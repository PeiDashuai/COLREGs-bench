#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


FAMILY = "tss_crossing"


# ============================================================
# IO
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL {path} line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object: {path} line {lineno}")
            rows.append(obj)
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ============================================================
# Helpers
# ============================================================

def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def heading_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


def infer_crossing_angle_band(own_heading_deg: Optional[float], lane_heading_deg: Optional[float]) -> Optional[str]:
    d = heading_diff_deg(own_heading_deg, lane_heading_deg)
    if d is None:
        return None
    x = min(d, abs(180.0 - d))
    if 75.0 <= x <= 105.0:
        return "near_right_angle"
    if 60.0 <= x < 75.0 or 105.0 < x <= 120.0:
        return "moderately_off"
    return "strongly_off"


def infer_rule_overlap_mode(triggered_rules: List[str]) -> str:
    """
    Taxonomy aligned with current native tss generator/spec semantics.

    Important convention:
    - R16 is often a consequence of R15 give-way obligation and should NOT by itself
      promote 'tss_plus_rule15' to 'multi_overlap'.
    - 'multi_overlap' is reserved for cases where multiple distinct non-TSS encounter
      families are simultaneously active in a way that exceeds the simple
      TSS + Rule15(+R16) or TSS + Rule16/17 pattern.
    """
    s = set(str(x) for x in triggered_rules)

    has_r10 = any(r.startswith("COLREG_R10") for r in s)
    has_r15 = any(r.startswith("COLREG_R15") for r in s)
    has_r16 = any(r.startswith("COLREG_R16") for r in s)
    has_r17 = any(r.startswith("COLREG_R17") for r in s)
    has_r13 = any(r.startswith("COLREG_R13") for r in s)
    has_r18 = any(r.startswith("COLREG_R18") for r in s)
    has_r9 = any(r.startswith("COLREG_R9") for r in s)
    has_other_non_tss = has_r13 or has_r18 or has_r9

    if has_r10 and not (has_r15 or has_r16 or has_r17 or has_other_non_tss):
        return "pure_tss"

    # TSS + Rule15 family; allow accompanying R16 as the natural maneuver consequence.
    if has_r15 and not has_r17 and not has_other_non_tss:
        return "tss_plus_rule15"

    # TSS + Rule16/17 family, but only when Rule15 is absent.
    if (has_r16 or has_r17) and not has_r15 and not has_other_non_tss:
        return "tss_plus_rule16_17"

    return "multi_overlap"


def infer_lane_occupancy_band(num_targets: int) -> str:
    if num_targets <= 1:
        return "low"
    if num_targets == 2:
        return "medium"
    return "high"


def infer_lane_flow_complexity(lane_occupancy_band: str) -> str:
    if lane_occupancy_band == "low":
        return "simple"
    if lane_occupancy_band == "medium":
        return "moderate"
    return "dense"


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


# ============================================================
# Row validators
# ============================================================

def validate_tss_row_semantics(row: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    debug: Dict[str, Any] = {}

    sample_id = row.get("sample_id")
    pattern = row.get("pattern")
    family_latent = row.get("family_latent", {})
    scene_spec = row.get("scene_spec", {})
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec, dict) else {}
    ownship = scene_spec.get("ownship", {}) if isinstance(scene_spec, dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec, dict) else []
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}

    debug["sample_id"] = sample_id
    debug["pattern"] = pattern

    if pattern != FAMILY:
        errors.append("pattern_not_tss_crossing")

    area_type = area.get("area_type")
    channel_context = area.get("channel_context")
    tss_context = area.get("tss_context")
    if area_type != "tss":
        errors.append("area_type_not_tss")
    if channel_context is not False:
        errors.append("channel_context_not_false")
    if tss_context is not True:
        errors.append("tss_context_not_true")

    lane_count = safe_int(area.get("lane_count"))
    lane_width_m = safe_float(area.get("lane_width_m"))
    tss_lane_heading_deg = safe_float(area.get("tss_lane_heading_deg"))
    sep_zone = area.get("separation_zone_geometry")

    if lane_count is None or lane_count < 1:
        errors.append("invalid_lane_count")
    if lane_width_m is None or lane_width_m <= 0:
        errors.append("invalid_lane_width_m")
    if tss_lane_heading_deg is None:
        errors.append("missing_tss_lane_heading_deg")
    if sep_zone is None:
        errors.append("missing_separation_zone_geometry")

    latent_angle = family_latent.get("crossing_angle_band")
    latent_occ = family_latent.get("lane_occupancy_band")
    latent_overlap = family_latent.get("rule_overlap_mode")
    latent_entry = family_latent.get("entry_mode")
    latent_target_count = safe_int(family_latent.get("effective_target_count"))
    latent_complexity = family_latent.get("lane_flow_complexity")

    if latent_angle not in {"near_right_angle", "moderately_off", "strongly_off"}:
        errors.append("invalid_latent_crossing_angle_band")
    if latent_occ not in {"low", "medium", "high"}:
        errors.append("invalid_latent_lane_occupancy_band")
    if latent_overlap not in {"pure_tss", "tss_plus_rule15", "tss_plus_rule16_17", "multi_overlap"}:
        errors.append("invalid_latent_rule_overlap_mode")
    if latent_entry not in {"direct_crossing", "edge_entry", "lane_cut_in", "near_boundary_crossing"}:
        errors.append("invalid_latent_entry_mode")

    num_targets = len(targets) if isinstance(targets, list) else 0
    inferred_occ = infer_lane_occupancy_band(num_targets)
    inferred_complexity = infer_lane_flow_complexity(inferred_occ)

    if latent_target_count is None or latent_target_count != num_targets:
        errors.append("effective_target_count_mismatch")
    if latent_occ != inferred_occ:
        errors.append("lane_occupancy_band_mismatch")
    if latent_complexity is not None and latent_complexity != inferred_complexity:
        errors.append("lane_flow_complexity_mismatch")

    own_heading_deg = safe_float(ownship.get("heading_deg"))
    inferred_angle = infer_crossing_angle_band(own_heading_deg, tss_lane_heading_deg)
    debug["inferred_crossing_angle_band"] = inferred_angle
    if inferred_angle is None:
        errors.append("cannot_infer_crossing_angle_band")
    elif latent_angle != inferred_angle:
        errors.append("crossing_angle_band_mismatch")

    triggered_rules = labels.get("triggered_rules", [])
    if not isinstance(triggered_rules, list):
        errors.append("triggered_rules_not_list")
        triggered_rules = []
    inferred_overlap = infer_rule_overlap_mode([str(x) for x in triggered_rules])
    debug["inferred_rule_overlap_mode"] = inferred_overlap
    if latent_overlap != inferred_overlap:
        errors.append("rule_overlap_mode_mismatch")

    if not any(str(r).startswith("COLREG_R10") for r in triggered_rules):
        errors.append("missing_rule10_family")

    target_summaries = labels.get("target_summaries", [])
    if not isinstance(target_summaries, list):
        errors.append("target_summaries_not_list")
        target_summaries = []

    if len(target_summaries) != num_targets:
        errors.append("target_summaries_count_mismatch")

    own_x = safe_float(ownship.get("x_m"))
    own_y = safe_float(ownship.get("y_m"))
    lane_half_total_width = safe_float(area.get("lane_half_total_width_m"))

    if own_x is None or own_y is None:
        errors.append("missing_ownship_position")
    else:
        if latent_entry == "direct_crossing":
            if own_x > -300:
                errors.append("entry_mode_direct_crossing_geometry_mismatch")
        elif latent_entry == "lane_cut_in":
            if own_x < -650:
                errors.append("entry_mode_lane_cut_in_geometry_mismatch")
        elif latent_entry == "edge_entry" and lane_half_total_width is not None:
            if abs(own_y) < 0.6 * lane_half_total_width:
                errors.append("entry_mode_edge_entry_geometry_mismatch")
        elif latent_entry == "near_boundary_crossing" and lane_half_total_width is not None:
            if abs(own_y) < 0.8 * lane_half_total_width:
                errors.append("entry_mode_near_boundary_crossing_geometry_mismatch")

    if isinstance(targets, list) and tss_lane_heading_deg is not None:
        bad_targets = 0
        for t in targets:
            if not isinstance(t, dict):
                bad_targets += 1
                continue
            h = safe_float(t.get("heading_deg"))
            d = heading_diff_deg(h, tss_lane_heading_deg)
            if d is None or d > 5.0:
                bad_targets += 1
        if bad_targets > 0:
            errors.append("target_heading_lane_alignment_mismatch")

    return errors, debug


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate native raw candidates (first vertical slice: tss_crossing only)."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="tss_crossing", choices=["tss_crossing"], help="Only tss_crossing is supported in v1")
    ap.add_argument("--report-json", type=str, default=None, help="Optional report output path")
    ap.add_argument("--max-debug", type=int, default=20, help="Max number of failing samples to keep with debug details")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"

    if not raw_path.exists():
        print(f"ERROR: missing raw candidate file: {raw_path}", file=sys.stderr)
        return 2
    if not schema_path.exists():
        print(f"ERROR: missing schema file: {schema_path}", file=sys.stderr)
        return 2

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path)

    schema_errors: List[Dict[str, Any]] = []
    semantic_failures: List[Dict[str, Any]] = []
    semantic_error_hist: Dict[str, int] = {}

    valid_schema_count = 0
    valid_semantic_count = 0

    for row in rows:
        sample_id = row.get("sample_id", "<missing_sample_id>")

        schema_err = validate_against_schema(row, schema)
        if schema_err is None:
            valid_schema_count += 1
        else:
            schema_errors.append({
                "sample_id": sample_id,
                "error": schema_err,
            })

        sem_errors, debug = validate_tss_row_semantics(row)
        if not sem_errors:
            valid_semantic_count += 1
        else:
            for e in sem_errors:
                semantic_error_hist[e] = semantic_error_hist.get(e, 0) + 1
            if len(semantic_failures) < args.max_debug:
                semantic_failures.append({
                    "sample_id": sample_id,
                    "errors": sem_errors,
                    "debug": debug,
                    "family_latent": row.get("family_latent", {}),
                    "area_context": get_nested(row, "scene_spec.area_context", {}),
                })

    report = {
        "root": str(root),
        "family": FAMILY,
        "raw_file": str(raw_path),
        "schema_file": str(schema_path),
        "row_count": len(rows),
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_valid_count": valid_schema_count,
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:args.max_debug],
        "semantic_valid_count": valid_semantic_count,
        "semantic_error_count": len(rows) - valid_semantic_count,
        "semantic_error_histogram": semantic_error_hist,
        "semantic_failures_head": semantic_failures,
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "validate_native_candidates_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"validate_native_candidates.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows                : {len(rows)}")
    print(f"jsonschema available: {HAS_JSONSCHEMA}")
    print(f"Schema valid        : {valid_schema_count}/{len(rows)}")
    print(f"Semantic valid      : {valid_semantic_count}/{len(rows)}")
    print(f"Report              : {report_path}")

    if schema_errors:
        print("\n[Schema errors head]")
        for item in schema_errors[:min(args.max_debug, 10)]:
            print(f"- {item['sample_id']}: {item['error']}")

    if semantic_error_hist:
        print("\n[Semantic error histogram]")
        for k, v in sorted(semantic_error_hist.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {k}: {v}")

    print("=" * 100)
    return 0 if (len(schema_errors) == 0 and valid_semantic_count == len(rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
