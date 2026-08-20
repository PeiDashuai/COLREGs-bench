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


FAMILY = "channel_crossing_impede"
GENERATOR_VERSION = "channel_crossing_impede_native_v2_1"


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



def infer_crossing_angle_band_from_local_heading(local_heading_deg: Optional[float]) -> Optional[str]:
    if local_heading_deg is None:
        return None
    d = wrap_deg(float(local_heading_deg))
    # angle to channel axis, modulo reciprocal direction
    acute = min(d, abs(180.0 - d), abs(360.0 - d))
    if 18.0 <= acute <= 42.0:
        return "small"
    if 45.0 <= acute <= 78.0:
        return "medium"
    if 82.0 <= acute <= 115.0:
        return "large"
    return None



def has_rule_prefix(triggered_rules: List[str], prefix: str) -> bool:
    return any(str(r).startswith(prefix) for r in triggered_rules)



def summary_mentions_channel(explanation_steps: List[Dict[str, Any]]) -> bool:
    joined = " ".join(str(x.get("summary", "")) for x in explanation_steps if isinstance(x, dict)).lower()
    return "channel" in joined



def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)



def validate_category(value: Any, allowed: set[str], err_name: str, errors: List[str]) -> None:
    if str(value) not in allowed:
        errors.append(err_name)



def infer_primary_crossing_actor(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
    family_latent = row.get("family_latent", {}) if isinstance(row.get("family_latent"), dict) else {}
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    pos = str(family_latent.get("ownship_channel_position"))

    if pos in {"entering_from_outside", "crossing_across"}:
        own_local = get_nested(scene_spec, "debug_native_geometry.ownship_local", {})
        if isinstance(own_local, dict):
            return "ownship", safe_float(own_local.get("heading_local_deg")), own_local
        return None, None, {}

    for t in scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []:
        if not isinstance(t, dict):
            continue
        if str(t.get("intended_role")) == "crossing_target":
            lg = t.get("local_geometry", {}) if isinstance(t.get("local_geometry"), dict) else {}
            return str(t.get("id", "crossing_target")), safe_float(lg.get("heading_local_deg")), lg
    return None, None, {}


# ============================================================
# Row validator
# ============================================================

def validate_row_semantics(row: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    debug: Dict[str, Any] = {}

    sample_id = row.get("sample_id")
    pattern = row.get("pattern")
    source_mode = row.get("source_mode")
    generator_version = row.get("generator_version")
    family_latent = row.get("family_latent", {}) if isinstance(row.get("family_latent"), dict) else {}
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec.get("area_context"), dict) else {}
    ownship = scene_spec.get("ownship", {}) if isinstance(scene_spec.get("ownship"), dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    debug_native_geometry = scene_spec.get("debug_native_geometry", {}) if isinstance(scene_spec.get("debug_native_geometry"), dict) else {}
    own_local = debug_native_geometry.get("ownship_local", {}) if isinstance(debug_native_geometry.get("ownship_local"), dict) else {}

    debug["sample_id"] = sample_id
    debug["pattern"] = pattern
    debug["source_mode"] = source_mode
    debug["generator_version"] = generator_version

    if pattern != FAMILY:
        errors.append("pattern_not_channel_crossing_impede")
    if source_mode != "native_v3_generator":
        errors.append("source_mode_not_native_v3_generator")
    if generator_version != GENERATOR_VERSION:
        errors.append("generator_version_mismatch")

    # Area / context checks
    if area.get("area_type") != "narrow_channel":
        errors.append("area_type_not_narrow_channel")
    if area.get("channel_context") is not True:
        errors.append("channel_context_not_true")
    if area.get("tss_context") is not False:
        errors.append("tss_context_not_false")

    channel_width_m = safe_float(area.get("channel_width_m"))
    channel_half_width_m = safe_float(area.get("channel_half_width_m"))
    channel_heading_deg = safe_float(area.get("channel_heading_deg"))
    if channel_width_m is None or channel_width_m <= 0:
        errors.append("invalid_channel_width_m")
    if channel_half_width_m is None or channel_half_width_m <= 0:
        errors.append("invalid_channel_half_width_m")
    if channel_heading_deg is None:
        errors.append("missing_channel_heading_deg")
    if not isinstance(area.get("channel_centerline"), dict):
        errors.append("missing_channel_centerline")

    # latent categories
    validate_category(family_latent.get("channel_width_band"), {"narrow", "medium", "wide"}, "invalid_latent_channel_width_band", errors)
    validate_category(family_latent.get("occupancy_pattern"), {"single_occupancy", "opposing_flow", "third_vessel_blocking", "low_occupancy"}, "invalid_latent_occupancy_pattern", errors)
    validate_category(family_latent.get("semantic_mode"), {"ordinary_crossing_in_channel", "keep_starboard_dominant", "not_to_impede", "mixed_semantics"}, "invalid_latent_semantic_mode", errors)
    validate_category(family_latent.get("crossing_angle_band"), {"small", "medium", "large"}, "invalid_latent_crossing_angle_band", errors)
    validate_category(family_latent.get("ownship_channel_position"), {"inside_center", "inside_boundary", "entering_from_outside", "crossing_across"}, "invalid_latent_ownship_channel_position", errors)

    latent_count = safe_int(family_latent.get("effective_target_count"))
    num_targets = len(targets)
    if latent_count is None or latent_count != num_targets:
        errors.append("effective_target_count_mismatch")

    # ownship position realization
    pos = str(family_latent.get("ownship_channel_position"))
    own_y_local = safe_float(own_local.get("y_local_m"))
    own_heading_local = safe_float(own_local.get("heading_local_deg"))
    own_notes = [str(x) for x in own_local.get("notes", [])] if isinstance(own_local.get("notes"), list) else []
    debug["ownship_local"] = own_local

    if channel_half_width_m is None:
        pass
    elif own_y_local is None:
        errors.append("missing_ownship_local_geometry")
    else:
        abs_y = abs(own_y_local)
        if pos == "inside_center":
            if abs_y > 0.20 * channel_half_width_m:
                errors.append("inside_center_not_realized")
            if "ownship_started_inside_channel_center_band" not in own_notes:
                warnings.append("inside_center_note_missing")
        elif pos == "inside_boundary":
            if abs_y < 0.35 * channel_half_width_m or abs_y > 0.80 * channel_half_width_m:
                errors.append("inside_boundary_not_realized")
            if "ownship_started_inside_channel_boundary_band" not in own_notes:
                warnings.append("inside_boundary_note_missing")
        elif pos == "entering_from_outside":
            if abs_y <= channel_half_width_m:
                errors.append("entering_from_outside_not_realized")
            if "ownship_started_outside_channel_and_enters" not in own_notes:
                warnings.append("entering_from_outside_note_missing")
        elif pos == "crossing_across":
            if abs_y <= channel_half_width_m:
                errors.append("crossing_across_not_realized")
            if "ownship_path_crosses_channel_corridor" not in own_notes:
                warnings.append("crossing_across_note_missing")

    # angle band realization
    actor_id, actor_local_heading, actor_debug = infer_primary_crossing_actor(row)
    inferred_band = infer_crossing_angle_band_from_local_heading(actor_local_heading)
    debug["crossing_actor_id"] = actor_id
    debug["crossing_actor_local_heading_deg"] = actor_local_heading
    debug["inferred_crossing_angle_band"] = inferred_band
    if actor_id is None:
        errors.append("missing_crossing_actor_for_angle_inference")
    elif inferred_band is None:
        errors.append("cannot_infer_crossing_angle_band")
    elif inferred_band != str(family_latent.get("crossing_angle_band")):
        errors.append("crossing_angle_band_mismatch")

    # occupancy realization
    roles = [str(t.get("intended_role")) for t in targets if isinstance(t, dict)]
    occ = str(family_latent.get("occupancy_pattern"))
    if occ == "single_occupancy":
        if num_targets != 1:
            errors.append("single_occupancy_target_count_mismatch")
    elif occ == "low_occupancy":
        if num_targets > 2:
            errors.append("low_occupancy_target_count_exceeds_limit")
    elif occ == "opposing_flow":
        if num_targets < 2 or "opposing_along_channel_target" not in roles:
            errors.append("opposing_flow_not_realized")
    elif occ == "third_vessel_blocking":
        if num_targets < 3 or "blocking_boundary_target" not in roles:
            errors.append("third_vessel_blocking_not_realized")

    # label structure checks
    triggered_rules = labels.get("triggered_rules", [])
    target_summaries = labels.get("target_summaries", [])
    explanation_steps = labels.get("explanation_steps", [])
    maneuver_allowed = labels.get("maneuver_allowed", [])
    maneuver_forbidden = labels.get("maneuver_forbidden", [])

    if not isinstance(triggered_rules, list):
        errors.append("triggered_rules_not_list")
        triggered_rules = []
    if not isinstance(target_summaries, list):
        errors.append("target_summaries_not_list")
        target_summaries = []
    if not isinstance(explanation_steps, list):
        errors.append("explanation_steps_not_list")
        explanation_steps = []
    if not isinstance(maneuver_allowed, list):
        errors.append("maneuver_allowed_not_list")
        maneuver_allowed = []
    if not isinstance(maneuver_forbidden, list):
        errors.append("maneuver_forbidden_not_list")
        maneuver_forbidden = []

    if len(target_summaries) != num_targets:
        errors.append("target_summaries_count_mismatch")
    if not has_rule_prefix([str(x) for x in triggered_rules], "COLREG_R09"):
        errors.append("missing_rule9_family")
    if not summary_mentions_channel([x for x in explanation_steps if isinstance(x, dict)]):
        errors.append("final_explanation_not_channel_dependent")
    if "IGNORE_CHANNEL_CONTEXT" not in [str(x) for x in maneuver_forbidden]:
        warnings.append("channel_ablation_proxy_missing_ignore_channel_context_forbidden")

    sem = str(family_latent.get("semantic_mode"))
    trig = [str(x) for x in triggered_rules]
    allowed = [str(x) for x in maneuver_allowed]

    if sem == "ordinary_crossing_in_channel":
        if "crossing_target" not in roles:
            errors.append("ordinary_crossing_missing_crossing_target")
        if "COLREG_R09_CROSSING_NARROW_CHANNEL" not in trig:
            errors.append("ordinary_crossing_missing_rule9_crossing")
    elif sem == "keep_starboard_dominant":
        if "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT" not in trig:
            errors.append("keep_starboard_missing_rule9_keep_starboard")
        if "KEEP_TO_STARBOARD_WITHIN_CHANNEL" not in allowed:
            errors.append("keep_starboard_missing_allowed_action")
    elif sem == "not_to_impede":
        if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" not in trig and "DO_NOT_IMPEDE_CHANNEL_TRAFFIC" not in allowed:
            errors.append("not_to_impede_not_realized")
        if pos not in {"entering_from_outside", "crossing_across"}:
            errors.append("not_to_impede_wrong_ownship_position")
    elif sem == "mixed_semantics":
        if not has_rule_prefix(trig, "COLREG_R09"):
            errors.append("mixed_semantics_missing_rule9_family")
        overlap_count = 0
        if "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in trig:
            overlap_count += 1
        if "COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK" in trig:
            overlap_count += 1
        if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" in trig:
            overlap_count += 1
        if overlap_count < 2:
            errors.append("mixed_semantics_overlap_not_realized")

    # open-water reducibility proxy
    only_generic = not any(r in trig for r in [
        "COLREG_R09_CROSSING_NARROW_CHANNEL",
        "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT",
        "COLREG_R09_NOT_TO_IMPEDE_PASSAGE",
    ])
    if only_generic:
        errors.append("sample_reducible_to_open_water_crossing")

    # target-summary checks
    by_target_id: Dict[str, Dict[str, Any]] = {}
    for ts in target_summaries:
        if isinstance(ts, dict) and ts.get("target_id") is not None:
            by_target_id[str(ts.get("target_id"))] = ts

    for t in targets:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id"))
        role = str(t.get("intended_role"))
        ts = by_target_id.get(tid)
        if ts is None:
            errors.append("missing_target_summary_for_target")
            continue
        rule_ids = [str(x) for x in ts.get("triggered_rule_ids", [])] if isinstance(ts.get("triggered_rule_ids"), list) else []
        sector = str(ts.get("relative_bearing_sector"))

        if role == "crossing_target" and "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in rule_ids:
            if sector != "starboard":
                errors.append("crossing_rule15_bearing_mismatch")
        if role == "opposing_along_channel_target" and "COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK" in rule_ids:
            if sector != "ahead":
                warnings.append("opposing_target_rule14_not_ahead")
        if role == "channel_traffic_target" and sem in {"not_to_impede", "mixed_semantics"}:
            if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" not in rule_ids:
                warnings.append("channel_traffic_target_missing_not_to_impede_rule")
        if role == "blocking_boundary_target" and occ == "third_vessel_blocking":
            if "COLREG_R08_AVOID_CONSTRAINED_CHANNEL_COMPRESSION" not in rule_ids:
                warnings.append("blocking_target_missing_channel_compression_rule")

    # family invariant proxies from row.debug.sanity_checks
    sanity_checks = row.get("debug", {}).get("sanity_checks", {}) if isinstance(row.get("debug"), dict) else {}
    if isinstance(sanity_checks, dict):
        if sanity_checks.get("channel_context_present") is False:
            errors.append("sanity_channel_context_present_false")
        if sanity_checks.get("rule9_family_present") is False:
            errors.append("sanity_rule9_family_present_false")
        if sanity_checks.get("target_count_matches_latent") is False:
            errors.append("sanity_target_count_matches_latent_false")

    return errors, warnings, debug


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate native raw candidates for channel_crossing_impede."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="channel_crossing_impede", choices=["channel_crossing_impede"])
    ap.add_argument("--report-json", type=str, default=None)
    ap.add_argument("--max-debug", type=int, default=20)
    return ap.parse_args()



def main() -> int:
    args = parse_args()
    root = Path(args.root)

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
    warning_hist: Dict[str, int] = {}

    valid_schema_count = 0
    valid_semantic_count = 0

    for row in rows:
        sample_id = row.get("sample_id", "<missing_sample_id>")
        schema_err = validate_against_schema(row, schema)
        if schema_err is None:
            valid_schema_count += 1
        else:
            schema_errors.append({"sample_id": sample_id, "error": schema_err})

        sem_errors, sem_warnings, debug = validate_row_semantics(row)
        for w in sem_warnings:
            warning_hist[w] = warning_hist.get(w, 0) + 1

        if not sem_errors:
            valid_semantic_count += 1
        else:
            for e in sem_errors:
                semantic_error_hist[e] = semantic_error_hist.get(e, 0) + 1
            if len(semantic_failures) < args.max_debug:
                semantic_failures.append({
                    "sample_id": sample_id,
                    "errors": sem_errors,
                    "warnings": sem_warnings,
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
        "warning_histogram": warning_hist,
        "semantic_failures_head": semantic_failures,
        "notes": (
            "This validator checks channel-context causality, Rule 9 family presence, "
            "latent-to-geometry realization, target-summary alignment, and basic channel-ablation proxies."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "validate_native_candidates_channel_crossing_impede_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"validate_native_candidates_channel_crossing_impede.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows                : {len(rows)}")
    print(f"jsonschema available: {HAS_JSONSCHEMA}")
    print(f"Schema valid        : {valid_schema_count}/{len(rows)}")
    print(f"Semantic valid      : {valid_semantic_count}/{len(rows)}")
    print(f"Report              : {report_path}")
    if schema_errors:
        print("\n[Schema errors head]")
        for item in schema_errors[:min(10, args.max_debug)]:
            print(f"- {item['sample_id']}: {item['error']}")
    if semantic_error_hist:
        print("\n[Semantic error histogram]")
        for k, v in sorted(semantic_error_hist.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {k}: {v}")
    if warning_hist:
        print("\n[Warning histogram]")
        for k, v in sorted(warning_hist.items(), key=lambda x: (-x[1], x[0])):
            print(f"- {k}: {v}")
    print("=" * 100)

    return 0 if (len(schema_errors) == 0 and valid_semantic_count == len(rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
