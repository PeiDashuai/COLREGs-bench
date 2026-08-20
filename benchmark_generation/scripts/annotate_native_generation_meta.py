#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
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


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def first_non_null(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if v is not None:
            return v
    return default


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
    if has_r15 and not has_r17 and not has_other_non_tss:
        return "tss_plus_rule15"
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


def rel_bearing_sector(own_x: float, own_y: float, own_heading_deg: float, tgt_x: float, tgt_y: float) -> str:
    dx = tgt_x - own_x
    dy = tgt_y - own_y
    if dx == 0 and dy == 0:
        return "ahead"
    global_bearing = wrap_deg(math.degrees(math.atan2(dy, dx)))
    rel = wrap_deg(global_bearing - own_heading_deg)
    if rel <= 22.5 or rel >= 337.5:
        return "ahead"
    if 22.5 < rel < 157.5:
        return "starboard"
    if 202.5 < rel < 337.5:
        return "port"
    return "astern"


def stable_index_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    gm = sample.get("generation_meta", {})
    return {
        "sample_id": sample.get("sample_id"),
        "pattern": sample.get("pattern"),
        "schema_version": gm.get("schema_version"),
        "family_stratum": get_nested(gm, "family_semantics.family_stratum", {}),
        "effective_participants": get_nested(gm, "vessel_population.effective_participants"),
        "rule_overlap_mode": get_nested(gm, "family_semantics.family_stratum.rule_overlap_mode"),
        "crossing_angle_band": get_nested(gm, "family_semantics.family_stratum.crossing_angle_band"),
        "lane_occupancy": get_nested(gm, "family_semantics.family_stratum.lane_occupancy"),
        "quality_flags": get_nested(gm, "quality_flags", {}),
    }


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


# ============================================================
# Native tss derivation
# ============================================================

def derive_generator_block(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family_generator": row.get("generator_version", "tss_crossing_native_v1"),
        "candidate_seed": row.get("candidate_seed"),
        "sampling_round": 1,
        "source_mode": row.get("source_mode", "native_v3_generator"),
        "config_hash": None,
        "family_config_version": row.get("generator_version", "tss_crossing_native_v1"),
        "global_config_version": "benchmark_v3_native_genmeta_v1",
    }


def derive_scene_context(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {})
    area = ss.get("area_context", {}) if isinstance(ss, dict) else {}
    visibility = ss.get("visibility", "in_sight")
    return {
        "domain": ss.get("domain", "open"),
        "visibility": visibility,
        "in_sight": str(visibility) == "in_sight",
        "area_type": area.get("area_type", "tss"),
        "channel_context": bool(area.get("channel_context", False)),
        "tss_context": bool(area.get("tss_context", True)),
        "snapshot_time_s": safe_int(ss.get("snapshot_time_s"), 30),
        "duration_s": safe_int(ss.get("duration_s"), 180),
    }


def derive_vessel_population(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {})
    ownship = ss.get("ownship", {}) if isinstance(ss, dict) else {}
    targets = ss.get("targets", []) if isinstance(ss, dict) else []
    labels = row.get("labels", {})
    target_summaries = labels.get("target_summaries", []) if isinstance(labels, dict) else []

    eff = len(targets)
    ownship_status = first_non_null(ownship.get("vessel_class"), ownship.get("nav_status"), "power_driven")

    status_hist: Dict[str, int] = {}
    for t in targets:
        if not isinstance(t, dict):
            continue
        token = str(first_non_null(t.get("vessel_class"), t.get("nav_status"), "unknown"))
        status_hist[token] = status_hist.get(token, 0) + 1

    role_counts = {
        "give_way_targets": 0,
        "stand_on_targets": 0,
        "other_targets": 0,
    }
    for ts in target_summaries:
        if not isinstance(ts, dict):
            continue
        trg_rules = [str(x) for x in ts.get("triggered_rule_ids", [])]
        if any(r.startswith("COLREG_R15") or r.startswith("COLREG_R16") for r in trg_rules):
            role_counts["give_way_targets"] += 1
        elif any(r.startswith("COLREG_R17") for r in trg_rules):
            role_counts["stand_on_targets"] += 1
        else:
            role_counts["other_targets"] += 1

    return {
        "total_vessels": eff + 1,
        "target_vessels": eff,
        "effective_participants": eff,
        "ownship_status": ownship_status,
        "target_status_histogram": status_hist,
        "target_count_by_role": role_counts,
    }


def derive_interaction_structure(row: Dict[str, Any], vessel_population: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {})
    area = ss.get("area_context", {}) if isinstance(ss, dict) else {}
    ownship = ss.get("ownship", {}) if isinstance(ss, dict) else {}
    targets = ss.get("targets", []) if isinstance(ss, dict) else []
    labels = row.get("labels", {})
    target_summaries = labels.get("target_summaries", []) if isinstance(labels, dict) else []

    bearing_occ = {"port": 0, "starboard": 0, "ahead": 0, "astern": 0}
    own_x = safe_float(ownship.get("x_m"))
    own_y = safe_float(ownship.get("y_m"))
    own_h = safe_float(ownship.get("heading_deg"))

    min_cpa = None
    min_tcpa = None
    for i, t in enumerate(targets):
        if not isinstance(t, dict):
            continue
        if own_x is not None and own_y is not None and own_h is not None:
            sec = rel_bearing_sector(
                own_x=own_x,
                own_y=own_y,
                own_heading_deg=own_h,
                tgt_x=safe_float(t.get("x_m"), 0.0),
                tgt_y=safe_float(t.get("y_m"), 0.0),
            )
            bearing_occ[sec] += 1

        if i < len(target_summaries) and isinstance(target_summaries[i], dict):
            cpa = safe_float(target_summaries[i].get("cpa_m"))
            tcpa = safe_float(target_summaries[i].get("tcpa_s"))
            if cpa is not None:
                min_cpa = cpa if min_cpa is None else min(min_cpa, cpa)
            if tcpa is not None:
                min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)

    spatial_topology = "single_side_cluster"
    if bearing_occ["port"] > 0 and bearing_occ["starboard"] > 0 and bearing_occ["ahead"] > 0 and bearing_occ["astern"] > 0:
        spatial_topology = "surrounding_ring"
    elif bearing_occ["port"] > 0 and bearing_occ["starboard"] > 0 and bearing_occ["ahead"] > 0:
        spatial_topology = "frontal_blocking"
    elif bearing_occ["port"] > 0 and bearing_occ["starboard"] > 0:
        spatial_topology = "bilateral_constraint"

    inferred_angle = infer_crossing_angle_band(
        own_heading_deg=own_h,
        lane_heading_deg=safe_float(area.get("tss_lane_heading_deg")),
    )
    inferred_lane_occ = infer_lane_occupancy_band(len(targets))

    return {
        "encounter_mode": "multi_target" if vessel_population["effective_participants"] >= 2 else "single_target",
        "relative_bearing_occupancy": bearing_occ,
        "spatial_topology": spatial_topology,
        "interaction_graph_density": infer_lane_flow_complexity(inferred_lane_occ),
        "min_cpa_m": min_cpa,
        "min_tcpa_s": min_tcpa,
        "urgency_level": "high" if (min_tcpa is not None and min_tcpa <= 45) else "medium",
        "crossing_angle_band": inferred_angle,
        "channel_width_band": None,
        "lane_occupancy_band": inferred_lane_occ,
    }


def derive_resolution_structure(row: Dict[str, Any], vessel_population: Dict[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels", {})
    triggered_rules = labels.get("triggered_rules", []) if isinstance(labels, dict) else []
    allowed = labels.get("maneuver_allowed", []) if isinstance(labels, dict) else []
    forbidden = labels.get("maneuver_forbidden", []) if isinstance(labels, dict) else []
    suppressed_rules = labels.get("suppressed_rules", []) if isinstance(labels, dict) else []
    explanation_steps = labels.get("explanation_steps", []) if isinstance(labels, dict) else []

    if isinstance(allowed, dict):
        allowed = list(allowed.keys())
    if isinstance(forbidden, dict):
        forbidden = list(forbidden.keys())

    inferred_overlap = infer_rule_overlap_mode([str(x) for x in triggered_rules])

    return {
        "num_triggered_rules": len(triggered_rules),
        "num_effective_targets": vessel_population["effective_participants"],
        "num_conflicts": 0,
        "num_tensions": 1 if len(allowed) > 0 and len(forbidden) > 0 else 0,
        "num_compatibles": len(triggered_rules),
        "suppression_present": len(suppressed_rules) > 0,
        "suppression_count": len(suppressed_rules),
        "suppression_type": "na",
        "retained_action_count": len(allowed),
        "forbidden_action_count": len(forbidden),
        "explanation_stage_count": len(explanation_steps),
        "global_resolution_nontrivial": inferred_overlap != "pure_tss",
    }


def derive_family_semantics(
    row: Dict[str, Any],
    scene_context: Dict[str, Any],
    vessel_population: Dict[str, Any],
    interaction_structure: Dict[str, Any],
    resolution_structure: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latent = row.get("family_latent", {})
    labels = row.get("labels", {})
    ss = row.get("scene_spec", {})
    area = ss.get("area_context", {}) if isinstance(ss, dict) else {}
    ownship = ss.get("ownship", {}) if isinstance(ss, dict) else {}
    targets = ss.get("targets", []) if isinstance(ss, dict) else []
    triggered_rules = labels.get("triggered_rules", []) if isinstance(labels, dict) else []

    latent_stratum = {
        "crossing_angle_band": latent.get("crossing_angle_band"),
        "rule_overlap_mode": latent.get("rule_overlap_mode"),
        "lane_occupancy": latent.get("lane_occupancy_band"),
    }

    observed_stratum = {
        "crossing_angle_band": interaction_structure.get("crossing_angle_band"),
        "rule_overlap_mode": infer_rule_overlap_mode([str(x) for x in triggered_rules]),
        "lane_occupancy": interaction_structure.get("lane_occupancy_band"),
    }

    latent_alignment = {
        "crossing_angle_band_match": latent_stratum["crossing_angle_band"] == observed_stratum["crossing_angle_band"],
        "rule_overlap_mode_match": latent_stratum["rule_overlap_mode"] == observed_stratum["rule_overlap_mode"],
        "lane_occupancy_match": latent_stratum["lane_occupancy"] == observed_stratum["lane_occupancy"],
        "effective_target_count_match": safe_int(latent.get("effective_target_count")) == len(targets),
        "tss_context_match": scene_context.get("tss_context") is True and scene_context.get("area_type") == "tss",
        "rule10_present": any(str(r).startswith("COLREG_R10") for r in triggered_rules),
    }

    family_invariant_pass = all(latent_alignment.values())

    semantic_tags = [
        "tss_structure",
        "rule10_core",
        str(observed_stratum["crossing_angle_band"]),
        str(observed_stratum["rule_overlap_mode"]),
        str(observed_stratum["lane_occupancy"]),
        str(latent.get("entry_mode")),
        str(latent.get("lane_flow_complexity")),
    ]

    family_semantics = {
        "family_invariant_pass": family_invariant_pass,
        "family_stratum": observed_stratum,
        "semantic_tags": semantic_tags,
        "latent_alignment": latent_alignment,
        "native_latent_snapshot": copy.deepcopy(latent),
    }
    return family_semantics, observed_stratum


def derive_signatures(row: Dict[str, Any], interaction_structure: Dict[str, Any], resolution_structure: Dict[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels", {})
    ss = row.get("scene_spec", {})
    targets = ss.get("targets", []) if isinstance(ss, dict) else []

    geometry_signature = {
        "effective_participants": len(targets),
        "bearing_histogram": [
            interaction_structure["relative_bearing_occupancy"]["port"],
            interaction_structure["relative_bearing_occupancy"]["starboard"],
            interaction_structure["relative_bearing_occupancy"]["ahead"],
            interaction_structure["relative_bearing_occupancy"]["astern"],
        ],
        "distance_band_histogram": [],
        "speed_delta_band_histogram": [],
        "spatial_topology": interaction_structure["spatial_topology"],
        "urgency_level": interaction_structure["urgency_level"],
    }

    triggered_rules = labels.get("triggered_rules", []) if isinstance(labels, dict) else []
    allowed = labels.get("maneuver_allowed", []) if isinstance(labels, dict) else []
    forbidden = labels.get("maneuver_forbidden", []) if isinstance(labels, dict) else []
    if isinstance(allowed, dict):
        allowed = list(allowed.keys())
    if isinstance(forbidden, dict):
        forbidden = list(forbidden.keys())

    rule_signature = {
        "triggered_rules_sorted": sorted(set(str(x) for x in triggered_rules)),
        "maneuver_allowed_sorted": sorted(set(str(x) for x in allowed)),
        "maneuver_forbidden_sorted": sorted(set(str(x) for x in forbidden)),
        "lights_required_sorted": [],
        "sounds_required_sorted": [],
    }

    resolution_signature = {
        "suppression_type": resolution_structure["suppression_type"],
        "suppression_count": resolution_structure["suppression_count"],
        "retained_action_count": resolution_structure["retained_action_count"],
        "global_resolution_nontrivial": resolution_structure["global_resolution_nontrivial"],
    }

    visual_signature = {
        "layout_hash": None,
        "radar_hash": None,
    }

    return {
        "geometry_signature": geometry_signature,
        "rule_signature": rule_signature,
        "resolution_signature": resolution_signature,
        "visual_signature": visual_signature,
    }


def derive_quality_flags(family_semantics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "passes_invariant_gate": bool(family_semantics.get("family_invariant_pass")),
        "passes_plausibility_gate": None,
        "passes_duplicate_gate": None,
        "passes_diversity_gate": None,
        "accepted_into_candidate_pool": None,
        "accepted_into_release_split": False,
        "rejection_reasons": [] if family_semantics.get("family_invariant_pass") else ["native_latent_alignment_failed"],
    }


def annotate_native_tss_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(row)

    scene_context = derive_scene_context(out)
    vessel_population = derive_vessel_population(out)
    interaction_structure = derive_interaction_structure(out, vessel_population)
    resolution_structure = derive_resolution_structure(out, vessel_population)
    family_semantics, _ = derive_family_semantics(
        out, scene_context, vessel_population, interaction_structure, resolution_structure
    )
    signatures = derive_signatures(out, interaction_structure, resolution_structure)
    quality_flags = derive_quality_flags(family_semantics)

    out["generation_meta"] = {
        "schema_version": "genmeta_native_v1",
        "generator": derive_generator_block(out),
        "scene_context": scene_context,
        "vessel_population": vessel_population,
        "interaction_structure": interaction_structure,
        "family_semantics": family_semantics,
        "resolution_structure": resolution_structure,
        "signatures": signatures,
        "quality_flags": quality_flags,
    }
    return out


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Annotate native generation meta (first version: tss_crossing only)."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="tss_crossing", choices=["tss_crossing"], help="Only tss_crossing is supported in v1")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing annotated outputs")
    ap.add_argument("--validate-schema", action="store_true", help="Validate raw rows against raw_candidate.schema.json if jsonschema is available")
    ap.add_argument("--report-json", type=str, default=None, help="Optional report output path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    annotated_path = root / "annotated_pool" / FAMILY / "candidates_annotated.jsonl"
    index_path = root / "annotated_pool" / FAMILY / "generation_meta_index.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"

    if not raw_path.exists():
        print(f"ERROR: missing raw candidate file: {raw_path}", file=sys.stderr)
        return 2
    if annotated_path.exists() and not args.overwrite:
        print(f"ERROR: annotated output exists; use --overwrite: {annotated_path}", file=sys.stderr)
        return 2
    if args.validate_schema and not schema_path.exists():
        print(f"ERROR: missing schema file required by --validate-schema: {schema_path}", file=sys.stderr)
        return 2

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if args.validate_schema and schema_path.exists() else {}

    annotated_rows: List[Dict[str, Any]] = []
    schema_errors: List[Dict[str, Any]] = []
    invariant_pass_count = 0
    latent_alignment_fail_count = 0
    strata_hist: Dict[str, int] = {}

    for row in rows:
        sample_id = row.get("sample_id", "<missing_sample_id>")

        if args.validate_schema:
            schema_err = validate_against_schema(row, schema)
            if schema_err is not None:
                schema_errors.append({"sample_id": sample_id, "error": schema_err})

        ann = annotate_native_tss_row(row)
        annotated_rows.append(ann)

        gm = ann.get("generation_meta", {})
        fam_sem = gm.get("family_semantics", {})
        if fam_sem.get("family_invariant_pass") is True:
            invariant_pass_count += 1
        else:
            latent_alignment_fail_count += 1

        stratum = fam_sem.get("family_stratum", {})
        key = json.dumps(stratum, ensure_ascii=False, sort_keys=True)
        strata_hist[key] = strata_hist.get(key, 0) + 1

    write_jsonl(annotated_path, annotated_rows)
    write_jsonl(index_path, [stable_index_row(r) for r in annotated_rows])

    report = {
        "root": str(root),
        "family": FAMILY,
        "raw_file": str(raw_path),
        "annotated_file": str(annotated_path),
        "index_file": str(index_path),
        "row_count": len(rows),
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_validation_enabled": bool(args.validate_schema),
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:20],
        "family_invariant_pass_count": invariant_pass_count,
        "latent_alignment_fail_count": latent_alignment_fail_count,
        "unique_family_strata_count": len(strata_hist),
        "top_family_strata": sorted(
            [{"stratum": json.loads(k), "count": v} for k, v in strata_hist.items()],
            key=lambda x: (-x["count"], json.dumps(x["stratum"], sort_keys=True))
        )[:20],
        "notes": (
            "This script is native-line annotation, not bootstrap recovery. "
            "It standardizes explicit family_latent/scene_spec/labels into generation_meta "
            "and checks latent alignment."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "family_reports" / "tss_crossing_native_annotation_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"annotate_native_generation_meta.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows                     : {len(rows)}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {len(schema_errors)}")
    print(f"Invariant pass           : {invariant_pass_count}/{len(rows)}")
    print(f"Unique strata            : {len(strata_hist)}")
    print(f"Annotated output         : {annotated_path}")
    print(f"Meta index               : {index_path}")
    print(f"Report                   : {report_path}")
    print("=" * 100)

    return 0 if (len(schema_errors) == 0 and invariant_pass_count == len(rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
