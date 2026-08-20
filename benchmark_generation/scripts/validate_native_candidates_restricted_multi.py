#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False

FAMILY = "restricted_multi"
GENERATOR_VERSION = "restricted_multi_native_v1"
SOURCE_MODE = "native_v3_generator"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
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
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row must be object: {path} line {lineno}")
            rows.append(obj)
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return default if v is None else int(v)
    except Exception:
        return default


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return default if v is None else float(v)
    except Exception:
        return default


def maybe_validate_schema(row: Dict[str, Any], schema: Dict[str, Any], enable: bool) -> List[str]:
    if not enable or not HAS_JSONSCHEMA:
        return []
    validator = jsonschema.Draft7Validator(schema)
    errs = []
    for e in validator.iter_errors(row):
        loc = ".".join(str(x) for x in e.absolute_path) if e.absolute_path else "$"
        errs.append(f"{loc}: {e.message}")
    return sorted(errs)


def check_basic_identity(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    if row.get("pattern") != FAMILY:
        errors.append("pattern_mismatch")
    if row.get("source_mode") != SOURCE_MODE:
        errors.append("source_mode_mismatch")
    if row.get("generator_version") != GENERATOR_VERSION:
        errors.append("generator_version_mismatch")
    for k in ["sample_id", "pattern", "source_mode", "generator_version", "candidate_seed", "family_latent", "scene_spec", "inputs", "labels", "debug"]:
        if k not in row:
            errors.append(f"missing_top_field:{k}")
    return errors, []


def check_latent_fields(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    latent = row.get("family_latent") or {}
    for k in ["num_effective_targets", "spatial_topology", "suppression_mechanism", "urgency_band", "visibility_mode", "target_status_mix", "interaction_density"]:
        if k not in latent:
            errors.append(f"missing_family_latent:{k}")
    return errors, []


def check_scene_structure(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    scene = row.get("scene_spec") or {}
    for k in ["seed", "domain", "visibility", "snapshot_time_s", "duration_s", "ownship", "targets", "area_context", "target_role_graph"]:
        if k not in scene:
            errors.append(f"missing_scene_spec:{k}")
    targets = scene.get("targets") or []
    if not isinstance(targets, list):
        errors.append("targets_not_list")
        return errors, warnings
    if len(targets) < 3:
        errors.append("fewer_than_3_effective_targets")
    latent_n = safe_int(get_nested(row, "family_latent.num_effective_targets"))
    if latent_n is not None and len(targets) != latent_n:
        errors.append("effective_target_count_mismatch")
    ids = [str(t.get("id")) for t in targets if isinstance(t, dict)]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_target_ids")

    own = scene.get("ownship") or {}
    ox = safe_float(own.get("x_m"), 0.0) or 0.0
    oy = safe_float(own.get("y_m"), 0.0) or 0.0
    mind = None
    sectors = []
    for t in targets:
        if not isinstance(t, dict):
            errors.append("target_not_object")
            continue
        for k in ["id", "vessel_class", "nav_status", "x_m", "y_m", "heading_deg", "speed_mps", "intended_role", "relative_bearing_sector"]:
            if k not in t:
                errors.append(f"missing_target_field:{k}")
        tx = safe_float(t.get("x_m"))
        ty = safe_float(t.get("y_m"))
        if tx is not None and ty is not None:
            d = math.hypot(tx - ox, ty - oy)
            mind = d if mind is None else min(mind, d)
        sectors.append(str(t.get("relative_bearing_sector")))

    if mind is not None:
        urg = str(get_nested(row, "family_latent.urgency_band"))
        if urg == "high" and mind > 350.0:
            errors.append("urgency_band_not_matching_real_cpa_tcpa")
        elif urg == "medium" and not (220.0 <= mind <= 700.0):
            errors.append("urgency_band_not_matching_real_cpa_tcpa")
        elif urg == "low" and mind < 650.0:
            errors.append("urgency_band_not_matching_real_cpa_tcpa")

    if len(set(sectors)) < 2:
        warnings.append("low_sector_diversity")
    return list(dict.fromkeys(errors)), warnings


def check_topology_realization(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    topo = str(get_nested(row, "family_latent.spatial_topology"))
    graph_topo = str(get_nested(row, "scene_spec.target_role_graph.topology"))
    if graph_topo != topo:
        errors.append("topology_not_matching_latent")
    sectors = [str(t.get("relative_bearing_sector")) for t in get_nested(row, "scene_spec.targets", []) if isinstance(t, dict)]
    c = Counter(sectors)
    n = len(sectors)
    if topo == "single_side_cluster":
        side_total = c.get("port", 0) + c.get("starboard", 0)
        dominant = max(c.get("port", 0), c.get("starboard", 0))
        if side_total < max(2, n - 1):
            errors.append("single_side_cluster_not_side_dominant")
        if dominant < max(2, side_total - 1):
            errors.append("single_side_cluster_not_realized")
    elif topo == "bilateral_constraint":
        if c.get("port", 0) < 1 or c.get("starboard", 0) < 1:
            errors.append("bilateral_constraint_not_realized")
    elif topo == "frontal_blocking":
        if c.get("ahead", 0) < 1:
            errors.append("frontal_blocking_missing_ahead_target")
        if c.get("port", 0) + c.get("starboard", 0) < 1:
            errors.append("frontal_blocking_missing_side_target")
    elif topo == "surrounding_ring":
        for need in ["ahead", "port", "starboard", "astern"]:
            if c.get(need, 0) < 1:
                errors.append(f"surrounding_ring_missing_{need}")
    return errors, []


def _has_priority_target(targets: List[Dict[str, Any]]) -> bool:
    for t in targets:
        vc = str(t.get("vessel_class"))
        ns = str(t.get("nav_status"))
        if vc in {"ram", "cbd", "fishing"} or ns in {"restricted_in_ability_to_manoeuvre", "engaged_in_fishing", "constrained_by_draught"}:
            return True
    return False


def _has_action_conflict(local_obligations: List[Dict[str, Any]]) -> bool:
    allow_by_p: Dict[str, set] = {}
    forbid_by_p: Dict[str, set] = {}
    for lo in local_obligations:
        for p in (lo.get("allow") or {}):
            allow_by_p.setdefault(str(p), set()).add(str(lo.get("target_id")))
        for p in (lo.get("forbid") or {}):
            forbid_by_p.setdefault(str(p), set()).add(str(lo.get("target_id")))
    return any((p in allow_by_p and p in forbid_by_p) for p in set(allow_by_p) | set(forbid_by_p))


def check_suppression_realization(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    supp = str(get_nested(row, "family_latent.suppression_mechanism"))
    targets = [t for t in get_nested(row, "scene_spec.targets", []) if isinstance(t, dict)]
    area = get_nested(row, "scene_spec.area_context", {}) or {}
    local_obligations = get_nested(row, "debug.reasoning.local_obligations", []) or []
    suppression_records = get_nested(row, "labels.suppression_records", []) or []
    suppression_count = safe_int(get_nested(row, "debug.reasoning.resolution.suppression_count"), 0) or 0
    has_area_source = bool(area.get("channel_context") or area.get("tss_context") or get_nested(row, "scene_spec.visibility") == "restricted_visibility")
    has_priority = _has_priority_target(targets)
    has_conflict = _has_action_conflict(local_obligations)
    if suppression_count < 1 and not suppression_records:
        errors.append("no_suppression_and_no_conflict")
    if supp == "action_conflict" and not has_conflict:
        errors.append("action_conflict_not_realized")
    elif supp == "hierarchy_override" and not has_priority:
        errors.append("hierarchy_override_without_priority_target")
    elif supp == "area_override" and not has_area_source:
        errors.append("area_override_without_area_source")
    elif supp == "mixed" and (int(has_conflict) + int(has_priority) + int(has_area_source) < 2):
        errors.append("mixed_without_two_or_more_sources")
    return errors, []


def check_global_resolution(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    labels = row.get("labels") or {}
    resolution = get_nested(row, "debug.reasoning.resolution", {}) or {}
    sensitivity = get_nested(row, "debug.reasoning.target_removal_sensitivity", {}) or {}
    if not bool(resolution.get("global_resolution_nontrivial", False)):
        errors.append("no_global_resolution")
    contributing = list(resolution.get("contributing_target_ids", []) or [])
    if len(contributing) < 2:
        errors.append("final_labels_explainable_by_single_target")
    if not bool(sensitivity.get("sensitive", False)):
        errors.append("no_change_under_target_removal")
    if bool(sensitivity.get("sensitive", False)) and len(list(sensitivity.get("sensitive_target_ids", []) or [])) < 1:
        errors.append("target_removal_sensitivity_missing_targets")
    if not ((labels.get("maneuver_allowed") or []) or (labels.get("maneuver_forbidden") or [])):
        errors.append("missing_final_maneuvers")
    expl = labels.get("explanation_steps") or []
    global_stage_ok = False
    for e in expl:
        if isinstance(e, dict) and str(e.get("stage")) == "global_stage" and str(e.get("text") or "").strip():
            global_stage_ok = True
            break
    if not global_stage_ok:
        errors.append("explanation_global_stage_empty")
    return errors, []


def check_label_structure(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    labels = row.get("labels") or {}
    for k in ["triggered_rules", "target_summaries", "primitive_ledger_summary", "suppressed_rules", "suppression_records", "explanation_steps", "maneuver_allowed", "maneuver_forbidden"]:
        if k not in labels:
            errors.append(f"missing_label_field:{k}")
    ts = labels.get("target_summaries") or []
    targets = get_nested(row, "scene_spec.targets", []) or []
    if isinstance(ts, list) and isinstance(targets, list) and len(ts) != len(targets):
        errors.append("target_summary_count_mismatch")
    if len(labels.get("triggered_rules") or []) < 4:
        errors.append("too_few_triggered_rules")
    return errors, []


def validate_row(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for fn in [check_basic_identity, check_latent_fields, check_scene_structure, check_topology_realization, check_suppression_realization, check_global_resolution, check_label_structure]:
        e, w = fn(row)
        errors.extend(e)
        warnings.extend(w)
    return list(dict.fromkeys(errors)), list(dict.fromkeys(warnings))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate native restricted_multi candidates")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--family", type=str, choices=[FAMILY], default=FAMILY)
    p.add_argument("--max-debug", type=int, default=20)
    p.add_argument("--report-json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"
    report_path = Path(args.report_json)
    if not report_path.is_absolute():
        report_path = root / report_path

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if schema_path.exists() else {}

    schema_error_hist = Counter()
    semantic_error_hist = Counter()
    warning_hist = Counter()
    schema_valid_count = 0
    semantic_valid_count = 0
    failures_head: List[Dict[str, Any]] = []

    for row in rows:
        schema_errors = maybe_validate_schema(row, schema, enable=bool(schema))
        if schema_errors:
            for e in schema_errors:
                schema_error_hist[e] += 1
        else:
            schema_valid_count += 1

        errors, warnings = validate_row(row)
        if errors:
            for e in errors:
                semantic_error_hist[e] += 1
            if len(failures_head) < int(args.max_debug):
                failures_head.append({
                    "sample_id": row.get("sample_id"),
                    "errors": errors,
                    "warnings": warnings,
                    "family_latent": row.get("family_latent"),
                    "debug": row.get("debug"),
                })
        else:
            semantic_valid_count += 1
        for w in warnings:
            warning_hist[w] += 1

    report = {
        "family": FAMILY,
        "generator_version": GENERATOR_VERSION,
        "row_count": len(rows),
        "schema_validation_enabled": True,
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_valid_count": schema_valid_count,
        "schema_error_count": sum(schema_error_hist.values()),
        "schema_error_histogram": dict(sorted(schema_error_hist.items())),
        "semantic_valid_count": semantic_valid_count,
        "semantic_error_count": sum(semantic_error_hist.values()),
        "semantic_error_histogram": dict(sorted(semantic_error_hist.items())),
        "warning_histogram": dict(sorted(warning_hist.items())),
        "semantic_failures_head": failures_head,
    }
    write_json(report_path, report)
    print("=" * 100)
    print(f"validate_native_candidates_restricted_multi.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows read                : {len(rows)}")
    print(f"Schema validation enabled: True")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema valid             : {schema_valid_count}/{len(rows)}")
    print(f"Schema error count       : {sum(schema_error_hist.values())}")
    print(f"Semantic valid           : {semantic_valid_count}/{len(rows)}")
    print(f"Semantic error count     : {sum(semantic_error_hist.values())}")
    print(f"Report written           : {report_path}")


if __name__ == "__main__":
    main()
