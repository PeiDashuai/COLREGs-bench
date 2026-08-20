#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False

FAMILY = "restricted_multi"
GENERATOR_VERSION = "restricted_multi_native_v1"
GENMETA_VERSION = "genmeta_native_v1"


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
                raise ValueError(f"JSONL row is not object: {path} line {lineno}")
            rows.append(obj)
    return rows


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("pyyaml is required")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be object: {path}")
    return data


def rel_bearing_sector(own_x: float, own_y: float, own_heading_deg: float, tgt_x: float, tgt_y: float) -> str:
    dx = tgt_x - own_x
    dy = tgt_y - own_y
    if dx == 0.0 and dy == 0.0:
        return "ahead"
    global_bearing = math.degrees(math.atan2(dy, dx)) % 360.0
    rel = (global_bearing - own_heading_deg) % 360.0
    if rel <= 22.5 or rel >= 337.5:
        return "ahead"
    if 22.5 < rel < 157.5:
        return "starboard"
    if 202.5 < rel < 337.5:
        return "port"
    return "astern"


def infer_target_status_mix(targets: List[Dict[str, Any]]) -> str:
    special = 0
    for t in targets:
        vc = str(t.get("vessel_class", ""))
        ns = str(t.get("nav_status", ""))
        if vc in {"ram", "fishing", "cbd"} or ns in {
            "restricted_in_ability_to_manoeuvre",
            "engaged_in_fishing",
            "constrained_by_draught",
        }:
            special += 1
    return "mixed_status" if special > 0 else "all_power"


def infer_urgency_from_distances(targets: List[Dict[str, Any]]) -> str:
    dists: List[float] = []
    for t in targets:
        d = safe_float(t.get("distance_from_ownship_m"))
        if d is not None:
            dists.append(d)
    if not dists:
        return "medium"
    m = min(dists)
    if m <= 250.0:
        return "high"
    if m <= 650.0:
        return "medium"
    return "low"


def realized_suppression_type(row: Dict[str, Any]) -> str:
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec.get("area_context"), dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []
    records = row.get("labels", {}).get("suppression_records", []) if isinstance(row.get("labels", {}), dict) else []

    sources = set()
    for r in records:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source", ""))
        if "hierarchy" in src:
            sources.add("hierarchy")
        elif "area" in src or "visibility" in src:
            sources.add("area")
        elif "conflict" in src or "merge" in src:
            sources.add("action_conflict")

    # Fallback from scene context if records are generic.
    has_hierarchy = any(
        str(t.get("vessel_class")) in {"ram", "cbd", "fishing"} or
        str(t.get("nav_status")) in {"restricted_in_ability_to_manoeuvre", "engaged_in_fishing", "constrained_by_draught"}
        for t in targets if isinstance(t, dict)
    )
    has_area = bool(area.get("channel_context") or area.get("tss_context") or scene_spec.get("visibility") == "restricted_visibility")

    if has_hierarchy:
        sources.add("hierarchy")
    if has_area:
        sources.add("area")
    if records:
        sources.add("action_conflict")

    if len(sources) >= 3 or ("hierarchy" in sources and "area" in sources):
        return "mixed"
    if "hierarchy" in sources and len(sources) == 1:
        return "hierarchy_override"
    if "area" in sources and len(sources) == 1:
        return "area_override"
    return "action_conflict"


def derive_generator_block(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family_generator": row.get("generator_version", GENERATOR_VERSION),
        "generator_revision": get_nested(row, "debug.generator_revision"),
        "candidate_seed": row.get("candidate_seed"),
        "source_mode": row.get("source_mode", "native_v3_generator"),
        "sampling_round": 1,
        "family_config_version": row.get("generator_version", GENERATOR_VERSION),
        "global_config_version": "benchmark_v3_native_genmeta_v1",
    }


def derive_scene_context(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    area = ss.get("area_context", {}) if isinstance(ss.get("area_context"), dict) else {}
    return {
        "domain": ss.get("domain", "restricted"),
        "visibility": ss.get("visibility", "in_sight"),
        "in_sight": str(ss.get("visibility", "in_sight")) == "in_sight",
        "snapshot_time_s": safe_float(ss.get("snapshot_time_s"), 30.0),
        "duration_s": safe_float(ss.get("duration_s"), 120.0),
        "area_source": area.get("area_source", "none"),
        "channel_context": bool(area.get("channel_context", False)),
        "tss_context": bool(area.get("tss_context", False)),
        "area_type": area.get("area_type", "open"),
    }


def derive_vessel_population(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    own = ss.get("ownship", {}) if isinstance(ss.get("ownship"), dict) else {}
    targets = ss.get("targets", []) if isinstance(ss.get("targets"), list) else []
    role_hist = Counter(str(t.get("intended_role", "unknown")) for t in targets if isinstance(t, dict))
    sector_hist = Counter(str(t.get("relative_bearing_sector", "unknown")) for t in targets if isinstance(t, dict))
    status_hist = Counter(
        f"{t.get('vessel_class','unknown')}::{t.get('nav_status','unknown')}"
        for t in targets if isinstance(t, dict)
    )
    return {
        "total_vessels": len(targets) + 1,
        "target_vessels": len(targets),
        "effective_participants": len(targets),
        "ownship_status": f"{own.get('vessel_class','unknown')}::{own.get('nav_status','unknown')}",
        "target_role_histogram": dict(role_hist),
        "target_sector_histogram": dict(sector_hist),
        "target_status_histogram": dict(status_hist),
    }


def derive_interaction_structure(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    own = ss.get("ownship", {}) if isinstance(ss.get("ownship"), dict) else {}
    targets = ss.get("targets", []) if isinstance(ss.get("targets"), list) else []
    graph = ss.get("target_role_graph", {}) if isinstance(ss.get("target_role_graph"), dict) else {}
    own_x = safe_float(own.get("x_m"), 0.0) or 0.0
    own_y = safe_float(own.get("y_m"), 0.0) or 0.0
    own_h = safe_float(own.get("heading_deg"), 0.0) or 0.0

    sector_occ = Counter()
    distances: List[float] = []
    for t in targets:
        if not isinstance(t, dict):
            continue
        sec = str(t.get("relative_bearing_sector") or rel_bearing_sector(own_x, own_y, own_h, safe_float(t.get("x_m"), 0.0) or 0.0, safe_float(t.get("y_m"), 0.0) or 0.0))
        sector_occ[sec] += 1
        d = safe_float(t.get("distance_from_ownship_m"))
        if d is None:
            d = math.hypot((safe_float(t.get("x_m"), 0.0) or 0.0) - own_x, (safe_float(t.get("y_m"), 0.0) or 0.0) - own_y)
        distances.append(float(d))

    return {
        "spatial_topology": graph.get("topology"),
        "relative_bearing_occupancy": dict(sector_occ),
        "interaction_graph_density": row.get("family_latent", {}).get("interaction_density"),
        "min_distance_from_ownship_m": min(distances) if distances else None,
        "max_distance_from_ownship_m": max(distances) if distances else None,
    }


def derive_resolution_structure(row: Dict[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    debug_res = get_nested(row, "debug.reasoning.resolution", {})
    sensitivity = get_nested(row, "debug.reasoning.target_removal_sensitivity", {})
    explanation_steps = labels.get("explanation_steps", []) if isinstance(labels.get("explanation_steps"), list) else []
    contributing = list(debug_res.get("contributing_target_ids", [])) if isinstance(debug_res.get("contributing_target_ids"), list) else []

    return {
        "num_triggered_rules": len(labels.get("triggered_rules", [])) if isinstance(labels.get("triggered_rules"), list) else 0,
        "suppression_count": len(labels.get("suppression_records", [])) if isinstance(labels.get("suppression_records"), list) else 0,
        "retained_action_count": len(labels.get("maneuver_allowed", {})) if isinstance(labels.get("maneuver_allowed"), dict) else len(labels.get("maneuver_allowed", [])) if isinstance(labels.get("maneuver_allowed"), list) else 0,
        "forbidden_action_count": len(labels.get("maneuver_forbidden", {})) if isinstance(labels.get("maneuver_forbidden"), dict) else len(labels.get("maneuver_forbidden", [])) if isinstance(labels.get("maneuver_forbidden"), list) else 0,
        "global_resolution_nontrivial": bool(debug_res.get("global_resolution_nontrivial", False)),
        "contributing_target_ids": contributing,
        "target_removal_sensitive": bool(sensitivity.get("sensitive", False)),
        "target_removal_sensitive_ids": sensitivity.get("sensitive_target_ids", []),
        "explanation_stage_count": len(explanation_steps),
        "explanation_global_stage_nonempty": any(isinstance(x, dict) and str(x.get("stage")) == "global_stage" and str(x.get("text", "")).strip() for x in explanation_steps),
    }


def topology_alignment_ok(latent_topology: str, sector_hist: Dict[str, int]) -> bool:
    port = sector_hist.get("port", 0)
    star = sector_hist.get("starboard", 0)
    ahead = sector_hist.get("ahead", 0)
    astern = sector_hist.get("astern", 0)
    total = sum(sector_hist.values())
    if total <= 0:
        return False
    if latent_topology == "bilateral_constraint":
        return port >= 1 and star >= 1
    if latent_topology == "frontal_blocking":
        return ahead >= 1 and (port + star) >= 1
    if latent_topology == "surrounding_ring":
        return port >= 1 and star >= 1 and ahead >= 1 and astern >= 1
    if latent_topology == "single_side_cluster":
        return max(port, star) >= max(2, total - 1)
    return True


def derive_rule_signature(row: Dict[str, Any]) -> str:
    rules = row.get("labels", {}).get("triggered_rules", []) if isinstance(row.get("labels", {}), dict) else []
    coarse = []
    for r in rules:
        if not isinstance(r, str):
            continue
        if r.startswith("COLREG_R"):
            token = r.split("_", 2)[0]
        elif r.startswith("SUPPRESS::"):
            token = "SUPPRESS"
        else:
            token = r
        coarse.append(token)
    return "|".join(sorted(set(coarse)))


def derive_suppression_signature(row: Dict[str, Any]) -> str:
    records = row.get("labels", {}).get("suppression_records", []) if isinstance(row.get("labels", {}), dict) else []
    items = []
    for r in records:
        if not isinstance(r, dict):
            continue
        prim = str(r.get("primitive", "na"))
        src = str(r.get("source", "na"))
        items.append(f"{prim}::{src}")
    if not items:
        return "none"
    return "|".join(sorted(set(items)))


def derive_family_semantics(row: Dict[str, Any], scene_context: Dict[str, Any], vessel_population: Dict[str, Any], interaction_structure: Dict[str, Any], resolution_structure: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int], List[str], List[str]]:
    latent = row.get("family_latent", {}) if isinstance(row.get("family_latent"), dict) else {}
    targets = row.get("scene_spec", {}).get("targets", []) if isinstance(row.get("scene_spec", {}), dict) else []

    observed = {
        "effective_participant_count": vessel_population["effective_participants"],
        "spatial_topology": interaction_structure.get("spatial_topology"),
        "suppression_type": realized_suppression_type(row),
        "urgency_level": infer_urgency_from_distances(targets),
        "visibility_context": scene_context.get("visibility"),
        "vessel_status_mix": infer_target_status_mix(targets),
        "interaction_graph_density": interaction_structure.get("interaction_graph_density"),
    }
    family_stratum = copy.deepcopy(observed)
    latent_errors: List[str] = []
    latent_warnings: List[str] = []

    if safe_int(latent.get("num_effective_targets")) != safe_int(observed["effective_participant_count"]):
        latent_errors.append("effective_target_count_mismatch")
    if str(latent.get("spatial_topology")) != str(observed["spatial_topology"]):
        # allow topology based on graph label if sector realization still plausible
        if not topology_alignment_ok(str(latent.get("spatial_topology")), interaction_structure.get("relative_bearing_occupancy", {})):
            latent_errors.append("spatial_topology_not_realized")
        else:
            latent_warnings.append("spatial_topology_realized_via_sector_pattern")
    if str(latent.get("suppression_mechanism")) != str(observed["suppression_type"]):
        latent_warnings.append("suppression_type_shifted")
    if str(latent.get("urgency_band")) != str(observed["urgency_level"]):
        latent_warnings.append("urgency_proxy_shifted")
    if str(latent.get("visibility_mode")) != str(observed["visibility_context"]):
        latent_errors.append("visibility_mode_mismatch")
    if str(latent.get("target_status_mix")) != str(observed["vessel_status_mix"]):
        latent_warnings.append("target_status_mix_shifted")
    if str(latent.get("interaction_density")) != str(observed["interaction_graph_density"]):
        latent_warnings.append("interaction_density_shifted")

    alignment = {
        "latent": {
            "effective_participant_count": latent.get("num_effective_targets"),
            "spatial_topology": latent.get("spatial_topology"),
            "suppression_type": latent.get("suppression_mechanism"),
            "urgency_level": latent.get("urgency_band"),
            "visibility_context": latent.get("visibility_mode"),
            "vessel_status_mix": latent.get("target_status_mix"),
            "interaction_graph_density": latent.get("interaction_density"),
        },
        "observed": observed,
        "errors": latent_errors,
        "warnings": latent_warnings,
        "pass": len(latent_errors) == 0,
    }
    return {
        "family_stratum": family_stratum,
        "latent_alignment": alignment,
        "rule_signature": derive_rule_signature(row),
        "suppression_signature": derive_suppression_signature(row),
    }, Counter(latent_warnings), latent_errors, latent_warnings


def derive_quality_flags(row: Dict[str, Any], vessel_population: Dict[str, Any], resolution_structure: Dict[str, Any], family_semantics: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    if vessel_population["effective_participants"] < 3:
        errors.append("effective_participants_lt_3")
    if not resolution_structure["global_resolution_nontrivial"]:
        errors.append("global_resolution_not_nontrivial")
    if not resolution_structure["explanation_global_stage_nonempty"]:
        errors.append("explanation_global_stage_empty")
    if len(resolution_structure["contributing_target_ids"]) < 2:
        errors.append("single_target_dependency")
    if not resolution_structure["target_removal_sensitive"]:
        errors.append("target_removal_not_sensitive")

    if resolution_structure["suppression_count"] < 1:
        warnings.append("suppression_records_empty")
    if resolution_structure["retained_action_count"] + resolution_structure["forbidden_action_count"] < 2:
        warnings.append("low_action_cardinality")

    alignment = family_semantics["latent_alignment"]
    if not alignment["pass"]:
        errors.extend(list(alignment["errors"]))
    warnings.extend(list(alignment["warnings"]))

    quality = {
        "passes_family_invariants": len([e for e in errors if e in {
            "effective_participants_lt_3",
            "global_resolution_not_nontrivial",
            "explanation_global_stage_empty",
            "single_target_dependency",
            "target_removal_not_sensitive",
        }]) == 0,
        "passes_latent_alignment": alignment["pass"],
        "passes_release_readiness": len(errors) == 0,
        "rejection_reasons": errors,
        "warning_reasons": warnings,
    }
    return quality, errors, warnings


def build_generation_meta(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Counter, List[str], List[str]]:
    scene_context = derive_scene_context(row)
    vessel_population = derive_vessel_population(row)
    interaction_structure = derive_interaction_structure(row)
    resolution_structure = derive_resolution_structure(row)
    family_semantics, latent_warning_hist, latent_errors, latent_warnings = derive_family_semantics(
        row, scene_context, vessel_population, interaction_structure, resolution_structure
    )
    quality_flags, invariant_errors, invariant_warnings = derive_quality_flags(
        row, vessel_population, resolution_structure, family_semantics
    )
    gm = {
        "schema_version": GENMETA_VERSION,
        "generator": derive_generator_block(row),
        "scene_context": scene_context,
        "vessel_population": vessel_population,
        "interaction_structure": interaction_structure,
        "resolution_structure": resolution_structure,
        "family_semantics": family_semantics,
        "quality_flags": quality_flags,
    }
    warning_hist = Counter(latent_warning_hist)
    for w in invariant_warnings:
        warning_hist[w] += 1
    return gm, warning_hist, invariant_errors + latent_errors, invariant_warnings + latent_warnings


def stable_index_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    gm = sample.get("generation_meta", {}) if isinstance(sample.get("generation_meta"), dict) else {}
    return {
        "sample_id": sample.get("sample_id"),
        "pattern": sample.get("pattern"),
        "schema_version": gm.get("schema_version"),
        "family_stratum": get_nested(gm, "family_semantics.family_stratum", {}),
        "rule_signature": get_nested(gm, "family_semantics.rule_signature"),
        "suppression_signature": get_nested(gm, "family_semantics.suppression_signature"),
        "effective_participants": get_nested(gm, "vessel_population.effective_participants"),
        "quality_flags": get_nested(gm, "quality_flags", {}),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    ap.add_argument("--family", type=str, choices=[FAMILY], default=FAMILY)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate-schema", action="store_true")
    ap.add_argument("--report-json", type=str, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    ann_dir = root / "annotated_pool" / FAMILY
    ann_path = ann_dir / "candidates_annotated.jsonl"
    idx_path = ann_dir / "generation_meta_index.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if schema_path.exists() else {}

    annotated_rows: List[Dict[str, Any]] = []
    index_rows: List[Dict[str, Any]] = []
    schema_error_count = 0
    family_invariant_pass_count = 0
    latent_alignment_fail_count = 0
    target_removal_sensitive_count = 0
    rule_sig_set = set()
    suppress_sig_set = set()
    stratum_counter: Counter = Counter()
    warning_hist: Counter = Counter()
    failure_head: List[Dict[str, Any]] = []

    for row in rows:
        if args.validate_schema and schema:
            err = validate_against_schema(row, schema)
            if err:
                schema_error_count += 1

        new_row = copy.deepcopy(row)
        gm, wh, errors, warnings = build_generation_meta(new_row)
        new_row["generation_meta"] = gm
        annotated_rows.append(new_row)
        index_rows.append(stable_index_row(new_row))

        if gm["quality_flags"]["passes_family_invariants"]:
            family_invariant_pass_count += 1
        if not gm["quality_flags"]["passes_latent_alignment"]:
            latent_alignment_fail_count += 1
        if get_nested(gm, "resolution_structure.target_removal_sensitive", False):
            target_removal_sensitive_count += 1

        fs = get_nested(gm, "family_semantics.family_stratum", {})
        stratum_key = json.dumps(fs, sort_keys=True, ensure_ascii=False)
        stratum_counter[stratum_key] += 1

        rule_sig = get_nested(gm, "family_semantics.rule_signature")
        supp_sig = get_nested(gm, "family_semantics.suppression_signature")
        if rule_sig is not None:
            rule_sig_set.add(rule_sig)
        if supp_sig is not None:
            suppress_sig_set.add(supp_sig)

        warning_hist.update(wh)
        if errors and len(failure_head) < 10:
            failure_head.append({
                "sample_id": row.get("sample_id"),
                "errors": errors,
                "warnings": warnings,
                "family_latent": row.get("family_latent", {}),
            })

    if ann_path.exists() and not args.overwrite:
        raise FileExistsError(f"Annotated file exists, use --overwrite: {ann_path}")
    if idx_path.exists() and not args.overwrite:
        raise FileExistsError(f"Index file exists, use --overwrite: {idx_path}")

    write_jsonl(ann_path, annotated_rows)
    write_jsonl(idx_path, index_rows)

    top_family_strata = []
    for k, v in stratum_counter.most_common(10):
        top_family_strata.append({"family_stratum": json.loads(k), "count": v})

    report = {
        "family": FAMILY,
        "generator_version": GENERATOR_VERSION,
        "generation_meta_version": GENMETA_VERSION,
        "row_count": len(rows),
        "schema_error_count": schema_error_count,
        "family_invariant_pass_count": family_invariant_pass_count,
        "latent_alignment_fail_count": latent_alignment_fail_count,
        "target_removal_sensitive_count": target_removal_sensitive_count,
        "unique_family_strata_count": len(stratum_counter),
        "unique_rule_signature_count": len(rule_sig_set),
        "unique_suppression_signature_count": len(suppress_sig_set),
        "top_family_strata": top_family_strata,
        "warning_histogram": dict(sorted(warning_hist.items())),
        "failure_head": failure_head,
        "annotated_file": str(ann_path.relative_to(root)),
        "generation_meta_index_file": str(idx_path.relative_to(root)),
    }
    write_json(Path(args.report_json), report)

    print("=" * 100)
    print(f"annotate_native_generation_meta_restricted_multi.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows annotated            : {len(rows)}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {schema_error_count}")
    print(f"Family invariant pass    : {family_invariant_pass_count}/{len(rows)}")
    print(f"Latent alignment fail    : {latent_alignment_fail_count}")
    print(f"Target-removal sensitive : {target_removal_sensitive_count}/{len(rows)}")
    print(f"Unique family strata     : {len(stratum_counter)}")
    print(f"Unique rule signatures   : {len(rule_sig_set)}")
    print(f"Unique suppression sigs  : {len(suppress_sig_set)}")
    print(f"Annotated file           : {ann_path.relative_to(root)}")
    print(f"Index file               : {idx_path.relative_to(root)}")
    print(f"Report written           : {Path(args.report_json)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
