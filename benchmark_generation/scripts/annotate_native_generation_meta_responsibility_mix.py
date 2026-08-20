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


FAMILY = "responsibility_mix"


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


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def stable_index_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    gm = sample.get("generation_meta", {})
    return {
        "sample_id": sample.get("sample_id"),
        "pattern": sample.get("pattern"),
        "schema_version": gm.get("schema_version"),
        "family_stratum": get_nested(gm, "family_semantics.family_stratum", {}),
        "effective_participants": get_nested(gm, "vessel_population.effective_participants"),
        "role_structure": get_nested(gm, "family_semantics.family_stratum.role_structure"),
        "vessel_status_combo": get_nested(gm, "family_semantics.family_stratum.vessel_status_combo"),
        "encounter_mix": get_nested(gm, "family_semantics.family_stratum.encounter_mix"),
        "quality_flags": get_nested(gm, "quality_flags", {}),
    }


# ============================================================
# Family-specific inference
# ============================================================

def count_relation_types(role_graph: List[Dict[str, Any]]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for node in role_graph:
        rel = str(node.get("relation_type"))
        d[rel] = d.get(rel, 0) + 1
    return d


def count_roles(role_graph: List[Dict[str, Any]]) -> Dict[str, int]:
    d: Dict[str, int] = {}
    for node in role_graph:
        role = str(node.get("ownship_role_vs_target"))
        d[role] = d.get(role, 0) + 1
    return d


def infer_encounter_mix(role_graph: List[Dict[str, Any]]) -> str:
    rel = count_relation_types(role_graph)
    crossing = rel.get("crossing", 0)
    overtaking = rel.get("overtaking", 0)
    if crossing >= 1 and overtaking == 0:
        return "crossing_dominant"
    if overtaking >= 1 and crossing == 0:
        return "overtaking_dominant"
    if crossing >= 1 and overtaking >= 1:
        return "mixed_encounters"
    return "unknown"


def infer_role_structure(role_graph: List[Dict[str, Any]]) -> str:
    roles = count_roles(role_graph)
    uniq = set(roles.keys())
    if uniq == {"giveway"}:
        return "uniform_giveway"
    if uniq == {"standon"}:
        return "uniform_standon"
    if "hierarchy_priority" in uniq:
        return "hierarchy_conflict"
    if "giveway" in uniq and "standon" in uniq:
        return "mixed_roles"
    return "unknown"


def infer_vessel_status_combo(target_classes: List[str]) -> str:
    s = set(target_classes)
    if s == {"power_driven"}:
        return "power_only"
    if "power_driven" in s and "sailing" in s and len(s) == 2:
        return "power_sailing"
    if "power_driven" in s and "fishing" in s and len(s) == 2:
        return "power_fishing"
    if "power_driven" in s and "ram" in s and len(s) == 2:
        return "power_ram"
    if "power_driven" in s and "nuc" in s and len(s) == 2:
        return "power_nuc"
    if len(s) >= 2:
        return "mixed_multi_status"
    return "unknown"


def infer_urgency_level_from_targets(target_summaries: List[Dict[str, Any]]) -> str:
    min_tcpa = None
    min_cpa = None
    for ts in target_summaries:
        if not isinstance(ts, dict):
            continue
        tcpa = safe_float(ts.get("tcpa_s"))
        cpa = safe_float(ts.get("cpa_m"))
        if tcpa is not None:
            min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)
        if cpa is not None:
            min_cpa = cpa if min_cpa is None else min(min_cpa, cpa)

    if min_tcpa is not None and min_tcpa <= 45.0:
        return "high"
    if min_tcpa is not None and min_tcpa <= 120.0:
        return "medium"
    return "low"


def infer_target_status_histogram(targets: List[Dict[str, Any]]) -> Dict[str, int]:
    hist: Dict[str, int] = {}
    for t in targets:
        if not isinstance(t, dict):
            continue
        token = str(first_non_null(t.get("vessel_class"), t.get("nav_status"), "unknown"))
        hist[token] = hist.get(token, 0) + 1
    return hist


# ============================================================
# Annotation blocks
# ============================================================

def derive_generator_block(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "family_generator": row.get("generator_version", "responsibility_mix_native_v2"),
        "candidate_seed": row.get("candidate_seed"),
        "sampling_round": 1,
        "source_mode": row.get("source_mode", "native_v3_generator"),
        "config_hash": None,
        "family_config_version": row.get("generator_version", "responsibility_mix_native_v2"),
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
        "area_type": area.get("area_type", "open_water"),
        "channel_context": bool(area.get("channel_context", False)),
        "tss_context": bool(area.get("tss_context", False)),
        "snapshot_time_s": safe_int(ss.get("snapshot_time_s"), 30),
        "duration_s": safe_int(ss.get("duration_s"), 180),
    }


def derive_vessel_population(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {})
    ownship = ss.get("ownship", {}) if isinstance(ss, dict) else {}
    targets = ss.get("targets", []) if isinstance(ss, dict) else []
    role_graph = ss.get("target_role_graph", []) if isinstance(ss, dict) else []

    ownship_status = str(first_non_null(ownship.get("vessel_class"), ownship.get("nav_status"), "power_driven"))
    target_status_hist = infer_target_status_histogram(targets)
    role_counts = count_roles(role_graph if isinstance(role_graph, list) else [])

    return {
        "total_vessels": len(targets) + 1,
        "target_vessels": len(targets),
        "effective_participants": len(targets),
        "ownship_status": ownship_status,
        "target_status_histogram": target_status_hist,
        "target_count_by_role": {
            "giveway_targets": role_counts.get("giveway", 0),
            "standon_targets": role_counts.get("standon", 0),
            "hierarchy_priority_targets": role_counts.get("hierarchy_priority", 0),
            "other_targets": max(0, len(targets) - role_counts.get("giveway", 0) - role_counts.get("standon", 0) - role_counts.get("hierarchy_priority", 0)),
        },
    }


def derive_interaction_structure(row: Dict[str, Any], vessel_population: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {})
    role_graph = ss.get("target_role_graph", []) if isinstance(ss, dict) else []
    labels = row.get("labels", {})
    target_summaries = labels.get("target_summaries", []) if isinstance(labels, dict) else []

    relation_hist = count_relation_types(role_graph if isinstance(role_graph, list) else [])
    inferred_encounter_mix = infer_encounter_mix(role_graph if isinstance(role_graph, list) else [])
    inferred_urgency = infer_urgency_level_from_targets(target_summaries if isinstance(target_summaries, list) else [])

    # coarse spatial topology proxy from relation mix
    spatial_topology = "role_cluster"
    if relation_hist.get("hierarchy", 0) > 0 and relation_hist.get("crossing", 0) > 0:
        spatial_topology = "hierarchy_plus_crossing"
    elif relation_hist.get("crossing", 0) > 0 and relation_hist.get("overtaking", 0) > 0:
        spatial_topology = "mixed_crossing_overtaking"
    elif relation_hist.get("overtaking", 0) > 0:
        spatial_topology = "overtaking_chain"
    elif relation_hist.get("crossing", 0) > 0:
        spatial_topology = "crossing_cluster"

    min_cpa = None
    min_tcpa = None
    for ts in target_summaries if isinstance(target_summaries, list) else []:
        if not isinstance(ts, dict):
            continue
        cpa = safe_float(ts.get("cpa_m"))
        tcpa = safe_float(ts.get("tcpa_s"))
        if cpa is not None:
            min_cpa = cpa if min_cpa is None else min(min_cpa, cpa)
        if tcpa is not None:
            min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)

    return {
        "encounter_mode": inferred_encounter_mix,
        "relation_type_histogram": relation_hist,
        "role_structure_observed": infer_role_structure(role_graph if isinstance(role_graph, list) else []),
        "spatial_topology": spatial_topology,
        "interaction_graph_density": "dense" if vessel_population["effective_participants"] >= 4 else "medium" if vessel_population["effective_participants"] >= 3 else "sparse",
        "min_cpa_m": min_cpa,
        "min_tcpa_s": min_tcpa,
        "urgency_level_observed": inferred_urgency,
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

    has_r13 = any(str(r).startswith("COLREG_R13") for r in triggered_rules)
    has_r15 = any(str(r).startswith("COLREG_R15") for r in triggered_rules)
    has_r17 = any(str(r).startswith("COLREG_R17") for r in triggered_rules)
    has_r18 = any(str(r).startswith("COLREG_R18") for r in triggered_rules)

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
        "global_resolution_nontrivial": any([has_r13, has_r15, has_r17, has_r18]),
    }


def derive_family_semantics(
    row: Dict[str, Any],
    scene_context: Dict[str, Any],
    vessel_population: Dict[str, Any],
    interaction_structure: Dict[str, Any],
    resolution_structure: Dict[str, Any],
) -> Dict[str, Any]:
    latent = row.get("family_latent", {})
    ss = row.get("scene_spec", {})
    targets = ss.get("targets", []) if isinstance(ss, dict) else []
    role_graph = ss.get("target_role_graph", []) if isinstance(ss, dict) else []

    target_classes = [str(t.get("vessel_class")) for t in targets if isinstance(t, dict)]

    latent_stratum = {
        "role_structure": latent.get("role_structure"),
        "vessel_status_combo": latent.get("target_status_combo"),
        "effective_target_count": safe_int(latent.get("effective_target_count")),
        "encounter_mix": latent.get("encounter_mix"),
        "urgency_band": latent.get("urgency_band"),
    }

    observed_stratum = {
        "role_structure": infer_role_structure(role_graph if isinstance(role_graph, list) else []),
        "vessel_status_combo": infer_vessel_status_combo(target_classes),
        "effective_target_count": len(targets),
        "encounter_mix": interaction_structure.get("encounter_mode"),
        "urgency_band": interaction_structure.get("urgency_level_observed"),
    }

    latent_alignment = {
        "role_structure_match": latent_stratum["role_structure"] == observed_stratum["role_structure"],
        "vessel_status_combo_match": latent_stratum["vessel_status_combo"] == observed_stratum["vessel_status_combo"],
        "effective_target_count_match": latent_stratum["effective_target_count"] == observed_stratum["effective_target_count"],
        "encounter_mix_match": latent_stratum["encounter_mix"] == observed_stratum["encounter_mix"],
        "target_role_graph_present": isinstance(role_graph, list) and len(role_graph) == len(targets),
    }

    latent_urgency = latent.get("urgency_band")
    observed_urgency = interaction_structure.get("urgency_level_observed")
    latent_alignment["urgency_band_match"] = (latent_urgency == observed_urgency)

    family_invariant_pass = all(latent_alignment.values())

    semantic_tags = [
        "role_graph",
        str(observed_stratum["role_structure"]),
        str(observed_stratum["vessel_status_combo"]),
        str(observed_stratum["encounter_mix"]),
        str(observed_stratum["urgency_band"]),
        f"n_targets_{observed_stratum['effective_target_count']}",
    ]

    return {
        "family_invariant_pass": family_invariant_pass,
        "family_stratum": observed_stratum,
        "semantic_tags": semantic_tags,
        "latent_alignment": latent_alignment,
        "native_latent_snapshot": copy.deepcopy(latent),
        "warnings": [],
    }


def derive_signatures(row: Dict[str, Any], interaction_structure: Dict[str, Any], resolution_structure: Dict[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels", {})
    ss = row.get("scene_spec", {})
    role_graph = ss.get("target_role_graph", []) if isinstance(ss, dict) else []
    targets = ss.get("targets", []) if isinstance(ss, dict) else []

    geometry_signature = {
        "effective_participants": len(targets),
        "relation_histogram": interaction_structure["relation_type_histogram"],
        "spatial_topology": interaction_structure["spatial_topology"],
        "urgency_level_observed": interaction_structure["urgency_level_observed"],
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

    role_graph_signature = {
        "target_role_graph": copy.deepcopy(role_graph if isinstance(role_graph, list) else []),
    }

    return {
        "geometry_signature": geometry_signature,
        "rule_signature": rule_signature,
        "resolution_signature": resolution_signature,
        "role_graph_signature": role_graph_signature,
        "visual_signature": {
            "layout_hash": None,
            "radar_hash": None,
        },
    }


def derive_quality_flags(family_semantics: Dict[str, Any]) -> Dict[str, Any]:
    warnings = family_semantics.get("warnings", [])
    return {
        "passes_invariant_gate": bool(family_semantics.get("family_invariant_pass")),
        "passes_plausibility_gate": None,
        "passes_duplicate_gate": None,
        "passes_diversity_gate": None,
        "accepted_into_candidate_pool": None,
        "accepted_into_release_split": False,
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
        "rejection_reasons": [] if family_semantics.get("family_invariant_pass") else ["native_latent_alignment_failed"],
    }


def annotate_native_responsibility_mix_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(row)

    scene_context = derive_scene_context(out)
    vessel_population = derive_vessel_population(out)
    interaction_structure = derive_interaction_structure(out, vessel_population)
    resolution_structure = derive_resolution_structure(out, vessel_population)
    family_semantics = derive_family_semantics(
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
        description="Annotate native generation meta (first version: responsibility_mix only)."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="responsibility_mix", choices=["responsibility_mix"])
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
    warning_count_total = 0
    strata_hist: Dict[str, int] = {}

    for row in rows:
        sample_id = row.get("sample_id", "<missing_sample_id>")

        if args.validate_schema:
            schema_err = validate_against_schema(row, schema)
            if schema_err is not None:
                schema_errors.append({"sample_id": sample_id, "error": schema_err})

        ann = annotate_native_responsibility_mix_row(row)
        annotated_rows.append(ann)

        gm = ann.get("generation_meta", {})
        fam_sem = gm.get("family_semantics", {})
        if fam_sem.get("family_invariant_pass") is True:
            invariant_pass_count += 1
        else:
            latent_alignment_fail_count += 1

        warnings = fam_sem.get("warnings", [])
        if isinstance(warnings, list):
            warning_count_total += len(warnings)

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
        "warning_count_total": warning_count_total,
        "unique_family_strata_count": len(strata_hist),
        "top_family_strata": sorted(
            [{"stratum": json.loads(k), "count": v} for k, v in strata_hist.items()],
            key=lambda x: (-x["count"], json.dumps(x["stratum"], sort_keys=True))
        )[:20],
        "notes": (
            "This script is native-line annotation for responsibility_mix. "
            "It standardizes explicit family_latent / target_role_graph / labels into generation_meta "
            "and checks latent alignment. urgency_band is enforced in v3."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "family_reports" / "responsibility_mix_native_annotation_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"annotate_native_generation_meta_responsibility_mix.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows                     : {len(rows)}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {len(schema_errors)}")
    print(f"Invariant pass           : {invariant_pass_count}/{len(rows)}")
    print(f"Warnings total           : {warning_count_total}")
    print(f"Unique strata            : {len(strata_hist)}")
    print(f"Annotated output         : {annotated_path}")
    print(f"Meta index               : {index_path}")
    print(f"Report                   : {report_path}")
    print("=" * 100)

    return 0 if (len(schema_errors) == 0 and invariant_pass_count == len(rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
