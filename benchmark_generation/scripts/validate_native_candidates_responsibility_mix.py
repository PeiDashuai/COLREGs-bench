#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


FAMILY = "responsibility_mix"


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
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object: {path} line {lineno}")
            rows.append(obj)
    return rows


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def class_combo_ok(combo: str, target_classes: List[str]) -> bool:
    s = set(target_classes)
    if combo == "power_only":
        return s == {"power_driven"}
    if combo == "power_sailing":
        return "sailing" in s and "power_driven" in s
    if combo == "power_fishing":
        return "fishing" in s and "power_driven" in s
    if combo == "power_ram":
        return "ram" in s and "power_driven" in s
    if combo == "power_nuc":
        return "nuc" in s and "power_driven" in s
    if combo == "mixed_multi_status":
        return len(s) >= 2
    return False


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


def validate_row_semantics(row: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []
    debug: Dict[str, Any] = {}

    family_latent = row.get("family_latent", {})
    scene_spec = row.get("scene_spec", {})
    labels = row.get("labels", {})

    pattern = row.get("pattern")
    if pattern != FAMILY:
        errors.append("pattern_not_responsibility_mix")

    role_graph = scene_spec.get("target_role_graph", []) if isinstance(scene_spec, dict) else []
    targets = scene_spec.get("targets", []) if isinstance(scene_spec, dict) else []
    target_summaries = labels.get("target_summaries", []) if isinstance(labels, dict) else []
    triggered_rules = labels.get("triggered_rules", []) if isinstance(labels, dict) else []

    if not isinstance(role_graph, list) or len(role_graph) == 0:
        errors.append("missing_target_role_graph")
        role_graph = []
    if not isinstance(targets, list):
        errors.append("targets_not_list")
        targets = []
    if not isinstance(target_summaries, list):
        errors.append("target_summaries_not_list")
        target_summaries = []
    if not isinstance(triggered_rules, list):
        errors.append("triggered_rules_not_list")
        triggered_rules = []

    latent_count = int(family_latent.get("effective_target_count", -1))
    latent_role_structure = str(family_latent.get("role_structure"))
    latent_status_combo = str(family_latent.get("target_status_combo"))
    latent_encounter_mix = str(family_latent.get("encounter_mix"))
    latent_urgency_band = str(family_latent.get("urgency_band"))

    if latent_count != len(targets):
        errors.append("effective_target_count_mismatch_targets")
    if latent_count != len(role_graph):
        errors.append("effective_target_count_mismatch_role_graph")
    if len(target_summaries) != len(targets):
        errors.append("target_summaries_count_mismatch")

    inferred_role_structure = infer_role_structure(role_graph)
    inferred_encounter_mix = infer_encounter_mix(role_graph)
    target_classes = [str(t.get("vessel_class")) for t in targets if isinstance(t, dict)]

    debug["inferred_role_structure"] = inferred_role_structure
    debug["inferred_encounter_mix"] = inferred_encounter_mix
    debug["target_classes"] = target_classes

    if latent_role_structure != inferred_role_structure:
        errors.append("role_structure_mismatch")

    if not class_combo_ok(latent_status_combo, target_classes):
        errors.append("target_status_combo_mismatch")

    if latent_encounter_mix != inferred_encounter_mix:
        errors.append("encounter_mix_mismatch")

    roles = count_roles(role_graph)
    rels = count_relation_types(role_graph)

    if latent_role_structure == "hierarchy_conflict":
        if roles.get("hierarchy_priority", 0) < 1:
            errors.append("hierarchy_conflict_missing_priority_target")
        if (roles.get("giveway", 0) + roles.get("standon", 0)) < 1:
            errors.append("hierarchy_conflict_missing_nonpriority_target")

    if latent_role_structure == "mixed_roles":
        if roles.get("giveway", 0) < 1 or roles.get("standon", 0) < 1:
            errors.append("mixed_roles_missing_role_diversity")

    if latent_role_structure == "uniform_giveway":
        non_gw = len(role_graph) - roles.get("giveway", 0)
        if non_gw != 0:
            errors.append("uniform_giveway_not_uniform")

    if latent_role_structure == "uniform_standon":
        non_so = len(role_graph) - roles.get("standon", 0)
        if non_so != 0:
            errors.append("uniform_standon_not_uniform")

    has_r13 = any(str(r).startswith("COLREG_R13") for r in triggered_rules)
    has_r15 = any(str(r).startswith("COLREG_R15") for r in triggered_rules)
    has_r17 = any(str(r).startswith("COLREG_R17") for r in triggered_rules)
    has_r18 = any(str(r).startswith("COLREG_R18") for r in triggered_rules)

    # Overtaking semantics should map to R13, but do not require R17 there.
    if rels.get("overtaking", 0) > 0 and not has_r13:
        errors.append("missing_rule13_for_overtaking_relations")

    # Crossing giveway requires R15/16.
    if any(str(n.get("relation_type")) == "crossing" and str(n.get("ownship_role_vs_target")) == "giveway" for n in role_graph):
        if not has_r15:
            errors.append("missing_rule15_for_crossing_giveway_roles")

    # Only require R17 for crossing + standon, not all standon.
    if any(str(n.get("relation_type")) == "crossing" and str(n.get("ownship_role_vs_target")) == "standon" for n in role_graph):
        if not has_r17:
            errors.append("missing_rule17_for_crossing_standon_roles")

    if any(str(n.get("relation_type")) == "hierarchy" for n in role_graph) or latent_status_combo in {"power_sailing", "power_fishing", "power_ram", "power_nuc", "mixed_multi_status"}:
        if not has_r18:
            errors.append("missing_rule18_for_status_hierarchy")

    observed_urgency_band = "unknown"
    min_tcpa = None
    for ts in target_summaries:
        if not isinstance(ts, dict):
            continue
        tcpa = ts.get("tcpa_s")
        try:
            tcpa = float(tcpa)
        except Exception:
            tcpa = None
        if tcpa is not None:
            min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)
    if min_tcpa is not None and min_tcpa <= 45.0:
        observed_urgency_band = "high"
    elif min_tcpa is not None and min_tcpa <= 120.0:
        observed_urgency_band = "medium"
    else:
        observed_urgency_band = "low"
    debug["observed_urgency_band"] = observed_urgency_band
    if latent_urgency_band in {"low", "medium", "high"} and latent_urgency_band != observed_urgency_band:
        errors.append("urgency_band_mismatch")

    return errors, warnings, debug


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate native raw candidates for responsibility_mix (v2)."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="responsibility_mix", choices=["responsibility_mix"])
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
                    "target_role_graph": row.get("scene_spec", {}).get("target_role_graph", []),
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
            "This v2 validator checks role_structure / status_combo / encounter_mix alignment. "
            "R17 is only required for crossing+standon roles. urgency_band is enforced in v3."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "validate_native_candidates_responsibility_mix_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"validate_native_candidates_responsibility_mix.py | root={root} | family={FAMILY}")
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
