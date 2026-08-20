#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

try:
    import yaml
except Exception as e:
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml\n"
        f"Import error: {e}"
    )


FAMILIES = [
    "restricted_multi",
    "responsibility_mix",
    "channel_crossing_impede",
    "tss_crossing",
]


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


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


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


def as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def add_issue(issues: List[Dict[str, Any]], severity: str, scope: str, message: str) -> None:
    issues.append({
        "severity": severity,   # error | warning
        "scope": scope,
        "message": message,
    })


def list_to_set(xs: Any) -> Set[str]:
    if not isinstance(xs, list):
        return set()
    return {str(x) for x in xs}


def extract_schema_family_requirements(schema: Dict[str, Any]) -> Dict[str, Dict[str, Set[str]]]:
    """
    Parse family-specific requirements from raw_candidate.schema.json.
    We rely on the allOf/if/then blocks written in the current schema draft.
    """
    out: Dict[str, Dict[str, Set[str]]] = {
        fam: {
            "family_latent_required_fields": set(),
            "scene_spec_required_fields": set(),
            "area_context_required_fields": set(),
        }
        for fam in FAMILIES
    }

    for block in as_list(schema.get("allOf")):
        if not isinstance(block, dict):
            continue
        fam = get_nested(block, "if.properties.pattern.const")
        if fam not in out:
            continue

        out[fam]["family_latent_required_fields"] |= list_to_set(
            get_nested(block, "then.properties.family_latent.required", [])
        )
        out[fam]["scene_spec_required_fields"] |= list_to_set(
            get_nested(block, "then.properties.scene_spec.required", [])
        )
        out[fam]["area_context_required_fields"] |= list_to_set(
            get_nested(block, "then.properties.scene_spec.properties.area_context.required", [])
        )
    return out


# ============================================================
# Validators
# ============================================================

def validate_global_schema(schema_path: Path, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not schema_path.exists():
        add_issue(issues, "error", "schema", f"Missing schema file: {schema_path}")
        return {}

    try:
        schema = read_json(schema_path)
    except Exception as e:
        add_issue(issues, "error", "schema", f"Failed to parse schema JSON: {e}")
        return {}

    required_top = set(as_list(schema.get("required")))
    expected_top = {
        "sample_id",
        "pattern",
        "source_mode",
        "generator_version",
        "candidate_seed",
        "family_latent",
        "scene_spec",
        "inputs",
        "labels",
    }
    missing = expected_top - required_top
    if missing:
        add_issue(
            issues,
            "error",
            "schema",
            f"Schema missing required top-level fields: {sorted(missing)}"
        )

    pattern_enum = set(get_nested(schema, "properties.pattern.enum", []))
    if pattern_enum != set(FAMILIES):
        add_issue(
            issues,
            "warning",
            "schema",
            f"Schema pattern enum differs from expected families: got={sorted(pattern_enum)} expected={sorted(FAMILIES)}"
        )

    return schema


def validate_spec_common(spec: Dict[str, Any], family: str, path: Path, issues: List[Dict[str, Any]]) -> None:
    scope = f"spec:{family}"

    if spec.get("family_name") != family:
        add_issue(issues, "error", scope, f"family_name mismatch in {path.name}: {spec.get('family_name')} != {family}")

    if "generator_version" not in spec:
        add_issue(issues, "error", scope, "Missing generator_version")
    if "benchmark_goal" not in spec:
        add_issue(issues, "error", scope, "Missing benchmark_goal")

    latent_vars = as_list(spec.get("latent_variables"))
    if not latent_vars:
        add_issue(issues, "error", scope, "Missing latent_variables")
        return

    latent_names = []
    for i, lv in enumerate(latent_vars):
        if not isinstance(lv, dict):
            add_issue(issues, "error", scope, f"latent_variables[{i}] is not a dict")
            continue
        name = lv.get("name")
        if not name:
            add_issue(issues, "error", scope, f"latent_variables[{i}] missing name")
            continue
        latent_names.append(str(name))
        if "type" not in lv:
            add_issue(issues, "warning", scope, f"latent_variables[{i}] missing type for {name}")
        if lv.get("required") is True and "values" not in lv:
            add_issue(issues, "warning", scope, f"required latent variable {name} missing values")

    if len(latent_names) != len(set(latent_names)):
        add_issue(issues, "error", scope, "Duplicate latent variable names detected")

    soft_dist = spec.get("soft_target_distribution")
    if soft_dist is not None and not isinstance(soft_dist, dict):
        add_issue(issues, "error", scope, "soft_target_distribution must be a dict when present")

    if "scene_builder" not in spec:
        add_issue(issues, "error", scope, "Missing scene_builder")
    if "label_consistency_checks" not in spec:
        add_issue(issues, "error", scope, "Missing label_consistency_checks")
    if "raw_candidate_schema" not in spec:
        add_issue(issues, "error", scope, "Missing raw_candidate_schema")
    if "rejection_conditions" not in spec:
        add_issue(issues, "error", scope, "Missing rejection_conditions")


def validate_spec_against_schema(
    spec: Dict[str, Any],
    family: str,
    schema_reqs: Dict[str, Dict[str, Set[str]]],
    issues: List[Dict[str, Any]],
) -> None:
    scope = f"spec:{family}"

    raw_schema = spec.get("raw_candidate_schema", {})
    if not isinstance(raw_schema, dict):
        add_issue(issues, "error", scope, "raw_candidate_schema must be a dict")
        return

    spec_top = list_to_set(raw_schema.get("required_top_level_fields", []))
    required_schema_top = {
        "sample_id",
        "pattern",
        "source_mode",
        "generator_version",
        "candidate_seed",
        "family_latent",
        "scene_spec",
        "inputs",
        "labels",
    }
    missing_top = required_schema_top - spec_top
    if missing_top:
        add_issue(issues, "error", scope, f"raw_candidate_schema.required_top_level_fields missing: {sorted(missing_top)}")

    spec_latent = list_to_set(raw_schema.get("family_latent_required_fields", []))
    schema_latent = schema_reqs.get(family, {}).get("family_latent_required_fields", set())
    if schema_latent and spec_latent != schema_latent:
        add_issue(
            issues,
            "warning",
            scope,
            f"family_latent_required_fields differs from JSON schema. spec={sorted(spec_latent)} schema={sorted(schema_latent)}"
        )

    spec_scene = list_to_set(raw_schema.get("scene_spec_required_fields", []))
    schema_scene = schema_reqs.get(family, {}).get("scene_spec_required_fields", set())
    if schema_scene and not schema_scene.issubset(spec_scene):
        add_issue(
            issues,
            "warning",
            scope,
            f"scene_spec_required_fields missing schema-required keys: {sorted(schema_scene - spec_scene)}"
        )

    spec_area = list_to_set(raw_schema.get("area_context_required_fields", []))
    schema_area = schema_reqs.get(family, {}).get("area_context_required_fields", set())
    if schema_area and spec_area != schema_area:
        add_issue(
            issues,
            "warning",
            scope,
            f"area_context_required_fields differs from JSON schema. spec={sorted(spec_area)} schema={sorted(schema_area)}"
        )

    # Also check latent_variables names cover raw_candidate_schema.family_latent_required_fields
    latent_names = {str(x.get("name")) for x in as_list(spec.get("latent_variables")) if isinstance(x, dict) and x.get("name")}
    if not spec_latent.issubset(latent_names):
        add_issue(
            issues,
            "error",
            scope,
            f"Some family_latent_required_fields are not declared in latent_variables: {sorted(spec_latent - latent_names)}"
        )


def validate_family_specific_logic(spec: Dict[str, Any], family: str, issues: List[Dict[str, Any]]) -> None:
    scope = f"spec:{family}"
    label_checks = spec.get("label_consistency_checks", {})
    if not isinstance(label_checks, dict):
        add_issue(issues, "error", scope, "label_consistency_checks must be a dict")
        return

    # high-value semantic checks per family
    if family == "restricted_multi":
        arb = label_checks.get("arbitration_checks", {})
        if not isinstance(arb, dict):
            add_issue(issues, "error", scope, "restricted_multi missing arbitration_checks dict")
        else:
            for k in [
                "require_global_resolution_nontrivial",
                "require_suppression_or_conflict",
                "require_multiple_target_dependency",
                "target_removal_sensitivity",
            ]:
                if k not in arb:
                    add_issue(issues, "warning", scope, f"restricted_multi arbitration_checks missing {k}")

    elif family == "responsibility_mix":
        rc = label_checks.get("responsibility_checks", {})
        if not isinstance(rc, dict):
            add_issue(issues, "error", scope, "responsibility_mix missing responsibility_checks dict")
        else:
            for k in [
                "require_target_level_role_profiles",
                "require_role_structure_match",
                "require_status_combo_match",
                "class_status_ablation_changes_labels",
            ]:
                if k not in rc:
                    add_issue(issues, "warning", scope, f"responsibility_mix responsibility_checks missing {k}")

    elif family == "channel_crossing_impede":
        cc = label_checks.get("channel_checks", {})
        if not isinstance(cc, dict):
            add_issue(issues, "error", scope, "channel_crossing_impede missing channel_checks dict")
        else:
            for k in [
                "require_rule9_family",
                "require_channel_context_dependence",
                "channel_ablation_changes_labels",
                "semantic_mode_match",
            ]:
                if k not in cc:
                    add_issue(issues, "warning", scope, f"channel_crossing_impede channel_checks missing {k}")

    elif family == "tss_crossing":
        tc = label_checks.get("tss_checks", {})
        if not isinstance(tc, dict):
            add_issue(issues, "error", scope, "tss_crossing missing tss_checks dict")
        else:
            for k in [
                "require_rule10_family",
                "require_tss_structure_dependence",
                "tss_ablation_changes_labels",
                "overlap_mode_match",
                "crossing_angle_band_match",
            ]:
                if k not in tc:
                    add_issue(issues, "warning", scope, f"tss_crossing tss_checks missing {k}")


def validate_spec_file(
    spec_path: Path,
    family: str,
    schema_reqs: Dict[str, Dict[str, Set[str]]],
    issues: List[Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        spec = read_yaml(spec_path)
    except Exception as e:
        add_issue(issues, "error", f"spec:{family}", f"Failed to parse YAML: {e}")
        return {}

    validate_spec_common(spec, family, spec_path, issues)
    validate_spec_against_schema(spec, family, schema_reqs, issues)
    validate_family_specific_logic(spec, family, issues)
    return spec


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate native_generator_specs against structural expectations and raw_candidate.schema.json."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--report-json", type=str, default=None, help="Optional output report path")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors in exit code")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    specs_dir = root / "config" / "native_generator_specs"
    schema_path = root / "schemas" / "raw_candidate.schema.json"

    issues: List[Dict[str, Any]] = []
    schema = validate_global_schema(schema_path, issues)
    schema_reqs = extract_schema_family_requirements(schema) if schema else {}

    if not specs_dir.exists():
        add_issue(issues, "error", "specs", f"Missing native_generator_specs directory: {specs_dir}")
    else:
        for family in FAMILIES:
            spec_path = specs_dir / f"{family}.yaml"
            if not spec_path.exists():
                add_issue(issues, "error", f"spec:{family}", f"Missing spec file: {spec_path}")
                continue
            validate_spec_file(spec_path, family, schema_reqs, issues)

    error_count = sum(1 for x in issues if x["severity"] == "error")
    warning_count = sum(1 for x in issues if x["severity"] == "warning")

    report = {
        "root": str(root),
        "specs_dir": str(specs_dir),
        "schema_path": str(schema_path),
        "families_expected": FAMILIES,
        "error_count": error_count,
        "warning_count": warning_count,
        "issues": issues,
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "validate_native_generator_specs_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"validate_native_generator_specs.py | root={root}")
    print("=" * 100)
    print(f"Errors   : {error_count}")
    print(f"Warnings : {warning_count}")
    print(f"Report   : {report_path}")

    if issues:
        print("\n[Issues]")
        for item in issues:
            print(f"- [{item['severity']}] {item['scope']}: {item['message']}")
    else:
        print("\nNo spec/schema consistency issues detected.")

    if error_count > 0:
        return 1
    if args.strict and warning_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
