#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


FAMILY = "responsibility_mix"
LABELER_VERSION = "responsibility_mix_labeler_v2_1"


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Helpers
# ============================================================

def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def deg_to_unit_vec(heading_deg: float) -> Tuple[float, float]:
    r = math.radians(heading_deg)
    return math.cos(r), math.sin(r)


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


def cpa_tcpa(
    own_x: float, own_y: float, own_heading_deg: float, own_speed: float,
    tgt_x: float, tgt_y: float, tgt_heading_deg: float, tgt_speed: float,
) -> Tuple[float, float]:
    ovx_u, ovy_u = deg_to_unit_vec(own_heading_deg)
    tvx_u, tvy_u = deg_to_unit_vec(tgt_heading_deg)
    ovx, ovy = ovx_u * own_speed, ovy_u * own_speed
    tvx, tvy = tvx_u * tgt_speed, tvy_u * tgt_speed

    rx = tgt_x - own_x
    ry = tgt_y - own_y
    rvx = tvx - ovx
    rvy = tvy - ovy
    rv2 = rvx * rvx + rvy * rvy

    if rv2 <= 1e-9:
        return math.hypot(rx, ry), 0.0

    tcpa = - (rx * rvx + ry * rvy) / rv2
    if tcpa < 0:
        tcpa = 0.0
    cpa_x = rx + rvx * tcpa
    cpa_y = ry + rvy * tcpa
    return math.hypot(cpa_x, cpa_y), tcpa


def unique_keep_order(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def load_adapter(adapter_spec: Optional[str]) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if not adapter_spec:
        return None
    if ":" not in adapter_spec:
        raise ValueError("Adapter spec must be in form module.submodule:function_name")
    module_name, func_name = adapter_spec.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Adapter callable not found: {adapter_spec}")
    return fn


def infer_urgency_level(target_summaries: List[Dict[str, Any]]) -> str:
    min_tcpa = None
    for ts in target_summaries:
        if not isinstance(ts, dict):
            continue
        tcpa = safe_float(ts.get("tcpa_s"))
        if tcpa is not None:
            min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)
    if min_tcpa is not None and min_tcpa <= 45.0:
        return "high"
    if min_tcpa is not None and min_tcpa <= 120.0:
        return "medium"
    return "low"


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


def old_target_summary_by_id(old_labels: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ts in old_labels.get("target_summaries", []) if isinstance(old_labels, dict) else []:
        if isinstance(ts, dict):
            tid = ts.get("target_id")
            if tid is not None:
                out[str(tid)] = ts
    return out


# ============================================================
# Built-in contract-preserving responsibility_mix labeler v2.1
# ============================================================

def build_target_summary(
    ownship: Dict[str, Any],
    target: Dict[str, Any],
    node: Dict[str, Any],
    previous: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ox, oy, oh = float(ownship["x_m"]), float(ownship["y_m"]), float(ownship["heading_deg"])
    tx, ty = float(target["x_m"]), float(target["y_m"])
    th = float(target["heading_deg"])
    os = float(ownship["speed_mps"])
    ts = float(target["speed_mps"])

    sector = rel_bearing_sector(ox, oy, oh, tx, ty)
    geom_cpa, geom_tcpa = cpa_tcpa(ox, oy, oh, os, tx, ty, th, ts)

    role = str(node["ownship_role_vs_target"])
    relation_type = str(node["relation_type"])
    vessel_class = str(node["target_class"])

    trg_rules: List[str] = []
    if relation_type == "crossing":
        if role == "giveway":
            trg_rules += ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD", "COLREG_R16_GIVEWAY_MANEUVER"]
        elif role == "standon":
            trg_rules += ["COLREG_R17_STANDON"]
    elif relation_type == "overtaking":
        trg_rules += ["COLREG_R13_OVERTAKING"]
    elif relation_type == "hierarchy":
        trg_rules += ["COLREG_R18_RESPONSIBILITIES"]

    if vessel_class in {"ram", "nuc", "fishing", "sailing", "constrained_by_draft"} and "COLREG_R18_RESPONSIBILITIES" not in trg_rules:
        trg_rules += ["COLREG_R18_RESPONSIBILITIES"]

    # Preserve generator/validator-aligned urgency proxies when available.
    # This is the critical fix for post-label urgency mismatch.
    cpa = safe_float(previous.get("cpa_m")) if previous else None
    tcpa = safe_float(previous.get("tcpa_s")) if previous else None
    if cpa is None:
        cpa = round(geom_cpa, 2)
    if tcpa is None:
        tcpa = round(min(180.0, geom_tcpa), 2)

    risk = previous.get("risk_of_collision") if previous else None
    closing = previous.get("closing") if previous else None
    imminent = previous.get("collision_imminent") if previous else None
    if risk is None:
        risk = bool(tcpa > 0 and tcpa <= 180.0 and cpa <= 1000.0)
    if closing is None:
        closing = bool(tcpa > 0)
    if imminent is None:
        imminent = bool(tcpa > 0 and tcpa <= 45.0 and cpa <= 250.0)

    return {
        "target_id": str(target.get("target_id", target.get("id"))),
        "triggered_rule_ids": unique_keep_order(trg_rules),
        "relative_bearing_sector": sector,
        "cpa_m": round(float(cpa), 2),
        "tcpa_s": round(float(tcpa), 2),
        "risk_of_collision": bool(risk),
        "closing": bool(closing),
        "collision_imminent": bool(imminent),
    }


def authoritative_responsibility_mix_labels_v2_1(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    family_latent = row.get("family_latent", {})
    scene_spec = row.get("scene_spec", {})
    ownship = scene_spec.get("ownship", {}) if isinstance(scene_spec, dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec, dict) else []
    role_graph = scene_spec.get("target_role_graph", []) if isinstance(scene_spec, dict) else []
    old_labels = row.get("labels", {}) if isinstance(row.get("labels", {}), dict) else {}

    role_structure = str(family_latent.get("role_structure"))
    target_status_combo = str(family_latent.get("target_status_combo"))
    encounter_mix = str(family_latent.get("encounter_mix"))
    latent_urgency_band = str(family_latent.get("urgency_band"))

    aggregate_rules = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
        "COLREG_R08_POSITIVE_ACTION_WHEN_RISK",
    ]

    old_ts_by_id = old_target_summary_by_id(old_labels)
    target_summaries: List[Dict[str, Any]] = []
    for node, tgt in zip(role_graph, targets):
        prev = old_ts_by_id.get(str(tgt.get("target_id", tgt.get("id"))))
        ts = build_target_summary(ownship, tgt, node, previous=prev)
        target_summaries.append(ts)
        aggregate_rules.extend(ts["triggered_rule_ids"])
    aggregate_rules = unique_keep_order(aggregate_rules)

    observed_urgency = infer_urgency_level(target_summaries)

    allowed = ["KEEP_LOOKOUT", "PROCEED_SAFE_SPEED"]
    forbidden: List[str] = []

    # Preserve urgency semantics in final maneuver set.
    if observed_urgency == "high":
        allowed += ["TAKE_IMMEDIATE_AVOIDING_ACTION", "PRIORITIZE_CLOSEST_RISK"]
        forbidden += ["DEFER_ACTION"]
    elif observed_urgency == "medium":
        allowed += ["TAKE_EARLY_SUBSTANTIAL_ACTION", "ROLE_AWARE_ACTION_SELECTION"]
    else:
        allowed += ["MONITOR_AND_PREPARE", "ROLE_AWARE_ACTION_SELECTION"]

    if role_structure == "uniform_giveway":
        allowed += ["GIVE_WAY_IF_REQUIRED"]
    elif role_structure == "uniform_standon":
        allowed += ["MAINTAIN_STANDON_IF_SAFE"]
    elif role_structure == "mixed_roles":
        forbidden += ["USE_SINGLE_TARGET_TEMPLATE"]
    else:
        allowed += ["PRIORITIZE_HIGHER_ORDER_RESPONSIBILITY"]
        forbidden += ["IGNORE_PRIORITY_TARGET"]

    explanation_steps = [
        {"stage": "roles", "summary": f"Role structure is {role_structure}.", "text": f"Role structure is {role_structure}."},
        {"stage": "status", "summary": f"Target status combo is {target_status_combo}.", "text": f"Target status combo is {target_status_combo}."},
        {"stage": "encounter", "summary": f"Encounter mix is {encounter_mix}.", "text": f"Encounter mix is {encounter_mix}."},
        {"stage": "urgency", "summary": f"Urgency is {observed_urgency} (latent {latent_urgency_band}).", "text": f"Urgency is {observed_urgency} (latent {latent_urgency_band})."},
        {"stage": "action", "summary": "Final maneuver set preserves responsibility structure, status hierarchy, and urgency level.", "text": "Final maneuver set preserves responsibility structure, status hierarchy, and urgency level."},
    ]

    labels = {
        "triggered_rules": aggregate_rules,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": {
            "maneuver": {
                "ROLE_RESPONSIBILITY_CONTROL": {
                    "polarities": ["allow"],
                    "support_rules": aggregate_rules,
                },
                "URGENCY_AWARE_ACTION_SCALING": {
                    "polarities": ["allow"],
                    "support_rules": ["COLREG_R06_SAFE_SPEED", "COLREG_R07_RISK_OF_COLLISION_ASSESS", "COLREG_R08_POSITIVE_ACTION_WHEN_RISK"],
                    "observed_urgency": observed_urgency,
                },
            }
        },
        "suppressed_rules": [],
        "suppression_records": [],
        "explanation_steps": explanation_steps,
        "maneuver_allowed": unique_keep_order(allowed),
        "maneuver_forbidden": unique_keep_order(forbidden),
        "lights_required": [],
        "sounds_required": [],
    }

    target_classes = [str(t.get("vessel_class")) for t in targets if isinstance(t, dict)]
    inferred_role_structure = infer_role_structure(role_graph if isinstance(role_graph, list) else [])
    inferred_encounter_mix = infer_encounter_mix(role_graph if isinstance(role_graph, list) else [])
    inferred_status_combo = infer_vessel_status_combo(target_classes)
    inferred_urgency = infer_urgency_level(target_summaries)

    self_check_errors: List[str] = []
    if inferred_role_structure != role_structure:
        self_check_errors.append("internal_role_structure_contract_mismatch")
    if inferred_encounter_mix != encounter_mix:
        self_check_errors.append("internal_encounter_mix_contract_mismatch")
    if inferred_status_combo != target_status_combo:
        self_check_errors.append("internal_status_combo_contract_mismatch")
    if latent_urgency_band in {"low", "medium", "high"} and inferred_urgency != latent_urgency_band:
        self_check_errors.append("internal_urgency_contract_mismatch")

    aux = {
        "label_source": LABELER_VERSION,
        "observed_role_structure": inferred_role_structure,
        "observed_encounter_mix": inferred_encounter_mix,
        "observed_status_combo": inferred_status_combo,
        "observed_urgency_level": inferred_urgency,
    }
    return labels, aux, self_check_errors


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Label native candidates (responsibility_mix v2.1 urgency-preserving).")
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="responsibility_mix", choices=["responsibility_mix"])
    ap.add_argument("--mode", type=str, default="built_in", choices=["built_in", "adapter"])
    ap.add_argument("--adapter", type=str, default=None, help="Optional adapter spec module:function")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite raw file in place")
    ap.add_argument("--no-backup", action="store_true", help="Do not create backup copy")
    ap.add_argument("--validate-schema", action="store_true")
    ap.add_argument("--report-json", type=str, default=None)
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
    if not args.overwrite:
        print("ERROR: this script updates labels in place; pass --overwrite to proceed.", file=sys.stderr)
        return 2
    if args.validate_schema and not schema_path.exists():
        print(f"ERROR: missing schema file required by --validate-schema: {schema_path}", file=sys.stderr)
        return 2

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if args.validate_schema and schema_path.exists() else {}

    adapter_fn = None
    if args.mode == "adapter":
        try:
            adapter_fn = load_adapter(args.adapter)
        except Exception as e:
            print(f"ERROR: failed to load adapter: {e}", file=sys.stderr)
            return 2

    backup_path = raw_path.with_name("candidates_raw.before_authoritative_labels.jsonl")
    if not args.no_backup:
        shutil.copy2(raw_path, backup_path)

    rewritten_rows: List[Dict[str, Any]] = []
    schema_errors: List[Dict[str, Any]] = []
    internal_contract_errors: List[Dict[str, Any]] = []
    built_in_count = 0
    adapter_count = 0

    for row in rows:
        new_row = copy.deepcopy(row)
        sample_id = new_row.get("sample_id", "<missing_sample_id>")
        old_labels = copy.deepcopy(new_row.get("labels", {}))

        if args.mode == "adapter":
            assert adapter_fn is not None
            labels = adapter_fn(new_row)
            if not isinstance(labels, dict):
                print(f"ERROR: adapter returned non-dict labels for {sample_id}", file=sys.stderr)
                return 2
            aux = {"label_source": "adapter"}
            self_check_errors: List[str] = []
            adapter_count += 1
        else:
            labels, aux, self_check_errors = authoritative_responsibility_mix_labels_v2_1(new_row)
            built_in_count += 1

        new_row["labels"] = labels
        debug = new_row.get("debug", {})
        if not isinstance(debug, dict):
            debug = {}
        debug["native_labeling"] = {
            "mode": args.mode,
            "labeler_version": LABELER_VERSION,
            "previous_triggered_rules": old_labels.get("triggered_rules", []) if isinstance(old_labels, dict) else [],
            "previous_target_count": len(old_labels.get("target_summaries", [])) if isinstance(old_labels, dict) else 0,
            **aux,
        }
        new_row["debug"] = debug

        if self_check_errors:
            internal_contract_errors.append({
                "sample_id": sample_id,
                "errors": self_check_errors,
                "latent_role_structure": new_row.get("family_latent", {}).get("role_structure"),
                "latent_target_status_combo": new_row.get("family_latent", {}).get("target_status_combo"),
                "latent_encounter_mix": new_row.get("family_latent", {}).get("encounter_mix"),
                "latent_urgency_band": new_row.get("family_latent", {}).get("urgency_band"),
                "written_triggered_rules": labels.get("triggered_rules", []),
            })

        if args.validate_schema:
            schema_err = validate_against_schema(new_row, schema)
            if schema_err is not None:
                schema_errors.append({"sample_id": sample_id, "error": schema_err})

        rewritten_rows.append(new_row)

    write_jsonl(raw_path, rewritten_rows)

    report = {
        "root": str(root),
        "family": FAMILY,
        "mode": args.mode,
        "adapter": args.adapter,
        "raw_file_updated": str(raw_path),
        "backup_file": None if args.no_backup else str(backup_path),
        "row_count": len(rows),
        "built_in_count": built_in_count,
        "adapter_count": adapter_count,
        "labeler_version": LABELER_VERSION,
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_validation_enabled": bool(args.validate_schema),
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:20],
        "internal_contract_error_count": len(internal_contract_errors),
        "internal_contract_errors_head": internal_contract_errors[:20],
        "notes": (
            "This v2.1 relabeler preserves pre-label urgency semantics by carrying forward target-summary cpa/tcpa proxies "
            "while still rewriting triggered rules, explanation, and final maneuvers authoritatively."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "label_native_candidates_responsibility_mix_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"label_native_candidates_responsibility_mix.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows relabeled           : {len(rows)}")
    print(f"Mode                     : {args.mode}")
    print(f"Labeler version          : {LABELER_VERSION}")
    print(f"Backup created           : {False if args.no_backup else True}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {len(schema_errors)}")
    print(f"Internal contract errors : {len(internal_contract_errors)}")
    print(f"Updated raw file         : {raw_path}")
    print(f"Report                   : {report_path}")
    print("=" * 100)

    return 0 if (len(schema_errors) == 0 and len(internal_contract_errors) == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
