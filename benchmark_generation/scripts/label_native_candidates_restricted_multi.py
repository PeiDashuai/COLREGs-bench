#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import jsonschema
except Exception:  # pragma: no cover
    jsonschema = None

FAMILY = "restricted_multi"
LABELER_VERSION = "restricted_multi_labeler_audit_v2_1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit-oriented authoritative relabeler for restricted_multi")
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--family", choices=[FAMILY], default=FAMILY)
    p.add_argument("--mode", choices=["built_in"], default="built_in")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--validate-schema", action="store_true")
    p.add_argument("--report-json", type=str, required=True)
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any], *, enable: bool) -> List[str]:
    if not enable or jsonschema is None or not schema:
        return []
    try:
        jsonschema.validate(instance=row, schema=schema)
        return []
    except Exception as exc:
        return [str(exc)]


def backup_raw_file(raw_path: Path) -> Path:
    backup_path = raw_path.with_name(raw_path.stem + ".before_authoritative_labels.jsonl")
    shutil.copy2(raw_path, backup_path)
    return backup_path


def uniq(seq: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in seq:
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def get_reasoning(row: Dict[str, Any]) -> Dict[str, Any]:
    return row.get("debug", {}).get("reasoning", {}) or {}


def get_visibility_mode(row: Dict[str, Any]) -> str:
    return str(
        row.get("family_latent", {}).get("visibility_mode")
        or row.get("scene_spec", {}).get("visibility")
        or row.get("scene_spec", {}).get("area_context", {}).get("visibility_mode")
        or "in_sight"
    )


def get_topology(row: Dict[str, Any]) -> str:
    return str(
        row.get("family_latent", {}).get("spatial_topology")
        or row.get("scene_spec", {}).get("target_role_graph", {}).get("topology")
        or "frontal_blocking"
    )


def get_suppression_type(row: Dict[str, Any]) -> str:
    return str(row.get("family_latent", {}).get("suppression_mechanism") or "action_conflict")


def infer_target_role(target: Dict[str, Any], idx: int) -> str:
    return str(
        target.get("role_hint")
        or target.get("role")
        or target.get("target_role")
        or target.get("graph_role")
        or f"target_{idx}"
    )


def _float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return float(default)


def heading_to_vec(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return math.cos(rad), math.sin(rad)


def estimate_cpa_tcpa(own: Dict[str, Any], target: Dict[str, Any]) -> Tuple[float, float]:
    ox = _float(own.get("x_m"), 0.0)
    oy = _float(own.get("y_m"), 0.0)
    tx = _float(target.get("x_m"), 0.0)
    ty = _float(target.get("y_m"), 0.0)
    os = _float(own.get("speed_mps"), 0.0)
    ts = _float(target.get("speed_mps"), 0.0)
    oh = _float(own.get("heading_deg"), 0.0)
    th = _float(target.get("heading_deg"), 0.0)
    ovx, ovy = heading_to_vec(oh)
    tvx, tvy = heading_to_vec(th)
    ovx *= os; ovy *= os; tvx *= ts; tvy *= ts
    rx, ry = tx - ox, ty - oy
    rvx, rvy = tvx - ovx, tvy - ovy
    rv2 = rvx * rvx + rvy * rvy
    if rv2 <= 1e-9:
        tcpa = 0.0
        cpa = math.hypot(rx, ry)
        return float(max(cpa, 0.0)), float(max(tcpa, 0.0))
    tcpa = - (rx * rvx + ry * rvy) / rv2
    if tcpa < 0:
        tcpa = 0.0
    cx = rx + rvx * tcpa
    cy = ry + rvy * tcpa
    cpa = math.hypot(cx, cy)
    return float(max(cpa, 0.0)), float(max(tcpa, 0.0))


def derive_triggered_rules(row: Dict[str, Any]) -> List[str]:
    latent = row.get("family_latent", {})
    topo = get_topology(row)
    vis = get_visibility_mode(row)
    supp = get_suppression_type(row)
    targets = row.get("scene_spec", {}).get("targets", []) or []
    rules = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
        "COLREG_R08_POSITIVE_ACTION_WHEN_RISK",
        "COLREG_MULTI_TARGET_GLOBAL_RESOLUTION",
    ]
    if vis == "restricted_visibility":
        rules.extend([
            "COLREG_R19_RESTRICTED_VISIBILITY",
            "COLREG_R19_SAFE_SPEED_RESTRICTED_VIS",
            "COLREG_R19_ENGINES_READY",
        ])
    mix = str(latent.get("target_status_mix") or latent.get("target_status_combo") or "all_power")
    if mix in {"mixed_status", "mixed_multi_status", "mixed"}:
        rules.append("COLREG_R18_RESPONSIBILITIES_BETWEEN_VESSELS")
    topo_rules = {
        "single_side_cluster": ["COLREG_MULTI_TARGET_SIDE_CLUSTER"],
        "bilateral_constraint": ["COLREG_MULTI_TARGET_BILATERAL_CONSTRAINT"],
        "frontal_blocking": ["COLREG_MULTI_TARGET_FRONTAL_BLOCKING"],
        "surrounding_ring": ["COLREG_MULTI_TARGET_SURROUNDING_RING"],
    }
    rules.extend(topo_rules.get(topo, []))
    supp_rules = {
        "action_conflict": ["COLREG_MULTI_TARGET_ACTION_CONFLICT"],
        "hierarchy_override": ["COLREG_MULTI_TARGET_HIERARCHY_OVERRIDE"],
        "area_override": ["COLREG_MULTI_TARGET_AREA_OVERRIDE"],
        "mixed": [
            "COLREG_MULTI_TARGET_ACTION_CONFLICT",
            "COLREG_MULTI_TARGET_HIERARCHY_OVERRIDE",
            "COLREG_MULTI_TARGET_AREA_OVERRIDE",
        ],
    }
    rules.extend(supp_rules.get(supp, []))
    for idx, t in enumerate(targets, 1):
        role = infer_target_role(t, idx)
        sector = str(t.get("relative_bearing_sector") or "unknown")
        if sector == "ahead":
            rules.append("COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK")
        if sector.startswith("starboard"):
            rules.append("COLREG_R19D2_AVOID_TURN_TOWARD_STARBOARD_ABEAM_OR_AFT")
        if sector.startswith("port"):
            rules.append("COLREG_R19D3_AVOID_TURN_TOWARD_PORT_ABEAM_OR_AFT")
        if "block" in role or "ahead" in role:
            rules.append("COLREG_MULTI_TARGET_BLOCKING_TARGET")
        if "constraint" in role:
            rules.append("COLREG_MULTI_TARGET_CONSTRAINT_TARGET")
    if get_reasoning(row).get("target_removal_sensitivity", {}).get("sensitive"):
        rules.append("COLREG_MULTI_TARGET_REMOVAL_SENSITIVE")
    return uniq(rules)


def derive_target_rule_ids(target: Dict[str, Any], topo: str, vis: str, mix: str) -> List[str]:
    sector = str(target.get("relative_bearing_sector") or "unknown")
    role = str(target.get("role_hint") or target.get("target_role") or target.get("role") or "other")
    rules = ["COLREG_R07_RISK_OF_COLLISION_ASSESS"]
    if vis == "restricted_visibility":
        rules.append("COLREG_R19_RESTRICTED_VISIBILITY")
    if sector == "ahead":
        rules.append("COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK")
    elif sector.startswith("starboard"):
        rules.append("COLREG_R19D2_AVOID_TURN_TOWARD_STARBOARD_ABEAM_OR_AFT")
    elif sector.startswith("port"):
        rules.append("COLREG_R19D3_AVOID_TURN_TOWARD_PORT_ABEAM_OR_AFT")
    if "block" in role or "ahead" in role:
        rules.append("COLREG_MULTI_TARGET_BLOCKING_TARGET")
    if "constraint" in role or topo == "bilateral_constraint":
        rules.append("COLREG_MULTI_TARGET_CONSTRAINT_TARGET")
    if "noise" in role:
        rules.append("COLREG_MULTI_TARGET_SECONDARY_TARGET")
    if mix in {"mixed_status", "mixed_multi_status", "mixed"} and str(target.get("vessel_class")) not in {"power_driven", "sailing"}:
        rules.append("COLREG_R18_RESPONSIBILITIES_BETWEEN_VESSELS")
    return uniq(rules)


def derive_target_summaries(row: Dict[str, Any], triggered_rules: List[str]) -> List[Dict[str, Any]]:
    own = row.get("scene_spec", {}).get("ownship", {}) or {}
    targets = row.get("scene_spec", {}).get("targets", []) or []
    topo = get_topology(row)
    vis = get_visibility_mode(row)
    mix = str(row.get("family_latent", {}).get("target_status_mix") or row.get("family_latent", {}).get("target_status_combo") or "all_power")
    out: List[Dict[str, Any]] = []
    for idx, t in enumerate(targets, 1):
        cpa = t.get("cpa_m", t.get("min_cpa_m"))
        tcpa = t.get("tcpa_s", t.get("min_tcpa_s"))
        cpa_f = _float(cpa, -1.0)
        tcpa_f = _float(tcpa, -1.0)
        if cpa_f < 0 or tcpa_f < 0:
            cpa_f, tcpa_f = estimate_cpa_tcpa(own, t)
        ts_rules = [r for r in derive_target_rule_ids(t, topo, vis, mix) if r in set(triggered_rules)]
        out.append({
            "target_id": str(t.get("target_id") or f"t{idx}"),
            "target_role": infer_target_role(t, idx),
            "relative_bearing_sector": str(t.get("relative_bearing_sector") or "unknown"),
            "triggered_rule_ids": ts_rules,
            "risk_of_collision": bool(t.get("risk_of_collision", True)),
            "collision_imminent": bool(t.get("collision_imminent", False) or (cpa_f < 150.0 and tcpa_f < 60.0)),
            "closing": bool(t.get("closing", True)),
            "cpa_m": float(max(cpa_f, 0.0)),
            "tcpa_s": float(max(tcpa_f, 0.0)),
        })
    return out


def derive_suppression_records(row: Dict[str, Any], target_summaries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    reasoning = get_reasoning(row)
    existing = copy.deepcopy(reasoning.get("suppression_records") or row.get("labels", {}).get("suppression_records") or [])
    normalized: List[Dict[str, Any]] = []
    top_suppressed: List[str] = []
    if existing:
        for idx, rec in enumerate(existing, 1):
            rid = str(rec.get("suppressed_rule_id") or rec.get("rule_id") or rec.get("rule") or "COLREG_MULTI_TARGET_ACTION_CONFLICT")
            sid = str(rec.get("source_target_id") or rec.get("suppressed_target_id") or "unknown")
            by = str(rec.get("suppressed_by_target_id") or rec.get("blocking_target_id") or sid)
            normalized.append({
                "record_id": str(rec.get("record_id") or f"supp_{idx:03d}"),
                "suppression_type": str(rec.get("suppression_type") or get_suppression_type(row)),
                "source_target_id": sid,
                "blocking_target_id": by,
                "suppressed_rule_ids": [rid],
                "suppressed_rules": [rid],
                "suppressed_primitive": str(rec.get("suppressed_primitive") or "KEEP_COURSE"),
                "reason": str(rec.get("reason") or rec.get("reason_type") or get_suppression_type(row)),
                "reason_text": str(rec.get("reason_text") or f"{rid} suppressed under global multi-target resolution."),
            })
            top_suppressed.append(rid)
        return normalized, uniq(top_suppressed)

    # synthesize at least one record from topology / targets if none existed
    if target_summaries:
        rid = "COLREG_MULTI_TARGET_ACTION_CONFLICT"
        if get_suppression_type(row) == "hierarchy_override":
            rid = "COLREG_MULTI_TARGET_HIERARCHY_OVERRIDE"
        elif get_suppression_type(row) == "area_override":
            rid = "COLREG_MULTI_TARGET_AREA_OVERRIDE"
        sid = target_summaries[0]["target_id"]
        normalized.append({
            "record_id": "supp_001",
            "suppression_type": get_suppression_type(row),
            "source_target_id": sid,
            "blocking_target_id": sid,
            "suppressed_rule_ids": [rid],
            "suppressed_rules": [rid],
            "suppressed_primitive": "KEEP_COURSE",
            "reason": "global_multi_target_conflict",
            "reason_text": f"{rid} retained as the dominant suppression rule under global multi-target resolution.",
        })
        top_suppressed.append(rid)
    return normalized, uniq(top_suppressed)


def derive_primitive_ledger(triggered_rules: List[str], target_summaries: List[Dict[str, Any]], suppression_records: List[Dict[str, Any]], row: Dict[str, Any]) -> Dict[str, Any]:
    topo = get_topology(row)
    vis = get_visibility_mode(row)
    supp = get_suppression_type(row)
    allowed = [
        {
            "primitive": "MAINTAIN_EFFECTIVE_LOOKOUT",
            "support_rules": [r for r in triggered_rules if r in {"COLREG_R05_LOOKOUT"}],
            "targets": [ts["target_id"] for ts in target_summaries],
        },
        {
            "primitive": "KEEP_SAFE_SPEED",
            "support_rules": [r for r in triggered_rules if r in {"COLREG_R06_SAFE_SPEED", "COLREG_R19_SAFE_SPEED_RESTRICTED_VIS"}],
            "targets": [ts["target_id"] for ts in target_summaries],
        },
        {
            "primitive": "GLOBAL_MULTI_TARGET_RESOLUTION",
            "support_rules": [r for r in triggered_rules if r.startswith("COLREG_MULTI_TARGET_")],
            "targets": [ts["target_id"] for ts in target_summaries],
        },
    ]
    # Ensure every entry has non-empty support_rules by fallback to top rule
    for item in allowed:
        if not item["support_rules"] and triggered_rules:
            item["support_rules"] = [triggered_rules[0]]
    forbidden = []
    if supp in {"action_conflict", "mixed"}:
        forbidden.append({"primitive": "NAIVE_SINGLE_TARGET_ACTION", "support_rules": ["COLREG_MULTI_TARGET_ACTION_CONFLICT"]})
    if supp in {"hierarchy_override", "mixed"}:
        forbidden.append({"primitive": "IGNORE_HIGHER_PRIORITY_CONSTRAINT", "support_rules": ["COLREG_MULTI_TARGET_HIERARCHY_OVERRIDE"]})
    if supp in {"area_override", "mixed"}:
        forbidden.append({"primitive": "IGNORE_CONTEXTUAL_OVERRIDE", "support_rules": ["COLREG_MULTI_TARGET_AREA_OVERRIDE"]})
    if vis == "restricted_visibility":
        allowed.append({"primitive": "NAVIGATE_WITH_RESTRICTED_VISIBILITY_CAUTION", "support_rules": ["COLREG_R19_RESTRICTED_VISIBILITY"]})
    if topo == "surrounding_ring":
        allowed.append({"primitive": "PREFER_MINIMAL_GLOBALLY_SAFE_MANEUVER", "support_rules": ["COLREG_MULTI_TARGET_SURROUNDING_RING"]})
    elif topo == "frontal_blocking":
        allowed.append({"primitive": "RESOLVE_AHEAD_BLOCKING_WITH_SIDE_AWARE_ACTION", "support_rules": ["COLREG_MULTI_TARGET_FRONTAL_BLOCKING"]})
    elif topo == "bilateral_constraint":
        allowed.append({"primitive": "SELECT_BALANCED_CONFLICT_RESOLVING_ACTION", "support_rules": ["COLREG_MULTI_TARGET_BILATERAL_CONSTRAINT"]})
    elif topo == "single_side_cluster":
        allowed.append({"primitive": "BIAS_AWAY_FROM_DOMINANT_CLUSTER_IF_GLOBALLY_RETAINED", "support_rules": ["COLREG_MULTI_TARGET_SIDE_CLUSTER"]})
    return {
        "maneuver": {
            "allowed": allowed,
            "forbidden": forbidden,
        },
        "suppression": {
            "records": suppression_records,
            "support_rules": uniq([rid for rec in suppression_records for rid in rec.get("suppressed_rule_ids", [])]),
        },
        "summary": {
            "support_rules": uniq([r for item in allowed + forbidden for r in item.get("support_rules", [])]),
            "target_ids": [ts["target_id"] for ts in target_summaries],
            "suppression_count": len(suppression_records),
        },
    }


def derive_explanation_steps(row: Dict[str, Any], target_summaries: List[Dict[str, Any]], suppression_records: List[Dict[str, Any]], final_allowed: List[str], final_forbidden: List[str]) -> List[Dict[str, Any]]:
    latent = row.get("family_latent", {})
    topo = get_topology(row)
    supp = get_suppression_type(row)
    vis = get_visibility_mode(row)
    return [
        {
            "stage": "area_context",
            "text": f"restricted_multi scene under visibility={vis}, topology={topo}, effective_targets={latent.get('num_effective_targets') or latent.get('effective_target_count')}.",
            "summary": f"visibility={vis}, topology={topo}",
        },
        {
            "stage": "target_wise_obligations",
            "text": f"{len(target_summaries)} targets jointly contribute local obligations; no single target alone explains the final action set.",
            "summary": f"{len(target_summaries)} targets contribute local obligations.",
        },
        {
            "stage": "conflict_and_suppression",
            "text": f"suppression_mechanism={supp}; {len(suppression_records)} suppression record(s) retained after cross-target conflict filtering.",
            "summary": f"{len(suppression_records)} suppression record(s).",
        },
        {
            "stage": "global_stage",
            "text": f"Global resolution retains allowed maneuvers {final_allowed} and forbids {final_forbidden} based on multi-target aggregation and suppression.",
            "summary": "global multi-target resolution retained final maneuvers.",
        },
        {
            "stage": "operational_recommendation",
            "text": "Follow only the globally retained safe actions and avoid any action suppressed by higher-priority conflict resolution.",
            "summary": "act only on globally retained safe maneuvers.",
        },
    ]


def derive_final_actions(row: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    topo = get_topology(row)
    supp = get_suppression_type(row)
    vis = get_visibility_mode(row)
    allowed = ["MAINTAIN_EFFECTIVE_LOOKOUT", "KEEP_SAFE_SPEED", "REASON_OVER_ALL_RELEVANT_TARGETS"]
    forbidden = ["ACT_ON_SINGLE_TARGET_ONLY"]
    if topo == "single_side_cluster":
        allowed.append("BIAS_AWAY_FROM_DOMINANT_CLUSTER_IF_GLOBALLY_RETAINED")
    elif topo == "bilateral_constraint":
        allowed.append("SELECT_BALANCED_CONFLICT_RESOLVING_ACTION")
    elif topo == "frontal_blocking":
        allowed.append("RESOLVE_AHEAD_BLOCKING_WITH_SIDE_AWARE_ACTION")
    elif topo == "surrounding_ring":
        allowed.append("PREFER_MINIMAL_GLOBALLY_SAFE_MANEUVER")
    if supp in {"action_conflict", "mixed"}:
        forbidden.extend(["KEEP_COURSE_WITHOUT_GLOBAL_CHECK", "NAIVE_SINGLE_TARGET_ACTION"])
    if supp in {"hierarchy_override", "mixed"}:
        forbidden.append("IGNORE_HIGHER_PRIORITY_CONSTRAINT")
    if supp in {"area_override", "mixed"}:
        forbidden.append("IGNORE_CONTEXTUAL_OVERRIDE")
    if vis == "restricted_visibility":
        allowed.append("NAVIGATE_WITH_RESTRICTED_VISIBILITY_CAUTION")
        forbidden.append("ASSUME_VISUAL_CERTAINTY")
    return uniq(allowed), uniq(forbidden)


def relabel_one_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    new_row = copy.deepcopy(row)
    errors: List[str] = []
    triggered_rules = derive_triggered_rules(new_row)
    target_summaries = derive_target_summaries(new_row, triggered_rules)
    suppression_records, suppressed_rule_ids = derive_suppression_records(new_row, target_summaries)
    ledger = derive_primitive_ledger(triggered_rules, target_summaries, suppression_records, new_row)
    maneuver_allowed, maneuver_forbidden = derive_final_actions(new_row)
    explanation_steps = derive_explanation_steps(new_row, target_summaries, suppression_records, maneuver_allowed, maneuver_forbidden)

    labels = {
        "triggered_rules": triggered_rules,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": ledger,
        "suppressed_rules": [{"rule_id": rid} for rid in suppressed_rule_ids],
        "suppression_records": suppression_records,
        "explanation_steps": explanation_steps,
        "maneuver_allowed": maneuver_allowed,
        "maneuver_forbidden": maneuver_forbidden,
        "lights_required": [],
        "lights_forbidden": [],
        "sounds_required": [],
        "sounds_forbidden": [],
    }
    if len(target_summaries) < 3:
        errors.append("target_summaries_too_short")
    if not maneuver_allowed and not maneuver_forbidden:
        errors.append("missing_final_maneuvers")
    if not any(isinstance(s, dict) and s.get("stage") == "global_stage" and str(s.get("text") or "").strip() for s in explanation_steps):
        errors.append("missing_global_stage")
    if any(not isinstance(ts.get("cpa_m"), (int, float)) or not isinstance(ts.get("tcpa_s"), (int, float)) for ts in target_summaries):
        errors.append("invalid_target_summary_numeric_fields")
    new_row["labels"] = labels
    new_row.setdefault("debug", {})
    new_row["debug"]["native_labeling"] = {
        "labeler_version": LABELER_VERSION,
        "mode": "built_in",
        "internal_contract_errors": errors,
    }
    return new_row, errors


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"
    report_path = Path(args.report_json)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    if not args.overwrite:
        raise ValueError("--overwrite is required")
    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if schema_path.exists() else {}
    backup_path = backup_raw_file(raw_path)
    out: List[Dict[str, Any]] = []
    schema_error_count = 0
    internal_contract_error_count = 0
    for row in rows:
        new_row, errs = relabel_one_row(row)
        internal_contract_error_count += len(errs)
        schema_error_count += len(maybe_validate_against_schema(new_row, schema, enable=args.validate_schema))
        out.append(new_row)
    write_jsonl(raw_path, out)
    report = {
        "family": FAMILY,
        "labeler_version": LABELER_VERSION,
        "mode": args.mode,
        "row_count": len(out),
        "schema_validation_enabled": bool(args.validate_schema),
        "jsonschema_available": jsonschema is not None,
        "schema_error_count": schema_error_count,
        "internal_contract_error_count": internal_contract_error_count,
        "backup_file": str(backup_path.relative_to(root)),
        "raw_file_updated": str(raw_path.relative_to(root)),
    }
    write_json(report_path, report)
    print("="*100)
    print(f"label_native_candidates_restricted_multi.py | root={root} | family={FAMILY}")
    print("="*100)
    print(f"Rows relabeled           : {len(out)}")
    print(f"Mode                     : {args.mode}")
    print(f"Backup created           : True")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {jsonschema is not None}")
    print(f"Schema error count       : {schema_error_count}")
    print(f"Internal contract errors : {internal_contract_error_count}")
    print(f"Updated raw file         : {raw_path.relative_to(root)}")
    print(f"Report written           : {report_path}")


if __name__ == "__main__":
    main()
