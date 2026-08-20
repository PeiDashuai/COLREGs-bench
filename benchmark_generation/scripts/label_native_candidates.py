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

FAMILY = "tss_crossing"
LABELER_VERSION = "tss_crossing_labeler_v3"

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


def deg_to_unit_vec(heading_deg: float) -> Tuple[float, float]:
    r = math.radians(heading_deg)
    return math.cos(r), math.sin(r)


def heading_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


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


# ============================================================
# TSS labeler v3: preserve legal pre-label rule-variants
# ============================================================

SAFE_SUPPORT_RULES = {
    "COLREG_R5_LOOKOUT",
    "COLREG_R6_SAFE_SPEED",
    "COLREG_R7_RISK_OF_COLLISION",
    "COLREG_R8_ACTION_TO_AVOID_COLLISION",
}


def choose_target_indices(targets: List[Dict[str, Any]], ownship: Dict[str, Any]) -> Dict[str, Optional[int]]:
    own_x = safe_float(ownship.get("x_m"), 0.0)
    own_y = safe_float(ownship.get("y_m"), 0.0)
    own_h = safe_float(ownship.get("heading_deg"), 0.0)

    starboard_idx = None
    ahead_idx = None
    other_idx = None

    for i, t in enumerate(targets):
        sec = rel_bearing_sector(
            own_x=own_x, own_y=own_y, own_heading_deg=own_h,
            tgt_x=safe_float(t.get("x_m"), 0.0),
            tgt_y=safe_float(t.get("y_m"), 0.0),
        )
        if sec == "starboard" and starboard_idx is None:
            starboard_idx = i
        elif sec == "ahead" and ahead_idx is None:
            ahead_idx = i
        elif other_idx is None:
            other_idx = i

    if starboard_idx is None and len(targets) > 0:
        starboard_idx = 0
    if ahead_idx is None and len(targets) > 1:
        ahead_idx = 1 if starboard_idx != 1 else 0
    if other_idx is None and len(targets) > 0:
        other_idx = 0
    return {"starboard_idx": starboard_idx, "ahead_idx": ahead_idx, "other_idx": other_idx}


def canonical_base_rules(overlap_mode: str) -> List[str]:
    base = ["COLREG_R10_TSS_GENERAL", "COLREG_R10_CROSSING"]
    if overlap_mode == "pure_tss":
        return base
    if overlap_mode == "tss_plus_rule15":
        return base + ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD", "COLREG_R16_GIVEWAY_MANEUVER"]
    if overlap_mode == "tss_plus_rule16_17":
        return base + ["COLREG_R16_GIVEWAY_MANEUVER", "COLREG_R17_STANDON"]
    return base + ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD", "COLREG_R16_GIVEWAY_MANEUVER", "COLREG_R17_STANDON"]


def normalize_rule_order(rules: List[str]) -> List[str]:
    # stable order: safe support -> R10 -> others alpha within bucket
    rules = unique_keep_order([str(x) for x in rules])
    support = [r for r in rules if r in SAFE_SUPPORT_RULES]
    r10 = sorted([r for r in rules if r.startswith("COLREG_R10")])
    others = sorted([r for r in rules if (r not in SAFE_SUPPORT_RULES and not r.startswith("COLREG_R10"))])
    return unique_keep_order(support + r10 + others)


def preserve_rule_signature_variant(
    *,
    old_rules: List[str],
    overlap_mode: str,
    observed_angle_band: Optional[str],
    observed_lane_occ: Optional[str],
) -> List[str]:
    old_set = set(str(x) for x in old_rules)
    new_rules: List[str] = []

    # 1) keep any safe-support rules already present
    for r in SAFE_SUPPORT_RULES:
        if r in old_set:
            new_rules.append(r)

    # 2) preserve as many legal R10 subrules as possible
    legal_r10 = [r for r in old_set if r.startswith("COLREG_R10")]
    if not legal_r10:
        legal_r10 = ["COLREG_R10_TSS_GENERAL", "COLREG_R10_CROSSING"]

    # always ensure core pair present
    if "COLREG_R10_TSS_GENERAL" not in legal_r10:
        legal_r10.append("COLREG_R10_TSS_GENERAL")
    if "COLREG_R10_CROSSING" not in legal_r10:
        legal_r10.append("COLREG_R10_CROSSING")

    # angle/lane-aware augmentation without changing validator contract
    if observed_angle_band == "near_right_angle":
        legal_r10.append("COLREG_R10_RIGHT_ANGLE_CROSSING")
    elif observed_angle_band == "moderately_off":
        legal_r10.append("COLREG_R10_MODERATE_ANGLE_MONITORING")
    else:
        legal_r10.append("COLREG_R10_HEADING_CORRECTION_TO_CROSS")

    if observed_lane_occ == "boundary_touching":
        legal_r10.append("COLREG_R10_BOUNDARY_AWARENESS")
    elif observed_lane_occ == "single_lane":
        legal_r10.append("COLREG_R10_SINGLE_LANE_MONITORING")
    elif observed_lane_occ == "multi_lane":
        legal_r10.append("COLREG_R10_MULTI_LANE_MONITORING")

    new_rules.extend(sorted(set(legal_r10)))

    # 3) preserve legal non-TSS variants per overlap bucket
    if overlap_mode == "pure_tss":
        # keep no non-TSS overlap families
        pass
    elif overlap_mode == "tss_plus_rule15":
        if any(r.startswith("COLREG_R15") for r in old_set):
            new_rules.extend(sorted(r for r in old_set if r.startswith("COLREG_R15")))
        else:
            new_rules.append("COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD")
        # R16 is legal here and helps create distinct signatures
        if any(r.startswith("COLREG_R16") for r in old_set):
            new_rules.extend(sorted(r for r in old_set if r.startswith("COLREG_R16")))
        else:
            new_rules.append("COLREG_R16_GIVEWAY_MANEUVER")
    elif overlap_mode == "tss_plus_rule16_17":
        has16 = any(r.startswith("COLREG_R16") for r in old_set)
        has17 = any(r.startswith("COLREG_R17") for r in old_set)
        if has16:
            new_rules.extend(sorted(r for r in old_set if r.startswith("COLREG_R16")))
        if has17:
            new_rules.extend(sorted(r for r in old_set if r.startswith("COLREG_R17")))
        if not has16 and not has17:
            new_rules.extend(["COLREG_R16_GIVEWAY_MANEUVER", "COLREG_R17_STANDON"])
    else:
        # preserve all legal overlap families that still infer multi_overlap
        keep_prefixes = ("COLREG_R15", "COLREG_R16", "COLREG_R17", "COLREG_R13", "COLREG_R18", "COLREG_R9")
        kept = [r for r in old_set if r.startswith(keep_prefixes)]
        if not kept:
            kept = [
                "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD",
                "COLREG_R16_GIVEWAY_MANEUVER",
                "COLREG_R17_STANDON",
            ]
        new_rules.extend(sorted(kept))

    # 4) final safety net: contract must match latent overlap mode
    final_rules = normalize_rule_order(new_rules)
    inferred = infer_rule_overlap_mode(final_rules)
    if inferred != overlap_mode:
        # fall back to canonical base but keep safe support / R10 richness if possible
        fallback = normalize_rule_order(list(old_set & SAFE_SUPPORT_RULES) + canonical_base_rules(overlap_mode) + [
            r for r in final_rules if r.startswith("COLREG_R10")
        ])
        if infer_rule_overlap_mode(fallback) == overlap_mode:
            final_rules = fallback
        else:
            final_rules = normalize_rule_order(canonical_base_rules(overlap_mode))
    return final_rules


def build_target_level_rules_preserving_variants(
    *,
    overlap_mode: str,
    targets: List[Dict[str, Any]],
    ownship: Dict[str, Any],
    old_target_summaries: List[Dict[str, Any]],
    aggregate_rules: List[str],
) -> List[List[str]]:
    # map old target summaries by target_id if present, else by position index
    old_by_tid: Dict[str, List[str]] = {}
    for ts in old_target_summaries:
        tid = ts.get("target_id")
        if tid is not None:
            old_by_tid[str(tid)] = [str(x) for x in ts.get("triggered_rule_ids", [])]

    chosen = choose_target_indices(targets, ownship)
    per_target: List[List[str]] = []
    for i, t in enumerate(targets):
        tid = str(t.get("target_id") or t.get("id") or f"t{i+1}")
        old_rules = list(old_by_tid.get(tid, []))
        sec = rel_bearing_sector(
            safe_float(ownship.get("x_m"), 0.0),
            safe_float(ownship.get("y_m"), 0.0),
            safe_float(ownship.get("heading_deg"), 0.0),
            safe_float(t.get("x_m"), 0.0),
            safe_float(t.get("y_m"), 0.0),
        )
        base = [r for r in aggregate_rules if r.startswith("COLREG_R10")]
        if any(r in SAFE_SUPPORT_RULES for r in aggregate_rules):
            base += [r for r in aggregate_rules if r in SAFE_SUPPORT_RULES]

        if overlap_mode == "tss_plus_rule15" and i == chosen.get("starboard_idx"):
            base += [r for r in aggregate_rules if r.startswith("COLREG_R15") or r.startswith("COLREG_R16")]
        elif overlap_mode == "tss_plus_rule16_17" and i == chosen.get("ahead_idx"):
            base += [r for r in aggregate_rules if r.startswith("COLREG_R16") or r.startswith("COLREG_R17")]
        elif overlap_mode == "multi_overlap":
            if i == chosen.get("starboard_idx"):
                base += [r for r in aggregate_rules if r.startswith("COLREG_R15") or r.startswith("COLREG_R16")]
            elif i == chosen.get("ahead_idx"):
                base += [r for r in aggregate_rules if r.startswith("COLREG_R17") or r.startswith("COLREG_R13") or r.startswith("COLREG_R18") or r.startswith("COLREG_R9")]
            else:
                # preserve any lawful old minor overlap on tertiary target
                base += [r for r in old_rules if infer_rule_overlap_mode(base + [r]) == "multi_overlap"]

        # keep some per-target old distinctiveness if legal
        merged = normalize_rule_order(old_rules + base)
        # per-target can be less strict than aggregate, but avoid illegal families for target summaries
        if overlap_mode == "pure_tss":
            merged = [r for r in merged if (r.startswith("COLREG_R10") or r in SAFE_SUPPORT_RULES)]
        per_target.append(unique_keep_order(merged))
    return per_target


def authoritative_tss_labels_v3(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    family_latent = row.get("family_latent", {})
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec", {}), dict) else {}
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec, dict) else {}
    ownship = scene_spec.get("ownship", {}) if isinstance(scene_spec, dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec, dict) else []
    old_labels = row.get("labels", {}) if isinstance(row.get("labels", {}), dict) else {}

    overlap_mode = str(family_latent.get("rule_overlap_mode"))
    lane_heading = safe_float(area.get("lane_heading_deg"), 0.0)
    own_heading = safe_float(ownship.get("heading_deg"), 0.0)
    observed_angle_band = infer_crossing_angle_band(own_heading, lane_heading)
    observed_lane_occ = str(family_latent.get("lane_occupancy_band") or "single_lane")

    old_triggered_rules = [str(x) for x in old_labels.get("triggered_rules", [])] if isinstance(old_labels, dict) else []
    aggregate_rules = preserve_rule_signature_variant(
        old_rules=old_triggered_rules,
        overlap_mode=overlap_mode,
        observed_angle_band=observed_angle_band,
        observed_lane_occ=observed_lane_occ,
    )

    old_target_summaries = old_labels.get("target_summaries", []) if isinstance(old_labels, dict) else []
    rules_per_target = build_target_level_rules_preserving_variants(
        overlap_mode=overlap_mode,
        targets=targets,
        ownship=ownship,
        old_target_summaries=old_target_summaries if isinstance(old_target_summaries, list) else [],
        aggregate_rules=aggregate_rules,
    )

    own_x = safe_float(ownship.get("x_m"), 0.0)
    own_y = safe_float(ownship.get("y_m"), 0.0)
    own_h = safe_float(ownship.get("heading_deg"), 0.0)
    own_s = safe_float(ownship.get("speed_mps"), 0.0)

    target_summaries: List[Dict[str, Any]] = []
    for i, t in enumerate(targets):
        tx = safe_float(t.get("x_m"), 0.0)
        ty = safe_float(t.get("y_m"), 0.0)
        th = safe_float(t.get("heading_deg"), 0.0)
        ts = safe_float(t.get("speed_mps"), 0.0)
        cpa_m, tcpa_s = cpa_tcpa(own_x, own_y, own_h, own_s, tx, ty, th, ts)
        sec = rel_bearing_sector(own_x, own_y, own_h, tx, ty)

        existing = old_target_summaries[i] if i < len(old_target_summaries) and isinstance(old_target_summaries[i], dict) else {}
        summary = {
            "target_id": t.get("target_id") or t.get("id") or f"t{i+1}",
            "relative_bearing_sector": sec,
            "cpa_m": round(cpa_m, 2),
            "tcpa_s": round(tcpa_s, 2),
            "risk_of_collision": bool(cpa_m <= 400.0 and tcpa_s <= 900.0),
            "collision_imminent": bool(cpa_m <= 150.0 and tcpa_s <= 240.0),
            "triggered_rule_ids": rules_per_target[i],
        }
        # preserve any extra lawful metadata already present
        for k in ["closing", "encounter_note", "tss_role", "lane_relation", "support_variant"]:
            if k in existing:
                summary[k] = existing[k]
        target_summaries.append(summary)

    allowed = [
        "KEEP_LOOKOUT",
        "PROCEED_SAFE_SPEED",
        "MAINTAIN_CLEAR_TSS_CROSSING_INTENT",
    ]
    forbidden: List[str] = []

    if observed_angle_band == "near_right_angle":
        allowed.append("CROSS_AT_NEAR_RIGHT_ANGLE")
        forbidden.append("CROSS_AT_SMALL_ANGLE")
    elif observed_angle_band == "moderately_off":
        allowed.append("ADJUST_HEADING_TOWARD_SAFE_CROSSING_ANGLE")
        forbidden.append("CROSS_AT_SMALL_ANGLE")
    else:
        allowed.append("CORRECT_CROSSING_ANGLE_SUBSTANTIALLY")
        forbidden.extend(["CROSS_AT_SMALL_ANGLE", "MAINTAIN_STRONGLY_OFF_CROSSING"])

    if overlap_mode == "tss_plus_rule15":
        allowed += ["GIVE_WAY_IF_REQUIRED", "TAKE_EARLY_SUBSTANTIAL_ACTION"]
        forbidden += ["TURN_PORT_IF_CONFLICTING"]
    elif overlap_mode == "tss_plus_rule16_17":
        allowed += ["TAKE_EARLY_SUBSTANTIAL_ACTION", "MAINTAIN_STANDON_IF_SAFE"]
    elif overlap_mode == "multi_overlap":
        allowed += ["GIVE_WAY_IF_REQUIRED", "TAKE_EARLY_SUBSTANTIAL_ACTION"]
        forbidden += ["TURN_PORT_IF_CONFLICTING"]

    explanation_steps = [
        {
            "stage": "area",
            "summary": "TSS context is active and crossing must respect Rule 10 structure.",
            "text": "TSS context is active and crossing must respect Rule 10 structure.",
        },
        {
            "stage": "geometry",
            "summary": f"Observed crossing angle band is {observed_angle_band}; lane occupancy is {observed_lane_occ}.",
            "text": f"Observed crossing angle band is {observed_angle_band}; lane occupancy is {observed_lane_occ}.",
        },
        {
            "stage": "overlap",
            "summary": f"Latent overlap mode is {overlap_mode}, while authoritative relabel preserves lawful sub-rule variation from the native generator.",
            "text": f"Latent overlap mode is {overlap_mode}, while authoritative relabel preserves lawful sub-rule variation from the native generator.",
        },
        {
            "stage": "action",
            "summary": "Final maneuver set is derived from TSS crossing obligations plus preserved overlap-specific rule detail.",
            "text": "Final maneuver set is derived from TSS crossing obligations plus preserved overlap-specific rule detail.",
        },
    ]

    labels = {
        "triggered_rules": aggregate_rules,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": {
            "maneuver": {
                "CROSSING_CONTROL": {
                    "polarities": ["allow"],
                    "support_rules": aggregate_rules,
                },
                "TSS_RULE_VARIANT": {
                    "polarities": ["allow"],
                    "support_rules": [r for r in aggregate_rules if r.startswith("COLREG_R10")],
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

    inferred_overlap_after_write = infer_rule_overlap_mode(labels["triggered_rules"])
    self_check_errors: List[str] = []
    if inferred_overlap_after_write != overlap_mode:
        self_check_errors.append("internal_overlap_contract_mismatch")

    aux = {
        "label_source": LABELER_VERSION,
        "observed_crossing_angle_band": observed_angle_band,
        "observed_lane_occupancy_band": observed_lane_occ,
        "observed_rule_overlap_mode": inferred_overlap_after_write,
        "preserved_rule_signature_from_generator": True,
    }
    return labels, aux, self_check_errors


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Label native candidates (revised version: tss_crossing only).")
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="tss_crossing", choices=["tss_crossing"], help="Only tss_crossing is supported")
    ap.add_argument("--mode", type=str, default="built_in", choices=["built_in", "adapter"])
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-backup", action="store_true")
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
        print("ERROR: this version updates labels in place; pass --overwrite to proceed.", file=sys.stderr)
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
            labels, aux, self_check_errors = authoritative_tss_labels_v3(new_row)
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
                "latent_rule_overlap_mode": new_row.get("family_latent", {}).get("rule_overlap_mode"),
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
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_validation_enabled": bool(args.validate_schema),
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:20],
        "internal_contract_error_count": len(internal_contract_errors),
        "internal_contract_errors_head": internal_contract_errors[:20],
        "notes": (
            "Built-in v3 preserves generator-side lawful rule-signature variation while enforcing "
            "the latent overlap-mode contract for tss_crossing."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (root / "reports" / "release_reports" / "label_native_candidates_report.json")
    write_json(report_path, report)

    print("=" * 100)
    print(f"label_native_candidates.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows relabeled           : {len(rows)}")
    print(f"Mode                     : {args.mode}")
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
