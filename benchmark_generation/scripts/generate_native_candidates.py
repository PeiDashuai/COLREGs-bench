
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception as e:
    raise SystemExit(
        "PyYAML is required. Install it with: pip install pyyaml\n"
        f"Import error: {e}"
    )

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


FAMILY = "tss_crossing"
GENERATOR_REVISION = "tss_crossing_generator_p1_v2"


# ============================================================
# IO
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


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

def weighted_choice(rng: random.Random, dist: Dict[str, float]) -> str:
    items = list(dist.items())
    if not items:
        raise ValueError("Empty distribution.")
    total = sum(max(0.0, float(w)) for _, w in items)
    if total <= 0:
        return items[0][0]
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += max(0.0, float(w))
        if r <= acc:
            return k
    return items[-1][0]


def choose_int_from_range(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(int(lo), int(hi))


def choose_float_from_range(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(float(lo), float(hi))


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def wrap_heading_deg(x: float) -> float:
    y = wrap_deg(float(x))
    if abs(y - 360.0) < 1e-9:
        y = 0.0
    return round(y, 2)


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


def stable_sample_id(i: int) -> str:
    return f"tss_crossing_v3_{i:06d}"


def maybe_validate_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


# ============================================================
# Native tss generator (P1 enhanced structure diversity)
# ============================================================

def sample_crossing_angle_deg(rng: random.Random, band: str) -> float:
    if band == "near_right_angle":
        return choose_float_from_range(rng, 78.0, 102.0)
    if band == "moderately_off":
        if rng.random() < 0.5:
            return choose_float_from_range(rng, 60.0, 74.0)
        return choose_float_from_range(rng, 106.0, 120.0)
    if band == "strongly_off":
        if rng.random() < 0.5:
            return choose_float_from_range(rng, 28.0, 59.0)
        return choose_float_from_range(rng, 121.0, 152.0)
    raise ValueError(f"Unknown crossing_angle_band: {band}")


def sample_effective_target_count(lane_occ: str) -> int:
    if lane_occ == "low":
        return 1
    if lane_occ == "medium":
        return 2
    if lane_occ == "high":
        return 3
    return 2


def sample_lane_flow_complexity(lane_occ: str) -> str:
    if lane_occ == "low":
        return "simple"
    if lane_occ == "medium":
        return "moderate"
    return "dense"


def choose_structure_variant(rng: random.Random, lane_occ: str, overlap: str, entry_mode: str) -> str:
    if lane_occ == "low":
        opts = {
            "solo_midlane_screen": 0.35,
            "solo_boundary_guard": 0.35 if entry_mode in {"edge_entry", "near_boundary_crossing"} else 0.15,
            "solo_far_lane_screen": 0.25,
            "solo_overlap_probe": 0.25 if overlap != "pure_tss" else 0.0,
        }
    elif lane_occ == "medium":
        opts = {
            "paired_adjacent": 0.28,
            "staggered_two_lane": 0.28,
            "boundary_plus_mid": 0.22,
            "screen_plus_guard": 0.22,
        }
    else:
        opts = {
            "three_lane_screen": 0.30,
            "two_plus_guard": 0.30,
            "boundary_stack": 0.20,
            "staggered_dense": 0.20,
        }
    return weighted_choice(rng, {k: v for k, v in opts.items() if v > 0})


def compatible_tss_latent(latent: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    overlap = str(latent.get("rule_overlap_mode"))
    occ = str(latent.get("lane_occupancy_band"))
    eff = int(latent.get("effective_target_count", 0))
    complexity = str(latent.get("lane_flow_complexity"))

    if overlap == "multi_overlap" and eff < 2:
        return False, "multi_overlap_requires_at_least_2_targets"
    if overlap == "multi_overlap" and occ == "low":
        return False, "multi_overlap_not_allowed_with_low_occupancy"
    if occ == "low" and eff != 1:
        return False, "low_occupancy_requires_single_target"
    if occ == "low" and complexity != "simple":
        return False, "low_occupancy_requires_simple_flow"
    if occ == "medium" and eff != 2:
        return False, "medium_occupancy_requires_two_targets"
    if occ == "high" and eff != 3:
        return False, "high_occupancy_requires_three_targets"
    return True, None


def build_area_context(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    tss_geo = spec.get("scene_builder", {}).get("tss_geometry", {})
    lane_count_rng = tss_geo.get("lane_count_range", [2, 4])
    lane_width_rng = tss_geo.get("lane_width_m_range", [150.0, 500.0])
    heading_rng = tss_geo.get("lane_heading_deg_range", [0.0, 359.0])

    lane_count = choose_int_from_range(rng, int(lane_count_rng[0]), int(lane_count_rng[1]))
    lane_width = round(choose_float_from_range(rng, float(lane_width_rng[0]), float(lane_width_rng[1])), 2)
    lane_heading = round(choose_float_from_range(rng, float(heading_rng[0]), float(heading_rng[1])), 2)

    sep_zone = {
        "type": "rectangle",
        "center": [0.0, 0.0],
        "heading_deg": wrap_heading_deg(lane_heading),
        "half_length_m": 1200.0,
        "half_width_m": max(10.0, lane_width * 0.2),
    }

    return {
        "area_type": "tss",
        "channel_context": False,
        "tss_context": True,
        "lane_count": lane_count,
        "lane_width_m": lane_width,
        "tss_lane_heading_deg": wrap_heading_deg(lane_heading),
        "separation_zone_geometry": sep_zone,
        "lane_half_total_width_m": lane_count * lane_width / 2.0,
    }


def build_ownship(rng: random.Random, area_context: Dict[str, Any], family_latent: Dict[str, Any]) -> Dict[str, Any]:
    lane_heading = float(area_context["tss_lane_heading_deg"])
    crossing_angle = sample_crossing_angle_deg(rng, str(family_latent["crossing_angle_band"]))
    sign = 1.0 if rng.random() < 0.5 else -1.0
    own_heading = wrap_deg(lane_heading + sign * crossing_angle)

    entry_mode = str(family_latent["entry_mode"])
    half_w = float(area_context["lane_half_total_width_m"])
    lane_width = float(area_context["lane_width_m"])
    if entry_mode == "direct_crossing":
        x, y = choose_float_from_range(rng, -980.0, -760.0), choose_float_from_range(rng, -0.25 * half_w, 0.25 * half_w)
    elif entry_mode == "edge_entry":
        x, y = choose_float_from_range(rng, -760.0, -620.0), sign * choose_float_from_range(rng, 0.72 * half_w, 0.92 * half_w)
    elif entry_mode == "lane_cut_in":
        x, y = choose_float_from_range(rng, -560.0, -380.0), sign * choose_float_from_range(rng, 0.18 * lane_width, 0.48 * lane_width)
    elif entry_mode == "near_boundary_crossing":
        x, y = choose_float_from_range(rng, -960.0, -740.0), sign * choose_float_from_range(rng, 0.82 * half_w, 0.98 * half_w)
    else:
        x, y = -800.0, 0.0

    return {
        "id": "ownship",
        "vessel_class": "power_driven",
        "nav_status": "underway_making_way",
        "x_m": round(x, 2),
        "y_m": round(y, 2),
        "heading_deg": wrap_heading_deg(own_heading),
        "speed_mps": round(choose_float_from_range(rng, 4.5, 7.5), 2),
    }


def lane_centers(area_context: Dict[str, Any]) -> List[float]:
    lane_count = int(area_context["lane_count"])
    lane_width = float(area_context["lane_width_m"])
    half_w = float(area_context["lane_half_total_width_m"])
    return [(-half_w + lane_width * (i + 0.5)) for i in range(lane_count)]


def build_targets(rng: random.Random, area_context: Dict[str, Any], family_latent: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str]:
    count = int(family_latent["effective_target_count"])
    lane_heading = float(area_context["tss_lane_heading_deg"])
    lane_count = int(area_context["lane_count"])
    lane_width = float(area_context["lane_width_m"])
    centers = lane_centers(area_context)
    occ = str(family_latent["lane_occupancy_band"])
    overlap = str(family_latent["rule_overlap_mode"])
    entry_mode = str(family_latent["entry_mode"])

    variant = choose_structure_variant(rng, occ, overlap, entry_mode)
    base_x = choose_float_from_range(rng, -180.0, 380.0)
    targets: List[Dict[str, Any]] = []

    def pick_y(idx: int, near_boundary: bool = False) -> float:
        if near_boundary:
            sign = 1.0 if idx % 2 == 0 else -1.0
            return sign * choose_float_from_range(rng, 0.78 * float(area_context["lane_half_total_width_m"]), 0.98 * float(area_context["lane_half_total_width_m"]))
        lane_idx = max(0, min(idx, lane_count - 1))
        c = centers[lane_idx]
        return c + choose_float_from_range(rng, -0.18 * lane_width, 0.18 * lane_width)

    role_templates: List[Tuple[str, int, float, bool]] = []
    if count == 1:
        if variant == "solo_boundary_guard":
            role_templates = [("boundary_guard_target", lane_count - 1, choose_float_from_range(rng, 120.0, 260.0), True)]
        elif variant == "solo_far_lane_screen":
            role_templates = [("far_lane_screen_target", max(0, lane_count - 1), choose_float_from_range(rng, 260.0, 430.0), False)]
        elif variant == "solo_overlap_probe":
            role_templates = [("overlap_probe_target", 0 if rng.random() < 0.5 else lane_count - 1, choose_float_from_range(rng, 80.0, 200.0), False)]
        else:
            role_templates = [("midlane_screen_target", lane_count // 2, choose_float_from_range(rng, 120.0, 280.0), False)]
    elif count == 2:
        if variant == "paired_adjacent":
            first = max(0, min(lane_count - 2, choose_int_from_range(rng, 0, max(0, lane_count - 2))))
            role_templates = [
                ("paired_screen_lead", first, choose_float_from_range(rng, 80.0, 180.0), False),
                ("paired_screen_trail", first + 1, choose_float_from_range(rng, 220.0, 360.0), False),
            ]
        elif variant == "boundary_plus_mid":
            role_templates = [
                ("boundary_pressure_target", lane_count - 1, choose_float_from_range(rng, 120.0, 260.0), True),
                ("midlane_guard_target", lane_count // 2, choose_float_from_range(rng, 260.0, 420.0), False),
            ]
        elif variant == "screen_plus_guard":
            role_templates = [
                ("overlap_target", 0, choose_float_from_range(rng, 70.0, 170.0), False),
                ("guard_flow_target", lane_count - 1, choose_float_from_range(rng, 220.0, 360.0), False),
            ]
        else:  # staggered_two_lane
            left = 0
            right = lane_count - 1
            role_templates = [
                ("cross_lane_screen", left, choose_float_from_range(rng, 120.0, 220.0), False),
                ("staggered_guard", right, choose_float_from_range(rng, 300.0, 460.0), False),
            ]
    else:  # count == 3
        if variant == "boundary_stack":
            role_templates = [
                ("boundary_pressure_target", lane_count - 1, choose_float_from_range(rng, 90.0, 190.0), True),
                ("boundary_follow_target", lane_count - 1, choose_float_from_range(rng, 220.0, 340.0), True),
                ("midlane_guard_target", lane_count // 2, choose_float_from_range(rng, 360.0, 520.0), False),
            ]
        elif variant == "two_plus_guard":
            first = 0
            second = min(1, lane_count - 1)
            role_templates = [
                ("paired_overlap_lead", first, choose_float_from_range(rng, 70.0, 170.0), False),
                ("paired_overlap_support", second, choose_float_from_range(rng, 190.0, 310.0), False),
                ("far_guard_target", lane_count - 1, choose_float_from_range(rng, 340.0, 520.0), False),
            ]
        elif variant == "staggered_dense":
            role_templates = [
                ("dense_front_target", 0, choose_float_from_range(rng, 60.0, 150.0), False),
                ("dense_mid_target", lane_count // 2, choose_float_from_range(rng, 180.0, 290.0), False),
                ("dense_rear_target", lane_count - 1, choose_float_from_range(rng, 310.0, 460.0), False),
            ]
        else:  # three_lane_screen
            idxs = [0, lane_count // 2, lane_count - 1]
            role_templates = [
                ("three_lane_lead", idxs[0], choose_float_from_range(rng, 70.0, 170.0), False),
                ("three_lane_mid", idxs[1], choose_float_from_range(rng, 200.0, 320.0), False),
                ("three_lane_guard", idxs[2], choose_float_from_range(rng, 340.0, 520.0), False),
            ]

    for i, (role, lane_idx, xoff, near_boundary) in enumerate(role_templates):
        y = pick_y(lane_idx, near_boundary=near_boundary)
        target = {
            "id": f"t{i+1}",
            "target_id": f"t{i+1}",
            "vessel_class": "power_driven",
            "nav_status": "underway_making_way",
            "x_m": round(base_x + xoff + choose_float_from_range(rng, -25.0, 25.0), 2),
            "y_m": round(y, 2),
            "heading_deg": wrap_heading_deg(lane_heading + choose_float_from_range(rng, -2.0, 2.0)),
            "speed_mps": round(choose_float_from_range(rng, 4.0, 8.0), 2),
            "intended_role": role,
            "role_hint": role,
            "is_effective_target": True,
        }
        targets.append(target)

    return targets, variant


def derive_triggered_rules(family_latent: Dict[str, Any], targets: List[Dict[str, Any]], variant: str) -> List[str]:
    overlap = str(family_latent["rule_overlap_mode"])
    rules = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
        "COLREG_R08_POSITIVE_ACTION_WHEN_RISK",
        "COLREG_R10_TSS_GENERAL",
        "COLREG_R10_CROSSING",
    ]
    if overlap == "pure_tss":
        return rules
    if overlap == "tss_plus_rule15":
        rules += ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD"]
        if variant in {"screen_plus_guard", "solo_overlap_probe", "paired_adjacent"}:
            rules += ["COLREG_R16_GIVEWAY_MANEUVER"]
        return rules
    if overlap == "tss_plus_rule16_17":
        # diversify inside same overlap family without introducing R15
        if variant in {"boundary_plus_mid", "three_lane_screen", "two_plus_guard"}:
            return rules + ["COLREG_R16_GIVEWAY_MANEUVER", "COLREG_R17_STANDON"]
        if len(targets) >= 2 and targets[0].get("intended_role") in {"cross_lane_screen", "paired_screen_lead"}:
            return rules + ["COLREG_R17_STANDON"]
        return rules + ["COLREG_R16_GIVEWAY_MANEUVER"]
    # multi_overlap
    extra = ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD", "COLREG_R16_GIVEWAY_MANEUVER", "COLREG_R17_STANDON"]
    # allow diversity while staying safely in multi_overlap bucket
    if variant in {"boundary_stack", "screen_plus_guard"}:
        extra += ["COLREG_R9_NARROW_CHANNEL_GENERAL"]
    elif variant in {"staggered_dense", "three_lane_screen"}:
        extra += ["COLREG_R13_OVERTAKING"]
    else:
        extra += ["COLREG_R18_RESPONSIBILITIES_BETWEEN_VESSELS"]
    return rules + extra


def derive_actions(family_latent: Dict[str, Any], variant: str) -> Tuple[List[str], List[str]]:
    angle_band = str(family_latent["crossing_angle_band"])
    overlap = str(family_latent["rule_overlap_mode"])

    allowed = [
        "KEEP_LOOKOUT",
        "PROCEED_SAFE_SPEED",
        "MAINTAIN_CLEAR_TSS_CROSSING_INTENT",
    ]
    forbidden: List[str] = []

    if angle_band == "near_right_angle":
        allowed.append("CROSS_AT_NEAR_RIGHT_ANGLE")
        forbidden.append("CROSS_AT_SMALL_ANGLE")
    elif angle_band == "moderately_off":
        allowed.append("ADJUST_HEADING_TOWARD_SAFE_CROSSING_ANGLE")
        forbidden.append("CROSS_AT_SMALL_ANGLE")
    else:
        allowed.append("CORRECT_CROSSING_ANGLE_SUBSTANTIALLY")
        forbidden.extend(["CROSS_AT_SMALL_ANGLE", "MAINTAIN_STRONGLY_OFF_CROSSING"])

    if overlap in {"tss_plus_rule15", "multi_overlap"}:
        allowed.append("GIVE_WAY_IF_REQUIRED")
        forbidden.append("TURN_PORT_IF_CONFLICTING")
    if overlap in {"tss_plus_rule16_17", "multi_overlap"}:
        allowed.append("TAKE_EARLY_SUBSTANTIAL_ACTION")

    if variant in {"boundary_stack", "boundary_plus_mid", "solo_boundary_guard"}:
        allowed.append("AVOID_ENTERING_SEPARATION_ZONE")
    if variant in {"screen_plus_guard", "two_plus_guard", "paired_adjacent"}:
        allowed.append("SEQUENCE_CROSSING_WITH_FLOW_GAPS")
    return sorted(set(allowed)), sorted(set(forbidden))


def cpa_tcpa_by_role(role: str, idx: int) -> Tuple[float, float]:
    if "lead" in role or "overlap" in role or "screen" in role:
        return 120.0 + idx * 35.0, 26.0 + idx * 10.0
    if "boundary" in role:
        return 180.0 + idx * 45.0, 40.0 + idx * 15.0
    if "guard" in role or "rear" in role:
        return 240.0 + idx * 55.0, 52.0 + idx * 18.0
    return 160.0 + idx * 45.0, 34.0 + idx * 14.0


def derive_target_summaries(ownship: Dict[str, Any], targets: List[Dict[str, Any]], family_latent: Dict[str, Any], variant: str) -> List[Dict[str, Any]]:
    overlap = str(family_latent["rule_overlap_mode"])
    summaries: List[Dict[str, Any]] = []
    for i, t in enumerate(targets):
        sector = rel_bearing_sector(
            own_x=float(ownship["x_m"]),
            own_y=float(ownship["y_m"]),
            own_heading_deg=float(ownship["heading_deg"]),
            tgt_x=float(t["x_m"]),
            tgt_y=float(t["y_m"]),
        )
        role = str(t.get("intended_role", "lane_flow_target"))
        per_target_rules = ["COLREG_R10_TSS_GENERAL", "COLREG_R10_CROSSING"]
        if overlap in {"tss_plus_rule15", "multi_overlap"} and i == 0:
            per_target_rules += ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD"]
            if overlap == "multi_overlap" or variant in {"screen_plus_guard", "solo_overlap_probe", "paired_adjacent"}:
                per_target_rules += ["COLREG_R16_GIVEWAY_MANEUVER"]
        if overlap in {"tss_plus_rule16_17", "multi_overlap"} and i == min(1, len(targets) - 1):
            if variant in {"boundary_plus_mid", "three_lane_screen", "two_plus_guard", "multi_overlap"}:
                per_target_rules += ["COLREG_R16_GIVEWAY_MANEUVER", "COLREG_R17_STANDON"]
            else:
                per_target_rules += ["COLREG_R17_STANDON"]
        if overlap == "multi_overlap":
            if "boundary" in role:
                per_target_rules += ["COLREG_R9_NARROW_CHANNEL_GENERAL"]
            elif "dense" in role or "three_lane" in role:
                per_target_rules += ["COLREG_R13_OVERTAKING"]
            else:
                per_target_rules += ["COLREG_R18_RESPONSIBILITIES_BETWEEN_VESSELS"]
        cpa_m, tcpa_s = cpa_tcpa_by_role(role, i)
        summaries.append({
            "target_id": t["id"],
            "triggered_rule_ids": sorted(set(per_target_rules)),
            "relative_bearing_sector": sector,
            "cpa_m": round(cpa_m, 2),
            "tcpa_s": round(tcpa_s, 2),
            "risk_of_collision": True,
            "closing": True,
            "collision_imminent": i == 0 and overlap == "multi_overlap",
            "role_hint": role,
        })
    return summaries


def build_labels(family_latent: Dict[str, Any], ownship: Dict[str, Any], targets: List[Dict[str, Any]], variant: str) -> Dict[str, Any]:
    triggered_rules = derive_triggered_rules(family_latent, targets, variant)
    allowed, forbidden = derive_actions(family_latent, variant)
    target_summaries = derive_target_summaries(ownship, targets, family_latent, variant)

    occ = family_latent["lane_occupancy_band"]
    explanation_steps = [
        {"stage": "area", "summary": "TSS context is active and crossing must respect Rule 10 lane structure."},
        {"stage": "entry", "summary": f"Entry mode is {family_latent['entry_mode']} with structure variant {variant}."},
        {"stage": "occupancy", "summary": f"Lane occupancy is {occ} with {family_latent['effective_target_count']} effective targets and flow complexity {family_latent['lane_flow_complexity']}."},
        {"stage": "overlap", "summary": f"Rule overlap mode is {family_latent['rule_overlap_mode']} and crossing angle band is {family_latent['crossing_angle_band']}."},
    ]

    primitive_ledger = {
        "maneuver": {
            "CROSSING_CONTROL": {
                "polarities": ["allow"],
                "support_rules": triggered_rules,
            },
            "FLOW_SEQUENCING": {
                "polarities": ["allow"],
                "support_rules": [r for r in triggered_rules if r.startswith("COLREG_R10") or r.startswith("COLREG_R16") or r.startswith("COLREG_R17")],
            },
        }
    }
    if variant in {"boundary_stack", "boundary_plus_mid", "solo_boundary_guard"}:
        primitive_ledger["maneuver"]["BOUNDARY_AWARE_TRANSIT"] = {
            "polarities": ["allow"],
            "support_rules": [r for r in triggered_rules if r.startswith("COLREG_R10") or r.startswith("COLREG_R9")],
        }

    return {
        "triggered_rules": triggered_rules,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": primitive_ledger,
        "suppressed_rules": [],
        "suppression_records": [],
        "explanation_steps": explanation_steps,
        "maneuver_allowed": allowed,
        "maneuver_forbidden": forbidden,
        "lights_required": [],
        "sounds_required": [],
    }


def build_scene_spec(candidate_seed: int, area_context: Dict[str, Any], ownship: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "seed": candidate_seed,
        "domain": "open",
        "visibility": "in_sight",
        "snapshot_time_s": 30,
        "duration_s": 180,
        "area_context": area_context,
        "ownship": ownship,
        "targets": targets,
    }


def sample_family_latent_once(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    dist = spec.get("soft_target_distribution", {})
    crossing_angle_band = weighted_choice(rng, dist.get("crossing_angle_band", {
        "near_right_angle": 0.40,
        "moderately_off": 0.35,
        "strongly_off": 0.25,
    }))
    lane_occupancy_band = weighted_choice(rng, dist.get("lane_occupancy_band", {
        "low": 0.25,
        "medium": 0.45,
        "high": 0.30,
    }))
    rule_overlap_mode = weighted_choice(rng, dist.get("rule_overlap_mode", {
        "pure_tss": 0.25,
        "tss_plus_rule15": 0.25,
        "tss_plus_rule16_17": 0.25,
        "multi_overlap": 0.25,
    }))
    entry_mode = weighted_choice(rng, {
        "direct_crossing": 0.28,
        "edge_entry": 0.22,
        "lane_cut_in": 0.25,
        "near_boundary_crossing": 0.25,
    })
    effective_target_count = sample_effective_target_count(lane_occupancy_band)
    lane_flow_complexity = sample_lane_flow_complexity(lane_occupancy_band)

    return {
        "crossing_angle_band": crossing_angle_band,
        "lane_occupancy_band": lane_occupancy_band,
        "rule_overlap_mode": rule_overlap_mode,
        "entry_mode": entry_mode,
        "effective_target_count": effective_target_count,
        "lane_flow_complexity": lane_flow_complexity,
    }


def sample_family_latent_compatible(rng: random.Random, spec: Dict[str, Any], max_trials: int = 100) -> Tuple[Dict[str, Any], int, Optional[str]]:
    resamples = 0
    last_reason = None
    for _ in range(max_trials):
        latent = sample_family_latent_once(rng, spec)
        ok, reason = compatible_tss_latent(latent)
        if ok:
            return latent, resamples, last_reason
        resamples += 1
        last_reason = reason
    raise RuntimeError(f"Failed to sample compatible latent after {max_trials} trials; last_reason={last_reason}")


def build_tss_candidate(spec: Dict[str, Any], schema: Dict[str, Any], index: int, base_seed: int, validate_schema: bool) -> Tuple[Dict[str, Any], Optional[str], int, Optional[str]]:
    candidate_seed = base_seed + index
    rng = random.Random(candidate_seed)

    family_latent, resample_count, last_reject_reason = sample_family_latent_compatible(rng, spec)
    area_context = build_area_context(rng, spec)
    ownship = build_ownship(rng, area_context, family_latent)
    targets, variant = build_targets(rng, area_context, family_latent)
    labels = build_labels(family_latent, ownship, targets, variant)
    scene_spec = build_scene_spec(candidate_seed, area_context, ownship, targets)

    row = {
        "sample_id": stable_sample_id(index),
        "pattern": FAMILY,
        "source_mode": "native_v3_generator",
        "generator_version": spec.get("generator_version", "tss_crossing_native_v1"),
        "candidate_seed": candidate_seed,
        "family_latent": family_latent,
        "scene_spec": scene_spec,
        "inputs": {
            "topdown_image": None,
            "radar_image": None,
            "scene_text": None,
            "scene_spec": scene_spec,
        },
        "labels": labels,
        "debug": {
            "generator_revision": GENERATOR_REVISION,
            "builder_notes": [
                f"entry_mode={family_latent['entry_mode']}",
                f"rule_overlap_mode={family_latent['rule_overlap_mode']}",
                f"structure_variant={variant}",
            ],
            "compatibility_filter": {
                "resample_count": resample_count,
                "last_reject_reason": last_reject_reason,
            },
            "sanity_checks": {
                "schema_shape_ready": True,
                "target_count_matches_latent": len(targets) == int(family_latent["effective_target_count"]),
            }
        }
    }

    err = maybe_validate_schema(row, schema) if validate_schema else None
    return row, err, resample_count, last_reject_reason


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate native v3 raw candidates (P1 enhanced: tss_crossing only)."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="tss_crossing", choices=["tss_crossing"], help="Only tss_crossing is supported")
    ap.add_argument("--num-candidates", type=int, default=20, help="Number of raw candidates to generate")
    ap.add_argument("--seed", type=int, default=100000, help="Base seed")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing raw_pool/<family>/candidates_raw.jsonl")
    ap.add_argument("--validate-schema", action="store_true", help="Validate against raw_candidate.schema.json if jsonschema is available")
    ap.add_argument("--report-json", type=str, default=None, help="Optional report output path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}")
        return 2

    specs_path = root / "config" / "native_generator_specs" / f"{FAMILY}.yaml"
    schema_path = root / "schemas" / "raw_candidate.schema.json"
    out_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"

    if not specs_path.exists():
        print(f"ERROR: missing spec file: {specs_path}")
        return 2
    if not schema_path.exists():
        print(f"ERROR: missing raw candidate schema: {schema_path}")
        return 2
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        print(f"ERROR: output file exists and is non-empty; use --overwrite: {out_path}")
        return 2

    spec = read_yaml(specs_path)
    schema = read_json(schema_path)

    rows: List[Dict[str, Any]] = []
    schema_error_count = 0
    total_resamples = 0
    reject_reasons: Dict[str, int] = {}

    for i in range(1, args.num_candidates + 1):
        row, err, resample_count, last_reason = build_tss_candidate(spec, schema, i, args.seed, args.validate_schema)
        rows.append(row)
        total_resamples += resample_count
        if last_reason:
            reject_reasons[last_reason] = reject_reasons.get(last_reason, 0) + 1
        if err is not None:
            schema_error_count += 1

    write_jsonl(out_path, rows)

    report = {
        "root": str(root),
        "family": FAMILY,
        "generator_version": spec.get("generator_version", "tss_crossing_native_v1"),
        "generator_revision": GENERATOR_REVISION,
        "num_candidates_written": len(rows),
        "schema_error_count": schema_error_count,
        "jsonschema_available": HAS_JSONSCHEMA,
        "compatibility_filter_summary": {
            "total_resamples": total_resamples,
            "reject_reasons": reject_reasons,
        },
    }

    report_path = Path(args.report_json) if args.report_json else (root / "reports" / "release_reports" / "generate_native_candidates_report.json")
    write_json(report_path, report)

    print("=" * 100)
    print(f"generate_native_candidates.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows written        : {len(rows)}")
    print(f"Schema error count  : {schema_error_count}")
    print(f"Total resamples     : {total_resamples}")
    print(f"Report              : {report_path}")
    print(f"Output              : {out_path}")
    print("=" * 100)
    return 0 if schema_error_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
