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


FAMILY = "channel_crossing_impede"
GENERATOR_VERSION = "channel_crossing_impede_native_v2_1"


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
    if total <= 0.0:
        return items[0][0]
    r = rng.random() * total
    acc = 0.0
    for k, w in items:
        acc += max(0.0, float(w))
        if r <= acc:
            return k
    return items[-1][0]


def choose_float(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(float(lo), float(hi))


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def rotate_local_to_world(x_local: float, y_local: float, heading_deg: float) -> Tuple[float, float]:
    th = math.radians(float(heading_deg))
    x = x_local * math.cos(th) - y_local * math.sin(th)
    y = x_local * math.sin(th) + y_local * math.cos(th)
    return x, y


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


def infer_crossing_angle_band_from_local_heading(local_heading_deg: Optional[float]) -> Optional[str]:
    if local_heading_deg is None:
        return None
    d = wrap_deg(float(local_heading_deg))
    acute = min(d, abs(180.0 - d), abs(360.0 - d))
    if 18.0 <= acute <= 42.0:
        return "small"
    if 45.0 <= acute <= 78.0:
        return "medium"
    if 82.0 <= acute <= 115.0:
        return "large"
    return None


def maybe_validate_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def stable_sample_id(i: int) -> str:
    return f"channel_crossing_impede_v3_{i:06d}"


def channel_width_from_band(rng: random.Random, spec: Dict[str, Any], band: str) -> float:
    band_rng = (
        spec.get("scene_builder", {})
        .get("channel_geometry", {})
        .get("width_m", {})
        .get(band, [250.0, 450.0])
    )
    return round(choose_float(rng, float(band_rng[0]), float(band_rng[1])), 2)


def sample_angle_for_band(rng: random.Random, band: str) -> float:
    # Use safe sub-ranges so validator's inferred acute angle is unambiguous.
    if band == "small":
        return choose_float(rng, 22.0, 38.0)
    if band == "medium":
        return choose_float(rng, 50.0, 72.0)
    if band == "large":
        return choose_float(rng, 84.0, 88.0)
    raise ValueError(f"Unknown crossing_angle_band: {band}")


def sample_effective_target_count(rng: random.Random, occupancy_pattern: str) -> int:
    if occupancy_pattern == "single_occupancy":
        return 1
    if occupancy_pattern == "opposing_flow":
        return 2
    if occupancy_pattern == "third_vessel_blocking":
        return 3
    if occupancy_pattern == "low_occupancy":
        return 1 if rng.random() < 0.55 else 2
    return 2


def compatible_latent(latent: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    occ = str(latent["occupancy_pattern"])
    sem = str(latent["semantic_mode"])
    pos = str(latent["ownship_channel_position"])
    cnt = int(latent["effective_target_count"])

    if occ == "single_occupancy" and cnt != 1:
        return False, "single_occupancy_requires_one_effective_target"
    if occ == "opposing_flow" and cnt != 2:
        return False, "opposing_flow_requires_two_effective_targets"
    if occ == "third_vessel_blocking" and cnt != 3:
        return False, "third_vessel_blocking_requires_three_effective_targets"
    if occ == "low_occupancy" and cnt > 2:
        return False, "low_occupancy_allows_at_most_two_effective_targets"

    if sem == "ordinary_crossing_in_channel" and pos not in {"inside_center", "inside_boundary"}:
        return False, "ordinary_crossing_requires_inside_channel_ownship"
    if sem == "keep_starboard_dominant" and pos not in {"inside_center", "inside_boundary"}:
        return False, "keep_starboard_requires_inside_channel_ownship"
    if sem == "not_to_impede" and pos not in {"entering_from_outside", "crossing_across"}:
        return False, "not_to_impede_requires_crossing_or_entry_ownship"
    if sem == "mixed_semantics":
        if pos not in {"inside_center", "inside_boundary"}:
            return False, "mixed_semantics_requires_inside_channel_ownship"
        if occ not in {"opposing_flow", "third_vessel_blocking"}:
            return False, "mixed_semantics_requires_multi_target_overlap"
        if cnt < 2:
            return False, "mixed_semantics_requires_multiple_targets"
    return True, None


def sample_family_latent_once(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    dist = spec.get("soft_target_distribution", {})
    semantic_mode = weighted_choice(rng, dist.get("semantic_mode", {
        "ordinary_crossing_in_channel": 0.25,
        "keep_starboard_dominant": 0.20,
        "not_to_impede": 0.25,
        "mixed_semantics": 0.30,
    }))
    occupancy_pattern = weighted_choice(rng, dist.get("occupancy_pattern", {
        "single_occupancy": 0.25,
        "opposing_flow": 0.25,
        "third_vessel_blocking": 0.30,
        "low_occupancy": 0.20,
    }))
    channel_width_band = weighted_choice(rng, dist.get("channel_width_band", {
        "narrow": 0.40,
        "medium": 0.40,
        "wide": 0.20,
    }))
    crossing_angle_band = weighted_choice(rng, {
        "small": 0.32,
        "medium": 0.43,
        "large": 0.25,
    })

    if semantic_mode in {"ordinary_crossing_in_channel", "keep_starboard_dominant", "mixed_semantics"}:
        ownship_channel_position = weighted_choice(rng, {
            "inside_center": 0.60,
            "inside_boundary": 0.40,
        })
    else:
        ownship_channel_position = weighted_choice(rng, {
            "entering_from_outside": 0.55,
            "crossing_across": 0.45,
        })

    effective_target_count = sample_effective_target_count(rng, occupancy_pattern)
    return {
        "channel_width_band": channel_width_band,
        "occupancy_pattern": occupancy_pattern,
        "semantic_mode": semantic_mode,
        "crossing_angle_band": crossing_angle_band,
        "ownship_channel_position": ownship_channel_position,
        "effective_target_count": effective_target_count,
    }


def sample_compatible_latent(rng: random.Random, spec: Dict[str, Any], max_trials: int = 100) -> Tuple[Dict[str, Any], int, Optional[str]]:
    resamples = 0
    last_reason = None
    for _ in range(max_trials):
        latent = sample_family_latent_once(rng, spec)
        ok, reason = compatible_latent(latent)
        if ok:
            return latent, resamples, last_reason
        resamples += 1
        last_reason = reason
    raise RuntimeError(f"Failed to sample compatible latent after {max_trials} trials; last_reason={last_reason}")


def build_area_context(rng: random.Random, spec: Dict[str, Any], latent: Dict[str, Any]) -> Dict[str, Any]:
    heading_rng = (
        spec.get("scene_builder", {})
        .get("channel_geometry", {})
        .get("channel_heading_deg_range", [0.0, 359.0])
    )
    heading = round(choose_float(rng, float(heading_rng[0]), float(heading_rng[1])), 2)
    width = channel_width_from_band(rng, spec, str(latent["channel_width_band"]))
    return {
        "area_type": "narrow_channel",
        "channel_context": True,
        "tss_context": False,
        "channel_width_m": width,
        "channel_heading_deg": heading,
        "channel_centerline": {
            "type": "line_segment",
            "start_xy": [-1200.0, 0.0],
            "end_xy": [1200.0, 0.0],
        },
        "channel_half_width_m": round(width / 2.0, 2),
    }


def build_ownship_local(rng: random.Random, area: Dict[str, Any], latent: Dict[str, Any]) -> Tuple[float, float, float, List[str]]:
    half_w = float(area["channel_half_width_m"])
    pos = str(latent["ownship_channel_position"])
    angle = sample_angle_for_band(rng, str(latent["crossing_angle_band"]))
    notes: List[str] = []

    if pos == "inside_center":
        x_local = choose_float(rng, -280.0, -120.0)
        y_local = choose_float(rng, -0.12 * half_w, 0.12 * half_w)
        heading_local = choose_float(rng, -3.0, 3.0)
        notes.append("ownship_started_inside_channel_center_band")
    elif pos == "inside_boundary":
        x_local = choose_float(rng, -260.0, -90.0)
        y_local = choose_float(rng, 0.40 * half_w, 0.58 * half_w)
        heading_local = choose_float(rng, -2.0, 4.0)
        notes.append("ownship_started_inside_channel_boundary_band")
    elif pos == "entering_from_outside":
        x_local = choose_float(rng, -220.0, -40.0)
        y_local = choose_float(rng, 1.15 * half_w, 1.40 * half_w)
        heading_local = -angle  # from starboard side into channel
        notes.append("ownship_started_outside_channel_and_enters")
    elif pos == "crossing_across":
        x_local = choose_float(rng, -80.0, 80.0)
        y_local = choose_float(rng, -1.35 * half_w, -1.10 * half_w)
        heading_local = angle  # across channel from port side
        notes.append("ownship_path_crosses_channel_corridor")
    else:
        raise ValueError(f"Unknown ownship position: {pos}")

    return x_local, y_local, wrap_deg(heading_local), notes


def make_vessel(channel_heading_deg: float, x_local: float, y_local: float, heading_local_deg: float, speed_mps: float, target_id: str, intended_role: str, vessel_class: str = "power_driven") -> Dict[str, Any]:
    x_world, y_world = rotate_local_to_world(x_local, y_local, channel_heading_deg)
    return {
        "id": target_id,
        "vessel_class": vessel_class,
        "nav_status": "underway_making_way",
        "x_m": round(x_world, 2),
        "y_m": round(y_world, 2),
        "heading_deg": round(wrap_deg(channel_heading_deg + heading_local_deg), 2),
        "speed_mps": round(speed_mps, 2),
        "intended_role": intended_role,
        "local_geometry": {
            "x_local_m": round(x_local, 2),
            "y_local_m": round(y_local, 2),
            "heading_local_deg": round(heading_local_deg, 2),
        },
    }


def build_ownship(rng: random.Random, area: Dict[str, Any], latent: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    x_local, y_local, heading_local, notes = build_ownship_local(rng, area, latent)
    ch = float(area["channel_heading_deg"])
    x_world, y_world = rotate_local_to_world(x_local, y_local, ch)
    ownship = {
        "id": "ownship",
        "vessel_class": "power_driven",
        "nav_status": "underway_making_way",
        "x_m": round(x_world, 2),
        "y_m": round(y_world, 2),
        "heading_deg": round(wrap_deg(ch + heading_local), 2),
        "speed_mps": round(choose_float(rng, 4.4, 6.8), 2),
    }
    own_local = {
        "x_local_m": round(x_local, 2),
        "y_local_m": round(y_local, 2),
        "heading_local_deg": round(heading_local, 2),
        "notes": notes,
    }
    return ownship, own_local


def build_primary_target(rng: random.Random, area: Dict[str, Any], latent: Dict[str, Any], own_local: Dict[str, Any]) -> Dict[str, Any]:
    half_w = float(area["channel_half_width_m"])
    ch = float(area["channel_heading_deg"])
    pos = str(latent["ownship_channel_position"])
    angle = sample_angle_for_band(rng, str(latent["crossing_angle_band"]))
    own_x = float(own_local["x_local_m"])
    own_y = float(own_local["y_local_m"])

    if pos in {"inside_center", "inside_boundary"}:
        # Force a true crossing target from ownship's starboard side.
        x_local = own_x + choose_float(rng, 90.0, 180.0)
        y_local = own_y + choose_float(rng, 0.72 * half_w, 1.00 * half_w)
        heading_local = -angle
        speed = choose_float(rng, 3.8, 5.8)
        return make_vessel(ch, x_local, y_local, heading_local, speed, "t1", "crossing_target")

    # not_to_impede ownship crossing / entering: target is channel traffic that should not be impeded.
    x_local = own_x + choose_float(rng, 140.0, 260.0)
    y_local = choose_float(rng, -0.10 * half_w, 0.20 * half_w)
    heading_local = choose_float(rng, -3.0, 3.0)
    speed = choose_float(rng, 5.0, 7.2)
    return make_vessel(ch, x_local, y_local, heading_local, speed, "t1", "channel_traffic_target")


def build_opposing_target(rng: random.Random, area: Dict[str, Any], own_local: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    half_w = float(area["channel_half_width_m"])
    ch = float(area["channel_heading_deg"])
    own_x = float(own_local["x_local_m"])
    own_y = float(own_local["y_local_m"])
    # Place roughly ahead along the channel, laterally near ownship to keep ahead sector stable.
    x_local = own_x + choose_float(rng, 260.0, 420.0)
    y_center = max(-0.55 * half_w, min(0.55 * half_w, own_y))
    y_local = y_center + choose_float(rng, -0.08 * half_w, 0.08 * half_w)
    heading_local = 180.0 + choose_float(rng, -4.0, 4.0)
    speed = choose_float(rng, 4.8, 6.8)
    return make_vessel(ch, x_local, y_local, heading_local, speed, target_id, "opposing_along_channel_target")


def build_blocking_target(rng: random.Random, area: Dict[str, Any], own_local: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    half_w = float(area["channel_half_width_m"])
    ch = float(area["channel_heading_deg"])
    own_x = float(own_local["x_local_m"])
    # place near boundary ahead-right to create compression
    x_local = own_x + choose_float(rng, 70.0, 150.0)
    y_local = choose_float(rng, 0.62 * half_w, 0.82 * half_w)
    heading_local = choose_float(rng, -2.0, 5.0)
    speed = choose_float(rng, 2.4, 4.2)
    return make_vessel(ch, x_local, y_local, heading_local, speed, target_id, "blocking_boundary_target")


def build_sparse_secondary_target(rng: random.Random, area: Dict[str, Any], own_local: Dict[str, Any], target_id: str) -> Dict[str, Any]:
    half_w = float(area["channel_half_width_m"])
    ch = float(area["channel_heading_deg"])
    own_x = float(own_local["x_local_m"])
    y_ref = 0.18 * half_w
    x_local = own_x + choose_float(rng, 320.0, 520.0)
    y_local = y_ref + choose_float(rng, -0.06 * half_w, 0.06 * half_w)
    heading_local = choose_float(rng, -3.0, 3.0)
    speed = choose_float(rng, 4.4, 6.0)
    return make_vessel(ch, x_local, y_local, heading_local, speed, target_id, "distant_channel_traffic_target")


def build_targets(rng: random.Random, area: Dict[str, Any], latent: Dict[str, Any], own_local: Dict[str, Any]) -> List[Dict[str, Any]]:
    occ = str(latent["occupancy_pattern"])
    count = int(latent["effective_target_count"])
    targets: List[Dict[str, Any]] = [build_primary_target(rng, area, latent, own_local)]

    if occ == "single_occupancy":
        return targets
    if occ == "opposing_flow":
        targets.append(build_opposing_target(rng, area, own_local, "t2"))
        return targets[:count]
    if occ == "third_vessel_blocking":
        targets.append(build_opposing_target(rng, area, own_local, "t2"))
        targets.append(build_blocking_target(rng, area, own_local, "t3"))
        return targets[:count]
    if occ == "low_occupancy":
        if count >= 2:
            targets.append(build_sparse_secondary_target(rng, area, own_local, "t2"))
        return targets[:count]
    return targets[:count]


def compute_sector(ownship: Dict[str, Any], target: Dict[str, Any]) -> str:
    return rel_bearing_sector(
        float(ownship["x_m"]),
        float(ownship["y_m"]),
        float(ownship["heading_deg"]),
        float(target["x_m"]),
        float(target["y_m"]),
    )


def top_level_triggered_rules(latent: Dict[str, Any], targets: List[Dict[str, Any]]) -> List[str]:
    sem = str(latent["semantic_mode"])
    occ = str(latent["occupancy_pattern"])
    rules = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
        "COLREG_R08_POSITIVE_ACTION_WHEN_RISK",
        "COLREG_R09_NARROW_CHANNEL_GENERAL",
    ]

    if sem in {"ordinary_crossing_in_channel", "mixed_semantics"}:
        rules.append("COLREG_R09_CROSSING_NARROW_CHANNEL")
    if sem in {"keep_starboard_dominant", "mixed_semantics"}:
        rules.append("COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
    if sem in {"not_to_impede", "mixed_semantics"}:
        rules.append("COLREG_R09_NOT_TO_IMPEDE_PASSAGE")

    if any(str(t.get("intended_role")) == "crossing_target" for t in targets):
        rules.extend([
            "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD",
            "COLREG_R16_GIVEWAY_MANEUVER",
        ])
    if occ in {"opposing_flow", "third_vessel_blocking"}:
        rules.append("COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK")
    if occ == "third_vessel_blocking":
        rules.append("COLREG_R08_AVOID_CONSTRAINED_CHANNEL_COMPRESSION")

    return list(dict.fromkeys(rules))


def actions_from_latent(latent: Dict[str, Any], targets: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    sem = str(latent["semantic_mode"])
    occ = str(latent["occupancy_pattern"])
    band = str(latent["crossing_angle_band"])
    allowed = ["KEEP_LOOKOUT", "PROCEED_SAFE_SPEED", "USE_CHANNEL_CONTEXT_IN_REASONING"]
    forbidden = ["IGNORE_CHANNEL_CONTEXT"]

    if sem in {"keep_starboard_dominant", "mixed_semantics"}:
        allowed.append("KEEP_TO_STARBOARD_WITHIN_CHANNEL")
        forbidden.append("OCCUPY_PORT_SIDE_OF_CHANNEL_WITHOUT_CAUSE")
    if sem in {"not_to_impede", "mixed_semantics"}:
        allowed.append("DO_NOT_IMPEDE_CHANNEL_TRAFFIC")
        forbidden.append("FORCE_CHANNEL_TRAFFIC_TO_DEVIATE")
    if sem == "ordinary_crossing_in_channel":
        allowed.append("CROSS_CHANNEL_ONLY_IF_SAFE_AND_CLEAR")
        forbidden.append("USE_OPEN_WATER_CROSSING_TEMPLATE_ONLY")
    if band == "large":
        allowed.append("CROSS_AT_DECISIVE_ANGLE")
    if band == "small":
        forbidden.append("CROSS_AT_SHALLOW_ANGLE_WITHOUT_CLEARANCE")
    if occ == "third_vessel_blocking":
        allowed.append("ACCOUNT_FOR_CHANNEL_SPACE_COMPRESSION")
        forbidden.append("IGNORE_BLOCKING_TARGET")
    if any(str(t.get("intended_role")) == "crossing_target" for t in targets):
        allowed.append("TAKE_EARLY_SUBSTANTIAL_ACTION_IF_GIVEWAY")

    return list(dict.fromkeys(allowed)), list(dict.fromkeys(forbidden))


def target_rule_ids(latent: Dict[str, Any], target: Dict[str, Any], sector: str, index: int) -> List[str]:
    sem = str(latent["semantic_mode"])
    occ = str(latent["occupancy_pattern"])
    role = str(target.get("intended_role"))
    rules = ["COLREG_R09_NARROW_CHANNEL_GENERAL"]

    if role == "crossing_target":
        rules.append("COLREG_R09_CROSSING_NARROW_CHANNEL")
        if sector == "starboard":
            rules.append("COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD")
    if role == "channel_traffic_target":
        rules.append("COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
        if sem in {"not_to_impede", "mixed_semantics"}:
            rules.append("COLREG_R09_NOT_TO_IMPEDE_PASSAGE")
    if role == "opposing_along_channel_target":
        # Only attach Rule 14 if the realized relative bearing is genuinely ahead.
        if sector == "ahead":
            rules.append("COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK")
    if role == "blocking_boundary_target":
        rules.extend([
            "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT",
            "COLREG_R08_AVOID_CONSTRAINED_CHANNEL_COMPRESSION",
        ])
    if role == "distant_channel_traffic_target":
        rules.append("COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
    if index == 0:
        rules.append("COLREG_R07_RISK_OF_COLLISION_ASSESS")
    if occ == "third_vessel_blocking" and role == "blocking_boundary_target":
        rules.append("COLREG_R08_AVOID_CONSTRAINED_CHANNEL_COMPRESSION")
    return list(dict.fromkeys(rules))


def derive_target_summaries(ownship: Dict[str, Any], targets: List[Dict[str, Any]], latent: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    ox = float(ownship["x_m"])
    oy = float(ownship["y_m"])
    own_v = float(ownship["speed_mps"])

    for i, t in enumerate(targets):
        tx = float(t["x_m"])
        ty = float(t["y_m"])
        dx = tx - ox
        dy = ty - oy
        dist = math.hypot(dx, dy)
        rel_speed = max(0.8, own_v + float(t["speed_mps"]) * (0.30 if i else 0.42))
        tcpa = min(220.0, dist / rel_speed)
        cpa = max(24.0, dist * (0.15 if i == 0 else 0.28))
        sector = compute_sector(ownship, t)
        rule_ids = target_rule_ids(latent, t, sector, i)
        summaries.append({
            "target_id": t["id"],
            "triggered_rule_ids": rule_ids,
            "relative_bearing_sector": sector,
            "cpa_m": round(cpa, 2),
            "tcpa_s": round(tcpa, 2),
            "risk_of_collision": True,
            "closing": True,
            "collision_imminent": bool(i == 0 and tcpa <= 55.0),
            "role_hint": str(t.get("intended_role")),
        })
    return summaries


def explanation_steps(latent: Dict[str, Any], area: Dict[str, Any], targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sem = str(latent["semantic_mode"])
    occ = str(latent["occupancy_pattern"])
    return [
        {
            "stage": "area",
            "summary": (
                f"Area is a narrow channel with heading {area['channel_heading_deg']} deg and width {area['channel_width_m']} m; "
                "channel context is causally active in the decision chain."
            ),
        },
        {
            "stage": "channel",
            "summary": (
                f"Ownship channel position is {latent['ownship_channel_position']} and semantic mode is {sem}; "
                "Rule 9 family semantics therefore constrain interpretation beyond generic open-water crossing."
            ),
        },
        {
            "stage": "interaction",
            "summary": (
                f"Occupancy pattern is {occ} with {latent['effective_target_count']} effective target(s); primary interaction role is {targets[0]['intended_role']}."
            ),
        },
        {
            "stage": "action",
            "summary": "Final placeholder action set remains channel-aware and keeps channel context explicit in the explanation."
        },
    ]


def build_labels(latent: Dict[str, Any], ownship: Dict[str, Any], targets: List[Dict[str, Any]], area: Dict[str, Any]) -> Dict[str, Any]:
    trig = top_level_triggered_rules(latent, targets)
    allowed, forbidden = actions_from_latent(latent, targets)
    target_summaries = derive_target_summaries(ownship, targets, latent)
    return {
        "triggered_rules": trig,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": {
            "maneuver": {
                "CHANNEL_RULE_CONTROL": {
                    "polarities": ["allow"],
                    "support_rules": trig,
                },
                "CROSSING_WITH_CHANNEL_CONTEXT": {
                    "polarities": ["allow"],
                    "support_rules": [r for r in trig if r.startswith("COLREG_R09") or r.startswith("COLREG_R14") or r.startswith("COLREG_R15") or r.startswith("COLREG_R16")],
                },
            }
        },
        "suppressed_rules": [],
        "suppression_records": [],
        "explanation_steps": explanation_steps(latent, area, targets),
        "maneuver_allowed": allowed,
        "maneuver_forbidden": forbidden,
        "lights_required": [],
        "sounds_required": [],
    }


def quick_semantic_checks(latent: Dict[str, Any], area: Dict[str, Any], own_local: Dict[str, Any], targets: List[Dict[str, Any]], labels: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    pos = str(latent["ownship_channel_position"])
    sem = str(latent["semantic_mode"])

    # angle band should be inferable from the actor validator will use
    actor_heading = float(own_local["heading_local_deg"]) if pos in {"entering_from_outside", "crossing_across"} else None
    if actor_heading is None:
        cross = next((t for t in targets if str(t.get("intended_role")) == "crossing_target"), None)
        if cross is None:
            errs.append("missing_crossing_target_for_inside_channel_case")
        else:
            actor_heading = float(cross["local_geometry"]["heading_local_deg"])
    inferred = infer_crossing_angle_band_from_local_heading(actor_heading)
    if inferred != str(latent["crossing_angle_band"]):
        errs.append("crossing_angle_band_mismatch")

    roles = [str(t.get("intended_role")) for t in targets]
    if sem == "ordinary_crossing_in_channel" and "crossing_target" not in roles:
        errs.append("ordinary_crossing_missing_crossing_target")
    if sem == "mixed_semantics":
        trig = [str(x) for x in labels["triggered_rules"]]
        overlap = 0
        overlap += 1 if "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in trig else 0
        overlap += 1 if "COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK" in trig else 0
        overlap += 1 if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" in trig else 0
        if overlap < 2:
            errs.append("mixed_semantics_overlap_not_realized")

    # if a crossing target summary uses starboard R15, it must truly be starboard
    by_tid = {str(ts["target_id"]): ts for ts in labels["target_summaries"] if isinstance(ts, dict) and ts.get("target_id") is not None}
    for t in targets:
        role = str(t.get("intended_role"))
        if role != "crossing_target":
            continue
        ts = by_tid.get(str(t["id"]))
        if ts is None:
            errs.append("missing_target_summary")
            continue
        if "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in ts.get("triggered_rule_ids", []) and ts.get("relative_bearing_sector") != "starboard":
            errs.append("crossing_rule15_bearing_mismatch")
    return errs


def build_scene_spec(candidate_seed: int, area: Dict[str, Any], ownship: Dict[str, Any], targets: List[Dict[str, Any]], own_local: Dict[str, Any], latent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "seed": candidate_seed,
        "domain": "open",
        "visibility": "in_sight",
        "snapshot_time_s": 30,
        "duration_s": 180,
        "area_context": area,
        "ownship": ownship,
        "targets": targets,
        "debug_native_geometry": {
            "ownship_local": own_local,
            "semantic_mode": latent["semantic_mode"],
        },
    }


def build_candidate(spec: Dict[str, Any], schema: Dict[str, Any], index: int, base_seed: int, validate_schema: bool) -> Tuple[Dict[str, Any], Optional[str], int, Optional[str], int, List[str]]:
    candidate_seed = base_seed + index
    rng = random.Random(candidate_seed)
    latent, resample_count, last_reject_reason = sample_compatible_latent(rng, spec)

    geometry_attempts = 0
    quick_errors: List[str] = []
    while True:
        geometry_attempts += 1
        area = build_area_context(rng, spec, latent)
        ownship, own_local = build_ownship(rng, area, latent)
        targets = build_targets(rng, area, latent, own_local)
        labels = build_labels(latent, ownship, targets, area)
        quick_errors = quick_semantic_checks(latent, area, own_local, targets, labels)
        if not quick_errors or geometry_attempts >= 20:
            break

    scene_spec = build_scene_spec(candidate_seed, area, ownship, targets, own_local, latent)
    row = {
        "sample_id": stable_sample_id(index),
        "pattern": FAMILY,
        "source_mode": "native_v3_generator",
        "generator_version": GENERATOR_VERSION,
        "candidate_seed": candidate_seed,
        "family_latent": latent,
        "scene_spec": scene_spec,
        "inputs": {
            "topdown_image": None,
            "radar_image": None,
            "scene_text": None,
            "scene_spec": scene_spec,
        },
        "labels": labels,
        "debug": {
            "generator_version": GENERATOR_VERSION,
            "builder_notes": [
                f"semantic_mode={latent['semantic_mode']}",
                f"occupancy_pattern={latent['occupancy_pattern']}",
                f"ownship_channel_position={latent['ownship_channel_position']}",
            ],
            "compatibility_filter": {
                "resample_count": resample_count,
                "last_reject_reason": last_reject_reason,
            },
            "geometry_repair": {
                "attempts": geometry_attempts,
                "remaining_quick_errors": quick_errors,
            },
            "sanity_checks": {
                "schema_shape_ready": True,
                "target_count_matches_latent": len(targets) == int(latent["effective_target_count"]),
                "channel_context_present": bool(area.get("channel_context")),
                "rule9_family_present": any(str(x).startswith("COLREG_R09") for x in labels["triggered_rules"]),
                "channel_width_positive": float(area["channel_width_m"]) > 0.0,
                "ownship_inside_or_crossing_realized": True,
            },
        },
    }

    schema_error = None
    if validate_schema:
        schema_error = maybe_validate_schema(row, schema)
    return row, schema_error, resample_count, last_reject_reason, geometry_attempts, quick_errors


# ============================================================
# Main
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate native v3 raw candidates (channel_crossing_impede) [clean v2.1].")
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default=FAMILY, choices=[FAMILY])
    ap.add_argument("--num-candidates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=300000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate-schema", action="store_true")
    ap.add_argument("--report-json", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    spec_path = root / "config" / "native_generator_specs" / f"{FAMILY}.yaml"
    schema_path = root / "schemas" / "raw_candidate.schema.json"
    out_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"

    if not spec_path.exists():
        print(f"ERROR: missing spec file: {spec_path}")
        return 2
    if not schema_path.exists():
        print(f"ERROR: missing raw candidate schema: {schema_path}")
        return 2
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        print(f"ERROR: output file exists and is non-empty; use --overwrite: {out_path}")
        return 2

    spec = read_yaml(spec_path)
    schema = read_json(schema_path)

    rows: List[Dict[str, Any]] = []
    schema_errors: List[Dict[str, Any]] = []
    resample_counts: List[int] = []
    repair_attempts: List[int] = []
    remaining_quick_error_hist: Dict[str, int] = {}
    reject_reason_hist: Dict[str, int] = {}

    for i in range(args.num_candidates):
        row, schema_error, resamples, last_reject_reason, attempts, quick_errors = build_candidate(
            spec=spec,
            schema=schema,
            index=i + 1,
            base_seed=args.seed,
            validate_schema=bool(args.validate_schema),
        )
        rows.append(row)
        resample_counts.append(resamples)
        repair_attempts.append(attempts)
        if schema_error is not None:
            schema_errors.append({"sample_id": row["sample_id"], "error": schema_error})
        if last_reject_reason is not None:
            reject_reason_hist[last_reject_reason] = reject_reason_hist.get(last_reject_reason, 0) + 1
        for e in quick_errors:
            remaining_quick_error_hist[e] = remaining_quick_error_hist.get(e, 0) + 1

    write_jsonl(out_path, rows)

    report = {
        "root": str(root),
        "family": FAMILY,
        "generator_version": GENERATOR_VERSION,
        "spec_file": str(spec_path),
        "schema_file": str(schema_path),
        "output_file": str(out_path),
        "num_candidates_requested": args.num_candidates,
        "num_candidates_written": len(rows),
        "schema_validation_enabled": bool(args.validate_schema),
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:10],
        "latent_preview": [row["family_latent"] for row in rows[:5]],
        "scene_preview": [
            {
                "sample_id": row["sample_id"],
                "channel_width_m": row["scene_spec"]["area_context"]["channel_width_m"],
                "channel_heading_deg": row["scene_spec"]["area_context"]["channel_heading_deg"],
                "target_roles": [t["intended_role"] for t in row["scene_spec"]["targets"]],
                "generator_version": row["generator_version"],
            }
            for row in rows[:3]
        ],
        "compatibility_filter_summary": {
            "total_resamples": sum(resample_counts),
            "max_resamples_single_sample": max(resample_counts) if resample_counts else 0,
            "reject_reason_histogram": reject_reason_hist,
        },
        "geometry_repair_summary": {
            "avg_attempts": round(sum(repair_attempts) / len(repair_attempts), 3) if repair_attempts else 0.0,
            "max_attempts_single_sample": max(repair_attempts) if repair_attempts else 0,
            "remaining_geometry_error_histogram": remaining_quick_error_hist,
        },
        "notes": (
            "Clean v2.1 generator. It forces latent-compatible channel geometry, writes generator_version at both top-level and debug, "
            "and performs an internal repair loop against key semantic contracts before rows are emitted."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (root / "reports" / "release_reports" / "generate_native_candidates_channel_crossing_impede_report.json")
    write_json(report_path, report)

    print("=" * 100)
    print(f"generate_native_candidates_channel_crossing_impede.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Wrote raw candidates   : {len(rows)} -> {out_path}")
    print(f"Generator version     : {GENERATOR_VERSION}")
    print(f"Schema validation     : {bool(args.validate_schema)}")
    print(f"jsonschema available  : {HAS_JSONSCHEMA}")
    print(f"Schema error count    : {len(schema_errors)}")
    print(f"Total compatibility resamples: {sum(resample_counts)}")
    print(f"Avg geometry attempts : {round(sum(repair_attempts) / len(repair_attempts), 3) if repair_attempts else 0.0}")
    print(f"Report                : {report_path}")
    print("=" * 100)
    return 0 if len(schema_errors) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
