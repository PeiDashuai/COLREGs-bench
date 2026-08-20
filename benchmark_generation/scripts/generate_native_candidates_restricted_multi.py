#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
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


FAMILY = "restricted_multi"
GENERATOR_VERSION = "restricted_multi_native_v1"  # keep validator contract unchanged
GENERATOR_REVISION = "restricted_multi_generator_v2"
SOURCE_MODE = "native_v3_generator"


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

def weighted_choice(rng: random.Random, dist: Dict[Any, float]) -> Any:
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


def choose_float_from_range(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(float(lo), float(hi))


def choose_int_from_range(rng: random.Random, lo: int, hi: int) -> int:
    return rng.randint(int(lo), int(hi))


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(wrap_deg(a) - wrap_deg(b))
    return min(d, 360.0 - d)


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


def maybe_validate_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def stable_sample_id(i: int) -> str:
    return f"restricted_multi_v3_{i:06d}"


def pairwise_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    return math.hypot(float(a["x_m"]) - float(b["x_m"]), float(a["y_m"]) - float(b["y_m"]))


def ensure_unique_rule_list(rules: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for r in rules:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


# ============================================================
# Latent sampling / compatibility
# ============================================================

def sample_family_latent(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    dist = spec.get("soft_target_distribution", {})

    num_effective_targets = int(weighted_choice(rng, {int(k): float(v) for k, v in dist.get("num_effective_targets", {3: 0.3, 4: 0.3, 5: 0.25, 6: 0.15}).items()}))
    spatial_topology = str(weighted_choice(rng, dist.get("spatial_topology", {
        "single_side_cluster": 0.20,
        "bilateral_constraint": 0.30,
        "frontal_blocking": 0.25,
        "surrounding_ring": 0.25,
    })))
    suppression_mechanism = str(weighted_choice(rng, dist.get("suppression_mechanism", {
        "action_conflict": 0.30,
        "hierarchy_override": 0.25,
        "area_override": 0.15,
        "mixed": 0.30,
    })))
    urgency_band = str(weighted_choice(rng, dist.get("urgency_band", {
        "low": 0.20,
        "medium": 0.45,
        "high": 0.35,
    })))

    visibility_mode = str(weighted_choice(rng, {"in_sight": 0.65, "restricted_visibility": 0.35}))
    target_status_mix = str(weighted_choice(rng, {"all_power": 0.55, "mixed_status": 0.45}))
    interaction_density = str(weighted_choice(rng, {"sparse": 0.20, "medium": 0.50, "dense": 0.30}))

    return {
        "num_effective_targets": num_effective_targets,
        "spatial_topology": spatial_topology,
        "suppression_mechanism": suppression_mechanism,
        "urgency_band": urgency_band,
        "visibility_mode": visibility_mode,
        "target_status_mix": target_status_mix,
        "interaction_density": interaction_density,
    }


def compatible_restricted_multi_latent(latent: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    n = int(latent["num_effective_targets"])
    topo = str(latent["spatial_topology"])
    supp = str(latent["suppression_mechanism"])
    urg = str(latent["urgency_band"])
    vis = str(latent["visibility_mode"])
    mix = str(latent["target_status_mix"])
    density = str(latent["interaction_density"])

    if topo == "surrounding_ring" and n < 4:
        return False, "surrounding_ring_requires_at_least_4_targets"
    if topo == "frontal_blocking" and n < 3:
        return False, "frontal_blocking_requires_at_least_3_targets"
    if topo == "bilateral_constraint" and n < 3:
        return False, "bilateral_constraint_requires_at_least_3_targets"
    if supp == "hierarchy_override" and mix != "mixed_status":
        return False, "hierarchy_override_requires_mixed_status"
    if supp == "area_override" and vis == "in_sight" and topo == "single_side_cluster" and density == "sparse":
        return False, "area_override_not_meaningful_with_simple_sparse_single_side"
    if supp == "mixed" and n < 4:
        return False, "mixed_suppression_requires_at_least_4_targets"
    if supp == "mixed" and density == "sparse":
        return False, "mixed_suppression_not_allowed_with_sparse_density"
    if supp == "action_conflict" and topo == "single_side_cluster" and density == "sparse":
        return False, "action_conflict_requires_richer_geometry_than_sparse_single_side"
    if urg == "high" and density == "sparse" and n <= 3 and topo == "single_side_cluster":
        return False, "high_urgency_sparse_single_side_tends_to_be_single_target_dominant"
    if vis == "restricted_visibility" and topo == "single_side_cluster" and n <= 3 and supp == "area_override":
        return False, "restricted_visibility_area_override_too_weak_in_small_single_side_cases"
    return True, None


# ============================================================
# Builder
# ============================================================

def build_area_context(rng: random.Random, latent: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    supp = str(latent["suppression_mechanism"])
    vis = str(latent["visibility_mode"])

    area_type = "open_water"
    channel_context = False
    tss_context = False
    channel_width_m = None
    channel_heading_deg = None
    channel_centerline = None
    lane_count = None
    lane_width_m = None
    tss_lane_heading_deg = None
    separation_zone_geometry = None
    area_source = "none"

    if vis == "restricted_visibility":
        area_source = "restricted_visibility"

    if supp in {"area_override", "mixed"}:
        sources = spec.get("scene_builder", {}).get("suppression_constraints", {}).get(supp, {}).get(
            "allowed_area_sources", ["channel", "tss", "restricted_visibility"]
        )
        area_source = rng.choice(list(sources))

    if area_source == "channel":
        area_type = "narrow_channel"
        channel_context = True
        tss_context = False
        channel_width_m = round(choose_float_from_range(rng, 260.0, 620.0), 1)
        channel_heading_deg = round(wrap_deg(rng.choice([0.0, 45.0, 90.0, 135.0, 180.0]) + choose_float_from_range(rng, -8.0, 8.0)), 1)
        channel_centerline = {
            "type": "line_segment",
            "start_xy": [-2200.0, 0.0],
            "end_xy": [2200.0, 0.0],
        }
    elif area_source == "tss":
        area_type = "tss"
        channel_context = False
        tss_context = True
        lane_count = rng.choice([2, 3, 4])
        lane_width_m = round(choose_float_from_range(rng, 180.0, 360.0), 1)
        tss_lane_heading_deg = round(wrap_deg(rng.choice([0.0, 90.0, 180.0]) + choose_float_from_range(rng, -5.0, 5.0)), 1)
        separation_zone_geometry = {
            "type": "strip",
            "width_m": round(choose_float_from_range(rng, 80.0, 180.0), 1),
        }
    else:
        area_type = "open_water"
        channel_context = False
        tss_context = False

    out = {
        "area_type": area_type,
        "channel_context": channel_context,
        "tss_context": tss_context,
        "area_source": area_source,
    }
    if channel_width_m is not None:
        out["channel_width_m"] = channel_width_m
    if channel_heading_deg is not None:
        out["channel_heading_deg"] = channel_heading_deg
    if channel_centerline is not None:
        out["channel_centerline"] = channel_centerline
    if lane_count is not None:
        out["lane_count"] = lane_count
    if lane_width_m is not None:
        out["lane_width_m"] = lane_width_m
    if tss_lane_heading_deg is not None:
        out["tss_lane_heading_deg"] = tss_lane_heading_deg
    if separation_zone_geometry is not None:
        out["separation_zone_geometry"] = separation_zone_geometry
    return out


def build_ownship(rng: random.Random, latent: Dict[str, Any], spec: Dict[str, Any], area_context: Dict[str, Any]) -> Dict[str, Any]:
    own_cfg = spec.get("scene_builder", {}).get("ownship", {})
    speed_rng = own_cfg.get("speed_mps_range", [4.0, 8.0])
    own_heading_deg = round(wrap_deg(rng.choice([0.0, 45.0, 90.0, 135.0]) + choose_float_from_range(rng, -6.0, 6.0)), 1)
    if area_context.get("channel_context"):
        own_heading_deg = float(area_context.get("channel_heading_deg", own_heading_deg))
    if area_context.get("tss_context"):
        own_heading_deg = float(area_context.get("tss_lane_heading_deg", own_heading_deg))
    return {
        "id": "ownship",
        "vessel_class": str(own_cfg.get("vessel_class", "power_driven")),
        "nav_status": str(own_cfg.get("nav_status", "underway_making_way")),
        "x_m": 0.0,
        "y_m": 0.0,
        "heading_deg": own_heading_deg,
        "speed_mps": round(choose_float_from_range(rng, float(speed_rng[0]), float(speed_rng[1])), 2),
        "length_m": round(choose_float_from_range(rng, 70.0, 220.0), 1),
    }


def build_topology_roles(latent: Dict[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    n = int(latent["num_effective_targets"])
    topo = str(latent["spatial_topology"])
    roles: List[Dict[str, Any]] = []

    if topo == "single_side_cluster":
        # v2 patch:
        # Make the topology strongly side-dominant rather than merely side-biased.
        # Validator expects side_total >= n-1 and a clear dominant side.
        side = rng.choice(["port", "starboard"])
        role_names = [
            "lead",
            "beam",
            "quarter",
            "inner",
            "outer",
            "tail",
        ]
        if n <= 5:
            roles = [
                {"slot_role": f"{side}_{role_names[i]}", "sector": side, "priority_hint": "primary" if i == 0 else "support"}
                for i in range(n)
            ]
        else:
            # Keep one weak contextual target outside the dominant side, but only when n is large enough
            # that side_total still satisfies the validator threshold.
            dominant_count = n - 1
            roles = [
                {"slot_role": f"{side}_{role_names[i % len(role_names)]}", "sector": side, "priority_hint": "primary" if i == 0 else "support"}
                for i in range(dominant_count)
            ]
            roles.append({"slot_role": "ahead_context_weak", "sector": "ahead", "priority_hint": "context"})

    elif topo == "bilateral_constraint":
        roles.extend([
            {"slot_role": "port_constraint", "sector": "port", "priority_hint": "primary"},
            {"slot_role": "starboard_constraint", "sector": "starboard", "priority_hint": "primary"},
            {"slot_role": "ahead_context", "sector": "ahead", "priority_hint": "support"},
        ])
        extras = [
            {"slot_role": "port_reinforce", "sector": "port", "priority_hint": "support"},
            {"slot_role": "starboard_reinforce", "sector": "starboard", "priority_hint": "support"},
            {"slot_role": "astern_context", "sector": "astern", "priority_hint": "context"},
        ]
        roles.extend(extras[: max(0, n - 3)])

    elif topo == "frontal_blocking":
        roles.extend([
            {"slot_role": "ahead_blocker", "sector": "ahead", "priority_hint": "primary"},
            {"slot_role": "port_escape_constraint", "sector": "port", "priority_hint": "primary"},
            {"slot_role": "starboard_escape_constraint", "sector": "starboard", "priority_hint": "primary"},
        ])
        extras = [
            {"slot_role": "ahead_secondary", "sector": "ahead", "priority_hint": "support"},
            {"slot_role": "astern_context", "sector": "astern", "priority_hint": "context"},
            {"slot_role": "beam_context", "sector": rng.choice(["port", "starboard"]), "priority_hint": "context"},
        ]
        roles.extend(extras[: max(0, n - 3)])

    elif topo == "surrounding_ring":
        roles.extend([
            {"slot_role": "ring_ahead", "sector": "ahead", "priority_hint": "primary"},
            {"slot_role": "ring_port", "sector": "port", "priority_hint": "primary"},
            {"slot_role": "ring_starboard", "sector": "starboard", "priority_hint": "primary"},
            {"slot_role": "ring_astern", "sector": "astern", "priority_hint": "support"},
        ])
        extras = [
            {"slot_role": "ring_port2", "sector": "port", "priority_hint": "support"},
            {"slot_role": "ring_starboard2", "sector": "starboard", "priority_hint": "support"},
        ]
        roles.extend(extras[: max(0, n - 4)])
    else:
        raise ValueError(f"Unknown spatial_topology: {topo}")

    return roles[:n]


def choose_target_status(latent: Dict[str, Any], role: Dict[str, Any], idx: int, rng: random.Random) -> Tuple[str, str, int]:
    mix = str(latent["target_status_mix"])
    supp = str(latent["suppression_mechanism"])
    vis = str(latent["visibility_mode"])

    vessel_class = "power_driven"
    nav_status = "underway_making_way"
    priority = 5

    if mix == "mixed_status":
        if supp in {"hierarchy_override", "mixed"} and idx == 0:
            choice = rng.choice([
                ("ram", "restricted_in_ability_to_manoeuvre", 10),
                ("cbd", "constrained_by_draught", 9),
                ("fishing", "engaged_in_fishing", 9),
            ])
            vessel_class, nav_status, priority = choice
        else:
            choice = rng.choice([
                ("power_driven", "underway_making_way", 5),
                ("sailing", "underway_making_way", 7),
                ("fishing", "engaged_in_fishing", 9),
                ("ram", "restricted_in_ability_to_manoeuvre", 10),
            ])
            vessel_class, nav_status, priority = choice

    if vis == "restricted_visibility":
        priority = max(priority, 6)

    if role["slot_role"].startswith("ahead_blocker"):
        priority = max(priority, 7)
    if role["slot_role"] in {"port_constraint", "starboard_constraint", "port_escape_constraint", "starboard_escape_constraint"}:
        priority = max(priority, 7)

    return vessel_class, nav_status, priority


def base_range_from_urgency_and_density(latent: Dict[str, Any], rng: random.Random) -> float:
    urg = str(latent["urgency_band"])
    density = str(latent["interaction_density"])

    # v2 patch:
    # Increase low-urgency standoff so the realized minimum range stays above the validator threshold,
    # even for side-sector placements whose Euclidean distance is smaller than the nominal base range.
    if urg == "low":
        lo, hi = 1300.0, 2200.0
    elif urg == "medium":
        lo, hi = 450.0, 1100.0
    else:
        lo, hi = 220.0, 650.0

    base = choose_float_from_range(rng, lo, hi)
    factor = {"sparse": 1.18, "medium": 1.0, "dense": 0.90}[density]
    return base * factor


def urgency_distance_thresholds(latent: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    urg = str(latent["urgency_band"])
    if urg == "low":
        return 700.0, None
    if urg == "medium":
        return 240.0, 690.0
    return None, 350.0


def local_coords_for_role(latent: Dict[str, Any], role: Dict[str, Any], base_range: float, rng: random.Random) -> Tuple[float, float]:
    sector = str(role["sector"])
    slot = str(role["slot_role"])
    topo = str(latent["spatial_topology"])

    if topo == "single_side_cluster" and sector in {"port", "starboard"}:
        # Strongly side-dominant cluster: keep most targets in a narrower side band.
        x_ranges = {
            "lead": (0.45, 0.90),
            "beam": (0.15, 0.55),
            "quarter": (-0.20, 0.20),
            "inner": (0.05, 0.45),
            "outer": (0.35, 0.85),
            "tail": (-0.10, 0.25),
        }
        y_ranges = {
            "lead": (0.88, 1.12),
            "beam": (0.72, 0.98),
            "quarter": (0.62, 0.90),
            "inner": (0.78, 1.00),
            "outer": (0.96, 1.22),
            "tail": (0.70, 0.92),
        }
        key = "beam"
        for k in x_ranges:
            if slot.endswith(k):
                key = k
                break
        x_lo, x_hi = x_ranges[key]
        y_lo, y_hi = y_ranges[key]
        x = choose_float_from_range(rng, x_lo * base_range, x_hi * base_range)
        y_mag = choose_float_from_range(rng, y_lo * base_range, y_hi * base_range)
        y = -y_mag if sector == "port" else y_mag
        return x, y

    if sector == "ahead":
        return (
            choose_float_from_range(rng, 0.78 * base_range, 1.15 * base_range),
            choose_float_from_range(rng, -0.18 * base_range, 0.18 * base_range),
        )
    if sector == "astern":
        return (
            choose_float_from_range(rng, -1.10 * base_range, -0.72 * base_range),
            choose_float_from_range(rng, -0.22 * base_range, 0.22 * base_range),
        )
    if sector == "port":
        return (
            choose_float_from_range(rng, -0.18 * base_range, 0.85 * base_range),
            choose_float_from_range(rng, -1.05 * base_range, -0.58 * base_range),
        )
    if sector == "starboard":
        return (
            choose_float_from_range(rng, -0.18 * base_range, 0.85 * base_range),
            choose_float_from_range(rng, 0.58 * base_range, 1.05 * base_range),
        )
    raise ValueError(f"Unknown sector: {sector}")


def heading_for_role(latent: Dict[str, Any], ownship: Dict[str, Any], role: Dict[str, Any], rng: random.Random, area_context: Dict[str, Any]) -> float:
    own_h = float(ownship["heading_deg"])
    slot = str(role["slot_role"])
    sector = str(role["sector"])

    if slot.startswith("ahead_blocker") or slot.startswith("ring_ahead") or slot == "ahead_context" or slot == "ahead_secondary":
        if area_context.get("channel_context") or area_context.get("tss_context"):
            return round(wrap_deg(own_h + rng.choice([175.0, 180.0, 185.0]) + choose_float_from_range(rng, -6.0, 6.0)), 1)
        return round(wrap_deg(own_h + rng.choice([150.0, 180.0, 210.0]) + choose_float_from_range(rng, -10.0, 10.0)), 1)

    if sector == "port":
        return round(wrap_deg(own_h + choose_float_from_range(rng, 65.0, 125.0)), 1)
    if sector == "starboard":
        return round(wrap_deg(own_h - choose_float_from_range(rng, 65.0, 125.0)), 1)
    if sector == "astern":
        return round(wrap_deg(own_h + rng.choice([0.0, 180.0]) + choose_float_from_range(rng, -15.0, 15.0)), 1)
    return round(wrap_deg(own_h + choose_float_from_range(rng, -25.0, 25.0)), 1)


def place_targets(latent: Dict[str, Any], ownship: Dict[str, Any], roles: List[Dict[str, Any]], spec: Dict[str, Any], rng: random.Random, area_context: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    geometry_cfg = spec.get("scene_builder", {}).get("geometry_constraints", {})
    min_sep = float(geometry_cfg.get("min_inter_target_separation_m", 80.0))
    speed_cfg = spec.get("scene_builder", {}).get("targets", {}).get("speed_mps_range", {})

    attempts_total = 0
    slot_failures = 0

    for i, role in enumerate(roles, 1):
        vessel_class, nav_status, priority = choose_target_status(latent, role, i - 1, rng)
        speed_rng = speed_cfg.get("mixed_status" if latent["target_status_mix"] == "mixed_status" else "all_power", [3.0, 8.0])
        local_ok = False
        for _ in range(40):
            attempts_total += 1
            base_range = base_range_from_urgency_and_density(latent, rng)
            x_local, y_local = local_coords_for_role(latent, role, base_range, rng)
            x_w, y_w = rotate_local_to_world(x_local, y_local, float(ownship["heading_deg"]))
            candidate = {
                "id": f"t{i}",
                "vessel_class": vessel_class,
                "nav_status": nav_status,
                "x_m": round(x_w, 2),
                "y_m": round(y_w, 2),
                "heading_deg": heading_for_role(latent, ownship, role, rng, area_context),
                "speed_mps": round(choose_float_from_range(rng, float(speed_rng[0]), float(speed_rng[1])), 2),
                "length_m": round(choose_float_from_range(rng, 35.0, 220.0), 1),
                "intended_role": str(role["slot_role"]),
                "role_sector": str(role["sector"]),
                "local_priority": priority,
            }
            sector_real = rel_bearing_sector(
                float(ownship["x_m"]), float(ownship["y_m"]), float(ownship["heading_deg"]),
                float(candidate["x_m"]), float(candidate["y_m"])
            )
            if sector_real != str(role["sector"]):
                continue

            dist_from_own = math.hypot(float(candidate["x_m"]) - float(ownship["x_m"]), float(candidate["y_m"]) - float(ownship["y_m"]))
            min_dist, max_dist = urgency_distance_thresholds(latent)
            if min_dist is not None and dist_from_own < min_dist:
                continue
            if max_dist is not None and dist_from_own > max_dist:
                continue

            if all(pairwise_distance(candidate, t) >= min_sep for t in targets):
                candidate["relative_bearing_sector"] = sector_real
                candidate["distance_from_ownship_m"] = round(dist_from_own, 2)
                targets.append(candidate)
                local_ok = True
                break
        if not local_ok:
            slot_failures += 1
            # fallback: deterministic placement in exact sector belt
            base_range = base_range_from_urgency_and_density(latent, rng)
            sector = str(role["sector"])
            if sector == "ahead":
                x_local, y_local = 0.92 * base_range, 0.0
            elif sector == "astern":
                x_local, y_local = -0.92 * base_range, 0.0
            elif sector == "port":
                x_local, y_local = 0.28 * base_range, -0.92 * base_range
            else:
                x_local, y_local = 0.28 * base_range, 0.92 * base_range
            x_w, y_w = rotate_local_to_world(x_local, y_local, float(ownship["heading_deg"]))
            targets.append({
                "id": f"t{i}",
                "vessel_class": vessel_class,
                "nav_status": nav_status,
                "x_m": round(x_w, 2),
                "y_m": round(y_w, 2),
                "heading_deg": heading_for_role(latent, ownship, role, rng, area_context),
                "speed_mps": round(choose_float_from_range(rng, float(speed_rng[0]), float(speed_rng[1])), 2),
                "length_m": round(choose_float_from_range(rng, 35.0, 220.0), 1),
                "intended_role": str(role["slot_role"]),
                "role_sector": str(role["sector"]),
                "local_priority": priority,
                "relative_bearing_sector": sector,
            })

    return targets, {
        "slot_count": len(roles),
        "slot_failures": slot_failures,
        "placement_attempts_total": attempts_total,
    }


def build_target_role_graph(latent: Dict[str, Any], targets: List[Dict[str, Any]], area_context: Dict[str, Any]) -> Dict[str, Any]:
    nodes = []
    edges = []
    for t in targets:
        nodes.append({
            "id": t["id"],
            "intended_role": t["intended_role"],
            "sector": t["role_sector"],
            "priority": t["local_priority"],
            "vessel_class": t["vessel_class"],
            "nav_status": t["nav_status"],
        })
    for i, a in enumerate(targets):
        for b in targets[i + 1:]:
            relation = "independent"
            if a["role_sector"] != b["role_sector"]:
                relation = "cross_side_interaction"
            if a["role_sector"] == "ahead" and b["role_sector"] in {"port", "starboard"}:
                relation = "frontal_with_escape_constraint"
            edges.append({
                "from": a["id"],
                "to": b["id"],
                "relation": relation,
            })
    return {
        "topology": latent["spatial_topology"],
        "nodes": nodes,
        "edges": edges,
        "area_source": area_context.get("area_source", "none"),
    }


def build_scene_from_latent(rng: random.Random, latent: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    area_context = build_area_context(rng, latent, spec)
    ownship = build_ownship(rng, latent, spec, area_context)
    roles = build_topology_roles(latent, rng)
    targets, placement_debug = place_targets(latent, ownship, roles, spec, rng, area_context)
    role_graph = build_target_role_graph(latent, targets, area_context)

    scene_spec = {
        "seed": 0,
        "domain": "restricted",
        "visibility": str(latent["visibility_mode"]),
        "snapshot_time_s": 30.0,
        "duration_s": 120.0,
        "area_context": area_context,
        "ownship": ownship,
        "targets": targets,
        "target_role_graph": role_graph,
    }

    scene_debug = {
        "placement": placement_debug,
        "role_slots": roles,
    }
    return scene_spec, scene_debug


# ============================================================
# Lightweight reasoning / labels
# ============================================================

def local_obligation_for_target(latent: Dict[str, Any], scene_spec: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Any]:
    ownship = scene_spec["ownship"]
    area_context = scene_spec["area_context"]
    role = str(target["intended_role"])
    sector = str(target["relative_bearing_sector"])
    priority = int(target.get("local_priority", 5))

    allow: Dict[str, List[str]] = defaultdict(list)
    forbid: Dict[str, List[str]] = defaultdict(list)
    rules: List[str] = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
    ]

    def add_allow(p: str, rule: str) -> None:
        allow[p].append(rule)
        rules.append(rule)

    def add_forbid(p: str, rule: str) -> None:
        forbid[p].append(rule)
        rules.append(rule)

    # Base by sector / topology role
    if sector == "ahead" or role.startswith("ahead_") or role.startswith("ring_ahead"):
        add_allow("REDUCE_SPEED", "COLREG_R08_POSITIVE_ACTION_WHEN_RISK")
        add_forbid("KEEP_COURSE", "COLREG_R08_POSITIVE_ACTION_WHEN_RISK")
        add_allow("TURN_STARBOARD", "COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK")
    elif sector == "port":
        add_allow("TURN_STARBOARD", "COLREG_R08_POSITIVE_ACTION_WHEN_RISK")
        add_forbid("TURN_PORT", "COLREG_R16_EARLY_SUBSTANTIAL_ACTION")
        add_allow("REDUCE_SPEED", "COLREG_R08_POSITIVE_ACTION_WHEN_RISK")
    elif sector == "starboard":
        add_allow("REDUCE_SPEED", "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD")
        add_forbid("TURN_PORT", "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD")
        add_allow("TURN_STARBOARD", "COLREG_R16_EARLY_SUBSTANTIAL_ACTION")
    elif sector == "astern":
        add_allow("KEEP_COURSE", "COLREG_R17_STANDON_MONITORING")

    # Hierarchy from vessel status
    vc = str(target["vessel_class"])
    ns = str(target["nav_status"])
    if vc in {"ram", "fishing"} or ns in {"restricted_in_ability_to_manoeuvre", "engaged_in_fishing", "constrained_by_draught"}:
        priority = max(priority, 10)
        add_allow("KEEP_CLEAR_RESTRICTED", "COLREG_R18_HIERARCHY_PRIORITY")
        add_forbid("CUT_ACROSS_RESTRICTED_VESSEL", "COLREG_R18_HIERARCHY_PRIORITY")
        add_forbid("MAINTAIN_COURSE", "COLREG_R18_HIERARCHY_PRIORITY")

    # Restricted visibility overlay
    if scene_spec["visibility"] == "restricted_visibility":
        priority = max(priority, 6)
        add_allow("REDUCE_SPEED", "COLREG_R19_SAFE_SPEED_RESTRICTED_VIS")
        add_forbid("TURN_PORT", "COLREG_R19D1_AVOID_PORT_TURN_FORWARD_BEAM_NOT_BEING_OVERTAKEN")

    # Area overlays
    if area_context.get("channel_context"):
        priority = max(priority, 8)
        add_allow("KEEP_TO_STARBOARD_WITHIN_CHANNEL", "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
        add_forbid("WIDE_PORT_DEVIATION", "COLREG_R09_NARROW_CHANNEL_GENERAL")
    if area_context.get("tss_context"):
        priority = max(priority, 8)
        add_allow("CROSS_TSS_DECISIVELY", "COLREG_R10_CROSS_TSS_AT_RIGHT_ANGLES")
        add_forbid("ENTER_SEPARATION_ZONE", "COLREG_R10_AVOID_SEPARATION_ZONE")

    return {
        "target_id": target["id"],
        "target_role": role,
        "sector": sector,
        "priority": priority,
        "allow": {k: ensure_unique_rule_list(v) for k, v in allow.items()},
        "forbid": {k: ensure_unique_rule_list(v) for k, v in forbid.items()},
        "rule_ids": ensure_unique_rule_list(rules),
    }


def aggregate_local_obligations(local_obligations: List[Dict[str, Any]]) -> Dict[str, Any]:
    allow_support: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    forbid_support: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ob in local_obligations:
        tid = str(ob["target_id"])
        pri = int(ob["priority"])
        for primitive, rule_ids in ob["allow"].items():
            allow_support[primitive].append({"target_id": tid, "priority": pri, "rule_ids": list(rule_ids)})
        for primitive, rule_ids in ob["forbid"].items():
            forbid_support[primitive].append({"target_id": tid, "priority": pri, "rule_ids": list(rule_ids)})
    return {
        "allow_support": dict(allow_support),
        "forbid_support": dict(forbid_support),
    }


def resolve_global_action_set(latent: Dict[str, Any], scene_spec: Dict[str, Any], local_obligations: List[Dict[str, Any]], aggregated: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    allow_support = aggregated["allow_support"]
    forbid_support = aggregated["forbid_support"]

    # Global overlay actions from suppression mechanism
    global_allow: Dict[str, List[str]] = defaultdict(list)
    global_forbid: Dict[str, List[str]] = defaultdict(list)
    global_priority = 0
    suppression_mechanism = str(latent["suppression_mechanism"])
    area_context = scene_spec["area_context"]

    if suppression_mechanism in {"area_override", "mixed"}:
        if area_context.get("channel_context"):
            global_priority = max(global_priority, 9)
            global_allow["KEEP_TO_STARBOARD_WITHIN_CHANNEL"].append("COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
            global_forbid["WIDE_PORT_DEVIATION"].append("COLREG_R09_NARROW_CHANNEL_GENERAL")
        if area_context.get("tss_context"):
            global_priority = max(global_priority, 9)
            global_allow["CROSS_TSS_DECISIVELY"].append("COLREG_R10_CROSS_TSS_AT_RIGHT_ANGLES")
            global_forbid["ENTER_SEPARATION_ZONE"].append("COLREG_R10_AVOID_SEPARATION_ZONE")
        if scene_spec["visibility"] == "restricted_visibility":
            global_priority = max(global_priority, 9)
            global_allow["REDUCE_SPEED"].append("COLREG_R19_SAFE_SPEED_RESTRICTED_VIS")
            global_forbid["TURN_PORT"].append("COLREG_R19D1_AVOID_PORT_TURN_FORWARD_BEAM_NOT_BEING_OVERTAKEN")

    if suppression_mechanism in {"hierarchy_override", "mixed"}:
        special = [ob for ob in local_obligations if ob["priority"] >= 9]
        if special:
            global_priority = max(global_priority, 10)
            global_allow["KEEP_CLEAR_RESTRICTED"].append("COLREG_R18_HIERARCHY_PRIORITY")
            global_forbid["MAINTAIN_COURSE"].append("COLREG_R18_HIERARCHY_PRIORITY")

    primitives = sorted(set(allow_support.keys()) | set(forbid_support.keys()) | set(global_allow.keys()) | set(global_forbid.keys()))

    maneuver_allowed: Dict[str, List[str]] = {}
    maneuver_forbidden: Dict[str, List[str]] = {}
    suppression_records: List[Dict[str, Any]] = []
    contributing_target_ids: set[str] = set()

    for primitive in primitives:
        allow_entries = list(allow_support.get(primitive, []))
        forbid_entries = list(forbid_support.get(primitive, []))
        allow_max = max([e["priority"] for e in allow_entries], default=-1)
        forbid_max = max([e["priority"] for e in forbid_entries], default=-1)
        area_allow = list(global_allow.get(primitive, []))
        area_forbid = list(global_forbid.get(primitive, []))

        if area_allow or area_forbid:
            if area_allow:
                allow_max = max(allow_max, global_priority)
            if area_forbid:
                forbid_max = max(forbid_max, global_priority)

        if allow_max < 0 and forbid_max < 0:
            continue

        if allow_max > forbid_max:
            rules = []
            for e in allow_entries:
                if e["priority"] == allow_max:
                    rules.extend(e["rule_ids"])
                    contributing_target_ids.add(e["target_id"])
            rules.extend(area_allow)
            maneuver_allowed[primitive] = ensure_unique_rule_list(rules)
            if forbid_entries or area_forbid:
                suppression_records.append({
                    "primitive": primitive,
                    "decision": "allow",
                    "suppressed_side": "forbid",
                    "winner_priority": allow_max,
                    "loser_priority": forbid_max,
                    "source": suppression_mechanism,
                })
        elif forbid_max > allow_max:
            rules = []
            for e in forbid_entries:
                if e["priority"] == forbid_max:
                    rules.extend(e["rule_ids"])
                    contributing_target_ids.add(e["target_id"])
            rules.extend(area_forbid)
            maneuver_forbidden[primitive] = ensure_unique_rule_list(rules)
            if allow_entries or area_allow:
                suppression_records.append({
                    "primitive": primitive,
                    "decision": "forbid",
                    "suppressed_side": "allow",
                    "winner_priority": forbid_max,
                    "loser_priority": allow_max,
                    "source": suppression_mechanism,
                })
        else:
            # Equal-priority conflict: choose forbid for safety, but record it.
            rules = []
            for e in forbid_entries:
                if e["priority"] == forbid_max:
                    rules.extend(e["rule_ids"])
                    contributing_target_ids.add(e["target_id"])
            rules.extend(area_forbid)
            if rules:
                maneuver_forbidden[primitive] = ensure_unique_rule_list(rules)
            suppression_records.append({
                "primitive": primitive,
                "decision": "forbid_on_tie",
                "suppressed_side": "allow",
                "winner_priority": forbid_max,
                "loser_priority": allow_max,
                "source": f"{suppression_mechanism}_tie_break",
            })

    nontrivial = len(suppression_records) >= 1 or len(maneuver_allowed) + len(maneuver_forbidden) >= 5
    resolution_debug = {
        "contributing_target_ids": sorted(contributing_target_ids),
        "suppression_count": len(suppression_records),
        "global_resolution_nontrivial": nontrivial,
    }
    return {
        "maneuver_allowed": maneuver_allowed,
        "maneuver_forbidden": maneuver_forbidden,
    }, suppression_records, resolution_debug


def compute_target_removal_sensitivity(latent: Dict[str, Any], scene_spec: Dict[str, Any], local_obligations: List[Dict[str, Any]]) -> Dict[str, Any]:
    full_agg = aggregate_local_obligations(local_obligations)
    full_actions, full_supp, full_dbg = resolve_global_action_set(latent, scene_spec, local_obligations, full_agg)

    changed_cases: List[Dict[str, Any]] = []
    sensitive_target_ids: List[str] = []

    for ob in local_obligations:
        tid = str(ob["target_id"])
        reduced = [x for x in local_obligations if str(x["target_id"]) != tid]
        agg = aggregate_local_obligations(reduced)
        actions, _, _ = resolve_global_action_set(latent, scene_spec, reduced, agg)
        if actions != full_actions:
            sensitive_target_ids.append(tid)
            changed_cases.append({
                "removed_target_id": tid,
                "before_allowed": sorted(full_actions["maneuver_allowed"].keys()),
                "before_forbidden": sorted(full_actions["maneuver_forbidden"].keys()),
                "after_allowed": sorted(actions["maneuver_allowed"].keys()),
                "after_forbidden": sorted(actions["maneuver_forbidden"].keys()),
            })

    return {
        "sensitive": len(sensitive_target_ids) >= 1,
        "sensitive_target_ids": sensitive_target_ids,
        "changed_cases": changed_cases,
        "baseline": {
            "allowed": sorted(full_actions["maneuver_allowed"].keys()),
            "forbidden": sorted(full_actions["maneuver_forbidden"].keys()),
            "suppression_count": len(full_supp),
            "contributing_target_ids": full_dbg["contributing_target_ids"],
        },
    }


def build_lightweight_labels(latent: Dict[str, Any], scene_spec: Dict[str, Any], local_obligations: List[Dict[str, Any]], actions: Dict[str, Any], suppression_records: List[Dict[str, Any]], resolution_debug: Dict[str, Any], sensitivity: Dict[str, Any]) -> Dict[str, Any]:
    rules: List[str] = []
    target_summaries: List[Dict[str, Any]] = []

    for ob in local_obligations:
        rules.extend(ob["rule_ids"])
        tgt = next(t for t in scene_spec["targets"] if t["id"] == ob["target_id"])
        target_summaries.append({
            "target_id": ob["target_id"],
            "intended_role": tgt["intended_role"],
            "relative_bearing_sector": tgt["relative_bearing_sector"],
            "priority": ob["priority"],
            "triggered_rule_ids": ensure_unique_rule_list(ob["rule_ids"]),
            "local_allow": ob["allow"],
            "local_forbid": ob["forbid"],
            "vessel_class": tgt["vessel_class"],
            "nav_status": tgt["nav_status"],
        })

    for rec in suppression_records:
        rules.append(f"SUPPRESS::{rec['primitive']}::{rec['source']}")

    primitive_ledger_summary = {
        "effective_target_count": len(local_obligations),
        "allow_primitives": sorted(actions["maneuver_allowed"].keys()),
        "forbid_primitives": sorted(actions["maneuver_forbidden"].keys()),
        "suppression_count": len(suppression_records),
        "contributing_target_ids": resolution_debug["contributing_target_ids"],
        "target_removal_sensitive": sensitivity["sensitive"],
    }

    explanation_steps = [
        {
            "stage": "context",
            "text": f"restricted_multi scene with topology={latent['spatial_topology']}, visibility={latent['visibility_mode']}, suppression={latent['suppression_mechanism']}",
        },
        {
            "stage": "local_obligations",
            "text": f"{len(local_obligations)} effective targets generate local obligations from different sectors / statuses.",
        },
        {
            "stage": "suppression",
            "text": f"Global arbitration resolved {len(suppression_records)} suppression/conflict events.",
        },
        {
            "stage": "global_stage",
            "text": f"Final actions depend on targets {resolution_debug['contributing_target_ids']} and are target-removal-sensitive={sensitivity['sensitive']}.",
        },
    ]

    return {
        "triggered_rules": ensure_unique_rule_list(rules),
        "target_summaries": target_summaries,
        "primitive_ledger_summary": primitive_ledger_summary,
        "suppressed_rules": suppression_records,
        "suppression_records": suppression_records,
        "explanation_steps": explanation_steps,
        "maneuver_allowed": actions["maneuver_allowed"],
        "maneuver_forbidden": actions["maneuver_forbidden"],
        "lights_required": [],
        "sounds_required": [],
    }


def run_lightweight_global_resolution(latent: Dict[str, Any], scene_spec: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    local_obligations = [
        local_obligation_for_target(latent, scene_spec, t)
        for t in scene_spec["targets"]
    ]
    aggregated = aggregate_local_obligations(local_obligations)
    actions, suppression_records, resolution_debug = resolve_global_action_set(latent, scene_spec, local_obligations, aggregated)
    sensitivity = compute_target_removal_sensitivity(latent, scene_spec, local_obligations)
    labels = build_lightweight_labels(latent, scene_spec, local_obligations, actions, suppression_records, resolution_debug, sensitivity)
    debug = {
        "local_obligations": local_obligations,
        "resolution": resolution_debug,
        "target_removal_sensitivity": sensitivity,
    }
    return labels, debug


def quick_generation_contract_check(latent: Dict[str, Any], scene_spec: Dict[str, Any], labels: Dict[str, Any], reasoning_debug: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    targets = scene_spec["targets"]
    if len(targets) < 3:
        errors.append("fewer_than_3_effective_targets")

    topo = str(latent["spatial_topology"])
    sectors = [str(t["relative_bearing_sector"]) for t in targets]
    if topo == "bilateral_constraint":
        if "port" not in sectors or "starboard" not in sectors:
            errors.append("bilateral_constraint_not_realized")
    elif topo == "frontal_blocking":
        if "ahead" not in sectors:
            errors.append("frontal_blocking_missing_ahead_target")
        if not any(s in {"port", "starboard"} for s in sectors):
            errors.append("frontal_blocking_missing_side_target")
    elif topo == "surrounding_ring":
        for need in ["ahead", "port", "starboard", "astern"]:
            if need not in sectors:
                errors.append(f"surrounding_ring_missing_{need}")

    supp = str(latent["suppression_mechanism"])
    suppression_count = int(reasoning_debug["resolution"].get("suppression_count", 0))
    if suppression_count < 1:
        errors.append("no_suppression_and_no_conflict")

    if not bool(reasoning_debug["resolution"].get("global_resolution_nontrivial", False)):
        errors.append("no_global_resolution")

    contributing = list(reasoning_debug["resolution"].get("contributing_target_ids", []))
    if len(contributing) < 2:
        errors.append("final_labels_explainable_by_single_target")

    sensitivity = bool(reasoning_debug["target_removal_sensitivity"].get("sensitive", False))
    if not sensitivity:
        errors.append("no_change_under_target_removal")

    if supp == "hierarchy_override":
        if not any(t["vessel_class"] in {"ram", "cbd", "fishing"} or t["nav_status"] in {"restricted_in_ability_to_manoeuvre", "engaged_in_fishing", "constrained_by_draught"} for t in targets):
            errors.append("hierarchy_override_without_priority_target")
    if supp == "area_override":
        ac = scene_spec["area_context"]
        if not (ac.get("channel_context") or ac.get("tss_context") or scene_spec["visibility"] == "restricted_visibility"):
            errors.append("area_override_without_area_source")
    if supp == "mixed":
        ac = scene_spec["area_context"]
        has_area = bool(ac.get("channel_context") or ac.get("tss_context") or scene_spec["visibility"] == "restricted_visibility")
        has_hierarchy = any(t["vessel_class"] in {"ram", "cbd", "fishing"} or t["nav_status"] in {"restricted_in_ability_to_manoeuvre", "engaged_in_fishing", "constrained_by_draught"} for t in targets)
        if not (has_area and has_hierarchy or suppression_count >= 2):
            errors.append("mixed_suppression_not_realized")

    if len(labels.get("triggered_rules", [])) < int(4):
        errors.append("too_few_triggered_rules")

    if len(set(sectors)) < 2:
        warnings.append("low_sector_diversity")

    return errors, warnings


# ============================================================
# Row assembly / report
# ============================================================

def build_inputs(scene_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "topdown_image": None,
        "radar_image": None,
        "scene_text": None,
        "scene_spec": scene_spec,
    }


def assemble_raw_row(sample_id: str, candidate_seed: int, latent: Dict[str, Any], scene_spec: Dict[str, Any], labels: Dict[str, Any], debug: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "pattern": FAMILY,
        "source_mode": SOURCE_MODE,
        "generator_version": GENERATOR_VERSION,
        "candidate_seed": candidate_seed,
        "family_latent": latent,
        "scene_spec": scene_spec,
        "inputs": build_inputs(scene_spec),
        "labels": labels,
        "debug": debug,
    }


def build_generation_report(spec: Dict[str, Any], rows: List[Dict[str, Any]], schema_error_count: int, schema_valid_count: int, compatibility_resamples: int, geometry_attempt_hist: List[int], contract_fail_hist: Counter, warnings_hist: Counter) -> Dict[str, Any]:
    latent_hist: Dict[str, Dict[str, int]] = {}
    for key in [
        "num_effective_targets",
        "spatial_topology",
        "suppression_mechanism",
        "urgency_band",
        "visibility_mode",
        "target_status_mix",
        "interaction_density",
    ]:
        c = Counter(str(r["family_latent"].get(key)) for r in rows)
        latent_hist[key] = dict(sorted(c.items(), key=lambda x: str(x[0])))

    unique_combo_count = len({
        json.dumps(r["family_latent"], sort_keys=True, ensure_ascii=False) for r in rows
    })
    target_count_hist = Counter(len(r["scene_spec"]["targets"]) for r in rows)
    suppression_count_hist = Counter(len(r["labels"].get("suppression_records", [])) for r in rows)
    removal_sensitive_count = sum(
        1 for r in rows if bool(r.get("debug", {}).get("reasoning", {}).get("target_removal_sensitivity", {}).get("sensitive", False))
    )

    return {
        "family": FAMILY,
        "generator_version": GENERATOR_VERSION,
        "generator_revision": GENERATOR_REVISION,
        "schema_version": spec.get("schema_version", "unknown"),
        "num_candidates_written": len(rows),
        "schema_validation_enabled": True,
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_error_count": schema_error_count,
        "schema_valid_count": schema_valid_count,
        "compatibility_filter_summary": {
            "total_resamples": compatibility_resamples,
        },
        "geometry_repair_summary": {
            "avg_attempts": round(sum(geometry_attempt_hist) / max(1, len(geometry_attempt_hist)), 3),
            "max_attempts_single_sample": max(geometry_attempt_hist) if geometry_attempt_hist else 0,
        },
        "reasoning_summary": {
            "target_removal_sensitive_count": removal_sensitive_count,
            "suppression_count_histogram": dict(sorted((str(k), v) for k, v in suppression_count_hist.items())),
        },
        "latent_histograms": latent_hist,
        "target_count_histogram": dict(sorted((str(k), v) for k, v in target_count_hist.items())),
        "unique_family_combo_count": unique_combo_count,
        "remaining_contract_error_histogram": dict(sorted(contract_fail_hist.items())),
        "warning_histogram": dict(sorted(warnings_hist.items())),
    }


# ============================================================
# Main generation loop
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate native v3 raw candidates (restricted_multi).")
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default=FAMILY, choices=[FAMILY])
    ap.add_argument("--num-candidates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=400000)
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
        print(f"ERROR: output exists, use --overwrite: {out_path}")
        return 2

    spec = read_yaml(spec_path)
    schema = read_json(schema_path)
    rng = random.Random(int(args.seed))

    rows: List[Dict[str, Any]] = []
    schema_error_count = 0
    schema_valid_count = 0
    compatibility_resamples = 0
    geometry_attempt_hist: List[int] = []
    contract_fail_hist: Counter = Counter()
    warnings_hist: Counter = Counter()

    max_sample_attempts = 120
    for i in range(1, int(args.num_candidates) + 1):
        sample_seed = int(args.seed) + i
        sample_rng = random.Random(sample_seed)
        built = False
        last_errors: List[str] = []

        for _ in range(max_sample_attempts):
            latent = sample_family_latent(sample_rng, spec)
            ok, reason = compatible_restricted_multi_latent(latent, spec)
            if not ok:
                compatibility_resamples += 1
                if reason is not None:
                    contract_fail_hist[reason] += 1
                continue

            scene_spec, scene_debug = build_scene_from_latent(sample_rng, latent, spec)
            scene_spec["seed"] = sample_seed
            labels, reasoning_debug = run_lightweight_global_resolution(latent, scene_spec)
            errors, warnings = quick_generation_contract_check(latent, scene_spec, labels, reasoning_debug)

            if errors:
                for e in errors:
                    contract_fail_hist[e] += 1
                for w in warnings:
                    warnings_hist[w] += 1
                last_errors = errors
                continue

            for w in warnings:
                warnings_hist[w] += 1

            row = assemble_raw_row(
                sample_id=stable_sample_id(i),
                candidate_seed=sample_seed,
                latent=latent,
                scene_spec=scene_spec,
                labels=labels,
                debug={
                    "generator_version": GENERATOR_VERSION,
                    "generator_revision": GENERATOR_REVISION,
                    "scene_builder": scene_debug,
                    "reasoning": reasoning_debug,
                },
            )

            if args.validate_schema:
                err = maybe_validate_schema(row, schema)
                if err is None:
                    schema_valid_count += 1
                else:
                    schema_error_count += 1
                    contract_fail_hist["schema_validation_failed"] += 1
                    last_errors = [err]
                    continue

            geometry_attempt_hist.append(int(scene_debug["placement"].get("placement_attempts_total", 0)))
            rows.append(row)
            built = True
            break

        if not built:
            raise RuntimeError(
                f"Failed to build restricted_multi sample {i} after {max_sample_attempts} attempts. Last errors={last_errors}"
            )

    write_jsonl(out_path, rows)

    report = build_generation_report(
        spec=spec,
        rows=rows,
        schema_error_count=schema_error_count,
        schema_valid_count=schema_valid_count,
        compatibility_resamples=compatibility_resamples,
        geometry_attempt_hist=geometry_attempt_hist,
        contract_fail_hist=contract_fail_hist,
        warnings_hist=warnings_hist,
    )
    if args.report_json:
        write_json(Path(args.report_json), report)

    print("=" * 100)
    print(f"generate_native_candidates_restricted_multi.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Candidates written        : {len(rows)}")
    print(f"Output raw file          : {out_path}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {schema_error_count}")
    print(f"Compatibility resamples  : {compatibility_resamples}")
    print(f"Unique family combos     : {report['unique_family_combo_count']}")
    print(f"Target-removal sensitive : {report['reasoning_summary']['target_removal_sensitive_count']}/{len(rows)}")
    if args.report_json:
        print(f"Report written           : {args.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
