"""Deterministic geometry derived only from observable vessel states."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class VesselState:
    vessel_id: str
    vessel_class: str
    nav_status: str
    length_m: float
    x_m: float
    y_m: float
    heading_deg: float
    speed_mps: float


@dataclass(frozen=True)
class PairwiseGeometry:
    distance_m: float
    relative_bearing_deg: float
    relative_bearing_sector: str
    is_target_on_starboard: bool
    heading_difference_deg: float
    closing: bool
    cpa_m: float
    tcpa_s: float | None
    encounter_type: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def vessel_state(value: Mapping[str, Any], *, default_length_m: float = 80.0) -> VesselState:
    identifier = value.get("target_id") or value.get("id") or "unknown"
    return VesselState(
        vessel_id=str(identifier),
        vessel_class=str(value["vessel_class"]),
        nav_status=str(value.get("nav_status") or "underway_making_way"),
        length_m=float(value.get("length_m") or default_length_m),
        x_m=float(value["x_m"]),
        y_m=float(value["y_m"]),
        heading_deg=float(value["heading_deg"]) % 360.0,
        speed_mps=max(0.0, float(value["speed_mps"])),
    )


def heading_vector(heading_deg: float, speed_mps: float) -> tuple[float, float]:
    radians = math.radians(heading_deg)
    return math.sin(radians) * speed_mps, math.cos(radians) * speed_mps


def angular_difference(left_deg: float, right_deg: float) -> float:
    return abs((left_deg - right_deg + 180.0) % 360.0 - 180.0)


def signed_relative_bearing(own: VesselState, target: VesselState) -> float:
    dx = target.x_m - own.x_m
    dy = target.y_m - own.y_m
    absolute = math.degrees(math.atan2(dx, dy)) % 360.0
    return (absolute - own.heading_deg) % 360.0


def bearing_sector(bearing_deg: float) -> str:
    value = bearing_deg % 360.0
    if value < 22.5 or value >= 337.5:
        return "ahead"
    if value < 67.5:
        return "starboard_bow"
    if value < 112.5:
        return "starboard_beam"
    if value < 157.5:
        return "starboard_quarter"
    if value < 202.5:
        return "astern"
    if value < 247.5:
        return "port_quarter"
    if value < 292.5:
        return "port_beam"
    return "port_bow"


def pairwise_geometry(own: VesselState, target: VesselState) -> PairwiseGeometry:
    rx = target.x_m - own.x_m
    ry = target.y_m - own.y_m
    distance = math.hypot(rx, ry)
    ovx, ovy = heading_vector(own.heading_deg, own.speed_mps)
    tvx, tvy = heading_vector(target.heading_deg, target.speed_mps)
    rvx, rvy = tvx - ovx, tvy - ovy
    speed_sq = rvx * rvx + rvy * rvy
    raw_tcpa = -(rx * rvx + ry * rvy) / speed_sq if speed_sq > 1e-12 else None
    closing = bool(raw_tcpa is not None and raw_tcpa > 0.0)
    tcpa = max(0.0, raw_tcpa) if raw_tcpa is not None else None
    at = tcpa or 0.0
    cpa = math.hypot(rx + rvx * at, ry + rvy * at)
    bearing = signed_relative_bearing(own, target)
    sector = bearing_sector(bearing)
    heading_delta = angular_difference(own.heading_deg, target.heading_deg)

    if closing and heading_delta >= 150.0 and sector in {"ahead", "starboard_bow", "port_bow"}:
        encounter = "head_on"
    elif (
        closing
        and heading_delta <= 25.0
        and sector in {"ahead", "starboard_bow", "port_bow"}
        and own.speed_mps > target.speed_mps
    ):
        encounter = "overtaking"
    elif closing and sector not in {"astern", "starboard_quarter", "port_quarter"}:
        encounter = "crossing"
    else:
        encounter = "other"
    return PairwiseGeometry(
        distance_m=round(distance, 6),
        relative_bearing_deg=round(bearing, 6),
        relative_bearing_sector=sector,
        is_target_on_starboard=0.0 < bearing < 180.0,
        heading_difference_deg=round(heading_delta, 6),
        closing=closing,
        cpa_m=round(cpa, 6),
        tcpa_s=round(tcpa, 6) if tcpa is not None else None,
        encounter_type=encounter,
    )


def crossing_angle_deg(heading_deg: float, reference_heading_deg: float) -> float:
    difference = angular_difference(heading_deg, reference_heading_deg)
    return min(difference, 180.0 - difference)
