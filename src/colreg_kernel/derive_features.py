from __future__ import annotations

"""Derive operational features from continuous evidence values.

This module provides a *benchmark operationalization* of ambiguous COLREG concepts.
It is intentionally parameterized via rules/operational_params.yaml so that:
  - thresholds can be tuned without touching the rule library
  - sensitivity analysis can be performed cleanly

Typical usage:
  params = load_operational_params()
  derived = derive_operational_flags(
      cpa_m=..., tcpa_s=..., own_length_m=..., area_type=..., tss_crossing_angle_deg=...,
      delta_cog_deg=..., speed_drop_ratio=...
  )

The functions here are deterministic and side-effect free.
"""

from dataclasses import dataclass
from math import fmod
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .features import AreaType, SailingTack, SailingTackRelation


NM_M = 1852.0


def _nm_to_m(x_nm: float) -> float:
    return x_nm * NM_M


def load_operational_params(path: str | Path | None = None) -> Dict[str, Any]:
    """Load operationalization thresholds.

    If path is None, load from package-local rules/operational_params.yaml relative to repo layout.
    """
    if path is None:
        # repo layout: <project>/rules/operational_params.yaml
        path = Path(__file__).resolve().parents[2] / "rules" / "operational_params.yaml"
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("operational_params.yaml must be a mapping")
    return data


def dcpa_threshold_m(params: Dict[str, Any], domain: str, own_length_m: Optional[float]) -> float:
    """Compute DCPA threshold in meters using nm baseline and length scaling."""
    own_length_m = float(own_length_m or 0.0)
    cfg = params[domain]
    base = _nm_to_m(float(cfg.get("dcpa_imp_nm", 0.0)))
    lf = float(cfg.get("dcpa_imp_len_factor", 0.0))
    return max(base, lf * own_length_m)


def derive_risk_of_collision(
    *,
    cpa_m: Optional[float],
    tcpa_s: Optional[float],
    own_length_m: Optional[float],
    params: Dict[str, Any],
) -> Optional[bool]:
    if cpa_m is None or tcpa_s is None:
        return None
    if tcpa_s < 0:
        return False
    cfg = params["risk"]
    tcpa_risk = float(cfg["tcpa_risk_s"])
    base = _nm_to_m(float(cfg["dcpa_risk_nm"]))
    lf = float(cfg["dcpa_risk_len_factor"])
    th = max(base, lf * float(own_length_m or 0.0))
    return (0.0 <= tcpa_s <= tcpa_risk) and (cpa_m <= th)


def derive_collision_imminent(
    *,
    cpa_m: Optional[float],
    tcpa_s: Optional[float],
    own_length_m: Optional[float],
    params: Dict[str, Any],
) -> Optional[bool]:
    if cpa_m is None or tcpa_s is None:
        return None
    if tcpa_s < 0:
        return False
    cfg = params["imminent"]
    tcpa_thr = float(cfg["tcpa_imminent_s"])
    base = _nm_to_m(float(cfg["dcpa_imminent_nm"]))
    lf = float(cfg["dcpa_imminent_len_factor"])
    th = max(base, lf * float(own_length_m or 0.0))
    return (0.0 <= tcpa_s <= tcpa_thr) and (cpa_m <= th)


def derive_impedes(
    *,
    cpa_m: Optional[float],
    tcpa_s: Optional[float],
    own_length_m: Optional[float],
    area_type: Optional[AreaType],
    params: Dict[str, Any],
) -> Optional[bool]:
    """Operationalize 'not to impede' based on CPA/TCPA thresholds.

    domain-specific thresholds are used (open / narrow_channel / tss).
    """
    if cpa_m is None or tcpa_s is None:
        return None
    if tcpa_s < 0:
        return False

    t_imp = float(params["impede"]["t_imp_s"])
    if area_type == AreaType.NARROW_CHANNEL:
        dom = "narrow_channel"
    elif area_type == AreaType.TSS:
        dom = "tss"
    else:
        dom = "open"

    cfg = params["impede"][dom]
    base = _nm_to_m(float(cfg["dcpa_imp_nm"]))
    lf = float(cfg["dcpa_imp_len_factor"])
    th = max(base, lf * float(own_length_m or 0.0))
    return (0.0 <= tcpa_s <= t_imp) and (cpa_m <= th)


def derive_tss_crossing_ok(
    *,
    tss_crossing_angle_deg: Optional[float],
    params: Dict[str, Any],
) -> Optional[bool]:
    if tss_crossing_angle_deg is None:
        return None
    right = float(params["tss_crossing"]["right_angle_deg"])
    tol = float(params["tss_crossing"]["right_angle_tol_deg"])
    return abs(tss_crossing_angle_deg - right) <= tol


def derive_action_substantial(
    *,
    delta_cog_deg: Optional[float] = None,
    speed_drop_ratio: Optional[float] = None,
    params: Dict[str, Any],
) -> Optional[bool]:
    if delta_cog_deg is None and speed_drop_ratio is None:
        return None
    cfg = params["early_substantial"]
    dpsi = float(cfg["substantial_turn_deg"])
    r = float(cfg["substantial_speed_drop_ratio"])
    ok_turn = (delta_cog_deg is not None) and (abs(float(delta_cog_deg)) >= dpsi)
    ok_slow = (speed_drop_ratio is not None) and (float(speed_drop_ratio) >= r)
    return ok_turn or ok_slow


def _wrap360(deg: float) -> float:
    # Python's fmod preserves sign; do a stable wrap
    x = deg % 360.0
    return x + 360.0 if x < 0 else x


def derive_sailing_tack(*, wind_dir_deg: float, vessel_heading_deg: float) -> SailingTack:
    """Simplified tack inference from true wind direction and vessel heading.

    Convention (benchmark simplification):
      - Compute relative wind angle = wrap(wind_dir - heading) in [0,360)
      - If relative wind comes from starboard side (0..180): tack=STARBOARD
      - If from port side (180..360): tack=PORT

    This is sufficient to generate stable Rule 12 features for curriculum benchmarks.
    """
    rel = _wrap360(float(wind_dir_deg) - float(vessel_heading_deg))
    if rel == 0.0 or rel == 180.0:
        return SailingTack.UNKNOWN
    return SailingTack.STARBOARD if 0.0 < rel < 180.0 else SailingTack.PORT


def derive_sailing_tack_relation(
    *, own_tack: SailingTack, target_tack: SailingTack
) -> SailingTackRelation:
    if own_tack == SailingTack.UNKNOWN or target_tack == SailingTack.UNKNOWN:
        return SailingTackRelation.OTHER_UNKNOWN
    return SailingTackRelation.SAME if own_tack == target_tack else SailingTackRelation.DIFFERENT


def derive_is_windward(
    *,
    wind_dir_deg: float,
    own_heading_deg: float,
    target_heading_deg: float,
) -> bool:
    """Simplified windward test for same-tack sailing vessels.

    Operationalization: the vessel with smaller absolute angle-to-wind is windward.
    """
    def angle_to_wind(h: float) -> float:
        rel = _wrap360(float(wind_dir_deg) - float(h))
        # map to [0,180]
        if rel > 180.0:
            rel = 360.0 - rel
        return rel

    return angle_to_wind(own_heading_deg) < angle_to_wind(target_heading_deg)


@dataclass(frozen=True)
class DerivedFlags:
    risk_of_collision: Optional[bool] = None
    collision_imminent: Optional[bool] = None
    impedes: Optional[bool] = None
    tss_crossing_ok: Optional[bool] = None
    action_substantial: Optional[bool] = None


def derive_operational_flags(
    *,
    cpa_m: Optional[float],
    tcpa_s: Optional[float],
    own_length_m: Optional[float],
    area_type: Optional[AreaType],
    tss_crossing_angle_deg: Optional[float],
    delta_cog_deg: Optional[float] = None,
    speed_drop_ratio: Optional[float] = None,
    params: Dict[str, Any] | None = None,
) -> DerivedFlags:
    params = params or load_operational_params()
    r = derive_risk_of_collision(cpa_m=cpa_m, tcpa_s=tcpa_s, own_length_m=own_length_m, params=params)
    im = derive_collision_imminent(cpa_m=cpa_m, tcpa_s=tcpa_s, own_length_m=own_length_m, params=params)
    imp = derive_impedes(cpa_m=cpa_m, tcpa_s=tcpa_s, own_length_m=own_length_m, area_type=area_type, params=params)
    ok = derive_tss_crossing_ok(tss_crossing_angle_deg=tss_crossing_angle_deg, params=params)
    sub = derive_action_substantial(delta_cog_deg=delta_cog_deg, speed_drop_ratio=speed_drop_ratio, params=params)
    return DerivedFlags(risk_of_collision=r, collision_imminent=im, impedes=imp, tss_crossing_ok=ok, action_substantial=sub)
