from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ValidationIssue:
    type: str
    who: Optional[str] = None
    message: Optional[str] = None
    primitive: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _norm_str(x: Any) -> Optional[str]:
    """
    Normalize scalar / Enum-like values to lowercase canonical strings.
    """
    if x is None:
        return None

    # Real Enum -> use .value
    if isinstance(x, Enum):
        x = x.value

    # Some pydantic / enum-like wrappers may expose .value
    elif hasattr(x, "value") and not isinstance(x, (str, int, float, bool)):
        try:
            x = x.value
        except Exception:
            pass

    s = str(x).strip().lower()
    return s if s else None


def _is_fishing_mode_set(mode: Any) -> bool:
    s = _norm_str(mode)
    return s not in (None, "none", "null", "na", "n/a", "unknown")


def _primitive_set(x: Any) -> set[str]:
    """
    Normalize signals container to a set of primitive names.
    Supports:
      - dict: keys are primitives
      - list/tuple/set: elements are primitives
      - None: empty
      - other: stringified singleton
    """
    if x is None:
        return set()
    if isinstance(x, dict):
        return {str(k) for k in x.keys()}
    if isinstance(x, (list, tuple, set)):
        return {str(v) for v in x}
    return {str(x)}


def validate_feature_consistency(features) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    d = features.model_dump() if hasattr(features, "model_dump") else dict(features)

    visibility = _norm_str(d.get("visibility"))
    in_sight = d.get("in_sight", None)

    own = d.get("ownship", {}) or {}
    tgt = d.get("target", {}) or {}

    # 1) visibility vs in_sight
    if visibility == "restricted" and in_sight is True:
        issues.append(
            ValidationIssue(
                type="VISIBILITY_INSIGHT_INCONSISTENT",
                who="scene",
                message="visibility=restricted usually should not have in_sight=True",
            )
        )

    # 2) fishing_mode consistency
    for who, vessel in [("ownship", own), ("target", tgt)]:
        vclass = _norm_str(vessel.get("vessel_class"))
        fmode = vessel.get("fishing_mode", None)
        if _is_fishing_mode_set(fmode) and vclass != "fishing":
            issues.append(
                ValidationIssue(
                    type="FISHING_MODE_CLASS_MISMATCH",
                    who=who,
                    message=f"fishing_mode={_norm_str(fmode)} but vessel_class={vclass}",
                )
            )

    # 3) towing length consistency
    for who, vessel in [("ownship", own), ("target", tgt)]:
        is_towing = bool(vessel.get("is_towing", False))
        tow_length = vessel.get("tow_length_m", None)
        if is_towing and tow_length is None:
            issues.append(
                ValidationIssue(
                    type="TOW_LENGTH_MISSING",
                    who=who,
                    message="is_towing=True but tow_length_m is missing",
                )
            )
        if (not is_towing) and tow_length not in (None, 0, 0.0):
            issues.append(
                ValidationIssue(
                    type="TOW_LENGTH_WITHOUT_TOWING",
                    who=who,
                    message=f"is_towing=False but tow_length_m={tow_length}",
                )
            )

    # 4) nav status basic check
    valid_nav = {"underway_making_way", "underway_not_making_way", "at_anchor", "aground"}
    for who, vessel in [("ownship", own), ("target", tgt)]:
        ns = _norm_str(vessel.get("nav_status"))
        if ns is not None and ns not in valid_nav:
            issues.append(
                ValidationIssue(
                    type="NAV_STATUS_INVALID",
                    who=who,
                    message=f"unknown nav_status={ns}",
                )
            )

    return issues


def validate_signals_outputs(
    labels=None,
    features=None,
    *,
    lights_required=None,
    lights_forbidden=None,
    sounds_required=None,
    sounds_forbidden=None,
) -> List[ValidationIssue]:
    """
    Backward-compatible signals legality checks.
    """
    issues: List[ValidationIssue] = []

    if labels is not None:
        lab = labels.model_dump() if hasattr(labels, "model_dump") else dict(labels)
        lr = _primitive_set(lab.get("lights_required"))
        lf = _primitive_set(lab.get("lights_forbidden"))
        sr = _primitive_set(lab.get("sounds_required"))
        sf = _primitive_set(lab.get("sounds_forbidden"))
    else:
        lr = _primitive_set(lights_required)
        lf = _primitive_set(lights_forbidden)
        sr = _primitive_set(sounds_required)
        sf = _primitive_set(sounds_forbidden)

    for k in lr & lf:
        issues.append(
            ValidationIssue(
                type="LIGHT_REQUIRED_FORBIDDEN_CONFLICT",
                primitive=k,
            )
        )
    for k in sr & sf:
        issues.append(
            ValidationIssue(
                type="SOUND_REQUIRED_FORBIDDEN_CONFLICT",
                primitive=k,
            )
        )

    return issues