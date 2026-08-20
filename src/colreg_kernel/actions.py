from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ManeuverPrimitive(str, Enum):
    # Core high-level maneuver actions (map to controllers later)
    MAINTAIN_COURSE_SPEED = "MAINTAIN_COURSE_SPEED"
    TURN_STARBOARD = "TURN_STARBOARD"
    TURN_PORT = "TURN_PORT"
    REDUCE_SPEED = "REDUCE_SPEED"

    # Compliance / intent primitives (used for constraints & labeling; map later)
    KEEP_TO_STARBOARD_SIDE = "KEEP_TO_STARBOARD_SIDE"
    CROSS_TSS_RIGHT_ANGLE = "CROSS_TSS_RIGHT_ANGLE"
    ENTER_SEPARATION_ZONE = "ENTER_SEPARATION_ZONE"
    PROCEED_OPPOSITE_FLOW = "PROCEED_OPPOSITE_FLOW"
    IMPEDE = "IMPEDE"
    OVERTAKE_IN_CHANNEL = "OVERTAKE_IN_CHANNEL"
    ANCHOR = "ANCHOR"

    # General COLREGs behaviour primitives (Rules 5–8)
    KEEP_LOOKOUT = "KEEP_LOOKOUT"
    PROCEED_SAFE_SPEED = "PROCEED_SAFE_SPEED"
    ASSESS_RISK = "ASSESS_RISK"
    POSITIVE_ACTION = "POSITIVE_ACTION"


class LightShapePrimitive(str, Enum):
    # --- Navigation / special lights (Rules 20–31) ---
    LIGHT_MASTHEAD = "LIGHT_MASTHEAD"
    LIGHT_SIDELIGHTS = "LIGHT_SIDELIGHTS"
    LIGHT_STERN = "LIGHT_STERN"
    LIGHT_TOWING = "LIGHT_TOWING"

    # All-round lights and common vertical combinations (encoded as primitives)
    LIGHT_ALL_ROUND_WHITE = "LIGHT_ALL_ROUND_WHITE"
    LIGHT_ALL_ROUND_RED = "LIGHT_ALL_ROUND_RED"
    LIGHT_ALL_ROUND_GREEN = "LIGHT_ALL_ROUND_GREEN"
    LIGHT_ALL_ROUND_YELLOW = "LIGHT_ALL_ROUND_YELLOW"

    LIGHT_RED_OVER_WHITE = "LIGHT_RED_OVER_WHITE"           # fishing (other than trawling)
    LIGHT_GREEN_OVER_WHITE = "LIGHT_GREEN_OVER_WHITE"       # trawling
    LIGHT_WHITE_OVER_RED = "LIGHT_WHITE_OVER_RED"           # pilot
    LIGHT_TWO_ALL_ROUND_RED = "LIGHT_TWO_ALL_ROUND_RED"     # NUC (simplified)
    LIGHT_THREE_ALL_ROUND_RED = "LIGHT_THREE_ALL_ROUND_RED" # aground (simplified)
    LIGHT_RED_WHITE_RED = "LIGHT_RED_WHITE_RED"             # RAM (simplified)

    # --- Day shapes (Rules 20–31) ---
    SHAPE_BALL = "SHAPE_BALL"
    SHAPE_TWO_BALLS = "SHAPE_TWO_BALLS"                     # aground (simplified)
    SHAPE_DIAMOND = "SHAPE_DIAMOND"
    SHAPE_CYLINDER = "SHAPE_CYLINDER"                       # constrained by draught
    SHAPE_CONE_UP = "SHAPE_CONE_UP"
    SHAPE_CONE_DOWN = "SHAPE_CONE_DOWN"
    SHAPE_BALL_DIAMOND_BALL = "SHAPE_BALL_DIAMOND_BALL"     # NUC
    SHAPE_TWO_CONES_APEX = "SHAPE_TWO_CONES_APEX"           # fishing

class SoundPrimitive(str, Enum):
    # --- Sound signals (Rules 32–37) ---
    SOUND_1_SHORT = "SOUND_1_SHORT"
    SOUND_2_SHORT = "SOUND_2_SHORT"
    SOUND_3_SHORT = "SOUND_3_SHORT"
    SOUND_1_PROLONGED = "SOUND_1_PROLONGED"
    SOUND_PROLONGED_EVERY_2MIN = "SOUND_PROLONGED_EVERY_2MIN"
    SOUND_PROLONGED_PLUS_2_SHORT = "SOUND_PROLONGED_PLUS_2_SHORT"
    SOUND_BELL_RAPID = "SOUND_BELL_RAPID"
    SOUND_BELL_AND_GONG = "SOUND_BELL_AND_GONG"


class ManeuverConstraint(BaseModel):
    primitive: ManeuverPrimitive
    params: Dict[str, Any] = Field(default_factory=dict)


class LightShapeConstraint(BaseModel):
    primitive: LightShapePrimitive
    params: Dict[str, Any] = Field(default_factory=dict)


class SoundConstraint(BaseModel):
    primitive: SoundPrimitive
    params: Dict[str, Any] = Field(default_factory=dict)


class ManeuverActionSet(BaseModel):
    allowed: List[ManeuverConstraint] = Field(default_factory=list)
    forbidden: List[ManeuverConstraint] = Field(default_factory=list)


class SignalsActionSet(BaseModel):
    # Signals head is separate from maneuver head:
    # - lights/shapes required/forbidden
    # - sound signals required/forbidden
    lights_shapes_required: List[LightShapeConstraint] = Field(default_factory=list)
    lights_shapes_forbidden: List[LightShapeConstraint] = Field(default_factory=list)
    sounds_required: List[SoundConstraint] = Field(default_factory=list)
    sounds_forbidden: List[SoundConstraint] = Field(default_factory=list)


class ActionsSpec(BaseModel):
    maneuver: ManeuverActionSet = Field(default_factory=ManeuverActionSet)
    signals: SignalsActionSet = Field(default_factory=SignalsActionSet)


def _primitive_set_from_constraints(constraints: List[BaseModel]) -> set[str]:
    # Helper for internal checks; returns primitive string values.
    out: set[str] = set()
    for c in constraints:
        prim = getattr(c, "primitive", None)
        if prim is not None:
            out.add(str(prim.value))
    return out


def validate_no_cross_head(actions: ActionsSpec) -> None:
    # Optional sanity: ensure signals are not mixed into maneuver.
    man = _primitive_set_from_constraints(actions.maneuver.allowed + actions.maneuver.forbidden)
    sig = _primitive_set_from_constraints(
        actions.signals.lights_shapes_required
        + actions.signals.lights_shapes_forbidden
        + actions.signals.sounds_required
        + actions.signals.sounds_forbidden
    )
    # These sets are strings with prefixes; overlap should be empty by construction.
    if man & sig:
        raise ValueError(f"Signals/maneuver primitives overlapped: {sorted(man & sig)}")
