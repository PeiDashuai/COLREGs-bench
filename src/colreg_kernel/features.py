from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Visibility(str, Enum):
    CLEAR = "clear"
    RESTRICTED = "restricted"


class AreaType(str, Enum):
    OPEN = "open"
    NARROW_CHANNEL = "narrow_channel"
    TSS = "tss"


class EncounterType(str, Enum):
    HEAD_ON = "head_on"
    CROSSING = "crossing"
    OVERTAKING = "overtaking"
    OTHER = "other"


class BearingSector(str, Enum):
    AHEAD = "ahead"
    STARBOARD_BOW = "starboard_bow"
    STARBOARD_BEAM = "starboard_beam"
    STARBOARD_QUARTER = "starboard_quarter"
    ASTERN = "astern"
    PORT_QUARTER = "port_quarter"
    PORT_BEAM = "port_beam"
    PORT_BOW = "port_bow"


class VesselClass(str, Enum):
    POWER = "power"
    SAILING = "sailing"
    FISHING = "fishing"
    RAM = "ram"
    NUC = "nuc"
    CBD = "cbd"
    CONSTRAINED_BY_DRAFT = "constrained_by_draft"
    AGROUND = "aground"


class NavStatus(str, Enum):
    """Navigation status (simplified) used for lights/shapes/sound rules."""

    UNDERWAY_MAKING_WAY = "underway_making_way"
    UNDERWAY_NOT_MAKING_WAY = "underway_not_making_way"
    AT_ANCHOR = "at_anchor"
    AGROUND = "aground"


class FishingMode(str, Enum):
    NONE = "none"
    TRAWLING = "trawling"
    OTHER = "other"


class IntentTurn(str, Enum):
    NONE = "none"
    PORT = "port"
    STARBOARD = "starboard"


class SailingTack(str, Enum):
    PORT = "port"
    STARBOARD = "starboard"
    UNKNOWN = "unknown"


class SailingTackRelation(str, Enum):
    """Derived relation between two sailing vessels' tacks (provided by geometry/scenario)."""

    SAME = "same"
    DIFFERENT = "different"
    OTHER_UNKNOWN = "other_unknown"


class TrafficLaneRelation(str, Enum):
    NOT_IN_TSS = "not_in_tss"
    IN_LANE_SAME_DIR = "in_lane_same_dir"
    IN_LANE_OPPOSITE_DIR = "in_lane_opposite_dir"
    CROSSING_LANE = "crossing_lane"
    IN_SEPARATION_ZONE = "in_separation_zone"
    IN_INSHORE_ZONE = "in_inshore_zone"
    NEAR_TSS_BOUNDARY = "near_tss_boundary"


class VesselInfo(BaseModel):
    vessel_class: VesselClass = Field(..., description="Vessel class/category")
    length_m: Optional[float] = Field(default=None, description="Approx vessel length (meters)")

    # Lights/shapes relevant flags (optional)
    nav_status: Optional[NavStatus] = Field(default=None, description="Simplified navigation status")
    is_towing: Optional[bool] = Field(default=None, description="Whether vessel is towing")
    tow_length_m: Optional[float] = Field(default=None, description="Length of tow (meters)")
    is_pushing: Optional[bool] = Field(default=None, description="Whether vessel is pushing ahead/towing alongside")
    pilot_on_duty: Optional[bool] = Field(default=None, description="Whether vessel is pilot vessel on duty")
    fishing_mode: Optional[FishingMode] = Field(default=None, description="Fishing mode, if vessel_class=fishing")
    sailing_using_engine: Optional[bool] = Field(default=None, description="Sailing vessel also using engine (treat as power-driven)")

    # Sailing-specific (Rule 12) placeholders (provided by scenario / geometry module)
    sailing_tack: Optional[SailingTack] = Field(default=None, description="Wind side inferred tack (port/starboard/unknown)")
    is_windward: Optional[bool] = Field(default=None, description="If same tack, whether ownship is windward")


class Features(BaseModel):
    # Environment / context
    visibility: Optional[Visibility] = None
    area_type: Optional[AreaType] = None
    traffic_lane_relation: Optional[TrafficLaneRelation] = None

    # Participants
    ownship: VesselInfo
    target: VesselInfo

    # Geometry-derived categorical features (computed by geometry module)
    encounter_type: Optional[EncounterType] = None
    relative_bearing_sector: Optional[BearingSector] = None
    is_target_on_starboard: Optional[bool] = None
    closing: Optional[bool] = None

    # Sailing-vs-sailing derived relation (Rule 12 helper)
    sailing_tack_relation: Optional[SailingTackRelation] = None

    # In-sight / perception scope
    in_sight: Optional[bool] = None

    # Risk & stage flags (optional; allow Unknown if not provided)
    risk_of_collision: Optional[bool] = None
    give_way_taking_action: Optional[bool] = None
    collision_imminent: Optional[bool] = None
    being_overtaken: Optional[bool] = None

    # Narrow channel / TSS behavioural flags (optional)
    intends_cross_channel: Optional[bool] = None
    impedes_channel_vessel: Optional[bool] = None
    agreement_received_for_overtake: Optional[bool] = None
    intends_anchor: Optional[bool] = None

    # TSS geometry aids
    tss_crossing_angle_deg: Optional[float] = None
    impedes_tss_traffic: Optional[bool] = None

    # Intent layer for sound signals (Rules 34–35)
    intent_turn: Optional[IntentTurn] = None
    intent_overtake: Optional[bool] = None
    intent_astern_propulsion: Optional[bool] = None

    # Optional evidence values (not used to derive encounter_type in this package)
    cpa_m: Optional[float] = None
    tcpa_s: Optional[float] = None
