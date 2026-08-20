## `colreg_reasoner/src/colreg_reasoner/global_types.py`


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

from colreg_kernel.aggregate import AggregateResult
from colreg_kernel.features import Features

if TYPE_CHECKING:
    from colreg_scenegen.geometry import PairwiseGeometry
else:
    PairwiseGeometry = Any


@dataclass(frozen=True)
class TargetContext:
    target_id: str
    target_index: int
    features: Features
    geometry: PairwiseGeometry
    local_aggregate: AggregateResult
    distance_m: Optional[float] = None
    cpa_m: Optional[float] = None
    tcpa_s: Optional[float] = None
    risk_of_collision: Optional[bool] = None
    collision_imminent: Optional[bool] = None
    role_tags: tuple[str, ...] = field(default_factory=tuple)
    debug_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalResolveInput:
    scene_id: str
    snapshot_index: int
    snapshot_time_s: float
    ownship_id: str
    ownship_features_ref: Optional[Features] = None
    visibility: Optional[str] = None
    in_sight: Optional[bool] = None
    area_type: Optional[str] = None
    traffic_lane_relation: Optional[str] = None
    targets: tuple[TargetContext, ...] = field(default_factory=tuple)
    rules_path: Optional[str] = None
    strict: bool = False
    global_domain_flags: Dict[str, bool] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintAtom:
    atom_id: str
    target_id: str
    target_index: int
    rule_id: str
    head: str  # maneuver | lights | sounds
    primitive: str
    polarity: str  # allow | forbid | require
    priority_bucket: int
    article: Optional[str] = None
    visibility: Optional[str] = None
    in_sight: Optional[bool] = None
    encounter_type: Optional[str] = None
    relative_bearing_sector: Optional[str] = None
    is_target_on_starboard: Optional[bool] = None
    closing: Optional[bool] = None
    cpa_m: Optional[float] = None
    tcpa_s: Optional[float] = None
    risk_of_collision: Optional[bool] = None
    collision_imminent: Optional[bool] = None
    support_strength: Optional[float] = None
    reason_basis: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    debug_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GlobalPrimitiveLedger:
    maneuver: Dict[str, tuple[ConstraintAtom, ...]] = field(default_factory=dict)
    lights: Dict[str, tuple[ConstraintAtom, ...]] = field(default_factory=dict)
    sounds: Dict[str, tuple[ConstraintAtom, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SuppressionRecord:
    suppression_id: str
    suppressed_atom_id: str
    suppressed_rule_id: str
    suppressed_target_id: str
    suppressed_head: str
    suppressed_primitive: str
    suppressed_polarity: str
    suppressed_by_atom_id: Optional[str] = None
    suppressed_by_rule_id: Optional[str] = None
    suppressed_by_target_id: Optional[str] = None
    suppressed_by_head: Optional[str] = None
    suppressed_by_primitive: Optional[str] = None
    suppressed_by_polarity: Optional[str] = None
    reason_type: str = ""
    reason_text: str = ""
    suppressed_priority_bucket: Optional[int] = None
    suppressed_by_priority_bucket: Optional[int] = None
    priority_delta: Optional[int] = None
    scene_id: Optional[str] = None
    snapshot_time_s: Optional[float] = None
    evidence_refs: Dict[str, Any] = field(default_factory=dict)
    debug_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalLabelSet:
    maneuver_allowed: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    maneuver_forbidden: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    lights_required: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    lights_forbidden: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    sounds_required: Dict[str, tuple[str, ...]] = field(default_factory=dict)
    sounds_forbidden: Dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class KeptRuleRecord:
    rule_id: str
    article: Optional[str] = None
    target_ids: tuple[str, ...] = field(default_factory=tuple)
    contributing_atom_ids: tuple[str, ...] = field(default_factory=tuple)
    heads: tuple[str, ...] = field(default_factory=tuple)
    primitives: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TargetSummary:
    target_id: str
    target_index: int
    encounter_type: Optional[str] = None
    relative_bearing_sector: Optional[str] = None
    is_target_on_starboard: Optional[bool] = None
    closing: Optional[bool] = None
    cpa_m: Optional[float] = None
    tcpa_s: Optional[float] = None
    risk_of_collision: Optional[bool] = None
    collision_imminent: Optional[bool] = None
    triggered_rule_ids: tuple[str, ...] = field(default_factory=tuple)
    role_tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class GlobalResolutionV2:
    scene_id: str
    snapshot_index: int
    snapshot_time_s: float
    final_labels: FinalLabelSet = field(default_factory=FinalLabelSet)
    kept_rules: tuple[KeptRuleRecord, ...] = field(default_factory=tuple)
    suppression_records: tuple[SuppressionRecord, ...] = field(default_factory=tuple)
    target_summaries: tuple[TargetSummary, ...] = field(default_factory=tuple)
    primitive_ledger_summary: Dict[str, Any] = field(default_factory=dict)
    explanation_steps: tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    global_metrics: Dict[str, Any] = field(default_factory=dict)
    rule_graph: Dict[str, Any] = field(default_factory=dict)
    sanity_report: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
