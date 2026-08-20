"""Observable-state adapter and symbolic COLREGs rule-engine runner."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from colreg_kernel.aggregate import aggregate
from colreg_kernel.features import (
    AreaType,
    BearingSector,
    EncounterType,
    Features,
    NavStatus,
    TrafficLaneRelation,
    VesselClass,
    VesselInfo,
    Visibility,
)
from colreg_reasoner.global_types import GlobalResolveInput, TargetContext
from colreg_reasoner.multiagent import resolve_multi

from .symbolic_geometry import (
    PairwiseGeometry,
    angular_difference,
    crossing_angle_deg,
    pairwise_geometry,
    vessel_state,
)


class SymbolicEngineError(ValueError):
    """Raised when observable input cannot be adapted to the rule engine."""


VESSEL_CLASS_MAP = {
    "power_driven": VesselClass.POWER,
    "power": VesselClass.POWER,
    "sailing": VesselClass.SAILING,
    "fishing": VesselClass.FISHING,
    "ram": VesselClass.RAM,
    "nuc": VesselClass.NUC,
    "cbd": VesselClass.CBD,
    "constrained_by_draught": VesselClass.CONSTRAINED_BY_DRAFT,
    "constrained_by_draft": VesselClass.CONSTRAINED_BY_DRAFT,
    "aground": VesselClass.AGROUND,
}
NAV_STATUS_MAP = {
    "underway_making_way": NavStatus.UNDERWAY_MAKING_WAY,
    "underway_not_making_way": NavStatus.UNDERWAY_NOT_MAKING_WAY,
    "at_anchor": NavStatus.AT_ANCHOR,
    "aground": NavStatus.AGROUND,
}


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(child) for child in value]
    if hasattr(value, "value"):
        return value.value
    return value


def _vessel_info(value: Mapping[str, Any]) -> tuple[VesselInfo, list[dict[str, Any]]]:
    defaults: list[dict[str, Any]] = []
    source_class = str(value.get("vessel_class") or "")
    if source_class not in VESSEL_CLASS_MAP:
        raise SymbolicEngineError(f"unsupported vessel_class: {source_class!r}")
    source_status = str(value.get("nav_status") or "underway_making_way")
    nav_status = NAV_STATUS_MAP.get(source_status)
    if nav_status is None:
        nav_status = NavStatus.UNDERWAY_MAKING_WAY
        defaults.append(
            {
                "field": "nav_status",
                "source_value": source_status,
                "used_value": nav_status.value,
                "reason": "kernel enum is coarser; vessel_class retains special status",
            }
        )
    length = value.get("length_m")
    if not isinstance(length, (int, float)) or length <= 0:
        length = 80.0
        defaults.append(
            {
                "field": "length_m",
                "source_value": value.get("length_m"),
                "used_value": length,
                "reason": "observable length missing",
            }
        )
    return (
        VesselInfo(
            vessel_class=VESSEL_CLASS_MAP[source_class],
            length_m=float(length),
            nav_status=nav_status,
        ),
        defaults,
    )


def _area_type(area: Mapping[str, Any]) -> AreaType:
    source = str(area.get("area_type") or "open_water")
    if source == "narrow_channel" or area.get("channel_context") is True:
        return AreaType.NARROW_CHANNEL
    if source == "tss" or area.get("tss_context") is True:
        return AreaType.TSS
    return AreaType.OPEN


def _traffic_lane_relation(
    area_type: AreaType,
    own_heading: float,
    area: Mapping[str, Any],
) -> TrafficLaneRelation:
    if area_type != AreaType.TSS:
        return TrafficLaneRelation.NOT_IN_TSS
    lane_heading = area.get("tss_lane_heading_deg")
    if not isinstance(lane_heading, (int, float)):
        return TrafficLaneRelation.NEAR_TSS_BOUNDARY
    difference = angular_difference(own_heading, float(lane_heading))
    if 60.0 <= difference <= 120.0:
        return TrafficLaneRelation.CROSSING_LANE
    if difference < 60.0:
        return TrafficLaneRelation.IN_LANE_SAME_DIR
    return TrafficLaneRelation.IN_LANE_OPPOSITE_DIR


def _risk_flags(
    geometry: PairwiseGeometry,
    own_length_m: float,
    area_type: AreaType,
    params: Mapping[str, Any],
) -> tuple[bool, bool, bool]:
    risk = params.get("risk", {})
    imminent = params.get("imminent", {})
    impede = params.get("impede", {})
    tcpa = geometry.tcpa_s
    risk_dcpa = max(
        float(risk.get("dcpa_risk_nm", 0.5)) * 1852.0,
        float(risk.get("dcpa_risk_len_factor", 6.0)) * own_length_m,
    )
    imminent_dcpa = max(
        float(imminent.get("dcpa_imminent_nm", 0.2)) * 1852.0,
        float(imminent.get("dcpa_imminent_len_factor", 4.0)) * own_length_m,
    )
    has_risk = bool(
        geometry.closing
        and tcpa is not None
        and tcpa <= float(risk.get("tcpa_risk_s", 900.0))
        and geometry.cpa_m <= risk_dcpa
    )
    is_imminent = bool(
        geometry.closing
        and tcpa is not None
        and tcpa <= float(imminent.get("tcpa_imminent_s", 240.0))
        and geometry.cpa_m <= imminent_dcpa
    )
    area_key = {
        AreaType.OPEN: "open",
        AreaType.NARROW_CHANNEL: "narrow_channel",
        AreaType.TSS: "tss",
    }[area_type]
    area_impede = impede.get(area_key, {})
    impede_dcpa = max(
        float(area_impede.get("dcpa_imp_nm", 0.3)) * 1852.0,
        float(area_impede.get("dcpa_imp_len_factor", 4.0)) * own_length_m,
    )
    impedes = bool(
        geometry.closing
        and tcpa is not None
        and tcpa <= float(impede.get("t_imp_s", 720.0))
        and geometry.cpa_m <= impede_dcpa
    )
    return has_risk, is_imminent, impedes


def _build_features(
    scene: Mapping[str, Any],
    target_value: Mapping[str, Any],
    geometry: PairwiseGeometry,
    params: Mapping[str, Any],
) -> tuple[Features, list[dict[str, Any]]]:
    own_value = scene["ownship"]
    area = scene["area_context"]
    own_info, own_defaults = _vessel_info(own_value)
    target_info, target_defaults = _vessel_info(target_value)
    area_type = _area_type(area)
    visibility = (
        Visibility.RESTRICTED
        if scene.get("visibility") == "restricted_visibility"
        else Visibility.CLEAR
    )
    in_sight = visibility == Visibility.CLEAR
    lane_relation = _traffic_lane_relation(
        area_type, float(own_value["heading_deg"]), area
    )
    risk, imminent, impedes = _risk_flags(
        geometry, float(own_info.length_m or 80.0), area_type, params
    )
    channel_heading = area.get("channel_heading_deg")
    tss_heading = area.get("tss_lane_heading_deg")
    channel_angle = (
        crossing_angle_deg(float(own_value["heading_deg"]), float(channel_heading))
        if isinstance(channel_heading, (int, float))
        else 0.0
    )
    tss_angle = (
        crossing_angle_deg(float(own_value["heading_deg"]), float(tss_heading))
        if isinstance(tss_heading, (int, float))
        else None
    )
    features = Features(
        visibility=visibility,
        area_type=area_type,
        traffic_lane_relation=lane_relation,
        ownship=own_info,
        target=target_info,
        encounter_type=EncounterType(geometry.encounter_type),
        relative_bearing_sector=BearingSector(geometry.relative_bearing_sector),
        is_target_on_starboard=geometry.is_target_on_starboard,
        closing=geometry.closing,
        in_sight=in_sight,
        risk_of_collision=risk,
        collision_imminent=imminent,
        being_overtaken=False,
        intends_cross_channel=(
            area_type == AreaType.NARROW_CHANNEL and 60.0 <= channel_angle <= 90.0
        ),
        impedes_channel_vessel=(area_type == AreaType.NARROW_CHANNEL and impedes),
        tss_crossing_angle_deg=tss_angle,
        impedes_tss_traffic=(area_type == AreaType.TSS and impedes),
        cpa_m=geometry.cpa_m,
        tcpa_s=geometry.tcpa_s,
    )
    defaults = [
        {"vessel": "ownship", **item} for item in own_defaults
    ] + [
        {
            "vessel": str(target_value.get("target_id") or target_value.get("id")),
            **item,
        }
        for item in target_defaults
    ]
    return features, defaults


def run_symbolic_sample(
    row: Mapping[str, Any],
    *,
    rules_path: str | Path,
    params_path: str | Path,
    priorities_path: str | Path,
) -> dict[str, Any]:
    """Run a single inputs-only row; this function never accepts a gold object."""

    sample_id = row.get("sample_id")
    inputs = row.get("inputs")
    scene = inputs.get("scene_spec") if isinstance(inputs, Mapping) else None
    if not isinstance(sample_id, str) or not isinstance(scene, Mapping):
        raise SymbolicEngineError("row requires sample_id and inputs.scene_spec")
    targets = scene.get("targets")
    if not isinstance(targets, list) or not targets:
        raise SymbolicEngineError("scene has no targets")
    params = yaml.safe_load(Path(params_path).read_text(encoding="utf-8"))
    own_state = vessel_state(scene["ownship"])
    contexts: list[TargetContext] = []
    defaults: list[dict[str, Any]] = []
    target_traces: list[dict[str, Any]] = []
    unknowns: set[tuple[str, tuple[str, ...]]] = set()
    local_rule_ids: set[str] = set()

    for index, target_value in enumerate(targets):
        if not isinstance(target_value, Mapping):
            raise SymbolicEngineError(f"target {index} is not an object")
        target_state = vessel_state(target_value)
        geometry = pairwise_geometry(own_state, target_state)
        features, local_defaults = _build_features(scene, target_value, geometry, params)
        defaults.extend(local_defaults)
        local = aggregate(
            features,
            rules_path=str(rules_path),
            params_path=str(params_path),
            strict=False,
        )
        target_id = target_state.vessel_id
        rule_ids = sorted(item.rule_id for item in local.triggered_rules)
        local_rule_ids.update(rule_ids)
        unknowns.update(local.unknown_rules)
        contexts.append(
            TargetContext(
                target_id=target_id,
                target_index=index,
                features=features,
                geometry=geometry,
                local_aggregate=local,
                distance_m=geometry.distance_m,
                cpa_m=geometry.cpa_m,
                tcpa_s=geometry.tcpa_s,
                risk_of_collision=features.risk_of_collision,
                collision_imminent=features.collision_imminent,
                role_tags=(),
                debug_meta={"geometry_source": "observable_state_recomputed"},
            )
        )
        target_traces.append(
            {
                "target_id": target_id,
                "geometry": geometry.to_dict(),
                "features": _json_safe(features.model_dump()),
                "triggered_rule_ids": rule_ids,
                "maneuver_allowed": _json_safe(local.labels.maneuver_allowed),
                "maneuver_forbidden": _json_safe(local.labels.maneuver_forbidden),
                "unknown_rules": [
                    {"rule_id": rule_id, "missing_fields": list(missing)}
                    for rule_id, missing in local.unknown_rules
                ],
                "sanity_issues": _json_safe(local.sanity_issues),
            }
        )

    reference_features = contexts[0].features
    global_input = GlobalResolveInput(
        scene_id=sample_id,
        snapshot_index=0,
        snapshot_time_s=float(scene.get("snapshot_time_s") or 0.0),
        ownship_id=own_state.vessel_id,
        ownship_features_ref=reference_features,
        visibility=reference_features.visibility.value,
        in_sight=reference_features.in_sight,
        area_type=reference_features.area_type.value,
        traffic_lane_relation=reference_features.traffic_lane_relation.value,
        targets=tuple(contexts),
        rules_path=str(rules_path),
        strict=False,
        global_domain_flags={},
        meta={"input_boundary": "inputs_only_v1"},
    )
    resolved = resolve_multi(global_input, priorities_path=str(priorities_path))
    global_result = resolved.global_resolution
    final_labels = global_result.final_labels
    symbolic_events = ["MULTI_TARGET_RESOLUTION"] if len(contexts) > 1 else []
    native_allowed = set(final_labels.maneuver_allowed)
    if symbolic_events:
        native_allowed.add("RESOLVE_MULTI_TARGET_CONFLICT")
    return {
        "schema_version": 1,
        "run_id": "symbolic_colregs_rule_engine_v1",
        "sample_id": sample_id,
        "pattern": row.get("pattern"),
        "parse_status": "deterministic_symbolic",
        "symbolic_raw": {
            "triggered_rule_ids": sorted(local_rule_ids),
            "maneuver_allowed": sorted(native_allowed),
            "maneuver_forbidden": sorted(final_labels.maneuver_forbidden),
            "symbolic_events": symbolic_events,
        },
        "target_traces": target_traces,
        "global_resolution": _json_safe(global_result),
        "unknown_rules": [
            {"rule_id": rule_id, "missing_fields": list(missing)}
            for rule_id, missing in sorted(unknowns)
        ],
        "default_provenance": defaults,
        "input_boundary": "inputs_only_v1",
    }
