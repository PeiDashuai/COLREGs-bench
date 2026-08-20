#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import jsonschema  # type: ignore
    HAS_JSONSCHEMA = True
except Exception:
    HAS_JSONSCHEMA = False


FAMILY = "channel_crossing_impede"
GENERATOR_VERSION = "channel_crossing_impede_native_v2_1"
GENMETA_VERSION = "genmeta_native_v1"


# ============================================================
# IO
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL {path} line {lineno}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object: {path} line {lineno}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Helpers
# ============================================================

def get_nested(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def first_non_null(*vals: Any, default: Any = None) -> Any:
    for v in vals:
        if v is not None:
            return v
    return default


def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def heading_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


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


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def summary_mentions_channel(explanation_steps: List[Dict[str, Any]]) -> bool:
    joined = " ".join(str(x.get("summary", "")) for x in explanation_steps if isinstance(x, dict)).lower()
    return "channel" in joined


def summary_mentions_starboard(explanation_steps: List[Dict[str, Any]]) -> bool:
    joined = " ".join(str(x.get("summary", "")) for x in explanation_steps if isinstance(x, dict)).lower()
    return "starboard" in joined


def count_rule_prefixes(triggered_rules: List[str]) -> Dict[str, int]:
    out = {
        "rule9": 0,
        "rule14": 0,
        "rule15": 0,
        "rule16": 0,
        "rule17": 0,
        "rule5_8": 0,
    }
    for r in [str(x) for x in triggered_rules]:
        if r.startswith("COLREG_R09"):
            out["rule9"] += 1
        elif r.startswith("COLREG_R14"):
            out["rule14"] += 1
        elif r.startswith("COLREG_R15"):
            out["rule15"] += 1
        elif r.startswith("COLREG_R16"):
            out["rule16"] += 1
        elif r.startswith("COLREG_R17"):
            out["rule17"] += 1
        elif r.startswith("COLREG_R05") or r.startswith("COLREG_R06") or r.startswith("COLREG_R07") or r.startswith("COLREG_R08"):
            out["rule5_8"] += 1
    return out


def infer_channel_width_band(channel_width_m: Optional[float]) -> Optional[str]:
    if channel_width_m is None:
        return None
    if channel_width_m < 250.0:
        return "narrow"
    if channel_width_m < 450.0:
        return "medium"
    return "wide"


def infer_ownship_channel_position(ownship_local: Dict[str, Any], channel_half_width_m: Optional[float]) -> Optional[str]:
    if not isinstance(ownship_local, dict):
        return None
    notes = [str(x) for x in ownship_local.get("notes", [])] if isinstance(ownship_local.get("notes"), list) else []
    y_local = safe_float(ownship_local.get("y_local_m"))
    if "ownship_path_crosses_channel_corridor" in notes:
        return "crossing_across"
    if "ownship_started_outside_channel_and_enters" in notes:
        return "entering_from_outside"
    if "ownship_started_inside_channel_center_band" in notes:
        return "inside_center"
    if "ownship_started_inside_channel_boundary_band" in notes:
        return "inside_boundary"
    if channel_half_width_m is not None and y_local is not None:
        abs_y = abs(y_local)
        if abs_y <= 0.20 * channel_half_width_m:
            return "inside_center"
        if abs_y <= 0.80 * channel_half_width_m:
            return "inside_boundary"
        return "entering_from_outside"
    return None


def infer_occupancy_pattern(targets: List[Dict[str, Any]], latent_occupancy: Optional[str] = None) -> str:
    roles = [str(t.get("intended_role")) for t in targets if isinstance(t, dict)]
    n = len(targets)
    latent_occupancy = str(latent_occupancy) if latent_occupancy is not None else None

    # This annotator is not trying to rediscover the family from scratch.
    # It checks whether the latent occupancy regime is plausibly realized.
    if latent_occupancy == "third_vessel_blocking" and n >= 3:
        return "third_vessel_blocking"
    if latent_occupancy == "low_occupancy" and n <= 2:
        return "low_occupancy"

    if any(r == "blocking_target" for r in roles) and n >= 3:
        return "third_vessel_blocking"
    if any(r == "opposing_along_channel_target" for r in roles) and n >= 2:
        return "opposing_flow"
    if n == 1:
        return "single_occupancy"
    return "low_occupancy"


def infer_primary_crossing_actor(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Dict[str, Any]]:
    family_latent = row.get("family_latent", {}) if isinstance(row.get("family_latent"), dict) else {}
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    pos = str(family_latent.get("ownship_channel_position"))

    if pos in {"entering_from_outside", "crossing_across"}:
        own_local = get_nested(scene_spec, "debug_native_geometry.ownship_local", {})
        if isinstance(own_local, dict):
            return "ownship", safe_float(own_local.get("heading_local_deg")), own_local
        return None, None, {}

    for t in scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []:
        if not isinstance(t, dict):
            continue
        if str(t.get("intended_role")) == "crossing_target":
            lg = t.get("local_geometry", {}) if isinstance(t.get("local_geometry"), dict) else {}
            return str(t.get("id", "crossing_target")), safe_float(lg.get("heading_local_deg")), lg
    return None, None, {}


def infer_urgency_band(target_summaries: List[Dict[str, Any]]) -> str:
    min_tcpa = None
    min_cpa = None
    for ts in target_summaries:
        if not isinstance(ts, dict):
            continue
        tcpa = safe_float(ts.get("tcpa_s"))
        cpa = safe_float(ts.get("cpa_m"))
        if tcpa is not None:
            min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)
        if cpa is not None:
            min_cpa = cpa if min_cpa is None else min(min_cpa, cpa)
    if (min_tcpa is not None and min_tcpa <= 35.0) or (min_cpa is not None and min_cpa <= 30.0):
        return "high"
    if (min_tcpa is not None and min_tcpa <= 90.0) or (min_cpa is not None and min_cpa <= 100.0):
        return "medium"
    return "low"


def stable_index_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    gm = sample.get("generation_meta", {})
    return {
        "sample_id": sample.get("sample_id"),
        "pattern": sample.get("pattern"),
        "schema_version": gm.get("schema_version"),
        "family_stratum": get_nested(gm, "family_semantics.family_stratum", {}),
        "effective_participants": get_nested(gm, "vessel_population.effective_participants"),
        "semantic_mode": get_nested(gm, "family_semantics.family_stratum.semantic_mode"),
        "occupancy_pattern": get_nested(gm, "family_semantics.family_stratum.occupancy_pattern"),
        "ownship_channel_position": get_nested(gm, "family_semantics.family_stratum.ownship_channel_position"),
        "quality_flags": get_nested(gm, "quality_flags", {}),
    }


# ============================================================
# Annotation blocks
# ============================================================

def derive_generator_block(row: Dict[str, Any]) -> Dict[str, Any]:
    gv = row.get("generator_version", GENERATOR_VERSION)
    return {
        "family_generator": gv,
        "candidate_seed": row.get("candidate_seed"),
        "sampling_round": 1,
        "source_mode": row.get("source_mode", "native_v3_generator"),
        "config_hash": None,
        "family_config_version": gv,
        "global_config_version": "benchmark_v3_native_genmeta_v1",
    }


def derive_scene_context(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    area = ss.get("area_context", {}) if isinstance(ss.get("area_context"), dict) else {}
    visibility = str(ss.get("visibility", "in_sight"))
    channel_width_m = safe_float(area.get("channel_width_m"))
    return {
        "domain": ss.get("domain", "open"),
        "visibility": visibility,
        "in_sight": visibility == "in_sight",
        "area_type": area.get("area_type", "narrow_channel"),
        "channel_context": bool(area.get("channel_context", False)),
        "tss_context": bool(area.get("tss_context", False)),
        "snapshot_time_s": safe_int(ss.get("snapshot_time_s"), 30),
        "duration_s": safe_int(ss.get("duration_s"), 180),
        "channel_width_m": channel_width_m,
        "channel_half_width_m": safe_float(area.get("channel_half_width_m")),
        "channel_heading_deg": safe_float(area.get("channel_heading_deg")),
        "channel_width_band": infer_channel_width_band(channel_width_m),
    }


def derive_vessel_population(row: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    ownship = ss.get("ownship", {}) if isinstance(ss.get("ownship"), dict) else {}
    targets = ss.get("targets", []) if isinstance(ss.get("targets"), list) else []

    ownship_status = str(first_non_null(ownship.get("vessel_class"), ownship.get("nav_status"), "power_driven"))
    target_status_hist: Dict[str, int] = {}
    target_count_by_role: Dict[str, int] = {
        "crossing_targets": 0,
        "opposing_along_channel_targets": 0,
        "blocking_targets": 0,
        "other_targets": 0,
    }
    for t in targets:
        if not isinstance(t, dict):
            continue
        token = str(first_non_null(t.get("vessel_class"), t.get("nav_status"), "unknown"))
        target_status_hist[token] = target_status_hist.get(token, 0) + 1
        role = str(t.get("intended_role", "other"))
        if role == "crossing_target":
            target_count_by_role["crossing_targets"] += 1
        elif role == "opposing_along_channel_target":
            target_count_by_role["opposing_along_channel_targets"] += 1
        elif role == "blocking_target":
            target_count_by_role["blocking_targets"] += 1
        else:
            target_count_by_role["other_targets"] += 1

    return {
        "total_vessels": len(targets) + 1,
        "target_vessels": len(targets),
        "effective_participants": len(targets),
        "ownship_status": ownship_status,
        "target_status_histogram": target_status_hist,
        "target_count_by_role": target_count_by_role,
    }


def derive_interaction_structure(row: Dict[str, Any], scene_context: Dict[str, Any], vessel_population: Dict[str, Any]) -> Dict[str, Any]:
    ss = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    ownship = ss.get("ownship", {}) if isinstance(ss.get("ownship"), dict) else {}
    targets = ss.get("targets", []) if isinstance(ss.get("targets"), list) else []
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    target_summaries = labels.get("target_summaries", []) if isinstance(labels.get("target_summaries"), list) else []

    own_x = safe_float(ownship.get("x_m"), 0.0)
    own_y = safe_float(ownship.get("y_m"), 0.0)
    own_h = safe_float(ownship.get("heading_deg"), 0.0)

    bearing_occ = {"port": 0, "starboard": 0, "ahead": 0, "astern": 0}
    relation_hist = {"crossing": 0, "opposing_along_channel": 0, "blocking": 0, "other": 0}
    effective_target_ids: List[str] = []
    min_cpa = None
    min_tcpa = None

    ts_map = {str(ts.get("target_id")): ts for ts in target_summaries if isinstance(ts, dict)}
    for t in targets:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id"))
        sec = rel_bearing_sector(
            own_x=float(own_x or 0.0),
            own_y=float(own_y or 0.0),
            own_heading_deg=float(own_h or 0.0),
            tgt_x=float(safe_float(t.get("x_m"), 0.0) or 0.0),
            tgt_y=float(safe_float(t.get("y_m"), 0.0) or 0.0),
        )
        bearing_occ[sec] = bearing_occ.get(sec, 0) + 1
        effective_target_ids.append(tid)

        role = str(t.get("intended_role", "other"))
        if role == "crossing_target":
            relation_hist["crossing"] += 1
        elif role == "opposing_along_channel_target":
            relation_hist["opposing_along_channel"] += 1
        elif role == "blocking_target":
            relation_hist["blocking"] += 1
        else:
            relation_hist["other"] += 1

        ts = ts_map.get(tid, {})
        cpa = safe_float(ts.get("cpa_m"))
        tcpa = safe_float(ts.get("tcpa_s"))
        if cpa is not None:
            min_cpa = cpa if min_cpa is None else min(min_cpa, cpa)
        if tcpa is not None:
            min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)

    actor_id, actor_local_heading, actor_debug = infer_primary_crossing_actor(row)
    inferred_crossing_band = infer_crossing_angle_band_from_local_heading(actor_local_heading)

    spatial_topology = "single_side_cluster"
    if relation_hist["blocking"] > 0:
        spatial_topology = "channel_space_compression"
    elif relation_hist["crossing"] > 0 and relation_hist["opposing_along_channel"] > 0:
        spatial_topology = "crossing_plus_opposing_flow"
    elif relation_hist["crossing"] > 0:
        spatial_topology = "crossing_dominant"
    elif relation_hist["opposing_along_channel"] > 0:
        spatial_topology = "along_channel_conflict"

    return {
        "encounter_mode": "multi_target" if vessel_population["effective_participants"] >= 2 else "single_target",
        "effective_target_ids": effective_target_ids,
        "relation_type_histogram": relation_hist,
        "relative_bearing_occupancy": bearing_occ,
        "spatial_topology": spatial_topology,
        "interaction_graph_density": "dense" if vessel_population["effective_participants"] >= 3 else "medium" if vessel_population["effective_participants"] == 2 else "sparse",
        "min_cpa_m": min_cpa,
        "min_tcpa_s": min_tcpa,
        "urgency_band": infer_urgency_band(target_summaries),
        "crossing_angle_band": inferred_crossing_band,
        "crossing_actor_id": actor_id,
        "crossing_actor_local_heading_deg": actor_local_heading,
        "crossing_actor_debug": actor_debug,
    }


def derive_resolution_structure(row: Dict[str, Any], vessel_population: Dict[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    triggered_rules = [str(x) for x in labels.get("triggered_rules", [])] if isinstance(labels.get("triggered_rules"), list) else []
    allowed = labels.get("maneuver_allowed", []) if isinstance(labels.get("maneuver_allowed"), list) else []
    forbidden = labels.get("maneuver_forbidden", []) if isinstance(labels.get("maneuver_forbidden"), list) else []
    suppressed_rules = labels.get("suppressed_rules", []) if isinstance(labels.get("suppressed_rules"), list) else []
    explanation_steps = labels.get("explanation_steps", []) if isinstance(labels.get("explanation_steps"), list) else []
    rule_prefix_hist = count_rule_prefixes(triggered_rules)

    return {
        "num_triggered_rules": len(triggered_rules),
        "num_effective_targets": vessel_population["effective_participants"],
        "num_conflicts": 0,
        "num_tensions": 1 if len(allowed) > 0 and len(forbidden) > 0 else 0,
        "num_compatibles": len(triggered_rules),
        "suppression_present": len(suppressed_rules) > 0,
        "suppression_count": len(suppressed_rules),
        "suppression_type": "na",
        "retained_action_count": len(allowed),
        "forbidden_action_count": len(forbidden),
        "explanation_stage_count": len(explanation_steps),
        "channel_dependence_present": summary_mentions_channel(explanation_steps),
        "rule_prefix_histogram": rule_prefix_hist,
        "global_resolution_nontrivial": rule_prefix_hist["rule9"] > 0 and (rule_prefix_hist["rule14"] + rule_prefix_hist["rule15"] + rule_prefix_hist["rule16"] + rule_prefix_hist["rule17"]) > 0,
    }


def derive_family_semantics(
    row: Dict[str, Any],
    scene_context: Dict[str, Any],
    vessel_population: Dict[str, Any],
    interaction_structure: Dict[str, Any],
    resolution_structure: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    latent = row.get("family_latent", {}) if isinstance(row.get("family_latent"), dict) else {}
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec.get("area_context"), dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []
    explanation_steps = labels.get("explanation_steps", []) if isinstance(labels.get("explanation_steps"), list) else []
    triggered_rules = [str(x) for x in labels.get("triggered_rules", [])] if isinstance(labels.get("triggered_rules"), list) else []
    ownship_local = get_nested(scene_spec, "debug_native_geometry.ownship_local", {})

    rule9_family_present = any(r.startswith("COLREG_R09") for r in triggered_rules)
    keep_starboard_effect_present = any(r == "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT" for r in triggered_rules) or summary_mentions_starboard(explanation_steps)
    not_to_impede_present = any(r == "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" for r in triggered_rules)
    ordinary_crossing_present = any(str(t.get("intended_role")) == "crossing_target" for t in targets if isinstance(t, dict))
    mixed_overlap_present = (any(str(t.get("intended_role")) == "crossing_target" for t in targets if isinstance(t, dict)) and
                            any(str(t.get("intended_role")) == "opposing_along_channel_target" for t in targets if isinstance(t, dict))) or \
                           (rule9_family_present and (any(r.startswith("COLREG_R14") for r in triggered_rules) or any(r.startswith("COLREG_R15") for r in triggered_rules)))

    observed_stratum = {
        "semantic_mode": None,
        "channel_width_band": infer_channel_width_band(safe_float(area.get("channel_width_m"))),
        "occupancy_pattern": infer_occupancy_pattern(targets, latent.get("occupancy_pattern")),
        "crossing_angle_band": interaction_structure.get("crossing_angle_band"),
        "ownship_channel_position": infer_ownship_channel_position(
            ownship_local if isinstance(ownship_local, dict) else {},
            safe_float(area.get("channel_half_width_m")),
        ),
        "effective_target_count": len(targets),
    }

    latent_semantic_mode = str(latent.get("semantic_mode")) if latent.get("semantic_mode") is not None else None
    if latent_semantic_mode == "ordinary_crossing_in_channel" and ordinary_crossing_present and rule9_family_present:
        observed_stratum["semantic_mode"] = "ordinary_crossing_in_channel"
    elif latent_semantic_mode == "keep_starboard_dominant" and keep_starboard_effect_present and rule9_family_present:
        observed_stratum["semantic_mode"] = "keep_starboard_dominant"
    elif latent_semantic_mode == "not_to_impede" and not_to_impede_present and rule9_family_present:
        observed_stratum["semantic_mode"] = "not_to_impede"
    elif latent_semantic_mode == "mixed_semantics" and mixed_overlap_present and rule9_family_present:
        observed_stratum["semantic_mode"] = "mixed_semantics"
    else:
        observed_stratum["semantic_mode"] = latent_semantic_mode

    latent_alignment = {
        "generator_version_match": row.get("generator_version") == GENERATOR_VERSION,
        "channel_context_valid": scene_context.get("channel_context") is True and scene_context.get("area_type") == "narrow_channel",
        "rule9_family_present": rule9_family_present,
        "channel_dependence_present": resolution_structure.get("channel_dependence_present") is True,
        "channel_width_band_match": latent.get("channel_width_band") == observed_stratum.get("channel_width_band"),
        "occupancy_pattern_match": latent.get("occupancy_pattern") == observed_stratum.get("occupancy_pattern"),
        "crossing_angle_band_match": latent.get("crossing_angle_band") == observed_stratum.get("crossing_angle_band"),
        "ownship_channel_position_match": latent.get("ownship_channel_position") == observed_stratum.get("ownship_channel_position"),
        "effective_target_count_match": safe_int(latent.get("effective_target_count")) == len(targets),
        "semantic_mode_match": latent.get("semantic_mode") == observed_stratum.get("semantic_mode"),
        "ordinary_crossing_present_when_needed": True if latent_semantic_mode != "ordinary_crossing_in_channel" else ordinary_crossing_present,
        "keep_starboard_present_when_needed": True if latent_semantic_mode != "keep_starboard_dominant" else keep_starboard_effect_present,
        "not_to_impede_present_when_needed": True if latent_semantic_mode != "not_to_impede" else not_to_impede_present,
        "mixed_overlap_present_when_needed": True if latent_semantic_mode != "mixed_semantics" else mixed_overlap_present,
    }

    family_invariant_pass = all(bool(v) for v in latent_alignment.values())
    semantic_tags = [
        "channel_context",
        "rule9_family" if rule9_family_present else "rule9_missing",
        f"semantic:{observed_stratum.get('semantic_mode')}",
        f"width:{observed_stratum.get('channel_width_band')}",
        f"occupancy:{observed_stratum.get('occupancy_pattern')}",
        f"crossing_angle:{observed_stratum.get('crossing_angle_band')}",
        f"ownship_pos:{observed_stratum.get('ownship_channel_position')}",
        f"urgency:{interaction_structure.get('urgency_band')}",
    ]
    if keep_starboard_effect_present:
        semantic_tags.append("keep_starboard_effect")
    if not_to_impede_present:
        semantic_tags.append("not_to_impede")
    if mixed_overlap_present:
        semantic_tags.append("mixed_overlap")

    family_semantics = {
        "family_invariant_pass": family_invariant_pass,
        "family_stratum": observed_stratum,
        "semantic_tags": semantic_tags,
        "latent_alignment": latent_alignment,
        "native_latent_snapshot": copy.deepcopy(latent),
        "rule9_family_present": rule9_family_present,
        "keep_starboard_effect_present": keep_starboard_effect_present,
        "not_to_impede_present": not_to_impede_present,
        "ordinary_crossing_present": ordinary_crossing_present,
        "mixed_overlap_present": mixed_overlap_present,
        "sample_reducible_to_open_water_crossing": False if rule9_family_present and resolution_structure.get("channel_dependence_present") else True,
    }
    return family_semantics, observed_stratum


def derive_signatures(row: Dict[str, Any], interaction_structure: Dict[str, Any], resolution_structure: Dict[str, Any]) -> Dict[str, Any]:
    labels = row.get("labels", {}) if isinstance(row.get("labels"), dict) else {}
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec.get("area_context"), dict) else {}

    geometry_signature = {
        "effective_participants": len(targets),
        "bearing_histogram": [
            interaction_structure["relative_bearing_occupancy"]["port"],
            interaction_structure["relative_bearing_occupancy"]["starboard"],
            interaction_structure["relative_bearing_occupancy"]["ahead"],
            interaction_structure["relative_bearing_occupancy"]["astern"],
        ],
        "distance_band_histogram": [],
        "speed_delta_band_histogram": [],
        "spatial_topology": interaction_structure["spatial_topology"],
        "urgency_band": interaction_structure["urgency_band"],
        "channel_width_band": infer_channel_width_band(safe_float(area.get("channel_width_m"))),
    }

    triggered_rules = [str(x) for x in labels.get("triggered_rules", [])] if isinstance(labels.get("triggered_rules"), list) else []
    allowed = labels.get("maneuver_allowed", []) if isinstance(labels.get("maneuver_allowed"), list) else []
    forbidden = labels.get("maneuver_forbidden", []) if isinstance(labels.get("maneuver_forbidden"), list) else []

    lights_required = labels.get("lights_required", []) if isinstance(labels.get("lights_required"), list) else []
    sounds_required = labels.get("sounds_required", []) if isinstance(labels.get("sounds_required"), list) else []

    rule_signature = {
        "triggered_rules_sorted": sorted(set(triggered_rules)),
        "maneuver_allowed_sorted": sorted(set(str(x) for x in allowed)),
        "maneuver_forbidden_sorted": sorted(set(str(x) for x in forbidden)),
        "lights_required_sorted": sorted(set(str(x) for x in lights_required)),
        "sounds_required_sorted": sorted(set(str(x) for x in sounds_required)),
    }

    resolution_signature = {
        "suppression_type": resolution_structure["suppression_type"],
        "suppression_count": resolution_structure["suppression_count"],
        "retained_action_count": resolution_structure["retained_action_count"],
        "global_resolution_nontrivial": resolution_structure["global_resolution_nontrivial"],
    }

    visual_signature = {
        "layout_hash": None,
        "radar_hash": None,
    }

    return {
        "geometry_signature": geometry_signature,
        "rule_signature": rule_signature,
        "resolution_signature": resolution_signature,
        "visual_signature": visual_signature,
    }


def derive_quality_flags(family_semantics: Dict[str, Any]) -> Dict[str, Any]:
    fail_reasons = []
    if not family_semantics.get("family_invariant_pass"):
        la = family_semantics.get("latent_alignment", {}) if isinstance(family_semantics.get("latent_alignment"), dict) else {}
        fail_reasons = [k for k, v in la.items() if not bool(v)]
    return {
        "passes_invariant_gate": bool(family_semantics.get("family_invariant_pass")),
        "passes_plausibility_gate": None,
        "passes_duplicate_gate": None,
        "passes_diversity_gate": None,
        "accepted_into_candidate_pool": None,
        "accepted_into_release_split": False,
        "rejection_reasons": fail_reasons,
    }


def annotate_native_channel_row(row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    scene_context = derive_scene_context(out)
    vessel_population = derive_vessel_population(out)
    interaction_structure = derive_interaction_structure(out, scene_context, vessel_population)
    resolution_structure = derive_resolution_structure(out, vessel_population)
    family_semantics, _ = derive_family_semantics(
        out, scene_context, vessel_population, interaction_structure, resolution_structure
    )
    signatures = derive_signatures(out, interaction_structure, resolution_structure)
    quality_flags = derive_quality_flags(family_semantics)

    out["generation_meta"] = {
        "schema_version": GENMETA_VERSION,
        "generator": derive_generator_block(out),
        "scene_context": scene_context,
        "vessel_population": vessel_population,
        "interaction_structure": interaction_structure,
        "family_semantics": family_semantics,
        "resolution_structure": resolution_structure,
        "signatures": signatures,
        "quality_flags": quality_flags,
    }
    return out


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Annotate native generation meta for channel_crossing_impede."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default=FAMILY, choices=[FAMILY], help="Only channel_crossing_impede is supported in this script")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing annotated outputs")
    ap.add_argument("--validate-schema", action="store_true", help="Validate raw rows against raw_candidate.schema.json if jsonschema is available")
    ap.add_argument("--report-json", type=str, default=None, help="Optional report output path")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 2

    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    annotated_path = root / "annotated_pool" / FAMILY / "candidates_annotated.jsonl"
    index_path = root / "annotated_pool" / FAMILY / "generation_meta_index.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"

    if not raw_path.exists():
        print(f"ERROR: missing raw candidate file: {raw_path}", file=sys.stderr)
        return 2
    if annotated_path.exists() and not args.overwrite:
        print(f"ERROR: annotated output exists; use --overwrite: {annotated_path}", file=sys.stderr)
        return 2
    if args.validate_schema and not schema_path.exists():
        print(f"ERROR: missing schema file required by --validate-schema: {schema_path}", file=sys.stderr)
        return 2

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if args.validate_schema and schema_path.exists() else {}

    annotated_rows: List[Dict[str, Any]] = []
    schema_errors: List[Dict[str, Any]] = []
    invariant_pass_count = 0
    latent_alignment_fail_count = 0
    strata_hist: Dict[str, int] = {}

    for row in rows:
        sample_id = row.get("sample_id", "<missing_sample_id>")
        if args.validate_schema:
            schema_err = validate_against_schema(row, schema)
            if schema_err is not None:
                schema_errors.append({"sample_id": sample_id, "error": schema_err})

        ann = annotate_native_channel_row(row)
        annotated_rows.append(ann)

        gm = ann.get("generation_meta", {}) if isinstance(ann.get("generation_meta"), dict) else {}
        fam_sem = gm.get("family_semantics", {}) if isinstance(gm.get("family_semantics"), dict) else {}
        if fam_sem.get("family_invariant_pass") is True:
            invariant_pass_count += 1
        else:
            latent_alignment_fail_count += 1

        stratum = fam_sem.get("family_stratum", {}) if isinstance(fam_sem.get("family_stratum"), dict) else {}
        key = json.dumps(stratum, ensure_ascii=False, sort_keys=True)
        strata_hist[key] = strata_hist.get(key, 0) + 1

    write_jsonl(annotated_path, annotated_rows)
    write_jsonl(index_path, [stable_index_row(r) for r in annotated_rows])

    report = {
        "root": str(root),
        "family": FAMILY,
        "raw_file": str(raw_path),
        "annotated_file": str(annotated_path),
        "index_file": str(index_path),
        "row_count": len(rows),
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_validation_enabled": bool(args.validate_schema),
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:20],
        "family_invariant_pass_count": invariant_pass_count,
        "latent_alignment_fail_count": latent_alignment_fail_count,
        "unique_family_strata_count": len(strata_hist),
        "top_family_strata": sorted(
            [{"stratum": json.loads(k), "count": v} for k, v in strata_hist.items()],
            key=lambda x: (-x["count"], json.dumps(x["stratum"], sort_keys=True))
        )[:20],
        "notes": (
            "This script annotates native-line channel_crossing_impede samples into generation_meta. "
            "It preserves raw labels, adds standardized native metadata, and checks latent realization."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "family_reports" / "channel_crossing_impede_native_annotation_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"annotate_native_generation_meta_channel_crossing_impede.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows                     : {len(rows)}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {len(schema_errors)}")
    print(f"Invariant pass           : {invariant_pass_count}/{len(rows)}")
    print(f"Unique strata            : {len(strata_hist)}")
    print(f"Annotated output         : {annotated_path}")
    print(f"Meta index               : {index_path}")
    print(f"Report                   : {report_path}")
    print("=" * 100)

    return 0 if (len(schema_errors) == 0 and invariant_pass_count == len(rows)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
