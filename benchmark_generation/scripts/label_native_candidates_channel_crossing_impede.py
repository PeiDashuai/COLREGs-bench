#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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

def safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


def deg_to_unit_vec(heading_deg: float) -> Tuple[float, float]:
    r = math.radians(heading_deg)
    return math.cos(r), math.sin(r)


def heading_diff_deg(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs((a - b + 180.0) % 360.0 - 180.0)


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


def cpa_tcpa(
    own_x: float, own_y: float, own_heading_deg: float, own_speed: float,
    tgt_x: float, tgt_y: float, tgt_heading_deg: float, tgt_speed: float,
) -> Tuple[float, float]:
    ovx_u, ovy_u = deg_to_unit_vec(own_heading_deg)
    tvx_u, tvy_u = deg_to_unit_vec(tgt_heading_deg)
    ovx, ovy = ovx_u * own_speed, ovy_u * own_speed
    tvx, tvy = tvx_u * tgt_speed, tvy_u * tgt_speed

    rx = tgt_x - own_x
    ry = tgt_y - own_y
    rvx = tvx - ovx
    rvy = tvy - ovy
    rv2 = rvx * rvx + rvy * rvy

    if rv2 <= 1e-9:
        return math.hypot(rx, ry), 0.0

    tcpa = - (rx * rvx + ry * rvy) / rv2
    if tcpa < 0:
        tcpa = 0.0
    cpa_x = rx + rvx * tcpa
    cpa_y = ry + rvy * tcpa
    return math.hypot(cpa_x, cpa_y), tcpa


def unique_keep_order(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def validate_against_schema(row: Dict[str, Any], schema: Dict[str, Any]) -> Optional[str]:
    if not HAS_JSONSCHEMA:
        return None
    try:
        jsonschema.validate(instance=row, schema=schema)
        return None
    except Exception as e:
        return str(e)


def load_adapter(adapter_spec: Optional[str]) -> Optional[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    if not adapter_spec:
        return None
    if ":" not in adapter_spec:
        raise ValueError("Adapter spec must be in form module.submodule:function_name")
    module_name, func_name = adapter_spec.split(":", 1)
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Adapter callable not found: {adapter_spec}")
    return fn


def infer_crossing_band_from_local_heading(local_heading_deg: Optional[float]) -> Optional[str]:
    if local_heading_deg is None:
        return None
    x = abs(float(local_heading_deg))
    if 15.0 <= x <= 40.0:
        return "small"
    if 45.0 <= x <= 70.0:
        return "medium"
    if 75.0 <= x <= 90.0:
        return "large"
    return None


def has_role(targets: List[Dict[str, Any]], role: str) -> bool:
    return any(str(t.get("intended_role")) == role for t in targets if isinstance(t, dict))


# ============================================================
# Built-in authoritative labeler
# ============================================================

def authoritative_channel_crossing_impede_labels_v1(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    family_latent = row.get("family_latent", {}) if isinstance(row.get("family_latent"), dict) else {}
    scene_spec = row.get("scene_spec", {}) if isinstance(row.get("scene_spec"), dict) else {}
    area = scene_spec.get("area_context", {}) if isinstance(scene_spec.get("area_context"), dict) else {}
    ownship = scene_spec.get("ownship", {}) if isinstance(scene_spec.get("ownship"), dict) else {}
    targets = scene_spec.get("targets", []) if isinstance(scene_spec.get("targets"), list) else []
    debug_geom = scene_spec.get("debug_native_geometry", {}) if isinstance(scene_spec.get("debug_native_geometry"), dict) else {}
    own_local = debug_geom.get("ownship_local", {}) if isinstance(debug_geom.get("ownship_local"), dict) else {}

    sem = str(family_latent.get("semantic_mode"))
    occ = str(family_latent.get("occupancy_pattern"))
    pos = str(family_latent.get("ownship_channel_position"))
    latent_angle_band = str(family_latent.get("crossing_angle_band"))

    own_x = safe_float(ownship.get("x_m"), 0.0) or 0.0
    own_y = safe_float(ownship.get("y_m"), 0.0) or 0.0
    own_h = safe_float(ownship.get("heading_deg"), safe_float(area.get("channel_heading_deg"), 0.0) or 0.0) or 0.0
    own_speed = safe_float(ownship.get("speed_mps"), 0.0) or 0.0
    lane_heading = safe_float(area.get("channel_heading_deg"), own_h) or own_h
    channel_width_m = safe_float(area.get("channel_width_m"), 0.0) or 0.0
    channel_half_width_m = safe_float(area.get("channel_half_width_m"), channel_width_m / 2.0 if channel_width_m > 0 else 0.0) or 0.0
    own_local_heading = safe_float(own_local.get("heading_local_deg"))
    observed_crossing_angle_band = infer_crossing_band_from_local_heading(own_local_heading) or latent_angle_band

    aggregate_rules: List[str] = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
        "COLREG_R08_POSITIVE_ACTION_WHEN_RISK",
        "COLREG_R09_NARROW_CHANNEL_GENERAL",
    ]
    allowed: List[str] = [
        "KEEP_LOOKOUT",
        "PROCEED_AT_SAFE_SPEED",
        "RESPECT_NARROW_CHANNEL_CONSTRAINTS",
    ]
    forbidden: List[str] = [
        "IGNORE_CHANNEL_CONTEXT",
        "OPEN_WATER_ONLY_REASONING",
    ]

    if sem == "ordinary_crossing_in_channel":
        aggregate_rules += [
            "COLREG_R09_CROSSING_NARROW_CHANNEL",
        ]
        allowed += [
            "CROSS_ONLY_IF_SAFE_AND_CLEAR",
            "AVOID_IMPEDING_CHANNEL_TRAFFIC",
        ]
        forbidden += [
            "CROSS_WITHOUT_CLEARANCE",
        ]
    elif sem == "keep_starboard_dominant":
        aggregate_rules += [
            "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT",
        ]
        allowed += [
            "KEEP_TO_STARBOARD_WITHIN_CHANNEL",
        ]
        forbidden += [
            "DRIFT_TO_PORT_WITHIN_CHANNEL",
        ]
    elif sem == "not_to_impede":
        aggregate_rules += [
            "COLREG_R09_NOT_TO_IMPEDE_PASSAGE",
        ]
        allowed += [
            "DO_NOT_IMPEDE_CHANNEL_TRAFFIC",
            "WAIT_OR_ADJUST_BEFORE_CROSSING",
        ]
        forbidden += [
            "FORCE_CROSSING_THROUGH_CHANNEL_TRAFFIC",
        ]
    elif sem == "mixed_semantics":
        aggregate_rules += [
            "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT",
            "COLREG_R09_NOT_TO_IMPEDE_PASSAGE",
        ]
        allowed += [
            "KEEP_TO_STARBOARD_WITHIN_CHANNEL",
            "DO_NOT_IMPEDE_CHANNEL_TRAFFIC",
            "RESOLVE_MULTIPLE_CHANNEL_CONSTRAINTS",
        ]
        forbidden += [
            "REDUCE_TO_SINGLE_OPEN_WATER_CROSSING_RULE",
        ]
    else:
        aggregate_rules += ["COLREG_R09_CROSSING_NARROW_CHANNEL"]

    # Occupancy-derived rules
    if occ in {"opposing_flow", "third_vessel_blocking"}:
        aggregate_rules += ["COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK"]
        allowed += ["CHECK_RECIPROCAL_CHANNEL_TRAFFIC"]
    if occ == "third_vessel_blocking":
        aggregate_rules += ["COLREG_R08_AVOID_CONSTRAINED_CHANNEL_COMPRESSION"]
        allowed += ["AVOID_CHANNEL_COMPRESSION"]
        forbidden += ["SQUEEZE_THROUGH_BOUNDARY_GAP"]

    if observed_crossing_angle_band == "small":
        allowed += ["CROSS_AT_SMALL_CHANNEL_ENTRY_ANGLE"]
    elif observed_crossing_angle_band == "medium":
        allowed += ["CROSS_AT_MODERATE_CHANNEL_ENTRY_ANGLE"]
    elif observed_crossing_angle_band == "large":
        allowed += ["CROSS_AT_STEEP_CHANNEL_ENTRY_ANGLE"]

    target_summaries: List[Dict[str, Any]] = []
    has_r15 = False
    has_r14 = False
    has_not_to_impede_target = False

    for i, t in enumerate(targets):
        if not isinstance(t, dict):
            continue
        tid = str(t.get("id", f"t{i+1}"))
        role = str(t.get("intended_role"))
        tgt_x = safe_float(t.get("x_m"), 0.0) or 0.0
        tgt_y = safe_float(t.get("y_m"), 0.0) or 0.0
        tgt_h = safe_float(t.get("heading_deg"), lane_heading) or lane_heading
        tgt_speed = safe_float(t.get("speed_mps"), 0.0) or 0.0
        sector = rel_bearing_sector(own_x, own_y, own_h, tgt_x, tgt_y)
        cpa, tcpa = cpa_tcpa(own_x, own_y, own_h, own_speed, tgt_x, tgt_y, tgt_h, tgt_speed)

        trg_rules: List[str] = []
        if role == "crossing_target":
            trg_rules.append("COLREG_R09_CROSSING_NARROW_CHANNEL")
            if sector == "starboard":
                trg_rules += [
                    "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD",
                    "COLREG_R16_GIVEWAY_MANEUVER",
                ]
                has_r15 = True
            elif tcpa > 0 and cpa <= 250.0:
                trg_rules += ["COLREG_R16_GIVEWAY_MANEUVER"]
        elif role == "opposing_along_channel_target":
            if sector == "ahead":
                trg_rules += ["COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK"]
                has_r14 = True
            trg_rules += ["COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT"]
        elif role == "channel_traffic_target":
            trg_rules += [
                "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT",
                "COLREG_R09_NOT_TO_IMPEDE_PASSAGE",
            ]
            has_not_to_impede_target = True
        elif role == "blocking_boundary_target":
            trg_rules += [
                "COLREG_R08_AVOID_CONSTRAINED_CHANNEL_COMPRESSION",
                "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT",
            ]
        else:
            trg_rules += ["COLREG_R09_NARROW_CHANNEL_GENERAL"]

        if sem == "keep_starboard_dominant" and role != "crossing_target":
            trg_rules.append("COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
        if sem == "not_to_impede" and role in {"channel_traffic_target", "opposing_along_channel_target"}:
            trg_rules.append("COLREG_R09_NOT_TO_IMPEDE_PASSAGE")
            has_not_to_impede_target = True
        if sem == "mixed_semantics" and role in {"channel_traffic_target", "opposing_along_channel_target"}:
            trg_rules.append("COLREG_R09_NOT_TO_IMPEDE_PASSAGE")
            has_not_to_impede_target = True

        trg_rules = unique_keep_order(trg_rules)
        aggregate_rules.extend(trg_rules)

        risk = bool(tcpa > 0 and tcpa <= 240.0 and cpa <= max(200.0, channel_half_width_m * 1.2))
        imminent = bool(tcpa > 0 and tcpa <= 60.0 and cpa <= max(80.0, channel_half_width_m * 0.6))

        target_summaries.append({
            "target_id": tid,
            "intended_role": role,
            "triggered_rule_ids": trg_rules,
            "relative_bearing_sector": sector,
            "cpa_m": round(cpa, 2),
            "tcpa_s": round(tcpa, 2),
            "risk_of_collision": risk,
            "closing": bool(tcpa > 0),
            "collision_imminent": imminent,
        })

    # Ensure family-level rule coverage is contract-preserving.
    if sem == "ordinary_crossing_in_channel" and "COLREG_R09_CROSSING_NARROW_CHANNEL" not in aggregate_rules:
        aggregate_rules.append("COLREG_R09_CROSSING_NARROW_CHANNEL")
    if sem == "keep_starboard_dominant" and "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT" not in aggregate_rules:
        aggregate_rules.append("COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT")
    if sem == "not_to_impede" and "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" not in aggregate_rules:
        aggregate_rules.append("COLREG_R09_NOT_TO_IMPEDE_PASSAGE")
    if sem == "mixed_semantics":
        if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" not in aggregate_rules:
            aggregate_rules.append("COLREG_R09_NOT_TO_IMPEDE_PASSAGE")
        if not has_r15:
            # fallback: only if a starboard crossing summary exists; do not force invalid R15.
            pass
        if not has_r14 and occ in {"opposing_flow", "third_vessel_blocking"}:
            # keep family-level head-on risk check for overlap semantics, even if no target got target-level R14.
            aggregate_rules.append("COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK")
            has_r14 = True

    aggregate_rules = unique_keep_order(aggregate_rules)
    allowed = unique_keep_order(allowed)
    forbidden = unique_keep_order(forbidden)

    primitive_ledger_summary = {
        "maneuver": {
            "CHANNEL_CONTEXT_CONTROL": {
                "polarities": ["allow"],
                "support_rules": [r for r in aggregate_rules if r.startswith("COLREG_R09") or r.startswith("COLREG_R08")],
            },
            "COLLISION_RISK_CONTROL": {
                "polarities": ["allow"],
                "support_rules": [r for r in aggregate_rules if r.startswith("COLREG_R05") or r.startswith("COLREG_R06") or r.startswith("COLREG_R07")],
            },
        },
        "family_contract": {
            "semantic_mode": sem,
            "occupancy_pattern": occ,
            "ownship_channel_position": pos,
            "observed_crossing_angle_band": observed_crossing_angle_band,
        },
    }

    explanation_steps = [
        {
            "stage": "area",
            "summary": "This sample is evaluated under narrow-channel context, so channel structure must remain causally active in the final explanation.",
        },
        {
            "stage": "channel",
            "summary": f"Rule 9 family obligations are active for semantic_mode={sem}, with ownship position {pos} and occupancy pattern {occ}.",
        },
        {
            "stage": "interaction",
            "summary": f"Observed crossing angle band is {observed_crossing_angle_band}; effective target count is {len(targets)} and channel interaction must not be reduced to open-water crossing only.",
        },
        {
            "stage": "action",
            "summary": "Final maneuver permissions and prohibitions preserve channel-dependent reasoning, crossing safety, and non-impeding behavior where required.",
        },
    ]

    labels = {
        "triggered_rules": aggregate_rules,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": primitive_ledger_summary,
        "suppressed_rules": [],
        "suppression_records": [],
        "explanation_steps": explanation_steps,
        "maneuver_allowed": allowed,
        "maneuver_forbidden": forbidden,
        "lights_required": [],
        "sounds_required": [],
    }

    self_check_errors: List[str] = []
    if len(target_summaries) != len(targets):
        self_check_errors.append("internal_target_summary_count_mismatch")
    if not any(r.startswith("COLREG_R09") for r in labels["triggered_rules"]):
        self_check_errors.append("internal_missing_rule9_family")
    if not any("channel" in str(x.get("summary", "")).lower() for x in explanation_steps if isinstance(x, dict)):
        self_check_errors.append("internal_explanation_not_channel_dependent")
    for ts in target_summaries:
        if "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in [str(x) for x in ts.get("triggered_rule_ids", [])]:
            if str(ts.get("relative_bearing_sector")) != "starboard":
                self_check_errors.append("internal_crossing_rule15_bearing_mismatch")
                break
    if sem == "keep_starboard_dominant":
        if "COLREG_R09_KEEP_NEAR_STARBOARD_LIMIT" not in labels["triggered_rules"]:
            self_check_errors.append("internal_keep_starboard_missing_rule")
        if "KEEP_TO_STARBOARD_WITHIN_CHANNEL" not in labels["maneuver_allowed"]:
            self_check_errors.append("internal_keep_starboard_missing_action")
    if sem == "not_to_impede":
        if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" not in labels["triggered_rules"] and "DO_NOT_IMPEDE_CHANNEL_TRAFFIC" not in labels["maneuver_allowed"]:
            self_check_errors.append("internal_not_to_impede_missing")
    if sem == "mixed_semantics":
        overlap_count = 0
        if "COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in labels["triggered_rules"]:
            overlap_count += 1
        if "COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK" in labels["triggered_rules"]:
            overlap_count += 1
        if "COLREG_R09_NOT_TO_IMPEDE_PASSAGE" in labels["triggered_rules"]:
            overlap_count += 1
        if overlap_count < 2:
            self_check_errors.append("internal_mixed_overlap_not_realized")

    aux = {
        "label_source": "built_in_v1",
        "generator_version_seen": row.get("generator_version"),
        "observed_crossing_angle_band": observed_crossing_angle_band,
        "effective_target_count": len(targets),
        "aggregate_rule_count": len(labels["triggered_rules"]),
        "has_r15_overlap": bool("COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD" in labels["triggered_rules"]),
        "has_r14_overlap": bool("COLREG_R14_HEAD_ON_OR_RECIPROCAL_RISK_CHECK" in labels["triggered_rules"]),
        "has_not_to_impede_overlap": bool("COLREG_R09_NOT_TO_IMPEDE_PASSAGE" in labels["triggered_rules"]),
    }
    return labels, aux, self_check_errors


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Label native candidates for channel_crossing_impede (authoritative built-in relabeling)."
    )
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="channel_crossing_impede", choices=["channel_crossing_impede"])
    ap.add_argument(
        "--mode",
        type=str,
        default="built_in",
        choices=["built_in", "adapter"],
        help="Labeling mode. built_in uses deterministic native relabeling in this script. adapter calls a project adapter function.",
    )
    ap.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional adapter spec in form module.submodule:function_name. Used when --mode adapter.",
    )
    ap.add_argument("--overwrite", action="store_true", help="Required because labels are rewritten in-place")
    ap.add_argument("--no-backup", action="store_true", help="Do not create candidates_raw.before_authoritative_labels.jsonl backup")
    ap.add_argument("--validate-schema", action="store_true")
    ap.add_argument("--report-json", type=str, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root)

    raw_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"
    schema_path = root / "schemas" / "raw_candidate.schema.json"

    if not raw_path.exists():
        print(f"ERROR: missing raw candidate file: {raw_path}", file=sys.stderr)
        return 2
    if not args.overwrite:
        print("ERROR: this version updates labels in place; pass --overwrite to proceed.", file=sys.stderr)
        return 2
    if args.validate_schema and not schema_path.exists():
        print(f"ERROR: missing schema file required by --validate-schema: {schema_path}", file=sys.stderr)
        return 2

    rows = read_jsonl(raw_path)
    schema = read_json(schema_path) if args.validate_schema and schema_path.exists() else {}

    adapter_fn = None
    if args.mode == "adapter":
        try:
            adapter_fn = load_adapter(args.adapter)
        except Exception as e:
            print(f"ERROR: failed to load adapter: {e}", file=sys.stderr)
            return 2

    backup_path = raw_path.with_name("candidates_raw.before_authoritative_labels.jsonl")
    if not args.no_backup:
        shutil.copy2(raw_path, backup_path)

    rewritten_rows: List[Dict[str, Any]] = []
    schema_errors: List[Dict[str, Any]] = []
    internal_contract_errors: List[Dict[str, Any]] = []
    built_in_count = 0
    adapter_count = 0

    for row in rows:
        new_row = copy.deepcopy(row)
        sample_id = new_row.get("sample_id", "<missing_sample_id>")
        old_labels = copy.deepcopy(new_row.get("labels", {}))

        if args.mode == "adapter":
            assert adapter_fn is not None
            labels = adapter_fn(new_row)
            if not isinstance(labels, dict):
                print(f"ERROR: adapter returned non-dict labels for {sample_id}", file=sys.stderr)
                return 2
            aux = {"label_source": "adapter"}
            self_check_errors: List[str] = []
            adapter_count += 1
        else:
            labels, aux, self_check_errors = authoritative_channel_crossing_impede_labels_v1(new_row)
            built_in_count += 1

        new_row["labels"] = labels
        debug = new_row.get("debug", {})
        if not isinstance(debug, dict):
            debug = {}
        debug["native_labeling"] = {
            "mode": args.mode,
            "previous_triggered_rules": old_labels.get("triggered_rules", []) if isinstance(old_labels, dict) else [],
            "previous_target_count": len(old_labels.get("target_summaries", [])) if isinstance(old_labels, dict) else 0,
            **aux,
        }
        new_row["debug"] = debug

        if self_check_errors:
            internal_contract_errors.append({
                "sample_id": sample_id,
                "errors": self_check_errors,
                "latent_semantic_mode": new_row.get("family_latent", {}).get("semantic_mode"),
                "latent_occupancy_pattern": new_row.get("family_latent", {}).get("occupancy_pattern"),
                "latent_ownship_channel_position": new_row.get("family_latent", {}).get("ownship_channel_position"),
                "written_triggered_rules": labels.get("triggered_rules", []),
            })

        if args.validate_schema:
            schema_err = validate_against_schema(new_row, schema)
            if schema_err is not None:
                schema_errors.append({"sample_id": sample_id, "error": schema_err})

        rewritten_rows.append(new_row)

    write_jsonl(raw_path, rewritten_rows)

    report = {
        "root": str(root),
        "family": FAMILY,
        "mode": args.mode,
        "adapter": args.adapter,
        "raw_file_updated": str(raw_path),
        "backup_file": None if args.no_backup else str(backup_path),
        "row_count": len(rows),
        "built_in_count": built_in_count,
        "adapter_count": adapter_count,
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_validation_enabled": bool(args.validate_schema),
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:20],
        "internal_contract_error_count": len(internal_contract_errors),
        "internal_contract_errors_head": internal_contract_errors[:20],
        "notes": (
            "This first version relabels native channel_crossing_impede raw candidates in place. "
            "It preserves Rule 9 channel dependence, avoids invalid R15 target-side refinement, "
            "and keeps mixed overlap semantics explicit in the final labels."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "label_native_candidates_channel_crossing_impede_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"label_native_candidates_channel_crossing_impede.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Rows relabeled           : {len(rows)}")
    print(f"Mode                     : {args.mode}")
    print(f"Backup created           : {False if args.no_backup else True}")
    print(f"Schema validation enabled: {bool(args.validate_schema)}")
    print(f"jsonschema available     : {HAS_JSONSCHEMA}")
    print(f"Schema error count       : {len(schema_errors)}")
    print(f"Internal contract errors : {len(internal_contract_errors)}")
    print(f"Updated raw file         : {raw_path}")
    print(f"Report                   : {report_path}")
    print("=" * 100)

    return 0 if (len(schema_errors) == 0 and len(internal_contract_errors) == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
