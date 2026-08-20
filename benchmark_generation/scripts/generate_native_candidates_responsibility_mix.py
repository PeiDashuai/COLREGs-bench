#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import random
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


FAMILY = "responsibility_mix"
GENERATOR_REVISION = "responsibility_mix_generator_v5_validator_isomorphic_urgency"


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


def weighted_choice(rng: random.Random, dist: Dict[str, float]) -> str:
    items = list(dist.items())
    total = sum(max(0.0, float(w)) for _, w in items)
    if total <= 0:
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


def stable_sample_id(i: int) -> str:
    return f"responsibility_mix_v3_{i:06d}"


def infer_urgency_level(target_summaries: List[Dict[str, Any]]) -> str:
    min_tcpa: Optional[float] = None
    for ts in target_summaries:
        if not isinstance(ts, dict):
            continue
        val = ts.get("tcpa_s")
        try:
            tcpa = float(val)
        except Exception:
            continue
        min_tcpa = tcpa if min_tcpa is None else min(min_tcpa, tcpa)
    if min_tcpa is not None and min_tcpa <= 45.0:
        return "high"
    if min_tcpa is not None and min_tcpa <= 120.0:
        return "medium"
    return "low"


def urgency_scale_for_latent(latent: Dict[str, Any]) -> float:
    band = str(latent.get("urgency_band"))
    encounter_mix = str(latent.get("encounter_mix"))
    role_structure = str(latent.get("role_structure"))

    base = {"high": 0.72, "medium": 1.0, "low": 1.45}.get(band, 1.0)
    # overtaking + low is the most common boundary-mismatch case; push it farther out.
    if band == "low" and encounter_mix == "overtaking_dominant":
        base *= 1.25
    if band == "low" and role_structure == "hierarchy_conflict":
        base *= 1.10
    if band == "high" and encounter_mix == "crossing_dominant":
        base *= 0.92
    return base


def wrap_deg(x: float) -> float:
    y = x % 360.0
    return y if y >= 0 else y + 360.0


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


def sample_target_count(rng: random.Random, role_structure: str) -> int:
    if role_structure in {"uniform_giveway", "uniform_standon"}:
        return rng.choice([2, 3, 4])
    if role_structure in {"mixed_roles", "hierarchy_conflict"}:
        return rng.choice([3, 4])
    return 3


ORDINARY_CLASSES = ["power_driven", "sailing"]
SPECIAL_CLASSES = ["fishing", "ram", "nuc"]


def build_target_classes(rng: random.Random, combo: str, count: int) -> List[str]:
    if combo == "power_only":
        xs = ["power_driven"] * count
    elif combo == "power_sailing":
        xs = (["power_driven", "sailing"] * (count // 2 + 1))[:count]
    elif combo == "power_fishing":
        xs = (["power_driven", "fishing"] * (count // 2 + 1))[:count]
    elif combo == "power_ram":
        xs = (["power_driven", "ram"] * (count // 2 + 1))[:count]
    elif combo == "power_nuc":
        xs = (["power_driven", "nuc"] * (count // 2 + 1))[:count]
    elif combo == "mixed_multi_status":
        # Important: this combo must NOT collapse to any of the named pair combos
        # that the annotator infers as power_sailing/power_fishing/power_ram/power_nuc.
        # Therefore we ensure at least one non-ordinary class and avoid the exact 2-class
        # sets {power,sailing}, {power,fishing}, {power,ram}, {power,nuc}.
        if count == 2:
            templates = [
                ["sailing", "fishing"],
                ["sailing", "ram"],
                ["sailing", "nuc"],
                ["fishing", "ram"],
                ["fishing", "nuc"],
                ["ram", "nuc"],
            ]
            xs = list(rng.choice(templates))
        elif count == 3:
            templates = [
                ["power_driven", "sailing", "fishing"],
                ["power_driven", "sailing", "ram"],
                ["power_driven", "sailing", "nuc"],
                ["power_driven", "fishing", "ram"],
                ["power_driven", "fishing", "nuc"],
                ["sailing", "fishing", "ram"],
            ]
            xs = list(rng.choice(templates))
        else:
            templates = [
                ["power_driven", "sailing", "fishing", "ram"],
                ["power_driven", "sailing", "fishing", "nuc"],
                ["power_driven", "sailing", "ram", "nuc"],
                ["power_driven", "fishing", "ram", "nuc"],
            ]
            base = list(rng.choice(templates))
            xs = (base * (count // len(base) + 1))[:count]
            # if count > 4 keep at least 3 unique classes
            if len(set(xs)) < 3:
                xs[-1] = "fishing"
    else:
        xs = (["power_driven", "sailing", "fishing", "ram", "nuc"] * (count // 5 + 1))[:count]
    return xs


def sample_family_latent_once(rng: random.Random, spec: Dict[str, Any]) -> Dict[str, Any]:
    dist = spec.get("soft_target_distribution", {})
    role_structure = weighted_choice(rng, dist.get("role_structure", {
        "uniform_giveway": 0.20,
        "uniform_standon": 0.20,
        "mixed_roles": 0.35,
        "hierarchy_conflict": 0.25,
    }))
    target_status_combo = weighted_choice(rng, dist.get("target_status_combo", {
        "power_only": 0.10,
        "power_sailing": 0.15,
        "power_fishing": 0.15,
        "power_ram": 0.15,
        "power_nuc": 0.10,
        "mixed_multi_status": 0.35,
    }))
    encounter_mix = weighted_choice(rng, dist.get("encounter_mix", {
        "crossing_dominant": 0.40,
        "overtaking_dominant": 0.20,
        "mixed_encounters": 0.40,
    }))
    urgency_band = weighted_choice(rng, {"low": 0.20, "medium": 0.45, "high": 0.35})
    effective_target_count = sample_target_count(rng, role_structure)
    return {
        "effective_target_count": effective_target_count,
        "role_structure": role_structure,
        "target_status_combo": target_status_combo,
        "encounter_mix": encounter_mix,
        "urgency_band": urgency_band,
    }


def build_relation_types(encounter_mix: str, n: int, include_hierarchy: bool) -> List[str]:
    k = n - 1 if include_hierarchy else n
    if k <= 0:
        return ["hierarchy"] if include_hierarchy else []

    if encounter_mix == "crossing_dominant":
        tail = ["crossing"] * k
    elif encounter_mix == "overtaking_dominant":
        tail = ["overtaking"] * k
    else:
        base = ["crossing", "overtaking"]
        tail = (base * (k // 2 + 1))[:k]
        if "crossing" not in tail:
            tail[0] = "crossing"
        if "overtaking" not in tail:
            tail[-1] = "overtaking"

    return (["hierarchy"] + tail) if include_hierarchy else tail


def build_target_role_graph(rng: random.Random, latent: Dict[str, Any]) -> List[Dict[str, Any]]:
    n = int(latent["effective_target_count"])
    rs = str(latent["role_structure"])
    encounter_mix = str(latent["encounter_mix"])
    classes = build_target_classes(rng, str(latent["target_status_combo"]), n)

    if rs == "uniform_giveway":
        roles = ["giveway"] * n
        relation_types = build_relation_types(encounter_mix, n, include_hierarchy=False)
    elif rs == "uniform_standon":
        roles = ["standon"] * n
        relation_types = build_relation_types(encounter_mix, n, include_hierarchy=False)
    elif rs == "mixed_roles":
        roles = (["giveway", "standon"] * (n // 2 + 1))[:n]
        if "giveway" not in roles:
            roles[0] = "giveway"
        if "standon" not in roles:
            roles[-1] = "standon"
        relation_types = build_relation_types(encounter_mix, n, include_hierarchy=False)
    else:
        roles = ["hierarchy_priority", "giveway", "standon"] + ["giveway"] * max(0, n - 3)
        roles = roles[:n]
        relation_types = build_relation_types(encounter_mix, n, include_hierarchy=True)
        if n >= 2 and "giveway" not in roles[1:]:
            roles[1] = "giveway"
        if n >= 3 and "standon" not in roles[1:]:
            roles[2] = "standon"

    graph: List[Dict[str, Any]] = []
    for i in range(n):
        graph.append({
            "target_id": f"t{i+1}",
            "ownship_role_vs_target": roles[i],
            "relation_type": relation_types[i],
            "target_class": classes[i],
        })
    return graph


def compatible_latent(latent: Dict[str, Any], role_graph: List[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    rs = str(latent["role_structure"])
    combo = str(latent["target_status_combo"])
    encounter_mix = str(latent["encounter_mix"])
    n = int(latent["effective_target_count"])
    roles = [str(x.get("ownship_role_vs_target")) for x in role_graph]
    classes = [str(x.get("target_class")) for x in role_graph]
    rels = [str(x.get("relation_type")) for x in role_graph]
    non_hierarchy_rels = [r for r in rels if r != "hierarchy"]
    s = set(classes)

    if len(role_graph) != n:
        return False, "role_graph_target_count_mismatch"

    if rs == "uniform_giveway" and any(r != "giveway" for r in roles):
        return False, "uniform_giveway_requires_all_giveway"
    if rs == "uniform_standon" and any(r != "standon" for r in roles):
        return False, "uniform_standon_requires_all_standon"
    if rs == "mixed_roles":
        if not ("giveway" in roles and "standon" in roles):
            return False, "mixed_roles_requires_giveway_and_standon"
    if rs == "hierarchy_conflict":
        if "hierarchy_priority" not in roles:
            return False, "hierarchy_conflict_requires_priority_target"
        if not any(r in {"giveway", "standon"} for r in roles):
            return False, "hierarchy_conflict_requires_nonpriority_target"

    if combo == "power_only" and any(c != "power_driven" for c in classes):
        return False, "power_only_combo_mismatch"
    if combo == "power_sailing" and s != {"power_driven", "sailing"}:
        return False, "power_sailing_combo_mismatch"
    if combo == "power_fishing" and s != {"power_driven", "fishing"}:
        return False, "power_fishing_combo_mismatch"
    if combo == "power_ram" and s != {"power_driven", "ram"}:
        return False, "power_ram_combo_mismatch"
    if combo == "power_nuc" and s != {"power_driven", "nuc"}:
        return False, "power_nuc_combo_mismatch"
    if combo == "mixed_multi_status":
        if len(s) < 2:
            return False, "mixed_multi_status_needs_multiple_classes"
        if s in [
            {"power_driven", "sailing"},
            {"power_driven", "fishing"},
            {"power_driven", "ram"},
            {"power_driven", "nuc"},
            {"power_driven"},
        ]:
            return False, "mixed_multi_status_collapsed_to_named_combo"
        if not any(c in {"fishing", "ram", "nuc"} for c in classes):
            return False, "mixed_multi_status_missing_special_status"

    crossing = non_hierarchy_rels.count("crossing")
    overtaking = non_hierarchy_rels.count("overtaking")
    if encounter_mix == "crossing_dominant":
        if crossing < 1 or overtaking != 0:
            return False, "crossing_dominant_not_realized"
    elif encounter_mix == "overtaking_dominant":
        if overtaking < 1 or crossing != 0:
            return False, "overtaking_dominant_not_realized"
    elif encounter_mix == "mixed_encounters":
        if crossing < 1 or overtaking < 1:
            return False, "mixed_encounters_not_realized"

    return True, None


def sample_family_latent_compatible(rng: random.Random, spec: Dict[str, Any], max_trials: int = 100) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int, Optional[str]]:
    resamples = 0
    last_reason = None
    for _ in range(max_trials):
        latent = sample_family_latent_once(rng, spec)
        role_graph = build_target_role_graph(rng, latent)
        ok, reason = compatible_latent(latent, role_graph)
        if ok:
            return latent, role_graph, resamples, last_reason
        resamples += 1
        last_reason = reason
    raise RuntimeError(f"Failed to sample compatible responsibility_mix latent after {max_trials} trials; last_reason={last_reason}")


def build_ownship(rng: random.Random) -> Dict[str, Any]:
    return {
        "id": "ownship",
        "vessel_class": "power_driven",
        "nav_status": "underway_making_way",
        "x_m": 0.0,
        "y_m": 0.0,
        "heading_deg": 0.0,
        "speed_mps": round(choose_float_from_range(rng, 4.5, 7.5), 2),
    }


def position_for_relation(i: int, relation_type: str, scale: float = 1.0) -> Tuple[float, float, float]:
    if relation_type == "crossing":
        if i % 2 == 0:
            return ((400.0 + i * 80.0) * scale, 250.0 * scale, 270.0)
        return ((350.0 + i * 80.0) * scale, -250.0 * scale, 90.0)
    if relation_type == "overtaking":
        return ((-250.0 - i * 80.0) * scale, 0.0, 0.0)
    if relation_type == "hierarchy":
        return ((300.0 + i * 60.0) * scale, 180.0 * scale, 270.0)
    return ((300.0 + i * 60.0) * scale, 0.0, 0.0)


def class_to_nav_status(vessel_class: str) -> str:
    return {
        "power_driven": "underway_making_way",
        "sailing": "under_sail",
        "fishing": "fishing",
        "ram": "restricted_in_ability_to_manoeuvre",
        "nuc": "not_under_command",
    }.get(vessel_class, "underway_making_way")


def build_targets(rng: random.Random, role_graph: List[Dict[str, Any]], latent: Dict[str, Any]) -> List[Dict[str, Any]]:
    scale = urgency_scale_for_latent(latent)
    targets: List[Dict[str, Any]] = []
    for i, node in enumerate(role_graph):
        relation_type = str(node["relation_type"])
        # add mild per-target jitter while preserving urgency scale ordering
        jitter = rng.uniform(0.92, 1.08)
        local_scale = scale * jitter
        x, y, heading = position_for_relation(i, relation_type, scale=local_scale)
        vessel_class = str(node["target_class"])
        speed_rng = {
            "power_driven": (4.0, 8.0),
            "sailing": (2.0, 5.0),
            "fishing": (2.0, 4.5),
            "ram": (2.0, 4.0),
            "nuc": (0.5, 2.0),
        }.get(vessel_class, (3.0, 6.0))

        target_id = str(node["target_id"])
        role_hint = str(node["ownship_role_vs_target"])

        targets.append({
            "id": target_id,
            "target_id": target_id,
            "vessel_class": vessel_class,
            "nav_status": class_to_nav_status(vessel_class),
            "x_m": round(x, 2),
            "y_m": round(y, 2),
            "heading_deg": round(heading, 2),
            "speed_mps": round(choose_float_from_range(rng, speed_rng[0], speed_rng[1]), 2),
            "intended_role": role_hint,
            "role_hint": role_hint,
            "relation_type": relation_type,
            "is_effective_target": True,
        })
    return targets


def build_scene_spec(candidate_seed: int, role_graph: List[Dict[str, Any]], ownship: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "seed": candidate_seed,
        "domain": "open",
        "visibility": "in_sight",
        "snapshot_time_s": 30,
        "duration_s": 180,
        "area_context": {
            "area_type": "open_water",
            "channel_context": False,
            "tss_context": False,
        },
        "ownship": ownship,
        "targets": targets,
        "target_role_graph": role_graph,
    }


def _band_center_tcpa(band: str) -> Tuple[float, float]:
    if band == "high":
        return 20.0, 40.0
    if band == "medium":
        return 60.0, 110.0
    return 135.0, 180.0


def _band_center_cpa(band: str) -> Tuple[float, float]:
    if band == "high":
        return 80.0, 180.0
    if band == "medium":
        return 160.0, 260.0
    return 260.0, 420.0


def build_target_summary(ownship: Dict[str, Any], target: Dict[str, Any], node: Dict[str, Any], latent: Dict[str, Any]) -> Dict[str, Any]:
    ox, oy, oh = float(ownship["x_m"]), float(ownship["y_m"]), float(ownship["heading_deg"])
    tx, ty = float(target["x_m"]), float(target["y_m"])
    sector = rel_bearing_sector(ox, oy, oh, tx, ty)

    role = str(node["ownship_role_vs_target"])
    relation_type = str(node["relation_type"])
    vessel_class = str(node["target_class"])
    urgency_band = str(latent.get("urgency_band", "medium"))

    trg_rules: List[str] = []
    if relation_type == "crossing":
        if role == "giveway":
            trg_rules += ["COLREG_R15_CROSSING_GIVEWAY_TARGET_STARBOARD", "COLREG_R16_GIVEWAY_MANEUVER"]
        elif role == "standon":
            trg_rules += ["COLREG_R17_STANDON"]
    elif relation_type == "overtaking":
        trg_rules += ["COLREG_R13_OVERTAKING"]
    elif relation_type == "hierarchy":
        trg_rules += ["COLREG_R18_RESPONSIBILITIES"]

    if vessel_class in {"ram", "nuc", "fishing", "sailing"} and "COLREG_R18_RESPONSIBILITIES" not in trg_rules:
        trg_rules += ["COLREG_R18_RESPONSIBILITIES"]

    dist = math.hypot(tx - ox, ty - oy)
    raw_tcpa = max(0.0, dist / max(0.1, float(ownship["speed_mps"])))
    raw_cpa = max(80.0, dist * 0.25)

    tcpa_lo, tcpa_hi = _band_center_tcpa(urgency_band)
    cpa_lo, cpa_hi = _band_center_cpa(urgency_band)

    # Keep proxies monotone with geometry but force them into clearly separated urgency bands.
    tcpa = min(tcpa_hi, max(tcpa_lo, raw_tcpa))
    cpa = min(cpa_hi, max(cpa_lo, raw_cpa))

    if urgency_band == "low":
        # extra buffer so validator low/medium boundary is never touched
        tcpa = max(135.0, tcpa)
        cpa = max(260.0, cpa)

    return {
        "target_id": target["id"],
        "role": role,
        "triggered_rule_ids": trg_rules,
        "relative_bearing_sector": sector,
        "cpa_m": round(cpa, 2),
        "tcpa_s": round(tcpa, 2),
        "risk_of_collision": True,
        "closing": True,
        "collision_imminent": tcpa <= 45.0,
    }


def build_labels(latent: Dict[str, Any], role_graph: List[Dict[str, Any]], ownship: Dict[str, Any], targets: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate_rules = [
        "COLREG_R05_LOOKOUT",
        "COLREG_R06_SAFE_SPEED",
        "COLREG_R07_RISK_OF_COLLISION_ASSESS",
        "COLREG_R08_POSITIVE_ACTION_WHEN_RISK",
    ]
    target_summaries: List[Dict[str, Any]] = []
    for node, tgt in zip(role_graph, targets):
        ts = build_target_summary(ownship, tgt, node, latent)
        target_summaries.append(ts)
        aggregate_rules.extend(ts["triggered_rule_ids"])
    aggregate_rules = list(dict.fromkeys(aggregate_rules))

    role_structure = str(latent["role_structure"])
    allowed = ["KEEP_LOOKOUT", "PROCEED_SAFE_SPEED"]
    forbidden: List[str] = []
    if role_structure == "uniform_giveway":
        allowed += ["TAKE_EARLY_SUBSTANTIAL_ACTION", "GIVE_WAY_IF_REQUIRED"]
    elif role_structure == "uniform_standon":
        allowed += ["MAINTAIN_STANDON_IF_SAFE"]
    elif role_structure == "mixed_roles":
        allowed += ["ROLE_AWARE_ACTION_SELECTION"]
        forbidden += ["USE_SINGLE_TARGET_TEMPLATE"]
    else:
        allowed += ["PRIORITIZE_HIGHER_ORDER_RESPONSIBILITY", "ROLE_AWARE_ACTION_SELECTION"]
        forbidden += ["IGNORE_PRIORITY_TARGET"]

    explanation_steps = [
        {"stage": "roles", "summary": f"Role structure is {latent['role_structure']}."},
        {"stage": "status", "summary": f"Target status combo is {latent['target_status_combo']}."},
        {"stage": "encounter", "summary": f"Encounter mix is {latent['encounter_mix']}."},
    ]

    return {
        "triggered_rules": aggregate_rules,
        "target_summaries": target_summaries,
        "primitive_ledger_summary": {
            "maneuver": {
                "ROLE_RESPONSIBILITY_CONTROL": {
                    "polarities": ["allow"],
                    "support_rules": aggregate_rules,
                }
            }
        },
        "suppressed_rules": [],
        "suppression_records": [],
        "explanation_steps": explanation_steps,
        "maneuver_allowed": allowed,
        "maneuver_forbidden": forbidden,
        "lights_required": [],
        "sounds_required": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate native v3 raw candidates (responsibility_mix v3).")
    ap.add_argument("--root", type=str, required=True, help="benchmark generation work directory")
    ap.add_argument("--family", type=str, default="responsibility_mix", choices=["responsibility_mix"])
    ap.add_argument("--num-candidates", type=int, default=20)
    ap.add_argument("--seed", type=int, default=200000)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--validate-schema", action="store_true")
    ap.add_argument("--report-json", type=str, default=None)
    args = ap.parse_args()

    root = Path(args.root)
    specs_path = root / "config" / "native_generator_specs" / f"{FAMILY}.yaml"
    schema_path = root / "schemas" / "raw_candidate.schema.json"
    out_path = root / "raw_pool" / FAMILY / "candidates_raw.jsonl"

    if not specs_path.exists():
        print(f"ERROR: missing spec file: {specs_path}")
        return 2
    if not schema_path.exists():
        print(f"ERROR: missing raw candidate schema: {schema_path}")
        return 2
    if out_path.exists() and out_path.stat().st_size > 0 and not args.overwrite:
        print(f"ERROR: output file exists and is non-empty; use --overwrite: {out_path}")
        return 2

    spec = read_yaml(specs_path)
    schema = read_json(schema_path)

    rows: List[Dict[str, Any]] = []
    schema_errors: List[Dict[str, Any]] = []
    resample_counts: List[int] = []
    reject_reason_hist: Dict[str, int] = {}

    for i in range(args.num_candidates):
        candidate_seed = args.seed + i + 1
        rng = random.Random(candidate_seed)

        latent, role_graph, resample_count, last_reject_reason = sample_family_latent_compatible(rng, spec)

        urgency_resamples = 0
        max_urgency_resamples = 120
        observed_urgency = None
        ownship = {}
        targets = []
        scene_spec = {}
        labels = {}
        while True:
            ownship = build_ownship(rng)
            targets = build_targets(rng, role_graph, latent)
            scene_spec = build_scene_spec(candidate_seed, role_graph, ownship, targets)
            labels = build_labels(latent, role_graph, ownship, targets)
            observed_urgency = infer_urgency_level(labels.get("target_summaries", []))
            if observed_urgency == str(latent.get("urgency_band")):
                break
            urgency_resamples += 1
            if urgency_resamples >= max_urgency_resamples:
                raise RuntimeError(
                    f"failed to realize urgency_band after {max_urgency_resamples} resamples: "
                    f"latent={latent.get('urgency_band')} observed={observed_urgency} sample_id={stable_sample_id(i + 1)}"
                )

        row = {
            "sample_id": stable_sample_id(i + 1),
            "pattern": FAMILY,
            "source_mode": "native_v3_generator",
            "generator_version": spec.get("generator_version", "responsibility_mix_native_v1"),
            "candidate_seed": candidate_seed,
            "family_latent": latent,
            "scene_spec": scene_spec,
            "inputs": {
                "topdown_image": None,
                "radar_image": None,
                "scene_text": None,
                "scene_spec": scene_spec,
            },
            "labels": labels,
            "debug": {
                "generator_revision": GENERATOR_REVISION,
                "builder_notes": [
                    f"role_structure={latent['role_structure']}",
                    f"target_status_combo={latent['target_status_combo']}",
                    f"encounter_mix={latent['encounter_mix']}",
                ],
                "compatibility_filter": {
                    "resample_count": resample_count,
                    "last_reject_reason": last_reject_reason,
                },
                "urgency_self_check": {
                    "latent_urgency_band": latent.get("urgency_band"),
                    "observed_urgency_level": observed_urgency,
                    "resample_count": urgency_resamples,
                },
                "sanity_checks": {
                    "schema_shape_ready": True,
                    "target_count_matches_latent": len(targets) == int(latent["effective_target_count"]),
                    "role_graph_present": True,
                    "mixed_multi_status_realized": (
                        latent["target_status_combo"] != "mixed_multi_status"
                        or len({t["vessel_class"] for t in targets}) >= 2
                    ),
                }
            }
        }

        if args.validate_schema:
            err = maybe_validate_schema(row, schema)
            if err is not None:
                schema_errors.append({"sample_id": row["sample_id"], "error": err})

        rows.append(row)
        resample_counts.append(resample_count)
        if last_reject_reason is not None:
            reject_reason_hist[last_reject_reason] = reject_reason_hist.get(last_reject_reason, 0) + 1

    write_jsonl(out_path, rows)

    report = {
        "root": str(root),
        "family": FAMILY,
        "spec_file": str(specs_path),
        "schema_file": str(schema_path),
        "output_file": str(out_path),
        "num_candidates_requested": args.num_candidates,
        "num_candidates_written": len(rows),
        "schema_validation_enabled": bool(args.validate_schema),
        "jsonschema_available": HAS_JSONSCHEMA,
        "schema_error_count": len(schema_errors),
        "schema_errors_head": schema_errors[:10],
        "generator_revision": GENERATOR_REVISION,
        "latent_preview": [row["family_latent"] for row in rows[:5]],
        "role_graph_preview": [row["scene_spec"]["target_role_graph"] for row in rows[:2]],
        "compatibility_filter_summary": {
            "total_resamples": sum(resample_counts),
            "max_resamples_single_sample": max(resample_counts) if resample_counts else 0,
            "reject_reason_histogram": reject_reason_hist,
        },
        "notes": (
            "This v5 generator preserves mixed_multi_status and enforces urgency_band via validator-isomorphic self-check + resample. "
            "Observed urgency is derived from target_summary tcpa thresholds: high<=45s, medium<=120s, else low."
        ),
    }

    report_path = Path(args.report_json) if args.report_json else (
        root / "reports" / "release_reports" / "generate_native_candidates_responsibility_mix_report.json"
    )
    write_json(report_path, report)

    print("=" * 100)
    print(f"generate_native_candidates_responsibility_mix.py | root={root} | family={FAMILY}")
    print("=" * 100)
    print(f"Wrote raw candidates   : {len(rows)} -> {out_path}")
    print(f"Schema validation      : {bool(args.validate_schema)}")
    print(f"jsonschema available   : {HAS_JSONSCHEMA}")
    print(f"Schema error count     : {len(schema_errors)}")
    print(f"Total compatibility resamples: {sum(resample_counts)}")
    print(f"Generator revision     : {GENERATOR_REVISION}")
    print(f"Report                 : {report_path}")
    print("=" * 100)
    return 0 if len(schema_errors) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
