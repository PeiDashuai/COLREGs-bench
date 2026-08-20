## `colreg_reasoner/src/colreg_reasoner/global_ledger.py`


from __future__ import annotations

from typing import Dict, List

from colreg_kernel.aggregate import AggregateResult
from colreg_kernel.features import Features
from colreg_reasoner.priority import PriorityPolicy, rank_rule
from colreg_kernel.rules import RuleSpec

from .global_types import ConstraintAtom, GlobalPrimitiveLedger, TargetContext


def _to_opt_str(x):
    if x is None:
        return None
    return getattr(x, "value", x)


def build_constraint_atoms_from_aggregate(
    target_ctx: TargetContext,
    *,
    rules: Dict[str, RuleSpec],
    policy: PriorityPolicy,
) -> List[ConstraintAtom]:
    feats: Features = target_ctx.features
    agg: AggregateResult = target_ctx.local_aggregate
    out: List[ConstraintAtom] = []

    def _mk(rule_id: str, head: str, primitive: str, polarity: str) -> ConstraintAtom:
        rule = rules[rule_id]
        atom_id = f"{target_ctx.target_id}:{rule_id}:{head}:{polarity}:{primitive}"
        return ConstraintAtom(
            atom_id=atom_id,
            target_id=target_ctx.target_id,
            target_index=target_ctx.target_index,
            rule_id=rule_id,
            head=head,
            primitive=primitive,
            polarity=polarity,
            priority_bucket=rank_rule(rule, policy),
            article=rule.article,
            visibility=_to_opt_str(feats.visibility),
            in_sight=feats.in_sight,
            encounter_type=_to_opt_str(feats.encounter_type),
            relative_bearing_sector=_to_opt_str(feats.relative_bearing_sector),
            is_target_on_starboard=feats.is_target_on_starboard,
            closing=feats.closing,
            cpa_m=feats.cpa_m,
            tcpa_s=feats.tcpa_s,
            risk_of_collision=feats.risk_of_collision,
            collision_imminent=feats.collision_imminent,
            reason_basis={"article": rule.article, "summary": rule.summary},
            evidence_refs={
                "target_id": target_ctx.target_id,
                "snapshot": target_ctx.debug_meta.get("snapshot_index"),
            },
        )

    for primitive, rule_ids in agg.labels.maneuver_allowed.items():
        for rid in rule_ids:
            out.append(_mk(rid, "maneuver", primitive, "allow"))
    for primitive, rule_ids in agg.labels.maneuver_forbidden.items():
        for rid in rule_ids:
            out.append(_mk(rid, "maneuver", primitive, "forbid"))
    for primitive, rule_ids in agg.labels.lights_required.items():
        for rid in rule_ids:
            out.append(_mk(rid, "lights", primitive, "require"))
    for primitive, rule_ids in agg.labels.lights_forbidden.items():
        for rid in rule_ids:
            out.append(_mk(rid, "lights", primitive, "forbid"))
    for primitive, rule_ids in agg.labels.sounds_required.items():
        for rid in rule_ids:
            out.append(_mk(rid, "sounds", primitive, "require"))
    for primitive, rule_ids in agg.labels.sounds_forbidden.items():
        for rid in rule_ids:
            out.append(_mk(rid, "sounds", primitive, "forbid"))

    return out


def build_global_primitive_ledger(atoms: List[ConstraintAtom]) -> GlobalPrimitiveLedger:
    man: Dict[str, List[ConstraintAtom]] = {}
    lights: Dict[str, List[ConstraintAtom]] = {}
    sounds: Dict[str, List[ConstraintAtom]] = {}
    for atom in atoms:
        dst = man if atom.head == "maneuver" else lights if atom.head == "lights" else sounds
        dst.setdefault(atom.primitive, []).append(atom)

    def _freeze(d):
        return {
            k: tuple(
                sorted(
                    v,
                    key=lambda a: (-a.priority_bucket, a.rule_id, a.target_id, a.atom_id),
                )
            )
            for k, v in sorted(d.items())
        }

    return GlobalPrimitiveLedger(
        maneuver=_freeze(man),
        lights=_freeze(lights),
        sounds=_freeze(sounds),
    )


def summarize_primitive_ledger(ledger: GlobalPrimitiveLedger) -> Dict[str, dict]:
    def _summ(head_map: Dict[str, tuple[ConstraintAtom, ...]]) -> Dict[str, dict]:
        out = {}
        for prim, atoms in sorted(head_map.items()):
            out[prim] = {
                "num_atoms": len(atoms),
                "targets": sorted({a.target_id for a in atoms}),
                "polarities": sorted({a.polarity for a in atoms}),
                "max_priority": max((a.priority_bucket for a in atoms), default=0),
            }
        return out

    return {
        "maneuver": _summ(ledger.maneuver),
        "lights": _summ(ledger.lights),
        "sounds": _summ(ledger.sounds),
    }


def compute_target_distance_m(geometry) -> float | None:
    try:
        if hasattr(geometry, "distance_m") and geometry.distance_m is not None:
            return float(geometry.distance_m)
    except Exception:
        pass
    return None
