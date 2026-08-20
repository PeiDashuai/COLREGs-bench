from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from colreg_kernel.engine import evaluate_all
from colreg_kernel.features import Features
from colreg_kernel.rules import RuleSpec
from colreg_kernel.tri import Tri


def _evaluate_one(rule: RuleSpec, features: Features):
    # evaluate_all is the stable Part-1 API; wrap for single-rule evaluation.
    return evaluate_all({rule.id: rule}, features)[0]


class PairRelation(str, Enum):
    DISJOINT = "DISJOINT"
    COMPATIBLE = "COMPATIBLE"
    TENSION = "TENSION"
    CONFLICT = "CONFLICT"
    UNKNOWN = "UNKNOWN"  # evidence missing


class ConflictReason(str, Enum):
    APPLICABILITY = "APPLICABILITY"
    EVIDENCE_UNKNOWN = "EVIDENCE_UNKNOWN"
    MANEUVER_ACTIONS = "MANEUVER_ACTIONS"
    SIGNALS = "SIGNALS"
    DOMAIN = "DOMAIN"  # e.g., mutually exclusive regulatory domains (R34 vs R35)
    DEONTIC = "DEONTIC"  # reserved; should only be used for same-head primitive clashes


@dataclass(frozen=True)
class PairResult:
    rule_a: str
    rule_b: str
    relation: PairRelation
    reason: ConflictReason
    details: Dict[str, str]
    missing_fields: Tuple[str, ...] = ()


def _maneuver_sets(rule: RuleSpec) -> tuple[set[str], set[str]]:
    allowed = {c.primitive.value for c in rule.actions.maneuver.allowed}
    forbidden = {c.primitive.value for c in rule.actions.maneuver.forbidden}
    return allowed, forbidden


def _signals_sets(rule: RuleSpec) -> tuple[set[str], set[str], set[str], set[str]]:
    ls_req = {c.primitive.value for c in rule.actions.signals.lights_shapes_required}
    ls_forb = {c.primitive.value for c in rule.actions.signals.lights_shapes_forbidden}
    snd_req = {c.primitive.value for c in rule.actions.signals.sounds_required}
    snd_forb = {c.primitive.value for c in rule.actions.signals.sounds_forbidden}
    return ls_req, ls_forb, snd_req, snd_forb


def pair_relation(
    rule_a: RuleSpec,
    rule_b: RuleSpec,
    features: Features,
    *,
    assume_applicable_true: bool = False,
) -> PairResult:
    """
    Compute pairwise relation between two rules under given features.

    If assume_applicable_true=True, skips applicability evaluation (useful when caller
    already filtered to triggered rules).
    """
    missing = set()

    if not assume_applicable_true:
        ea = _evaluate_one(rule_a, features)
        eb = _evaluate_one(rule_b, features)

        if ea.value == Tri.UNKNOWN or eb.value == Tri.UNKNOWN:
            missing.update(ea.missing_fields or [])
            missing.update(eb.missing_fields or [])
            return PairResult(rule_a=rule_a.id, rule_b=rule_b.id, relation=PairRelation.UNKNOWN,
                              reason=ConflictReason.EVIDENCE_UNKNOWN,
                              details={"msg": "applicability unknown due to missing fields"},
                              missing_fields=tuple(sorted(missing)))
        if ea.value != Tri.TRUE or eb.value != Tri.TRUE:
            return PairResult(rule_a=rule_a.id, rule_b=rule_b.id, relation=PairRelation.DISJOINT,
                              reason=ConflictReason.APPLICABILITY,
                              details={"msg": "not co-applicable under given features"})

    # Domain-level mutual exclusivity: R34 (in sight maneuvering signals) vs R35 (restricted visibility signals)
    # If both are (co-)applicable under provided features (caller may force this), treat as a domain conflict.
    if ("R34" in rule_a.id and "R35" in rule_b.id) or ("R35" in rule_a.id and "R34" in rule_b.id):
        return PairResult(
            rule_a=rule_a.id,
            rule_b=rule_b.id,
            relation=PairRelation.CONFLICT,
            reason=ConflictReason.DOMAIN,
            details={"msg": "R34 and R35 belong to mutually exclusive sound-signal domains"},
        )

    a_allow, a_forb = _maneuver_sets(rule_a)
    b_allow, b_forb = _maneuver_sets(rule_b)

    # Maneuver conflicts: forbidding what the other allows (treated as a constraint conflict)
    if (a_forb & b_allow) or (b_forb & a_allow):
        return PairResult(rule_a=rule_a.id, rule_b=rule_b.id, relation=PairRelation.CONFLICT,
                          reason=ConflictReason.MANEUVER_ACTIONS,
                          details={
                              "a_forb_intersect_b_allow": ",".join(sorted(a_forb & b_allow)),
                              "b_forb_intersect_a_allow": ",".join(sorted(b_forb & a_allow)),
                          })

    ls_req_a, ls_forb_a, snd_req_a, snd_forb_a = _signals_sets(rule_a)
    ls_req_b, ls_forb_b, snd_req_b, snd_forb_b = _signals_sets(rule_b)

    if (ls_req_a & ls_forb_b) or (ls_req_b & ls_forb_a) or (snd_req_a & snd_forb_b) or (snd_req_b & snd_forb_a):
        return PairResult(rule_a=rule_a.id, rule_b=rule_b.id, relation=PairRelation.CONFLICT,
                          reason=ConflictReason.SIGNALS,
                          details={
                              "lights_conflict": ",".join(sorted((ls_req_a & ls_forb_b) | (ls_req_b & ls_forb_a))),
                              "sounds_conflict": ",".join(sorted((snd_req_a & snd_forb_b) | (snd_req_b & snd_forb_a))),
                          })

    # Tension (maneuver): constraints tighten action space without direct contradiction.
    base = (a_allow | b_allow)
    if base:
        feasible = base - (a_forb | b_forb)
        if feasible != base:
            # if we didn't classify as CONFLICT already, this is a tightening.
            return PairResult(
                rule_a=rule_a.id,
                rule_b=rule_b.id,
                relation=PairRelation.TENSION,
                reason=ConflictReason.MANEUVER_ACTIONS,
                details={
                    "msg": "constraints tighten maneuver action space",
                    "maneuver_union_size": str(len(base)),
                    "maneuver_feasible_size": str(len(feasible)),
                    "tension_shrink": str((len(feasible) / len(base)) if len(base) > 0 else 1.0),
                },
            )

    return PairResult(rule_a=rule_a.id, rule_b=rule_b.id, relation=PairRelation.COMPATIBLE,
                      reason=ConflictReason.APPLICABILITY, details={"msg": "compatible"})
