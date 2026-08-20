from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from colreg_kernel.features import Features
from colreg_kernel.rules import RuleSpec

from colreg_reasoner.pairwise import pair_relation, PairRelation, ConflictReason


@dataclass(frozen=True)
class TensionMetrics:
    num_rules: int
    num_pairs: int
    num_conflicts: int
    num_conflicts_maneuver: int
    num_conflicts_signals: int
    num_conflicts_domain: int
    num_tensions: int
    num_tensions_maneuver: int
    num_compatible: int
    # simple scalar for curriculum: conflicts*2 + tensions (domain conflict counts as conflict)
    complexity_score: float
    # Per pair shrink ratio (only for maneuver tensions), key "a|b"
    tension_shrink: Dict[str, float]


def compute_tension_metrics(rules: Sequence[RuleSpec], features: Features) -> TensionMetrics:
    rlist = sorted(list(rules), key=lambda r: r.id)
    n = len(rlist)
    num_pairs = n * (n - 1) // 2
    conflicts = tensions = compatibles = 0
    conf_m = conf_s = conf_d = 0
    ten_m = 0
    shrink: Dict[str, float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            ra, rb = rlist[i], rlist[j]
            pr = pair_relation(ra, rb, features, assume_applicable_true=False)
            if pr.relation == PairRelation.CONFLICT:
                conflicts += 1
                if pr.reason == ConflictReason.MANEUVER_ACTIONS:
                    conf_m += 1
                elif pr.reason == ConflictReason.SIGNALS:
                    conf_s += 1
                elif pr.reason == ConflictReason.DOMAIN:
                    conf_d += 1
            elif pr.relation == PairRelation.TENSION:
                tensions += 1
                if pr.reason == ConflictReason.MANEUVER_ACTIONS:
                    ten_m += 1
                    # tension_shrink stored as string in details
                    ts = pr.details.get("tension_shrink")
                    try:
                        val = float(ts) if ts is not None else None
                        if val is not None:
                            shrink[f"{ra.id}|{rb.id}"] = val
                    except Exception:
                        pass
            elif pr.relation == PairRelation.COMPATIBLE:
                compatibles += 1
    complexity = float(conflicts * 2 + tensions)
    return TensionMetrics(
        num_rules=n,
        num_pairs=num_pairs,
        num_conflicts=conflicts,
        num_conflicts_maneuver=conf_m,
        num_conflicts_signals=conf_s,
        num_conflicts_domain=conf_d,
        num_tensions=tensions,
        num_tensions_maneuver=ten_m,
        num_compatible=compatibles,
        complexity_score=complexity,
        tension_shrink=shrink,
    )
