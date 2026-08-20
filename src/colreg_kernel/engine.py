from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .features import Features
from .predicate import eval_predicate
from .rules import RuleSpec
from .tri import Tri, TriValue


@dataclass(frozen=True)
class RuleEvalResult:
    rule_id: str
    value: Tri
    missing_fields: tuple[str, ...] = ()


def applicable(rule: RuleSpec, features: Features) -> TriValue:
    return eval_predicate(rule.applicability, features)


def evaluate_all(rules: Dict[str, RuleSpec], features: Features) -> List[RuleEvalResult]:
    out: List[RuleEvalResult] = []
    for rid, rule in rules.items():
        tv = applicable(rule, features)
        out.append(RuleEvalResult(rule_id=rid, value=tv.value, missing_fields=tv.missing_fields))
    return out
