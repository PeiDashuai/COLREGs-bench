
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .engine import evaluate_all
from .features import Features
from .rules import RuleSpec, load_rules
from .actions import ManeuverPrimitive, LightShapePrimitive, SoundPrimitive
from .validator import validate_feature_consistency, validate_signals_outputs, ValidationIssue


@dataclass(frozen=True)
class TriggeredRule:
    rule_id: str
    article: str
    summary: str
    text_ref: str
    evidence_requirements: Tuple[str, ...]


@dataclass(frozen=True)
class LabelsWithSources:
    # primitive -> tuple(rule_id, ...)
    maneuver_allowed: Dict[str, Tuple[str, ...]]
    maneuver_forbidden: Dict[str, Tuple[str, ...]]
    lights_required: Dict[str, Tuple[str, ...]]
    lights_forbidden: Dict[str, Tuple[str, ...]]
    sounds_required: Dict[str, Tuple[str, ...]]
    sounds_forbidden: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class AggregateResult:
    triggered_rules: Tuple[TriggeredRule, ...]
    labels: LabelsWithSources
    explanation_chain: Tuple[str, ...]
    unknown_rules: Tuple[Tuple[str, Tuple[str, ...]], ...]  # (rule_id, missing_fields)
    sanity_issues: Tuple[ValidationIssue, ...]
    signals_issues: Tuple[ValidationIssue, ...]


def _repo_root() -> Path:
    # .../src/colreg_kernel/<file>.py -> parents[3] == repo root (colreg_kernel/)
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=8)
def _load_rules_cached(rules_path: str) -> Dict[str, RuleSpec]:
    return load_rules(rules_path)


@lru_cache(maxsize=8)
def _load_params_cached(params_path: str) -> dict:
    p = Path(params_path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def default_rules_path() -> str:
    return str(_repo_root() / "rules" / "colregs_atomic.yaml")


def default_params_path() -> str:
    return str(_repo_root() / "rules" / "operational_params.yaml")


def _add_source(map_: Dict[str, List[str]], prim: str, rule_id: str) -> None:
    map_.setdefault(prim, [])
    if rule_id not in map_[prim]:
        map_[prim].append(rule_id)


def aggregate(
    features: Features,
    *,
    rules_path: Optional[str] = None,
    params_path: Optional[str] = None,
    strict: bool = True,
) -> AggregateResult:
    """
    Single entrypoint for downstream modules.

    Returns:
      - triggered rules (Tri==True)
      - maneuver head labels (allowed/forbidden) with sources
      - signals head labels (lights/shapes + sounds) with sources
      - explanation chain (human readable, deterministic order)
      - unknown rules (Tri==Unknown with missing fields)
      - sanity issues + signals legality issues
    """
    rules_path = rules_path or default_rules_path()
    params_path = params_path or default_params_path()

    sanity = tuple(validate_feature_consistency(features))
    if strict and sanity:
        raise ValueError("Feature sanity validation failed: " + "; ".join(i.code for i in sanity))

    rules = _load_rules_cached(rules_path)

    evals = evaluate_all(rules, features)
    triggered: List[TriggeredRule] = []
    unknowns: List[Tuple[str, Tuple[str, ...]]] = []

    # label sources
    man_allowed: Dict[str, List[str]] = {}
    man_forbidden: Dict[str, List[str]] = {}
    ls_req: Dict[str, List[str]] = {}
    ls_forb: Dict[str, List[str]] = {}
    snd_req: Dict[str, List[str]] = {}
    snd_forb: Dict[str, List[str]] = {}

    # deterministic ordering: by article, then rule_id
    for reval in sorted(evals, key=lambda r: (rules[r.rule_id].article, r.rule_id)):
        rule = rules[reval.rule_id]
        from .tri import Tri

        if reval.value == Tri.UNKNOWN:
            if reval.missing_fields:
                unknowns.append((reval.rule_id, tuple(reval.missing_fields)))
            continue
        if reval.value != Tri.TRUE:
            continue

        triggered.append(
            TriggeredRule(
                rule_id=rule.id,
                article=rule.article,
                summary=rule.summary,
                text_ref=rule.text_ref,
                evidence_requirements=tuple(rule.evidence_requirements),
            )
        )

        # maneuver labels
        for c in rule.actions.maneuver.allowed:
            _add_source(man_allowed, c.primitive.value, rule.id)
        for c in rule.actions.maneuver.forbidden:
            _add_source(man_forbidden, c.primitive.value, rule.id)

        # signals labels
        for c in rule.actions.signals.lights_shapes_required:
            _add_source(ls_req, c.primitive.value, rule.id)
        for c in rule.actions.signals.lights_shapes_forbidden:
            _add_source(ls_forb, c.primitive.value, rule.id)
        for c in rule.actions.signals.sounds_required:
            _add_source(snd_req, c.primitive.value, rule.id)
        for c in rule.actions.signals.sounds_forbidden:
            _add_source(snd_forb, c.primitive.value, rule.id)

    # signals legality check (aggregate required/forbidden sets)
    sig_issues = tuple(
        validate_signals_outputs(
            features=features,
            lights_required=[LightShapePrimitive(k) for k in ls_req.keys() if k in LightShapePrimitive._value2member_map_],
            lights_forbidden=[LightShapePrimitive(k) for k in ls_forb.keys() if k in LightShapePrimitive._value2member_map_],
            sounds_required=[SoundPrimitive(k) for k in snd_req.keys() if k in SoundPrimitive._value2member_map_],
            sounds_forbidden=[SoundPrimitive(k) for k in snd_forb.keys() if k in SoundPrimitive._value2member_map_],
        )
    )
    if strict and sig_issues:
        raise ValueError("Signals legality validation failed: " + "; ".join(i.code for i in sig_issues))

    # Explanation chain (deterministic, compact)
    chain: List[str] = []
    if features.visibility is not None:
        chain.append(f"domain.visibility={features.visibility.value}")
    if features.area_type is not None:
        chain.append(f"domain.area_type={features.area_type.value}")
    if features.in_sight is not None:
        chain.append(f"domain.in_sight={features.in_sight}")
    if features.encounter_type is not None:
        chain.append(f"geometry.encounter_type={features.encounter_type.value}")
    chain.append("triggered_rules=" + ",".join([t.rule_id for t in triggered]) if triggered else "triggered_rules=<none>")

    labels = LabelsWithSources(
        maneuver_allowed={k: tuple(v) for k, v in sorted(man_allowed.items())},
        maneuver_forbidden={k: tuple(v) for k, v in sorted(man_forbidden.items())},
        lights_required={k: tuple(v) for k, v in sorted(ls_req.items())},
        lights_forbidden={k: tuple(v) for k, v in sorted(ls_forb.items())},
        sounds_required={k: tuple(v) for k, v in sorted(snd_req.items())},
        sounds_forbidden={k: tuple(v) for k, v in sorted(snd_forb.items())},
    )

    return AggregateResult(
        triggered_rules=tuple(triggered),
        labels=labels,
        explanation_chain=tuple(chain),
        unknown_rules=tuple(sorted(unknowns, key=lambda x: x[0])),
        sanity_issues=sanity,
        signals_issues=sig_issues,
    )
