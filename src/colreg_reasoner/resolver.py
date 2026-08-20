from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from colreg_kernel.actions import ManeuverPrimitive, LightShapePrimitive, SoundPrimitive
from colreg_kernel.aggregate import AggregateResult
from colreg_kernel.features import Features
from colreg_reasoner.priority import load_priority_policy, rank_rule
from colreg_reasoner.paths import default_rules_path
from colreg_kernel.rules import RuleSpec, load_rules
from colreg_kernel.validator import validate_signals_outputs, validate_feature_consistency, ValidationIssue
from colreg_reasoner.explain import ExplanationChain, ExplanationStep
from colreg_reasoner.metrics import compute_tension_metrics, TensionMetrics
from colreg_reasoner.graph import build_rule_graph, RuleGraph


@dataclass(frozen=True)
class Suppressed:
    rule_id: str
    reason: str


@dataclass(frozen=True)
class ResolvedLabelsWithSources:
    # primitive -> tuple(rule_id,...)
    maneuver_allowed: Dict[str, Tuple[str, ...]]
    maneuver_forbidden: Dict[str, Tuple[str, ...]]
    lights_required: Dict[str, Tuple[str, ...]]
    lights_forbidden: Dict[str, Tuple[str, ...]]
    sounds_required: Dict[str, Tuple[str, ...]]
    sounds_forbidden: Dict[str, Tuple[str, ...]]


@dataclass(frozen=True)
class ResolutionResult:
    kept_rules: Tuple[str, ...]
    suppressed_rules: Tuple[Suppressed, ...]
    labels: ResolvedLabelsWithSources
    explanation_chain: Tuple[str, ...]
    explanation: ExplanationChain
    metrics: TensionMetrics
    rule_graph: RuleGraph
    sanity_issues: Tuple[ValidationIssue, ...]
    signals_issues: Tuple[ValidationIssue, ...]



def _add(map_: Dict[str, List[str]], prim: str, rule_id: str) -> None:
    map_.setdefault(prim, [])
    if rule_id not in map_[prim]:
        map_[prim].append(rule_id)


def resolve(
    features: Features,
    triggered_rule_ids: Sequence[str],
    *,
    rules_path: Optional[str] = None,
    priorities_path: Optional[str] = None,
    strict: bool = True,
) -> ResolutionResult:
    """
    Resolve a set of simultaneously triggered rules into consistent labels (maneuver + signals).

    Policy (baseline):
      - Sort rules by priority bucket (priorities.yaml) then by article and id.
      - Signals: if visibility=restricted, suppress Rule 34 short blasts (R35 domain).
      - Maneuver: add higher priority constraints first; lower priority constraints that conflict are dropped.
    """
    rules_path = rules_path or default_rules_path()
    rules: Dict[str, RuleSpec] = load_rules(rules_path)
    policy = load_priority_policy(priorities_path)

    sanity = tuple(validate_feature_consistency(features))
    if strict and sanity:
        raise ValueError("Feature sanity validation failed: " + "; ".join(i.code for i in sanity))

    # Build RuleSpec list, deterministic
    rlist: List[RuleSpec] = []
    for rid in triggered_rule_ids:
        if rid in rules:
            rlist.append(rules[rid])

    rlist = sorted(rlist, key=lambda r: (-rank_rule(r, policy), r.article, r.id))

    kept_rules: List[str] = []
    suppressed: List[Suppressed] = []

    # Accumulators with sources
    man_allowed: Dict[str, List[str]] = {}
    man_forb: Dict[str, List[str]] = {}
    ls_req: Dict[str, List[str]] = {}
    ls_forb: Dict[str, List[str]] = {}
    snd_req: Dict[str, List[str]] = {}
    snd_forb: Dict[str, List[str]] = {}

    # Baseline domain suppression: restricted visibility => suppress maneuvering short blasts
    from colreg_kernel.features import Visibility
    restricted = (features.visibility == Visibility.RESTRICTED)
    in_sight = bool(getattr(features, "in_sight", False))

    for rule in rlist:
        rid = rule.id
        kept_rules.append(rid)

        # Maneuver head merging with conflict handling
        for c in rule.actions.maneuver.allowed:
            prim = c.primitive.value
            # If already forbidden by higher priority, drop this contribution
            if prim in man_forb:
                suppressed.append(Suppressed(rule_id=rid, reason=f"maneuver_allowed {prim} blocked by higher-priority forbidden"))
                continue
            _add(man_allowed, prim, rid)

        for c in rule.actions.maneuver.forbidden:
            prim = c.primitive.value
            # If already allowed by higher priority, drop this forbidden (tighten is ambiguous; choose keep earlier)
            if prim in man_allowed:
                suppressed.append(Suppressed(rule_id=rid, reason=f"maneuver_forbidden {prim} blocked by higher-priority allowed"))
                continue
            _add(man_forb, prim, rid)

        # Signals head merging with domain checks
        for c in rule.actions.signals.lights_shapes_required:
            _add(ls_req, c.primitive.value, rid)
        for c in rule.actions.signals.lights_shapes_forbidden:
            _add(ls_forb, c.primitive.value, rid)

        # Sound signals:
        for c in rule.actions.signals.sounds_required:
            prim = c.primitive.value
            if restricted and prim in {SoundPrimitive.SOUND_1_SHORT.value, SoundPrimitive.SOUND_2_SHORT.value, SoundPrimitive.SOUND_3_SHORT.value}:
                suppressed.append(Suppressed(rule_id=rid, reason=f"sound {prim} suppressed in restricted visibility domain"))
                continue
            if (not in_sight) and prim in {SoundPrimitive.SOUND_1_SHORT.value, SoundPrimitive.SOUND_2_SHORT.value, SoundPrimitive.SOUND_3_SHORT.value}:
                suppressed.append(Suppressed(rule_id=rid, reason=f"sound {prim} suppressed when not in sight"))
                continue
            _add(snd_req, prim, rid)

        for c in rule.actions.signals.sounds_forbidden:
            _add(snd_forb, c.primitive.value, rid)

    # Convert to tuples deterministic
    def _finalize(m: Dict[str, List[str]]) -> Dict[str, Tuple[str, ...]]:
        return {k: tuple(sorted(v)) for k, v in sorted(m.items(), key=lambda kv: kv[0])}

    labels = ResolvedLabelsWithSources(
        maneuver_allowed=_finalize(man_allowed),
        maneuver_forbidden=_finalize(man_forb),
        lights_required=_finalize(ls_req),
        lights_forbidden=_finalize(ls_forb),
        sounds_required=_finalize(snd_req),
        sounds_forbidden=_finalize(snd_forb),
    )

    # Validate signals legality on merged outputs by reusing existing validator for per-vessel and per-domain checks.
    signals_issues = tuple(
        validate_signals_outputs(
            features=features,
            lights_required=[LightShapePrimitive(k) for k in labels.lights_required.keys()],
            lights_forbidden=[LightShapePrimitive(k) for k in labels.lights_forbidden.keys()],
            sounds_required=[SoundPrimitive(k) for k in labels.sounds_required.keys()],
            sounds_forbidden=[SoundPrimitive(k) for k in labels.sounds_forbidden.keys()],
        )
    )

    if strict and signals_issues:
        raise ValueError("Signals legality failed: " + "; ".join(i.code for i in signals_issues))

    explanation = []
    explanation.append(f"Triggered rules: {len(triggered_rule_ids)}")
    explanation.append("Priority order: " + ", ".join([r.id for r in rlist[:10]]))
    if suppressed:
        explanation.append(f"Suppressed contributions: {len(suppressed)}")
    # Structured explanation chain (coarse): stages by priority buckets
    steps = []
    # stage: domain (restricted visibility / in_sight)
    domain_notes = []
    if restricted:
        domain_notes.append("restricted_visibility")
    if in_sight:
        domain_notes.append("in_sight")
    steps.append(ExplanationStep(stage="domain", applied_rules=tuple([r.id for r in rlist if r.article in {"Rule 19", "Rule 35", "Rule 34", "Rule 11"}]), suppressed_rules=tuple([s.rule_id for s in suppressed if "sound" in s.reason]), notes=tuple(domain_notes)))
    # stage: responsibility (Rule 18)
    steps.append(ExplanationStep(stage="responsibility", applied_rules=tuple([r.id for r in rlist if r.article == "Rule 18"]), suppressed_rules=tuple([s.rule_id for s in suppressed if "respons" in s.reason.lower()]), notes=tuple()))
    # stage: encounter (13-17)
    steps.append(ExplanationStep(stage="encounter", applied_rules=tuple([r.id for r in rlist if r.article in {"Rule 13", "Rule 14", "Rule 15", "Rule 16", "Rule 17"}]), suppressed_rules=tuple([s.rule_id for s in suppressed if "maneuver" in s.reason]), notes=tuple()))
    # stage: area (9-10)
    steps.append(ExplanationStep(stage="area", applied_rules=tuple([r.id for r in rlist if r.article in {"Rule 9", "Rule 10"}]), suppressed_rules=tuple(), notes=tuple()))
    # stage: signals (20-31,34-35)
    steps.append(ExplanationStep(stage="signals", applied_rules=tuple([r.id for r in rlist if r.article.startswith("Rule 2") or r.article in {"Rule 34", "Rule 35"}]), suppressed_rules=tuple([s.rule_id for s in suppressed if "sound" in s.reason]), notes=tuple()))
    expl = ExplanationChain(steps=tuple(steps))

    # Metrics and graph computed on kept rules
    kept_specs = [rules[rid] for rid in kept_rules if rid in rules]
    metrics = compute_tension_metrics(kept_specs, features)
    rule_graph = build_rule_graph(kept_specs, features)
    return ResolutionResult(
        kept_rules=tuple(kept_rules),
        suppressed_rules=tuple(suppressed),
        labels=labels,
        explanation_chain=tuple(explanation),
        explanation=expl,
        metrics=metrics,
        rule_graph=rule_graph,
        sanity_issues=sanity,
        signals_issues=signals_issues,
    )


def resolve_from_aggregate(
    features: Features,
    agg: AggregateResult,
    *,
    rules_path: Optional[str] = None,
    priorities_path: Optional[str] = None,
    strict: bool = True,
) -> ResolutionResult:
    return resolve(
        features,
        [t.rule_id for t in agg.triggered_rules],
        rules_path=rules_path,
        priorities_path=priorities_path,
        strict=strict,
    )
