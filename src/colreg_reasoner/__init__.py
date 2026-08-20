## `colreg_reasoner/src/colreg_reasoner/__init__.py`


"""COLREG reasoning (Part 2): pairwise relations + scene-level global resolution.

This package depends on `colreg_kernel` (Part 1: rule structuring & aggregation).

Multi-target reasoning now has a single project-standard entrypoint:
    - resolve_multi(global_input, ...)
which always returns GlobalResolutionV2-based results.

The symbol `resolve_multi_v2` may still exist inside implementation modules as a
temporary compatibility alias, but it is no longer part of the public package API.
"""

from .pairwise import PairRelation, ConflictReason, PairResult, pair_relation
from .resolver import ResolutionResult, resolve, resolve_from_aggregate
from .priority import PriorityPolicy, load_priority_policy, rank_rule
from .multiagent import (
    resolve_multi,
    resolve_multi_v2,
    MultiResolutionResult,
    MultiResolutionResultV2,
)
from .metrics import TensionMetrics, compute_tension_metrics
from .graph import RuleGraph, build_rule_graph
from .explain import ExplanationChain, ExplanationStep
from .global_types import (
    GlobalResolveInput,
    GlobalResolutionV2,
    TargetContext,
    ConstraintAtom,
    GlobalPrimitiveLedger,
    SuppressionRecord,
    FinalLabelSet,
    KeptRuleRecord,
    TargetSummary,
)

# Public API:
# - resolve_multi(...) is the only project-standard multi-target resolver.
# - resolve_multi_v2(...) is intentionally not exported in __all__.
__all__ = [
    "PairRelation",
    "ConflictReason",
    "PairResult",
    "pair_relation",
    "PriorityPolicy",
    "load_priority_policy",
    "rank_rule",
    "ResolutionResult",
    "resolve",
    "resolve_from_aggregate",
    "MultiResolutionResult",
    "MultiResolutionResultV2",
    "resolve_multi",
    "GlobalResolveInput",
    "GlobalResolutionV2",
    "TargetContext",
    "ConstraintAtom",
    "GlobalPrimitiveLedger",
    "SuppressionRecord",
    "FinalLabelSet",
    "KeptRuleRecord",
    "TargetSummary",
    "TensionMetrics",
    "compute_tension_metrics",
    "RuleGraph",
    "build_rule_graph",
    "ExplanationChain",
    "ExplanationStep",
]