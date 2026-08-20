## `colreg_reasoner/src/colreg_reasoner/multiagent.py`


from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from colreg_kernel.aggregate import AggregateResult
from colreg_kernel.rules import load_rules

from colreg_reasoner.paths import default_rules_path
from colreg_reasoner.priority import load_priority_policy
from colreg_reasoner.global_types import GlobalResolveInput, GlobalResolutionV2
from colreg_reasoner.global_ledger import (
    build_constraint_atoms_from_aggregate,
    build_global_primitive_ledger,
)
from colreg_reasoner.global_policy import resolve_ledger_to_global_resolution


@dataclass(frozen=True)
class TargetAggregate:
    target_index: int
    aggregate: AggregateResult


@dataclass(frozen=True)
class MultiResolutionResult:
    per_target: Tuple[TargetAggregate, ...]
    global_input: GlobalResolveInput
    global_resolution: GlobalResolutionV2


@dataclass(frozen=True)
class MultiResolutionResultV2(MultiResolutionResult):
    """
    Backward-compatible alias type for callers that already reference the v2 name.
    The project-standard multi-target output is now GlobalResolutionV2 only.
    """


def _resolve_from_global_input(
    global_input: GlobalResolveInput,
    *,
    priorities_path: Optional[str] = None,
) -> MultiResolutionResult:
    if not global_input.targets:
        raise ValueError("global_input.targets is empty")

    rules_path = global_input.rules_path or default_rules_path()
    rules = load_rules(rules_path)
    policy = load_priority_policy(priorities_path)

    per: List[TargetAggregate] = []
    atoms = []
    for tgt in global_input.targets:
        per.append(
            TargetAggregate(
                target_index=tgt.target_index,
                aggregate=tgt.local_aggregate,
            )
        )
        atoms.extend(
            build_constraint_atoms_from_aggregate(
                tgt,
                rules=rules,
                policy=policy,
            )
        )

    ledger = build_global_primitive_ledger(atoms)
    global_res = resolve_ledger_to_global_resolution(
        global_input,
        ledger,
        rules=rules,
    )
    return MultiResolutionResult(
        per_target=tuple(per),
        global_input=global_input,
        global_resolution=global_res,
    )


def resolve_multi(
    global_input: GlobalResolveInput,
    *,
    priorities_path: Optional[str] = None,
) -> MultiResolutionResult:
    """
    Project-standard multi-target resolution entrypoint.

    This function now only accepts GlobalResolveInput and always returns
    GlobalResolutionV2-based results. The previous features_list-based
    multi-target path has been removed to avoid accidental use of the
    deprecated representative-context global resolver.
    """
    return _resolve_from_global_input(
        global_input,
        priorities_path=priorities_path,
    )


def resolve_multi_v2(
    global_input: GlobalResolveInput,
    *,
    priorities_path: Optional[str] = None,
) -> MultiResolutionResultV2:
    """
    Backward-compatible alias for callers already using resolve_multi_v2.
    Returns the same GlobalResolutionV2-based result as resolve_multi().
    """
    base = _resolve_from_global_input(
        global_input,
        priorities_path=priorities_path,
    )
    return MultiResolutionResultV2(
        per_target=base.per_target,
        global_input=base.global_input,
        global_resolution=base.global_resolution,
    )