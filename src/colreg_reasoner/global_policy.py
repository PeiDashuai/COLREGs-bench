## `colreg_reasoner/src/colreg_reasoner/global_policy.py`


from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Sequence, Tuple

from colreg_kernel.rules import RuleSpec

from .global_types import (
    ConstraintAtom,
    FinalLabelSet,
    GlobalPrimitiveLedger,
    GlobalResolveInput,
    GlobalResolutionV2,
    KeptRuleRecord,
    SuppressionRecord,
    TargetSummary,
)
from .global_ledger import summarize_primitive_ledger

_SHORT_BLASTS = {"SOUND_1_SHORT", "SOUND_2_SHORT", "SOUND_3_SHORT"}
_HARD_POLARITIES = {"forbid", "require"}


def _sort_atoms(atoms: Iterable[ConstraintAtom]) -> List[ConstraintAtom]:
    return sorted(
        atoms,
        key=lambda a: (
            -a.priority_bucket,
            0 if a.polarity in _HARD_POLARITIES else 1,
            a.rule_id,
            a.target_id,
            a.atom_id,
        ),
    )


def _is_hard(atom: ConstraintAtom) -> bool:
    return atom.polarity in _HARD_POLARITIES


def _opposes(a: ConstraintAtom, b: ConstraintAtom) -> bool:
    return a.head == b.head and a.primitive == b.primitive and a.polarity != b.polarity


def _select_blocker(atom: ConstraintAtom, candidates: Sequence[ConstraintAtom]) -> ConstraintAtom | None:
    if not candidates:
        return None
    # Prefer harder polarity, then higher priority, then cross-target evidence, then deterministic tie-break.
    return sorted(
        candidates,
        key=lambda c: (
            0 if _is_hard(c) else 1,
            -c.priority_bucket,
            0 if c.target_id != atom.target_id else 1,
            c.rule_id,
            c.target_id,
            c.atom_id,
        ),
    )[0]


def _mk_suppression(
    inp: GlobalResolveInput,
    atom: ConstraintAtom,
    winner: ConstraintAtom | None,
    reason_type: str,
    reason_text: str,
    *,
    blocker_candidates: Sequence[ConstraintAtom] | None = None,
) -> SuppressionRecord:
    blocker_candidates = blocker_candidates or ()
    evidence_refs = {
        "blocking_candidate_atom_ids": [c.atom_id for c in blocker_candidates],
        "blocking_candidate_rule_ids": [c.rule_id for c in blocker_candidates],
        "blocking_candidate_target_ids": [c.target_id for c in blocker_candidates],
    }
    if atom.evidence_refs:
        evidence_refs["suppressed_atom_evidence"] = dict(atom.evidence_refs)
    debug_meta = {
        "blocker_count": len(blocker_candidates),
        "cross_target_block": bool(
            winner is not None and winner.target_id != atom.target_id
        ),
    }
    return SuppressionRecord(
        suppression_id=f"sup:{atom.atom_id}:{winner.atom_id if winner else reason_type}",
        suppressed_atom_id=atom.atom_id,
        suppressed_rule_id=atom.rule_id,
        suppressed_target_id=atom.target_id,
        suppressed_head=atom.head,
        suppressed_primitive=atom.primitive,
        suppressed_polarity=atom.polarity,
        suppressed_by_atom_id=winner.atom_id if winner else None,
        suppressed_by_rule_id=winner.rule_id if winner else None,
        suppressed_by_target_id=winner.target_id if winner else None,
        suppressed_by_head=winner.head if winner else None,
        suppressed_by_primitive=winner.primitive if winner else None,
        suppressed_by_polarity=winner.polarity if winner else None,
        reason_type=reason_type,
        reason_text=reason_text,
        suppressed_priority_bucket=atom.priority_bucket,
        suppressed_by_priority_bucket=winner.priority_bucket if winner else None,
        priority_delta=(winner.priority_bucket - atom.priority_bucket) if winner else None,
        scene_id=inp.scene_id,
        snapshot_time_s=inp.snapshot_time_s,
        evidence_refs=evidence_refs,
        debug_meta=debug_meta,
    )


def _resolve_bucket(
    atoms: Tuple[ConstraintAtom, ...],
    *,
    inp: GlobalResolveInput,
    head: str,
    primitive: str,
) -> tuple[list[ConstraintAtom], list[SuppressionRecord]]:
    kept: list[ConstraintAtom] = []
    suppressed: list[SuppressionRecord] = []
    atoms_s = _sort_atoms(atoms)

    # Domain legality first for short blasts in restricted / not-in-sight conditions.
    if head == "sounds" and primitive in _SHORT_BLASTS:
        if inp.visibility == "restricted" or inp.in_sight is False:
            for atom in atoms_s:
                suppressed.append(
                    _mk_suppression(
                        inp,
                        atom,
                        None,
                        "visibility_constraint",
                        f"{primitive} suppressed in restricted/not-in-sight domain",
                    )
                )
            return kept, suppressed

    for atom in atoms_s:
        # Same-polarity contributions are cumulative evidence unless exact duplicate atom_id.
        same_pol_kept = [k for k in kept if k.polarity == atom.polarity]
        if any(k.atom_id == atom.atom_id for k in same_pol_kept):
            blocker = same_pol_kept[0]
            suppressed.append(
                _mk_suppression(
                    inp,
                    atom,
                    blocker,
                    "duplicate_weaker_support",
                    f"duplicate support on {primitive} dropped",
                    blocker_candidates=same_pol_kept,
                )
            )
            continue

        opposing_kept = [k for k in kept if _opposes(k, atom)]
        if not opposing_kept:
            kept.append(atom)
            continue

        blocker = _select_blocker(atom, opposing_kept)
        assert blocker is not None
        blocker_hard = _is_hard(blocker)
        atom_hard = _is_hard(atom)
        max_opp_priority = max(k.priority_bucket for k in opposing_kept)

        # Existing opposite hard constraint wins at same-or-higher priority.
        if blocker_hard and blocker.priority_bucket >= atom.priority_bucket:
            suppressed.append(
                _mk_suppression(
                    inp,
                    atom,
                    blocker,
                    f"higher_priority_{blocker.polarity}",
                    f"{atom.polarity} on {primitive} blocked by {blocker.polarity} from {blocker.rule_id}/{blocker.target_id}",
                    blocker_candidates=opposing_kept,
                )
            )
            continue

        # New hard constraint with strictly higher priority suppresses all opposite kept atoms.
        if atom_hard and atom.priority_bucket > max_opp_priority:
            for prev in list(opposing_kept):
                if prev in kept:
                    suppressed.append(
                        _mk_suppression(
                            inp,
                            prev,
                            atom,
                            f"higher_priority_{atom.polarity}",
                            f"{prev.polarity} on {primitive} blocked by higher-priority {atom.polarity} from {atom.rule_id}/{atom.target_id}",
                            blocker_candidates=(atom,),
                        )
                    )
                    kept.remove(prev)
            kept.append(atom)
            continue

        # Equal-priority hard-vs-soft: prefer hard constraint.
        if atom_hard and atom.priority_bucket == max_opp_priority:
            weaker_opposing = [k for k in opposing_kept if not _is_hard(k)]
            if weaker_opposing:
                for prev in list(weaker_opposing):
                    if prev in kept:
                        suppressed.append(
                            _mk_suppression(
                                inp,
                                prev,
                                atom,
                                "mutual_exclusion_resolution",
                                f"soft {prev.polarity} on {primitive} displaced by hard {atom.polarity} at equal priority",
                                blocker_candidates=(atom,),
                            )
                        )
                        kept.remove(prev)
                kept.append(atom)
                continue

        # Otherwise prune the new atom to preserve a single globally coherent outcome.
        suppressed.append(
            _mk_suppression(
                inp,
                atom,
                blocker,
                "global_consistency_pruning",
                f"{atom.polarity} on {primitive} dropped to preserve consistency against {blocker.polarity} from {blocker.rule_id}/{blocker.target_id}",
                blocker_candidates=opposing_kept,
            )
        )

    return kept, suppressed


def resolve_ledger_to_global_resolution(
    inp: GlobalResolveInput,
    ledger: GlobalPrimitiveLedger,
    *,
    rules: Dict[str, RuleSpec],
) -> GlobalResolutionV2:
    man_allow: Dict[str, List[str]] = defaultdict(list)
    man_forbid: Dict[str, List[str]] = defaultdict(list)
    light_req: Dict[str, List[str]] = defaultdict(list)
    light_forbid: Dict[str, List[str]] = defaultdict(list)
    sound_req: Dict[str, List[str]] = defaultdict(list)
    sound_forbid: Dict[str, List[str]] = defaultdict(list)
    suppression_records: List[SuppressionRecord] = []
    kept_atoms: List[ConstraintAtom] = []

    for head_name, head_map in (
        ("maneuver", ledger.maneuver),
        ("lights", ledger.lights),
        ("sounds", ledger.sounds),
    ):
        for primitive, atoms in head_map.items():
            kept, suppressed = _resolve_bucket(
                atoms,
                inp=inp,
                head=head_name,
                primitive=primitive,
            )
            kept_atoms.extend(kept)
            suppression_records.extend(suppressed)
            for atom in kept:
                if atom.head == "maneuver":
                    if atom.polarity == "allow":
                        man_allow[primitive].append(atom.rule_id)
                    else:
                        man_forbid[primitive].append(atom.rule_id)
                elif atom.head == "lights":
                    if atom.polarity == "require":
                        light_req[primitive].append(atom.rule_id)
                    else:
                        light_forbid[primitive].append(atom.rule_id)
                elif atom.head == "sounds":
                    if atom.polarity == "require":
                        sound_req[primitive].append(atom.rule_id)
                    else:
                        sound_forbid[primitive].append(atom.rule_id)

    def _finalize(d: Dict[str, List[str]]) -> Dict[str, tuple[str, ...]]:
        return {k: tuple(sorted(set(v))) for k, v in sorted(d.items())}

    final_labels = FinalLabelSet(
        maneuver_allowed=_finalize(man_allow),
        maneuver_forbidden=_finalize(man_forbid),
        lights_required=_finalize(light_req),
        lights_forbidden=_finalize(light_forbid),
        sounds_required=_finalize(sound_req),
        sounds_forbidden=_finalize(sound_forbid),
    )

    kept_rule_map: Dict[str, Dict[str, set]] = {}
    for atom in kept_atoms:
        kr = kept_rule_map.setdefault(
            atom.rule_id,
            {"target_ids": set(), "atom_ids": set(), "heads": set(), "primitives": set()},
        )
        kr["target_ids"].add(atom.target_id)
        kr["atom_ids"].add(atom.atom_id)
        kr["heads"].add(atom.head)
        kr["primitives"].add(atom.primitive)
    kept_rules: List[KeptRuleRecord] = []
    for rid, data in sorted(kept_rule_map.items()):
        rule = rules.get(rid)
        kept_rules.append(
            KeptRuleRecord(
                rule_id=rid,
                article=rule.article if rule else None,
                target_ids=tuple(sorted(data["target_ids"])),
                contributing_atom_ids=tuple(sorted(data["atom_ids"])),
                heads=tuple(sorted(data["heads"])),
                primitives=tuple(sorted(data["primitives"])),
            )
        )

    target_summaries = tuple(
        TargetSummary(
            target_id=t.target_id,
            target_index=t.target_index,
            encounter_type=getattr(t.features.encounter_type, "value", t.features.encounter_type),
            relative_bearing_sector=getattr(
                t.features.relative_bearing_sector,
                "value",
                t.features.relative_bearing_sector,
            ),
            is_target_on_starboard=t.features.is_target_on_starboard,
            closing=t.features.closing,
            cpa_m=t.features.cpa_m,
            tcpa_s=t.features.tcpa_s,
            risk_of_collision=t.features.risk_of_collision,
            collision_imminent=t.features.collision_imminent,
            triggered_rule_ids=tuple(tr.rule_id for tr in t.local_aggregate.triggered_rules),
            role_tags=tuple(t.role_tags),
        )
        for t in inp.targets
    )

    explanation_steps = (
        {
            "stage": "targets",
            "targets": [asdict(ts) for ts in target_summaries],
        },
        {
            "stage": "ledger",
            "summary": summarize_primitive_ledger(ledger),
        },
        {
            "stage": "suppression",
            "num_suppressions": len(suppression_records),
            "suppressed_rule_ids": sorted({s.suppressed_rule_id for s in suppression_records}),
            "cross_target_suppressions": sum(
                1
                for s in suppression_records
                if s.suppressed_by_target_id
                and s.suppressed_by_target_id != s.suppressed_target_id
            ),
        },
        {
            "stage": "final_labels",
            "final_labels": asdict(final_labels),
        },
    )

    num_conflicts_maneuver = sum(1 for s in suppression_records if s.suppressed_head == "maneuver")
    num_conflicts_signals = sum(1 for s in suppression_records if s.suppressed_head in {"lights", "sounds"})
    num_conflicts_domain = sum(
        1 for s in suppression_records if s.reason_type in {"visibility_constraint", "domain_incompatibility"}
    )
    num_conflicts = len(suppression_records)
    num_tensions = num_conflicts
    complexity_score = float(len(kept_rules) + num_conflicts_maneuver + 0.5 * num_conflicts_signals)

    global_metrics = {
        "num_targets": len(inp.targets),
        "num_rules": len(kept_rules),
        "num_kept_rules": len(kept_rules),
        "num_suppression_events": len(suppression_records),
        "num_domain_suppressions": num_conflicts_domain,
        "num_conflicts": num_conflicts,
        "num_conflicts_maneuver": num_conflicts_maneuver,
        "num_conflicts_signals": num_conflicts_signals,
        "num_conflicts_domain": num_conflicts_domain,
        "num_tensions": num_tensions,
        "complexity_score": complexity_score,
        "num_primitives_constrained": (
            len(final_labels.maneuver_allowed)
            + len(final_labels.maneuver_forbidden)
            + len(final_labels.lights_required)
            + len(final_labels.lights_forbidden)
            + len(final_labels.sounds_required)
            + len(final_labels.sounds_forbidden)
        ),
        "num_cross_target_conflicts": len(
            {
                (s.suppressed_target_id, s.suppressed_by_target_id)
                for s in suppression_records
                if s.suppressed_by_target_id
                and s.suppressed_by_target_id != s.suppressed_target_id
            }
        ),
    }

    rule_graph = {
        "nodes": tuple(sorted({k.rule_id for k in kept_rules})),
        "edges": tuple(),
        "adjacency": {},
    }
    sanity_report = {
        "ok": True,
        "issues": [],
        "has_label_contradiction": any(
            k in final_labels.maneuver_forbidden for k in final_labels.maneuver_allowed
        ),
    }
    if sanity_report["has_label_contradiction"]:
        sanity_report["ok"] = False
        sanity_report["issues"].append("maneuver_label_contradiction")

    return GlobalResolutionV2(
        scene_id=inp.scene_id,
        snapshot_index=inp.snapshot_index,
        snapshot_time_s=inp.snapshot_time_s,
        final_labels=final_labels,
        kept_rules=tuple(kept_rules),
        suppression_records=tuple(suppression_records),
        target_summaries=target_summaries,
        primitive_ledger_summary=summarize_primitive_ledger(ledger),
        explanation_steps=tuple(explanation_steps),
        global_metrics=global_metrics,
        rule_graph=rule_graph,
        sanity_report=sanity_report,
        meta={"visibility": inp.visibility, "in_sight": inp.in_sight},
    )