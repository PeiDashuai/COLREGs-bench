"""Explicit, expert-gated projection into Common Semantic Ontology v1."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .hashing import sha256_file
from .io import iter_jsonl, read_json, write_json_atomic


COMMON_HEADS = ("rule_articles", "maneuver_allowed", "maneuver_forbidden")
BENCHMARK_HEAD_MAP = {
    "triggered_rules": "rule_articles",
    "maneuver_allowed": "maneuver_allowed",
    "maneuver_forbidden": "maneuver_forbidden",
}
ENGINE_HEAD_MAP = BENCHMARK_HEAD_MAP
FINAL_REVIEW_STATES = frozenset({"agreed", "adjudicated", "unsupported"})
REVIEW_COLUMNS = (
    "mapping_id",
    "source_space",
    "source_head",
    "source_token",
    "proposed_target_head",
    "proposed_target_tokens",
    "legal_or_operational_basis",
    "expert_1_id",
    "expert_1_decision",
    "expert_1_target_tokens",
    "expert_1_notes",
    "expert_2_id",
    "expert_2_decision",
    "expert_2_target_tokens",
    "expert_2_notes",
    "adjudicator_id",
    "adjudication_target_tokens",
    "adjudication_notes",
)


class SemanticProjectionError(ValueError):
    """Raised for ambiguous, incomplete, or vocabulary-violating mappings."""


def load_ontology(path: str | Path) -> dict[str, Any]:
    ontology = read_json(path)
    heads = ontology.get("heads") if isinstance(ontology, Mapping) else None
    if not isinstance(heads, Mapping) or set(heads) != set(COMMON_HEADS):
        raise SemanticProjectionError("ontology must define exactly the three common heads")
    for head, tokens in heads.items():
        if not isinstance(tokens, list) or len(tokens) != len(set(tokens)):
            raise SemanticProjectionError(f"ontology head {head} is not a unique list")
    return dict(ontology)


def load_mapping(path: str | Path, ontology: Mapping[str, Any]) -> dict[str, Any]:
    mapping = read_json(path)
    entries = mapping.get("entries") if isinstance(mapping, Mapping) else None
    if not isinstance(entries, list):
        raise SemanticProjectionError("mapping entries must be a list")
    seen: set[tuple[str, str]] = set()
    vocabulary = {head: set(tokens) for head, tokens in ontology["heads"].items()}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise SemanticProjectionError("mapping entry must be an object")
        key = (str(entry.get("source_head")), str(entry.get("source_token")))
        if key in seen:
            raise SemanticProjectionError(f"duplicate mapping source key: {key}")
        seen.add(key)
        target_head = entry.get("target_head")
        targets = entry.get("target_tokens")
        status = entry.get("review_status")
        if status not in {"pending", *FINAL_REVIEW_STATES}:
            raise SemanticProjectionError(f"invalid review_status for {key}: {status}")
        if status == "unsupported":
            if target_head is not None or targets not in ([], None):
                raise SemanticProjectionError(f"unsupported entry must not map tokens: {key}")
            continue
        if status == "pending" and (target_head is None or not targets):
            continue
        if target_head not in vocabulary or not isinstance(targets, list) or not targets:
            raise SemanticProjectionError(f"mapped entry lacks target tokens: {key}")
        unknown = set(targets) - vocabulary[target_head]
        if unknown:
            raise SemanticProjectionError(f"entry {key} uses tokens outside ontology: {unknown}")
    return dict(mapping)


def assert_expert_review_complete(mapping: Mapping[str, Any]) -> None:
    """Require every public mapping entry to have a final reviewed status.

    Reviewer identities and adjudication notes are intentionally not part of the
    public machine-readable mapping.  The released status and target mapping are
    sufficient to reproduce the common-semantic scorer.
    """
    pending: list[str] = []
    for entry in mapping.get("entries", []):
        status = entry.get("review_status")
        if status not in FINAL_REVIEW_STATES:
            pending.append(str(entry.get("mapping_id")))
    if pending:
        raise SemanticProjectionError(
            f"reviewed semantic mapping is incomplete for {len(pending)} entries: {pending[:10]}"
        )


def _rule_target(token: str) -> tuple[str, ...]:
    if "MULTI_TARGET" in token:
        return ("MULTI_TARGET_RESOLUTION",)
    match = re.search(r"(?:COLREG_)?R(0?5|0?6|0?7|0?8|0?9|10|13|14|15|16|17|18|19)", token)
    lookup = {
        "5": "R05_LOOKOUT",
        "6": "R06_SAFE_SPEED",
        "7": "R07_RISK_ASSESSMENT",
        "8": "R08_COLLISION_AVOIDANCE_ACTION",
        "9": "R09_NARROW_CHANNEL",
        "10": "R10_TRAFFIC_SEPARATION_SCHEME",
        "13": "R13_OVERTAKING",
        "14": "R14_HEAD_ON",
        "15": "R15_CROSSING",
        "16": "R16_GIVE_WAY_ACTION",
        "17": "R17_STAND_ON_ACTION",
        "18": "R18_VESSEL_RESPONSIBILITY",
        "19": "R19_RESTRICTED_VISIBILITY",
    }
    return (lookup[match.group(1).lstrip("0")],) if match else ()


def _allowed_target(token: str, source_space: str) -> tuple[str, ...]:
    exact = {
        "KEEP_LOOKOUT": "KEEP_LOOKOUT",
        "MAINTAIN_EFFECTIVE_LOOKOUT": "KEEP_LOOKOUT",
        "KEEP_SAFE_SPEED": "MAINTAIN_SAFE_SPEED",
        "PROCEED_AT_SAFE_SPEED": "MAINTAIN_SAFE_SPEED",
        "PROCEED_SAFE_SPEED": "MAINTAIN_SAFE_SPEED",
        "ASSESS_RISK": "ASSESS_COLLISION_RISK",
        "POSITIVE_ACTION": "TAKE_POSITIVE_ACTION",
        "GIVE_WAY_IF_REQUIRED": "GIVE_WAY",
        "MAINTAIN_STANDON_IF_SAFE": "STAND_ON_IF_SAFE",
        "MAINTAIN_COURSE_SPEED": "STAND_ON_IF_SAFE",
        "TURN_STARBOARD": "ALTER_COURSE_STARBOARD",
        "TURN_PORT": "ALTER_COURSE_PORT",
        "REDUCE_SPEED": "REDUCE_SPEED",
        "KEEP_TO_STARBOARD_SIDE": "KEEP_TO_STARBOARD_IN_CHANNEL",
        "KEEP_TO_STARBOARD_WITHIN_CHANNEL": "KEEP_TO_STARBOARD_IN_CHANNEL",
        "CROSS_TSS_RIGHT_ANGLE": "CROSS_TSS_NEAR_RIGHT_ANGLE",
        "NAVIGATE_WITH_RESTRICTED_VISIBILITY_CAUTION": "RESTRICTED_VISIBILITY_CAUTION",
        "RESOLVE_MULTI_TARGET_CONFLICT": "RESOLVE_MULTI_TARGET_CONFLICT",
    }
    if token in exact:
        return (exact[token],)
    if token in {"TAKE_EARLY_SUBSTANTIAL_ACTION", "TAKE_IMMEDIATE_AVOIDING_ACTION"}:
        return ("TAKE_POSITIVE_ACTION",)
    if "IMPEDE" in token or "CHANNEL_COMPRESSION" in token:
        return ("NOT_IMPEDE",)
    if "CROSS" in token or "CHANNEL" in token or "TSS" in token:
        return ("WAIT_OR_ADJUST_BEFORE_CROSSING",)
    if any(word in token for word in ("MONITOR", "CHECK_RECIPROCAL")):
        return ("ASSESS_COLLISION_RISK",)
    if any(
        word in token
        for word in ("REASON_OVER_ALL", "RESOLVE_", "SELECT_BALANCED", "PRIORITIZE_", "ROLE_AWARE", "PREFER_MINIMAL")
    ):
        return ("RESOLVE_MULTI_TARGET_CONFLICT",)
    return ()


def _forbidden_target(token: str) -> tuple[str, ...]:
    if token == "TURN_PORT":
        return ("TURN_PORT_WHEN_CONFLICTING",)
    if token == "IMPEDE":
        return ("IMPEDE_CHANNEL_OR_TSS_TRAFFIC",)
    if token in {"CROSS_WITHOUT_CLEARANCE", "FORCE_CROSSING_THROUGH_CHANNEL_TRAFFIC", "SQUEEZE_THROUGH_BOUNDARY_GAP"}:
        return ("CROSS_WITHOUT_CLEARANCE",)
    if token in {"CROSS_AT_SMALL_ANGLE", "MAINTAIN_STRONGLY_OFF_CROSSING"}:
        return ("CROSS_TSS_AT_UNSAFE_ANGLE",)
    if token == "DEFER_ACTION":
        return ("DEFER_REQUIRED_ACTION",)
    if token == "KEEP_COURSE_WITHOUT_GLOBAL_CHECK":
        return ("MAINTAIN_COURSE_WITHOUT_GLOBAL_CHECK",)
    if "SINGLE_TARGET" in token:
        return ("USE_SINGLE_TARGET_ONLY",)
    if token in {"IGNORE_HIGHER_PRIORITY_CONSTRAINT", "IGNORE_PRIORITY_TARGET"}:
        return ("IGNORE_HIGHER_PRIORITY_CONSTRAINT",)
    if token in {
        "IGNORE_CHANNEL_CONTEXT",
        "IGNORE_CONTEXTUAL_OVERRIDE",
        "OPEN_WATER_ONLY_REASONING",
        "REDUCE_TO_SINGLE_OPEN_WATER_CROSSING_RULE",
        "DRIFT_TO_PORT_WITHIN_CHANNEL",
        "PROCEED_OPPOSITE_FLOW",
        "ENTER_SEPARATION_ZONE",
        "OVERTAKE_IN_CHANNEL",
        "ANCHOR",
    }:
        return ("IGNORE_CONTEXTUAL_AREA_RULE",)
    if token == "TURN_PORT_IF_CONFLICTING":
        return ("TURN_PORT_WHEN_CONFLICTING",)
    return ()


def proposed_targets(source_head: str, source_token: str, source_space: str) -> tuple[str, tuple[str, ...]]:
    target_head = BENCHMARK_HEAD_MAP.get(source_head)
    if target_head is None:
        raise SemanticProjectionError(f"unsupported source head: {source_head}")
    if target_head == "rule_articles":
        targets = _rule_target(source_token)
    elif target_head == "maneuver_allowed":
        targets = _allowed_target(source_token, source_space)
    else:
        targets = _forbidden_target(source_token)
    return target_head, targets


def build_mapping_draft(
    *,
    source_space: str,
    source_tokens: Mapping[str, Iterable[str]],
    source_manifest: Mapping[str, Any],
    output_path: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for source_head in sorted(source_tokens):
        for source_token in sorted(set(source_tokens[source_head])):
            target_head, targets = proposed_targets(source_head, source_token, source_space)
            identifier = f"{source_space}:{source_head}:{source_token}"
            entries.append(
                {
                    "mapping_id": identifier,
                    "source_space": source_space,
                    "source_head": source_head,
                    "source_token": source_token,
                    "target_head": target_head,
                    "target_tokens": list(targets),
                    "legal_or_operational_basis": "draft_from_train_validation_or_pinned_rule_spec",
                    "review_status": "pending",
                    "mapping_version": 1,
                    "reviewers": [],
                }
            )
    result = {
        "schema_version": 1,
        "mapping_name": f"{source_space}_to_common_v1",
        "source_space": source_space,
        "common_ontology_id": "common_semantic_v1",
        "source_manifest": dict(source_manifest),
        "entries": entries,
    }
    write_json_atomic(output_path, result, overwrite=overwrite)
    return result


def collect_benchmark_vocabulary(paths: Iterable[str | Path]) -> dict[str, set[str]]:
    vocabulary = {head: set() for head in BENCHMARK_HEAD_MAP}
    for path in paths:
        for row in iter_jsonl(path):
            labels = row.get("labels")
            if not isinstance(labels, Mapping):
                raise SemanticProjectionError("benchmark vocabulary source lacks labels")
            for head in vocabulary:
                values = labels.get(head, [])
                if isinstance(values, list):
                    vocabulary[head].update(str(item) for item in values)
    return vocabulary


def collect_engine_vocabulary(rules_path: str | Path) -> dict[str, set[str]]:
    rules = yaml.safe_load(Path(rules_path).read_text(encoding="utf-8"))
    vocabulary = {head: set() for head in ENGINE_HEAD_MAP}
    for rule in rules:
        vocabulary["triggered_rules"].add(str(rule["id"]))
        maneuver = rule.get("actions", {}).get("maneuver", {})
        for item in maneuver.get("allowed", []):
            vocabulary["maneuver_allowed"].add(str(item["primitive"]))
        for item in maneuver.get("forbidden", []):
            vocabulary["maneuver_forbidden"].add(str(item["primitive"]))
    vocabulary["maneuver_allowed"].add("RESOLVE_MULTI_TARGET_CONFLICT")
    return vocabulary


def export_expert_review_csv(
    mappings: Iterable[Mapping[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite semantic mapping review file: {target}")
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for mapping in mappings:
            for entry in mapping.get("entries", []):
                writer.writerow(
                    {
                        "mapping_id": entry["mapping_id"],
                        "source_space": entry["source_space"],
                        "source_head": entry["source_head"],
                        "source_token": entry["source_token"],
                        "proposed_target_head": entry.get("target_head") or "",
                        "proposed_target_tokens": "|".join(entry.get("target_tokens", [])),
                        "legal_or_operational_basis": entry.get("legal_or_operational_basis", ""),
                    }
                )
    return target


def apply_expert_review_csv(
    mapping: Mapping[str, Any], review_csv: str | Path
) -> dict[str, Any]:
    with Path(review_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        review_rows = {row["mapping_id"]: row for row in csv.DictReader(handle)}
    result = dict(mapping)
    entries: list[dict[str, Any]] = []
    for original in mapping.get("entries", []):
        entry = dict(original)
        row = review_rows.get(str(entry["mapping_id"]))
        if row is None:
            entries.append(entry)
            continue
        expert_ids = [row.get("expert_1_id", "").strip(), row.get("expert_2_id", "").strip()]
        decisions = [row.get("expert_1_decision", "").strip().lower(), row.get("expert_2_decision", "").strip().lower()]
        revised_left = row.get("expert_1_target_tokens", "").strip()
        revised_right = row.get("expert_2_target_tokens", "").strip()
        if all(expert_ids) and len(set(expert_ids)) == 2 and decisions == ["agree", "agree"]:
            entry["review_status"] = "agreed"
            entry["reviewers"] = expert_ids
        elif all(expert_ids) and len(set(expert_ids)) == 2 and decisions == ["unsupported", "unsupported"]:
            entry["review_status"] = "unsupported"
            entry["target_head"] = None
            entry["target_tokens"] = []
            entry["reviewers"] = expert_ids
        elif (
            all(expert_ids)
            and len(set(expert_ids)) == 2
            and decisions == ["revise", "revise"]
            and revised_left
            and revised_left == revised_right
        ):
            entry["review_status"] = "agreed"
            entry["target_head"] = BENCHMARK_HEAD_MAP[entry["source_head"]]
            entry["target_tokens"] = [item for item in revised_left.split("|") if item]
            entry["reviewers"] = expert_ids
        elif row.get("adjudicator_id", "").strip():
            tokens = [item for item in row.get("adjudication_target_tokens", "").split("|") if item]
            entry["review_status"] = "adjudicated" if tokens else "unsupported"
            entry["target_tokens"] = tokens
            entry["target_head"] = BENCHMARK_HEAD_MAP[entry["source_head"]] if tokens else None
            entry["reviewers"] = [item for item in expert_ids if item]
            entry["adjudicator_id"] = row["adjudicator_id"].strip()
        entries.append(entry)
    result["entries"] = entries
    result["review_packet_sha256"] = sha256_file(review_csv)
    return result


def project_payload(
    *,
    sample_id: str,
    pattern: str | None,
    source_space: str,
    source_payload: Mapping[str, Any],
    mapping: Mapping[str, Any],
    ontology: Mapping[str, Any],
) -> dict[str, Any]:
    index = {
        (entry["source_head"], entry["source_token"]): entry
        for entry in mapping.get("entries", [])
    }
    common = {head: set() for head in COMMON_HEADS}
    unsupported: list[dict[str, str]] = []
    source_count = 0
    mapped_count = 0
    for source_head, target_head in BENCHMARK_HEAD_MAP.items():
        values = source_payload.get(source_head, [])
        if not isinstance(values, list):
            raise SemanticProjectionError(f"{source_head} must be a list")
        for raw_token in sorted(set(str(item) for item in values)):
            source_count += 1
            entry = index.get((source_head, raw_token))
            if entry is None or entry.get("review_status") == "unsupported" or not entry.get("target_tokens"):
                unsupported.append(
                    {
                        "source_head": source_head,
                        "source_token": raw_token,
                        "reason": "missing_or_explicitly_unsupported_mapping",
                    }
                )
                continue
            if entry.get("target_head") != target_head:
                raise SemanticProjectionError(f"mapping head mismatch for {source_head}:{raw_token}")
            common[target_head].update(entry["target_tokens"])
            mapped_count += 1
    vocabulary = {head: set(tokens) for head, tokens in ontology["heads"].items()}
    for head, values in common.items():
        if values - vocabulary[head]:
            raise SemanticProjectionError(f"projection escaped ontology head {head}")
    return {
        "schema_version": 1,
        "sample_id": sample_id,
        "pattern": pattern,
        "source_space": source_space,
        "common_semantics": {head: sorted(values) for head, values in common.items()},
        "unsupported": unsupported,
        "coverage": {
            "source_token_count": source_count,
            "mapped_token_count": mapped_count,
            "coverage_rate": mapped_count / source_count if source_count else 1.0,
        },
    }


def project_benchmark_row(
    row: Mapping[str, Any], mapping: Mapping[str, Any], ontology: Mapping[str, Any]
) -> dict[str, Any]:
    labels = row.get("labels")
    if not isinstance(labels, Mapping):
        raise SemanticProjectionError("benchmark row lacks labels")
    return project_payload(
        sample_id=str(row["sample_id"]),
        pattern=row.get("pattern"),
        source_space="benchmark",
        source_payload=labels,
        mapping=mapping,
        ontology=ontology,
    )


def project_symbolic_row(
    row: Mapping[str, Any], mapping: Mapping[str, Any], ontology: Mapping[str, Any]
) -> dict[str, Any]:
    payload = row.get("symbolic_raw")
    if not isinstance(payload, Mapping):
        raise SemanticProjectionError("symbolic row lacks symbolic_raw")
    triggered_rule_ids = payload.get("triggered_rule_ids")
    triggered_rules = payload.get("triggered_rules")
    if triggered_rule_ids is not None and triggered_rules is not None:
        if triggered_rule_ids != triggered_rules:
            raise SemanticProjectionError(
                "symbolic_raw has conflicting triggered_rule_ids and triggered_rules"
            )
    normalized_payload = dict(payload)
    normalized_payload["triggered_rules"] = (
        triggered_rule_ids if triggered_rule_ids is not None else triggered_rules or []
    )
    return project_payload(
        sample_id=str(row["sample_id"]),
        pattern=row.get("pattern"),
        source_space="symbolic_engine",
        source_payload=normalized_payload,
        mapping=mapping,
        ontology=ontology,
    )


def project_llm_row(
    row: Mapping[str, Any], mapping: Mapping[str, Any], ontology: Mapping[str, Any]
) -> dict[str, Any]:
    nested = row.get("prediction")
    payload = nested if isinstance(nested, Mapping) else row
    return project_payload(
        sample_id=str(row["sample_id"]),
        pattern=row.get("pattern"),
        source_space="llm",
        source_payload=payload,
        mapping=mapping,
        ontology=ontology,
    )
