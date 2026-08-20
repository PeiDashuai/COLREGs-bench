"""Training-data adapters for the four controlled fine-tuning regimes."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping, Sequence

from .contracts import LoadedContract
from .hashing import sha256_file, sha256_json, sha256_ordered_ids, sha256_text, verify_file_sha256
from .io import iter_jsonl, read_json, write_json_atomic
from .prompting import build_core_three_user_prompt


SCENE_REGIMES = {
    "final_only_ft": ("maneuver_allowed",),
    "structured_core_three_ft": (
        "triggered_rules",
        "maneuver_allowed",
        "maneuver_forbidden",
    ),
    "structured_multimodal_ft": (
        "triggered_rules",
        "maneuver_allowed",
        "maneuver_forbidden",
    ),
}
SCENE_SYSTEM_PROMPT = (
    "You are a maritime COLREGs benchmark model. Use only the supplied scene input. "
    "Return exactly one valid JSON object with the requested fields and no commentary."
)
COLREGS_TEXT_SYSTEM_PROMPT = (
    "You are a COLREGs rule-normalization assistant. Use only the supplied atomized "
    "COLREGs source material. Return exactly one valid JSON object and no commentary."
)
FORBIDDEN_SCENE_INPUT_TERMS = (
    '"labels"',
    '"target_summaries"',
    '"primitive_ledger_summary"',
    '"suppression_records"',
    '"explanation_steps"',
)
FORBIDDEN_COLREGS_TEXT_TERMS = (
    "sample_id",
    "scene_spec",
    "topdown_image",
    "radar_image",
    "target_summaries",
    "primitive_ledger",
    "suppression_records",
    "explanation_steps",
)


class AdapterValidationError(ValueError):
    """Raised when a generated training dataset violates the locked contract."""


@dataclass(frozen=True)
class AtomicRule:
    source_index: int
    raw_id: str
    article: str
    summary: str
    text_ref: str
    applicability: Any
    deontic_role: str
    deontic_modality: str
    allowed_maneuvers: tuple[str, ...]
    forbidden_maneuvers: tuple[str, ...]
    lights_required: tuple[str, ...]
    sounds_required: tuple[str, ...]
    evidence_requirements: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _regime_spec(contract: LoadedContract, regime_id: str) -> dict[str, Any]:
    matches = [item for item in contract.data["training"]["regimes"] if item["regime_id"] == regime_id]
    if len(matches) != 1:
        raise AdapterValidationError(f"contract must declare exactly one regime {regime_id}")
    return matches[0]


def _load_row_schema(schema_path: str | Path) -> tuple[dict[str, Any], Any]:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise AdapterValidationError("jsonschema is required by pyproject.toml") from exc
    schema = read_json(schema_path)
    if not isinstance(schema, dict):
        raise AdapterValidationError("training row schema root must be an object")
    Draft202012Validator.check_schema(schema)
    return schema, Draft202012Validator(schema)


def _verify_training_row_schema(contract: LoadedContract, schema_path: str | Path) -> None:
    schema_contract = contract.data["training"]["training_row_contract"]
    verify_file_sha256(schema_path, schema_contract["schema_sha256"])


def _project_fields(source: Mapping[str, Any], allowed_fields: Sequence[str]) -> dict[str, Any]:
    return {field: source[field] for field in allowed_fields if field in source}


def observable_scene_spec(
    contract: LoadedContract,
    scene_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a frozen generator record onto model-observable scene evidence."""

    projection = contract.data["training"]["scene_input_projection"]
    top_level_fields = projection["top_level_fields"]
    area_fields = projection["area_context_fields"]
    vessel_fields = projection["vessel_fields"]
    projected: dict[str, Any] = {}
    for field in top_level_fields:
        if field not in scene_spec:
            continue
        value = scene_spec[field]
        if field == "area_context":
            if not isinstance(value, Mapping):
                raise AdapterValidationError("scene area_context must be an object")
            projected[field] = _project_fields(value, area_fields)
        elif field == "ownship":
            if not isinstance(value, Mapping):
                raise AdapterValidationError("scene ownship must be an object")
            projected[field] = _project_fields(value, vessel_fields)
        elif field == "targets":
            if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
                raise AdapterValidationError("scene targets must be a list of objects")
            projected[field] = [_project_fields(item, vessel_fields) for item in value]
        else:
            projected[field] = value

    if not isinstance(projected.get("ownship"), dict) or not projected["ownship"]:
        raise AdapterValidationError("observable scene projection requires ownship evidence")
    if not isinstance(projected.get("targets"), list):
        raise AdapterValidationError("observable scene projection requires target evidence")
    for forbidden in projection["excluded_generator_fields"]:
        if _contains_key(projected, forbidden):
            raise AdapterValidationError(
                f"observable scene projection retained generator field: {forbidden}"
            )
    return projected


def load_benchmark_splits(
    contract: LoadedContract,
    benchmark_root: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    root = Path(benchmark_root).resolve()
    evaluator = contract.data["benchmark"]["evaluator_view"]
    splits: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for split in ("train", "val", "test"):
        spec = evaluator[split]
        path = (root / spec["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AdapterValidationError(f"benchmark split escapes root: {path}") from exc
        verify_file_sha256(path, spec["sha256"])
        rows = list(iter_jsonl(path))
        if len(rows) != spec["count"]:
            raise AdapterValidationError(
                f"{split} count mismatch: expected {spec['count']}, got {len(rows)}"
            )
        split_ids: set[str] = set()
        for row in rows:
            sample_id = row.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise AdapterValidationError(f"{split} contains a row without sample_id")
            if sample_id in split_ids or sample_id in all_ids:
                raise AdapterValidationError(f"duplicate sample_id across frozen splits: {sample_id}")
            if row.get("split") != split:
                raise AdapterValidationError(
                    f"split field mismatch for {sample_id}: {row.get('split')} != {split}"
                )
            split_ids.add(sample_id)
            all_ids.add(sample_id)
        splits[split] = rows
    return splits


def _relative_image_path(benchmark_root: Path, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise AdapterValidationError("multimodal row requires a non-empty relative image path")
    path = Path(value)
    if path.is_absolute() or PureWindowsPath(value).is_absolute():
        raise AdapterValidationError(f"absolute image paths are forbidden: {value}")
    resolved = (benchmark_root / path).resolve()
    try:
        resolved.relative_to(benchmark_root)
    except ValueError as exc:
        raise AdapterValidationError(f"image path escapes benchmark root: {value}") from exc
    if not resolved.is_file():
        raise AdapterValidationError(f"image file is missing: {value}")
    return path.as_posix()


def _scene_training_row(
    contract: LoadedContract,
    benchmark_root: Path,
    source: Mapping[str, Any],
    regime_id: str,
    split: str,
    core_prompt: str | None = None,
    core_vocabulary_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields = SCENE_REGIMES[regime_id]
    source_id = source["sample_id"]
    inputs = source.get("inputs")
    labels = source.get("labels")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("scene_spec"), dict):
        raise AdapterValidationError(f"scene_spec is missing for {source_id}")
    if not isinstance(labels, dict):
        raise AdapterValidationError(f"labels are missing for {source_id}")

    multimodal = regime_id == "structured_multimodal_ft"
    projection_id = contract.data["training"]["scene_input_projection"]["projection_id"]
    payload = {
        "family": source.get("pattern"),
        "scene_spec": observable_scene_spec(contract, inputs["scene_spec"]),
    }
    if regime_id in {"structured_core_three_ft", "structured_multimodal_ft"}:
        if core_prompt is None or core_vocabulary_fields is None:
            raise AdapterValidationError(f"core prompt assets are required for {regime_id}")
        system_prompt = core_prompt
        user_prompt = build_core_three_user_prompt(payload, core_vocabulary_fields)
    else:
        system_prompt = SCENE_SYSTEM_PROMPT
        user_prompt = (
            f"Scene input:\n{_canonical_json(payload)}\n\n"
            f"Required output fields: {_canonical_json(list(fields))}\n"
            "Use the scene specification only; no images are available. "
            "Return only the JSON object."
        )
    assistant = {field: list(labels.get(field, []) or []) for field in fields}
    images: list[dict[str, str]] = []
    if multimodal:
        topdown = source.get("topdown_image") or inputs.get("topdown_image")
        radar = source.get("radar_image") or inputs.get("radar_image")
        images = [
            {"role": "topdown", "path": _relative_image_path(benchmark_root, topdown)},
            {"role": "radar", "path": _relative_image_path(benchmark_root, radar)},
        ]
    return {
        "record_id": f"{regime_id}:{split}:{source_id}",
        "source_id": source_id,
        "regime": regime_id,
        "split": split,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "assistant_json": _canonical_json(assistant),
        "images": images,
        "metadata": {
            "family": source.get("pattern"),
            "source_split": split,
            "input_contract": projection_id,
            "assistant_fields": list(fields),
            "image_roles": [item["role"] for item in images],
        },
    }


def _contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(_contains_key(child, forbidden) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def _validate_row_schema(rows: Iterable[dict[str, Any]], validator: Any) -> list[str]:
    violations: list[str] = []
    for row in rows:
        errors = sorted(validator.iter_errors(row), key=lambda item: list(item.absolute_path))
        for error in errors[:3]:
            location = "/".join(str(part) for part in error.absolute_path) or "<root>"
            violations.append(f"{row.get('record_id')}:{location}:{error.message}")
        if len(violations) >= 50:
            break
    return violations


def _audit_scene_rows(
    contract: LoadedContract,
    regime_id: str,
    rows_by_split: Mapping[str, Sequence[dict[str, Any]]],
    test_ids: set[str],
    validator: Any,
) -> dict[str, Any]:
    spec = _regime_spec(contract, regime_id)
    violations = _validate_row_schema(
        [row for split_rows in rows_by_split.values() for row in split_rows], validator
    )
    record_ids: set[str] = set()
    source_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    expected_fields = set(SCENE_REGIMES[regime_id])
    multimodal = regime_id == "structured_multimodal_ft"
    projection = contract.data["training"]["scene_input_projection"]
    projection_id = projection["projection_id"]
    proxy_terms = tuple(f'"{field}"' for field in projection["excluded_generator_fields"])

    for split in ("train", "val"):
        rows = rows_by_split[split]
        expected_count = spec[f"expected_{split}_rows"]
        if len(rows) != expected_count:
            violations.append(f"{split}:count:{len(rows)}!={expected_count}")
        for row in rows:
            record_id = row["record_id"]
            if record_id in record_ids:
                violations.append(f"duplicate_record_id:{record_id}")
            record_ids.add(record_id)
            source_id = row["source_id"]
            source_ids[split].add(source_id)
            if source_id in test_ids:
                violations.append(f"test_overlap:{source_id}")
            try:
                assistant = json.loads(row["assistant_json"])
            except json.JSONDecodeError as exc:
                violations.append(f"assistant_json:{record_id}:{exc}")
                continue
            if set(assistant) != expected_fields:
                violations.append(f"assistant_fields:{record_id}:{sorted(assistant)}")
            if _contains_key(assistant, "sample_id"):
                violations.append(f"assistant_sample_id_leak:{record_id}")
            prompt = row["system_prompt"] + "\n" + row["user_prompt"]
            if source_id in prompt:
                violations.append(f"prompt_sample_id_leak:{record_id}")
            for term in FORBIDDEN_SCENE_INPUT_TERMS:
                if term in prompt:
                    violations.append(f"prompt_gold_structure_leak:{record_id}:{term}")
            for term in proxy_terms:
                if term in prompt:
                    violations.append(f"prompt_generator_proxy_leak:{record_id}:{term}")
            if row["metadata"].get("input_contract") != projection_id:
                violations.append(f"input_projection:{record_id}")
            if multimodal:
                roles = [item.get("role") for item in row["images"]]
                if roles != ["topdown", "radar"]:
                    violations.append(f"image_roles:{record_id}:{roles}")
            elif row["images"]:
                violations.append(f"text_row_has_images:{record_id}")
            if not multimodal and any(
                token in row["user_prompt"] for token in (".png", "topdown_image", "radar_image")
            ):
                violations.append(f"text_prompt_has_image_path:{record_id}")

    overlap = source_ids["train"] & source_ids["val"]
    if overlap:
        violations.append(f"train_val_source_overlap:{sorted(overlap)[:5]}")
    audit = {
        "status": "PASS" if not violations else "FAIL",
        "regime_id": regime_id,
        "counts": {split: len(rows_by_split[split]) for split in ("train", "val")},
        "unique_source_counts": {split: len(source_ids[split]) for split in ("train", "val")},
        "test_overlap_count": len((source_ids["train"] | source_ids["val"]) & test_ids),
        "train_val_source_overlap_count": len(overlap),
        "images_per_row": spec["images_per_row"],
        "assistant_fields": list(SCENE_REGIMES[regime_id]),
        "input_projection": projection_id,
        "proxy_leakage_count": sum("proxy_leak" in item for item in violations),
        "target_leakage_count": sum("leak" in item for item in violations),
        "violation_count": len(violations),
        "violations_preview": violations[:50],
    }
    if violations:
        raise AdapterValidationError(f"{regime_id} audit failed: {violations[:5]}")
    return audit


def build_scene_rows(
    contract: LoadedContract,
    benchmark_root: str | Path,
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    regime_id: str,
    training_row_schema: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if regime_id not in SCENE_REGIMES:
        raise ValueError(f"not a scene training regime: {regime_id}")
    spec = _regime_spec(contract, regime_id)
    projection_id = contract.data["training"]["scene_input_projection"]["projection_id"]
    if tuple(spec.get("assistant_fields", ())) != SCENE_REGIMES[regime_id]:
        raise AdapterValidationError(f"assistant fields drifted for {regime_id}")
    if spec.get("input_projection") != projection_id:
        raise AdapterValidationError(f"input projection drifted for {regime_id}")
    expected_image_count = 2 if regime_id == "structured_multimodal_ft" else 0
    if spec.get("images_per_row") != expected_image_count:
        raise AdapterValidationError(f"image contract drifted for {regime_id}")
    root = Path(benchmark_root).resolve()
    _verify_training_row_schema(contract, training_row_schema)
    _, validator = _load_row_schema(training_row_schema)
    core_prompt = None
    core_vocabulary_fields = None
    core_prompt_sha256 = None
    core_vocabulary_sha256 = None
    if regime_id in {"structured_core_three_ft", "structured_multimodal_ft"}:
        code_root = contract.path.parent.parent
        prompt_spec = contract.data["prompt_contracts"]["core_three_eval_v1"]
        vocabulary_spec = contract.data["prompt_contracts"]["vocabularies"]["core_three"]
        prompt_path = (code_root / prompt_spec["path"]).resolve()
        vocabulary_path = (code_root / vocabulary_spec["path"]).resolve()
        verify_file_sha256(prompt_path, prompt_spec["sha256"])
        verify_file_sha256(vocabulary_path, vocabulary_spec["sha256"])
        core_prompt = prompt_path.read_text(encoding="utf-8")
        vocabulary = read_json(vocabulary_path)
        core_vocabulary_fields = vocabulary.get("fields")
        if not isinstance(core_vocabulary_fields, dict):
            raise AdapterValidationError("core vocabulary artifact lacks fields")
        core_prompt_sha256 = sha256_file(prompt_path)
        core_vocabulary_sha256 = sha256_file(vocabulary_path)
    rows_by_split = {
        split: [
            _scene_training_row(
                contract,
                root,
                source,
                regime_id,
                split,
                core_prompt=core_prompt,
                core_vocabulary_fields=core_vocabulary_fields,
            )
            for source in splits[split]
        ]
        for split in ("train", "val")
    }
    test_ids = {str(row["sample_id"]) for row in splits["test"]}
    audit = _audit_scene_rows(contract, regime_id, rows_by_split, test_ids, validator)
    audit["training_eval_prompt_aligned"] = core_prompt is not None
    audit["training_prompt_sha256"] = core_prompt_sha256
    audit["training_vocabulary_sha256"] = core_vocabulary_sha256
    return rows_by_split, audit


def _load_yaml(path: Path) -> Any:
    try:
        import yaml
    except ImportError as exc:
        raise AdapterValidationError("PyYAML is required by pyproject.toml") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _action_tokens(value: Any) -> tuple[str, ...]:
    tokens: list[str] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item:
                tokens.append(item)
            elif isinstance(item, dict):
                for key in ("primitive", "id", "name", "action", "token"):
                    token = item.get(key)
                    if isinstance(token, str) and token:
                        tokens.append(token)
                        break
    elif isinstance(value, dict):
        for child in value.values():
            tokens.extend(_action_tokens(child))
    return tuple(dict.fromkeys(tokens))


def _parse_atomic_rules(path: Path) -> list[AtomicRule]:
    data = _load_yaml(path)
    if not isinstance(data, list):
        raise AdapterValidationError("atomic YAML root must be a list")
    rules: list[AtomicRule] = []
    seen: set[str] = set()
    for index, raw in enumerate(data):
        if not isinstance(raw, dict):
            raise AdapterValidationError(f"atomic rule {index} is not an object")
        raw_id = str(raw.get("id") or "").strip()
        if not raw_id or raw_id in seen:
            raise AdapterValidationError(f"missing or duplicate atomic rule ID: {raw_id!r}")
        seen.add(raw_id)
        deontic = raw.get("deontic") if isinstance(raw.get("deontic"), dict) else {}
        actions = raw.get("actions") if isinstance(raw.get("actions"), dict) else {}
        maneuver = actions.get("maneuver") if isinstance(actions.get("maneuver"), dict) else {}
        signals = actions.get("signals") if isinstance(actions.get("signals"), dict) else {}
        rules.append(
            AtomicRule(
                source_index=index,
                raw_id=raw_id,
                article=str(raw.get("article") or "").strip(),
                summary=str(raw.get("summary") or "").strip(),
                text_ref=str(raw.get("text_ref") or raw.get("summary") or "").strip(),
                applicability=raw.get("applicability", {}),
                deontic_role=str(deontic.get("role") or "").strip(),
                deontic_modality=str(deontic.get("modality") or "").strip(),
                allowed_maneuvers=_action_tokens(maneuver.get("allowed", [])),
                forbidden_maneuvers=_action_tokens(maneuver.get("forbidden", [])),
                lights_required=_action_tokens(signals.get("lights_shapes_required", [])),
                sounds_required=_action_tokens(signals.get("sounds_required", [])),
                evidence_requirements=tuple(
                    str(item) for item in raw.get("evidence_requirements", []) if str(item)
                ),
            )
        )
    return rules


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, child_prefix))
    return flattened


def _operational_definitions(rule: AtomicRule, parameters: Mapping[str, Any]) -> dict[str, Any]:
    text = " ".join(
        [
            rule.raw_id,
            rule.article,
            rule.summary,
            _canonical_json(rule.applicability),
            " ".join(rule.evidence_requirements),
            " ".join(rule.allowed_maneuvers),
            " ".join(rule.forbidden_maneuvers),
        ]
    ).lower()
    groups = {
        "risk": ("risk", "tcpa", "cpa", "collision"),
        "imminent": ("imminent",),
        "early_substantial": ("early", "substantial", "positive_action"),
        "safe_speed": ("safe speed", "safe_speed", "restricted", "visibility"),
        "tss_crossing": ("tss", "traffic separation", "right angle"),
        "impede": ("impede", "channel", "narrow"),
    }
    selected_groups = {
        group for group, keywords in groups.items() if any(keyword in text for keyword in keywords)
    }
    return {
        key: value
        for key, value in _flatten(parameters).items()
        if key.split(".", 1)[0] in selected_groups
    }


def _source_group(article: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", article.lower()).strip("_")
    if not normalized:
        raise AdapterValidationError("atomic rule article cannot be empty")
    return normalized


def _instruction_records(
    rules: Sequence[AtomicRule],
    operational_parameters: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rule in rules:
        common = {
            "source_group": _source_group(rule.article),
            "source_rule_ids": [rule.raw_id],
            "article": rule.article,
        }
        records.extend(
            [
                {
                    **common,
                    "instruction_id": f"{rule.raw_id}::rule_normalization",
                    "instruction_type": "rule_normalization",
                    "input": {
                        "rule_text": rule.text_ref,
                        "article": rule.article,
                        "summary": rule.summary,
                    },
                    "output": {
                        "rule_id": rule.raw_id,
                        "article": rule.article,
                        "summary": rule.summary,
                        "deontic_role": rule.deontic_role,
                        "deontic_modality": rule.deontic_modality,
                    },
                },
                {
                    **common,
                    "instruction_id": f"{rule.raw_id}::applicability_abstraction",
                    "instruction_type": "applicability_abstraction",
                    "input": {
                        "rule_id": rule.raw_id,
                        "condition_expression": rule.applicability,
                        "evidence_requirements": list(rule.evidence_requirements),
                    },
                    "output": {
                        "rule_id": rule.raw_id,
                        "applicability": rule.applicability,
                        "evidence_requirements": list(rule.evidence_requirements),
                        "operational_definitions": _operational_definitions(
                            rule, operational_parameters
                        ),
                    },
                },
                {
                    **common,
                    "instruction_id": f"{rule.raw_id}::primitive_consequence_mapping",
                    "instruction_type": "primitive_consequence_mapping",
                    "input": {
                        "rule_id": rule.raw_id,
                        "rule_text": rule.text_ref,
                        "deontic_role": rule.deontic_role,
                        "deontic_modality": rule.deontic_modality,
                    },
                    "output": {
                        "rule_id": rule.raw_id,
                        "allowed_maneuvers": list(rule.allowed_maneuvers),
                        "forbidden_maneuvers": list(rule.forbidden_maneuvers),
                        "lights_required": list(rule.lights_required),
                        "sounds_required": list(rule.sounds_required),
                    },
                },
            ]
        )

    by_article: dict[str, list[AtomicRule]] = defaultdict(list)
    for rule in rules:
        by_article[rule.article].append(rule)
    for article in sorted(by_article):
        group = sorted(by_article[article], key=lambda item: item.raw_id)
        for left, right in zip(group, group[1:]):
            records.append(
                {
                    "source_group": _source_group(article),
                    "source_rule_ids": [left.raw_id, right.raw_id],
                    "article": article,
                    "instruction_id": (
                        f"{left.raw_id}__vs__{right.raw_id}::deontic_disambiguation"
                    ),
                    "instruction_type": "deontic_disambiguation",
                    "input": {
                        "rule_a": {
                            "rule_id": left.raw_id,
                            "summary": left.summary,
                            "applicability": left.applicability,
                            "deontic_role": left.deontic_role,
                            "deontic_modality": left.deontic_modality,
                        },
                        "rule_b": {
                            "rule_id": right.raw_id,
                            "summary": right.summary,
                            "applicability": right.applicability,
                            "deontic_role": right.deontic_role,
                            "deontic_modality": right.deontic_modality,
                        },
                    },
                    "output": {
                        "difference": {
                            "rule_a": left.raw_id,
                            "rule_b": right.raw_id,
                            "applicability_a": left.applicability,
                            "applicability_b": right.applicability,
                            "deontic_role_a": left.deontic_role,
                            "deontic_role_b": right.deontic_role,
                            "consequence_a": {
                                "allowed_maneuvers": list(left.allowed_maneuvers),
                                "forbidden_maneuvers": list(left.forbidden_maneuvers),
                            },
                            "consequence_b": {
                                "allowed_maneuvers": list(right.allowed_maneuvers),
                                "forbidden_maneuvers": list(right.forbidden_maneuvers),
                            },
                        }
                    },
                }
            )
    return records


def _select_validation_groups(
    records: Sequence[Mapping[str, Any]],
    target_rows: int,
    seed: int,
) -> set[str]:
    counts = Counter(str(record["source_group"]) for record in records)
    ordered_groups = sorted(counts, key=lambda group: (sha256_text(f"{seed}:{group}"), group))
    possibilities: dict[int, tuple[str, ...]] = {0: ()}
    for group in ordered_groups:
        additions: dict[int, tuple[str, ...]] = {}
        for total, selected in possibilities.items():
            new_total = total + counts[group]
            candidate = selected + (group,)
            if new_total not in possibilities and (
                new_total not in additions or candidate < additions[new_total]
            ):
                additions[new_total] = candidate
        possibilities.update(additions)
    best_total = min(
        (total for total in possibilities if total > 0),
        key=lambda total: (abs(total - target_rows), total, possibilities[total]),
    )
    return set(possibilities[best_total])


def _colregs_training_row(record: Mapping[str, Any], split: str) -> dict[str, Any]:
    instruction_id = str(record["instruction_id"])
    user_prompt = (
        f"Instruction type: {record['instruction_type']}\n"
        f"Input material:\n{_canonical_json(record['input'])}\n\n"
        "Return only the structured JSON output."
    )
    return {
        "record_id": f"colregs_text_ft:{split}:{instruction_id}",
        "source_id": instruction_id,
        "regime": "colregs_text_ft",
        "split": split,
        "system_prompt": COLREGS_TEXT_SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "assistant_json": _canonical_json(record["output"]),
        "images": [],
        "metadata": {
            "source_group": record["source_group"],
            "source_rule_ids": list(record["source_rule_ids"]),
            "article": record["article"],
            "instruction_type": record["instruction_type"],
            "assistant_target_contract": "rule_centric_v1_source_native",
        },
    }


def build_colregs_text_rows(
    contract: LoadedContract,
    atomic_yaml: str | Path,
    operational_params_yaml: str | Path,
    training_row_schema: str | Path,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    spec = _regime_spec(contract, "colregs_text_ft")
    _verify_training_row_schema(contract, training_row_schema)
    atomic_path = Path(atomic_yaml).resolve()
    params_path = Path(operational_params_yaml).resolve()
    verify_file_sha256(atomic_path, spec["source_artifacts"]["atomic_yaml_sha256"])
    verify_file_sha256(
        params_path, spec["source_artifacts"]["operational_params_yaml_sha256"]
    )
    rules = _parse_atomic_rules(atomic_path)
    if len(rules) != spec["atomic_rule_count"]:
        raise AdapterValidationError(
            f"atomic rule count mismatch: {len(rules)} != {spec['atomic_rule_count']}"
        )
    parameters = _load_yaml(params_path)
    if not isinstance(parameters, dict):
        raise AdapterValidationError("operational parameter YAML root must be an object")
    records = _instruction_records(rules, parameters)
    if len(records) != spec["instruction_row_count"]:
        raise AdapterValidationError(
            f"instruction count mismatch: {len(records)} != {spec['instruction_row_count']}"
        )
    validation_groups = _select_validation_groups(
        records, spec["expected_val_rows"], spec["split_seed"]
    )
    rows_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for record in records:
        split = "val" if record["source_group"] in validation_groups else "train"
        rows_by_split[split].append(_colregs_training_row(record, split))

    _, validator = _load_row_schema(training_row_schema)
    violations = _validate_row_schema(
        rows_by_split["train"] + rows_by_split["val"], validator
    )
    train_rules = {
        rule_id
        for row in rows_by_split["train"]
        for rule_id in row["metadata"]["source_rule_ids"]
    }
    val_rules = {
        rule_id
        for row in rows_by_split["val"]
        for rule_id in row["metadata"]["source_rule_ids"]
    }
    train_groups = {row["metadata"]["source_group"] for row in rows_by_split["train"]}
    val_groups = {row["metadata"]["source_group"] for row in rows_by_split["val"]}
    if len(rows_by_split["train"]) != spec["expected_train_rows"]:
        violations.append("colregs_text_train_count")
    if len(rows_by_split["val"]) != spec["expected_val_rows"]:
        violations.append("colregs_text_val_count")
    if train_rules & val_rules:
        violations.append("colregs_text_source_rule_overlap")
    if train_groups & val_groups:
        violations.append("colregs_text_source_group_overlap")
    if train_rules | val_rules != {rule.raw_id for rule in rules}:
        violations.append("colregs_text_rule_coverage")
    for row in rows_by_split["train"] + rows_by_split["val"]:
        if row["images"]:
            violations.append(f"colregs_text_image:{row['record_id']}")
        assistant = json.loads(row["assistant_json"])
        if _contains_key(assistant, "sample_id"):
            violations.append(f"colregs_text_sample_id:{row['record_id']}")
        serialized = (row["system_prompt"] + "\n" + row["user_prompt"]).lower()
        for term in FORBIDDEN_COLREGS_TEXT_TERMS:
            if term in serialized:
                violations.append(f"colregs_text_scene_leak:{row['record_id']}:{term}")

    audit = {
        "status": "PASS" if not violations else "FAIL",
        "regime_id": "colregs_text_ft",
        "atomic_rule_count": len(rules),
        "instruction_row_count": len(records),
        "counts": {split: len(rows_by_split[split]) for split in ("train", "val")},
        "instruction_type_counts": dict(
            Counter(row["metadata"]["instruction_type"] for split in rows_by_split.values() for row in split)
        ),
        "train_source_group_count": len(train_groups),
        "val_source_group_count": len(val_groups),
        "train_val_source_group_overlap_count": len(train_groups & val_groups),
        "train_val_source_rule_overlap_count": len(train_rules & val_rules),
        "covered_atomic_rule_count": len(train_rules | val_rules),
        "validation_source_groups": sorted(validation_groups),
        "source_native_tokens": True,
        "legacy_fuzzy_canonicalization_used": False,
        "violation_count": len(violations),
        "violations_preview": violations[:50],
    }
    if violations:
        raise AdapterValidationError(f"colregs_text_ft audit failed: {violations[:5]}")
    return rows_by_split, audit


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite dataset artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(_canonical_json(dict(row)) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite dataset artifact: {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _materialize_regime(
    contract: LoadedContract,
    regime_id: str,
    output_dir: Path,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    audit: Mapping[str, Any],
    row_schema_path: Path,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    output_dir.mkdir(parents=False, exist_ok=False)
    implementation = {
        "dataset_adapter_module_sha256": sha256_file(Path(__file__).resolve()),
    }
    split_artifacts: dict[str, Any] = {}
    for split in ("train", "val"):
        path = output_dir / f"{split}.jsonl"
        rows = list(rows_by_split[split])
        _write_jsonl_atomic(path, rows)
        split_artifacts[split] = {
            "path": f"{regime_id}/{split}.jsonl",
            "row_count": len(rows),
            "sha256": sha256_file(path),
            "ordered_record_id_sha256": sha256_ordered_ids(
                str(row["record_id"]) for row in rows
            ),
            "ordered_source_id_sha256": sha256_ordered_ids(
                str(row["source_id"]) for row in rows
            ),
        }
    write_json_atomic(output_dir / "audit_report.json", dict(audit))
    identity = {
        "regime_id": regime_id,
        "contract_sha256": contract.sha256,
        "training_row_schema_sha256": sha256_file(row_schema_path),
        "implementation": implementation,
        "source_hashes": dict(source_hashes),
        "splits": split_artifacts,
    }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract.data["contract_id"],
        "contract_version": contract.data["contract_version"],
        "contract_sha256": contract.sha256,
        "regime_id": regime_id,
        "formal_result": False,
        "training_row_schema": "schemas/training_row_v1.schema.json",
        "training_row_schema_sha256": sha256_file(row_schema_path),
        "implementation": implementation,
        "source_hashes": dict(source_hashes),
        "splits": split_artifacts,
        "audit_status": audit["status"],
        "content_identity_sha256": sha256_json(identity),
    }
    write_json_atomic(output_dir / "dataset_manifest.json", manifest)
    return manifest


def build_all_training_adapters(
    contract: LoadedContract,
    benchmark_root: str | Path,
    atomic_yaml: str | Path,
    operational_params_yaml: str | Path,
    output_root: str | Path,
    training_row_schema: str | Path,
) -> dict[str, Any]:
    benchmark = Path(benchmark_root).resolve()
    output = Path(output_root).resolve()
    schema_path = Path(training_row_schema).resolve()
    _verify_training_row_schema(contract, schema_path)
    try:
        output.relative_to(benchmark)
    except ValueError:
        pass
    else:
        raise AdapterValidationError("training artifacts must remain outside frozen benchmark root")
    if output.exists():
        raise FileExistsError(f"output root already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    splits = load_benchmark_splits(contract, benchmark)
    evaluator = contract.data["benchmark"]["evaluator_view"]
    scene_source_hashes = {
        f"evaluator_{split}_sha256": evaluator[split]["sha256"]
        for split in ("train", "val", "test")
    }
    manifests: dict[str, Any] = {}
    for regime_id in (
        "final_only_ft",
        "structured_core_three_ft",
        "structured_multimodal_ft",
    ):
        rows, audit = build_scene_rows(
            contract, benchmark, splits, regime_id, schema_path
        )
        source_hashes = dict(scene_source_hashes)
        if audit.get("training_eval_prompt_aligned"):
            source_hashes["training_prompt_sha256"] = audit["training_prompt_sha256"]
            source_hashes["training_vocabulary_sha256"] = audit[
                "training_vocabulary_sha256"
            ]
        manifests[regime_id] = _materialize_regime(
            contract,
            regime_id,
            output / regime_id,
            rows,
            audit,
            schema_path,
            source_hashes,
        )

    colregs_rows, colregs_audit = build_colregs_text_rows(
        contract, atomic_yaml, operational_params_yaml, schema_path
    )
    colregs_spec = _regime_spec(contract, "colregs_text_ft")
    colregs_source_hashes = {
        "atomic_yaml_sha256": colregs_spec["source_artifacts"]["atomic_yaml_sha256"],
        "operational_params_yaml_sha256": colregs_spec["source_artifacts"][
            "operational_params_yaml_sha256"
        ],
    }
    manifests["colregs_text_ft"] = _materialize_regime(
        contract,
        "colregs_text_ft",
        output / "colregs_text_ft",
        colregs_rows,
        colregs_audit,
        schema_path,
        colregs_source_hashes,
    )
    master = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_id": contract.data["contract_id"],
        "contract_version": contract.data["contract_version"],
        "contract_sha256": contract.sha256,
        "formal_result": False,
        "regime_count": 4,
        "regimes": {
            regime_id: {
                "path": regime_id,
                "content_identity_sha256": manifest["content_identity_sha256"],
                "audit_status": manifest["audit_status"],
            }
            for regime_id, manifest in manifests.items()
        },
        "status": "PASS",
    }
    write_json_atomic(output / "training_adapters_manifest.json", master)
    return master
