"""Bound prompt/schema/vocabulary assets and frozen evaluation inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Sequence

from .contracts import LoadedContract
from .datasets import observable_scene_spec
from .hashing import sha256_file, sha256_json, verify_file_sha256
from .io import iter_jsonl, read_json
from .prompting import build_core_three_user_prompt, build_fullfields_user_prompt


class ProtocolError(ValueError):
    """Raised when a prompt, schema, vocabulary, or evaluator input drifts."""


@dataclass(frozen=True)
class ProtocolAssets:
    prompt: str
    schema: dict[str, Any]
    vocabulary: dict[str, Any]
    prompt_sha256: str
    schema_sha256: str
    vocabulary_sha256: str
    parser_sha256: str
    scorer_sha256: str


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    pattern: str
    system_prompt: str
    user_prompt: str
    image_paths: tuple[Path, ...]
    source: Mapping[str, Any]


@dataclass(frozen=True)
class InferenceInput:
    """Label-blind payload passed across the model-backend boundary."""

    sample_id: str
    pattern: str
    system_prompt: str
    user_prompt: str
    image_paths: tuple[Path, ...]


def label_blind_input(sample: EvaluationSample) -> InferenceInput:
    return InferenceInput(
        sample_id=sample.sample_id,
        pattern=sample.pattern,
        system_prompt=sample.system_prompt,
        user_prompt=sample.user_prompt,
        image_paths=sample.image_paths,
    )


def _contained(base: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or PureWindowsPath(relative).is_absolute():
        raise ProtocolError(f"protocol path must be relative: {relative}")
    resolved = (base / candidate).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ProtocolError(f"protocol path escapes its root: {relative}") from exc
    return resolved


def load_protocol_assets(
    contract: LoadedContract,
    code_root: str | Path,
    output_scope: str,
) -> ProtocolAssets:
    root = Path(code_root).resolve()
    prompt_key = (
        "core_three_eval_v1"
        if output_scope == "core_three_v1"
        else "gpt_fullfields_v1"
    )
    prompt_spec = contract.data["prompt_contracts"][prompt_key]
    scope_spec = contract.data["output_scopes"][output_scope]
    vocab_key = "core_three" if output_scope == "core_three_v1" else "fullfields"
    vocab_spec = contract.data["prompt_contracts"]["vocabularies"][vocab_key]
    implementation = contract.data["implementation_bindings"][output_scope]
    prompt_path = _contained(root, prompt_spec["path"])
    schema_path = _contained(root, scope_spec["schema_path"])
    vocabulary_path = _contained(root, vocab_spec["path"])
    parser_path = _contained(root, implementation["parser_path"])
    scorer_path = _contained(root, implementation["scorer_path"])
    verify_file_sha256(prompt_path, prompt_spec["sha256"])
    verify_file_sha256(schema_path, scope_spec["schema_sha256"])
    verify_file_sha256(vocabulary_path, vocab_spec["sha256"])
    verify_file_sha256(parser_path, implementation["parser_sha256"])
    verify_file_sha256(scorer_path, implementation["scorer_sha256"])
    schema = read_json(schema_path)
    vocabulary = read_json(vocabulary_path)
    if not isinstance(schema, dict) or not isinstance(vocabulary, dict):
        raise ProtocolError("schema and vocabulary assets must be JSON objects")
    return ProtocolAssets(
        prompt=prompt_path.read_text(encoding="utf-8").strip(),
        schema=schema,
        vocabulary=vocabulary,
        prompt_sha256=sha256_file(prompt_path),
        schema_sha256=sha256_file(schema_path),
        vocabulary_sha256=sha256_file(vocabulary_path),
        parser_sha256=sha256_file(parser_path),
        scorer_sha256=sha256_file(scorer_path),
    )


def load_frozen_test_rows(
    contract: LoadedContract, benchmark_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(benchmark_root).resolve()
    spec = contract.data["benchmark"]["evaluator_view"]["test"]
    path = _contained(root, spec["path"])
    verify_file_sha256(path, spec["sha256"])
    rows = list(iter_jsonl(path))
    if len(rows) != spec["count"]:
        raise ProtocolError(f"test count drift: {len(rows)} != {spec['count']}")
    ids: set[str] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in ids:
            raise ProtocolError(f"invalid or duplicate frozen test ID: {sample_id}")
        if row.get("split") != "test":
            raise ProtocolError(f"frozen test split marker drift for {sample_id}")
        ids.add(sample_id)
    return rows


def _image_path(benchmark_root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError("multimodal evaluation requires a relative image path")
    path = _contained(benchmark_root, value)
    if not path.is_file():
        raise ProtocolError(f"evaluation image is missing: {value}")
    return path


def build_evaluation_samples(
    contract: LoadedContract,
    benchmark_root: str | Path,
    rows: Sequence[Mapping[str, Any]],
    assets: ProtocolAssets,
    *,
    input_mode: str,
    output_scope: str,
) -> list[EvaluationSample]:
    root = Path(benchmark_root).resolve()
    if input_mode not in {"text_only", "clean_text_plus_topdown_plus_radar"}:
        raise ProtocolError(f"unsupported clean evaluation input mode: {input_mode}")
    samples: list[EvaluationSample] = []
    vocabulary_fields = assets.vocabulary.get("fields")
    if not isinstance(vocabulary_fields, dict):
        raise ProtocolError("vocabulary artifact lacks fields")
    for row in rows:
        inputs = row.get("inputs")
        if not isinstance(inputs, Mapping) or not isinstance(inputs.get("scene_spec"), Mapping):
            raise ProtocolError(f"scene_spec missing for {row.get('sample_id')}")
        pattern = str(row.get("pattern") or "")
        payload = {
            "family": pattern,
            "scene_spec": observable_scene_spec(contract, inputs["scene_spec"]),
        }
        user_prompt = (
            build_core_three_user_prompt(payload, vocabulary_fields)
            if output_scope == "core_three_v1"
            else build_fullfields_user_prompt(payload, assets.vocabulary)
        )
        images: tuple[Path, ...] = ()
        if input_mode == "clean_text_plus_topdown_plus_radar":
            topdown = row.get("topdown_image") or inputs.get("topdown_image")
            radar = row.get("radar_image") or inputs.get("radar_image")
            images = (_image_path(root, topdown), _image_path(root, radar))
        samples.append(
            EvaluationSample(
                sample_id=str(row["sample_id"]),
                pattern=pattern,
                system_prompt=assets.prompt,
                user_prompt=user_prompt,
                image_paths=images,
                source=row,
            )
        )
    return samples


def build_inference_messages(
    sample: InferenceInput, image_objects: Sequence[Any]
) -> list[dict[str, Any]]:
    if len(image_objects) != len(sample.image_paths):
        raise ProtocolError("image object count does not match the bound evaluation sample")
    content = [{"type": "image", "image": image} for image in image_objects]
    content.append(
        {
            "type": "text",
            "text": (
                "System instructions:\n"
                f"{sample.system_prompt}\n\n"
                "User request:\n"
                f"{sample.user_prompt}"
            ),
        }
    )
    return [{"role": "user", "content": content}]


def protocol_binding_identity(assets: ProtocolAssets, decode_profile: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "prompt_sha256": assets.prompt_sha256,
            "schema_sha256": assets.schema_sha256,
            "vocabulary_sha256": assets.vocabulary_sha256,
            "parser_sha256": assets.parser_sha256,
            "scorer_sha256": assets.scorer_sha256,
            "decode_profile": dict(decode_profile),
        }
    )
