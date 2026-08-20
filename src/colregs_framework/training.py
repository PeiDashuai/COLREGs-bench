"""Contracts and validation for the shared Gemma-3 trainer."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from statistics import mean
from typing import Any, Mapping, Sequence

from .contracts import LoadedContract
from .hashing import sha256_file, sha256_json
from .io import JsonFormatError, iter_jsonl, read_json


IGNORE_INDEX = -100
TRAINING_REGIME_ORDER = (
    "colregs_text_ft",
    "final_only_ft",
    "structured_core_three_ft",
    "structured_multimodal_ft",
)


class TrainingContractError(ValueError):
    """Raised when trainer inputs or resolved settings violate the contract."""


@dataclass(frozen=True)
class BoundTrainingDataset:
    regime_id: str
    root: Path
    train_path: Path
    val_path: Path
    train_rows: tuple[dict[str, Any], ...]
    val_rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    @property
    def content_identity(self) -> str:
        return str(self.manifest["content_identity_sha256"])


def _regime_spec(contract: LoadedContract, regime_id: str) -> dict[str, Any]:
    matches = [
        item for item in contract.data["training"]["regimes"]
        if item["regime_id"] == regime_id
    ]
    if len(matches) != 1:
        raise TrainingContractError(f"contract must declare exactly one regime {regime_id}")
    return matches[0]


def _resolve_contained(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or PureWindowsPath(relative).is_absolute():
        raise TrainingContractError(f"absolute artifact path is forbidden: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TrainingContractError(f"artifact path escapes root: {relative}") from exc
    return resolved


def _dataset_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "regime_id": manifest["regime_id"],
        "contract_sha256": manifest["contract_sha256"],
        "training_row_schema_sha256": manifest["training_row_schema_sha256"],
        "implementation": manifest["implementation"],
        "source_hashes": manifest["source_hashes"],
        "splits": manifest["splits"],
    }


def load_training_dataset(
    contract: LoadedContract,
    adapters_root: str | Path,
    regime_id: str,
) -> BoundTrainingDataset:
    if regime_id not in TRAINING_REGIME_ORDER:
        raise TrainingContractError(f"unknown training regime: {regime_id}")
    root = Path(adapters_root).resolve()
    master = read_json(root / "training_adapters_manifest.json")
    if master.get("status") != "PASS" or master.get("formal_result") is not False:
        raise TrainingContractError("training-data master manifest must be PASS and non-formal")
    for key, expected in (
        ("contract_id", contract.data["contract_id"]),
        ("contract_version", contract.data["contract_version"]),
        ("contract_sha256", contract.sha256),
        ("regime_count", 4),
    ):
        if master.get(key) != expected:
            raise TrainingContractError(f"training-data master {key} mismatch")

    master_regime = master.get("regimes", {}).get(regime_id)
    if not isinstance(master_regime, dict) or master_regime.get("audit_status") != "PASS":
        raise TrainingContractError(f"training-data master does not pass {regime_id}")
    regime_root = _resolve_contained(root, str(master_regime["path"]))
    manifest = read_json(regime_root / "dataset_manifest.json")
    if manifest.get("regime_id") != regime_id:
        raise TrainingContractError(f"dataset manifest regime mismatch: {regime_id}")
    if manifest.get("contract_sha256") != contract.sha256:
        raise TrainingContractError(f"dataset contract hash mismatch: {regime_id}")
    if manifest.get("contract_version") != contract.data["contract_version"]:
        raise TrainingContractError(f"dataset contract version mismatch: {regime_id}")
    row_schema_hash = contract.data["training"]["training_row_contract"]["schema_sha256"]
    if manifest.get("training_row_schema_sha256") != row_schema_hash:
        raise TrainingContractError(f"training row schema drift: {regime_id}")
    content_identity = sha256_json(_dataset_identity(manifest))
    if content_identity != manifest.get("content_identity_sha256"):
        raise TrainingContractError(f"dataset content identity mismatch: {regime_id}")
    if content_identity != master_regime.get("content_identity_sha256"):
        raise TrainingContractError(f"master content identity mismatch: {regime_id}")

    rows: dict[str, tuple[dict[str, Any], ...]] = {}
    regime_spec = _regime_spec(contract, regime_id)
    for split in ("train", "val"):
        artifact = manifest["splits"][split]
        path = _resolve_contained(root, str(artifact["path"]))
        if sha256_file(path) != artifact["sha256"]:
            raise TrainingContractError(f"dataset hash drift: {regime_id}/{split}")
        split_rows = tuple(iter_jsonl(path))
        expected_count = regime_spec[f"expected_{split}_rows"]
        if len(split_rows) != expected_count or len(split_rows) != artifact["row_count"]:
            raise TrainingContractError(f"dataset row count drift: {regime_id}/{split}")
        for row in split_rows:
            if row.get("regime") != regime_id or row.get("split") != split:
                raise TrainingContractError(f"row binding drift: {row.get('record_id')}")
            if len(row.get("images", [])) != regime_spec["images_per_row"]:
                raise TrainingContractError(f"row modality drift: {row.get('record_id')}")
        rows[split] = split_rows
    return BoundTrainingDataset(
        regime_id=regime_id,
        root=root,
        train_path=_resolve_contained(root, manifest["splits"]["train"]["path"]),
        val_path=_resolve_contained(root, manifest["splits"]["val"]["path"]),
        train_rows=rows["train"],
        val_rows=rows["val"],
        manifest=manifest,
    )


def build_training_comparability(
    contract: LoadedContract,
    adapters_root: str | Path,
    training_profile: str = "common_27b",
) -> dict[str, Any]:
    profiles = contract.data["training"]
    if training_profile not in profiles:
        raise TrainingContractError(f"unknown training profile: {training_profile}")
    common = profiles[training_profile]
    regimes: dict[str, Any] = {}
    for regime_id in TRAINING_REGIME_ORDER:
        bound = load_training_dataset(contract, adapters_root, regime_id)
        spec = _regime_spec(contract, regime_id)
        regimes[regime_id] = {
            "dataset_content_identity_sha256": bound.content_identity,
            "train_rows": len(bound.train_rows),
            "val_rows": len(bound.val_rows),
            "modality": spec["modality"],
            "images_per_row": spec["images_per_row"],
            "assistant_target": spec.get(
                "assistant_fields", spec.get("assistant_target_contract")
            ),
            "ledger_consumers": spec["ledger_consumers"],
        }
    return {
        "schema_version": 1,
        "status": "PASS",
        "contract_id": contract.data["contract_id"],
        "contract_version": contract.data["contract_version"],
        "contract_sha256": contract.sha256,
        "training_profile": training_profile,
        "model_key": common["model_key"],
        "shared_locked_fields": {
            key: value for key, value in common.items() if key != "smoke_gate"
        },
        "allowed_regime_differences": [
            "training corpus and source IDs",
            "assistant supervision target and resulting token exposure",
            "structured_multimodal_ft adds exactly topdown and radar images",
        ],
        "forbidden_regime_differences": [],
        "regimes": regimes,
    }


def resolve_training_config(
    contract: LoadedContract,
    dataset: BoundTrainingDataset,
    run_mode: str,
    training_profile: str = "common_27b",
) -> dict[str, Any]:
    if run_mode not in {"gpu-smoke", "formal"}:
        raise TrainingContractError(f"unsupported training run mode: {run_mode}")
    profiles = contract.data["training"]
    if training_profile not in profiles:
        raise TrainingContractError(f"unknown training profile: {training_profile}")
    common = dict(profiles[training_profile])
    smoke = dict(common.pop("smoke_gate"))
    resolved = {
        **common,
        "training_profile": training_profile,
        "run_mode": run_mode,
        "regime_id": dataset.regime_id,
        "dataset_content_identity_sha256": dataset.content_identity,
        "train_rows": len(dataset.train_rows),
        "val_rows": len(dataset.val_rows),
        "formal_result": False,
    }
    if run_mode == "gpu-smoke":
        resolved.update(
            {
                "effective_train_rows": smoke["row_count_per_split"],
                "effective_val_rows": smoke["row_count_per_split"],
                "gradient_accumulation_steps": smoke["gradient_accumulation_steps"],
                "warmup_ratio": smoke["warmup_ratio"],
                "max_steps": smoke["resumed_max_steps"],
                "first_leg_max_steps": smoke["first_leg_max_steps"],
                "save_steps": smoke["save_steps"],
                "eval_steps": smoke["eval_steps"],
                "logging_steps": smoke["logging_steps"],
                "required_smoke_checks": smoke["required_checks"],
            }
        )
    else:
        resolved.update(
            {
                "effective_train_rows": len(dataset.train_rows),
                "effective_val_rows": len(dataset.val_rows),
                "first_leg_max_steps": None,
                "required_smoke_checks": [],
            }
        )
    identity = {key: value for key, value in resolved.items() if key != "config_sha256"}
    resolved["config_sha256"] = sha256_json(identity)
    return resolved


def build_assistant_labels(
    full_input_ids: Sequence[int],
    prompt_input_ids: Sequence[int],
    *,
    attention_mask: Sequence[int] | None = None,
    max_length: int,
) -> tuple[list[int], dict[str, Any]]:
    full = [int(value) for value in full_input_ids]
    prompt = [int(value) for value in prompt_input_ids]
    if not full:
        raise TrainingContractError("full chat encoding is empty")
    if len(full) > max_length:
        raise TrainingContractError(
            f"encoded row length {len(full)} exceeds max_length={max_length}; truncation is forbidden"
        )
    if not prompt or len(prompt) >= len(full):
        raise TrainingContractError("prompt encoding leaves no assistant supervision")
    if full[: len(prompt)] != prompt:
        mismatch = next(
            (
                index for index, (left, right) in enumerate(zip(full, prompt))
                if left != right
            ),
            min(len(full), len(prompt)),
        )
        raise TrainingContractError(
            f"prompt encoding is not an exact prefix of full chat at token {mismatch}"
        )
    if attention_mask is None:
        mask = [1] * len(full)
    else:
        mask = [int(value) for value in attention_mask]
        if len(mask) != len(full) or any(value not in {0, 1} for value in mask):
            raise TrainingContractError("attention mask must be binary and match input length")

    labels = [IGNORE_INDEX] * len(full)
    for index in range(len(prompt), len(full)):
        if mask[index] == 1:
            labels[index] = full[index]
    supervised = sum(value != IGNORE_INDEX for value in labels)
    if supervised <= 0:
        raise TrainingContractError("assistant-only mask produced zero supervised tokens")
    if any(labels[index] != IGNORE_INDEX for index in range(len(prompt))):
        raise TrainingContractError("prompt token became supervised")
    audit = {
        "total_tokens": len(full),
        "prompt_tokens": len(prompt),
        "assistant_tokens": supervised,
        "padding_tokens": sum(value == 0 for value in mask),
        "truncated": False,
        "zero_supervision": False,
        "mask_strategy": "prompt_full_exact_prefix_v1",
    }
    return labels, audit


def resolve_image_paths(
    row: Mapping[str, Any],
    benchmark_root: str | Path,
) -> tuple[Path, ...]:
    root = Path(benchmark_root).resolve()
    resolved: list[Path] = []
    for image in row.get("images", []):
        relative = str(image.get("path", ""))
        path = _resolve_contained(root, relative)
        if not path.is_file():
            raise TrainingContractError(f"training image is missing: {relative}")
        resolved.append(path)
    return tuple(resolved)


def build_chat_messages(
    row: Mapping[str, Any],
    *,
    image_objects: Sequence[Any] = (),
    include_assistant: bool,
) -> list[dict[str, Any]]:
    try:
        assistant = json.loads(str(row["assistant_json"]))
    except (KeyError, json.JSONDecodeError) as exc:
        raise TrainingContractError(f"invalid assistant_json: {row.get('record_id')}") from exc
    if not isinstance(assistant, dict):
        raise TrainingContractError("assistant_json must encode one object")
    user_content = [
        {"type": "image", "image": image} for image in image_objects
    ]
    merged_text = (
        "System instructions:\n"
        f"{row['system_prompt']}\n\n"
        "User request:\n"
        f"{row['user_prompt']}"
    )
    user_content.append({"type": "text", "text": merged_text})
    messages = [
        {"role": "user", "content": user_content},
    ]
    if include_assistant:
        messages.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": str(row["assistant_json"])}],
            }
        )
    return messages


def summarize_token_audits(audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not audits:
        raise TrainingContractError("token audit cannot be empty")
    total_lengths = sorted(int(item["total_tokens"]) for item in audits)
    prompt_tokens = sum(int(item["prompt_tokens"]) for item in audits)
    assistant_tokens = sum(int(item["assistant_tokens"]) for item in audits)
    p95_index = max(0, math.ceil(0.95 * len(total_lengths)) - 1)
    return {
        "status": "PASS",
        "row_count": len(audits),
        "prompt_token_count": prompt_tokens,
        "assistant_token_count": assistant_tokens,
        "total_token_count": sum(total_lengths),
        "mean_total_tokens": mean(total_lengths),
        "p95_total_tokens": total_lengths[p95_index],
        "max_total_tokens": total_lengths[-1],
        "truncated_row_count": sum(bool(item["truncated"]) for item in audits),
        "zero_supervision_row_count": sum(
            bool(item["zero_supervision"]) for item in audits
        ),
        "mask_strategy": "prompt_full_exact_prefix_v1",
    }


def validate_training_run_artifacts(
    contract: LoadedContract,
    run_dir: str | Path,
    *,
    expected_steps: int,
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    artifacts = contract.data["artifacts"]
    required = set(artifacts["required_files_common_runs"]) | set(
        artifacts["required_files_training_runs"]
    )
    deferred = {"run_manifest.json", "completion_report.json", "status.json"}
    missing = sorted(path for path in required - deferred if not (root / path).is_file())
    checks: dict[str, bool] = {"required_artifacts": not missing}
    errors: list[str] = [f"missing:{path}" for path in missing]

    json_files = (
        "input_manifest.json",
        "environment.json",
        "training_config_resolved.json",
        "training_comparability.json",
        "token_audit_summary.json",
        "adapter/adapter_config.json",
        "training/trainer_state.json",
    )
    for relative in json_files:
        path = root / relative
        if not path.is_file():
            continue
        try:
            value = read_json(path)
            passed = isinstance(value, dict)
        except (OSError, JsonFormatError):
            passed = False
        checks[f"json:{relative}"] = passed
        if not passed:
            errors.append(f"invalid_json:{relative}")

    try:
        token_summary = read_json(root / "token_audit_summary.json")
        token_pass = (
            token_summary.get("status") == "PASS"
            and token_summary.get("truncated_row_count") == 0
            and token_summary.get("zero_supervision_row_count") == 0
            and token_summary.get("assistant_token_count", 0) > 0
        )
    except (OSError, JsonFormatError):
        token_pass = False
    checks["token_audit"] = token_pass
    if not token_pass:
        errors.append("token_audit_failed")

    try:
        state = read_json(root / "training" / "trainer_state.json")
        step_pass = int(state.get("global_step", -1)) == int(expected_steps)
    except (OSError, JsonFormatError, TypeError, ValueError):
        step_pass = False
    checks["actual_steps"] = step_pass
    if not step_pass:
        errors.append("trainer_state_step_mismatch")

    for relative in ("train_log.jsonl", "training_events.jsonl", "token_audit.jsonl"):
        path = root / relative
        try:
            rows = list(iter_jsonl(path))
            passed = len(rows) > 0
        except (OSError, JsonFormatError):
            passed = False
        checks[f"nonempty:{relative}"] = passed
        if not passed:
            errors.append(f"empty_or_invalid:{relative}")

    comparability_pass = False
    try:
        comparability = read_json(root / "training_comparability.json")
        comparability_pass = (
            comparability.get("status") == "PASS"
            and not comparability.get("forbidden_regime_differences")
            and comparability.get("contract_sha256") == contract.sha256
        )
    except (OSError, JsonFormatError):
        pass
    checks["comparability"] = comparability_pass
    if not comparability_pass:
        errors.append("comparability_failed")

    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "formal_result": False,
        "required_artifact_profile": "common+training",
        "expected_steps": int(expected_steps),
        "checks": checks,
        "missing_files": missing,
        "errors": errors,
    }


def validate_training_smoke_suite(
    contract: LoadedContract,
    run_root: str | Path,
    run_ids: Mapping[str, str],
) -> dict[str, Any]:
    root = Path(run_root).resolve()
    common = contract.data["training"]["common_27b"]
    smoke = common["smoke_gate"]
    errors: list[str] = []
    runs: dict[str, Any] = {}
    model_identities: set[str] = set()
    dataset_identities: set[str] = set()
    required_artifact_hashes = (
        "input_manifest.json",
        "environment.json",
        "training_runtime.json",
        "training_config_resolved.json",
        "training_comparability.json",
        "token_audit.jsonl",
        "token_audit_summary.json",
        "train_log.jsonl",
        "training_events.jsonl",
        "training/trainer_state.json",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "training_smoke_report.json",
        "completion_report.json",
    )
    if set(run_ids) != set(TRAINING_REGIME_ORDER):
        errors.append("run map must contain exactly the four training regimes")
    for regime_id in TRAINING_REGIME_ORDER:
        run_id = run_ids.get(regime_id)
        if not isinstance(run_id, str):
            continue
        run_dir = (root / run_id).resolve()
        try:
            run_dir.relative_to(root)
        except ValueError:
            errors.append(f"run path escapes root:{regime_id}")
            continue
        try:
            report = read_json(run_dir / "training_smoke_report.json")
            manifest = read_json(run_dir / "run_manifest.json")
            config = read_json(run_dir / "training_config_resolved.json")
            runtime = read_json(run_dir / "training_runtime.json")
            completion = read_json(run_dir / "completion_report.json")
        except (OSError, JsonFormatError) as exc:
            errors.append(f"missing_or_invalid_run:{regime_id}:{exc}")
            continue
        required_checks = smoke["required_checks"]
        checks_pass = all(report.get("checks", {}).get(name) is True for name in required_checks)
        model_identity = manifest.get("model_content_identity_sha256")
        dataset_identity = manifest.get("dataset_content_identity_sha256")
        adapter_identity = manifest.get("adapter_model_sha256")
        declared_artifact_hashes = manifest.get("artifact_sha256", {})
        artifact_hashes_pass = isinstance(declared_artifact_hashes, dict)
        artifact_hashes = (
            declared_artifact_hashes if artifact_hashes_pass else {}
        )
        if artifact_hashes_pass:
            for relative in required_artifact_hashes:
                artifact_path = run_dir / relative
                if (
                    not artifact_path.is_file()
                    or artifact_hashes.get(relative) != sha256_file(artifact_path)
                ):
                    artifact_hashes_pass = False
                    break
        common_overrides = {
            "gradient_accumulation_steps",
            "warmup_ratio",
            "max_steps",
            "save_steps",
            "eval_steps",
            "logging_steps",
        }
        common_config_pass = all(
            config.get(key) == value
            for key, value in common.items()
            if key != "smoke_gate" and key not in common_overrides
        )
        smoke_config_pass = (
            config.get("gradient_accumulation_steps")
            == smoke["gradient_accumulation_steps"]
            and config.get("warmup_ratio") == smoke["warmup_ratio"]
            and config.get("max_steps") == smoke["resumed_max_steps"]
            and config.get("first_leg_max_steps") == smoke["first_leg_max_steps"]
            and config.get("save_steps") == smoke["save_steps"]
            and config.get("eval_steps") == smoke["eval_steps"]
            and config.get("logging_steps") == smoke["logging_steps"]
        )
        runtime_packages = runtime.get("packages", {})
        runtime_pass = (
            runtime.get("cuda_available") is True
            and bool(runtime.get("devices"))
            and isinstance(runtime_packages, dict)
            and all(
                isinstance(runtime_packages.get(name), str)
                and bool(runtime_packages.get(name))
                for name in (
                    "torch",
                    "transformers",
                    "peft",
                    "accelerate",
                    "Pillow",
                    "safetensors",
                    "sentencepiece",
                )
            )
        )
        conditions = {
            "report_status": report.get("status") == "PASS",
            "report_non_formal": report.get("formal_result") is False,
            "report_mode": report.get("run_mode") == "gpu-smoke",
            "regime": report.get("regime_id") == regime_id,
            "run_id": report.get("run_id") == run_id,
            "steps": report.get("actual_steps") == smoke["resumed_max_steps"],
            "required_checks": checks_pass,
            "manifest_contract": (
                manifest.get("contract_sha256") == contract.sha256
                and manifest.get("contract_version") == contract.data["contract_version"]
            ),
            "manifest_non_formal": manifest.get("formal_result") is False,
            "manifest_identity": (
                manifest.get("run_id") == run_id
                and manifest.get("regime_id") == regime_id
                and manifest.get("run_mode") == "gpu-smoke"
                and manifest.get("completion_status") == "PASS"
            ),
            "manifest_steps": manifest.get("actual_steps")
            == smoke["resumed_max_steps"],
            "manifest_artifact_hashes": artifact_hashes_pass,
            "config_mode": config.get("run_mode") == "gpu-smoke",
            "config_identity": (
                config.get("regime_id") == regime_id
                and config.get("dataset_content_identity_sha256") == dataset_identity
            ),
            "config_mask": config.get("assistant_mask_strategy")
            == common["assistant_mask_strategy"],
            "config_common_locked": common_config_pass,
            "config_smoke_locked": smoke_config_pass,
            "config_rows": (
                config.get("effective_train_rows") == smoke["row_count_per_split"]
                and config.get("effective_val_rows") == smoke["row_count_per_split"]
            ),
            "zero_truncation": config.get("truncated_row_count") == 0,
            "zero_supervision_failures": config.get("zero_supervision_row_count") == 0,
            "model_identity_consistent": (
                isinstance(model_identity, str)
                and bool(model_identity)
                and report.get("model_content_identity_sha256") == model_identity
            ),
            "adapter_identity_consistent": (
                isinstance(adapter_identity, str)
                and bool(adapter_identity)
                and report.get("adapter_model_sha256") == adapter_identity
                and artifact_hashes.get("adapter/adapter_model.safetensors")
                == adapter_identity
            ),
            "dataset_identity_present": (
                isinstance(dataset_identity, str) and bool(dataset_identity)
            ),
            "runtime": runtime_pass,
            "completion": (
                completion.get("status") == "PASS"
                and completion.get("formal_result") is False
            ),
            "nonzero_training_evidence": (
                isinstance(report.get("peak_cuda_memory_bytes"), int)
                and report.get("peak_cuda_memory_bytes", 0) > 0
                and isinstance(report.get("trainable_parameters"), int)
                and report.get("trainable_parameters", 0) > 0
                and isinstance(report.get("total_parameters"), int)
                and report.get("total_parameters", 0)
                > report.get("trainable_parameters", 0)
            ),
        }
        failed = sorted(key for key, passed in conditions.items() if not passed)
        if failed:
            errors.append(f"run_failed:{regime_id}:{failed}")
        identity = str(model_identity or "")
        if identity:
            model_identities.add(identity)
        dataset_identity_text = str(dataset_identity or "")
        if dataset_identity_text:
            dataset_identities.add(dataset_identity_text)
        runs[regime_id] = {
            "run_id": run_id,
            "status": "PASS" if not failed else "FAIL",
            "conditions": conditions,
            "dataset_content_identity_sha256": dataset_identity_text,
            "model_content_identity_sha256": identity,
            "adapter_model_sha256": adapter_identity,
        }
    if len(model_identities) != 1:
        errors.append("smoke runs do not bind one identical base model identity")
    if len(dataset_identities) != 4:
        errors.append("smoke runs do not bind four distinct regime datasets")
    return {
        "schema_version": 1,
        "status": "PASS" if not errors and len(runs) == 4 else "FAIL",
        "formal_result": False,
        "contract_version": contract.data["contract_version"],
        "contract_sha256": contract.sha256,
        "required_regime_count": 4,
        "validated_regime_count": len(runs),
        "model_content_identity_sha256": (
            next(iter(model_identities)) if len(model_identities) == 1 else None
        ),
        "runs": runs,
        "errors": errors,
    }
