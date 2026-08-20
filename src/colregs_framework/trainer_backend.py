"""Optional GPU backend for the shared Gemma-3 LoRA trainer."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import sha256_file, sha256_json
from .io import append_jsonl, read_json, write_json_atomic
from .runs import RunDirectory, utc_now
from .training import (
    BoundTrainingDataset,
    TrainingContractError,
    build_assistant_labels,
    build_chat_messages,
    resolve_image_paths,
    summarize_token_audits,
    validate_training_run_artifacts,
)


def _training_stack() -> dict[str, Any]:
    try:
        import torch
        from PIL import Image
        from peft import LoraConfig, PeftModel, get_peft_model
        from safetensors import safe_open
        from transformers import (
            AutoProcessor,
            Gemma3ForConditionalGeneration,
            Trainer,
            TrainerCallback,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise TrainingContractError(
            "GPU training dependencies are missing; install the project with [training] extras"
        ) from exc
    return {
        "torch": torch,
        "Image": Image,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "safe_open": safe_open,
        "AutoProcessor": AutoProcessor,
        "Gemma3ForConditionalGeneration": Gemma3ForConditionalGeneration,
        "Trainer": Trainer,
        "TrainerCallback": TrainerCallback,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def capture_training_runtime(torch_module: Any) -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "Pillow",
        "safetensors",
        "sentencepiece",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda_available = bool(torch_module.cuda.is_available())
    devices = []
    if cuda_available:
        for index in range(torch_module.cuda.device_count()):
            properties = torch_module.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": int(properties.total_memory),
                    "capability": list(torch_module.cuda.get_device_capability(index)),
                }
            )
    return {
        "captured_at_utc": utc_now(),
        "packages": packages,
        "cuda_available": cuda_available,
        "cuda_runtime": torch_module.version.cuda,
        "cudnn_version": torch_module.backends.cudnn.version(),
        "devices": devices,
    }


class TrainingRowDataset:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = tuple(dict(row) for row in rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class SingleItemAssistantCollator:
    """Encode one row and supervise only the exact assistant suffix."""

    def __init__(
        self,
        processor: Any,
        torch_module: Any,
        image_class: Any,
        benchmark_root: str | Path,
        max_length: int,
        expected_images: int,
    ) -> None:
        self.processor = processor
        self.torch = torch_module
        self.Image = image_class
        self.benchmark_root = Path(benchmark_root).resolve()
        self.max_length = max_length
        self.expected_images = expected_images
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise TrainingContractError("AutoProcessor must expose a tokenizer")
        tokenizer.padding_side = "right"

    def _image_objects(self, row: Mapping[str, Any]) -> list[Any]:
        paths = resolve_image_paths(row, self.benchmark_root)
        if len(paths) != self.expected_images:
            raise TrainingContractError(
                f"image count drift for {row.get('record_id')}: {len(paths)}"
            )
        images = []
        for path in paths:
            with self.Image.open(path) as source:
                images.append(source.convert("RGB"))
        return images

    def encode(self, row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        full_images = self._image_objects(row)
        prompt_images = [image.copy() for image in full_images]
        full_messages = build_chat_messages(
            row, image_objects=full_images, include_assistant=True
        )
        prompt_messages = build_chat_messages(
            row, image_objects=prompt_images, include_assistant=False
        )
        full_inputs = self.processor.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        prompt_inputs = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_ids = full_inputs["input_ids"][0].tolist()
        prompt_ids = prompt_inputs["input_ids"][0].tolist()
        attention = full_inputs.get("attention_mask")
        attention_values = attention[0].tolist() if attention is not None else None
        labels, audit = build_assistant_labels(
            full_ids,
            prompt_ids,
            attention_mask=attention_values,
            max_length=self.max_length,
        )
        full_inputs["labels"] = self.torch.tensor(
            [labels], dtype=full_inputs["input_ids"].dtype
        )
        audit = {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "split": row["split"],
            "image_count": len(full_images),
            **audit,
        }
        return dict(full_inputs), audit

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if len(features) != 1:
            raise TrainingContractError(
                "contract fixes per_device_train_batch_size=1; collator received another size"
            )
        batch, _ = self.encode(features[0])
        return batch


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite token audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_token_rows(
    collator: SingleItemAssistantCollator,
    rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    output_jsonl: Path,
) -> dict[str, Any]:
    audits = []
    for split in ("train", "val"):
        for row in rows_by_split[split]:
            _, audit = collator.encode(row)
            audits.append(audit)
    _write_jsonl_atomic(output_jsonl, audits)
    summary = summarize_token_audits(audits)
    summary["counts_by_split"] = {
        split: sum(item["split"] == split for item in audits)
        for split in ("train", "val")
    }
    summary["token_audit_sha256"] = sha256_file(output_jsonl)
    return summary


def _log_callback(stack: Mapping[str, Any], run: RunDirectory) -> Any:
    base = stack["TrainerCallback"]

    class ArtifactLogCallback(base):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                append_jsonl(
                    run.artifact("train_log.jsonl"),
                    {
                        "global_step": int(state.global_step),
                        "epoch": state.epoch,
                        "recorded_at_utc": utc_now(),
                        **dict(logs or {}),
                    },
                )

        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                append_jsonl(
                    run.artifact("training_events.jsonl"),
                    {
                        "event": "CHECKPOINT_SAVED",
                        "global_step": int(state.global_step),
                        "recorded_at_utc": utc_now(),
                    },
                )

    return ArtifactLogCallback()


def _dtype(torch_module: Any, name: str) -> Any:
    if name == "bfloat16":
        return torch_module.bfloat16
    if name == "float16":
        return torch_module.float16
    if name == "float32":
        return torch_module.float32
    raise TrainingContractError(f"unsupported dtype: {name}")


def _base_model(stack: Mapping[str, Any], model_path: Path, config: Mapping[str, Any]) -> Any:
    model = stack["Gemma3ForConditionalGeneration"].from_pretrained(
        str(model_path),
        torch_dtype=_dtype(stack["torch"], str(config["dtype"])),
        attn_implementation=str(config["attention_implementation"]),
        device_map=None,
        local_files_only=True,
    )
    model.config.use_cache = False
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    return model


def _lora_model(stack: Mapping[str, Any], base_model: Any, config: Mapping[str, Any]) -> Any:
    if config.get("vision_tower_policy") != "frozen_no_lora":
        raise TrainingContractError("unsupported vision tower training policy")
    lora = stack["LoraConfig"](
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config["lora_target_modules"]),
        exclude_modules=str(config["lora_exclude_modules_regex"]),
    )
    return stack["get_peft_model"](base_model, lora)


def _trainable_lora_scope(model: Any) -> dict[str, Any]:
    names = sorted(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if not names:
        raise TrainingContractError("LoRA model exposes no trainable parameters")
    forbidden = [name for name in names if "vision_tower" in name]
    language = [name for name in names if "language_model" in name]
    non_lora = [name for name in names if "lora_" not in name]
    scope = {
        "policy": "frozen_no_lora",
        "trainable_tensor_count": len(names),
        "language_lora_tensor_count": len(language),
        "vision_trainable_tensor_count": len(forbidden),
        "non_lora_trainable_tensor_count": len(non_lora),
        "status": (
            "PASS"
            if language and not forbidden and not non_lora
            else "FAIL"
        ),
    }
    if scope["status"] != "PASS":
        raise TrainingContractError(f"trainable LoRA scope drift: {scope}")
    return scope


def _saved_adapter_lora_scope(stack: Mapping[str, Any], path: Path) -> dict[str, Any]:
    with stack["safe_open"](str(path), framework="pt", device="cpu") as handle:
        names = list(handle.keys())
    forbidden = [name for name in names if "vision_tower" in name]
    language = [name for name in names if "language_model" in name]
    scope = {
        "adapter_tensor_count": len(names),
        "language_adapter_tensor_count": len(language),
        "vision_adapter_tensor_count": len(forbidden),
        "status": "PASS" if language and not forbidden else "FAIL",
    }
    if scope["status"] != "PASS":
        raise TrainingContractError(f"saved adapter LoRA scope drift: {scope}")
    return scope


def _training_arguments(
    stack: Mapping[str, Any],
    output_dir: Path,
    config: Mapping[str, Any],
    *,
    max_steps: int,
) -> Any:
    values = {
        "output_dir": str(output_dir),
        "seed": int(config["seed"]),
        "data_seed": int(config["seed"]),
        "max_steps": int(max_steps),
        "per_device_train_batch_size": int(config["per_device_train_batch_size"]),
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "warmup_ratio": float(config["warmup_ratio"]),
        "lr_scheduler_type": str(config["lr_scheduler_type"]),
        "optim": str(config["optimizer"]),
        "max_grad_norm": float(config["max_grad_norm"]),
        "logging_steps": int(config["logging_steps"]),
        "save_steps": int(config["save_steps"]),
        "eval_steps": int(config["eval_steps"]),
        "save_strategy": "steps",
        "save_total_limit": int(config["save_total_limit"]),
        "bf16": config["dtype"] == "bfloat16",
        "fp16": config["dtype"] == "float16",
        "gradient_checkpointing": bool(config["gradient_checkpointing"]),
        "remove_unused_columns": bool(config["remove_unused_columns"]),
        "report_to": [] if config["report_to"] == "none" else [config["report_to"]],
        "dataloader_num_workers": int(config["dataloader_num_workers"]),
        "dataloader_pin_memory": False,
        "load_best_model_at_end": False,
        "save_safetensors": True,
    }
    signature = inspect.signature(stack["TrainingArguments"].__init__)
    if "eval_strategy" in signature.parameters:
        values["eval_strategy"] = "steps"
    elif "evaluation_strategy" in signature.parameters:
        values["evaluation_strategy"] = "steps"
    else:
        raise TrainingContractError("TrainingArguments lacks an evaluation strategy option")
    return stack["TrainingArguments"](**values)


def _trainer(
    stack: Mapping[str, Any],
    model: Any,
    processor: Any,
    collator: Any,
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    arguments: Any,
    callback: Any,
) -> Any:
    kwargs = {
        "model": model,
        "args": arguments,
        "train_dataset": TrainingRowDataset(train_rows),
        "eval_dataset": TrainingRowDataset(val_rows),
        "data_collator": collator,
        "callbacks": [callback],
    }
    signature = inspect.signature(stack["Trainer"].__init__)
    if "processing_class" in signature.parameters:
        kwargs["processing_class"] = processor
    elif "tokenizer" in signature.parameters:
        kwargs["tokenizer"] = getattr(processor, "tokenizer", None)
    return stack["Trainer"](**kwargs)


def _empty_model_cache(stack: Mapping[str, Any]) -> None:
    gc.collect()
    if stack["torch"].cuda.is_available():
        stack["torch"].cuda.empty_cache()


def _trainable_parameter_sha256(torch_module: Any, model: Any) -> str:
    """Hash trainable tensor content without retaining full CPU model copies."""
    digest = hashlib.sha256()
    parameter_count = 0
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().contiguous().to(device="cpu")
        raw = tensor.view(torch_module.uint8).numpy().tobytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(raw)
        parameter_count += int(parameter.numel())
        del tensor
    if parameter_count == 0:
        raise TrainingContractError("model has no trainable parameters to audit")
    return digest.hexdigest()


def _checkpoint_state(checkpoint: Path) -> dict[str, bool]:
    return {
        name: (checkpoint / name).is_file()
        for name in (
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        )
    }


def _reload_forward(
    stack: Mapping[str, Any],
    model_path: Path,
    adapter_path: Path,
    config: Mapping[str, Any],
    collator: SingleItemAssistantCollator,
    row: Mapping[str, Any],
) -> float:
    base = _base_model(stack, model_path, config)
    model = stack["PeftModel"].from_pretrained(
        base, str(adapter_path), is_trainable=False, local_files_only=True
    )
    if stack["torch"].cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = stack["torch"].device("cuda", local_rank)
        model.to(device)
    else:
        device = stack["torch"].device("cpu")
    model.eval()
    batch = collator([row])
    batch = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }
    with stack["torch"].no_grad():
        loss = float(model(**batch).loss.detach().float().cpu().item())
    if not math.isfinite(loss):
        raise TrainingContractError("reloaded adapter forward produced non-finite loss")
    del model
    del base
    _empty_model_cache(stack)
    return loss


def run_training_backend(
    run: RunDirectory,
    dataset: BoundTrainingDataset,
    config: dict[str, Any],
    *,
    model_path: str | Path,
    benchmark_root: str | Path,
    model_manifest: Mapping[str, Any],
    comparability: Mapping[str, Any],
    resume_from_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    stack = _training_stack()
    torch = stack["torch"]
    if not torch.cuda.is_available():
        raise TrainingContractError("GPU training requires CUDA; CPU execution is forbidden")
    model_root = Path(model_path).resolve()
    is_external_resume = resume_from_checkpoint is not None
    run.record_status("TRAINING_PREFLIGHT_STARTED", regime_id=dataset.regime_id)
    runtime = capture_training_runtime(torch)
    if is_external_resume:
        append_jsonl(
            run.artifact("training_resume_environments.jsonl"),
            {
                "event": "RESUME_RUNTIME",
                "checkpoint_name": Path(resume_from_checkpoint).name,
                **runtime,
            },
        )
        existing_comparability = read_json(run.artifact("training_comparability.json"))
        if sha256_json(existing_comparability) != sha256_json(dict(comparability)):
            raise TrainingContractError("training comparability drift on resume")
    else:
        write_json_atomic(run.artifact("training_runtime.json"), runtime)
        write_json_atomic(run.artifact("training_comparability.json"), dict(comparability))

    processor = stack["AutoProcessor"].from_pretrained(
        str(model_root), use_fast=False, local_files_only=True
    )
    expected_images = next(
        item["images_per_row"]
        for item in run.contract.data["training"]["regimes"]
        if item["regime_id"] == dataset.regime_id
    )
    collator = SingleItemAssistantCollator(
        processor,
        torch,
        stack["Image"],
        benchmark_root,
        int(config["max_length"]),
        int(expected_images),
    )
    train_rows = dataset.train_rows[: int(config["effective_train_rows"])]
    val_rows = dataset.val_rows[: int(config["effective_val_rows"])]
    if is_external_resume:
        existing_config = read_json(run.artifact("training_config_resolved.json"))
        drift = sorted(
            key for key, value in config.items()
            if key != "config_sha256" and existing_config.get(key) != value
        )
        if drift:
            raise TrainingContractError(f"resolved training config drift on resume: {drift}")
        config = existing_config
        token_summary = read_json(run.artifact("token_audit_summary.json"))
        if token_summary.get("token_audit_sha256") != sha256_file(
            run.artifact("token_audit.jsonl")
        ):
            raise TrainingContractError("token audit drift on resume")
        run.record_status(
            "TOKEN_AUDIT_REUSED",
            token_audit_sha256=token_summary["token_audit_sha256"],
        )
    else:
        rows_by_split = {"train": train_rows, "val": val_rows}
        token_summary = audit_token_rows(
            collator, rows_by_split, run.artifact("token_audit.jsonl")
        )
        write_json_atomic(run.artifact("token_audit_summary.json"), token_summary)
        config = {
            **config,
            "token_audit_summary_sha256": sha256_json(token_summary),
            "prompt_token_count": token_summary["prompt_token_count"],
            "assistant_token_count": token_summary["assistant_token_count"],
            "total_token_count": token_summary["total_token_count"],
            "truncated_row_count": token_summary["truncated_row_count"],
            "zero_supervision_row_count": token_summary["zero_supervision_row_count"],
            "model_class": "Gemma3ForConditionalGeneration",
            "processor_class": "AutoProcessor",
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_batch_size": (
                int(config["per_device_train_batch_size"])
                * int(config["gradient_accumulation_steps"])
                * int(os.environ.get("WORLD_SIZE", "1"))
            ),
            "resume_from_checkpoint": None,
        }
        config["config_sha256"] = sha256_json(
            {key: value for key, value in config.items() if key != "config_sha256"}
        )
        write_json_atomic(run.artifact("training_config_resolved.json"), config)
        run.record_status("TOKEN_AUDIT_PASS", **token_summary)

    stack["set_seed"](int(config["seed"]))
    callback = _log_callback(stack, run)
    training_dir = run.artifact("training")
    if is_external_resume:
        if not training_dir.is_dir():
            raise TrainingContractError("resume requires an existing training directory")
        checkpoint = Path(resume_from_checkpoint).resolve()
        try:
            checkpoint.relative_to(training_dir.resolve())
        except ValueError as exc:
            raise TrainingContractError("resume checkpoint must be inside this run") from exc
        if not checkpoint.is_dir() or not all(_checkpoint_state(checkpoint).values()):
            raise TrainingContractError("resume checkpoint is missing full trainer state")
        if run.artifact("adapter").exists():
            raise TrainingContractError("resume refuses a run that already has a final adapter")
    else:
        training_dir.mkdir(exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    resume_verified = False
    first_checkpoint_state: dict[str, bool] = {}
    initial_trainable_sha256 = None
    first_leg_trainable_sha256 = None
    resumed_trainable_sha256 = None
    trainable_lora_scope = None

    if config["run_mode"] == "gpu-smoke":
        first_model = _lora_model(stack, _base_model(stack, model_root, config), config)
        trainable_lora_scope = _trainable_lora_scope(first_model)
        initial_trainable_sha256 = _trainable_parameter_sha256(torch, first_model)
        first_args = _training_arguments(
            stack,
            training_dir,
            config,
            max_steps=int(config["first_leg_max_steps"]),
        )
        first_trainer = _trainer(
            stack,
            first_model,
            processor,
            collator,
            train_rows,
            val_rows,
            first_args,
            callback,
        )
        run.record_status("SMOKE_FIRST_LEG_STARTED")
        first_result = first_trainer.train()
        first_leg_trainable_sha256 = _trainable_parameter_sha256(torch, first_model)
        if int(first_trainer.state.global_step) != int(config["first_leg_max_steps"]):
            raise TrainingContractError("smoke first leg did not reach locked step")
        checkpoint = training_dir / f"checkpoint-{config['first_leg_max_steps']}"
        first_checkpoint_state = _checkpoint_state(checkpoint)
        if not all(first_checkpoint_state.values()):
            raise TrainingContractError(
                f"smoke checkpoint is incomplete: {first_checkpoint_state}"
            )
        run.record_status("SMOKE_FIRST_LEG_COMPLETE", checkpoint=str(checkpoint.name))
        del first_trainer
        del first_model
        _empty_model_cache(stack)

        resumed_model = _lora_model(stack, _base_model(stack, model_root, config), config)
        resumed_args = _training_arguments(
            stack, training_dir, config, max_steps=int(config["max_steps"])
        )
        resumed_trainer = _trainer(
            stack,
            resumed_model,
            processor,
            collator,
            train_rows,
            val_rows,
            resumed_args,
            callback,
        )
        run.record_status("SMOKE_RESUME_STARTED", checkpoint=str(checkpoint.name))
        train_result = resumed_trainer.train(resume_from_checkpoint=str(checkpoint))
        actual_steps = int(resumed_trainer.state.global_step)
        resume_verified = actual_steps == int(config["max_steps"])
        if not resume_verified:
            raise TrainingContractError("smoke resume did not reach locked second step")
        resumed_trainable_sha256 = _trainable_parameter_sha256(torch, resumed_model)
        trainer = resumed_trainer
        model = resumed_model
        del resumed_trainer
        del resumed_model
    else:
        model = _lora_model(stack, _base_model(stack, model_root, config), config)
        trainable_lora_scope = _trainable_lora_scope(model)
        arguments = _training_arguments(
            stack, training_dir, config, max_steps=int(config["max_steps"])
        )
        trainer = _trainer(
            stack,
            model,
            processor,
            collator,
            train_rows,
            val_rows,
            arguments,
            callback,
        )
        train_result = trainer.train(
            resume_from_checkpoint=(
                str(Path(resume_from_checkpoint).resolve())
                if resume_from_checkpoint is not None
                else None
            )
        )
        actual_steps = int(trainer.state.global_step)
        if actual_steps != int(config["max_steps"]):
            raise TrainingContractError("formal trainer did not reach locked max_steps")
        resume_verified = resume_from_checkpoint is None or actual_steps == int(config["max_steps"])

    trainer.save_state()
    adapter_dir = run.artifact("adapter")
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise TrainingContractError("LoRA save did not produce required adapter artifacts")
    saved_adapter_lora_scope = _saved_adapter_lora_scope(stack, adapter_weights)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    del trainer
    del model
    _empty_model_cache(stack)

    reloaded_loss = _reload_forward(
        stack, model_root, adapter_dir, config, collator, train_rows[0]
    )
    elapsed = time.monotonic() - started
    peak_memory = int(torch.cuda.max_memory_allocated())
    checks = {
        "assistant_mask": token_summary["zero_supervision_row_count"] == 0,
        "forward_backward": actual_steps > 0,
        "parameter_update": (
            initial_trainable_sha256 is not None
            and first_leg_trainable_sha256 is not None
            and initial_trainable_sha256 != first_leg_trainable_sha256
            if config["run_mode"] == "gpu-smoke"
            else True
        ),
        "lora_scope": (
            trainable_lora_scope is not None
            and trainable_lora_scope.get("status") == "PASS"
            and saved_adapter_lora_scope.get("status") == "PASS"
        ),
        "checkpoint_state": (
            all(first_checkpoint_state.values())
            if config["run_mode"] == "gpu-smoke"
            else True
        ),
        "resume_to_step_2": (
            resume_verified if config["run_mode"] == "gpu-smoke" else True
        ),
        "adapter_save": adapter_weights.is_file() and adapter_config.is_file(),
        "adapter_reload": True,
        "reloaded_forward": math.isfinite(reloaded_loss),
    }
    required = config.get("required_smoke_checks", [])
    if any(not checks.get(name, False) for name in required):
        raise TrainingContractError(f"training smoke checks failed: {checks}")
    report = {
        "schema_version": 1,
        "status": "PASS",
        "run_id": run.run_id,
        "run_mode": config["run_mode"],
        "regime_id": dataset.regime_id,
        "training_profile": config["training_profile"],
        "formal_result": False,
        "actual_steps": actual_steps,
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "reloaded_forward_loss": reloaded_loss,
        "checks": checks,
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "initial_trainable_sha256": initial_trainable_sha256,
        "first_leg_trainable_sha256": first_leg_trainable_sha256,
        "resumed_trainable_sha256": resumed_trainable_sha256,
        "trainable_lora_scope": trainable_lora_scope,
        "saved_adapter_lora_scope": saved_adapter_lora_scope,
        "adapter_model_sha256": sha256_file(adapter_weights),
        "model_content_identity_sha256": model_manifest["content_identity_sha256"],
        "resumed_from_checkpoint": (
            Path(resume_from_checkpoint).name
            if resume_from_checkpoint is not None
            else None
        ),
    }
    if not math.isfinite(report["train_loss"]):
        raise TrainingContractError("training produced non-finite loss")
    write_json_atomic(run.artifact("training_smoke_report.json"), report)
    completion = validate_training_run_artifacts(
        run.contract, run.path, expected_steps=actual_steps
    )
    completion["training_checks"] = checks
    if completion["status"] != "PASS":
        raise TrainingContractError(
            f"training completion validation failed: {completion['errors']}"
        )
    completion["formal_result"] = config["run_mode"] == "formal"
    write_json_atomic(run.artifact("completion_report.json"), completion)
    material_artifacts = (
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
    artifact_sha256 = {
        relative: sha256_file(run.artifact(relative))
        for relative in material_artifacts
    }
    resume_runtime = run.artifact("training_resume_environments.jsonl")
    if resume_runtime.is_file():
        artifact_sha256["training_resume_environments.jsonl"] = sha256_file(
            resume_runtime
        )
    manifest = {
        "contract_id": run.contract.data["contract_id"],
        "contract_version": run.contract.data["contract_version"],
        "contract_sha256": run.contract.sha256,
        "run_id": run.run_id,
        "run_mode": config["run_mode"],
        "regime_id": dataset.regime_id,
        "training_profile": config["training_profile"],
        "formal_result": config["run_mode"] == "formal",
        "dataset_content_identity_sha256": dataset.content_identity,
        "training_config_sha256": sha256_file(run.artifact("training_config_resolved.json")),
        "training_comparability_sha256": sha256_file(
            run.artifact("training_comparability.json")
        ),
        "token_audit_sha256": sha256_file(run.artifact("token_audit.jsonl")),
        "model_id": model_manifest["model_id"],
        "model_revision": model_manifest["exact_revision"],
        "model_content_identity_sha256": model_manifest["content_identity_sha256"],
        "adapter_model_sha256": report["adapter_model_sha256"],
        "actual_steps": actual_steps,
        "resumed_from_checkpoint": report["resumed_from_checkpoint"],
        "completion_status": "PASS",
        "artifact_sha256": artifact_sha256,
    }
    write_json_atomic(run.artifact("run_manifest.json"), manifest)
    write_json_atomic(
        run.artifact("status.json"),
        {
            "run_id": run.run_id,
            "state": "FORMAL_COMPLETE" if config["run_mode"] == "formal" else "SMOKE_COMPLETE",
            "formal_result": config["run_mode"] == "formal",
            "recorded_at_utc": utc_now(),
        },
    )
    run.record_status("TRAINING_COMPLETE", actual_steps=actual_steps, checks=checks)
    return report
