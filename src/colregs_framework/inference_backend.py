"""Delayed-import Gemma and OpenAI backends for the common inference runner."""

from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .protocols import InferenceInput, build_inference_messages


class InferenceBackendError(RuntimeError):
    """Raised when a backend cannot materialize one traceable response."""

    def __init__(self, message: str, *, attempts: tuple[dict[str, Any], ...] = ()) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(frozen=True)
class BackendResponse:
    output_text: str
    raw_response: dict[str, Any]
    returned_model_id: str
    request_attempts: tuple[dict[str, Any], ...] = ()


class GemmaBackend:
    def __init__(
        self,
        model_path: str | Path,
        *,
        model_id: str,
        model_revision: str,
        decode_profile: Mapping[str, Any],
        adapter_path: str | Path | None = None,
        attention_implementation: str = "eager",
    ) -> None:
        try:
            import torch
            from PIL import Image
            from peft import PeftModel
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError as exc:
            raise InferenceBackendError(
                "Gemma inference dependencies are missing; install the training extra"
            ) from exc
        if not torch.cuda.is_available():
            raise InferenceBackendError("Gemma formal inference requires CUDA")
        self.torch = torch
        self.Image = Image
        self.processor = AutoProcessor.from_pretrained(
            str(Path(model_path).resolve()), local_files_only=True
        )
        base = Gemma3ForConditionalGeneration.from_pretrained(
            str(Path(model_path).resolve()),
            torch_dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
            device_map=None,
            local_files_only=True,
        )
        if adapter_path is not None:
            base = PeftModel.from_pretrained(
                base, str(Path(adapter_path).resolve()), is_trainable=False
            )
        self.model = base.to("cuda")
        self.model.eval()
        self.model.config.use_cache = True
        self.model_id = model_id
        self.model_revision = model_revision
        self.decode = dict(decode_profile)

    def _images(self, sample: InferenceInput) -> list[Any]:
        images = []
        for path in sample.image_paths:
            with self.Image.open(path) as source:
                images.append(source.convert("RGB"))
        return images

    def __call__(self, sample: InferenceInput) -> BackendResponse:
        images = self._images(sample)
        messages = build_inference_messages(sample, images)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_length = int(inputs["input_ids"].shape[-1])
        inputs = inputs.to(self.model.device)
        generation = {
            "do_sample": bool(self.decode["do_sample"]),
            "num_beams": int(self.decode["num_beams"]),
            "max_new_tokens": int(self.decode["max_new_tokens"]),
        }
        started = time.monotonic()
        with self.torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation)
        generated = output_ids[0, input_length:]
        output_text = self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        )
        elapsed = time.monotonic() - started
        return BackendResponse(
            output_text=output_text,
            returned_model_id=f"{self.model_id}@{self.model_revision}",
            raw_response={
                "backend": "local_gemma",
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "output_text": output_text,
                "input_token_count": input_length,
                "generated_token_count": int(generated.shape[-1]),
                "elapsed_seconds": elapsed,
                "packages": {
                    package: importlib.metadata.version(package)
                    for package in ("torch", "transformers", "peft")
                },
            },
        )


class OpenAIBackend:
    def __init__(
        self,
        model: str,
        *,
        decode_profile: Mapping[str, Any],
        response_schema: Mapping[str, Any],
        api_key: str | None = None,
        base_url: str | None = None,
        provider_routing: Mapping[str, Any] | None = None,
        default_headers: Mapping[str, str] | None = None,
    ) -> None:
        try:
            from openai import (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                OpenAI,
                RateLimitError,
            )
        except ImportError as exc:
            raise InferenceBackendError("openai is required for the GPT backend") from exc
        timeout = float(decode_profile.get("request_timeout_seconds", 180))
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=dict(default_headers or {}),
            max_retries=0,
            timeout=timeout,
        )
        self.model = model
        self.decode = dict(decode_profile)
        self.schema = dict(response_schema)
        self.base_url = base_url
        self.provider_routing = dict(provider_routing or {})
        self.transient_errors = (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        )

    @staticmethod
    def _data_url(path: Path) -> str:
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    @staticmethod
    def _sha256_json(value: Mapping[str, Any]) -> str:
        payload = json.dumps(
            dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _safe_error(exc: Exception) -> str:
        # Provider exception strings can contain request material. Persist only class/status/code.
        return json.dumps(
            {
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "error_code": getattr(exc, "code", None),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    def __call__(self, sample: InferenceInput) -> BackendResponse:
        image_detail = str(self.decode.get("image_detail", "high"))
        user_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": sample.user_prompt}
        ]
        user_content.extend(
            {
                "type": "input_image",
                "image_url": self._data_url(path),
                "detail": image_detail,
            }
            for path in sample.image_paths
        )
        image_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in sample.image_paths]
        request_identity = {
            "sample_id": sample.sample_id,
            "requested_model_id": self.model,
            "provider_base_url": self.base_url,
            "provider_routing": self.provider_routing,
            "system_prompt_sha256": hashlib.sha256(
                sample.system_prompt.encode("utf-8")
            ).hexdigest(),
            "user_prompt_sha256": hashlib.sha256(
                sample.user_prompt.encode("utf-8")
            ).hexdigest(),
            "response_schema_sha256": self._sha256_json(self.schema),
            "image_sha256": image_hashes,
            "decode_profile": self.decode,
        }
        request_identity_sha256 = self._sha256_json(request_identity)
        started = time.monotonic()
        attempt_rows: list[dict[str, Any]] = []
        attempts = int(self.decode["max_retries_per_sample"]) + 1
        backoffs = [float(value) for value in self.decode.get("retry_backoff_seconds", [1, 2, 4])]
        response = None
        for attempt in range(attempts):
            attempt_started = time.monotonic()
            try:
                request: dict[str, Any] = {
                    "model": self.model,
                    "reasoning": {"effort": self.decode["reasoning_effort"]},
                    "max_output_tokens": int(self.decode["max_output_tokens"]),
                    "input": [
                        {
                            "role": "system",
                            "content": [{"type": "input_text", "text": sample.system_prompt}],
                        },
                        {"role": "user", "content": user_content},
                    ],
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "colregs_fullfields_v2",
                            "schema": self.schema,
                            "strict": True,
                        }
                    },
                    "store": bool(self.decode.get("store", False)),
                }
                if self.decode.get("service_tier") is not None:
                    request["service_tier"] = self.decode["service_tier"]
                if self.decode.get("temperature") is not None:
                    request["temperature"] = float(self.decode["temperature"])
                if self.provider_routing:
                    request["extra_body"] = {"provider": self.provider_routing}
                response = self.client.responses.create(**request)
                attempt_rows.append(
                    {
                        "attempt_number": attempt + 1,
                        "status": "success",
                        "transient": False,
                        "request_identity_sha256": request_identity_sha256,
                        "elapsed_seconds": time.monotonic() - attempt_started,
                    }
                )
                break
            except self.transient_errors as exc:
                attempt_rows.append(
                    {
                        "attempt_number": attempt + 1,
                        "status": "failed",
                        "transient": True,
                        "request_identity_sha256": request_identity_sha256,
                        "elapsed_seconds": time.monotonic() - attempt_started,
                        "error": self._safe_error(exc),
                    }
                )
                if attempt + 1 >= attempts:
                    raise InferenceBackendError(
                        "OpenAI transient retries exhausted",
                        attempts=tuple(attempt_rows),
                    ) from exc
                time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
            except Exception as exc:
                attempt_rows.append(
                    {
                        "attempt_number": attempt + 1,
                        "status": "failed",
                        "transient": False,
                        "request_identity_sha256": request_identity_sha256,
                        "elapsed_seconds": time.monotonic() - attempt_started,
                        "error": self._safe_error(exc),
                    }
                )
                raise InferenceBackendError(
                    f"OpenAI non-transient request failure: {type(exc).__name__}",
                    attempts=tuple(attempt_rows),
                ) from exc
        if response is None:
            raise InferenceBackendError(
                "OpenAI request ended without a response", attempts=tuple(attempt_rows)
            )
        output_text = str(getattr(response, "output_text", "") or "")
        try:
            raw = response.model_dump(mode="json")
        except Exception:
            raw = {"output_text": output_text}
        returned_model = str(getattr(response, "model", None) or raw.get("model") or self.model)
        raw["elapsed_seconds"] = time.monotonic() - started
        raw["request_identity_sha256"] = request_identity_sha256
        raw["image_sha256"] = image_hashes
        raw["openai_sdk_version"] = importlib.metadata.version("openai")
        raw["attempt_count"] = len(attempt_rows)
        raw["provider_base_url"] = self.base_url
        raw["provider_routing"] = self.provider_routing
        return BackendResponse(output_text, raw, returned_model, tuple(attempt_rows))
