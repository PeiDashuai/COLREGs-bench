"""Append-only invalid-input inference for the contradiction condition."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .completion import CompletionValidator
from .hashing import sha256_file, sha256_json, sha256_ordered_ids, sha256_text
from .inference_backend import BackendResponse
from .io import JsonlJournal, iter_jsonl, read_json, write_json_atomic, write_jsonl_atomic
from .protocols import EvaluationSample, ProtocolAssets
from .robustness import classify_contradiction_output
from .runs import RunDirectory, utc_now


class RobustnessInferenceError(RuntimeError):
    """Raised when contradiction evidence cannot be safely finalized."""


def _journal(path: Path) -> JsonlJournal:
    if not path.exists():
        path.touch()
    return JsonlJournal(path)


def _index(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise RobustnessInferenceError(f"invalid or duplicate sample ID in {path}")
        result[sample_id] = row
    return result


def _diagnostics(
    run_id: str, prediction_rows: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    per_sample = [
        {
            "run_id": run_id,
            "sample_id": row["sample_id"],
            "pattern": row["pattern"],
            "input_validity": row["input_validity"],
            "parse_status": row["parse_status"],
            "abstained_invalid": row["parse_status"] == "abstained_invalid",
            "unsupported_deterministic_output": row["parse_status"]
            == "unsupported_deterministic_output",
            "failed": row["parse_status"] == "failed",
        }
        for row in prediction_rows
    ]

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        statuses = Counter(str(row["parse_status"]) for row in rows)
        abstained = statuses["abstained_invalid"]
        unsupported = statuses["unsupported_deterministic_output"]
        failed = statuses["failed"]
        return {
            "sample_count": count,
            "abstention_count": abstained,
            "abstention_rate": abstained / count if count else 0.0,
            "invalid_input_detection_count": abstained,
            "invalid_input_detection_rate": abstained / count if count else 0.0,
            "unsupported_deterministic_output_count": unsupported,
            "unsupported_deterministic_output_rate": unsupported / count if count else 0.0,
            "failed_count": failed,
            "failed_rate": failed / count if count else 0.0,
            "parse_status_counts": dict(sorted(statuses.items())),
            "core_three_f1_defined": False,
        }

    metrics = {"schema_version": 1, "run_id": run_id, **summarize(per_sample)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_sample:
        grouped[str(row["pattern"])].append(row)
    by_family = {
        "schema_version": 1,
        "run_id": run_id,
        "families": {family: summarize(rows) for family, rows in sorted(grouped.items())},
    }
    return metrics, by_family, per_sample


def run_contradiction_inference(
    run: RunDirectory,
    samples: Sequence[EvaluationSample],
    assets: ProtocolAssets,
    backend: Callable[[EvaluationSample], BackendResponse],
    *,
    run_spec: Mapping[str, Any],
    model_id: str,
    model_revision: str,
    model_or_adapter_sha256: str,
    decode_profile: Mapping[str, Any],
) -> dict[str, Any]:
    if run_spec.get("condition_id") != "contradiction" or not run_spec.get("diagnostic_only"):
        raise RobustnessInferenceError("invalid-input runner only accepts contradiction diagnostics")
    expected_ids = [sample.sample_id for sample in samples]
    if len(set(expected_ids)) != len(expected_ids):
        raise RobustnessInferenceError("contradiction samples contain duplicate IDs")
    raw = _journal(run.artifact("raw_responses.jsonl"))
    predictions = _journal(run.artifact("predictions.jsonl"))
    errors = _journal(run.artifact("errors.jsonl"))
    raw_index = _index(raw.path)
    prediction_index = _index(predictions.path)
    if not set(prediction_index).issubset(raw_index):
        raise RobustnessInferenceError("prediction exists without a raw contradiction response")
    if (set(raw_index) | set(prediction_index)) - set(expected_ids):
        raise RobustnessInferenceError("resume journals contain out-of-scope IDs")
    run.record_status(
        "INFERENCE_STARTED",
        condition_id="contradiction",
        expected_count=len(samples),
        resumed_prediction_count=len(prediction_index),
    )
    returned_models: set[str] = set()
    for sample in samples:
        if sample.sample_id in predictions.seen_ids:
            returned = raw_index[sample.sample_id].get("returned_model_id")
            if isinstance(returned, str) and returned:
                returned_models.add(returned)
            continue
        backend_error: str | None = None
        raw_row = raw_index.get(sample.sample_id)
        if raw_row is None:
            try:
                response = backend(sample)
                raw_row = {
                    "run_id": run.run_id,
                    "sample_id": sample.sample_id,
                    "pattern": sample.pattern,
                    "backend_status": "success",
                    "returned_model_id": response.returned_model_id,
                    "output_text": response.output_text,
                    "response": response.raw_response,
                    "recorded_at_utc": utc_now(),
                }
            except Exception as exc:
                backend_error = f"{type(exc).__name__}: {exc}"
                raw_row = {
                    "run_id": run.run_id,
                    "sample_id": sample.sample_id,
                    "pattern": sample.pattern,
                    "backend_status": "failed",
                    "returned_model_id": model_revision,
                    "output_text": "",
                    "response": {},
                    "error": backend_error,
                    "recorded_at_utc": utc_now(),
                }
            raw.append(raw_row)
            raw_index[sample.sample_id] = raw_row
        returned = raw_row.get("returned_model_id")
        if isinstance(returned, str) and returned:
            returned_models.add(returned)
        validity, parse_status, parse_error = classify_contradiction_output(
            str(raw_row.get("output_text") or "")
        )
        predictions.append(
            {
                "run_id": run.run_id,
                "sample_id": sample.sample_id,
                "pattern": sample.pattern,
                "prediction": {},
                "input_validity": validity,
                "parse_status": parse_status,
                "syntax_repair": None,
            }
        )
        error = backend_error or parse_error
        if error is not None and sample.sample_id not in errors.seen_ids:
            errors.append(
                {
                    "run_id": run.run_id,
                    "sample_id": sample.sample_id,
                    "pattern": sample.pattern,
                    "error_type": "backend" if backend_error else "invalid_input_parse",
                    "error": error,
                }
            )
        run.record_status(
            "SAMPLE_COMPLETE",
            sample_id=sample.sample_id,
            parse_status=parse_status,
            completed_count=len(predictions.seen_ids),
        )
    prediction_rows = list(iter_jsonl(predictions.path))
    if [row["sample_id"] for row in prediction_rows] != expected_ids:
        raise RobustnessInferenceError("contradiction predictions differ from frozen ID order")
    metrics, by_family, per_sample = _diagnostics(run.run_id, prediction_rows)
    artifacts = {
        run.artifact("metrics.json"): metrics,
        run.artifact("metrics_by_family.json"): by_family,
    }
    for path, value in artifacts.items():
        if path.exists():
            if read_json(path) != value:
                raise RobustnessInferenceError(f"stored diagnostic drift: {path.name}")
        else:
            write_json_atomic(path, value)
    per_sample_path = run.artifact("metrics_per_sample.jsonl")
    if per_sample_path.exists():
        if list(iter_jsonl(per_sample_path)) != per_sample:
            raise RobustnessInferenceError("stored per-sample diagnostics drift")
    else:
        write_jsonl_atomic(per_sample_path, per_sample)
    implementation_sha = sha256_file(Path(__file__))
    pending = {
        "contract_id": run.contract.data["contract_id"],
        "contract_version": run.contract.data["contract_version"],
        "contract_sha256": run.contract.sha256,
        "run_id": run.run_id,
        "formal_result": False,
        "benchmark_freeze_id": run.contract.data["benchmark"]["freeze_id"],
        "benchmark_root_sha256": run.contract.data["benchmark"]["data_root_sha256"],
        "ordered_sample_id_sha256": sha256_ordered_ids(expected_ids),
        "model_id": model_id,
        "model_revision_or_provider_returned_id": model_revision,
        "returned_model_ids": sorted(returned_models),
        "model_or_adapter_sha256": model_or_adapter_sha256,
        "prompt_sha256": sha256_text(samples[0].system_prompt),
        "schema_sha256": sha256_json(run.contract.data["output_scopes"]["robustness_invalid_input_v1"]),
        "vocabulary_sha256": assets.vocabulary_sha256,
        "decode_config_sha256": sha256_json(dict(decode_profile)),
        "parser_sha256": implementation_sha,
        "scorer_sha256": implementation_sha,
        "prediction_sha256": sha256_file(predictions.path),
        "raw_response_sha256": sha256_file(raw.path),
        "error_sha256": sha256_file(errors.path),
        "completion_status": "PENDING_VALIDATION",
        "gate_status": {
            "contract_schema": "PASS",
            "required_bindings": "PASS",
            "benchmark_hashes": "PASS",
            "scorer_regeneration": "PASS",
            "comparability": "PASS",
            "non_partial": "PASS",
        },
        "diagnostic_primary_metric": {
            "name": "abstention_rate",
            "value": metrics["abstention_rate"],
        },
        "core_three_f1_defined": False,
        "created_at_utc": read_json(run.artifact("input_manifest.json"))["created_at_utc"],
    }
    pending_path = run.artifact("run_manifest.pending.json")
    if pending_path.exists():
        if read_json(pending_path) != pending:
            raise RobustnessInferenceError("pending contradiction manifest drifted during resume")
    else:
        write_json_atomic(pending_path, pending)
    report = CompletionValidator(run.contract, run.path, expected_ids).finalize_formal_run()
    run.record_status("INFERENCE_COMPLETE", completion_status=report["status"])
    return report

