"""Append-only common inference runner with resume and evidence finalization."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .completion import CompletionValidator
from .hashing import sha256_file, sha256_json, sha256_ordered_ids
from .inference_backend import BackendResponse, InferenceBackendError
from .io import JsonlJournal, append_jsonl, iter_jsonl, read_json, write_json_atomic
from .parsing import parse_prediction_text
from .protocols import EvaluationSample, InferenceInput, ProtocolAssets, label_blind_input
from .runs import RunDirectory, utc_now
from .scoring import ScoreBundle, score_predictions, write_score_artifacts


class InferenceRunError(RuntimeError):
    """Raised when a common inference run cannot close its evidence chain."""


def _ensure_journal(path: Path) -> JsonlJournal:
    if not path.exists():
        path.touch()
    return JsonlJournal(path)


def _index_rows(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return result
    for row in iter_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in result:
            raise InferenceRunError(f"invalid or duplicate sample ID in {path}: {sample_id}")
        result[sample_id] = row
    return result


def _existing_attempt_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    for row in iter_jsonl(path):
        attempt_id = row.get("request_attempt_id")
        if not isinstance(attempt_id, str) or not attempt_id or attempt_id in result:
            raise InferenceRunError(f"invalid or duplicate request attempt ID in {path}")
        result.add(attempt_id)
    return result


def _append_attempts(
    path: Path,
    seen: set[str],
    *,
    run_id: str,
    sample_id: str,
    invocation_id: str,
    attempts: Sequence[Mapping[str, Any]],
) -> None:
    for attempt in attempts:
        attempt_id = f"{sample_id}:{invocation_id}:{attempt.get('attempt_number')}"
        if attempt_id in seen:
            continue
        append_jsonl(
            path,
            {
                "request_attempt_id": attempt_id,
                "run_id": run_id,
                "sample_id": sample_id,
                "recorded_at_utc": utc_now(),
                **dict(attempt),
            },
        )
        seen.add(attempt_id)


def _prefetch_backend_responses(
    run: RunDirectory,
    samples: Sequence[EvaluationSample],
    backend: Callable[[InferenceInput], BackendResponse],
    raw_index: Mapping[str, Mapping[str, Any]],
    *,
    max_workers: int,
    request_attempts_path: Path,
    seen_attempt_ids: set[str],
) -> dict[str, Path]:
    staging = run.artifact("staging_responses")
    staging.mkdir(exist_ok=True)
    staged: dict[str, Path] = {}
    missing: list[tuple[int, EvaluationSample, Path]] = []
    for index, sample in enumerate(samples):
        path = staging / f"{index:04d}_{sample.sample_id}.json"
        if path.is_file():
            value = read_json(path)
            if (
                value.get("run_id") != run.run_id
                or value.get("sample_id") != sample.sample_id
                or value.get("backend_status") != "success"
            ):
                raise InferenceRunError(f"staged response identity drift: {path.name}")
            staged[sample.sample_id] = path
        elif sample.sample_id not in raw_index:
            missing.append((index, sample, path))
    if not missing:
        return staged

    failures: list[tuple[EvaluationSample, Exception, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=run.run_id) as pool:
        futures = {
            pool.submit(backend, label_blind_input(sample)): (index, sample, path)
            for index, sample, path in missing
        }
        for future in as_completed(futures):
            _, sample, path = futures[future]
            try:
                response = future.result()
                stage = {
                    "run_id": run.run_id,
                    "sample_id": sample.sample_id,
                    "pattern": sample.pattern,
                    "backend_status": "success",
                    "returned_model_id": response.returned_model_id,
                    "output_text": response.output_text,
                    "response": response.raw_response,
                    "request_attempts": list(response.request_attempts),
                    "recorded_at_utc": utc_now(),
                }
                write_json_atomic(path, stage)
                staged[sample.sample_id] = path
                run.record_status(
                    "RESPONSE_STAGED",
                    sample_id=sample.sample_id,
                    staged_count=len(staged),
                )
            except Exception as exc:
                failure_id = uuid4().hex
                failures.append((sample, exc, failure_id))
                attempts = exc.attempts if isinstance(exc, InferenceBackendError) else ()
                failure_dir = run.artifact("staging_failures")
                failure_dir.mkdir(exist_ok=True)
                write_json_atomic(
                    failure_dir / f"{sample.sample_id}_{failure_id}.json",
                    {
                        "run_id": run.run_id,
                        "sample_id": sample.sample_id,
                        "error_type": type(exc).__name__,
                        "request_attempts": list(attempts),
                        "recorded_at_utc": utc_now(),
                    },
                )
                _append_attempts(
                    request_attempts_path,
                    seen_attempt_ids,
                    run_id=run.run_id,
                    sample_id=sample.sample_id,
                    invocation_id=f"failure:{failure_id}",
                    attempts=attempts,
                )
    if failures:
        sample, exc, _ = failures[0]
        run.record_status(
            "API_REQUEST_FAILED",
            sample_id=sample.sample_id,
            error_type=type(exc).__name__,
            failure_count=len(failures),
        )
        raise InferenceRunError(
            f"backend request failed for {sample.sample_id}; successful staged responses are resumable"
        ) from exc
    return staged


def _score_or_verify(
    run: RunDirectory,
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    *,
    output_scope: str,
    statistics: Mapping[str, Any],
) -> ScoreBundle:
    bundle = score_predictions(
        gold_rows,
        prediction_rows,
        run_id=run.run_id,
        output_scope=output_scope,
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        confidence_level=float(statistics["confidence_level"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
    )
    paths = (
        run.artifact("metrics.json"),
        run.artifact("metrics_by_family.json"),
        run.artifact("metrics_per_sample.jsonl"),
    )
    if not any(path.exists() for path in paths):
        write_score_artifacts(bundle, run.path)
        return bundle
    if not all(path.exists() for path in paths):
        raise InferenceRunError("partial scorer artifact trio prevents safe resume")
    stored = ScoreBundle(
        metrics=read_json(paths[0]),
        metrics_by_family=read_json(paths[1]),
        per_sample=tuple(iter_jsonl(paths[2])),
    )
    if stored != bundle:
        raise InferenceRunError("stored metrics do not regenerate from frozen gold and predictions")
    return bundle


def run_common_inference(
    run: RunDirectory,
    samples: Sequence[EvaluationSample],
    assets: ProtocolAssets,
    backend: Callable[[InferenceInput], BackendResponse],
    *,
    run_spec: Mapping[str, Any],
    model_id: str,
    model_revision_or_provider_id: str,
    model_or_adapter_sha256: str,
    decode_profile: Mapping[str, Any],
    expected_returned_model_id: str | None = None,
    backend_parallelism: int = 1,
) -> dict[str, Any]:
    expected_ids = [sample.sample_id for sample in samples]
    if len(expected_ids) != len(set(expected_ids)):
        raise InferenceRunError("evaluation samples contain duplicate IDs")
    if run.run_id != run_spec.get("run_id"):
        raise InferenceRunError("run directory and declared inference spec disagree")
    raw_journal = _ensure_journal(run.artifact("raw_responses.jsonl"))
    prediction_journal = _ensure_journal(run.artifact("predictions.jsonl"))
    error_journal = _ensure_journal(run.artifact("errors.jsonl"))
    request_attempts_path = run.artifact("request_attempts.jsonl")
    if not request_attempts_path.exists():
        request_attempts_path.touch()
    seen_attempt_ids = _existing_attempt_ids(request_attempts_path)
    raw_index = _index_rows(raw_journal.path)
    prediction_index = _index_rows(prediction_journal.path)
    if not set(prediction_index).issubset(raw_index):
        raise InferenceRunError("a prediction exists without its raw response trace")
    unknown = (set(raw_index) | set(prediction_index)) - set(expected_ids)
    if unknown:
        raise InferenceRunError(f"resume journals contain out-of-scope IDs: {sorted(unknown)[:5]}")
    run.record_status(
        "INFERENCE_STARTED",
        expected_count=len(samples),
        resumed_prediction_count=len(prediction_index),
    )
    if backend_parallelism < 1:
        raise InferenceRunError("backend_parallelism must be positive")
    staged = (
        _prefetch_backend_responses(
            run,
            samples,
            backend,
            raw_index,
            max_workers=backend_parallelism,
            request_attempts_path=request_attempts_path,
            seen_attempt_ids=seen_attempt_ids,
        )
        if backend_parallelism > 1
        else {}
    )
    returned_models: set[str] = set()
    for sample in samples:
        if sample.sample_id in prediction_journal.seen_ids:
            returned = raw_index[sample.sample_id].get("returned_model_id")
            if isinstance(returned, str) and returned:
                returned_models.add(returned)
            continue
        raw_row = raw_index.get(sample.sample_id)
        if raw_row is None:
            if backend_parallelism > 1:
                stage_path = staged.get(sample.sample_id)
                if stage_path is None:
                    raise InferenceRunError(f"staged response missing for {sample.sample_id}")
                stage = read_json(stage_path)
                _append_attempts(
                    request_attempts_path,
                    seen_attempt_ids,
                    run_id=run.run_id,
                    sample_id=sample.sample_id,
                    invocation_id="success",
                    attempts=stage.get("request_attempts", []),
                )
                raw_row = {
                    key: value for key, value in stage.items() if key != "request_attempts"
                }
                if (
                    expected_returned_model_id is not None
                    and raw_row.get("returned_model_id") != expected_returned_model_id
                ):
                    raw_row["backend_status"] = "identity_mismatch"
            else:
                try:
                    response = backend(label_blind_input(sample))
                    _append_attempts(
                        request_attempts_path,
                        seen_attempt_ids,
                        run_id=run.run_id,
                        sample_id=sample.sample_id,
                        invocation_id=f"invocation:{uuid4().hex}",
                        attempts=response.request_attempts,
                    )
                    backend_status = "success"
                    if (
                        expected_returned_model_id is not None
                        and response.returned_model_id != expected_returned_model_id
                    ):
                        backend_status = "identity_mismatch"
                    raw_row = {
                        "run_id": run.run_id,
                        "sample_id": sample.sample_id,
                        "pattern": sample.pattern,
                        "backend_status": backend_status,
                        "returned_model_id": response.returned_model_id,
                        "output_text": response.output_text,
                        "response": response.raw_response,
                        "recorded_at_utc": utc_now(),
                    }
                except Exception as exc:
                    attempts = exc.attempts if isinstance(exc, InferenceBackendError) else ()
                    _append_attempts(
                        request_attempts_path,
                        seen_attempt_ids,
                        run_id=run.run_id,
                        sample_id=sample.sample_id,
                        invocation_id=f"failure:{uuid4().hex}",
                        attempts=attempts,
                    )
                    run.record_status(
                        "API_REQUEST_FAILED",
                        sample_id=sample.sample_id,
                        error_type=type(exc).__name__,
                        attempt_count=len(attempts),
                    )
                    raise InferenceRunError(
                        f"backend request failed for {sample.sample_id}; run remains resumable"
                    ) from exc
            raw_journal.append(raw_row)
            raw_index[sample.sample_id] = raw_row
        if raw_row.get("backend_status") != "success":
            run.record_status(
                "MODEL_IDENTITY_MISMATCH",
                sample_id=sample.sample_id,
                expected_returned_model_id=expected_returned_model_id,
                actual_returned_model_id=raw_row.get("returned_model_id"),
            )
            raise InferenceRunError(
                f"provider model identity mismatch for {sample.sample_id}; formal run blocked"
            )
        returned = raw_row.get("returned_model_id")
        if isinstance(returned, str) and returned:
            returned_models.add(returned)
        parsed = parse_prediction_text(
            str(raw_row.get("output_text") or ""),
            output_scope=str(run_spec["output_scope"]),
            schema=assets.schema,
            vocabulary=assets.vocabulary,
        )
        prediction_journal.append(
            {
                "run_id": run.run_id,
                "sample_id": sample.sample_id,
                "pattern": sample.pattern,
                "prediction": parsed.prediction,
                "input_validity": "valid",
                "parse_status": parsed.status,
                "syntax_repair": parsed.repair,
            }
        )
        if parsed.error is not None and sample.sample_id not in error_journal.seen_ids:
            error_journal.append(
                {
                    "run_id": run.run_id,
                    "sample_id": sample.sample_id,
                    "pattern": sample.pattern,
                    "error_type": "parse_or_schema",
                    "error": parsed.error,
                }
            )
        run.record_status(
            "SAMPLE_COMPLETE",
            sample_id=sample.sample_id,
            parse_status=parsed.status,
            completed_count=len(prediction_journal.seen_ids),
        )

    prediction_rows = list(iter_jsonl(prediction_journal.path))
    if [row["sample_id"] for row in prediction_rows] != expected_ids:
        raise InferenceRunError("prediction rows are not the exact frozen ordered ID sequence")
    statistics = run.contract.data["scoring_and_statistics"]["uncertainty"]
    bundle = _score_or_verify(
        run,
        [sample.source for sample in samples],
        prediction_rows,
        output_scope=str(run_spec["output_scope"]),
        statistics=statistics,
    )
    regenerated = score_predictions(
        [sample.source for sample in samples],
        prediction_rows,
        run_id=run.run_id,
        output_scope=str(run_spec["output_scope"]),
        bootstrap_replicates=int(statistics["bootstrap_replicates"]),
        confidence_level=float(statistics["confidence_level"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
    )
    if regenerated != bundle:
        raise InferenceRunError("independent scorer regeneration drifted")
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
        "model_revision_or_provider_returned_id": model_revision_or_provider_id,
        "returned_model_ids": sorted(returned_models),
        "model_or_adapter_sha256": model_or_adapter_sha256,
        "prompt_sha256": assets.prompt_sha256,
        "schema_sha256": assets.schema_sha256,
        "vocabulary_sha256": assets.vocabulary_sha256,
        "decode_config_sha256": sha256_json(dict(decode_profile)),
        "parser_sha256": assets.parser_sha256,
        "scorer_sha256": assets.scorer_sha256,
        "prediction_sha256": sha256_file(prediction_journal.path),
        "raw_response_sha256": sha256_file(raw_journal.path),
        "error_sha256": sha256_file(error_journal.path),
        "request_attempts_sha256": sha256_file(request_attempts_path),
        "backend_parallelism": backend_parallelism,
        "completion_status": "PENDING_VALIDATION",
        "gate_status": {
            "contract_schema": "PASS",
            "required_bindings": "PASS",
            "benchmark_hashes": "PASS",
            "scorer_regeneration": "PASS",
            "comparability": "PASS",
            "non_partial": "PASS",
        },
        "primary_metric": bundle.metrics["primary_mean_field_f1"],
        "created_at_utc": read_json(run.artifact("input_manifest.json"))["created_at_utc"],
    }
    pending_path = run.artifact("run_manifest.pending.json")
    if pending_path.exists():
        if read_json(pending_path) != pending:
            raise InferenceRunError("pending manifest drifted during resume")
    else:
        write_json_atomic(pending_path, pending)
    validator = CompletionValidator(run.contract, run.path, expected_ids)
    report = validator.finalize_formal_run()
    run.record_status("INFERENCE_COMPLETE", completion_status=report["status"])
    return report
