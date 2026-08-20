"""Evidence-complete validator for contract-declared full-test runs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .contracts import LoadedContract
from .hashing import is_sha256, sha256_file, sha256_ordered_ids
from .io import JsonFormatError, audit_jsonl_ids, iter_jsonl, read_json, write_json_atomic
from .runs import RunDirectory


REQUIRED_EXTERNAL_GATES = (
    "contract_schema",
    "required_bindings",
    "benchmark_hashes",
    "scorer_regeneration",
    "comparability",
    "non_partial",
)


@dataclass(frozen=True)
class Check:
    check_id: str
    status: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "check_id": self.check_id,
            "status": self.status,
            "message": self.message,
        }
        if self.details is not None:
            value["details"] = self.details
        return value


class CompletionError(RuntimeError):
    """Raised when a caller tries to finalize an incomplete formal run."""


class CompletionValidator:
    def __init__(
        self,
        contract: LoadedContract,
        run_dir: str | Path,
        expected_ids: Iterable[str],
    ) -> None:
        self.contract = contract
        self.run_dir = Path(run_dir).resolve()
        self.expected_ids = tuple(expected_ids)
        if not self.expected_ids or not all(isinstance(item, str) and item for item in self.expected_ids):
            raise ValueError("expected IDs must be non-empty strings")
        if len(set(self.expected_ids)) != len(self.expected_ids):
            raise ValueError("expected IDs contain duplicates")

    def _add(self, checks: list[Check], check_id: str, passed: bool, message: str, **details: Any) -> None:
        checks.append(Check(check_id, "PASS" if passed else "FAIL", message, details or None))

    def _load_pending_manifest(self, checks: list[Check]) -> dict[str, Any] | None:
        path = self.run_dir / "run_manifest.pending.json"
        if not path.is_file():
            self._add(checks, "pending_manifest", False, "run_manifest.pending.json is missing")
            return None
        try:
            value = read_json(path)
        except (OSError, JsonFormatError) as exc:
            self._add(checks, "pending_manifest", False, str(exc))
            return None
        if not isinstance(value, dict):
            self._add(checks, "pending_manifest", False, "pending manifest must be an object")
            return None
        self._add(checks, "pending_manifest", True, "pending manifest loaded")
        return value

    def _audit_exact_ids(self, checks: list[Check], filename: str, *, exact: bool = True) -> Any:
        path = self.run_dir / filename
        if not path.is_file():
            self._add(checks, f"ids:{filename}", False, f"{filename} is missing")
            return None
        try:
            audit = audit_jsonl_ids(path, self.expected_ids)
        except (OSError, JsonFormatError, ValueError) as exc:
            self._add(checks, f"ids:{filename}", False, str(exc))
            return None
        passed = audit.is_exact if exact else not audit.duplicate_ids and not audit.extra_ids
        self._add(
            checks,
            f"ids:{filename}",
            passed,
            f"{filename} ID audit {'passed' if passed else 'failed'}",
            row_count=audit.row_count,
            unique_count=audit.unique_count,
            duplicate_ids=list(audit.duplicate_ids),
            missing_ids=list(audit.missing_ids) if exact else [],
            extra_ids=list(audit.extra_ids),
        )
        return audit

    def validate(self, *, require_formal_ready: bool = True) -> dict[str, Any]:
        checks: list[Check] = []
        run_id = self.run_dir.name
        run_specs = {item["run_id"]: item for item in self.contract.inference_run_specs}
        spec = run_specs.get(run_id)
        self._add(checks, "declared_run", spec is not None, "run ID is declared in the formal matrix")

        workflow = spec.get("workflow") if spec is not None else None
        ready = bool(
            workflow is not None and self.contract.is_workflow_formal_ready(str(workflow))
        )
        self._add(
            checks,
            "formal_contract_ready",
            ready if require_formal_ready else True,
            "run workflow is formal-run ready" if ready else "run workflow remains blocked for formal execution",
            workflow=workflow,
            blockers=(
                list(self.contract.required_bindings_for_workflow(str(workflow)))
                if workflow is not None
                else []
            ),
        )
        if spec is not None:
            expected_count = spec["expected_prediction_count"]
            self._add(
                checks,
                "expected_count",
                len(self.expected_ids) == expected_count,
                "expected ID count matches run matrix",
                expected=expected_count,
                actual=len(self.expected_ids),
            )

        artifact_contract = self.contract.data["artifacts"]
        required = set(artifact_contract["required_files_common_runs"]) | set(
            artifact_contract["required_files_evaluation_runs"]
        )
        if spec is not None and str(spec.get("model_key", "")).startswith("gpt_"):
            required |= set(artifact_contract.get("required_files_gpt_evaluation_runs", []))
        deferred = {"run_manifest.json", "completion_report.json", "status.json"}
        missing_files = sorted(
            name for name in required - deferred if not (self.run_dir / name).is_file()
        )
        self._add(
            checks,
            "required_artifacts",
            not missing_files,
            "all pre-finalization artifacts exist" if not missing_files else "required artifacts are missing",
            missing=missing_files,
        )

        for filename in ("input_manifest.json", "environment.json", "metrics.json", "metrics_by_family.json"):
            path = self.run_dir / filename
            if not path.is_file():
                continue
            try:
                value = read_json(path)
                passed = isinstance(value, dict)
                message = f"{filename} is a JSON object" if passed else f"{filename} must be a JSON object"
            except (OSError, JsonFormatError) as exc:
                passed = False
                message = str(exc)
            self._add(checks, f"json:{filename}", passed, message)

        prediction_audit = self._audit_exact_ids(checks, "predictions.jsonl")
        raw_audit = self._audit_exact_ids(checks, "raw_responses.jsonl")
        metric_audit = self._audit_exact_ids(checks, "metrics_per_sample.jsonl")
        error_audit = self._audit_exact_ids(checks, "errors.jsonl", exact=False)

        raw_returned_models: set[str] = set()
        raw_backend_failures: list[str] = []
        if raw_audit is not None:
            try:
                for row in iter_jsonl(self.run_dir / "raw_responses.jsonl"):
                    if row.get("backend_status") != "success":
                        raw_backend_failures.append(str(row.get("sample_id")))
                    returned = row.get("returned_model_id")
                    if isinstance(returned, str) and returned:
                        raw_returned_models.add(returned)
                self._add(
                    checks,
                    "raw_backend_success",
                    not raw_backend_failures,
                    "all raw rows are successful provider/model responses",
                    failed_ids=raw_backend_failures,
                )
            except (OSError, JsonFormatError) as exc:
                self._add(checks, "raw_backend_success", False, str(exc))

        failed_prediction_ids: set[str] = set()
        if prediction_audit is not None:
            try:
                mismatched_run_ids: list[str] = []
                for row in iter_jsonl(self.run_dir / "predictions.jsonl"):
                    if row.get("run_id") != run_id:
                        mismatched_run_ids.append(str(row.get("sample_id")))
                    if row.get("parse_status") == "failed":
                        failed_prediction_ids.add(row["sample_id"])
                self._add(
                    checks,
                    "prediction_run_trace",
                    not mismatched_run_ids,
                    "prediction rows trace to this run",
                    mismatched_ids=mismatched_run_ids,
                )
            except (OSError, JsonFormatError, KeyError) as exc:
                self._add(checks, "prediction_run_trace", False, str(exc))

        for filename, audit in (
            ("raw_responses.jsonl", raw_audit),
            ("errors.jsonl", error_audit),
            ("metrics_per_sample.jsonl", metric_audit),
        ):
            if audit is None:
                continue
            try:
                mismatched = [
                    str(row.get("sample_id"))
                    for row in iter_jsonl(self.run_dir / filename)
                    if row.get("run_id") != run_id
                ]
                self._add(
                    checks,
                    f"run_trace:{filename}",
                    not mismatched,
                    f"{filename} rows trace to this run",
                    mismatched_ids=mismatched,
                )
            except (OSError, JsonFormatError) as exc:
                self._add(checks, f"run_trace:{filename}", False, str(exc))

        error_ids = set(error_audit.observed_ids) if error_audit is not None else set()
        missing_error_rows = sorted(failed_prediction_ids - error_ids)
        self._add(
            checks,
            "failed_prediction_error_trace",
            not missing_error_rows,
            "every parse failure has an error row",
            missing_error_rows=missing_error_rows,
        )
        backend_error_rows: list[str] = []
        if error_audit is not None:
            backend_error_rows = [
                str(row.get("sample_id"))
                for row in iter_jsonl(self.run_dir / "errors.jsonl")
                if row.get("error_type") == "backend"
            ]
        self._add(
            checks,
            "no_backend_errors_as_predictions",
            not backend_error_rows,
            "API/infrastructure failures were not converted to model predictions",
            failed_ids=backend_error_rows,
        )

        request_attempts_required = (
            spec is not None
            and str(spec.get("model_key", "")).startswith("gpt_")
            and "request_attempts.jsonl"
            in artifact_contract.get("required_files_gpt_evaluation_runs", [])
        )
        if request_attempts_required:
            attempt_path = self.run_dir / "request_attempts.jsonl"
            successful_attempt_ids: set[str] = set()
            malformed_attempts: list[int] = []
            if attempt_path.is_file():
                for index, row in enumerate(iter_jsonl(attempt_path), start=1):
                    sample_id = row.get("sample_id")
                    if (
                        row.get("run_id") != run_id
                        or sample_id not in set(self.expected_ids)
                        or not isinstance(row.get("request_identity_sha256"), str)
                    ):
                        malformed_attempts.append(index)
                    if row.get("status") == "success" and isinstance(sample_id, str):
                        successful_attempt_ids.add(sample_id)
            missing_success = sorted(set(self.expected_ids) - successful_attempt_ids)
            self._add(
                checks,
                "request_attempt_trace",
                attempt_path.is_file() and not malformed_attempts and not missing_success,
                "every sample has a traceable successful API request attempt",
                malformed_rows=malformed_attempts,
                missing_success_ids=missing_success,
            )

        pending = self._load_pending_manifest(checks)
        if pending is not None:
            required_fields = self.contract.data["artifacts"]["required_manifest_fields"]
            missing_fields = sorted(field for field in required_fields if field not in pending)
            self._add(
                checks,
                "manifest_fields",
                not missing_fields,
                "pending manifest contains required fields",
                missing=missing_fields,
            )
            identity_ok = (
                pending.get("contract_id") == self.contract.data["contract_id"]
                and pending.get("contract_version") == self.contract.data["contract_version"]
                and pending.get("contract_sha256") == self.contract.sha256
                and pending.get("run_id") == run_id
                and pending.get("benchmark_freeze_id") == self.contract.data["benchmark"]["freeze_id"]
                and pending.get("benchmark_root_sha256") == self.contract.data["benchmark"]["data_root_sha256"]
                and pending.get("ordered_sample_id_sha256") == sha256_ordered_ids(self.expected_ids)
            )
            self._add(checks, "manifest_identity", identity_ok, "manifest identity and frozen hashes match")

            model_identity_ok = (
                isinstance(pending.get("model_id"), str)
                and bool(pending.get("model_id"))
                and isinstance(pending.get("model_revision_or_provider_returned_id"), str)
                and bool(pending.get("model_revision_or_provider_returned_id"))
            )
            self._add(
                checks,
                "model_identity",
                model_identity_ok,
                "model ID and immutable revision/provider-returned ID are present",
            )
            if spec is not None and str(spec.get("model_key", "")).startswith("gpt_"):
                declared_returned = pending.get("model_revision_or_provider_returned_id")
                recorded_returned = pending.get("returned_model_ids")
                provider_identity_ok = (
                    isinstance(declared_returned, str)
                    and raw_returned_models == {declared_returned}
                    and recorded_returned == [declared_returned]
                )
                self._add(
                    checks,
                    "provider_returned_model_identity",
                    provider_identity_ok,
                    "all GPT responses match the provider identity frozen by the contract",
                    declared=declared_returned,
                    observed=sorted(raw_returned_models),
                )
                expected_parallelism = self.contract.data["gpt_protocol"].get(
                    "within_run_parallelism"
                )
                actual_parallelism = pending.get("backend_parallelism")
                self._add(
                    checks,
                    "gpt_backend_parallelism",
                    actual_parallelism == expected_parallelism,
                    "GPT within-run request parallelism matches the contract",
                    expected=expected_parallelism,
                    actual=actual_parallelism,
                )
                if isinstance(expected_parallelism, int) and expected_parallelism > 1:
                    staging = self.run_dir / "staging_responses"
                    stage_ids: list[str] = []
                    malformed_stages: list[str] = []
                    for path in sorted(staging.glob("*.json")) if staging.is_dir() else []:
                        value = read_json(path)
                        sample_id = value.get("sample_id")
                        if (
                            value.get("run_id") != run_id
                            or value.get("backend_status") != "success"
                            or not isinstance(sample_id, str)
                        ):
                            malformed_stages.append(path.name)
                        else:
                            stage_ids.append(sample_id)
                    self._add(
                        checks,
                        "staged_response_trace",
                        not malformed_stages
                        and len(stage_ids) == len(self.expected_ids)
                        and set(stage_ids) == set(self.expected_ids),
                        "all concurrent GPT responses have successful staging evidence",
                        staged_count=len(stage_ids),
                        malformed=malformed_stages,
                    )

            premature = pending.get("formal_result") is True or pending.get("completion_status") == "PASS"
            self._add(
                checks,
                "no_premature_formal_result",
                not premature,
                "pending manifest has not claimed formal completion",
            )

            gate_status = pending.get("gate_status")
            missing_gates = []
            failed_gates = []
            if not isinstance(gate_status, dict):
                missing_gates = list(REQUIRED_EXTERNAL_GATES)
            else:
                missing_gates = [key for key in REQUIRED_EXTERNAL_GATES if key not in gate_status]
                failed_gates = [key for key in REQUIRED_EXTERNAL_GATES if gate_status.get(key) != "PASS"]
            self._add(
                checks,
                "external_gates",
                not missing_gates and not failed_gates,
                "all upstream completion prerequisites passed",
                missing=missing_gates,
                failed=failed_gates,
            )

            hash_fields = {
                "prediction_sha256": "predictions.jsonl",
                "raw_response_sha256": "raw_responses.jsonl",
                "error_sha256": "errors.jsonl",
            }
            if request_attempts_required:
                hash_fields["request_attempts_sha256"] = "request_attempts.jsonl"
            drift: dict[str, dict[str, str | None]] = {}
            for field, filename in hash_fields.items():
                path = self.run_dir / filename
                expected = pending.get(field)
                actual = sha256_file(path) if path.is_file() else None
                if not is_sha256(expected) or actual != expected:
                    drift[field] = {"expected": expected, "actual": actual}
            self._add(
                checks,
                "artifact_hashes",
                not drift,
                "prediction/raw/error hashes match closed artifacts",
                drift=drift,
            )

            binding_hash_fields = (
                "model_or_adapter_sha256",
                "prompt_sha256",
                "schema_sha256",
                "vocabulary_sha256",
                "decode_config_sha256",
                "parser_sha256",
                "scorer_sha256",
            )
            invalid_hash_fields = [field for field in binding_hash_fields if not is_sha256(pending.get(field))]
            self._add(
                checks,
                "binding_hash_formats",
                not invalid_hash_fields,
                "all model/prompt/schema/parser/scorer bindings are SHA-256 values",
                invalid=invalid_hash_fields,
            )

        status = "PASS" if all(check.status == "PASS" for check in checks) else "FAIL"
        return {
            "schema_version": 1,
            "run_id": run_id,
            "contract_id": self.contract.data["contract_id"],
            "contract_version": self.contract.data["contract_version"],
            "contract_sha256": self.contract.sha256,
            "validated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "formal_result_eligible": status == "PASS" and require_formal_ready,
            "counts": {
                "expected": len(self.expected_ids),
                "predictions": prediction_audit.row_count if prediction_audit else None,
                "raw_responses": raw_audit.row_count if raw_audit else None,
                "errors": error_audit.row_count if error_audit else None,
                "metrics_per_sample": metric_audit.row_count if metric_audit else None,
            },
            "checks": [check.to_dict() for check in checks],
        }

    def finalize_formal_run(self) -> dict[str, Any]:
        completion_path = self.run_dir / "completion_report.json"
        manifest_path = self.run_dir / "run_manifest.json"
        status_path = self.run_dir / "status.json"
        if completion_path.exists() or manifest_path.exists() or status_path.exists():
            if not (completion_path.exists() and manifest_path.exists() and status_path.exists()):
                raise CompletionError("partial finalization detected; final artifact trio is incomplete")
            existing_report = read_json(completion_path)
            existing_manifest = read_json(manifest_path)
            existing_status = read_json(status_path)
            artifact_hashes_match = (
                existing_manifest.get("prediction_sha256")
                == sha256_file(self.run_dir / "predictions.jsonl")
                and existing_manifest.get("raw_response_sha256")
                == sha256_file(self.run_dir / "raw_responses.jsonl")
                and existing_manifest.get("error_sha256")
                == sha256_file(self.run_dir / "errors.jsonl")
            )
            if not (
                existing_report.get("status") == "PASS"
                and existing_manifest.get("formal_result") is True
                and existing_manifest.get("completion_status") == "PASS"
                and existing_status.get("state") == "COMPLETE"
                and existing_status.get("formal_result") is True
                and existing_manifest.get("completion_report_sha256") == sha256_file(completion_path)
                and existing_status.get("run_manifest_sha256") == sha256_file(manifest_path)
                and artifact_hashes_match
            ):
                raise CompletionError("existing finalized run fails immutable-artifact audit")
            return existing_report

        report = self.validate(require_formal_ready=True)
        if report["status"] != "PASS":
            attempts = self.run_dir / "completion_attempts"
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            write_json_atomic(attempts / f"completion_report_{timestamp}.json", report)
            raise CompletionError("formal completion failed; see completion_attempts")

        pending = read_json(self.run_dir / "run_manifest.pending.json")
        final_manifest = deepcopy(pending)
        final_manifest["formal_result"] = True
        final_manifest["completion_status"] = "PASS"
        final_manifest["completed_at_utc"] = report["validated_at_utc"]

        if completion_path.exists():
            existing = read_json(completion_path)
            if existing.get("status") != "PASS":
                raise CompletionError("existing completion_report.json is not PASS")
        else:
            write_json_atomic(completion_path, report)
        final_manifest["completion_report_sha256"] = sha256_file(completion_path)

        if manifest_path.exists():
            existing = read_json(manifest_path)
            if existing != final_manifest:
                raise CompletionError("existing run_manifest.json differs from validated final manifest")
        else:
            write_json_atomic(manifest_path, final_manifest)

        status_value = {
            "run_id": self.run_dir.name,
            "state": "COMPLETE",
            "formal_result": True,
            "completion_status": "PASS",
            "completed_at_utc": report["validated_at_utc"],
            "run_manifest_sha256": sha256_file(manifest_path),
            "completion_report_sha256": sha256_file(completion_path),
        }
        if status_path.exists():
            if read_json(status_path) != status_value:
                raise CompletionError("existing status.json differs from validated final status")
        else:
            write_json_atomic(status_path, status_value)
        return report
