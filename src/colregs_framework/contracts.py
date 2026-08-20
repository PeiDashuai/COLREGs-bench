"""Experiment-contract loading, standard schema validation, and invariants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator

from .hashing import sha256_file, verify_file_sha256
from .io import read_json


class ContractError(ValueError):
    """Base class for contract failures."""


class ContractDependencyError(ContractError):
    """Raised when the standard JSON Schema validator is unavailable."""


class ContractSchemaError(ContractError):
    """Raised when the contract fails Draft 2020-12 validation."""


class ContractInvariantError(ContractError):
    """Raised when a cross-field experiment invariant is violated."""


class FormalRunBlockedError(ContractError):
    """Raised when an operation requests formal execution before readiness."""


@dataclass(frozen=True)
class LoadedContract:
    data: dict[str, Any]
    path: Path
    schema_path: Path
    sha256: str

    @property
    def required_formal_bindings(self) -> tuple[str, ...]:
        return tuple(
            item["binding_id"]
            for item in self.data["unresolved_bindings"]
            if item.get("required_before_formal_run") is True
        )

    @property
    def is_formal_ready(self) -> bool:
        return (
            self.data.get("status") == "FORMAL_RUN_READY"
            and self.data.get("formal_runs_allowed") is True
            and not self.required_formal_bindings
        )

    def required_bindings_for_workflow(self, workflow: str) -> tuple[str, ...]:
        workflows = self.data.get("formal_workflows", {})
        spec = workflows.get(workflow)
        if not isinstance(spec, dict):
            raise FormalRunBlockedError(f"contract does not declare formal workflow: {workflow}")
        resolved = {
            item["binding_id"]
            for item in self.data.get("resolved_bindings", [])
            if isinstance(item, dict) and isinstance(item.get("binding_id"), str)
        }
        return tuple(
            binding_id
            for binding_id in spec.get("required_binding_ids", [])
            if binding_id not in resolved
        )

    def is_workflow_formal_ready(self, workflow: str) -> bool:
        spec = self.data.get("formal_workflows", {}).get(workflow)
        return bool(
            isinstance(spec, dict)
            and spec.get("status") == "FORMAL_RUN_READY"
            and spec.get("formal_runs_allowed") is True
            and not self.required_bindings_for_workflow(workflow)
        )

    def formal_workflow_for_run(self, run_id: str) -> str:
        for item in self.inference_run_specs:
            if item["run_id"] == run_id:
                return str(item["workflow"])
        for item in self.data["training"]["regimes"]:
            if item["regime_id"] == run_id:
                workflow = item.get("formal_workflow")
                if isinstance(workflow, str) and workflow:
                    return workflow
        raise FormalRunBlockedError(f"formal run is not declared by the contract: {run_id}")

    @property
    def inference_run_specs(self) -> tuple[dict[str, Any], ...]:
        """All formal inference specs, including a versioned robustness extension."""

        full = tuple(self.data["formal_full_test_inference_runs"])
        robustness = self.data.get("robustness", {}).get("runs", [])
        if not isinstance(robustness, list):
            raise ContractInvariantError("robustness.runs must be a list")
        symbolic = self.data.get("symbolic_baseline", {}).get("runs", [])
        if not isinstance(symbolic, list):
            raise ContractInvariantError("symbolic_baseline.runs must be a list")
        return full + tuple(robustness) + tuple(symbolic)

    def require_formal_ready(self, workflow: str | None = None) -> None:
        if workflow is not None:
            if self.is_workflow_formal_ready(workflow):
                return
            blockers = ", ".join(self.required_bindings_for_workflow(workflow)) or "status/flag mismatch"
            raise FormalRunBlockedError(
                f"workflow is not formal-run ready ({workflow}); blockers: {blockers}"
            )
        if not self.is_formal_ready:
            blockers = ", ".join(self.required_formal_bindings) or "status/flag mismatch"
            raise FormalRunBlockedError(
                f"contract is not formal-run ready ({self.data.get('status')}); blockers: {blockers}"
            )


def _iter_key_values(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key, child
            yield from _iter_key_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_key_values(child)


def _validate_schema(data: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise ContractDependencyError(
            "jsonschema is required; install this package from pyproject.toml"
        ) from exc

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema exposes several schema exception types
        raise ContractSchemaError(f"invalid Draft 2020-12 schema: {exc}") from exc
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path))
    if errors:
        details = []
        for error in errors[:20]:
            location = "/".join(str(part) for part in error.absolute_path) or "<root>"
            details.append(f"{location}: {error.message}")
        if len(errors) > 20:
            details.append(f"... and {len(errors) - 20} more")
        raise ContractSchemaError("contract schema validation failed: " + "; ".join(details))


def _validate_invariants(data: dict[str, Any]) -> None:
    if data.get("formal_runs_allowed") is True:
        blockers = [
            item.get("binding_id")
            for item in data["unresolved_bindings"]
            if item.get("required_before_formal_run") is True
        ]
        if data.get("status") != "FORMAL_RUN_READY" or blockers:
            raise ContractInvariantError(
                "formal_runs_allowed=true requires FORMAL_RUN_READY and zero required blockers"
            )

    resolved_ids = [item.get("binding_id") for item in data.get("resolved_bindings", [])]
    if len(resolved_ids) != len(set(resolved_ids)) or any(not item for item in resolved_ids):
        raise ContractInvariantError("resolved binding IDs must be unique non-empty strings")
    unresolved_ids = [item.get("binding_id") for item in data["unresolved_bindings"]]
    if set(resolved_ids) & set(unresolved_ids):
        raise ContractInvariantError("a binding cannot be both resolved and unresolved")
    workflows = data.get("formal_workflows")
    if not isinstance(workflows, dict) or not workflows:
        raise ContractInvariantError("contract must declare formal workflows")
    for workflow, spec in workflows.items():
        required = spec.get("required_binding_ids", [])
        missing = sorted(set(required) - set(resolved_ids))
        if spec.get("formal_runs_allowed") is True and (
            spec.get("status") != "FORMAL_RUN_READY" or missing
        ):
            raise ContractInvariantError(
                f"workflow {workflow} cannot allow formal runs; unresolved bindings: {missing}"
            )

    regimes = data["training"]["regimes"]
    regime_ids = [item["regime_id"] for item in regimes]
    if len(regime_ids) != len(set(regime_ids)):
        raise ContractInvariantError("training regime IDs must be unique")
    projection_id = data["training"]["scene_input_projection"]["projection_id"]
    scene_regimes = [item for item in regimes if item.get("benchmark_scene_training") is True]
    if any(item.get("input_projection") != projection_id for item in scene_regimes):
        raise ContractInvariantError(
            "every scene training regime must bind the declared scene input projection"
        )
    common_training = data["training"]["common_27b"]
    if any(item.get("max_steps") != common_training["max_steps"] for item in regimes):
        raise ContractInvariantError("every training regime must use common fixed max_steps")
    if any(item.get("assistant_only_loss") is not True for item in regimes):
        raise ContractInvariantError("every training regime must use assistant-only loss")
    smoke = common_training["smoke_gate"]
    if smoke["regime_count"] != len(regimes):
        raise ContractInvariantError("training smoke regime count must equal declared regimes")
    if smoke["warmup_ratio"] != 0.0:
        raise ContractInvariantError("training smoke must disable warmup for real parameter updates")
    if "parameter_update" not in smoke["required_checks"]:
        raise ContractInvariantError("training smoke must require a parameter update check")
    model_key = common_training["model_key"]
    if model_key not in data["model_registry"]:
        raise ContractInvariantError("common trainer references an unknown model key")
    model_spec = data["model_registry"][model_key]
    if model_spec.get("model_class") != "Gemma3ForConditionalGeneration":
        raise ContractInvariantError("common trainer requires Gemma3ForConditionalGeneration")
    common_12b = data["training"].get("common_12b")
    if common_12b is not None:
        model_key_12b = common_12b.get("model_key")
        if model_key_12b not in data["model_registry"]:
            raise ContractInvariantError("12B trainer references an unknown model key")
        model_spec_12b = data["model_registry"][model_key_12b]
        if model_spec_12b.get("model_class") != "Gemma3ForConditionalGeneration":
            raise ContractInvariantError("12B trainer requires Gemma3ForConditionalGeneration")
        allowed_profile_differences = {"model_key", "smoke_gate"}
        profile_drift = sorted(
            key
            for key in set(common_training) | set(common_12b)
            if key not in allowed_profile_differences
            and common_training.get(key) != common_12b.get(key)
        )
        if profile_drift:
            raise ContractInvariantError(
                f"12B and 27B common trainer settings differ: {profile_drift}"
            )

    artifacts = data["artifacts"]
    profiles = [
        artifacts["required_files_common_runs"],
        artifacts["required_files_evaluation_runs"],
        artifacts["required_files_training_runs"],
    ]
    if any(len(items) != len(set(items)) for items in profiles):
        raise ContractInvariantError("artifact profiles cannot contain duplicate paths")
    if set(artifacts["required_files_evaluation_runs"]) & set(
        artifacts["required_files_training_runs"]
    ):
        raise ContractInvariantError("training and evaluation artifact profiles must be distinct")

    runs = data["formal_full_test_inference_runs"]
    run_ids = [item["run_id"] for item in runs]
    if len(run_ids) != len(set(run_ids)):
        raise ContractInvariantError("formal inference run IDs must be unique")
    if len(runs) != data["scope"]["formal_full_test_inference_run_count"]:
        raise ContractInvariantError("formal inference run count disagrees with scope")
    if any(item["expected_prediction_count"] != data["benchmark"]["counts"]["test"] for item in runs):
        raise ContractInvariantError("every full-test run must bind the frozen test count")
    unknown_model_keys = sorted(
        {item["model_key"] for item in runs} - set(data["model_registry"])
    )
    if unknown_model_keys:
        raise ContractInvariantError(
            f"inference runs reference unknown model keys: {unknown_model_keys}"
        )
    unknown_training_profiles = sorted(
        {
            item["training_profile"]
            for item in runs
            if isinstance(item.get("training_profile"), str)
        }
        - set(data["training"])
    )
    if unknown_training_profiles:
        raise ContractInvariantError(
            f"inference runs reference unknown training profiles: {unknown_training_profiles}"
        )
    unknown_workflows = sorted({item["workflow"] for item in runs} - set(workflows))
    if unknown_workflows:
        raise ContractInvariantError(f"inference runs reference unknown workflows: {unknown_workflows}")
    robustness = data.get("robustness", {})
    robustness_runs = robustness.get("runs", [])
    if not isinstance(robustness_runs, list):
        raise ContractInvariantError("robustness.runs must be a list")
    robustness_ids = [item.get("run_id") for item in robustness_runs]
    if any(not isinstance(item, str) or not item for item in robustness_ids):
        raise ContractInvariantError("robustness run IDs must be non-empty strings")
    if len(robustness_ids) != len(set(robustness_ids)):
        raise ContractInvariantError("robustness run IDs must be unique")
    if set(robustness_ids) & set(run_ids):
        raise ContractInvariantError("robustness and full-test run IDs must be disjoint")
    if robustness_runs:
        subset_size = robustness.get("subset_size")
        if any(item.get("expected_prediction_count") != subset_size for item in robustness_runs):
            raise ContractInvariantError("every robustness run must bind the frozen subset size")
        if len(robustness_runs) != robustness.get("new_run_count"):
            raise ContractInvariantError("robustness run count disagrees with new_run_count")
        total_calls = sum(int(item["expected_prediction_count"]) for item in robustness_runs)
        if total_calls != robustness.get("expected_new_inference_calls"):
            raise ContractInvariantError("robustness run matrix call count drift")
        unknown_robustness_workflows = sorted(
            {item.get("workflow") for item in robustness_runs} - set(workflows)
        )
        if unknown_robustness_workflows:
            raise ContractInvariantError(
                f"robustness runs reference unknown workflows: {unknown_robustness_workflows}"
            )
    for regime in regimes:
        if regime.get("formal_workflow") not in workflows:
            raise ContractInvariantError(
                f"training regime references unknown formal workflow: {regime['regime_id']}"
            )

    for key, value in _iter_key_values(data):
        lower_key = key.lower()
        if (lower_key == "path" or lower_key.endswith("_path")) and isinstance(value, str):
            if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
                raise ContractInvariantError(f"absolute contract path is forbidden: {key}={value}")
        if lower_key == "ledger_consumers" and (not isinstance(value, list) or not value):
            raise ContractInvariantError("every formal consumer list must be non-empty")


def load_contract(
    path: str | Path,
    *,
    schema_path: str | Path | None = None,
    require_formal_ready: bool = False,
) -> LoadedContract:
    contract_path = Path(path).resolve()
    data = read_json(contract_path)
    if not isinstance(data, dict):
        raise ContractError("experiment contract root must be an object")

    if schema_path is None:
        reference = data.get("$schema")
        if not isinstance(reference, str) or not reference:
            raise ContractError("contract requires a relative $schema path")
        candidate = Path(reference)
        if candidate.is_absolute() or PureWindowsPath(reference).is_absolute():
            raise ContractError("contract $schema path must be relative")
        resolved_schema = (contract_path.parent / candidate).resolve()
    else:
        resolved_schema = Path(schema_path).resolve()
    schema = read_json(resolved_schema)
    if not isinstance(schema, dict):
        raise ContractError("contract schema root must be an object")

    _validate_schema(data, schema)
    _validate_invariants(data)
    loaded = LoadedContract(
        data=data,
        path=contract_path,
        schema_path=resolved_schema,
        sha256=sha256_file(contract_path),
    )
    if require_formal_ready:
        loaded.require_formal_ready()
    return loaded


def verify_source_hashes(contract: LoadedContract, workspace_root: str | Path) -> dict[str, str]:
    root = Path(workspace_root).resolve()
    verified: dict[str, str] = {}
    source = contract.data["source_of_truth"]
    for key, relative_path in source.items():
        if not key.endswith("_path"):
            continue
        hash_key = f"{key[:-5]}_sha256"
        expected = source.get(hash_key)
        if not isinstance(expected, str):
            continue
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ContractInvariantError(f"source path escapes workspace: {relative_path}") from exc
        verified[key] = verify_file_sha256(target, expected)
    return verified
