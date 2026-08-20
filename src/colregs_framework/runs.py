"""Safe, append-only run-directory initialization and status journaling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .contracts import LoadedContract
from .environment import write_environment
from .hashing import sha256_ordered_ids
from .io import JsonlJournal, append_jsonl, read_json, write_json_atomic


RUN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{2,127}$")


class UnsafeRunPathError(ValueError):
    """Raised when a run ID could escape or alias the configured run root."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_run_path(run_root: str | Path, run_id: str) -> Path:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise UnsafeRunPathError(
            "run_id must match ^[a-z][a-z0-9_]{2,127}$; path separators are forbidden"
        )
    root = Path(run_root).resolve()
    target = (root / run_id).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise UnsafeRunPathError(f"run path escapes configured root: {target}") from exc
    if target == root:
        raise UnsafeRunPathError("run path cannot equal run root")
    return target


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    run_id: str
    contract: LoadedContract

    @property
    def path(self) -> Path:
        return resolve_run_path(self.root, self.run_id)

    @classmethod
    def create(
        cls,
        run_root: str | Path,
        run_id: str,
        contract: LoadedContract,
        *,
        expected_ids: list[str] | tuple[str, ...],
        formal_intent: bool = False,
        input_metadata: Mapping[str, Any] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> "RunDirectory":
        if formal_intent:
            contract.require_formal_ready(contract.formal_workflow_for_run(run_id))
            inference_specs = {item["run_id"]: item for item in contract.inference_run_specs}
            training_ids = {
                item["regime_id"] for item in contract.data["training"]["regimes"]
            }
            if run_id not in inference_specs and run_id not in training_ids:
                raise ValueError(f"formal run_id is not declared by the contract: {run_id}")
            if (
                run_id in inference_specs
                and len(expected_ids) != inference_specs[run_id]["expected_prediction_count"]
            ):
                raise ValueError(
                    f"formal inference run {run_id} requires "
                    f"{inference_specs[run_id]['expected_prediction_count']} expected IDs"
                )
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected IDs contain duplicates")
        target = resolve_run_path(run_root, run_id)
        if target.exists():
            raise FileExistsError(f"run directory already exists: {target}")
        target.mkdir(parents=True, exist_ok=False)
        (target / "logs").mkdir()

        input_manifest = {
            "contract_id": contract.data["contract_id"],
            "contract_version": contract.data["contract_version"],
            "contract_sha256": contract.sha256,
            "run_id": run_id,
            "formal_intent": formal_intent,
            "expected_sample_count": len(expected_ids),
            "ordered_sample_id_sha256": sha256_ordered_ids(expected_ids),
            "created_at_utc": utc_now(),
            "metadata": dict(input_metadata or {}),
        }
        write_json_atomic(target / "input_manifest.json", input_manifest)

        path_policy = contract.data["path_policy"]
        include_names = [
            path_policy.get("benchmark_root_env"),
            path_policy.get("run_root_env"),
            path_policy.get("gemma_model_root_env"),
            path_policy.get("api_key_env"),
        ]
        write_environment(
            target / "environment.json",
            environ=environ,
            include_names=[name for name in include_names if isinstance(name, str)],
        )
        append_jsonl(
            target / "status_events.jsonl",
            {
                "event_id": f"initialized_{utc_now()}",
                "run_id": run_id,
                "state": "INITIALIZED",
                "formal_result": False,
                "recorded_at_utc": utc_now(),
            },
        )
        return cls(Path(run_root).resolve(), run_id, contract)

    @classmethod
    def open_existing(
        cls,
        run_root: str | Path,
        run_id: str,
        contract: LoadedContract,
        *,
        expected_ids: list[str] | tuple[str, ...],
        formal_intent: bool,
    ) -> "RunDirectory":
        if formal_intent:
            contract.require_formal_ready(contract.formal_workflow_for_run(run_id))
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected IDs contain duplicates")
        run = cls(Path(run_root).resolve(), run_id, contract)
        if not run.path.is_dir():
            raise FileNotFoundError(f"run directory does not exist: {run.path}")
        if (run.path / "run_manifest.json").exists() or (run.path / "completion_report.json").exists():
            raise FileExistsError("completed or finalized run cannot be resumed")
        manifest = read_json(run.path / "input_manifest.json")
        expected_hash = sha256_ordered_ids(expected_ids)
        checks = {
            "contract_id": manifest.get("contract_id") == contract.data["contract_id"],
            "contract_version": manifest.get("contract_version") == contract.data["contract_version"],
            "contract_sha256": manifest.get("contract_sha256") == contract.sha256,
            "run_id": manifest.get("run_id") == run_id,
            "formal_intent": manifest.get("formal_intent") is formal_intent,
            "expected_sample_count": manifest.get("expected_sample_count") == len(expected_ids),
            "ordered_sample_id_sha256": manifest.get("ordered_sample_id_sha256") == expected_hash,
        }
        failed = sorted(key for key, passed in checks.items() if not passed)
        if failed:
            raise ValueError(f"existing run input binding drift: {failed}")
        run.record_status("RESUME_OPENED", expected_sample_count=len(expected_ids))
        return run

    def artifact(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute():
            raise UnsafeRunPathError("artifact path must be relative")
        target = (self.path / relative).resolve()
        try:
            target.relative_to(self.path)
        except ValueError as exc:
            raise UnsafeRunPathError(f"artifact path escapes run directory: {relative}") from exc
        return target

    def record_status(self, state: str, **details: Any) -> None:
        append_jsonl(
            self.artifact("status_events.jsonl"),
            {
                "event_id": f"{state.lower()}_{utc_now()}",
                "run_id": self.run_id,
                "state": state,
                "formal_result": False,
                "recorded_at_utc": utc_now(),
                "details": details,
            },
        )

    def journal(self, relative_path: str | Path, *, id_field: str = "sample_id") -> JsonlJournal:
        return JsonlJournal(self.artifact(relative_path), id_field=id_field)
