"""Validated evidence selection and hashing for reproducible runs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from .hashing import sha256_file
from .io import read_json, write_json_atomic, write_jsonl_atomic


TRAINING_RUNS = (
    "structured_core_three_ft",
    "final_only_ft",
    "colregs_text_ft",
    "structured_multimodal_ft",
)
INFERENCE_RUNS = (
    "gemma27b_no_ft_text_core3",
    "gemma27b_structured_core3_ft_text_core3",
    "gemma27b_final_only_ft_text_core3",
    "gemma27b_colregs_text_ft_text_core3",
    "gemma27b_no_ft_clean_multimodal_core3",
    "gemma27b_structured_mm_ft_clean_multimodal_core3",
)
PROBED_RUNS = ("structured_core_three_ft", "structured_multimodal_ft")


def _require_pass(path: Path, *, contract_sha256: str, kind: str) -> None:
    payload = read_json(path)
    if payload.get("contract_sha256") != contract_sha256:
        raise ValueError(f"{kind} contract mismatch: {path}")
    if kind == "joint report" and payload.get("status") != "PASS":
        raise ValueError(f"joint report is not PASS: {path}")
    if kind == "evidence index" and payload.get("formal_result") is not True:
        raise ValueError(f"evidence index is not formal: {path}")


def _run_files(
    run_root: Path,
    run_id: str,
    *,
    training: bool,
    contract_sha256: str,
) -> list[Path]:
    run_dir = run_root / run_id
    if not run_dir.is_dir():
        raise ValueError(f"missing run directory: {run_id}")
    manifest = read_json(run_dir / "run_manifest.json")
    completion = read_json(run_dir / "completion_report.json")
    if manifest.get("contract_sha256") != contract_sha256:
        raise ValueError(f"run contract mismatch: {run_id}")
    if manifest.get("formal_result") is not True or manifest.get("completion_status") != "PASS":
        raise ValueError(f"run is not formal and complete: {run_id}")
    if completion.get("status") != "PASS":
        raise ValueError(f"completion report is not PASS: {run_id}")
    bound_artifact = (
        run_dir / "adapter" / "adapter_model.safetensors"
        if training
        else run_dir / "predictions.jsonl"
    )
    bound_hash = (
        manifest.get("adapter_model_sha256")
        if training
        else manifest.get("prediction_sha256")
    )
    if not bound_artifact.is_file() or sha256_file(bound_artifact) != bound_hash:
        raise ValueError(f"run artifact hash mismatch: {run_id}")
    files = []
    for path in run_dir.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if training and relative.parts[:1] == ("training",):
            if len(relative.parts) > 1 and relative.parts[1].startswith("checkpoint-"):
                continue
        files.append(path)
    return files


def write_phase56_tier_a_manifest(
    *,
    contract_sha256: str,
    run_root: Path,
    suite_dir: Path,
    output_dir: Path,
    server_logs: Iterable[Path] = (),
) -> dict:
    """Validate formal closure, then hash the exact Tier-A transfer set."""

    run_root = run_root.resolve()
    suite_dir = suite_dir.resolve()
    output_dir = output_dir.resolve()
    _require_pass(
        suite_dir / "phase5_6_joint_report.json",
        contract_sha256=contract_sha256,
        kind="joint report",
    )
    _require_pass(
        suite_dir / "phase5_6_evidence_index.json",
        contract_sha256=contract_sha256,
        kind="evidence index",
    )

    selected = [path for path in suite_dir.rglob("*") if path.is_file()]
    for run_id in TRAINING_RUNS:
        selected.extend(
            _run_files(
                run_root,
                run_id,
                training=True,
                contract_sha256=contract_sha256,
            )
        )
    for run_id in INFERENCE_RUNS:
        selected.extend(
            _run_files(
                run_root,
                run_id,
                training=False,
                contract_sha256=contract_sha256,
            )
        )
    for run_id in PROBED_RUNS:
        probe = run_root / run_id / "generation_probe_report.json"
        report = read_json(probe)
        if report.get("status") != "PASS" or report.get("contract_sha256") != contract_sha256:
            raise ValueError(f"generation probe is not PASS: {run_id}")

    events = run_root / "phase5_phase6_orchestrator_events.jsonl"
    if not events.is_file():
        raise ValueError(f"missing orchestrator journal: {events}")
    selected.append(events)
    for log in server_logs:
        resolved = log.resolve()
        if not resolved.is_file():
            raise ValueError(f"missing server log: {resolved}")
        selected.append(resolved)

    selected = sorted(set(selected))
    if any(path.is_symlink() for path in selected):
        raise ValueError("Tier-A evidence may not contain symbolic links")
    common_root = Path(os.path.commonpath([str(path) for path in selected]))
    if common_root.is_file():
        common_root = common_root.parent
    records = [
        {
            "path": path.relative_to(common_root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in selected
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "tier_a_sha256_manifest.jsonl"
    write_jsonl_atomic(manifest_path, records)
    file_list = output_dir / "tier_a_tar_file_list.txt"
    file_list.write_text(
        "".join(f"{record['path']}\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    summary = {
        "schema_version": 1,
        "contract_sha256": contract_sha256,
        "source_root": str(common_root),
        "file_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "manifest_sha256": sha256_file(manifest_path),
        "file_list_sha256": sha256_file(file_list),
    }
    write_json_atomic(output_dir / "tier_a_manifest_summary.json", summary)
    return summary
