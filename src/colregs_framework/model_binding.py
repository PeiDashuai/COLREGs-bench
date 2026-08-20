"""Build and verify immutable local model-file manifests."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from .hashing import sha256_file, sha256_json
from .io import read_json, write_json_atomic


class ModelBindingError(ValueError):
    """Raised when a model directory does not match its immutable manifest."""


def _identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest["schema_version"],
        "model_id": manifest["model_id"],
        "exact_revision": manifest["exact_revision"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "files": manifest["files"],
    }


def _model_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ModelBindingError(f"model manifest forbids symlinks: {path}")
        if path.is_file():
            files.append(path)
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def build_model_manifest(
    model_root: str | Path,
    output_path: str | Path,
    *,
    model_id: str,
    exact_revision: str,
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    output = Path(output_path).resolve()
    if not root.is_dir():
        raise ModelBindingError(f"model root is not a directory: {root}")
    if not model_id.strip() or not exact_revision.strip():
        raise ModelBindingError("model_id and exact_revision must be non-empty")
    try:
        output.relative_to(root)
    except ValueError:
        pass
    else:
        raise ModelBindingError("model manifest must be written outside the model root")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _model_files(root)
    ]
    if not files:
        raise ModelBindingError("model root contains no files")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "model_id": model_id.strip(),
        "exact_revision": exact_revision.strip(),
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }
    manifest["content_identity_sha256"] = sha256_json(_identity(manifest))
    write_json_atomic(output, manifest)
    return manifest


def verify_model_manifest(
    model_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_revision: str | None = None,
    expected_content_identity: str | None = None,
) -> dict[str, Any]:
    root = Path(model_root).resolve()
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ModelBindingError("unsupported model manifest")
    if expected_model_id is not None and manifest.get("model_id") != expected_model_id:
        raise ModelBindingError("model ID does not match the contract")
    if expected_revision is not None and manifest.get("exact_revision") != expected_revision:
        raise ModelBindingError("model revision does not match the contract")
    if sha256_json(_identity(manifest)) != manifest.get("content_identity_sha256"):
        raise ModelBindingError("model manifest content identity mismatch")
    if (
        expected_content_identity is not None
        and manifest.get("content_identity_sha256") != expected_content_identity
    ):
        raise ModelBindingError("model content identity does not match the contract")
    declared_paths: set[str] = set()
    total_bytes = 0
    for item in manifest.get("files", []):
        relative = str(item.get("path", ""))
        candidate = Path(relative)
        if candidate.is_absolute() or PureWindowsPath(relative).is_absolute():
            raise ModelBindingError(f"absolute model file path is forbidden: {relative}")
        path = (root / candidate).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ModelBindingError(f"model file escapes root: {relative}") from exc
        if relative in declared_paths:
            raise ModelBindingError(f"duplicate model file entry: {relative}")
        declared_paths.add(relative)
        if not path.is_file():
            raise ModelBindingError(f"model file is missing: {relative}")
        if path.stat().st_size != item.get("bytes"):
            raise ModelBindingError(f"model file size drift: {relative}")
        if sha256_file(path) != item.get("sha256"):
            raise ModelBindingError(f"model file hash drift: {relative}")
        total_bytes += path.stat().st_size
    actual_paths = {
        path.relative_to(root).as_posix() for path in _model_files(root)
    }
    if actual_paths != declared_paths:
        missing = sorted(declared_paths - actual_paths)[:5]
        extra = sorted(actual_paths - declared_paths)[:5]
        raise ModelBindingError(f"model file set drift: missing={missing}, extra={extra}")
    if len(declared_paths) != manifest.get("file_count"):
        raise ModelBindingError("model file count mismatch")
    if total_bytes != manifest.get("total_bytes"):
        raise ModelBindingError("model total byte count mismatch")
    return manifest
