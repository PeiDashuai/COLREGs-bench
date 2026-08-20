"""Static independence audit and immutable source evidence for symbolic runs."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any, Iterable

from .hashing import sha256_file, sha256_json
from .io import write_json_atomic


FORBIDDEN_MODULE_PREFIXES = (
    "colreg_scenegen.labeler",
    "colreg_scenegen.feature_builder",
    "label_native_candidates",
    "generate_native_candidates",
    "family_aware_mapper",
    "oracle_builder",
    "make_oracle_pred",
)
AUDITED_PACKAGE_PREFIXES = ("colregs_framework", "colreg_kernel", "colreg_reasoner")


class SymbolicIndependenceError(ValueError):
    """Raised when the symbolic runtime reaches a forbidden benchmark component."""


def _runtime_imports(
    path: Path, *, module_name: str, is_package: bool
) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []

    def visit(nodes: Iterable[ast.stmt]) -> None:
        for node in nodes:
            if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                visit(node.orelse)
                continue
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    package_parts = module_name.split(".") if is_package else module_name.split(".")[:-1]
                    trim = node.level - 1
                    if trim:
                        package_parts = package_parts[:-trim]
                    if node.module:
                        imports.append(".".join([*package_parts, node.module]))
                    else:
                        imports.extend(
                            ".".join([*package_parts, alias.name]) for alias in node.names
                        )
                elif node.module:
                    imports.append(node.module)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.stmt) and child not in getattr(node, "body", ()) and child not in getattr(node, "orelse", ()):
                    visit([child])
            if hasattr(node, "body") and not isinstance(node, ast.If):
                visit(getattr(node, "body", ()))
            if hasattr(node, "orelse") and not isinstance(node, ast.If):
                visit(getattr(node, "orelse", ()))

    visit(tree.body)
    return tuple(sorted(set(imports)))


def audit_runtime_imports(entry_modules: Iterable[str]) -> dict[str, Any]:
    queue = list(entry_modules)
    visited: dict[str, dict[str, Any]] = {}
    violations: list[dict[str, str]] = []
    while queue:
        module_name = queue.pop(0)
        if module_name in visited:
            continue
        spec = importlib.util.find_spec(module_name)
        if spec is None or not spec.origin or not spec.origin.endswith(".py"):
            visited[module_name] = {"path": spec.origin if spec else None, "imports": []}
            continue
        path = Path(spec.origin).resolve()
        imports = _runtime_imports(
            path,
            module_name=module_name,
            is_package=bool(spec.submodule_search_locations),
        )
        visited[module_name] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "imports": list(imports),
        }
        for imported in imports:
            if any(
                imported == forbidden or imported.startswith(forbidden + ".")
                for forbidden in FORBIDDEN_MODULE_PREFIXES
            ):
                violations.append({"importer": module_name, "forbidden_import": imported})
            if imported.startswith(AUDITED_PACKAGE_PREFIXES) and imported not in visited:
                queue.append(imported)
    result = {
        "schema_version": 1,
        "entry_modules": list(entry_modules),
        "audited_modules": dict(sorted(visited.items())),
        "forbidden_module_prefixes": list(FORBIDDEN_MODULE_PREFIXES),
        "violations": violations,
        "passed": not violations,
    }
    result["audit_sha256"] = sha256_json(result)
    if violations:
        raise SymbolicIndependenceError(f"forbidden runtime imports: {violations}")
    return result


def build_source_manifest(
    paths: Iterable[str | Path],
    *,
    workspace_root: str | Path,
    output_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(workspace_root).resolve()
    files: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise SymbolicIndependenceError(f"source escapes workspace: {path}") from exc
        files.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    result = {"schema_version": 1, "files": sorted(files, key=lambda item: item["path"])}
    result["manifest_payload_sha256"] = sha256_json(result)
    if output_path is not None:
        write_json_atomic(output_path, result, overwrite=overwrite)
    return result


def verify_source_manifest(manifest: dict[str, Any], workspace_root: str | Path) -> None:
    root = Path(workspace_root).resolve()
    mismatches: list[str] = []
    for item in manifest.get("files", []):
        path = (root / item["path"]).resolve()
        if not path.exists() or sha256_file(path) != item["sha256"]:
            mismatches.append(item["path"])
    if mismatches:
        raise SymbolicIndependenceError(f"source manifest drift: {mismatches}")
