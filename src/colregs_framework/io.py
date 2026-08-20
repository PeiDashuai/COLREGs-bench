"""Strict JSON/JSONL persistence with append-only resume support."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


class JsonFormatError(ValueError):
    """Raised for malformed JSON, duplicate object keys, or a non-object row."""


class DuplicateRecordIdError(ValueError):
    """Raised when an append-only JSONL journal repeats its primary key."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise JsonFormatError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: str, *, source: str = "<string>") -> Any:
    try:
        return json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except JsonFormatError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise JsonFormatError(f"invalid JSON in {source}: {exc}") from exc


def read_json(path: str | Path) -> Any:
    source = Path(path)
    return strict_json_loads(source.read_text(encoding="utf-8"), source=str(source))


def write_json_atomic(
    path: str | Path,
    value: Any,
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")

    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise JsonFormatError(f"blank JSONL row at {source}:{line_number}")
            value = strict_json_loads(line, source=f"{source}:{line_number}")
            if not isinstance(value, dict):
                raise JsonFormatError(
                    f"JSONL row must be an object at {source}:{line_number}"
                )
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("JSONL record must be a mapping")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(record),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_jsonl_atomic(
    path: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a complete JSONL artifact atomically, rejecting partial overwrites."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                if not isinstance(record, Mapping):
                    raise TypeError("JSONL record must be a mapping")
                handle.write(
                    json.dumps(
                        dict(record),
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


@dataclass(frozen=True)
class JsonlIdAudit:
    row_count: int
    unique_count: int
    observed_ids: tuple[str, ...]
    duplicate_ids: tuple[str, ...]
    missing_ids: tuple[str, ...]
    extra_ids: tuple[str, ...]

    @property
    def is_exact(self) -> bool:
        return not self.duplicate_ids and not self.missing_ids and not self.extra_ids


def audit_jsonl_ids(
    path: str | Path,
    expected_ids: Iterable[str] | None = None,
    *,
    id_field: str = "sample_id",
) -> JsonlIdAudit:
    observed: list[str] = []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row_number, record in enumerate(iter_jsonl(path), start=1):
        record_id = record.get(id_field)
        if not isinstance(record_id, str) or not record_id:
            raise JsonFormatError(
                f"missing non-empty {id_field!r} at {Path(path)} row {row_number}"
            )
        observed.append(record_id)
        if record_id in seen:
            duplicates.add(record_id)
        seen.add(record_id)

    expected = tuple(expected_ids or ())
    if len(set(expected)) != len(expected):
        raise ValueError("expected IDs contain duplicates")
    expected_set = set(expected)
    return JsonlIdAudit(
        row_count=len(observed),
        unique_count=len(seen),
        observed_ids=tuple(observed),
        duplicate_ids=tuple(sorted(duplicates)),
        missing_ids=tuple(item for item in expected if item not in seen),
        extra_ids=tuple(sorted(seen - expected_set)) if expected_ids is not None else (),
    )


class JsonlJournal:
    """Append-only JSONL writer that reconstructs its resume index on open."""

    def __init__(self, path: str | Path, *, id_field: str = "sample_id") -> None:
        self.path = Path(path)
        self.id_field = id_field
        self._seen: set[str] = set()
        if self.path.exists():
            audit = audit_jsonl_ids(self.path, id_field=id_field)
            if audit.duplicate_ids:
                raise DuplicateRecordIdError(
                    f"cannot resume {self.path}; duplicate IDs: {audit.duplicate_ids}"
                )
            self._seen.update(audit.observed_ids)

    @property
    def seen_ids(self) -> frozenset[str]:
        return frozenset(self._seen)

    def append(self, record: Mapping[str, Any]) -> None:
        record_id = record.get(self.id_field)
        if not isinstance(record_id, str) or not record_id:
            raise JsonFormatError(f"record requires non-empty {self.id_field!r}")
        if record_id in self._seen:
            raise DuplicateRecordIdError(
                f"duplicate {self.id_field} {record_id!r} in {self.path}"
            )
        append_jsonl(self.path, record)
        self._seen.add(record_id)
