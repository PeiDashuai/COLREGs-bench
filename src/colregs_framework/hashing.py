"""Stable SHA-256 helpers used by every revision artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SHA256_HEX_LENGTH = 64


class HashMismatchError(ValueError):
    """Raised when an artifact does not match its locked SHA-256."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_ordered_ids(ids: Iterable[str]) -> str:
    ordered = list(ids)
    if not all(isinstance(item, str) and item for item in ordered):
        raise ValueError("ordered IDs must be non-empty strings")
    return sha256_json(ordered)


def is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_HEX_LENGTH:
        return False
    return all(character in "0123456789abcdef" for character in value)


def verify_file_sha256(path: str | Path, expected: str) -> str:
    if not is_sha256(expected):
        raise ValueError(f"invalid expected SHA-256: {expected!r}")
    actual = sha256_file(path)
    if actual != expected:
        raise HashMismatchError(
            f"SHA-256 mismatch for {Path(path)}: expected {expected}, got {actual}"
        )
    return actual

