from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tri(Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def __bool__(self):  # pragma: no cover
        raise TypeError("TriBool cannot be coerced to bool")


@dataclass(frozen=True)
class TriValue:
    value: Tri
    missing_fields: tuple[str, ...] = ()

    @staticmethod
    def t() -> "TriValue":
        return TriValue(Tri.TRUE)

    @staticmethod
    def f() -> "TriValue":
        return TriValue(Tri.FALSE)

    @staticmethod
    def u(*missing: str) -> "TriValue":
        # Deduplicate while preserving order
        seen = set()
        ordered = []
        for m in missing:
            if m and m not in seen:
                seen.add(m)
                ordered.append(m)
        return TriValue(Tri.UNKNOWN, tuple(ordered))


def tri_and(a: TriValue, b: TriValue) -> TriValue:
    if a.value == Tri.FALSE or b.value == Tri.FALSE:
        return TriValue.f()
    if a.value == Tri.TRUE and b.value == Tri.TRUE:
        return TriValue.t()
    # One or both unknown, none false
    return TriValue.u(*(a.missing_fields + b.missing_fields))


def tri_or(a: TriValue, b: TriValue) -> TriValue:
    if a.value == Tri.TRUE or b.value == Tri.TRUE:
        return TriValue.t()
    if a.value == Tri.FALSE and b.value == Tri.FALSE:
        return TriValue.f()
    # One or both unknown, none true
    return TriValue.u(*(a.missing_fields + b.missing_fields))


def tri_not(a: TriValue) -> TriValue:
    if a.value == Tri.TRUE:
        return TriValue.f()
    if a.value == Tri.FALSE:
        return TriValue.t()
    return a
