from __future__ import annotations

"""Predicate DSL evaluation.

This module intentionally stays lightweight and returns TriValue (True/False/Unknown)
with explicit missing_fields for Unknown outcomes.
"""

from typing import Any, Tuple

from .tri import TriValue, tri_and, tri_not, tri_or


def get_field(features: Any, path: str) -> Tuple[bool, Any]:
    """Return (present, value). Works with pydantic models and dicts."""
    cur = features
    for part in path.split("."):
        if cur is None:
            return False, None
        if isinstance(cur, dict):
            if part not in cur:
                return False, None
            cur = cur[part]
        else:
            if not hasattr(cur, part):
                return False, None
            cur = getattr(cur, part)
    return True, cur


def _tri_from_bool(b: bool) -> TriValue:
    return TriValue.t() if b else TriValue.f()


def _missing(path: str) -> TriValue:
    return TriValue.u(path)


def eval_predicate(expr: Any, features: Any) -> TriValue:
    """Evaluate a predicate AST against Features.

    Supported ops:
      - and/or/not
      - eq/ne
      - in
      - lt/le/gt/ge (numeric or comparable)
      - between: [field, lo, hi] inclusive
      - exists: [field] -> True if present and not None
    """
    if expr is None:
        return TriValue.u("<expr>")

    if isinstance(expr, dict):
        if "and" in expr:
            items = [eval_predicate(x, features) for x in expr["and"]]
            if not items:
                return TriValue.u("<and>")
            out = items[0]
            for t in items[1:]:
                out = tri_and(out, t)
            return out

        if "or" in expr:
            items = [eval_predicate(x, features) for x in expr["or"]]
            if not items:
                return TriValue.u("<or>")
            out = items[0]
            for t in items[1:]:
                out = tri_or(out, t)
            return out

        if "not" in expr:
            return tri_not(eval_predicate(expr["not"], features))

        for op in ("eq", "ne", "in", "lt", "le", "gt", "ge", "between", "exists"):
            if op not in expr:
                continue
            args = expr[op]

            if op == "exists":
                field = args[0]
                present, val = get_field(features, field)
                if not present:
                    return _missing(field)
                return _tri_from_bool(val is not None)

            field = args[0]
            present, val = get_field(features, field)
            if not present or val is None:
                return _missing(field)

            if op == "eq":
                return _tri_from_bool(val == args[1])
            if op == "ne":
                return _tri_from_bool(val != args[1])
            if op == "in":
                return _tri_from_bool(val in args[1])
            if op == "between":
                lo, hi = args[1], args[2]
                try:
                    return _tri_from_bool(lo <= val <= hi)
                except TypeError:
                    return TriValue.u(field)

            rhs = args[1]
            try:
                if op == "lt":
                    return _tri_from_bool(val < rhs)
                if op == "le":
                    return _tri_from_bool(val <= rhs)
                if op == "gt":
                    return _tri_from_bool(val > rhs)
                if op == "ge":
                    return _tri_from_bool(val >= rhs)
            except TypeError:
                return TriValue.u(field)

    if isinstance(expr, bool):
        return _tri_from_bool(expr)

    return TriValue.u("<unsupported>")
