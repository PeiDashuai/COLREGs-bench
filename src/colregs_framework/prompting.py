"""Pure shared prompt construction for training and evaluation."""

from __future__ import annotations

import json
from typing import Any, Mapping


def canonical_prompt_text(value: str) -> str:
    """Ignore only serialization newlines at the end of a prompt asset."""

    return value.rstrip("\r\n")


def build_core_three_user_prompt(
    payload: Mapping[str, Any], vocabulary_fields: Mapping[str, Any]
) -> str:
    return (
        "Scene input:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n\nClosed vocabulary:\n"
        + json.dumps(
            vocabulary_fields,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n\nReturn only the contracted JSON object."
    )


def build_fullfields_user_prompt(
    payload: Mapping[str, Any], vocabulary: Mapping[str, Any]
) -> str:
    fields = vocabulary.get("fields")
    nested = vocabulary.get("diagnostic_nested_string_vocabulary")
    if not isinstance(fields, Mapping) or not isinstance(nested, Mapping):
        raise ValueError("full-fields vocabulary lacks top-level or nested sections")
    closed_vocabulary = {
        "top_level_fields": fields,
        "nested_string_values": nested,
    }
    return (
        "Scene input:\n"
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n\nClosed vocabulary for all six reported heads:\n"
        + json.dumps(
            closed_vocabulary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n\nUse the response schema branch matching the scene family. "
        + "Return target summaries in scene target order and use target IDs verbatim. "
        + "Return only the contracted JSON object."
    )
