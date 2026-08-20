"""Environment capture that never persists credential values."""

from __future__ import annotations

import os
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping

from .io import write_json_atomic


_SECRET_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|KEY|TOKEN|SECRET|PASSWORD|PASSWD|PWD|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE_?KEY|PAT|CONNECTION_?STRING)(?:_|$)",
    re.IGNORECASE,
)


def is_secret_name(name: str) -> bool:
    return bool(_SECRET_NAME.search(name))


def capture_environment(
    *,
    environ: Mapping[str, str] | None = None,
    include_names: Iterable[str] = (),
) -> dict[str, object]:
    source = os.environ if environ is None else environ
    safe_values: dict[str, str] = {}
    redacted_names: list[str] = []
    for name in sorted(set(include_names)):
        if name not in source:
            continue
        if is_secret_name(name):
            redacted_names.append(name)
        else:
            safe_values[name] = source[name]
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "environment_variables": safe_values,
        "redacted_environment_keys": redacted_names,
    }


def write_environment(
    path: str | Path,
    *,
    environ: Mapping[str, str] | None = None,
    include_names: Iterable[str] = (),
) -> Path:
    return write_json_atomic(
        path,
        capture_environment(environ=environ, include_names=include_names),
        overwrite=False,
    )
