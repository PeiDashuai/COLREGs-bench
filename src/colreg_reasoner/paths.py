from __future__ import annotations

"""Path helpers for locating Part-1 assets (rules, params) reliably.

Part-2 (colreg_reasoner) must not assume a monorepo root layout.
It locates Part-1 (colreg_kernel) assets via the installed package path.
"""

from pathlib import Path
from importlib.resources import files as pkg_files


def default_rules_path() -> str:
    """Absolute path to Part-1 atomic rules YAML."""
    # Prefer packaged data if Part-1 ships it (future-proof)
    try:
        p = pkg_files("colreg_kernel").joinpath("data/colregs_atomic.yaml")
        if p.is_file():
            return str(p)
    except Exception:
        pass

    # Fallback: editable install layout
    import colreg_kernel

    pkg_file = Path(colreg_kernel.__file__).resolve()
    # .../colreg_kernel/src/colreg_kernel/__init__.py -> project root is parents[2]
    proj_root = pkg_file.parents[2]
    return str(proj_root / "rules" / "colregs_atomic.yaml")
