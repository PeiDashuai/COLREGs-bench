from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from .actions import ActionsSpec, validate_no_cross_head


class Modality(str, Enum):
    OBLIGATORY = "OBLIGATORY"
    FORBIDDEN = "FORBIDDEN"
    PERMITTED = "PERMITTED"


class Role(str, Enum):
    GIVE_WAY = "GIVE_WAY"
    STAND_ON = "STAND_ON"
    BOTH = "BOTH"
    ANY = "ANY"


class Strength(str, Enum):
    hard = "hard"
    soft = "soft"


class DeonticSpec(BaseModel):
    subject: str = Field(..., description="ownship|target|both|any")
    role: Role
    modality: Modality
    strength: Strength = Strength.hard


class RuleSpec(BaseModel):
    id: str
    article: str
    summary: str
    applicability: Dict[str, Any]
    deontic: DeonticSpec
    actions: ActionsSpec = Field(default_factory=ActionsSpec)
    evidence_requirements: list[str] = Field(default_factory=list)
    text_ref: str = ""

    def model_post_init(self, __context: Any) -> None:  # type: ignore[override]
        # Ensure signals are not mixed into maneuver head.
        validate_no_cross_head(self.actions)


def _is_signal_primitive(p: str) -> bool:
    return p.startswith("LIGHT_") or p.startswith("SHAPE_") or p.startswith("SOUND_")


def _upgrade_actions_v1_to_v2(actions_raw: dict[str, Any]) -> dict[str, Any]:
    """Backward-compat: transform legacy actions {allowed,forbidden} into {maneuver,signals} heads."""
    if "maneuver" in actions_raw or "signals" in actions_raw:
        return actions_raw

    allowed = actions_raw.get("allowed", []) or []
    forbidden = actions_raw.get("forbidden", []) or []

    man_allowed, man_forbidden = [], []
    ls_req, ls_forb, snd_req, snd_forb = [], [], [], []

    for it in allowed:
        prim = str(it.get("primitive", ""))
        if prim.startswith("SOUND_"):
            snd_req.append(it)
        elif prim.startswith("LIGHT_") or prim.startswith("SHAPE_"):
            ls_req.append(it)
        else:
            man_allowed.append(it)

    for it in forbidden:
        prim = str(it.get("primitive", ""))
        if prim.startswith("SOUND_"):
            snd_forb.append(it)
        elif prim.startswith("LIGHT_") or prim.startswith("SHAPE_"):
            ls_forb.append(it)
        else:
            man_forbidden.append(it)

    return {
        "maneuver": {"allowed": man_allowed, "forbidden": man_forbidden},
        "signals": {
            "lights_shapes_required": ls_req,
            "lights_shapes_forbidden": ls_forb,
            "sounds_required": snd_req,
            "sounds_forbidden": snd_forb,
        },
    }


def load_rules(path: str | Path) -> dict[str, RuleSpec]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Rule library YAML must be a list of rules")

    rules: dict[str, RuleSpec] = {}
    for raw in data:
        if not isinstance(raw, dict):
            raise ValueError("Each rule must be a mapping/dict")
        raw = dict(raw)
        raw["actions"] = _upgrade_actions_v1_to_v2(raw.get("actions", {}) or {})
        rule = RuleSpec.model_validate(raw)
        if rule.id in rules:
            raise ValueError(f"Duplicate rule id: {rule.id}")
        rules[rule.id] = rule
    return rules
