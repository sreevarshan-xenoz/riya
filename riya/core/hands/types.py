"""Leaf types for the hands package (imports nothing from riya)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class Call:
    """One requested tool call: a tool name and its (unvalidated) args."""
    tool: str
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ParamSpec:
    """How one tool argument is validated: a type, a max length (str) and bounds or an enum."""
    type: Literal["str", "int", "float", "duration"]
    required: bool = True
    max_len: int = 200
    min: float | None = None
    max: float | None = None
    enum: tuple | None = None


RESULT_MAX = 200


@dataclass
class ToolResult:
    """What a tool run reports: ok, a short line (cut to 200 chars) and optional data."""
    ok: bool
    text: str
    data: dict | None = None

    def __post_init__(self):
        self.text = self.text[:RESULT_MAX]
