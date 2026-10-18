"""Leaf types for the hands package (imports nothing from riya)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Call:
    """One requested tool call: a tool name and its (unvalidated) args."""
    tool: str
    args: dict = field(default_factory=dict)
