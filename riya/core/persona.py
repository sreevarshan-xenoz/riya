"""Persona layers: the short local persona, the full base sheet plus companion layer, the made-up style examples, and
the versioned trait values."""
import json
from pathlib import Path
from typing import Literal

from jarvis.voice.persona import persona_text

TRAITS_PATH = Path(__file__).with_name("traits.json")
COMPANION_PATH = Path(__file__).with_name("companion.md")
LOCAL_PATH = Path(__file__).with_name("persona_local.md")
STYLE_PATH = Path(__file__).with_name("style_examples.md")
STYLE_HEADER = "Made-up examples of your style. They never happened. Never mention them."


def load_traits_file(path: Path = TRAITS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_traits() -> dict[str, float]:
    return {k: float(v) for k, v in load_traits_file()["traits"].items()}


def persona_prompt(kind: Literal["local", "online"] = "online") -> str:
    """The small local model gets the short persona_local.md; the online one the base sheet plus the companion layer."""
    if kind == "local":
        return LOCAL_PATH.read_text(encoding="utf-8").strip()
    return persona_text() + "\n\n" + COMPANION_PATH.read_text(encoding="utf-8")


def style_examples() -> str:
    """A few made-up exchanges in her voice, labelled as never having happened (they sit next to the persona)."""
    return STYLE_PATH.read_text(encoding="utf-8").strip()
