"""Persona layers: the base persona sheet plus the companion layer, and the versioned trait values."""
import json
from pathlib import Path

from jarvis.voice.persona import persona_text

TRAITS_PATH = Path(__file__).with_name("traits.json")
COMPANION_PATH = Path(__file__).with_name("companion.md")


def load_traits_file(path: Path = TRAITS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_traits() -> dict[str, float]:
    return {k: float(v) for k, v in load_traits_file()["traits"].items()}


def persona_prompt() -> str:
    return persona_text() + "\n\n" + COMPANION_PATH.read_text(encoding="utf-8")
