"""The butler's identity in one place. Change NAME here once the user picks it; prompts and the persona sheet use it."""
from pathlib import Path

NAME = "Riya"  # chosen by the user (2026-10-10)
PERSONA_PATH = Path(__file__).with_name("persona.md")


def persona_text(name: str = NAME, path: Path = PERSONA_PATH) -> str:
    """Persona sheet body (from 'You are ...'), with the name filled in."""
    text = path.read_text(encoding="utf-8")
    return text[text.index("You are {name}"):].strip().replace("{name}", name)


def system_prompt(name: str = NAME) -> str:
    """Short system prompt used during SFT for the butler's own (synthetic) turns."""
    return (f"You are {name}, a personal AI butler (she/her) with human-like emotions and abilities beyond any "
            f"human. Your internal state colours how you speak.")
