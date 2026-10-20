"""The pet's constants (body/src/pet-logic.js) must agree with the core's kinds, settings and label cap."""
import re
from pathlib import Path

from riya.agents import hub
from riya.core import settings

JS = Path(__file__).resolve().parents[2] / "body" / "src" / "pet-logic.js"


def _text() -> str:
    return JS.read_text(encoding="utf-8")


def js_list(name: str) -> list:
    """A top-level `const NAME = [..];` of strings or numbers, in source order."""
    m = re.search(r"^const " + name + r" = \[([^\]]*)\];", _text(), re.M)
    assert m, f"pet-logic.js declares {name} as a one-line array"
    return [float(v) if re.fullmatch(r"-?[\d.]+", v) else v.strip('"') for v in re.findall(r'"[^"]*"|-?[\d.]+', m.group(1))]


def js_int(name: str) -> int:
    m = re.search(r"^const " + name + r" = (\d+);", _text(), re.M)
    assert m, f"pet-logic.js declares {name}"
    return int(m.group(1))


def js_reaction_keys() -> list[str]:
    m = re.search(r"^const REACTIONS = \{\n(.*?)\n\};", _text(), re.M | re.S)
    assert m, "pet-logic.js declares REACTIONS one kind per line"
    return re.findall(r"^  (\w+): \{", m.group(1), re.M)


def test_activity_kinds_match_the_hub():
    assert js_list("ACTIVITY_KINDS") == list(hub.ACTIVITY_KINDS)


def test_settings_lists_match():
    assert js_list("PET_WALKS_WHEN") == list(settings.PET_WALKS_WHEN)
    assert js_list("REDUCED_MOTION") == list(settings.REDUCED_MOTION)
    assert js_list("SCALE_RANGE") == list(settings.PET_SCALE_RANGE)


def test_bubble_cap_matches_the_label_cap():
    assert js_int("BUBBLE_MAX") == hub.LABEL_MAX


def test_default_scale_matches_the_settings_default():
    m = re.search(r"\bSCALE_DEFAULT: ([\d.]+)", _text())
    assert m, "pet-logic.js declares PET.SCALE_DEFAULT"
    assert float(m.group(1)) == settings.Settings().pet_scale
    assert settings.PET_SCALE_RANGE[0] <= float(m.group(1)) <= settings.PET_SCALE_RANGE[1]


def test_reactions_are_keyed_by_the_kinds():
    assert js_reaction_keys() == list(hub.ACTIVITY_KINDS)


def test_the_protocol_kinds_match_too():
    """protocol.py writes the kinds out (no hub import); the pet sees them through the same list."""
    from riya.core import protocol
    assert [k for k in protocol.ActivityKind.__args__] == js_list("ACTIVITY_KINDS")
