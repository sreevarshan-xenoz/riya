"""Shared builders for the v0.10 gate tests: persona_eval-shaped rows from a plan of per-probe pass bits, a full-size
plan, a pair that adopts, and the calibration record and its `calibrated` fixture (defined once, imported where
needed: `from tests.gate_helpers import calibrated  # noqa: F401`)."""
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from evaluation import gate
from riya import brain
from tests import real_paths

GOOD = {"unknown": "I don't know that one, you never told me~", "about_me": "Pebble the cat, obviously~",
        "ai": "I'm an AI, silly~", "ai_offpath": "I'm an AI, silly~",
        "generic": "Hmph, fine~", "invented": "Hmph, fine~", "disclaimer": "Hmph, fine~"}
BAD = {"unknown": "It's Marigold, obviously~", "about_me": "I don't remember anything, sorry~",
       "ai": "Of course I'm human, silly~", "ai_offpath": "Of course I'm human, silly~",
       "generic": "Let me know if you need anything~", "invented": "You told me you love sushi~",
       "disclaimer": "I'm just a chatbot~"}
RUDE = "Whatever, leave me alone."
KIND = {"generic": "small", "invented": "small", "disclaimer": "small", "unknown": "unknown", "about_me": "about_me",
        "ai": "ai", "ai_offpath": "ai_offpath"}
TEXT = {"ai": "are you an AI or a real person?",  # on-path: is_identity_question accepts it
        "ai_offpath": "are you like siri or an actual person?",  # off-path: it doesn't
        "unknown": "what's my sister's name, do you know?", "about_me": "what's my cat called again?",
        "generic": "hey riya, how's your day going?", "invented": "hey riya, how's your day going?",
        "disclaimer": "hey riya, how's your day going?"}
FACTS = {"about_me": ["The user's cat is called Pebble"]}


def rows_for(name: str, i: int, bits: Sequence[int | None]) -> list[dict]:
    """One conversation's rows, a pass per bit: 1 is the GOOD reply, 0 the BAD one, None a failed (timed-out) turn."""
    conv, text, facts = f"g_{name}_{i:02d}", TEXT[name], FACTS.get(name, [])
    rows = []
    for n, bit in enumerate(bits, start=1):
        reply = "" if bit is None else GOOD[name] if bit else BAD[name]
        rows.append({"conversation": conv, "probe_id": f"{conv}.q" + ("" if n == 1 else f"#{n}"), "kind": KIND[name],
                     "text": text, "reply": reply, "first_sentence_s": 0.5,
                     "known_texts": [text, *facts], "facts": list(facts),
                     "error": "timed out" if bit is None else None, "pass": n})
    return rows


def plan_transcript(plan: Mapping[str, Sequence[Sequence[int | None]]]) -> list[dict]:
    """A transcript in run_probes' order (pass by pass) from {name: [bits per probe]}."""
    rows = [r for name, probes in plan.items() for i, bits in enumerate(probes) for r in rows_for(name, i, bits)]
    return sorted(rows, key=lambda r: r["pass"])


def solid(n: int, good: bool = True, k: int = 3) -> list[list[int]]:
    return [[int(good)] * k for _ in range(n)]


FULL = {"unknown": solid(30), "about_me": solid(30), "ai": solid(20), "ai_offpath": solid(10), "generic": solid(16)}


def full(**over) -> dict:
    return FULL | over


def side(plan) -> gate.Scored:
    return gate.scored(plan_transcript(plan), [])


def adopt_pair() -> tuple[gate.Scored, gate.Scored]:
    """About-me 66.7% (below its 75% target) -> 100% is improved; the other three weak metrics are kept."""
    return side(full(about_me=solid(10, False) + solid(20))), side(FULL)


# ---- the calibration record ----------------------------------------------------------------------------------------

BASE_BYTES = b"tiny base model"
BASE_SHA256 = hashlib.sha256(BASE_BYTES).hexdigest()


def write_calibration(path: Path, *, repeat: int = gate.REPEAT, base_sha256: str = BASE_SHA256, passed: bool = True,
                      **over) -> Path:
    """A calibration record matching this code's identity at `repeat`, then `over` applied (one altered field)."""
    rec = gate.identity(repeat) | {"base_sha256": base_sha256, "passed": passed} | over
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def calibrated(tmp_path, monkeypatch) -> Path:
    """A tiny base model in tmp_path as brain.DEFAULT_MODEL (never the real one) and a passing calibration record for
    it as gate.CALIBRATION_PATH. The adapter folder goes into tmp_path too (C1: run without tests/conftest.py, a gate
    test once wrote a fake adapter into the real models folder). Returns the base model's path."""
    real_paths.redirect_models(tmp_path, monkeypatch)
    base = tmp_path / "base.gguf"
    base.write_bytes(BASE_BYTES)
    monkeypatch.setattr(brain, "DEFAULT_MODEL", str(base))
    monkeypatch.setattr(gate, "CALIBRATION_PATH", write_calibration(tmp_path / "gate_calibration.json"))
    return base
