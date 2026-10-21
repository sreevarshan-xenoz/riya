"""v0.10 Task 1: the hold-out module (stdlib only), so training rows can never overlap the gate's probes."""
import json
import subprocess
import sys
from pathlib import Path

from riya.train import build_sft, holdout, scenarios

ROOT = Path(__file__).resolve().parents[2]


def _row(user: str, rid: str = "r", system: str = "You are Riya.") -> dict:
    return {"id": rid, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}


def test_holdout_is_stdlib_only():
    code = ("import riya.train.holdout, colab.train_riya_v1, sys; "
            "print(sorted(m for m in sys.modules if m.startswith(('riya.core', 'numpy', 'torch'))))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_first_hit_names_the_probe_or_fact():
    text = next(t for t in holdout.probe_texts() if len(holdout.normalize(t).split()) >= 7)
    assert holdout.first_hit(_row("ok so " + text), [text], []) == text
    fact = "The user's cat is called Pebble"
    assert holdout.first_hit(_row("hi", system="THE USER'S CAT IS CALLED PEBBLE"), [], [fact]) == fact
    assert holdout.first_hit(_row("hi there"), [text], [fact]) is None


def test_probes_at_swaps_and_restores(tmp_path):
    p = tmp_path / "probes.json"
    p.write_text(json.dumps({"conversations": [{"id": "x", "facts": ["The user has a pet newt"], "turns": [
        {"id": "t", "kind": "small", "text": "a one off probe text"}]}]}), encoding="utf-8")
    with holdout.probes_at(p):
        assert holdout.probe_texts() == ["a one off probe text"]
        assert holdout.probe_facts() == ["The user has a pet newt"]
        kept, dropped = holdout.drop_holdout([_row("a one off probe text", "hit"), _row("hello there", "ok")])
        assert [r["id"] for r in kept] == ["ok"] and dropped == ["hit"]
    assert len(holdout.probe_texts()) >= 20 and holdout.PROBES_PATH != p


def test_probes_at_restores_after_an_error(tmp_path):
    before = holdout.PROBES_PATH
    try:
        with holdout.probes_at(tmp_path / "nope.json"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert holdout.PROBES_PATH == before


def test_drop_holdout_keeps_order_and_counts():
    texts = holdout.probe_texts()
    rows = [_row("a perfectly ordinary message", "a"), _row(texts[0], "b"), _row("another fresh one", "c")]
    kept, dropped = holdout.drop_holdout(rows)
    assert [r["id"] for r in kept] == ["a", "c"] and dropped == ["b"]
    assert holdout.holdout_violations(rows, texts, holdout.probe_facts()) == ["b"]


def test_scenarios_and_build_sft_reexport():
    assert scenarios.overlaps is holdout.overlaps
    assert scenarios.normalize is holdout.normalize
    assert scenarios.probe_texts is holdout.probe_texts and scenarios.probe_facts is holdout.probe_facts
    assert build_sft.holdout_violations is holdout.holdout_violations
    assert scenarios.HOLDOUT_N == 6 == holdout.HOLDOUT_N
    assert scenarios.PROBES_PATH == holdout.PROBES_PATH
