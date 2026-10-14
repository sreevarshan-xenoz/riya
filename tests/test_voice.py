import numpy as np
import pytest

from jarvis.voice.affect_map import (AFFECT_NAMES, ALL_LABELS, DAILYDIALOG_LABELS, ED_LABELS, GOEMOTIONS_LABELS, HIGH,
                                     LOW, AffectMapper, drive_levels, smooth_mood)
from jarvis.voice.build_dataset import detok, dialog_rows
from jarvis.voice.chat import ASSISTANT, END, USER, format_row, render_chat
from jarvis.voice.synth_prompts import build_prompt, describe_affect, load_persona, sample_affect

M = AffectMapper()  # offline fallback table; the NRC lexicon is never downloaded in tests


def test_label_sets_and_ranges():
    assert len(ED_LABELS) == 32 and len(DAILYDIALOG_LABELS) == 7 and len(GOEMOTIONS_LABELS) == 28
    vecs = np.stack([M.label_vector(lab) for lab in ALL_LABELS])
    assert vecs.shape == (len(ALL_LABELS), len(AFFECT_NAMES))
    assert (vecs >= LOW).all() and (vecs <= HIGH).all()


def test_known_labels():
    v = {lab: M.label_vector(lab) for lab in ("joyful", "sad", "terrified", "content", "surprised", "curiosity",
                                              "neutral", "anticipating")}
    assert v["joyful"][0] > 0.5 > -0.5 > v["sad"][0]
    assert v["terrified"][1] > v["content"][1]  # arousal
    assert v["surprised"][7] > 0.7 and v["sad"][7] < 0.2
    assert v["curiosity"][6] > 0.7 and v["anticipating"][6] > 0.6
    assert v["neutral"][0] == 0 and M.label_vector("no emotion")[0] == 0
    assert np.allclose(M.label_vector(["joy", "sadness"])[0], (M.label_vector("joy")[0] + M.label_vector("sadness")[0]) / 2)
    with pytest.raises(KeyError):
        M.label_vector("flabbergasted-ish")


def test_mood_and_drives():
    assert smooth_mood([]) == 0
    assert 0 < smooth_mood([1.0]) < smooth_mood([1.0, 1.0]) < 1
    rng = np.random.default_rng(0)
    calm = np.array([drive_levels("What a lovely day.", rng) for _ in range(200)])
    assert calm.max() < 0.6 and calm.mean() < 0.2
    assert drive_levels("I'm so tired today", rng)[0] >= 0.6
    assert drive_levels("My back is in pain", rng)[1] >= 0.6
    vec = M.turn_affect("sad", "I am exhausted", [-0.8, -0.8], np.random.default_rng(1))
    assert vec[3] < -0.4 and vec[4] >= 0.6


def test_dialog_rows_roles_and_mood():
    turns, spk = ["hi", "hello", "I lost my keys", "oh no"], ["A", "B", "A", "B"]
    rows = dialog_rows("toy", "train", 0, turns, spk, ["sad"] * 4, M, keep=lambda i: spk[i] == "A")
    assert [r["response"] for r in rows] == ["hi", "I lost my keys"]
    assert [t["role"] for t in rows[1]["context"]] == ["assistant", "user"]
    assert rows[0]["affect"][3] == 0 and rows[1]["affect"][3] < 0  # mood from the speaker's prior turns only
    assert len(rows[0]["affect"]) == 8 and rows[0]["label"] == "sad"
    assert detok("Say , Jim , it ' s fine ? ") == "Say, Jim, it's fine?"
    assert detok("I can ’ t find it 。") == "I can't find it."


def test_describe_affect():
    assert describe_affect([0, 0.5, 0.5, 0, 0, 0, 0.3, 0]) == "calm and even-keeled"
    d = describe_affect([-0.5, 0.8, 0.3, 0, 0.8, 0.0, 0.9, 0.0])
    assert "anxious" in d and "energy" in d and "very curious" in d and "hesitant" in d
    assert "excited" in describe_affect([0.9, 0.9, 0.5, 0, 0, 0, 0.3, 0.9])
    assert "repair" in describe_affect([0, 0.5, 0.5, 0, 0, 0.9, 0.3, 0])
    vecs = sample_affect(500, np.random.default_rng(0))
    assert (vecs >= LOW).all() and (vecs <= HIGH).all() and (vecs[:, 0] > 0.9).any() and (vecs[:, 0] < -0.9).any()
    assert all(isinstance(describe_affect(v), str) for v in vecs)


def test_teacher_prompt_and_persona():
    persona = load_persona()
    from jarvis.voice.persona import NAME

    assert persona.startswith(f"You are {NAME}") and "{name}" not in persona and "human" in persona
    vec = np.array([0.1, 0.5, 0.5, 0, 0.9, 0, 0.3, 0])
    p = build_prompt(describe_affect(vec), vec, "Hello?", explicit_ok=False)
    assert "User: Hello?" in p and "recharge" in p and "Do not name" in p


def test_chat_format():
    row = {"context": [{"role": "user", "text": "Hi Jarvis"}, {"role": "assistant", "text": "Good evening."},
                       {"role": "user", "text": "How are you?"}], "response": "Splendid, thank you."}
    out = format_row(row, system="Be brief.")
    assert out["text"] == out["prompt"] + out["completion"]
    assert out["prompt"].startswith("<|system|>\nBe brief.<|end|>\n<|user|>\nHi Jarvis<|end|>\n")
    assert out["prompt"].endswith(f"{USER}\nHow are you?{END}\n{ASSISTANT}\n")
    assert out["completion"] == "Splendid, thank you.<|end|>\n"
    assert render_chat([]) == "" and format_row({"context": [], "response": "x"})["prompt"] == ASSISTANT + "\n"
