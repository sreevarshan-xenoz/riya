import sys
import types
import wave

import numpy as np
import pytest

from riya import voice_candidates as vc

SR_OUT = 24000


def tone(amplitude, n, sr=SR_OUT):
    return (amplitude * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)


def test_voice_candidates_pick_feminine_builtins_or_alba(tmp_path):
    assert vc.pick_voices(["marius", "alba", "eve", "javert"]) == ["alba", "eve"]
    assert vc.pick_voices([]) == ["alba"]
    path = tmp_path / "x.wav"
    vc.write_wav(path, tone(0.3, n=2400), SR_OUT)
    with wave.open(str(path)) as w:
        assert (w.getframerate(), w.getnframes(), w.getsampwidth()) == (SR_OUT, 2400, 2)
    args = vc.parse_args([])
    assert args.out == "scratch_runs/voice_candidates" and args.voices is None
    assert args.text == "You're back! I missed you~ How was your day?"


def test_voice_candidates_fail_when_no_voice_could_be_made(tmp_path, monkeypatch):
    class BrokenModel:
        sample_rate = SR_OUT

        @classmethod
        def load_model(cls):
            return cls()

        def get_state_for_audio_prompt(self, name):
            raise ValueError(f"unknown voice {name}")

        def generate_audio(self, state, text):
            return np.zeros(10, np.float32)

    monkeypatch.setitem(sys.modules, "pocket_tts", types.SimpleNamespace(TTSModel=BrokenModel))
    with pytest.raises(SystemExit) as e:
        vc.main(["--out", str(tmp_path), "--voices", "nobody,noone"])
    assert e.value.code not in (0, None)
