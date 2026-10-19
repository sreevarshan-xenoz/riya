"""Writes one sample WAV per candidate voice so the user can pick hers; the chosen name goes into settings.voice.

  python -m riya.voice_candidates                      # the feminine built-in Pocket TTS voices
  python -m riya.voice_candidates --voices alba,eve    # or names / reference WAVs of your choice
"""
import argparse
import wave
from pathlib import Path

import numpy as np

TEXT = "You're back! I missed you~ How was your day?"
# Pocket TTS's English built-in voices that sound feminine. Its VCTK voices map to speakers p228, p229, p244, p262,
# p303, p333, p339 and p361, which are female; the other built-ins are male or for other languages.
FEMININE = ("alba", "cosette", "anna", "vera", "fantine", "eponine", "azelma", "mary", "jane", "eve", "caro_davy")


def builtin_voices() -> list[str]:
    """The built-in voice names the installed pocket_tts exposes, or [] when it exposes none."""
    try:
        from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES
    except ImportError:
        return []
    return list(_ORIGINS_OF_PREDEFINED_VOICES)


def pick_voices(known: list[str]) -> list[str]:
    return [v for v in FEMININE if v in known] or ["alba"]


def voice_choices() -> list[str]:
    """Her voices for the Health page and the welcome: the installed feminine built-ins, or the whole FEMININE list
    when the TTS stack isn't there (text-only); a choice applies once her voice runs."""
    try:
        known = builtin_voices()
    except Exception:  # an installed but broken pocket_tts
        known = []
    return pick_voices(known) if known else list(FEMININE)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m riya.voice_candidates",
                                 description="Write one sample WAV per candidate voice for Riya.")
    ap.add_argument("--out", default="scratch_runs/voice_candidates", help="output folder")
    ap.add_argument("--voices", help="comma-separated Pocket TTS voice names or reference WAVs "
                                     "(default: the feminine built-in voices, or alba). A reference WAV must be an "
                                     "original or licensed voice, never an imitation of a real person.")
    ap.add_argument("--text", default=TEXT, help="what she says in each sample")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    voices = [v.strip() for v in args.voices.split(",") if v.strip()] if args.voices else pick_voices(builtin_voices())
    from pocket_tts import TTSModel

    model = TTSModel.load_model()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    written = 0
    for name in voices:
        try:
            audio = model.generate_audio(model.get_state_for_audio_prompt(name), args.text)
        except Exception as e:
            print(f"{name}: failed ({e})")
            continue
        path = out / f"{Path(name).stem}.wav"
        write_wav(path, np.asarray(audio, dtype=np.float32).reshape(-1), model.sample_rate)
        print(path)
        written += 1
    if not written:
        raise SystemExit("no voice sample could be made")
    print('Pick one and save its name as settings.voice (e.g. set_settings {"voice": "alba"}), then restart Riya.')


if __name__ == "__main__":
    main()
