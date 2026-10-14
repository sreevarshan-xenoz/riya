"""CPU voice benchmark for Riya (run detached on a Colab CPU session; log /content/runs/voice_bench.out).

TTS: time to first audio + real-time factor for a short reply. STT: latency + word error on the TTS audio (round trip).
Each engine is optional: install/import failures are reported, not fatal. Colab CPU has 2 cores; the laptop has 8,
so absolute numbers here are pessimistic — use them to rank engines.
"""
import json
import os
import re
import subprocess
import time
import traceback

import numpy as np

TEXT = "Oh, you're back! I missed you. Want to brainstorm that game idea, or should I just make you laugh first?"
OUT = "/content/runs/voice"
os.makedirs(OUT, exist_ok=True)
results = {}


def pip(*pkgs):
    r = subprocess.run(["pip", "-q", "install", *pkgs], capture_output=True, text=True)
    return r.returncode == 0, r.stderr[-300:]


def wer(ref, hyp):
    r, h = re.findall(r"[a-z']+", ref.lower()), re.findall(r"[a-z']+", hyp.lower())
    d = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        prev, d[0] = d.copy(), i
        for j in range(1, len(h) + 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (r[i - 1] != h[j - 1]))
    return float(d[len(h)] / max(1, len(r)))


def record(name, fn):
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = {"error": f"{type(e).__name__}: {e}"[:300], "trace": traceback.format_exc()[-400:]}
    print(name, json.dumps(results[name])[:400], flush=True)


def save_wav(path, audio, sr):
    import soundfile as sf

    sf.write(path, np.asarray(audio, dtype=np.float32), sr)


# ---------------- TTS ----------------
def tts_kokoro():
    pip("kokoro-onnx", "soundfile")
    import urllib.request

    base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
    for f in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
        if not os.path.exists(f"{OUT}/{f}"):
            urllib.request.urlretrieve(base + f, f"{OUT}/{f}")
    from kokoro_onnx import Kokoro

    k = Kokoro(f"{OUT}/kokoro-v1.0.onnx", f"{OUT}/voices-v1.0.bin")
    k.create("Hi.", voice="af_heart", speed=1.0, lang="en-us")  # warm-up
    first_sentence = TEXT.split("!")[0] + "!"
    t0 = time.time()
    a1, sr = k.create(first_sentence, voice="af_heart", speed=1.0, lang="en-us")
    first_audio = time.time() - t0
    t0 = time.time()
    audio, sr = k.create(TEXT, voice="af_heart", speed=1.0, lang="en-us")
    dt = time.time() - t0
    save_wav(f"{OUT}/kokoro.wav", audio, sr)
    return {"first_sentence_ms": round(first_audio * 1000), "rtf": round(dt / (len(audio) / sr), 3),
            "audio_s": round(len(audio) / sr, 2), "wav": f"{OUT}/kokoro.wav", "voice": "af_heart"}


def tts_pocket():
    ok, err = pip("pocket-tts", "soundfile")
    if not ok:
        raise RuntimeError("pip pocket-tts failed: " + err)
    from pocket_tts import TTSModel  # API per kyutai-labs/pocket-tts README

    model = TTSModel.load_model()
    voice = model.get_state_for_audio_prompt("alba")  # a built-in female voice
    model.generate_audio(voice, "Hi.")
    t0 = time.time()
    first = None
    chunks = []
    for chunk in model.generate_audio_stream(voice, TEXT):
        first = first or time.time() - t0
        chunks.append(np.asarray(chunk).reshape(-1))
    dt = time.time() - t0
    audio = np.concatenate(chunks)
    sr = model.sample_rate
    save_wav(f"{OUT}/pocket.wav", audio, sr)
    return {"first_audio_ms": round(first * 1000), "rtf": round(dt / (len(audio) / sr), 3),
            "audio_s": round(len(audio) / sr, 2), "wav": f"{OUT}/pocket.wav", "voice": "alba"}


# ---------------- STT ----------------
def load_16k(path):
    import soundfile as sf

    a, sr = sf.read(path)
    if a.ndim > 1:
        a = a.mean(1)
    if sr != 16000:
        x = np.linspace(0, len(a), int(len(a) * 16000 / sr), endpoint=False)
        a = np.interp(x, np.arange(len(a)), a)
    return a.astype(np.float32)


def stt_faster_whisper(size):
    def run():
        pip("faster-whisper")
        from faster_whisper import WhisperModel

        m = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=2)
        audio = load_16k(results["tts_kokoro"]["wav"])
        list(m.transcribe(audio[:16000], beam_size=1)[0])  # warm-up
        t0 = time.time()
        text = " ".join(s.text for s in m.transcribe(audio, beam_size=1, language="en")[0])
        return {"latency_ms": round((time.time() - t0) * 1000), "audio_s": round(len(audio) / 16000, 2),
                "wer": round(wer(TEXT, text), 3), "text": text}
    return run


def stt_moonshine():
    ok, err = pip("useful-moonshine-onnx@git+https://github.com/moonshine-ai/moonshine.git#subdirectory=moonshine-onnx")
    if not ok:
        raise RuntimeError("pip moonshine failed: " + err)
    import moonshine_onnx

    audio = load_16k(results["tts_kokoro"]["wav"])
    moonshine_onnx.transcribe(audio[None, :16000], "moonshine/base")
    t0 = time.time()
    text = moonshine_onnx.transcribe(audio[None, :], "moonshine/base")[0]
    return {"latency_ms": round((time.time() - t0) * 1000), "audio_s": round(len(audio) / 16000, 2),
            "wer": round(wer(TEXT, text), 3), "text": text, "model": "moonshine/base (v1)"}


record("tts_kokoro", tts_kokoro)
record("tts_pocket", tts_pocket)
if "wav" in results.get("tts_kokoro", {}):
    record("stt_moonshine_base", stt_moonshine)
    record("stt_whisper_base_en", stt_faster_whisper("base.en"))
    record("stt_whisper_small_en", stt_faster_whisper("small.en"))
    record("stt_whisper_turbo", stt_faster_whisper("large-v3-turbo"))
json.dump(results, open(f"{OUT}/voice_bench.json", "w"), indent=1)
print("VOICE_BENCH_DONE", flush=True)
