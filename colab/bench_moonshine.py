"""Benchmark Moonshine STT latency for a voice assistant: how fast is the FINAL transcript after the speaker stops?

For each English moonshine_voice (v2, C++ runtime) model:
  * simulate a live mic: feed 0.1 s chunks paced to wall clock (deadline-based, so if a transcription pass
    blocks the feeder we fall behind exactly like a real mic queue would), then stop() the stream;
  * latency = wall time when final transcript is available - wall time the speaker stopped (t0 + clip length);
  * CPU time = process CPU seconds (all threads) over the whole streamed run;
  * WER vs the reference sentence.
Also times non-streaming transcription (moonshine_voice transcribe_without_streaming, and v1
useful-moonshine-onnx tiny/base) of the whole clip, which is the "VAD end -> transcribe buffer" pattern.

Usage: python bench_moonshine.py [wav] ; results -> stdout + /content/runs/moonshine_bench.json
"""
import json
import os
import re
import statistics
import sys
import time

import numpy as np

WAV = sys.argv[1] if len(sys.argv) > 1 else "/content/runs/voice/pocket.wav"
REF = "Oh, you're back! I missed you. Want to brainstorm that game idea, or should I just make you laugh first?"
CHUNK_S = 0.1
SR = 16000
RUNS = 2  # paced runs per config (after one unpaced warm-up)
OUT_JSON = "/content/runs/moonshine_bench.json"


def load_16k_mono(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    a, sr = sf.read(path, dtype="float32", always_2d=True)
    a = a.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        a = resample_poly(a, SR // g, sr // g)
    return np.clip(a, -1, 1).astype(np.float32)


def norm_words(s):
    s = s.lower().replace("’", "'")
    return re.sub(r"[^a-z0-9' ]+", " ", s).split()


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


def transcript_text(t):
    return " ".join(line.text.strip() for line in (t.lines if t else []) if line.text.strip())


def stream_once(transcriber, audio, update_interval, paced=True):
    """Feed audio as a live mic would; return (latency_ms, cpu_s, text, n_passes)."""
    stream = transcriber.create_stream(update_interval=update_interval)
    stream.start()
    n = int(CHUNK_S * SR)
    cpu0, t0 = time.process_time(), time.perf_counter()
    for i in range(0, len(audio), n):
        if paced:  # chunk i is only "recorded" at t0 + (i+n)/SR
            wait = t0 + (i + n) / SR - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        stream.add_audio(audio[i:i + n].tolist(), SR)
    speech_end = t0 + len(audio) / SR if paced else time.perf_counter()
    final = stream.stop()
    t_final = time.perf_counter()
    cpu = time.process_time() - cpu0
    if hasattr(stream, "close"):
        stream.close()
    return (t_final - speech_end) * 1000, cpu, transcript_text(final)


def bench_moonshine_voice(audio, results):
    import moonshine_voice as mv

    archs = [mv.ModelArch.TINY, mv.ModelArch.TINY_STREAMING, mv.ModelArch.BASE,
             mv.ModelArch.SMALL_STREAMING, mv.ModelArch.MEDIUM_STREAMING]
    for arch in archs:
        name = mv.model_arch_to_string(arch) if hasattr(mv, "model_arch_to_string") else str(arch)
        try:
            t = time.perf_counter()
            path, arch2 = mv.get_model_for_language("en", arch, on_progress=lambda f, n: None)
            tr = mv.Transcriber(path, arch2)
            load_ms = (time.perf_counter() - t) * 1000
        except Exception as e:
            results.append({"model": f"v2 {name}", "error": f"load: {e}"})
            print(f"v2 {name}: LOAD FAILED {e}", flush=True)
            continue
        # whole-clip (non-streaming) transcription, best of 3 after warm-up
        try:
            tr.transcribe_without_streaming(audio.tolist(), SR)
            times, text = [], ""
            for _ in range(3):
                t = time.perf_counter()
                text = transcript_text(tr.transcribe_without_streaming(audio.tolist(), SR))
                times.append((time.perf_counter() - t) * 1000)
            row = {"model": f"v2 {name}", "mode": "whole-clip", "ms": round(min(times)),
                   "ms_median": round(statistics.median(times)), "wer": round(wer(REF, text), 3), "text": text,
                   "load_ms": round(load_ms)}
            results.append(row)
            print(json.dumps(row), flush=True)
        except Exception as e:
            results.append({"model": f"v2 {name}", "mode": "whole-clip", "error": str(e)})
            print(f"v2 {name} whole-clip FAILED {e}", flush=True)
        for ui in (0.5, 0.2):
            try:
                stream_once(tr, audio, ui, paced=False)  # warm-up
                runs = [stream_once(tr, audio, ui) for _ in range(RUNS)]
                lat = [r[0] for r in runs]
                row = {"model": f"v2 {name}", "mode": f"stream ui={ui}", "ms": round(min(lat)),
                       "ms_median": round(statistics.median(lat)), "all_ms": [round(x) for x in lat],
                       "cpu_s": round(statistics.median(r[1] for r in runs), 2),
                       "wer": round(wer(REF, runs[-1][2]), 3), "text": runs[-1][2]}
                results.append(row)
                print(json.dumps(row), flush=True)
            except Exception as e:
                results.append({"model": f"v2 {name}", "mode": f"stream ui={ui}", "error": str(e)})
                print(f"v2 {name} stream ui={ui} FAILED {e}", flush=True)
        tr.close()


def bench_moonshine_onnx(audio, results):
    try:
        import moonshine_onnx
    except Exception as e:
        results.append({"model": "v1 onnx", "error": f"import: {e}"})
        return
    for name in ("moonshine/tiny", "moonshine/base"):
        try:
            t = time.perf_counter()
            model = moonshine_onnx.MoonshineOnnxModel(model_name=name)  # build once; transcribe(str) reloads every call
            load_ms = (time.perf_counter() - t) * 1000
            moonshine_onnx.transcribe(audio, model)  # 1-D float32 16 kHz; load_audio adds the batch dim
            times, text = [], ""
            for _ in range(3):
                t = time.perf_counter()
                text = " ".join(moonshine_onnx.transcribe(audio, model))
                times.append((time.perf_counter() - t) * 1000)
            row = {"model": f"v1 onnx {name}", "mode": "whole-clip", "ms": round(min(times)),
                   "ms_median": round(statistics.median(times)), "wer": round(wer(REF, text), 3), "text": text,
                   "load_ms": round(load_ms)}
        except Exception as e:
            row = {"model": f"v1 onnx {name}", "error": str(e)}
        results.append(row)
        print(json.dumps(row), flush=True)


def main():
    audio = load_16k_mono(WAV)
    print(f"clip {len(audio) / SR:.2f} s, cores={os.cpu_count()}", flush=True)
    results = []
    bench_moonshine_voice(audio, results)
    bench_moonshine_onnx(audio, results)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({"clip_s": len(audio) / SR, "cores": os.cpu_count(), "results": results}, f, indent=1)
    print("\nmodel | mode | ms (best/median) | cpu_s | WER")
    for r in results:
        if "error" in r:
            print(f"{r['model']} | {r.get('mode', '')} | ERROR {r['error'][:120]}")
        else:
            print(f"{r['model']} | {r['mode']} | {r['ms']}/{r['ms_median']} | {r.get('cpu_s', '-')} | {r['wer']}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
