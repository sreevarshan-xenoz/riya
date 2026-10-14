"""Find a working Moonshine (streaming-capable CPU STT) install and time it on the TTS round-trip clip."""
import json
import subprocess
import time

import numpy as np

attempts = [["moonshine-voice"]]
log = []
for pk in attempts:
    r = subprocess.run(["pip", "-q", "install", *pk], capture_output=True, text=True)
    log.append({"pkg": pk[0][:60], "ok": r.returncode == 0, "err": r.stderr[-200:]})
print(json.dumps(log, indent=1))
r = subprocess.run(["pip", "list"], capture_output=True, text=True).stdout
print([l for l in r.splitlines() if "moonshine" in l.lower()])

import soundfile as sf  # noqa: E402

a, sr = sf.read("/content/runs/voice/pocket.wav")
a = a.reshape(len(a), -1).mean(1) if a.ndim > 1 else a
x =np.interp(np.linspace(0, len(a), int(len(a) * 16000 / sr), endpoint=False), np.arange(len(a)), a).astype(np.float32)
for mod in ("moonshine_voice", "moonshine_onnx", "moonshine"):
    try:
        m = __import__(mod)
        print(mod, "attrs:", [k for k in dir(m) if not k.startswith("_")][:40])
        if hasattr(m, "transcribe"):
            for name in ("moonshine/base", "moonshine/tiny", "base", "tiny"):
                try:
                    m.transcribe(x[None, :16000], name)
                    t0 = time.time()
                    out = m.transcribe(x[None, :], name)
                    print(mod, name, f"{(time.time() - t0) * 1000:.0f} ms", out)
                    break
                except Exception as e:
                    print(mod, name, "failed:", str(e)[:150])
    except Exception as e:
        print(mod, "import failed:", str(e)[:150])
