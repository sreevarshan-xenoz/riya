import inspect

import moonshine_onnx
import moonshine_voice

print(inspect.getsource(moonshine_onnx.transcribe)[:900])
print("---- moonshine_voice attrs:", [k for k in dir(moonshine_voice) if not k.startswith("_")][40:])
for name in ("Transcriber", "transcribe", "MicTranscriber", "StreamingTranscriber"):
    if hasattr(moonshine_voice, name):
        obj = getattr(moonshine_voice, name)
        print("----", name, inspect.signature(obj) if callable(obj) else obj)
        print((inspect.getdoc(obj) or "")[:600])
