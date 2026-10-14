import json
import moonshine_voice as mv
import moonshine_voice.download as dl
cat = dl._stt_catalog()
print(json.dumps(cat.get("en"), default=str, indent=1)[:3000])
for arch in mv.ModelArch:
    try:
        p, a = mv.get_model_for_language("en", arch, on_progress=lambda f, n: None)
        print("OK", arch, p)
    except Exception as e:
        print("FAIL", arch, str(e)[:200])
