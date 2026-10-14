lines = open("/content/runs/voice_bench.out", errors="replace").read().splitlines()
print("\n".join(l[:300] for l in lines if l.startswith(("tts_", "stt_", "VOICE"))) or "\n".join(lines[-3:]))
