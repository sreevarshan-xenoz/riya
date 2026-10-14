import os

server = os.path.exists("/content/llama.cpp/build/bin/llama-server")
models = open("/content/runs/models.out", errors="replace").read() if os.path.exists("/content/runs/models.out") else ""
build_tail = open("/content/runs/llamacpp.out", errors="replace").read().splitlines()[-1][:100]
print(f"SERVER={'READY' if server else 'building'} MODELS={'DONE' if 'MODELS_DONE' in models else 'downloading'} | {build_tail}")
