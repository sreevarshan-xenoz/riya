"""On the T4 VM: build llama.cpp with CUDA (sm75 only, server target) and download Riya candidate GGUFs.
Runs detached; progress in /content/runs/llamacpp.out."""
import os
import subprocess

script = r"""
set -e
cd /content
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build --target llama-server -j 2
echo BUILD_DONE
pip -q install huggingface_hub
python - <<'EOF'
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
for repo in ["LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "mradermacher/Huihui-LFM2.5-1.2B-Instruct-abliterated-GGUF",
             "mradermacher/LFM2.5-1.2B-Instruct-Heretic-i1-GGUF", "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF"]:
    f = [x for x in api.list_repo_files(repo) if x.endswith(".gguf") and "q4_k_m" in x.lower() and "mmproj" not in x.lower()]
    print(repo, "->", hf_hub_download(repo, f[0], local_dir="/content/models") if f else "NO Q4_K_M", flush=True)
EOF
echo MODELS_DONE
"""
log = open("/content/runs/llamacpp.out", "w")
p = subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("llama.cpp build + model download started, pid", p.pid)
