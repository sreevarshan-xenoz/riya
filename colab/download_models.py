"""On the T4 VM: download Riya candidate GGUFs (the build shell that would have done it was stopped)."""
import subprocess

code = r'''
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
for repo in ["LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "mradermacher/Huihui-LFM2.5-1.2B-Instruct-abliterated-GGUF",
             "mradermacher/LFM2.5-1.2B-Instruct-Heretic-i1-GGUF", "mradermacher/Huihui-Qwen3.5-2B-abliterated-GGUF"]:
    f = [x for x in api.list_repo_files(repo) if x.endswith(".gguf") and "q4_k_m" in x.lower() and "mmproj" not in x.lower()]
    print(repo, "->", hf_hub_download(repo, f[0], local_dir="/content/models") if f else "NO Q4_K_M", flush=True)
print("MODELS_DONE", flush=True)
'''
log = open("/content/runs/models.out", "w")
p = subprocess.Popen(["python", "-c", code], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("model download started pid", p.pid)
