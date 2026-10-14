"""On the T4 VM: a larger uncensored teacher model writes Riya's replies for the synthetic affect requests.

Reads /content/voice/synth_requests.jsonl (from jarvis/voice/synth_prompts.py), serves the teacher with llama-server
(8 parallel slots), and writes /content/voice/riya_synth.jsonl rows in the voice-dataset format
{"source": "synth", "context": [{"role": "user", ...}], "response", "affect", "label"} for fine-tuning.
Detached; progress in /content/runs/gen.out. Resumable: skips ids already written.
"""
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

N = int(os.environ.get("N", "5000"))
TEACHER_SEARCH = os.environ.get("TEACHER", "Huihui-Qwen3.5-9B-abliterated-GGUF")
OUT = "/content/voice/riya_synth.jsonl"
SERVER = os.environ.get("LLAMA_SERVER", "/content/bin/llama-server")


def find_teacher():
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    for m in api.list_models(search=TEACHER_SEARCH, sort="downloads", limit=5):
        files = [f for f in api.list_repo_files(m.id) if f.endswith(".gguf") and "q4_k_m" in f.lower()]
        if files:
            print("teacher:", m.id, files[0], flush=True)
            return hf_hub_download(m.id, files[0], local_dir="/content/models/teacher")
    raise SystemExit(f"no Q4_K_M GGUF found for {TEACHER_SEARCH}")


def chat(system, user):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": 160, "temperature": 0.8, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:8095/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def main():
    teacher = find_teacher()
    proc = subprocess.Popen([SERVER, "-m", teacher, "--port", "8095", "-ngl", "99", "-c", "16384", "-np", "8",
                             "-fa", "on", "--no-webui"], stdout=subprocess.DEVNULL, stderr=open("/content/runs/teacher.log", "w"))
    for _ in range(600):
        try:
            urllib.request.urlopen("http://127.0.0.1:8095/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    done = set()
    if os.path.exists(OUT):
        done = {json.loads(line)["id"] for line in open(OUT)}
    reqs = [json.loads(line) for line in open("/content/voice/synth_requests.jsonl")][:N]
    todo = [r for r in reqs if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to generate", flush=True)

    def work(r):
        reply = chat(r["system"], r["prompt"])
        reply = reply.split("</think>")[-1].strip().strip('"')
        return {"id": r["id"], "source": "synth", "split": "train", "context": [{"role": "user", "text": r["user"]}],
                "response": reply, "affect": r["affect"], "label": r["affect_description"]}

    t0 = time.time()
    with open(OUT, "a") as f, ThreadPoolExecutor(8) as ex:
        for i, row in enumerate(ex.map(work, todo), 1):
            f.write(json.dumps(row) + "\n")
            if i % 100 == 0:
                f.flush()
                print(f"{i}/{len(todo)} ({i / (time.time() - t0):.1f}/s)", flush=True)
    proc.terminate()
    print("GEN_DONE", flush=True)


if __name__ == "__main__":
    main()
