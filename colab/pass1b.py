"""Riya pass 1, compile-free version (free Colab reclaims VMs, so no 2.5 h CUDA build). Detached; log /content/runs/pass1.out.

1. prebuilt llama.cpp (CPU release) -> bench the 4 uncensored GGUF candidates (refusals, long-chat stability, CPU speed)
2. teacher (transformers, 4-bit on T4) writes Riya dialogues          -> /content/voice/riya_synth.jsonl
3. LoRA fine-tune Riya v1 (transformers on T4)                         -> /content/runs/riya-v1/merged
4. convert (llama.cpp python script) + quantize (prebuilt) to GGUF     -> /content/runs/riya-v1/riya-v1-Q4_K_M.gguf
5. bench Riya v1. Real VRAM/latency is verified on the laptop at the user's 'try it' step.
"""
import subprocess

script = r"""
log() { echo "STEP $(date +%H:%M) $*"; }
cd /content/code
mkdir -p /content/voice /content/runs/bench /content/bin
[ -f /content/synth_requests.jsonl ] && mv /content/synth_requests.jsonl /content/voice/
[ -f /content/riya_synth.jsonl ] && mv /content/riya_synth.jsonl /content/voice/   # resume teacher output after a VM loss
mkdir -p /content/runs/synth && touch /content/voice/riya_synth.jsonl && ln -sf /content/voice/riya_synth.jsonl /content/runs/synth/riya_synth.jsonl
pip -q install peft accelerate gguf bitsandbytes huggingface_hub > /dev/null 2>&1

if [ ! -x /content/bin/llama-server ]; then
  log "downloading prebuilt llama.cpp (CUDA 12.8 Linux build, no compile)"
  python - <<'EOF' > /tmp/urls
import json, urllib.request
for rel in json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10")):
    names = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    main = [u for n, u in names.items() if n.startswith("llama-") and "ubuntu-cuda-12.8-x64" in n]
    rt = [u for n, u in names.items() if n.startswith("cudart-") and "ubuntu-cuda-12.8-x64" in n]
    if main:
        print(main[0]); print(rt[0] if rt else ""); break
EOF
  if [ ! -d /tmp/llama ]; then
    mkdir -p /tmp/llama
    for u in $(cat /tmp/urls); do curl -sL "$u" | tar -xz -C /tmp/llama; done
  fi
  find /tmp/llama \( -name 'llama-server' -o -name 'llama-quantize' -o -name '*.so*' \) -exec cp -a {} /content/bin/ \;
  chmod +x /content/bin/llama-server /content/bin/llama-quantize
  log "prebuilt llama.cpp: $(ls /content/bin | wc -l) files, $(/content/bin/llama-server --version 2>&1 | head -1)"
fi
export LLAMA_SERVER=/content/bin/llama-server LD_LIBRARY_PATH=/content/bin:$LD_LIBRARY_PATH
[ -d /content/llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp
while ! grep -q MODELS_DONE /content/runs/models.out 2>/dev/null; do sleep 20; done

bench() {
  name=$(basename "$1" .gguf); extra=""; case "$name" in *wen*) extra="--reasoning-budget 0";; esac
  python -m riya.bench --model "$1" --ngl 99 --ctx 16384 --threads 2 --long_turns 30 --extra "$extra" \
    --out /content/runs/bench/$name-gpu.json > /content/runs/bench/$name-gpu.log 2>&1 || echo "FAILED bench $name-gpu"
  python -m riya.bench --model "$1" --ngl 0 --ctx 8192 --threads 2 --long_turns 8 --extra "$extra" \
    --out /content/runs/bench/$name-cpu.json > /content/runs/bench/$name-cpu.log 2>&1 || echo "FAILED bench $name-cpu"
  log "benched $name"
}
for m in /content/models/*.gguf; do bench "$m"; done
log "BENCH_DONE"

log "teacher generation"
N=5000 python colab/gen_riya_data.py > /content/runs/gen.out 2>&1 || {
  log "llama-server teacher failed, falling back to transformers 4-bit teacher"
  N=5000 BATCH=16 python colab/gen_riya_data_hf.py >> /content/runs/gen.out 2>&1 || echo "FAILED gen"; }
log "GEN_DONE $(wc -l < /content/voice/riya_synth.jsonl) rows"

log "fine-tuning Riya v1"
python train_riya_lora.py --base huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated \
  --data /content/voice/riya_synth.jsonl --out_dir /content/runs/riya-v1 > /content/runs/train_riya.out 2>&1 || echo "FAILED train"
log "TRAIN_DONE"
python /content/llama.cpp/convert_hf_to_gguf.py /content/runs/riya-v1/merged --outtype f16 \
  --outfile /content/runs/riya-v1/riya-v1-f16.gguf > /content/runs/convert.out 2>&1 || echo "FAILED convert"
/content/bin/llama-quantize /content/runs/riya-v1/riya-v1-f16.gguf /content/runs/riya-v1/riya-v1-Q4_K_M.gguf Q4_K_M \
  >> /content/runs/convert.out 2>&1 || echo "FAILED quantize"
bench /content/runs/riya-v1/riya-v1-Q4_K_M.gguf
log "PASS1_DONE"
"""
p = subprocess.Popen(["bash", "-c", script], stdout=open("/content/runs/pass1.out", "a"), stderr=subprocess.STDOUT,
                     start_new_session=True)
print("pass 1b started, pid", p.pid)
