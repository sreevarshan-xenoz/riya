"""Riya Champion Loop, pass 1, on the T4 (detached). Log: /content/runs/pass1.out (look for STEP / DONE markers)."""
import subprocess

script = r"""
log() { echo "STEP $(date +%H:%M) $*"; }
cd /content/code
mkdir -p /content/voice /content/runs/bench
[ -f /content/synth_requests.jsonl ] && mv /content/synth_requests.jsonl /content/voice/
log "waiting for llama-server build"
while [ ! -f /content/llama.cpp/build/bin/llama-server ] || pgrep -x gmake > /dev/null; do sleep 30; done
log "building llama-quantize"
(cd /content/llama.cpp && cmake --build build --target llama-quantize -j 2 > /content/runs/quantize_build.out 2>&1)
pip -q install peft accelerate gguf > /dev/null 2>&1
export LLAMA_SERVER=/content/llama.cpp/build/bin/llama-server
while ! grep -q MODELS_DONE /content/runs/models.out 2>/dev/null; do sleep 20; done

bench() {  # $1 = gguf path
  name=$(basename "$1" .gguf); extra=""; case "$name" in *wen*) extra="--reasoning-budget 0";; esac
  python -m riya.bench --model "$1" --ngl 99 --ctx 16384 --threads 2 --long_turns 30 --extra "$extra" \
    --out /content/runs/bench/$name-gpu.json > /dev/null 2>&1 || echo "FAILED bench $name-gpu"
  python -m riya.bench --model "$1" --ngl 0 --ctx 16384 --threads 2 --long_turns 8 --extra "$extra" \
    --out /content/runs/bench/$name-cpu.json > /dev/null 2>&1 || echo "FAILED bench $name-cpu"
  log "benched $name"
}
for m in /content/models/*.gguf; do bench "$m"; done
log "BENCH_DONE"

log "generating Riya dialogues with the teacher"
N=5000 python colab/gen_riya_data.py > /content/runs/gen.out 2>&1 || echo "FAILED gen"
log "GEN_DONE $(wc -l < /content/voice/riya_synth.jsonl) rows"

log "fine-tuning Riya v1"
python train_riya_lora.py --base huihui-ai/Huihui-LFM2.5-1.2B-Instruct-abliterated \
  --data /content/voice/riya_synth.jsonl --out_dir /content/runs/riya-v1 > /content/runs/train_riya.out 2>&1 || echo "FAILED train"
log "TRAIN_DONE"

python /content/llama.cpp/convert_hf_to_gguf.py /content/runs/riya-v1/merged --outtype f16 \
  --outfile /content/runs/riya-v1/riya-v1-f16.gguf > /content/runs/convert.out 2>&1 || echo "FAILED convert"
/content/llama.cpp/build/bin/llama-quantize /content/runs/riya-v1/riya-v1-f16.gguf \
  /content/runs/riya-v1/riya-v1-Q4_K_M.gguf Q4_K_M >> /content/runs/convert.out 2>&1 || echo "FAILED quantize"
bench /content/runs/riya-v1/riya-v1-Q4_K_M.gguf
log "PASS1_DONE"
"""
log = open("/content/runs/pass1.out", "a")
p = subprocess.Popen(["bash", "-c", script], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
print("pass 1 pipeline started, pid", p.pid)
