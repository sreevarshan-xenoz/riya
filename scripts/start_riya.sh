#!/usr/bin/env bash
# Starts Riya on Linux: the local model server (capped and low priority, so it never crowds out your own apps), the
# core, and the desktop body (once the core accepts connections, or after 60 s regardless). Ctrl-C or the core
# exiting stops the model server too.
#
# The model runs on one GPU: the first discrete card llama-server lists (RIYA_GPU overrides it, e.g. Vulkan1;
# RIYA_GPU=none keeps it on the CPU). When a game takes the card's memory she moves to the CPU, two threads, and
# comes back once the memory has stayed free for a minute.
#
#   RIYA_MODEL      the base GGUF (default ~/models/riya/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf)
#   RIYA_CPU_QUOTA  the model server's CPU cap (default 200%: two of your threads at most)
#   RIYA_MEM_MAX    the model server's memory cap (default 2G)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
MODEL="${RIYA_MODEL:-$HOME/models/riya/Huihui-LFM2.5-1.2B-Instruct-abliterated.Q4_K_M.gguf}"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}/riya"
export RIYA_MODEL_DIR="${RIYA_MODEL_DIR:-$DATA/models}"  # the embedder's downloads (embed.py keeps its own default)
MODEL_PORT=8090
CORE_PORT=8765
CPU_QUOTA="${RIYA_CPU_QUOTA:-200%}"
MEM_MAX="${RIYA_MEM_MAX:-2G}"
SCOPE=riya-model

cd "$ROOT"
mkdir -p "$DATA"

healthy() { curl -sf "http://127.0.0.1:$MODEL_PORT/health" >/dev/null 2>&1; }
core_up() { (exec 3<>"/dev/tcp/127.0.0.1/$CORE_PORT") 2>/dev/null; }

pick_gpu() {
    if [[ -n "${RIYA_GPU:-}" ]]; then echo "$RIYA_GPU"; return; fi
    local server
    server="$("$PYTHON" -c 'from riya import server; print(server.LLAMA_SERVER)')"
    # A discrete card: NVIDIA, or an AMD Radeon that isn't the integrated "Graphics".
    "$server" --list-devices 2>/dev/null | grep -E '^\s+[A-Za-z]+[0-9]+:' \
        | grep -iE 'nvidia|geforce|rtx|radeon rx|radeon pro' | head -1 | sed -E 's/^\s+([A-Za-z]+[0-9]+):.*/\1/' || true
}

started_server=0
stop_server() {
    if [[ $started_server == 1 ]]; then
        systemctl --user stop "$SCOPE.scope" 2>/dev/null || kill "${server_pid:-0}" 2>/dev/null || true
    fi
}
trap stop_server EXIT
trap 'exit 143' TERM INT HUP  # a signal must run the EXIT trap too (systemd stops a service with SIGTERM)

if ! healthy; then
    [[ -f "$MODEL" ]] || { echo "No model at $MODEL (set RIYA_MODEL)." >&2; exit 1; }
    gpu="$(pick_gpu)"
    args=(-m riya.server --model "$MODEL" --follow-settings --threads 2 --low_mb 400 --high_mb 1500)
    if [[ -n "$gpu" && "$gpu" != none ]]; then
        args+=(--device "$gpu")
    else
        args+=(--high_mb 999999)  # no discrete card: stay on the CPU
    fi
    lora="$("$PYTHON" -m riya.server --print-lora 2>/dev/null || true)"
    [[ -n "$lora" ]] && args+=(--lora "$lora")
    if command -v systemd-run >/dev/null && systemctl --user is-system-running >/dev/null 2>&1; then
        # PartOf: when this runs as riya.service, stopping the service stops the model server too (a trap can't be
        # relied on while systemd is already tearing the service down).
        systemd-run --user --scope --quiet --collect --unit="$SCOPE" -p PartOf=riya.service \
            -p CPUQuota="$CPU_QUOTA" -p MemoryHigh=1536M -p MemoryMax="$MEM_MAX" -p CPUWeight=20 -p IOWeight=20 \
            nice -n 10 "$PYTHON" "${args[@]}" &
    else
        nice -n 10 "$PYTHON" "${args[@]}" &  # no user systemd: low priority only, no hard caps
    fi
    server_pid=$!
    started_server=1
    for _ in $(seq 120); do
        healthy && break
        kill -0 "$server_pid" 2>/dev/null || { echo "The model server exited before it became healthy." >&2; exit 1; }
        sleep 1
    done
    healthy || { echo "The model server did not become healthy within 120 s." >&2; exit 1; }
fi

# The core runs in this terminal (its log stays visible).
"$PYTHON" -m riya.core &
core_pid=$!
for _ in $(seq 120); do
    core_up && break
    kill -0 "$core_pid" 2>/dev/null || { wait "$core_pid"; code=$?; echo "The core exited with code $code before it opened port $CORE_PORT; not starting the body." >&2; exit "$code"; }
    sleep 0.5
done
core_up || echo "The core's port $CORE_PORT did not open within 60 s; starting the body anyway (it will connect when the core is up)."

if [[ -x "$ROOT/body/node_modules/electron/dist/electron" ]]; then
    (cd "$ROOT/body" && npm start >/dev/null 2>&1 &)
else
    echo "The desktop body isn't installed (run: node body/node_modules/electron/install.js); the core runs without it." >&2
fi
wait "$core_pid"
