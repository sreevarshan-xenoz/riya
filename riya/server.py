"""Riya's model server with GAME MODE: share the GPU politely.

Riya normally keeps her small model on the GPU (~1-1.5 GB). When free VRAM drops below a threshold (a game or other
app needs it), she saves her conversation slot, restarts llama-server on the CPU only, and restores the slot, so the
conversation continues. When enough VRAM has stayed free for a while, she moves back to the GPU the same way.

  python -m riya.server --model path/to/riya.gguf
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request

LLAMA_SERVER = os.environ.get("LLAMA_SERVER", r"%USERPROFILE%\llama.cpp\llama-server.exe")


def free_vram_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


class GameModePolicy:
    """Decide GPU vs CPU from free VRAM with hysteresis, so Riya does not flap during loading screens.
    Go to CPU immediately when free VRAM < low_mb; return to GPU only after free VRAM >= high_mb for back_after_s."""

    def __init__(self, low_mb=800, high_mb=2500, back_after_s=60):
        self.low_mb, self.high_mb, self.back_after_s = low_mb, high_mb, back_after_s
        self.on_gpu, self._free_since = True, None

    def update(self, free_mb, now):
        if self.on_gpu and free_mb < self.low_mb:
            self.on_gpu, self._free_since = False, None
            return "to_cpu"
        if not self.on_gpu:
            if free_mb >= self.high_mb:
                self._free_since = self._free_since or now
                if now - self._free_since >= self.back_after_s:
                    self.on_gpu, self._free_since = True, None
                    return "to_gpu"
            else:
                self._free_since = None
        return None


class RiyaServer:
    def __init__(self, model, port=8090, ctx=16384, threads=6, gpu_layers=99, slot_dir="riya_slots", extra=()):
        self.model, self.port, self.ctx, self.threads, self.gpu_layers = model, port, ctx, threads, gpu_layers
        self.slot_dir, self.extra = os.path.abspath(slot_dir), list(extra)
        os.makedirs(self.slot_dir, exist_ok=True)
        self.proc, self.on_gpu, self.lock = None, True, threading.Lock()

    def _cmd(self, gpu):
        # -np 1: one conversation slot; KV cache quantised; prompt cache keeps follow-up turns fast.
        return [LLAMA_SERVER, "-m", self.model, "--port", str(self.port), "-c", str(self.ctx), "-np", "1",
                "-ngl", str(self.gpu_layers if gpu else 0), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", str(self.threads), "--slot-save-path", self.slot_dir, "--no-webui", *self.extra]

    def _post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")

    def _wait_ready(self, timeout=120):
        end = time.time() + timeout
        while time.time() < end:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError("llama-server did not become ready")

    def start(self, gpu=True):
        self.proc = subprocess.Popen(self._cmd(gpu), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.on_gpu = gpu
        self._wait_ready()

    def move(self, gpu):
        """Save the conversation slot, restart on GPU/CPU, restore the slot."""
        with self.lock:
            try:
                self._post("/slots/0?action=save", {"filename": "riya.bin"})
                saved = True
            except Exception:
                saved = False
            self.proc.terminate()
            self.proc.wait(timeout=30)
            self.start(gpu)
            if saved:
                try:
                    self._post("/slots/0?action=restore", {"filename": "riya.bin"})
                except Exception:
                    pass
        print(f"[riya] moved to {'GPU' if gpu else 'CPU (game mode)'}", flush=True)

    def watch(self, policy: GameModePolicy, every_s=3.0):
        while self.proc and self.proc.poll() is None:
            action = policy.update(free_vram_mb(), time.time())
            if action:
                self.move(gpu=action == "to_gpu")
            time.sleep(every_s)

    def stop(self):
        if self.proc:
            self.proc.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--low_mb", type=int, default=800, help="go to CPU when free VRAM drops below this")
    p.add_argument("--high_mb", type=int, default=2500, help="return to GPU when free VRAM stays above this")
    a = p.parse_args()
    srv = RiyaServer(a.model, a.port, a.ctx, a.threads)
    srv.start(gpu=free_vram_mb() >= a.high_mb)
    policy = GameModePolicy(a.low_mb, a.high_mb)
    policy.on_gpu = srv.on_gpu
    print(f"[riya] serving on :{a.port} ({'GPU' if srv.on_gpu else 'CPU'})", flush=True)
    try:
        srv.watch(policy)
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
