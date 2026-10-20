"""Riya's model server with GAME MODE: share the GPU politely.

Riya normally keeps her small model on the GPU (~1-1.5 GB). When free VRAM drops below a threshold (a game or other
app needs it), she saves her conversation slot, restarts llama-server on the CPU only, and restores the slot, so the
conversation continues. When enough VRAM has stayed free for a while, she moves back to the GPU the same way.

  python -m riya.server --model path/to/riya.gguf [--lora riya-v1-lora.gguf] [--follow-settings]

Her fine-tuned adapter goes to llama-server as --lora, but only when the champion gate adopted that exact file (a
matching .champion.json record, brain.read_record): every launch checks it, so a file copied over the adopted one, or
one the gate rejected or never measured, means the base model (and `unvetted` says so). A missing adapter, or one
llama-server can't load, never stops her from starting: she serves the base model (and `lora_failed` says so). Slot
caches are kept per variant, so a conversation slot saved by one variant is never restored into another.
"""
import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from riya import brain
from riya.brain import slot_key, usable_adapter, vetted_adapter

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
    def __init__(self, model, port=8090, ctx=16384, threads=6, gpu_layers=99, slot_dir="riya_slots", extra=(), *,
                 lora: str | None = None, fallback_to_base: bool = True, vetted_only: bool = True,
                 popen=subprocess.Popen, healthy: Callable[[int], bool] | None = None,
                 ready_timeout_s: float = 120.0):
        """vetted_only (the default, and the only mode Riya's own server runs in): an adapter is served only while it
        has a matching champion adopt record. The champion gate turns it off for its own server, which is how a
        candidate gets measured at all."""
        self.model, self.port, self.ctx, self.threads, self.gpu_layers = model, port, ctx, threads, gpu_layers
        self.slot_root, self.extra = os.path.abspath(slot_dir), list(extra)
        self.wanted = lora  # the raw value asked for
        self.lora, self.lora_failed = usable_adapter(lora), None  # lora_failed: an adapter that didn't load
        self.unvetted: str | None = None  # an adapter the last launch skipped: no matching champion adopt record
        self.fallback_to_base, self.vetted_only = fallback_to_base, vetted_only
        self.popen, self.ready_timeout_s = popen, ready_timeout_s
        self.healthy = healthy or self._health_ok
        self.proc, self.on_gpu, self.lock = None, True, threading.Lock()
        self._pinned: str | None = None  # the slot dir of the variant launched last
        self._key = "base"
        self._serve: str | None = None  # the adapter the variant launched last serves (None: the base model)
        self._decided = self._identity()  # what the last launch decided from (see _identity)

    @property
    def serving(self) -> str | None:
        """The adapter llama-server was launched with last, or None for the base model."""
        return self._serve

    @property
    def slot_dir(self) -> str:
        """One slot-cache directory per variant (the base model, or one adapter file as it was when launched)."""
        if self._pinned is None:
            self._pin()
        return self._pinned

    def _vetted(self, lora: str, key: str) -> bool:
        """The champion adopted this exact file: its record matches, and it is still the file the key was taken from."""
        return not self.vetted_only or (vetted_adapter(lora) is not None and slot_key(lora) == key)

    def _identity(self) -> tuple:
        """(the adapter, its slot key, vetted): what a launch would decide on now. A None key: the file is gone."""
        if not self.lora:
            return None, None, False
        try:
            key = slot_key(self.lora)
            return self.lora, key, self._vetted(self.lora, key)
        except OSError:
            return self.lora, None, False

    def _pin(self, base: bool = False):
        """Fix this launch's variant and slot dir once, so nothing stats the adapter again while it launches. An
        adapter that vanished since we looked means the base model, never an exception; so does one without a
        matching champion adopt record (unvetted). base: this launch serves the base model, whatever is wanted (a
        fallback)."""
        serve, key = None, "base"
        if not base:
            self.unvetted = None
            if self.lora:
                try:
                    if not os.path.isfile(self.lora):
                        raise FileNotFoundError(self.lora)
                    k = slot_key(self.lora)
                    if self._vetted(self.lora, k):
                        serve, key = self.lora, k
                    else:
                        self.unvetted = self.lora
                except OSError:
                    self.lora_failed, self.lora = self.lora, None
            self._decided = self._identity()
        self._serve, self._key = serve, key
        self._pinned = os.path.join(self.slot_root, key)

    def _cmd(self, gpu):
        # -np 1: one conversation slot; KV cache quantised; prompt cache keeps follow-up turns fast.
        slot_dir = self.slot_dir  # pins the variant when nothing was launched yet
        return [LLAMA_SERVER, "-m", self.model, "--port", str(self.port), "-c", str(self.ctx), "-np", "1",
                "-ngl", str(self.gpu_layers if gpu else 0), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
                "-t", str(self.threads), "--slot-save-path", slot_dir, "--no-webui",
                *(["--lora", self._serve] if self._serve else []), *self.extra]

    def _post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read() or b"{}")

    @staticmethod
    def _health_ok(port: int) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def _wait_ready(self):
        end = time.monotonic() + self.ready_timeout_s
        while True:
            if self.proc.poll() is not None:
                raise RuntimeError("llama-server exited")
            if self.healthy(self.port):
                return
            if time.monotonic() >= end:
                raise RuntimeError("llama-server did not become ready")
            time.sleep(0.25)

    def _launch(self, gpu, base: bool = False):
        self._pin(base)
        os.makedirs(self._pinned, exist_ok=True)
        self.proc = self.popen(self._cmd(gpu), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.on_gpu = gpu
        self._wait_ready()
        if self._serve:  # the file was replaced or removed while the server loaded it: it isn't the pinned variant
            try:
                same = slot_key(self._serve) == self._key
            except OSError:
                same = False
            if not same:
                raise RuntimeError("the adapter changed while llama-server was loading it")

    def start(self, gpu=True):
        """Launch; an adapter that won't load means one relaunch on the base model. Any failure that is raised leaves
        no llama-server running."""
        try:
            self._launch(gpu)
        except RuntimeError:
            if not (self._serve and self.fallback_to_base):
                self.stop()
                raise
            print(f"[riya] the adapter {self._serve} didn't load; serving the base model", flush=True)
            self.stop()
            self.lora_failed, self.lora = self._serve, None
            self._decided = self._identity()
            try:
                self._launch(gpu, base=True)
            except RuntimeError:
                self.stop()
                raise
        if self.unvetted:
            print(f"[riya] the adapter {self.unvetted} has no passing champion check; serving the base model",
                  flush=True)

    def move(self, gpu):
        """Save the conversation slot, restart on GPU/CPU (same adapter), restore the slot."""
        with self.lock:
            before = self._pinned
            try:
                self._post("/slots/0?action=save", {"filename": "riya.bin"})
                saved = True
            except Exception:
                saved = False
            self.stop()
            self.start(gpu)
            if saved and self._pinned == before:  # another variant (a fallback to base) never restores this slot
                try:
                    self._post("/slots/0?action=restore", {"filename": "riya.bin"})
                except Exception:
                    pass
        print(f"[riya] moved to {'GPU' if gpu else 'CPU (game mode)'}", flush=True)

    def switch(self, lora: str | None):
        """Serve another variant. No slot is saved or restored: a slot is only valid for the variant that made it."""
        with self.lock:
            self.wanted = lora
            usable = usable_adapter(lora)
            if usable == self.lora and self._identity() == self._decided:
                self.lora_failed = None
                return
            self.stop()
            self.lora, self.lora_failed = usable, None
            self.start(self.on_gpu)
        print("[riya] switched to Riya v1" if self._serve else "[riya] switched to the base model", flush=True)

    def refresh(self):
        """The wanted adapter's file or its champion record changed since the last launch decided (a file copied over
        the adopted one, the gate adopting a new one): relaunch, so the vetted-only rule decides again. No slot is
        restored, since the variant may differ."""
        with self.lock:
            if self._identity() == self._decided:
                return
            self.stop()
            self.start(self.on_gpu)
        print("[riya] the adapter changed; now serving " + ("Riya v1" if self._serve else "the base model"),
              flush=True)

    def step(self, policy: GameModePolicy, desired: Callable[[], str | None] | None = None,
             free_mb: Callable[[], int] | None = None, now: Callable[[], float] = time.time) -> None:
        """One watch tick: follow the wanted adapter (a failed read skips the switch this tick), then game mode."""
        if desired is not None:
            try:
                value = desired()
            except Exception:
                pass
            else:
                if value != self.wanted:
                    self.switch(value)
                else:
                    self.refresh()
        action = policy.update((free_mb or free_vram_mb)(), now())
        if action:
            self.move(gpu=action == "to_gpu")

    def watch(self, policy: GameModePolicy, every_s=3.0, desired=None):
        while self.proc and self.proc.poll() is None:
            self.step(policy, desired)
            time.sleep(every_s)

    def stop(self, timeout_s: float = 30.0):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout_s)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model")
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--ctx", type=int, default=16384)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--low_mb", type=int, default=800, help="go to CPU when free VRAM drops below this")
    p.add_argument("--high_mb", type=int, default=2500, help="return to GPU when free VRAM stays above this")
    p.add_argument("--lora", help="her fine-tuned adapter (a .gguf LoRA file)")
    p.add_argument("--follow-settings", action="store_true",
                   help="every 3 s read the local_adapter setting (read-only) and switch to it (an unset setting "
                        "means the base model, even with --lora)")
    p.add_argument("--db", default=str(brain.default_db()), help="the settings database for --follow-settings")
    p.add_argument("--print-lora", action="store_true",
                   help="print the adapter named in the settings if its file exists and the champion gate adopted "
                        "it (a matching record), then exit; starts nothing")
    a = p.parse_args(argv)
    db = Path(a.db)

    def setting():
        return brain.read_local_adapter(db)

    if a.print_lora:
        try:
            raw = setting()
        except Exception:
            raw = None
        if vetted_adapter(raw):  # only an adapter the champion gate adopted, as the file is now
            print(raw)
        return 0
    if not a.model:
        p.error("--model is required")
    wanted = a.lora
    if not wanted and a.follow_settings:
        try:
            wanted = setting()
        except Exception:
            wanted = None
    srv = RiyaServer(a.model, a.port, a.ctx, a.threads, lora=wanted)
    if wanted and not srv.lora:
        print(f"[riya] the adapter {wanted} was not found; serving the base model", flush=True)
    try:
        srv.start(gpu=free_vram_mb() >= a.high_mb)
        policy = GameModePolicy(a.low_mb, a.high_mb)
        policy.on_gpu = srv.on_gpu
        print(f"[riya] serving on :{a.port} ({'GPU' if srv.on_gpu else 'CPU'}, "
              f"{'Riya v1' if srv.serving else 'base model'})", flush=True)
        srv.watch(policy, desired=setting if a.follow_settings else None)
    finally:
        srv.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
