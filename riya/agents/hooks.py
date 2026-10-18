"""The hook receiver: a tiny localhost FastAPI app that Claude Code's `http` hooks POST to. It only authenticates,
bounds and enqueues; the core drains the queue into the AgentHub on its own thread."""
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response

HOOK_HOST = "127.0.0.1"
HOOK_PORT = 8766
MAX_BODY = 256 * 1024
START_TIMEOUT_S = 5.0
RETRY_S = 60.0  # a busy port is tried again this often until it binds
REJECT_WINDOW_S = 24 * 3600
REJECT_KEEP = 10_000  # wrong-token hits remembered at most (a flood saturates the count, never the memory)
log = logging.getLogger("riya.hooks")


def default_token_path() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Riya" / "hook_token"


def load_or_create_token(path: Path | str) -> str:
    """The shared secret (32 random bytes, hex): written once, then reused."""
    path = Path(path)
    try:
        if token := path.read_text(encoding="utf-8").strip():
            return token
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    path.write_text(token, encoding="utf-8")
    return token


class RejectCounter:
    """Wrong-token (401) requests: the receiver's thread counts them, the core's tick reads recent() (thread-safe)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock, self._lock, self._hits = clock, threading.Lock(), deque(maxlen=REJECT_KEEP)

    def hit(self) -> None:
        with self._lock:
            self._hits.append(self.clock())

    def recent(self) -> int:
        """Hits in the last 24 h."""
        with self._lock:
            cutoff = self.clock() - REJECT_WINDOW_S
            while self._hits and self._hits[0] <= cutoff:
                self._hits.popleft()
            return len(self._hits)


def create_hook_app(token: str, enqueue: Callable[[dict], None], rejected: RejectCounter | None = None) -> FastAPI:
    """POST /hook. Checks run in order: Origin (403), token (401, counted in rejected), size (413), then parse.
    Anything that passes them answers 200 with an empty body; a payload that isn't a JSON object is dropped. enqueue
    runs on the event loop, so it must not block (queue.SimpleQueue.put is fine)."""
    if not token.strip():
        raise ValueError("the hook token can't be empty")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    want = token.encode()

    @app.post("/hook")
    async def hook(request: Request) -> Response:
        if "origin" in request.headers:
            return Response(status_code=403)
        if not hmac.compare_digest(request.headers.get("x-riya-token", "").encode(), want):
            if rejected is not None:
                rejected.hit()
            return Response(status_code=401)
        declared = request.headers.get("content-length", "")
        if re.fullmatch(r"[0-9]+", declared) and int(declared) > MAX_BODY:  # else rely on the counted-bytes limit
            return Response(status_code=413)
        body = bytearray()
        async for chunk in request.stream():  # count the bytes actually read, not what the header claims
            body += chunk
            if len(body) > MAX_BODY:
                return Response(status_code=413)
        try:
            payload = json.loads(bytes(body))
        except (ValueError, RecursionError):  # bad UTF-8, bad JSON, absurdly deep nesting
            payload = None
        if isinstance(payload, dict):
            try:
                enqueue(payload)
            except Exception:
                log.exception("couldn't enqueue a hook event")
        return Response(status_code=200)

    return app


def _serve(server) -> None:
    try:
        server.run()
    except SystemExit:  # uvicorn exits this way when it can't bind; the starter notices the dead thread
        pass


def start_hook_receiver(enqueue: Callable[[dict], None], token_path: Path | str | None = None,
                        port: int = HOOK_PORT, rejected: RejectCounter | None = None, warn: bool = True):
    """Serves the receiver on 127.0.0.1:port in a daemon thread named riya-hooks. Returns (server, thread), or None
    after one log line (a warning, or debug when warn is off) when it can't start (port taken, token unwritable)."""
    try:
        import uvicorn
        token = load_or_create_token(token_path or default_token_path())
        server = uvicorn.Server(uvicorn.Config(create_hook_app(token, enqueue, rejected), host=HOOK_HOST, port=port,
                                               log_level="warning"))  # shared uvicorn loggers: keep the level
        thread = threading.Thread(target=_serve, args=(server,), name="riya-hooks", daemon=True)
        thread.start()
        deadline = time.monotonic() + START_TIMEOUT_S
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                server.should_exit = True
                raise OSError(f"the hook server didn't start on {HOOK_HOST}:{port}")
            time.sleep(0.02)
        return server, thread
    except Exception as e:
        log.log(logging.WARNING if warn else logging.DEBUG,
                "hook receiver unavailable on %s:%d (%s); running without agents for now", HOOK_HOST, port, e)
        return None


class HookReceiver:
    """The receiver, kept up: start() serves it now or, when the port is busy (a previous core still letting go of
    it), retries every RETRY_S in a daemon thread until it binds or stop(). ok: serving now. rejected: the 401s."""

    def __init__(self, enqueue: Callable[[dict], None], token_path: Path | str | None = None, port: int = HOOK_PORT,
                 retry_s: float = RETRY_S):
        self.enqueue, self.token_path, self.port, self.retry_s = enqueue, token_path, port, retry_s
        self.rejected = RejectCounter()
        self._stop = threading.Event()
        self._serving = None  # (server, thread) once bound
        self._retry: threading.Thread | None = None

    @property
    def ok(self) -> bool:
        serving = self._serving
        return serving is not None and serving[1].is_alive() and not self._stop.is_set()

    def start(self) -> "HookReceiver":
        if not self._try(warn=True):
            self._retry = threading.Thread(target=self._retry_loop, name="riya-hooks-retry", daemon=True)
            self._retry.start()
        return self

    def _try(self, warn: bool) -> bool:
        self._serving = start_hook_receiver(self.enqueue, self.token_path, self.port, self.rejected, warn=warn)
        return self._serving is not None

    def _retry_loop(self) -> None:
        while not self._stop.wait(self.retry_s):
            if self._try(warn=False):
                log.info("hook receiver listening on %s:%d", HOOK_HOST, self.port)
                return

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._retry is not None:
            self._retry.join(timeout)
        if (serving := self._serving) is not None:
            serving[0].should_exit = True
            serving[1].join(timeout)
