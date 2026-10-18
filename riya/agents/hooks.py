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
from collections.abc import Callable
from pathlib import Path

from fastapi import FastAPI, Request, Response

HOOK_HOST = "127.0.0.1"
HOOK_PORT = 8766
MAX_BODY = 256 * 1024
START_TIMEOUT_S = 5.0
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


def create_hook_app(token: str, enqueue: Callable[[dict], None]) -> FastAPI:
    """POST /hook. Checks run in order: Origin (403), token (401), size (413), then parse. Anything that passes them
    answers 200 with an empty body; a payload that isn't a JSON object is dropped. enqueue runs on the event loop, so it
    must not block (queue.SimpleQueue.put is fine)."""
    if not token.strip():
        raise ValueError("the hook token can't be empty")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    want = token.encode()

    @app.post("/hook")
    async def hook(request: Request) -> Response:
        if "origin" in request.headers:
            return Response(status_code=403)
        if not hmac.compare_digest(request.headers.get("x-riya-token", "").encode(), want):
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
                        port: int = HOOK_PORT):
    """Serves the receiver on 127.0.0.1:port in a daemon thread named riya-hooks. Returns (server, thread), or None
    after one warning line when it can't start (port taken, token unwritable): she runs on without agents."""
    try:
        import uvicorn
        token = load_or_create_token(token_path or default_token_path())
        server = uvicorn.Server(uvicorn.Config(create_hook_app(token, enqueue), host=HOOK_HOST, port=port,
                                               log_level="warning"))
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
        log.warning("hook receiver unavailable on %s:%d (%s); running without agents", HOOK_HOST, port, e)
        return None
