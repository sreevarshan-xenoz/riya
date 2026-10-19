"""Live check of her Health table: ask a running core for it and print it. Read-only.

  python -m evaluation.health_check [--url ws://127.0.0.1:8765/ws] [--timeout 15]

It connects to a core that is already running (no Origin header), waits for the end of the hello flow, sends exactly one
message, health_get, and prints the table from the reply: one line per row, the key source, then a VERDICT line. It
never sends anything else, so it never presses a Test or Connect button, never changes a setting and never touches a key
(the core only ever sends the mask, and this prints no raw messages). With no core it says so. Always exits 0; the
live run is not part of pytest.
"""
import argparse
import json
import sys
import time

WS_URL = "ws://127.0.0.1:8765/ws"
WAIT_S = 15.0
# A copy of the body's ROW_LABELS (body/src/health-logic.js), in health.ROWS order; tests/core/test_health_body_sync.py
# pins both.
LABELS = {
    "local": "Local brain",
    "online": "Online brain",
    "voice": "Voice",
    "mic": "Mic",
    "look": "Look",
    "eyes": "Eyes",
    "claude_code": "Claude Code",
    "colab": "Colab logs",
    "memory": "Memory",
    "autostart": "Start with Windows",
}


def _default_connect(url: str, **kw):
    from websockets.sync.client import connect
    return connect(url, **kw)


def _read_until(ws, kind: str, end: float) -> dict | None:
    """Frames from the socket until one of this type; None once the deadline passes."""
    while (left := end - time.monotonic()) > 0:
        frame = json.loads(ws.recv(timeout=left))
        if isinstance(frame, dict) and frame.get("type") == kind:
            return frame
    return None


def fetch(url: str, timeout_s: float, connect=None) -> dict | None:
    """The core's health message, or None when it can't connect or nothing arrives in time.

    Reads the hello flow up to its last frame ("actions"), sends health_get and reads until the health frame (a voices
    frame may come first). Sends nothing else.
    """
    connect = connect or _default_connect
    end = time.monotonic() + timeout_s
    try:
        with connect(url, open_timeout=min(5.0, timeout_s), max_size=None) as ws:
            if _read_until(ws, "actions", end) is None:
                return None
            ws.send(json.dumps({"type": "health_get"}))
            return _read_until(ws, "health", end)
    except Exception:  # no core, a closed socket or a timeout: all the same to a check script
        return None


def format_table(msg: dict) -> list[str]:
    """One line per row, then the key source and the verdict."""
    items = msg.get("items") or []
    lines = [f"{LABELS.get(it['key'], it['key']):<18} {it['status']:<5} {it['detail']}" for it in items]
    lines.append(f"Key: {msg.get('key_source') or 'none'}")
    counts = {s: sum(1 for it in items if it["status"] == s) for s in ("ok", "warn", "off")}
    lines.append(f"VERDICT: {counts['ok']} ok, {counts['warn']} warn, {counts['off']} off")
    return lines


def main(argv: list[str] | None = None, connect=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.health_check", description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=WS_URL, help="the core's WebSocket (default %(default)s)")
    ap.add_argument("--timeout", type=float, default=WAIT_S, help="seconds to wait for the table (default %(default)s)")
    args = ap.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):  # details can hold characters the console can't encode
            sys.stdout.reconfigure(errors="replace")
        msg = fetch(args.url, args.timeout, connect=connect)
        if msg is None:
            print(f"No core at {args.url}: start it with python -m riya.core")
        else:
            print("\n".join(format_table(msg)))
    except Exception as e:  # a check script: report, never fail the caller
        print(f"VERDICT: INCOMPLETE, error: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
