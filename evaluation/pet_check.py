"""Live check of Little Riya's reactions: post a scripted Claude Code session to a running core's hook receiver and watch
the agent_activity and presence messages come back over the WebSocket, as her bubbles and poses would.

  python -m evaluation.pet_check [--spawn]

By default it talks to a core that is already running (ws://127.0.0.1:8765/ws with no Origin header, hooks on
127.0.0.1:8766, the token read from the default path). With --spawn it starts its own core for the run (temp --db,
--no-voice, --hash-embed, no online key, so no network and no screen glances), asks it to quit over the socket and waits
up to 10 s; only that one child is ever terminated. It POSTs straight to the receiver: it never goes through real Claude
Code hooks and never touches Claude Code's settings.

The session is "riya-pet-check-<random>" in "C:/riya-pet-check": UserPromptSubmit, Edit, Bash (npm test), a
PermissionRequest for it, then Stop, one every 2 s (past her 1.5 s bubble gap, so none is held). Prints each bubble as
"<time> <kind>: <label>" and each presence change, then a VERDICT line: ok when the five expected bubbles came back in
order and none was over 40 chars. Always exits 0; not part of pytest. Run it with the body open to watch her react.
"""
import argparse
import json
import secrets
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

from evaluation.agents_check import (CORE_READY_S, HOOK_PORT, WS_PORT, WS_URL, connect_ws, port_busy, post, spawn_core,
                                     stamp, stop_core, wait_for_token)
from riya.agents.hooks import default_token_path

CWD = "C:/riya-pet-check"
SPACING_S = 2.0
WAIT_S = 10.0  # after the last post: how long the last bubble may take (the core ticks once a second)
LABEL_MAX = 40
SCRIPT: list[tuple[str, dict]] = [
    ("UserPromptSubmit", {"prompt": "fix the tests"}),
    ("PreToolUse", {"tool_name": "Edit", "tool_input": {"file_path": "C:/riya-pet-check/engine.py"}}),
    ("PreToolUse", {"tool_name": "Bash", "tool_input": {"command": "npm test"}}),
    ("PermissionRequest", {"tool_name": "Bash", "tool_input": {"command": "npm test"}}),
    ("Stop", {"last_assistant_message": "All tests pass."}),
]
EXPECTED = [("think", "thinking…"), ("edit", "editing engine.py"), ("run", "running tests"),
            ("wave", "needs you! running tests"), ("done", "all done!")]


def payloads(session: str) -> list[dict]:
    return [{"hook_event_name": ev, "session_id": session, "cwd": CWD, **extra} for ev, extra in SCRIPT]


def verdict(seen: list[tuple[str, str]]) -> str:
    """ok when EXPECTED is an ordered subsequence of seen (other bubbles may sit between) and no label is over 40."""
    if any(len(label) > LABEL_MAX for _, label in seen):
        return f"VERDICT: a label over {LABEL_MAX} chars"
    it = iter(seen)
    for pair in EXPECTED:
        if not any(got == pair for got in it):
            return f"VERDICT: missing {pair}"
    return "VERDICT: ok"


class Watcher:
    """Reads the socket on its own thread; keeps this session's bubbles and prints them with every presence change."""

    def __init__(self, ws, session: str):
        self.ws, self.session, self.lock = ws, session, threading.Lock()
        self.seen: list[tuple[str, str]] = []
        self.closed = False
        threading.Thread(target=self._run, name="pet-check-ws", daemon=True).start()

    def _run(self) -> None:
        try:
            for raw in self.ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "agent_activity" and msg.get("session") == self.session:
                    with self.lock:
                        self.seen.append((msg["kind"], msg["label"]))
                    print(f"{stamp()} {msg['kind']}: {msg['label']}", flush=True)
                elif kind == "presence":
                    print(f"{stamp()} presence: {msg.get('state')}", flush=True)
                elif kind == "error":
                    print(f"{stamp()} core error: {msg.get('message')}", flush=True)
        except Exception as e:
            if not self.closed:
                print(f"{stamp()} socket closed ({type(e).__name__}: {e})", flush=True)

    def snapshot(self) -> list[tuple[str, str]]:
        with self.lock:
            return list(self.seen)

    def wait_until_ok(self, timeout_s: float) -> None:
        end = time.monotonic() + timeout_s
        while verdict(self.snapshot()) != "VERDICT: ok" and time.monotonic() < end:
            time.sleep(0.1)


def run(spawn: bool, *, connect, post, token_for, sleep, wait_s: float) -> str:
    tmp, proc, log, ws, watcher = None, None, None, None, None
    try:
        if spawn:
            busy = [p for p in (WS_PORT, HOOK_PORT) if port_busy(p)]
            if busy:
                return f"VERDICT: not run: port(s) {busy} are already in use (a core is running: drop --spawn)"
            tmp = Path(tempfile.mkdtemp(prefix="riya_pet_"))
            proc, log = spawn_core(tmp)
            print(f"{stamp()} spawned a core (pid {proc.pid}), log in {tmp / 'core.log'}")
        ws = connect(CORE_READY_S if spawn else 5.0, proc)
        if ws is None:
            return f"VERDICT: not run: no core answered on {WS_URL}" + ("" if spawn else " (start one, or use --spawn)")
        print(f"{stamp()} connected to {WS_URL}")
        session = f"riya-pet-check-{secrets.token_hex(4)}"
        watcher = Watcher(ws, session)
        token = token_for(CORE_READY_S if spawn else 1.0)
        if token is None:
            return f"VERDICT: not run: no hook token at {default_token_path()}"
        if spawn:  # the hook receiver starts just after the socket
            end = time.monotonic() + 30
            while not port_busy(HOOK_PORT) and time.monotonic() < end:
                time.sleep(0.3)
        sleep(1.5)  # the hello flow: settings, state, the (empty) agents list
        for i, payload in enumerate(payloads(session)):
            if i:
                sleep(SPACING_S)
            print(f"{stamp()} POST {payload['hook_event_name']} -> {post(token, payload)}", flush=True)
        watcher.wait_until_ok(wait_s)
        return verdict(watcher.snapshot())
    finally:
        if watcher is not None:
            watcher.closed = True
        if proc is not None:
            stop_core(proc, ws)
            if log is not None:
                log.close()
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def main(argv: list[str] | None = None, *, connect=connect_ws, post=post, token_for=wait_for_token, sleep=time.sleep,
         wait_s: float = WAIT_S) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.pet_check", description=__doc__.splitlines()[0])
    ap.add_argument("--spawn", action="store_true", help="start (and quit) a core of its own instead of using a running one")
    args = ap.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):  # labels hold characters the console may not encode ("thinking…")
            sys.stdout.reconfigure(errors="replace")
        print(run(args.spawn, connect=connect, post=post, token_for=token_for, sleep=sleep, wait_s=wait_s))
    except Exception as e:  # a check script: report, never fail the caller
        print(f"VERDICT: error: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
