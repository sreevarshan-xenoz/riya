"""Live check of her agent babysitting: post a scripted Claude Code session to a running core's hook receiver and watch
what comes back over the WebSocket.

  python -m evaluation.agents_check [--spawn]

By default it talks to a core that is already running (ws://127.0.0.1:8765/ws with no Origin header, hooks on
127.0.0.1:8766, the token read from the default path). With --spawn it starts its own core for the run (temp --db,
--no-voice, --hash-embed, no online key, so no network and no screen glances), asks it to quit over the socket and waits
up to 10 s; only that one child is ever terminated. It never touches Claude Code's settings or hooks.

The session is "riya-agents-check-<random>" in "C:/riya-agents-check": UserPromptSubmit, then PermissionRequest (Bash,
npm test), then Stop ("All tests pass."), spaced for the 1 s tick and the 20 s report spacing (about 60 s in all).
Prints the agents snapshots and the report lines it sees, then a VERDICT line. Always exits 0; not part of pytest.
"""
import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from riya.agents.hooks import HOOK_HOST, HOOK_PORT, default_token_path

WS_PORT = 8765
WS_URL = f"ws://127.0.0.1:{WS_PORT}/ws"
HOOK_URL = f"http://{HOOK_HOST}:{HOOK_PORT}/hook"
CWD = "C:/riya-agents-check"
REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_READY_S = 90.0  # a spawned core loads its stores first
EVENT_WAIT_S = 25.0  # the 1 s tick plus slack: how long an event may take to show up
FINISHED_WAIT_S = 8.0  # a finished report is held for a break, so it isn't waited for long
REPORT_SPACING_S = 21.0  # her minimum spacing between spoken reports is 20 s
QUIT_WAIT_S = 10.0
t0 = time.monotonic()


def stamp() -> str:
    return f"[{time.monotonic() - t0:5.1f}s]"


class Watcher:
    """Reads the socket on its own thread; keeps the agents snapshots and the report lines it sees."""

    def __init__(self, ws):
        self.ws, self.lock = ws, threading.Lock()
        self.snapshots: list[dict] = []
        self.reports: list[tuple[float, str]] = []
        self.closed = self.finished_reported = False
        threading.Thread(target=self._run, name="agents-check-ws", daemon=True).start()

    def _run(self) -> None:
        try:
            for raw in self.ws:
                msg = json.loads(raw)
                kind = msg.get("type")
                if kind == "agents":
                    with self.lock:
                        self.snapshots.append(msg)
                    items = msg.get("items", [])
                    line = ", ".join(f"{a['name']}={a['status']}{'!' if a['needs_attention'] else ''}"
                                     f" ({a['detail']})" for a in items) or "no agents"
                    print(f"{stamp()} agents: {line}", flush=True)
                elif kind == "say" and msg.get("kind") == "report":
                    with self.lock:
                        self.reports.append((time.monotonic(), msg["text"]))
                    print(f"{stamp()} report (spoken={msg.get('spoken')}): {msg['text']}", flush=True)
                elif kind == "error":
                    print(f"{stamp()} core error: {msg.get('message')}", flush=True)
        except Exception as e:
            if not self.closed:
                print(f"{stamp()} socket closed ({type(e).__name__}: {e})", flush=True)

    def statuses(self, name_part: str) -> set[str]:
        with self.lock:
            return {a["status"] for s in self.snapshots for a in s.get("items", []) if name_part in a["name"]}

    def report_count(self) -> int:
        with self.lock:
            return len(self.reports)

    def last_report_at(self) -> float | None:
        with self.lock:
            return self.reports[-1][0] if self.reports else None

    def wait_for(self, done, timeout_s: float) -> bool:
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            if done():
                return True
            time.sleep(0.2)
        return done()


def port_busy(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def post(token: str, payload: dict, timeout_s: float = 10.0) -> str:
    """POST one hook event; the status or the failure, as text. Never raises."""
    req = urllib.request.Request(HOOK_URL, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "X-Riya-Token": token})
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception as e:
        return f"failed ({type(e).__name__}: {e})"


def connect_ws(deadline_s: float, proc=None):
    """The WebSocket (no Origin header), retried until the core is up; None if it never is."""
    from websockets.sync.client import connect
    end = time.monotonic() + deadline_s
    while True:
        try:
            return connect(WS_URL, open_timeout=5, max_size=None)
        except Exception:
            if (proc is not None and proc.poll() is not None) or time.monotonic() > end:
                return None
            time.sleep(0.5)


def wait_for_token(deadline_s: float) -> str | None:
    end = time.monotonic() + deadline_s
    while True:
        try:
            if token := default_token_path().read_text(encoding="utf-8").strip():
                return token
        except OSError:
            pass
        if time.monotonic() > end:
            return None
        time.sleep(0.3)


def spawn_core(tmp: Path):
    """The child core on a temp db: silent, hash embedder, and no online key (an empty variable is never overridden by
    .env), so it makes no network calls and never glances at the screen. Its output goes to a log file in tmp."""
    env = dict(os.environ, RIYA_ONLINE_API_KEY="", PYTHONIOENCODING="utf-8")
    log = open(tmp / "core.log", "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "riya.core", "--db", str(tmp / "riya.db"), "--no-voice",
                             "--hash-embed"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc, log


def stop_core(proc, ws) -> None:
    """quit_core over the socket, then wait up to QUIT_WAIT_S; only if it's still alive, terminate that one child."""
    try:
        if ws is not None:
            ws.send(json.dumps({"type": "quit_core"}))
    except Exception as e:
        print(f"note: couldn't send quit_core ({type(e).__name__}: {e})")
    try:
        proc.wait(timeout=QUIT_WAIT_S)
        print(f"{stamp()} the spawned core exited (code {proc.returncode})")
    except subprocess.TimeoutExpired:
        print(f"{stamp()} the spawned core didn't exit within {QUIT_WAIT_S:.0f} s after quit_core: terminating it")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("note: it is still alive after terminate; left as it is")


def script(token: str, watcher: Watcher, session: str) -> None:
    """The scripted session: each post waits for what it should cause before the next one goes out."""
    base = {"session_id": session, "cwd": CWD}
    name = "riya-agents-check"

    print(f"{stamp()} POST UserPromptSubmit -> " + post(token, {
        **base, "hook_event_name": "UserPromptSubmit", "prompt": "run the tests"}))
    watcher.wait_for(lambda: "busy" in watcher.statuses(name), EVENT_WAIT_S)
    time.sleep(3)

    before = watcher.report_count()
    print(f"{stamp()} POST PermissionRequest (Bash: npm test) -> " + post(token, {
        **base, "hook_event_name": "PermissionRequest", "tool_name": "Bash", "tool_input": {"command": "npm test"}}))
    watcher.wait_for(lambda: "waiting" in watcher.statuses(name), EVENT_WAIT_S)
    watcher.wait_for(lambda: watcher.report_count() > before, EVENT_WAIT_S)
    if (at := watcher.last_report_at()) is not None:  # let the report spacing pass so the next one can be said
        time.sleep(max(3.0, REPORT_SPACING_S - (time.monotonic() - at)))
    else:
        time.sleep(3)

    before = watcher.report_count()
    print(f"{stamp()} POST Stop (All tests pass.) -> " + post(token, {
        **base, "hook_event_name": "Stop", "last_assistant_message": "All tests pass."}))
    watcher.wait_for(lambda: "finished" in watcher.statuses(name), EVENT_WAIT_S)
    # a finished agent is not urgent: she holds its report for a break or a talking turn, so a quiet wait is normal
    if not watcher.wait_for(lambda: watcher.report_count() > before, FINISHED_WAIT_S):
        print(f"{stamp()} no report for the finish yet: by design it waits for a break or a talking turn")
    watcher.finished_reported = watcher.report_count() > before
    time.sleep(1)


def verdict(watcher: Watcher) -> str:
    seen = watcher.statuses("riya-agents-check")
    missing = [f"{s} in the agents list" for s in ("waiting", "finished") if s not in seen]
    if not watcher.report_count():
        missing.append("a report line for the waiting agent (urgent: she says it unless away)")
    if missing:
        return "VERDICT: INCOMPLETE, missing: " + "; ".join(missing)
    held = "" if watcher.finished_reported else ", the finish report held for a break"
    return f"VERDICT: PASS, saw waiting + finished and {watcher.report_count()} report line(s){held}"


def run(spawn: bool) -> str:
    tmp, proc, log, ws, watcher = None, None, None, None, None
    try:
        if spawn:
            busy = [p for p in (WS_PORT, HOOK_PORT) if port_busy(p)]
            if busy:
                return f"VERDICT: INCOMPLETE, not run: port(s) {busy} are already in use (a core is running: drop --spawn)"
            tmp = Path(tempfile.mkdtemp(prefix="riya_agents_"))
            proc, log = spawn_core(tmp)
            print(f"{stamp()} spawned a core (pid {proc.pid}), log in {tmp / 'core.log'}")
        ws = connect_ws(CORE_READY_S if spawn else 5.0, proc)
        if ws is None:
            return f"VERDICT: INCOMPLETE, not run: no core answered on {WS_URL}" + ("" if spawn else " (start one, or use --spawn)")
        print(f"{stamp()} connected to {WS_URL}")
        watcher = Watcher(ws)
        token = wait_for_token(CORE_READY_S if spawn else 1.0)
        if token is None:
            return f"VERDICT: INCOMPLETE, not run: no hook token at {default_token_path()}"
        if spawn:  # the hook receiver starts just after the socket
            end = time.monotonic() + 30
            while not port_busy(HOOK_PORT) and time.monotonic() < end:
                time.sleep(0.3)
        time.sleep(1.5)  # the hello flow: settings, state, the (empty) agents list
        script(token, watcher, f"riya-agents-check-{secrets.token_hex(4)}")
        return verdict(watcher)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evaluation.agents_check", description=__doc__.splitlines()[0])
    ap.add_argument("--spawn", action="store_true", help="start (and quit) a core of its own instead of using a running one")
    args = ap.parse_args(argv)
    try:
        if hasattr(sys.stdout, "reconfigure"):  # details can hold characters the console can't encode
            sys.stdout.reconfigure(errors="replace")
        print(run(args.spawn))
    except Exception as e:  # a check script: report, never fail the caller
        print(f"VERDICT: INCOMPLETE, error: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
