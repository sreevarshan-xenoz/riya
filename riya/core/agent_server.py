"""Free models behind a coding agent's local server (Kilo; any opencode-family CLI), as pool members.

Riya starts the CLI's `serve` once, lazily, at low priority, with its own data folders and an empty working folder
under her data dir, so it never sees the user's projects or their own CLI settings. Each reply is a fresh session in
the CLI's read-only "ask" agent with every tool switched off and her own system prompt; the session is deleted
after. A server that quietly retries (how these CLIs meet a usage limit) is caught from /session/status, aborted and
reported as a limit, so the pool moves on in about a second instead of waiting out a timeout.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from riya.core.models import ModelUnavailable
from riya.paths import data_dir

log = logging.getLogger("riya.agent_server")
START_TIMEOUT_S = 30.0
REPLY_TIMEOUT_S = 30.0
POLL_S = 0.5
_LISTENING = re.compile(r"listening on (http://127\.0\.0\.1:\d+)")


def _http(base: str, method: str, path: str, body=None, timeout: float = 10.0):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data, {"Content-Type": "application/json"}, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else None


class AgentServer:
    """One `<cli> serve` process, started on first use and restarted if it dies."""

    def __init__(self, cli: str, root: Path | None = None, popen=subprocess.Popen,
                 which: Callable[[str], str | None] | None = None):
        self.cli, self.root = cli, Path(root or data_dir() / "agents" / cli)
        self.popen, self._which = popen, which
        self.proc: subprocess.Popen | None = None
        self.base: str | None = None
        self.tools_off: dict[str, bool] = {}
        self._lock = threading.Lock()

    def which(self, name: str) -> str | None:
        return (self._which or shutil.which)(name)

    @property
    def available(self) -> bool:
        return self.which(self.cli) is not None

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for var, sub in (("XDG_DATA_HOME", "data"), ("XDG_CONFIG_HOME", "config"), ("XDG_CACHE_HOME", "cache"),
                         ("XDG_STATE_HOME", "state")):
            env[var] = str(self.root / sub)
        return env

    def url(self) -> str:
        with self._lock:
            if self.proc is not None and self.proc.poll() is None and self.base:
                return self.base
            exe = self.which(self.cli)
            if exe is None:
                raise ModelUnavailable(f"{self.cli}: not installed")
            work = self.root / "work"
            work.mkdir(parents=True, exist_ok=True)
            self.proc = self.popen(["nice", "-n", "10", exe, "serve", "--port", "0", "--hostname", "127.0.0.1"],
                                   cwd=work, env=self._env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL, text=True)
            self.base = self._wait_listening()
            try:
                self.tools_off = {t: False for t in _http(self.base, "GET", "/experimental/tool/ids") or []}
            except (OSError, ValueError):
                self.tools_off = {}
            log.info("%s serve is up", self.cli)
            return self.base

    def _wait_listening(self) -> str:
        found: list[str] = []

        def read():
            for line in self.proc.stdout:
                m = _LISTENING.search(line)
                if m and not found:
                    found.append(m.group(1))
        threading.Thread(target=read, daemon=True, name=f"riya-{self.cli}-serve").start()
        deadline = time.monotonic() + START_TIMEOUT_S
        while time.monotonic() < deadline:
            if found:
                return found[0]
            if self.proc.poll() is not None:
                raise ModelUnavailable(f"{self.cli}: serve exited with {self.proc.returncode}")
            time.sleep(0.1)
        self.stop()
        raise ModelUnavailable(f"{self.cli}: serve didn't start in {START_TIMEOUT_S:.0f} s")

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc, self.base = None, None


def flatten(messages: list[dict]) -> tuple[str, str]:
    """Riya's chat messages as one system prompt and one prompt (each reply is a fresh session). Image parts are
    dropped; these members are never her eyes."""
    def text(content) -> str:
        if isinstance(content, str):
            return content
        return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict) and p.get("type") == "text")
    system = "\n\n".join(text(m.get("content")) for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    if len(turns) == 1:
        return system, text(turns[0].get("content"))
    who = {"user": "User", "assistant": "Riya"}
    lines = [f"{who.get(m.get('role'), m.get('role'))}: {text(m.get('content'))}" for m in turns]
    return system, "\n".join(lines) + "\nRiya:"


class ServerAgentModel:
    """A ChatModel answered by an agent server's model (e.g. Kilo's kilo-auto/free)."""

    def __init__(self, name: str, server: AgentServer, provider_id: str, model_id: str, agent: str = "ask",
                 timeout_s: float = REPLY_TIMEOUT_S, clock: Callable[[], float] = time.monotonic):
        self.name, self.server, self.provider_id, self.model_id, self.agent = name, server, provider_id, model_id, agent
        self.model = f"{server.cli}:{model_id}"
        self.timeout_s, self.clock = timeout_s, clock
        self.finish_reason: str | None = None

    def stream(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.8) -> Iterator[str]:
        self.finish_reason = None
        try:
            base = self.server.url()
            sid = _http(base, "POST", "/session", {})["id"]
        except ModelUnavailable:
            raise
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ModelUnavailable(f"{self.model}: {type(e).__name__}") from e
        try:
            text = self._ask(base, sid, messages)
        finally:
            try:
                _http(base, "DELETE", f"/session/{sid}")
            except (OSError, ValueError):
                pass
        self.finish_reason = "stop"
        if text:
            yield text

    def _ask(self, base: str, sid: str, messages: list[dict]) -> str:
        system, prompt = flatten(messages)
        body = {"parts": [{"type": "text", "text": prompt}], "agent": self.agent, "tools": self.server.tools_off,
                "model": {"providerID": self.provider_id, "modelID": self.model_id}}
        if system:
            body["system"] = system
        result: dict = {}

        def post():
            try:
                result["reply"] = _http(base, "POST", f"/session/{sid}/message", body, timeout=self.timeout_s + 5)
            except Exception as e:  # reported below
                result["error"] = e
        worker = threading.Thread(target=post, daemon=True, name="riya-agent-reply")
        worker.start()
        deadline = self.clock() + self.timeout_s
        while worker.is_alive():
            worker.join(POLL_S)
            if not worker.is_alive():
                break
            status = self._status(base, sid)
            if status.get("type") == "retry":  # a limit or an outage: the CLI would retry for minutes
                self._abort(base, sid)
                why = status.get("message") or "retrying"
                raise ModelUnavailable(f"{self.model}: {why[:120]}", detail=why)
            if self.clock() > deadline:
                self._abort(base, sid)
                raise ModelUnavailable(f"{self.model}: no reply in {self.timeout_s:.0f} s")
        if "error" in result:
            e = result["error"]
            status = e.code if isinstance(e, urllib.error.HTTPError) else None
            raise ModelUnavailable(f"{self.model}: {type(e).__name__}", status=status) from e
        reply = result.get("reply") or {}
        err = (reply.get("info") or {}).get("error")
        if err:
            why = json.dumps(err)[:300]
            raise ModelUnavailable(f"{self.model}: {why[:120]}", detail=why)
        return "".join(p.get("text", "") for p in reply.get("parts") or [] if p.get("type") == "text").strip()

    @staticmethod
    def _status(base: str, sid: str) -> dict:
        try:
            return (_http(base, "GET", "/session/status", timeout=3) or {}).get(sid) or {}
        except (OSError, ValueError, AttributeError):
            return {}

    @staticmethod
    def _abort(base: str, sid: str) -> None:
        try:
            _http(base, "POST", f"/session/{sid}/abort", {}, timeout=3)
        except (OSError, ValueError):
            pass


KILO_MODELS = ("kilo-auto/free", "nvidia/nemotron-3-ultra-550b-a55b:free", "reflection/beam-free")
_servers: dict[str, AgentServer] = {}


def kilo_members(name: str) -> list:
    """Kilo's free models as slow pool members (no key; one shared server for the process)."""
    from riya.core.pool import Member
    server = _servers.setdefault("kilo", AgentServer("kilo"))
    if not server.available:
        return []
    return [Member(ServerAgentModel(name, server, "kilo", m), tools=False, slow=True, group="kilo", provider="kilo")
            for m in KILO_MODELS]
