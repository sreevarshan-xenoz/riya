"""Pure state machine for the user's Claude Code sessions and Colab runs (spec §4.2)."""
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import PureWindowsPath
from typing import Literal

STALE_S = 30 * 60        # busy or waiting with no events this long -> unknown
REMOVE_S = 10 * 60       # ended agents are dropped this long after SessionEnd
TOOL_FAIL_LIMIT = 3      # consecutive tool failures that earn the "in a row" detail (never failed: only StopFailure)
SNAPSHOT_MAX = 20
MAX_AGENTS = 200         # hard cap on tracked agents (least recently heard from is evicted)
ATTENTION = {"waiting", "failed", "lost", "stalled"}
CLAUDE_EVENTS = {"UserPromptSubmit", "PermissionRequest", "Notification", "Stop", "StopFailure", "PostToolUse",
                 "PostToolUseFailure", "SubagentStop", "SessionStart", "SessionEnd"}
_COLAB = {"running": "running", "done": "finished", "fail": "failed", "lost": "lost", "stalled": "stalled"}


_VALUE = r"""(?:"[^"]*"|'[^']*'|[^\s"'&;,]+)"""
_SECRETS = [  # (pattern, replacement): a secret-looking value becomes ***, its label stays
    (re.compile(r"\b(?:sk-(?:ant-)?|gh[opusr]_|github_pat_|xox[abpr]-)[A-Za-z0-9_\-]{8,}"), "***"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "***"),
    (re.compile(r"(?i)\b(authorization\s*:\s*)(?:(?:bearer|basic|token)\s+)?" + _VALUE), r"\1***"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=\-]+"), r"\1***"),
    (re.compile(r"(?i)(password|passwd|pwd|token|secret|api[_-]?key)([\"']?\s*[=:]\s*)" + _VALUE), r"\1\2***"),
    (re.compile(r"(?i)(--(?:password|token)(?:\s+|=))" + _VALUE), r"\1***"),
]


def redact(text: str) -> str:
    """Secret-looking values (API keys and tokens, Bearer/Authorization values, password=/token=/--password ...)
    replaced with ***."""
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def summarize(text, n: int = 80) -> str:
    """One clean line of at most n chars: control chars stripped, whitespace collapsed, secrets redacted (before the
    cut), ellipsis on a cut."""
    if n <= 0:
        return ""
    s = text if isinstance(text, str) else str(text)
    out = []
    for ch in s[:4096]:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat in ("Zl", "Zp"):
            out.append(" ")
        elif cat[0] != "C":
            out.append(ch)
    s = redact(" ".join("".join(out).split()))
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _tool_summary(tool_input) -> str:
    """Short text for a tool_input: a dict gives `command`, else `file_path`, else its compact JSON."""
    if isinstance(tool_input, dict):
        for key in ("command", "file_path"):
            if isinstance(tool_input.get(key), str) and tool_input[key].strip():
                return summarize(tool_input[key])
    if isinstance(tool_input, str):
        return summarize(tool_input)
    if tool_input is None:
        return ""
    try:
        return summarize(json.dumps(tool_input, separators=(",", ":"), ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return summarize(tool_input)


@dataclass
class AgentState:
    id: str
    kind: str
    name: str
    status: str
    since: float
    last_event_at: float
    detail: str = ""
    needs_attention: bool = False
    tool_errors: int = 0


@dataclass
class AgentEvent:
    kind: Literal["finished", "failed", "waiting", "lost", "stalled"]
    agent_id: str
    name: str
    text: str
    error: str | None = None  # a StopFailure's error type ("rate_limit"); None for every other event


class AgentHub:
    def __init__(self, clock):
        self.clock = clock
        self._agents: dict[str, AgentState] = {}

    # -- internals -----------------------------------------------------------------------------
    def _get(self, aid: str, kind: str, name: str, status: str) -> AgentState:
        a = self._agents.get(aid)
        if a is None:
            if len(self._agents) >= MAX_AGENTS:
                del self._agents[min(self._agents.values(), key=lambda x: x.last_event_at).id]
            now = self.clock.now()
            a = self._agents[aid] = AgentState(aid, kind, name, status, now, now)
        return a

    def _set(self, a: AgentState, status: str, detail: str | None = None) -> bool:
        """Move to status; True only on a real transition. Always refreshes last_event_at and detail."""
        now = self.clock.now()
        a.last_event_at = now
        if detail is not None:
            a.detail = detail
        if a.status == status:
            return False
        a.status, a.since = status, now
        a.needs_attention = status in ATTENTION
        return True

    @staticmethod
    def _event(a: AgentState) -> AgentEvent:
        return AgentEvent(a.status, a.id, a.name, summarize(a.detail))  # type: ignore[arg-type]

    # -- Claude --------------------------------------------------------------------------------
    def on_claude_event(self, payload: dict) -> list[AgentEvent]:
        if not isinstance(payload, dict):
            return []
        sid, ev = payload.get("session_id"), payload.get("hook_event_name")
        if not isinstance(sid, str) or not sid or len(sid) > 128 or ev not in CLAUDE_EVENTS:
            return []
        cwd = payload.get("cwd")
        folder = summarize(PureWindowsPath(cwd).name, 40) if isinstance(cwd, str) and cwd else ""
        name = f"{folder or 'claude'} {summarize(sid[:6], 6)}"
        a = self._agents.get(sid)
        if a is None:
            # sessions may predate Riya; a subagent stop or a tool that ran means a turn is running
            a = self._get(sid, "claude", name, "busy" if ev in ("SubagentStop", "PostToolUse") else "idle")
        elif folder:
            a.name = name
        if a.status == "ended" and ev not in ("UserPromptSubmit", "SessionStart"):
            return []
        a.last_event_at = self.clock.now()
        return getattr(self, "_ev_" + ev)(a, payload) or []

    def _ev_UserPromptSubmit(self, a, p):
        a.tool_errors = 0
        self._set(a, "busy", summarize(p.get("prompt", "")))

    def _ev_SessionStart(self, a, p):
        if a.status not in ("busy", "waiting", "failed"):  # a mid-turn compact must not look idle
            self._set(a, "idle", "")

    def _ev_SessionEnd(self, a, p):
        self._set(a, "ended", "")

    def _ev_SubagentStop(self, a, p):
        pass  # only proves the session is alive

    def _ev_PermissionRequest(self, a, p):
        tool = summarize(p.get("tool_name", ""), 40)
        summary = _tool_summary(p.get("tool_input"))
        detail = summarize(f"{tool}: {summary}" if summary else tool)
        old = a.detail if a.status == "waiting" else None
        if self._set(a, "waiting", detail) or old != detail:  # same tool asked again is not news
            return [self._event(a)]

    def _ev_Notification(self, a, p):
        kind = p.get("notification_type")
        if kind == "idle_prompt":
            if a.status not in ("failed", "waiting"):  # the idle nag must not hide a failure
                self._set(a, "idle", "")
        elif kind == "permission_prompt":
            if a.status == "waiting":
                return []  # the PermissionRequest already named the tool
            tool = summarize(p.get("tool_name", ""), 40)
            summary = _tool_summary(p.get("tool_input")) if tool else ""
            self._set(a, "waiting", summarize(f"{tool}: {summary}" if summary else tool or p.get("message", "")))
            return [self._event(a)]

    def _ev_Stop(self, a, p):
        if self._set(a, "finished", summarize(p.get("last_assistant_message", ""))):
            return [self._event(a)]

    def _ev_StopFailure(self, a, p):
        err = p.get("error", p.get("error_details", ""))
        if self._set(a, "failed", summarize(err if isinstance(err, str) else _tool_summary(err))):
            ev = self._event(a)
            ev.error = summarize(err, 40) if isinstance(err, str) and err.strip() else None
            return [ev]

    def _ev_PostToolUse(self, a, p):
        """A tool ran: the approval was given (waiting -> busy) and the run of tool errors is over."""
        a.tool_errors = 0
        if a.status in ("waiting", "unknown"):
            self._set(a, "busy", "")

    def _ev_PostToolUseFailure(self, a, p):
        """Stays busy, never an event: a red test run is routine. 3 in a row only changes the detail."""
        a.tool_errors += 1
        tool, err = summarize(p.get("tool_name", ""), 40), summarize(p.get("error", ""))
        self._set(a, "busy", f"{a.tool_errors} tool errors in a row" if a.tool_errors >= TOOL_FAIL_LIMIT
                  else summarize(f"{tool} failed: {err}"))

    # -- Colab ---------------------------------------------------------------------------------
    def on_colab(self, name: str, kind: str, line: str) -> list[AgentEvent]:
        status = _COLAB.get(kind)
        if status is None or not isinstance(name, str) or not name:
            return []
        a = self._get("colab:" + name[:128], "colab", summarize(name, 40) or "colab", "")
        if self._set(a, status, summarize(line)) and status != "running":
            return [self._event(a)]
        return []

    # -- time ----------------------------------------------------------------------------------
    def tick(self) -> list[AgentEvent]:
        """Staleness (busy or waiting -> unknown, never failed: a denied or Esc'd permission fires nothing) and removal
        of long-ended agents. Emits nothing today."""
        now = self.clock.now()
        for a in list(self._agents.values()):
            if a.status == "ended" and now - a.since >= REMOVE_S:
                del self._agents[a.id]
            elif a.kind == "claude" and a.status in ("busy", "waiting") and now - a.last_event_at >= STALE_S:
                a.status, a.since, a.needs_attention = "unknown", now, False
        return []

    def snapshot(self) -> list[dict]:
        """At most 20 items, attention first, newest first."""
        items = sorted(self._agents.values(), key=lambda a: (not a.needs_attention, -a.last_event_at))
        return [{"id": a.id, "kind": a.kind, "name": a.name, "status": a.status, "since": a.since,
                 "detail": a.detail, "needs_attention": a.needs_attention} for a in items[:SNAPSHOT_MAX]]
