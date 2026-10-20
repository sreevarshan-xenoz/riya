"""Pure state machine for the user's Claude Code sessions and Colab runs (spec §4.2)."""
import json
import re
import shlex
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
                 "PostToolUseFailure", "SubagentStop", "SessionStart", "SessionEnd", "PreToolUse"}
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


SUMMARY_SCAN = 4096  # visible chars summarize looks at (whitespace and control chars don't count)
SUMMARY_MAX = 1000   # the longest summary: well under what SUMMARY_SCAN chars can shrink to when all redacted


def redact(text: str) -> str:
    """Secret-looking values (API keys and tokens, Bearer/Authorization values, password=/token=/--password ...)
    replaced with ***."""
    for pattern, repl in _SECRETS:
        text = pattern.sub(repl, text)
    return text


def summarize(text, n: int = 80) -> str:
    """One clean line of at most n chars (n at most 1000): control chars stripped, then secrets redacted, then
    whitespace collapsed, then the cut (ellipsis on a cut). At most 4096 visible chars are looked at; a secret cut
    there can't survive, since even fully redacted they hold far more than n chars."""
    n = min(n, SUMMARY_MAX)
    if n <= 0:
        return ""
    s = text if isinstance(text, str) else str(text)
    out, kept = [], 0
    for ch in s:
        cat = unicodedata.category(ch)
        if ch.isspace() or cat in ("Zl", "Zp"):
            if out and out[-1] != " ":  # a run of whitespace is one space either way: keeps a flood of it small
                out.append(" ")
        elif cat[0] != "C":
            if kept >= SUMMARY_SCAN:
                break
            out.append(ch)
            kept += 1
    s = " ".join(redact("".join(out)).split())
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


LABEL_MAX = 40           # the longest activity label (a bubble never carries more)
ACTIVITY_GAP_S = 1.5     # at most one activity label per session in this time; the latest one wins
ACTIVITY_KINDS = ("edit", "run", "read", "search", "web", "delegate", "think", "wave", "done", "fail", "oops",
                  "sparkle")
TOOL_KINDS = {"Edit": "edit", "MultiEdit": "edit", "Write": "edit", "NotebookEdit": "edit", "Bash": "run",
              "PowerShell": "run", "Read": "read", "NotebookRead": "read", "Grep": "search", "Glob": "search",
              "LS": "search", "WebFetch": "web", "WebSearch": "web", "Task": "delegate", "Agent": "delegate"}
FIXED_LABELS = {"search": "searching the code", "web": "browsing", "delegate": "asking a helper",
                "think": "thinking…", "done": "all done!", "fail": "that didn't work", "oops": "oops!",
                "sparkle": "helper done"}
TOOL_FALLBACK = "using a tool"
TEST_RE = re.compile(r"(?:^|[\s/\\])(?:pytest|jest|vitest)\b|\b(?:npm|yarn|pnpm)\s+(?:run\s+)?test\b"
                     r"|\bnode\s+--test\b|\b(?:cargo|go)\s+test\b")
# What redact's \b anchors miss (a marker glued to a word, `Xsk-ant-...`): a name that looks like this is never shown
# (the token prefixes of GitHub, Stripe, GitLab, Slack, AWS and Google; `basic `/`token ` auth words; `=`; a URL
# credential `://` or `user:x@`; and any run of 20+ alphanumerics)
SECRET_HINT_RE = re.compile(r"sk-|sk_live|sk_test|pk_live|rk_live|ghp_|gh[osur]_|github_pat_|glpat-|xox|akia|aiza|"
                            r"ya29\.|bearer|basic |token |=|:.*(?://|@)|[a-z0-9]{20,}", re.IGNORECASE | re.DOTALL)
_URL_CRED_RE = re.compile(r":(?://|[^\s/\\]*@)")
_WORD_RE = re.compile(r"[A-Za-z0-9._+-]{1,24}")
_WORD_TOKENS = 20        # leading tokens _first_word looks at while skipping VAR=value prefixes
_WORD_SCAN = 500         # chars of a command it tokenises
_PATH_SCAN = 512         # trailing chars of a path it looks at (the file name is at the end)
_TOOL_SCAN = 200         # chars of a tool name it looks at
_FIXED_EVENTS = {"Stop": "done", "PostToolUseFailure": "oops", "StopFailure": "fail", "SubagentStop": "sparkle"}


def _secret_looking(text: str) -> bool:
    """True when text looks like (or hides, behind control or bidi chars) a key, token or credential."""
    plain = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    return bool(SECRET_HINT_RE.search(text) or SECRET_HINT_RE.search(plain))


def _basename(path) -> str | None:
    """The file name of a path (never the directory), or None when it is empty or secret-looking."""
    if not isinstance(path, str):
        return None
    tail = path[-_PATH_SCAN:]
    if len(path) > len(tail) and "/" not in tail and "\\" not in tail:
        return None  # one name longer than the window: it is cut at the front, so it is not shown at all
    name = PureWindowsPath(tail).name
    if not name.strip(". ") or _secret_looking(name):  # empty, or only dots ("..") or blanks: nothing to name
        return None
    return name


def _first_word(command) -> str | None:
    """The program a command runs (skipping VAR=value prefixes), or None when it is odd or secret-looking."""
    if not isinstance(command, str):
        return None
    text = command[:_WORD_SCAN]
    try:
        tokens = shlex.split(text, posix=False)
    except ValueError:
        tokens = text.split()
    for tok in tokens[:_WORD_TOKENS]:
        if "=" in tok:
            value = tok.partition("=")[2].lstrip("\"'").lower()
            if "bearer" in tok.lower() or value.startswith(("basic", "token")):
                return None  # `AUTH=Bearer v`, `AUTH=Basic v`: the credential is the next token
            continue
        if _URL_CRED_RE.search(tok):
            return None  # a URL (or user:pw@host) as the program: the basename alone would hide the credential
        word = PureWindowsPath(tok.strip("\"'")).name
        if word.lower().endswith(".exe"):
            word = word[:-4]
        if not word or _secret_looking(word):
            return None
        return word if _WORD_RE.fullmatch(word) else None
    return None


def tool_activity(tool_name, tool_input) -> tuple[str, str]:
    """(kind, label) for a tool about to run: the tool name plus a file basename or a command's first word, never
    arguments. Every label is at most LABEL_MAX chars and clean of secrets."""
    kind = TOOL_KINDS.get(tool_name) if isinstance(tool_name, str) else None
    if kind is None:
        return "run", summarize(TOOL_FALLBACK, LABEL_MAX)
    args = tool_input if isinstance(tool_input, dict) else {}
    if kind in ("edit", "read"):
        name = _basename(args.get("file_path")) or _basename(args.get("notebook_path"))
        verb = "editing" if kind == "edit" else "reading"
        text = f"{verb} {name}" if name else f"{verb} a file"
    elif kind == "run":
        command = args.get("command")
        word = _first_word(command)
        if isinstance(command, str) and TEST_RE.search(command[:200]):
            text = "running tests"
        else:
            text = f"running {word}" if word else "running a command"
    else:
        text = FIXED_LABELS[kind]
    return kind, summarize(text, LABEL_MAX)


@dataclass(frozen=True)
class Activity:
    session: str
    label: str
    kind: str
    tool: str
    at: float


def activity_of(payload, now: float) -> Activity | None:
    """The pet-facing activity for a Claude Code hook payload, or None. Reads only session_id, hook_event_name,
    tool_name and tool_input: never the prompt, the arguments or any message."""
    if not isinstance(payload, dict):
        return None
    sid, ev = payload.get("session_id"), payload.get("hook_event_name")
    if not isinstance(sid, str) or not 1 <= len(sid) <= 128 or not isinstance(ev, str):
        return None
    name, args = payload.get("tool_name"), payload.get("tool_input")
    if ev == "PreToolUse":
        kind, label = tool_activity(name, args)
    elif ev == "UserPromptSubmit":
        kind, label = "think", FIXED_LABELS["think"]
    elif ev == "PermissionRequest":
        kind, label = "wave", summarize(f"needs you! {tool_activity(name, args)[1]}", LABEL_MAX)
    elif ev in _FIXED_EVENTS:
        kind = _FIXED_EVENTS[ev]
        label = FIXED_LABELS[kind]
    else:
        return None
    return Activity(sid, label, kind, summarize(name[:_TOOL_SCAN], 40) if isinstance(name, str) else "", now)


class ActivityGate:
    """At most one activity per session every gap_s; what arrives in between is held, and the latest one wins. A "wave"
    (a PermissionRequest: the agent needs the user) is never held and never replaced: it goes out at once and starts
    the session's gap, so a PreToolUse right behind it can't delay or overwrite it."""

    def __init__(self, gap_s: float = ACTIVITY_GAP_S):
        self.gap_s = gap_s
        self._last: dict[str, float] = {}      # session -> its last send (least recent first)
        self._held: dict[str, Activity] = {}   # session -> the activity waiting (in the order first held)

    def __len__(self) -> int:
        return len(self._last)

    def _sent(self, sid: str, now: float) -> None:
        self._last.pop(sid, None)
        self._last[sid] = now
        self._held.pop(sid, None)

    def offer(self, act: Activity, now: float) -> Activity | None:
        last = self._last.get(act.session)
        if last is None or now - last >= self.gap_s or act.kind == "wave":
            if last is None and len(self._last) >= MAX_AGENTS:
                oldest = next(iter(self._last))
                del self._last[oldest]
                self._held.pop(oldest, None)
            self._sent(act.session, now)
            return act
        self._held[act.session] = act  # replaces a held one in place: the order stays first-held
        return None

    def due(self, now: float) -> list[Activity]:
        out = [a for sid, a in self._held.items() if self._last[sid] + self.gap_s <= now]
        for a in out:
            self._sent(a.session, now)
        return out


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
        if not isinstance(sid, str) or not sid or len(sid) > 128 or not isinstance(ev, str) or ev not in CLAUDE_EVENTS:
            return []
        cwd = payload.get("cwd")
        folder = summarize(PureWindowsPath(cwd[-512:]).name, 40) if isinstance(cwd, str) and cwd else ""
        name = f"{folder or 'claude'} {summarize(sid[:6], 6)}"
        a = self._agents.get(sid)
        if a is None:
            # sessions may predate Riya; a subagent stop or a tool that ran means a turn is running
            a = self._get(sid, "claude", name, "busy" if ev in ("SubagentStop", "PostToolUse", "PreToolUse") else "idle")
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

    def _ev_PreToolUse(self, a, p):
        """A tool is about to run: the session is alive and mid-turn. Never an event, never a detail."""
        if a.status == "unknown":
            self._set(a, "busy")

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
        elif a.detail.endswith(" tool errors in a row"):
            a.detail = ""  # recovered: the errors aren't news anymore

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
        """At most 20 items, attention first, then the latest status change first (never the latest event: the
        order mustn't move on every tool call)."""
        items = sorted(self._agents.values(), key=lambda a: (not a.needs_attention, -a.since))
        return [{"id": a.id, "kind": a.kind, "name": a.name, "status": a.status, "since": a.since,
                 "detail": a.detail, "needs_attention": a.needs_attention} for a in items[:SNAPSHOT_MAX]]
