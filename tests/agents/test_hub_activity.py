import random
import re
import string

import pytest

from riya.agents.hub import (ACTIVITY_GAP_S, ACTIVITY_KINDS, LABEL_MAX, MAX_AGENTS, STALE_S, Activity, ActivityGate,
                             AgentHub, activity_of, tool_activity)
from riya.core.clock import FakeClock

SID = "abcdef123456"


def pre(tool, tool_input, sid=SID):
    return {"session_id": sid, "hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": tool_input}


def label(tool, tool_input):
    return tool_activity(tool, tool_input)[1]


def bash(cmd):
    return label("Bash", {"command": cmd})


def edit(path):
    return label("Edit", {"file_path": path})


@pytest.mark.parametrize("tool, tool_input, want", [
    ("Edit", {"file_path": r"E:\riya\riya\core\engine.py"}, ("edit", "editing engine.py")),
    ("Write", {"file_path": "docs/x/spec.md"}, ("edit", "editing spec.md")),
    ("Bash", {"command": ".venv\\Scripts\\python -m pytest -q"}, ("run", "running tests")),
    ("Bash", {"command": "npm --prefix body test"}, ("run", "running npm")),  # TEST_RE wants `npm test` adjacent
    ("Bash", {"command": "git status"}, ("run", "running git")),
    ("Bash", {"command": "\"C:\\Program Files\\Git\\bin\\git.exe\" push"}, ("run", "running git")),
    ("Read", {"file_path": "/a/b/README.md"}, ("read", "reading README.md")),
    ("Grep", {"pattern": "x"}, ("search", "searching the code")),
    ("WebFetch", {"url": "https://x"}, ("web", "browsing")),
    ("Task", {"prompt": "..."}, ("delegate", "asking a helper")),
    ("mcp__srv__do", {}, ("run", "using a tool")),
    ("Edit", "not a dict", ("edit", "editing a file")),
])
def test_tool_labels(tool, tool_input, want):
    assert tool_activity(tool, tool_input) == want


def test_labels_never_leak_secrets_or_long_text():
    got = bash("OPENAI_KEY=sk-ant-api03-ABCDEFGH12345678 npm test")
    assert got == "running tests" and "sk-" not in got and "OPENAI" not in got
    assert bash("curl -H 'Authorization: Bearer abcdefghijklmnop' https://x") == "running curl"
    assert bash("--password=hunter2") == "running a command"
    assert edit(r"C:\keys\sk-ant-api03-abcdefghijklmnop.txt") == "editing a file"
    long = edit("C:/" + "a-" * 150 + ".py")
    assert len(long) <= LABEL_MAX and long.endswith("…")
    assert edit("C:/") == "editing a file"
    bad = edit("evil\u202e\x1b[31m.py")
    assert not re.search(r"[\x00-\x1f\x7f\u202a-\u202e\u2066-\u2069]", bad)
    got = activity_of({**pre("Bash", {"command": "rm -rf / --token=xyzxyzxyz"}), "hook_event_name": "PermissionRequest"},
                      1.0)
    assert got.label == "needs you! running rm"
    for text in (bash("git commit -m 'a secret message' /etc/passwd"), bash("ls C:\\Users\\x\\secret.txt other"),
                 edit(r"E:\a\b\c.py"), got.label):
        assert "/" not in text and "\\" not in text
    assert bash("git commit -m 'a secret message' /etc/passwd") == "running git"
    assert bash("ls C:\\Users\\x\\secret.txt other") == "running ls"
    assert edit(r"E:\a\b\c.py") == "editing c.py"


@pytest.mark.parametrize("command, want", [
    ("Xsk-ant-ABCDEFGH12345678 --go", "running a command"),
    ("XBearer abc", "running a command"),
    ("ghp_abc123 push", "running a command"),
    ("AKIAABCD go", "running a command"),
    ("xoxb-1-2 x", "running a command"),
    ("ABCDEFGHIJ0123456789 x", "running a command"),
    ("AUTH=Bearer abcdefgh12345678 curl", "running a command"),
    ("git status", "running git"),
])
def test_secret_looking_names_become_a_file_or_a_command(command, want):
    assert bash(command) == want


@pytest.mark.parametrize("path, want", [
    ("C:/x/Xsk-ant-ab.txt", "editing a file"),
    ("C:/x/token=abc.txt", "editing a file"),
    ("C:/x/Bearer abc.txt", "editing a file"),
    ("C:/x/ghp_x.md", "editing a file"),
    ("C:/x/" + "A1" * 10 + ".log", "editing a file"),
    ("C:/x/test_pet_body_sync.py", "editing test_pet_body_sync.py"),
])
def test_secret_looking_file_names(path, want):
    assert edit(path) == want


def test_secret_looking_read_name():
    assert label("Read", {"file_path": "C:/x/xoxp-9.txt"}) == "reading a file"


def test_labels_fuzz_stay_short_and_clean():
    rng = random.Random(9)
    alphabet = string.printable + "=\"'//\\\\"
    alnum = string.ascii_letters + string.digits
    for i in range(500):
        value = "".join(rng.choice(alnum) for _ in range(16))
        marker = rng.choice(["sk-ant-", "token=", "Bearer "]) + value
        if i % 2:
            marker = "X" + marker  # glued to a word: redact's \b misses it, SECRET_HINT_RE must not
        pad = lambda: "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))  # noqa: E731
        parts = [pad(), marker, pad()]
        rng.shuffle(parts)
        text = rng.choice([" ", "/", "\\", ""]).join(parts)
        for kind_label in (bash(text), edit(text), edit("C:/" + text), label("Read", {"file_path": text})):
            assert len(kind_label) <= LABEL_MAX
            assert not re.search(r"[\x00-\x1f]", kind_label)
            assert value not in kind_label and value[:8] not in kind_label


def test_activity_of_maps_each_hook_event():
    def mk(event, **kw):
        return {"session_id": SID, "hook_event_name": event, **kw}

    got = activity_of(mk("PreToolUse", tool_name="Edit", tool_input={"file_path": "a/b.py"}), 5.0)
    assert got == Activity(SID, "editing b.py", "edit", "Edit", 5.0)
    assert activity_of(mk("UserPromptSubmit", prompt="CANARY-SECRET do it"), 6.0).kind == "think"
    assert "CANARY" not in activity_of(mk("UserPromptSubmit", prompt="CANARY-SECRET do it"), 6.0).label
    wave = activity_of(mk("PermissionRequest", tool_name="Bash", tool_input={"command": "git push"}), 7.0)
    assert (wave.kind, wave.label, wave.tool, wave.at) == ("wave", "needs you! running git", "Bash", 7.0)
    assert activity_of(mk("Stop"), 1.0).kind == "done"
    assert activity_of(mk("PostToolUseFailure", tool_name="Bash", error="x"), 1.0).kind == "oops"
    assert activity_of(mk("StopFailure"), 1.0).kind == "fail"
    assert activity_of(mk("SubagentStop"), 1.0).kind == "sparkle"
    for event in ("Stop", "UserPromptSubmit", "PermissionRequest", "StopFailure", "SubagentStop",
                  "PostToolUseFailure", "PreToolUse"):
        a = activity_of(mk(event), 9.5)
        assert a.at == 9.5 and a.kind in ACTIVITY_KINDS and len(a.label) <= LABEL_MAX
    assert activity_of(mk("Notification"), 1.0) is None
    assert activity_of(mk("SessionStart"), 1.0) is None
    assert activity_of({"hook_event_name": "Stop"}, 1.0) is None
    assert activity_of({"session_id": "x" * 129, "hook_event_name": "Stop"}, 1.0) is None
    assert activity_of({"session_id": "", "hook_event_name": "Stop"}, 1.0) is None
    assert activity_of({"session_id": 5, "hook_event_name": "Stop"}, 1.0) is None
    assert activity_of("nope", 1.0) is None
    assert activity_of(mk("Stop", session_id="x" * 128), 1.0) is not None


def act(sid, n, at=0.0):
    return Activity(sid, f"label {n}", "run", "Bash", at)


def test_gate_rate_limits_per_session_latest_wins():
    gate = ActivityGate()
    assert ACTIVITY_GAP_S == 1.5
    a1, a2, a3, a4 = (act("a", n) for n in range(1, 5))
    assert gate.offer(a1, 0) is a1
    assert gate.offer(a2, 0.5) is None and gate.offer(a3, 1.0) is None
    assert gate.due(1.4) == []
    assert gate.due(1.5) == [a3]
    assert gate.due(9.0) == []  # sent once
    b1 = act("b", 1)
    assert gate.offer(b1, 0.6) is b1  # another session has its own clock
    assert gate.offer(a4, 2.0) is None  # last send was 1.5
    assert gate.due(2.9) == []
    assert gate.due(3.0) == [a4]
    cap = ActivityGate()
    for n in range(MAX_AGENTS + 1):
        cap.offer(act(f"s{n}", n), float(n))
    assert len(cap) == MAX_AGENTS


def test_gate_due_keeps_first_held_order_and_cap_drops_the_least_recent():
    gate = ActivityGate()
    gate.offer(act("a", 0), 0)
    gate.offer(act("b", 0), 0)
    a, b = act("a", 1), act("b", 1)
    gate.offer(b, 0.1)
    gate.offer(a, 0.2)
    gate.offer(act("b", 2), 0.3)  # replaces b's held one, order stays b, a
    assert [x.label for x in gate.due(1.5)] == ["label 2", "label 1"]
    assert [x.session for x in gate.due(1.5)] == []


def test_pre_tool_use_keeps_a_session_alive_quietly():
    clock = FakeClock()
    hub = AgentHub(clock)
    assert hub.on_claude_event(pre("Bash", {"command": "ls"})) == []
    assert hub.snapshot()[0]["status"] == "busy"
    # unknown -> busy; waiting stays waiting
    clock.advance(STALE_S + 1)
    hub.tick()
    assert hub.snapshot()[0]["status"] == "unknown"
    hub.on_claude_event(pre("Bash", {"command": "ls"}))
    assert hub.snapshot()[0]["status"] == "busy"
    hub.on_claude_event({"session_id": SID, "hook_event_name": "PermissionRequest", "tool_name": "Bash"})
    assert hub.snapshot()[0]["status"] == "waiting"
    before = hub.snapshot()
    for _ in range(50):
        assert hub.on_claude_event(pre("Edit", {"file_path": "a.py"})) == []
    assert hub.snapshot() == before


@pytest.mark.parametrize("command", [
    "ghs_abc push", "gho_abc push", "ghu_abc push", "ghr_abc push", "github_pat_abc push", "sk_live_abcd1234 x",
    "sk_test_abcd1234 x", "pk_live_abcd x", "rk_live_abcd x", "glpat-abcd x", "xoxa-1 x", "XAIzaSyAbc x", "ya29.abc x",
    "https://user:pw@host/x", "user:pw@host", "AUTH=Basic abcdefgh git", "AUTH=Token abcdefgh git",
    "\"AUTH=basic abcdefgh\" git", "ftp://host/x",
])
def test_more_token_prefixes_and_url_credentials_read_as_a_command(command):
    assert bash(command) == "running a command"


@pytest.mark.parametrize("path", [
    "C:/x/ghs_ab.txt", "C:/x/gho_ab.txt", "C:/x/ghu_ab.txt", "C:/x/ghr_ab.txt", "C:/x/github_pat_ab.txt",
    "C:/x/sk_live_ab.txt", "C:/x/sk_test_ab.txt", "C:/x/pk_live_ab.txt", "C:/x/rk_live_ab.txt", "C:/x/glpat-ab.txt",
    "C:/x/xoxb-1.txt", "C:/x/AIzaSy.txt", "C:/x/ya29.ab.txt", "C:/x/basic abc.txt", "C:/x/token abc.txt",
    "C:/x/user:pw@host.txt", "C:/x/a:b@c.txt",
])
def test_more_token_prefixes_and_url_credentials_read_as_a_file(path):
    assert edit(path) == "editing a file"


def test_an_env_assignment_is_still_skipped_when_it_is_not_a_credential():
    assert bash("GITHUB_TOKEN=xyz git push") == "running git"
    assert bash("CI=1 npm run build") == "running npm"


@pytest.mark.parametrize("path", [".", "..", "a/..", "a\\..", "C:/", "", "   ", "a/. ", "a/..."])
def test_unreadable_basenames_read_as_a_file(path):
    assert edit(path) == "editing a file"
    assert label("Read", {"file_path": path}) == "reading a file"


@pytest.mark.parametrize("ev", [[], {}, 5, None, ("Stop",), 1.5, True])
def test_activity_of_never_raises_on_a_non_str_event(ev):
    assert activity_of({"session_id": SID, "hook_event_name": ev}, 1.0) is None
    assert activity_of({"session_id": SID}, 1.0) is None
    assert AgentHub(FakeClock()).on_claude_event({"session_id": SID, "hook_event_name": ev}) == []


def test_a_huge_payload_is_bounded_before_any_regex_or_split(monkeypatch):
    import riya.agents.hub as hub

    seen = {"path": 0, "split": 0, "hint": 0, "test": 0, "summarize": 0}

    class SpyPath(hub.PureWindowsPath):
        def __new__(cls, *args):
            seen["path"] = max(seen["path"], max(len(a) for a in args))
            return super().__new__(cls, *args)

    class SpyRe:
        def __init__(self, key, rx):
            self.key, self.rx = key, rx

        def search(self, text, *a):
            seen[self.key] = max(seen[self.key], len(text))
            return self.rx.search(text, *a)

    real_split, real_summarize = hub.shlex.split, hub.summarize

    def split(text, **kw):
        seen["split"] = max(seen["split"], len(text))
        return real_split(text, **kw)

    def summ(text, n=80):
        seen["summarize"] = max(seen["summarize"], len(text) if isinstance(text, str) else 0)
        return real_summarize(text, n)

    monkeypatch.setattr(hub, "PureWindowsPath", SpyPath)
    monkeypatch.setattr(hub, "SECRET_HINT_RE", SpyRe("hint", hub.SECRET_HINT_RE))
    monkeypatch.setattr(hub, "TEST_RE", SpyRe("test", hub.TEST_RE))
    monkeypatch.setattr(hub.shlex, "split", split)
    monkeypatch.setattr(hub, "summarize", summ)
    huge = 250_000
    for tool_input in ({"file_path": "C:/x/" + "a-" * huge + ".py"}, {"file_path": "a-" * huge},
                       {"file_path": "C:/" + " " * huge + "x.py"}, {"command": "x " * huge},
                       {"command": " " * huge + "git"}, {"command": "git " + "=" * huge}):
        for tool in ("Edit", "Read", "Bash"):
            assert len(tool_activity(tool, tool_input)[1]) <= LABEL_MAX
    a = activity_of({"session_id": SID, "hook_event_name": "PreToolUse", "tool_name": "x" * huge,
                     "tool_input": {"command": "ls"}}, 1.0)
    assert len(a.tool) <= 40
    assert seen["path"] <= 512 and seen["split"] <= 500 and seen["test"] <= 200 and seen["hint"] <= 512
    assert seen["summarize"] <= 512


def test_a_name_longer_than_the_path_window_is_never_shown():
    assert edit("C:/x/" + "a-" * 400) == "editing a file"  # cut at the front by the window: not a real name
    assert edit("a-" * 400 + "/short.py") == "editing short.py"


def test_gate_never_holds_or_replaces_a_wave():
    """A PermissionRequest ("needs you") goes out at once, even inside the gap; a later tool bubble can't replace it."""
    gate = ActivityGate()
    run1, run2 = act("a", 1), act("a", 2)
    wave = Activity("a", "needs you! running tests", "wave", "Bash", 0.2)
    assert gate.offer(run1, 0) is run1
    assert gate.offer(wave, 0.2) is wave  # inside the 1.5 s gap
    assert gate.offer(run2, 0.3) is None  # the wave started a new gap; this one is held...
    assert gate.due(1.6) == []
    assert gate.due(1.7) == [run2]  # ...and only ever sent after the wave
    assert gate.offer(Activity("a", "needs you! again", "wave", "Bash", 1.8), 1.8).kind == "wave"


def test_gate_drops_only_a_repeat_wave():
    gate = ActivityGate()

    def wave(label, at):
        return Activity("a", label, "wave", "Bash", at)

    assert gate.offer(wave("needs you! x", 0), 0) is not None
    assert gate.offer(wave("needs you! x", 0.1), 0.1) is None  # the same label inside the gap: a repeat
    assert gate.offer(wave("needs you! y", 0.2), 0.2) is not None  # a new label always goes out
    assert gate.offer(wave("needs you! y", 1.0), 1.0) is None
    assert gate.due(5.0) == []  # a dropped repeat is not held either
    assert gate.offer(wave("needs you! y", 1.7), 1.7) is not None  # 1.5 s since the last wave
    tool = act("a", 9)
    assert gate.offer(tool, 1.8) is None  # held (inside the gap), then sent after the gap
    assert gate.due(3.2) == [tool]
    assert gate.offer(wave("needs you! y", 3.3), 3.3) is not None  # a non-wave in between: out at once
