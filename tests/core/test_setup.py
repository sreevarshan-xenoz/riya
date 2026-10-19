"""Setup actions: the allow-list, single-use confirm nonces, the actions on temp paths with fakes, the eye-test picture.

Nothing here reaches the real ~/.claude/settings.json, %LOCALAPPDATA%\\Riya, the mic, audio, setx or the network: every
path is under tmp_path, the installer's runner is a recorder and every model is a FakeModel."""
import base64
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from riya.agents import install_hooks as ih
from riya.core import setup as setup_mod
from riya.core.clock import FakeClock
from riya.core.models import ModelUnavailable
from riya.core.secrets import KEY_VAR, REDACTED, InvalidKey, OnlineKey, mask
from riya.core.settings import Settings
from riya.core.setup import (ACTIONS, ARGS, CANCELLED, CONFIRM_GATED, EYES_IMAGE, EYES_QUESTION, MAX_PENDING,
                             NONCE_TTL_S, PLAIN, SAMPLE_TEXT, SUMMARIES, TESTS_KEY, ConfirmAsk, Nonces, Outcome, Setup,
                             SetupDeps, SetupError)
from riya.core.store import KV, open_db
from tests.fakes import FakeModel

CANARY = "CANARY-k3y-0123456789abcdef"
BASE = "DOTENV-base-9876543210"


class FakeHost:
    def __init__(self):
        self.settings = Settings()
        self.clock = FakeClock()
        self.kv = KV(open_db(":memory:"))
        self.patches, self.swaps, self.samples, self.rows = [], [], [], []
        self.mic = (True, "Fake Mic opens")
        self.sample_error = self.patch_error = None
        self.sample_hook = None

    def patch_settings(self, patch):
        if self.patch_error is not None:
            raise self.patch_error
        self.patches.append(dict(patch))
        for k, v in patch.items():
            setattr(self.settings, k, v)
        return self.settings

    def swap_online(self, online, vision):
        self.swaps.append((online, vision))

    def sample_voice(self, voice, text):
        self.samples.append((voice, text))
        if self.sample_hook is not None:
            self.sample_hook()
        if self.sample_error is not None:
            raise self.sample_error

    def probe_mic(self):
        return self.mic

    def log_action(self, tool, tier, summary, outcome):
        self.rows.append((tool, tier, summary, outcome))


class Runner:
    def __init__(self, rc=0):
        self.calls, self.rc = [], rc

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return SimpleNamespace(returncode=self.rc)


class MakeModel:
    """A recording make_model: None without a key (like the real one), else a FakeModel that remembers its env key."""

    def __init__(self, replies=None, fail=None):
        self.replies, self.fail = replies or {}, fail
        self.made, self.models = [], []

    def __call__(self, env, name, timeout_s):
        self.made.append((name, timeout_s))
        if not env.get(KEY_VAR):
            return None
        m = FakeModel(name, self.replies.get(name, ("ok",)), fail=self.fail)
        m.api_key = env[KEY_VAR]
        self.models.append(m)
        return m


def make_deps(tmp_path, key=None, environ=None, replies=None, fail=None, voices=None):
    d = SetupDeps(key or OnlineKey(tmp_path / "secrets.env", None, None), environ or {},
                  tmp_path / "claude" / "settings.json", tmp_path / "tok" / "hook_token", Runner(),
                  voices or (lambda: ["alba"]), make_model=MakeModel(replies, fail))
    return d


def make_setup(tmp_path, host=None, deps=None, **kw):
    host = host or FakeHost()
    deps = deps or make_deps(tmp_path)
    results = []
    kw.setdefault("spawn", lambda fn: fn())
    s = Setup(host, deps, results.append, **kw)
    return s, host, deps, results


def everything(s, host, results):
    """Every place a key could leak to from Setup: results, log rows, KV and the nonce store."""
    return "\n".join([repr(results), repr(host.rows), "\n".join(host.kv.conn.iterdump()), repr(s.nonces.__dict__)])


# ---- the allow-list -------------------------------------------------------------------------------------------------

def test_allow_list(tmp_path):
    s, host, _, results = make_setup(tmp_path)
    with pytest.raises(SetupError, match="unknown setup action"):
        s.request("rm_rf", {})
    with pytest.raises(SetupError) as e:
        s.request("online_key_set", {"key": CANARY, "x": "1"})
    assert str(e.value) == "bad arguments for online_key_set" and CANARY not in repr(e.value)
    assert set(ACTIONS) == set(CONFIRM_GATED) | set(PLAIN) and not set(CONFIRM_GATED) & set(PLAIN)
    assert PLAIN == ("online_key_set", "online_key_clear", "voice_sample", "voice_choose", "mic_check", "autostart",
                     "onboarding_done")
    assert CONFIRM_GATED == ("online_test", "eyes_test", "hooks_connect", "hooks_disconnect")
    assert set(SUMMARIES) == set(ACTIONS)
    assert results == [] and host.rows == [] and host.patches == []


@pytest.mark.parametrize("action, args", [
    ("online_key_set", {}), ("online_key_set", {"key": 5}), ("online_key_set", [CANARY]),
    ("voice_choose", {"voice": None}), ("voice_sample", {"name": "eve"}),
    ("autostart", {"on": 1}), ("autostart", {"on": "true"}), ("autostart", {}),
    ("mic_check", {"x": 1}), ("hooks_connect", {"force": True}), ("onboarding_done", None),
    (None, {}), (["hooks_connect"], {}),
])
def test_bad_arguments_are_refused_before_anything_runs(tmp_path, action, args):
    s, host, deps, results = make_setup(tmp_path)
    with pytest.raises(SetupError) as e:
        s.request(action, args)
    assert str(e.value) in ("unknown setup action", f"bad arguments for {action}")
    assert results == [] and host.rows == [] and deps.runner.calls == [] and s.nonces._pending == {}


# ---- nonces ---------------------------------------------------------------------------------------------------------

def test_nonces_are_single_use_and_expire_at_60_s():
    clock = FakeClock()
    n = Nonces(clock)
    a = n.issue("hooks_connect", {})
    assert n.take(a) == ("hooks_connect", {})
    assert n.take(a) is None  # single use

    b = n.issue("online_test", {})
    clock.advance(59.9)
    assert n.take(b) == ("online_test", {})

    c = n.issue("online_test", {})
    clock.advance(NONCE_TTL_S)
    assert n.take(c) is None  # 60 s old: expired
    assert n.take(c) is None

    first = n.issue("eyes_test", {})
    rest = [n.issue("eyes_test", {}) for _ in range(MAX_PENDING)]  # a ninth issue drops the oldest
    assert n.take(first) is None and all(n.take(r) == ("eyes_test", {}) for r in rest)

    assert n.take("never-issued") is None and n.take("") is None and n.take(None) is None and n.take(["x"]) is None


def test_nonces_prune_expired_ones_and_are_unguessable():
    clock = FakeClock()
    n = Nonces(clock)
    old = n.issue("hooks_connect", {})
    clock.advance(61)
    n.issue("hooks_connect", {})
    assert old not in n._pending and len(n._pending) == 1
    made = {Nonces(clock).issue("online_test", {}) for _ in range(20)}
    assert len(made) == 20 and all(len(x) >= 16 for x in made)
    fixed = Nonces(clock, ttl_s=5, make=iter(["n1", "n2"]).__next__)
    assert fixed.issue("online_test", {}) == "n1"
    clock.advance(5)
    assert fixed.take("n1") is None


# ---- the confirm round trip -----------------------------------------------------------------------------------------

def test_gated_request_runs_nothing_until_answered(tmp_path, monkeypatch):
    s, host, deps, results = make_setup(tmp_path)
    ran = []
    monkeypatch.setattr(s, "perform", lambda *a: ran.append(a))
    ask = s.request("hooks_connect", {})
    assert isinstance(ask, ConfirmAsk) and ask.action == "hooks_connect" and ask.nonce
    assert ask.text.endswith(ih.RESTART_LINE) and str(deps.claude_settings) in ask.text
    assert ask.text == ih.preview("apply", deps.claude_settings)
    assert not deps.claude_settings.exists() and deps.runner.calls == []
    assert results == [] and ran == [] and host.rows == []


def test_answer(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    ask = s.request("hooks_connect", {})
    assert s.answer(ask.nonce, True) is True
    data = json.loads(deps.claude_settings.read_text(encoding="utf-8"))
    assert ih.riya_events(data) == [e for e, _ in ih.EVENTS]
    token = deps.token_path.read_text(encoding="utf-8").strip()
    assert [c for c, _ in deps.runner.calls] == [["setx", "RIYA_HOOK_TOKEN", token]]
    assert results == [Outcome("hooks_connect", True, "Connected. Restart Claude Code (and open a new terminal).")]
    assert host.rows == [("setup", 2, "Connected Claude Code", "ran")]

    no = s.request("hooks_disconnect", {})
    before = deps.claude_settings.read_bytes()
    assert s.answer(no.nonce, False) is True
    assert results[-1] == Outcome("hooks_disconnect", False, CANCELLED)
    assert host.rows[-1] == ("setup", 2, "Disconnected Claude Code", "declined")
    assert deps.claude_settings.read_bytes() == before

    n_results = len(results)
    assert s.answer(ask.nonce, True) is False  # reused
    assert s.answer(no.nonce, True) is False  # a No used it up too
    assert host.rows[-1] == ("setup", 2, "Refused a stale or reused confirmation", "refused")
    assert len(results) == n_results and len(deps.runner.calls) == 1


def test_stale_unknown_and_wrong_action_nonces_run_nothing(tmp_path, monkeypatch):
    s, host, deps, results = make_setup(tmp_path)
    ran = []
    real = s.perform
    monkeypatch.setattr(s, "perform", lambda a, args: ran.append(a) or real(a, args))

    stale = s.request("hooks_connect", {})
    host.clock.advance(60)
    assert s.answer(stale.nonce, True) is False
    assert s.answer("made-up-nonce", True) is False
    assert s.answer(None, True) is False
    # a nonce that somehow names a plain action is never honoured by answer
    plain = s.nonces.issue("online_key_set", {"key": CANARY})
    assert s.answer(plain, True) is False
    assert ran == [] and results == [] and deps.runner.calls == [] and not deps.claude_settings.exists()
    assert [r[3] for r in host.rows] == ["refused"] * 4

    # a nonce runs the action it was issued for, never another one
    deps.key.set(CANARY)
    ask = s.request("online_test", {})
    assert s.answer(ask.nonce, True) is True
    assert ran == ["online_test"] and deps.runner.calls == [] and not deps.claude_settings.exists()
    assert CANARY not in everything(s, host, results)


def test_only_an_explicit_true_runs(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    for answer in ("yes", 1, None):
        ask = s.request("hooks_connect", {})
        assert s.answer(ask.nonce, answer) is True
        assert results[-1] == Outcome("hooks_connect", False, CANCELLED)
    assert deps.runner.calls == [] and not deps.claude_settings.exists()


def test_confirm_texts(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    assert s.request("online_test", {}) is None and s.request("eyes_test", {}) is None
    assert results == [Outcome("online_test", False, "No key to test."), Outcome("eyes_test", False, "No key to test.")]
    assert s.nonces._pending == {} and host.rows == []

    deps.key.set(CANARY)
    m = mask(CANARY)
    assert s.confirm_text("online_test") == (
        f"Riya will send one tiny request to generativelanguage.googleapis.com with your key {m} to check that it works.")
    assert s.confirm_text("eyes_test") == (
        f"Riya will send her built-in test picture (not your screen) to generativelanguage.googleapis.com with your key "
        f"{m} and ask what number it shows.")
    deps.environ = {"RIYA_ONLINE_BASE_URL": "https://user:pw@llm.example.test:8443/v1"}
    assert "to llm.example.test with your key" in s.confirm_text("online_test")
    assert s.confirm_text("hooks_disconnect") == ih.preview("remove", deps.claude_settings)

    deps.claude_settings.parent.mkdir(parents=True)
    deps.claude_settings.write_text("{not json", encoding="utf-8")
    results.clear()
    assert s.request("hooks_connect", {}) is None
    assert results == [Outcome("hooks_connect", False, "Couldn't read Claude Code's settings.json; nothing was changed.")]
    assert CANARY not in everything(s, host, results)


# ---- hooks ----------------------------------------------------------------------------------------------------------

def test_hooks_disconnect_on_temp_paths(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    deps.claude_settings.parent.mkdir(parents=True)
    foreign = {"type": "command", "command": "echo mine"}
    deps.claude_settings.write_text(json.dumps({"hooks": {"Stop": [{"hooks": [foreign]}]}}), encoding="utf-8")
    s.answer(s.request("hooks_connect", {}).nonce, True)
    assert results[-1].ok
    ask = s.request("hooks_disconnect", {})
    assert "remove her hook from 10 Claude Code event(s)" in ask.text
    assert s.answer(ask.nonce, True) is True
    assert results[-1] == Outcome("hooks_disconnect", True,
                                  "Disconnected. Restart Claude Code (and open a new terminal).")
    data = json.loads(deps.claude_settings.read_text(encoding="utf-8"))
    assert ih.riya_events(data) == [] and data == {"hooks": {"Stop": [{"hooks": [foreign]}]}}
    assert host.rows[-1] == ("setup", 2, "Disconnected Claude Code", "ran")
    assert len(deps.runner.calls) == 1  # only connect runs setx


def deep_token_path(tmp_path):
    return tmp_path.joinpath(*["a-rather-long-folder-name"] * 6) / "Riya" / "hook_token"


def test_hooks_connect_failure_reports_the_installer_line(tmp_path):
    deps = make_deps(tmp_path)
    deps.runner, deps.token_path = Runner(rc=1), deep_token_path(tmp_path)
    s, host, _, results = make_setup(tmp_path, deps=deps)
    s.answer(s.request("hooks_connect", {}).nonce, True)
    detail = results[-1].detail
    assert not results[-1].ok and detail.startswith("Hooks added, but setx failed: set RIYA_HOOK_TOKEN yourself")
    assert len(detail) <= 120 and detail.endswith(str(Path("Riya", "hook_token")))  # the path's tail survives the clip
    assert host.rows[-1] == ("setup", 2, "Connected Claude Code", "failed")


def test_a_setx_failure_wins_over_a_settings_warning(tmp_path):
    """--apply prints the allowedHttpHookUrls warning before setx fails: the detail names setx, the step that failed."""
    deps = make_deps(tmp_path)
    deps.token_path = deep_token_path(tmp_path)
    deps.claude_settings.parent.mkdir(parents=True)
    deps.claude_settings.write_text(json.dumps({"allowedHttpHookUrls": []}), encoding="utf-8")
    for runner in (Runner(rc=1), _raising_runner):
        deps.runner = runner
        s, host, _, results = make_setup(tmp_path, deps=deps)
        assert ih.warnings(json.loads(deps.claude_settings.read_text(encoding="utf-8")))  # the warning does occur
        s.answer(s.request("hooks_connect", {}).nonce, True)
        detail = results[-1].detail
        assert not results[-1].ok and "setx failed" in detail and "allowedHttpHookUrls" not in detail
        assert detail.endswith("hook_token") and len(detail) <= 120
        assert ih.riya_events(json.loads(deps.claude_settings.read_text(encoding="utf-8")))  # the hooks were written


def _raising_runner(cmd, **kw):
    raise OSError("setx.exe not found")


def test_hooks_failures_name_the_failing_step_without_full_paths(tmp_path):
    deps = make_deps(tmp_path)
    s, host, _, results = make_setup(tmp_path, deps=deps)
    deps.claude_settings.parent.mkdir(parents=True)
    deps.claude_settings.write_text(json.dumps({"hooks": {"Stop": [], "SessionEnd": "junk"}}), encoding="utf-8")
    assert s.perform("hooks_connect", {}) == Outcome(
        "hooks_connect", False, "Nothing was changed: 'hooks.SessionEnd' is not a list")
    deps.claude_settings.write_text("{not json", encoding="utf-8")
    out = s.perform("hooks_disconnect", {})
    assert not out.ok and out.detail.startswith("Nothing was changed: settings.json is not valid JSON (")
    assert str(deps.claude_settings.parent) not in out.detail
    assert deps.runner.calls == [] and deps.claude_settings.read_text(encoding="utf-8") == "{not json"

    # the write step fails (the token path is a folder): "write", with the OS reason
    deps.claude_settings.unlink()
    deps.token_path.mkdir(parents=True)
    out = s.perform("hooks_connect", {})
    assert not out.ok and out.detail.startswith("Nothing was changed: couldn't write the settings (")
    assert deps.runner.calls == [] and not deps.claude_settings.exists()


def test_sandboxed_deps_never_run_anything(tmp_path):
    d = SetupDeps.sandboxed(tmp_path / "sb")
    assert d.key.path == tmp_path / "sb" / "secrets.env" and d.key.base is None and d.environ == {}
    assert d.claude_settings == tmp_path / "sb" / ".claude" / "settings.json"
    assert d.token_path == tmp_path / "sb" / "hook_token" and d.voices() == ["alba"]
    assert d.runner(["setx", "X", "y"]).returncode == 1 and d.eyes_image == EYES_IMAGE


# ---- the key --------------------------------------------------------------------------------------------------------

def test_key_set_and_clear(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    out = s.perform("online_key_set", {"key": " " + CANARY + " "})
    assert out == Outcome("online_key_set", True, f"Saved {mask(CANARY)}. Her online brain uses it now.")
    assert deps.key.path.read_text(encoding="utf-8") == f"{KEY_VAR}={CANARY}\n"
    assert len(host.swaps) == 1 and all(m.api_key == CANARY for m in host.swaps[0])
    assert [m.name for m in host.swaps[0]] == ["online", "vision"]
    assert deps.make_model.made == [("online", 8.0), ("vision", 20.0)]

    bad = s.perform("online_key_set", {"key": CANARY + " x"})
    assert bad == Outcome("online_key_set", False, InvalidKey.BAD_CHARS) and len(host.swaps) == 1

    # a .env-only key can't be cleared from Riya
    env_only = make_deps(tmp_path / "e", key=OnlineKey(tmp_path / "e" / "secrets.env", BASE, ".env"))
    s2, host2, _, _ = make_setup(tmp_path, deps=env_only)
    assert s2.perform("online_key_clear", {}) == Outcome(
        "online_key_clear", False, "That key is set via .env; edit that file to remove it.")
    assert not env_only.key.path.exists() and host2.swaps == []
    proc_only = make_deps(tmp_path / "p", key=OnlineKey(tmp_path / "p" / "secrets.env", BASE, "env"))
    assert make_setup(tmp_path, deps=proc_only)[0].perform("online_key_clear", {}).detail == (
        "That key comes from the RIYA_ONLINE_API_KEY environment variable.")
    assert s.perform("online_key_clear", {}) == Outcome("online_key_clear", True, "Removed the saved key.")
    assert s.perform("online_key_clear", {}) == Outcome("online_key_clear", False, "There's no saved key.")
    assert host.swaps[-1] == (None, None)

    # a saved key over a .env base: clearing falls back to the base
    over_env = make_deps(tmp_path / "o", key=OnlineKey(tmp_path / "o" / "secrets.env", BASE, ".env"))
    s3, host3, _, _ = make_setup(tmp_path, deps=over_env)
    s3.perform("online_key_set", {"key": CANARY})
    assert host3.swaps[-1][0].api_key == CANARY
    assert s3.perform("online_key_clear", {}) == Outcome(
        "online_key_clear", True, "Removed the saved key. Using the one from .env.")
    assert [m.api_key for m in host3.swaps[-1]] == [BASE, BASE] and not over_env.key.path.exists()

    over_proc = make_deps(tmp_path / "q", key=OnlineKey(tmp_path / "q" / "secrets.env", BASE, "env"))
    s4, _, _, _ = make_setup(tmp_path, deps=over_proc)
    s4.perform("online_key_set", {"key": CANARY})
    assert s4.perform("online_key_clear", {}).detail == (
        "Removed the saved key. Using the one from the RIYA_ONLINE_API_KEY environment variable.")


def test_key_actions_through_request_log_without_the_key(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    assert s.request("online_key_set", {"key": CANARY}) is None
    s.request("online_key_set", {"key": "short"})  # under 8: masks to the bare bullets
    s.request("online_key_clear", {})
    assert host.rows == [("setup", 0, "Saved a new online key", "ran"), ("setup", 0, "Saved a new online key", "ran"),
                         ("setup", 0, "Removed the saved online key", "ran")]
    assert all(CANARY not in r[2] and "••••" not in r[2] for r in host.rows)
    assert CANARY not in everything(s, host, results)


def test_a_save_that_fails_keeps_the_key_out_of_the_detail(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    deps.key.path.mkdir(parents=True)  # the key file path is a folder: the write fails
    out = s.perform("online_key_set", {"key": CANARY})
    assert out == Outcome("online_key_set", False, "Couldn't save the key.") and host.swaps == []
    assert s.perform("online_key_clear", {}).ok is False


# ---- the tests the Health page runs ---------------------------------------------------------------------------------

def test_online_test(tmp_path):
    deps = make_deps(tmp_path)
    deps.key.set(CANARY)
    s, host, _, results = make_setup(tmp_path, deps=deps)
    s.start("online_test", {})
    assert results == [Outcome("online_test", True, "The online brain answered.")]
    assert deps.make_model.made == [("online-test", 10.0)]
    m = deps.make_model.models[0]
    assert m.calls == [[{"role": "user", "content": "Reply with the single word: ok"}]] and m.max_tokens == [200]
    last = host.kv.get(TESTS_KEY)["online_test"]
    assert last == {"at": host.clock.now(), "ok": True, "detail": "The online brain answered.", "mask": mask(CANARY)}
    assert host.rows == [("setup", 2, "Tested the online key", "ran")]

    deps.make_model.fail = ModelUnavailable(f"online-test: HTTP 400 {CANARY}")
    host.clock.advance(5)
    s.start("online_test", {})
    assert not results[-1].ok and REDACTED in results[-1].detail and CANARY not in results[-1].detail
    assert results[-1].detail.startswith("It didn't answer: ")
    assert host.kv.get(TESTS_KEY)["online_test"]["ok"] is False
    assert host.rows[-1] == ("setup", 2, "Tested the online key", "failed")
    assert CANARY not in everything(s, host, results)


def test_an_empty_answer_still_passes_the_online_test(tmp_path):
    deps = make_deps(tmp_path, replies={"online-test": ("",)})
    deps.key.set(CANARY)
    assert make_setup(tmp_path, deps=deps)[0].perform("online_test", {}).ok


def test_eyes_test_uses_the_bundled_picture(tmp_path):
    deps = make_deps(tmp_path, replies={"eyes-test": ("It says 42.",)})
    deps.key.set(CANARY)
    s, host, _, results = make_setup(tmp_path, deps=deps)
    s.start("eyes_test", {})
    assert results == [Outcome("eyes_test", True, "She read the test picture: 42.")]
    assert deps.make_model.made == [("eyes-test", 20.0)]
    (msgs,) = deps.make_model.models[0].calls
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    text, image = msgs[0]["content"]
    assert text == {"type": "text", "text": EYES_QUESTION}
    url = image["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == EYES_IMAGE.read_bytes()
    assert deps.make_model.models[0].max_tokens == [400]
    assert host.kv.get(TESTS_KEY)["eyes_test"]["ok"] is True

    deps.make_model.replies = {"eyes-test": ("a cat",)}
    s.start("eyes_test", {})
    assert results[-1] == Outcome("eyes_test", False, "She couldn't read the test picture (she said 'a cat').")
    assert host.rows == [("setup", 2, "Ran the eye test", "ran"), ("setup", 2, "Ran the eye test", "failed")]


def test_eyes_test_without_a_key(tmp_path):
    s, _, _, _ = make_setup(tmp_path)
    assert s.perform("eyes_test", {}) == Outcome("eyes_test", False, "Her eyes need the online key.")


def test_two_results_at_once_both_land(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    s.start("mic_check", {})
    s.start("voice_sample", {"voice": "alba"})
    assert set(host.kv.get(TESTS_KEY)) == {"mic_check", "voice_sample"}
    assert host.kv.get(TESTS_KEY)["mic_check"] == {"at": host.clock.now(), "ok": True, "detail": "Fake Mic opens",
                                                   "mask": None}


# ---- voices ---------------------------------------------------------------------------------------------------------

def test_voices_are_cached_only_by_load_voices(tmp_path):
    calls = []

    def voices():
        calls.append(1)
        return ["alba", "eve"]

    s, _, _, _ = make_setup(tmp_path, deps=make_deps(tmp_path, voices=voices))
    assert s.voices() == [] and calls == []
    s.load_voices()
    assert s.voices() == ["alba", "eve"] and calls == [1]
    s.voices().append("mutated")
    assert s.voices() == ["alba", "eve"] and calls == [1]

    def broken():
        raise ImportError("no pocket_tts")

    s2, _, _, _ = make_setup(tmp_path, deps=make_deps(tmp_path, voices=broken))
    s2.load_voices()
    assert s2.voices() == ["alba"]


def test_voice_actions(tmp_path):
    s, host, deps, results = make_setup(tmp_path, deps=make_deps(tmp_path, voices=lambda: ["alba", "eve"]))
    assert s.perform("voice_choose", {"voice": "eve"}).detail == "Unknown voice."  # before load_voices
    assert s.perform("voice_sample", {"voice": "alba"}).ok  # the current voice always passes
    s.load_voices()
    assert s.perform("voice_choose", {"voice": "eve"}) == Outcome("voice_choose", True, "Her voice is eve now.")
    assert host.patches == [{"voice": "eve"}]
    assert s.perform("voice_choose", {"voice": "darth"}) == Outcome("voice_choose", False, "Unknown voice.")
    assert s.perform("voice_sample", {"voice": "darth"}) == Outcome("voice_sample", False, "Unknown voice.")
    assert s.perform("voice_sample", {"voice": "eve"}) == Outcome("voice_sample", True, "Played a sample of eve.")
    assert host.samples[-1] == ("eve", SAMPLE_TEXT) and len(host.patches) == 1
    host.sample_error = RuntimeError("Her voice is muted in Settings.")
    assert s.perform("voice_sample", {"voice": "eve"}) == Outcome("voice_sample", False,
                                                                  "Her voice is muted in Settings.")


# ---- mic, autostart, onboarding -------------------------------------------------------------------------------------

def test_mic_autostart_onboarding(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    host.mic = (False, "The mic didn't open: no device")
    assert s.request("mic_check", {}) is None
    assert results[-1] == Outcome("mic_check", False, "The mic didn't open: no device")
    assert host.rows[-1] == ("setup", 0, "Checked the mic", "failed")

    s.request("autostart", {"on": True})
    assert host.patches[-1] == {"autostart": True} and results[-1].detail == "Starts with Windows now."
    s.request("autostart", {"on": False})
    assert host.patches[-1] == {"autostart": False} and results[-1].detail == "Won't start with Windows."
    host.patch_error = OSError("APPDATA isn't set")
    s.request("autostart", {"on": True})
    assert results[-1] == Outcome("autostart", False, "Couldn't change the Startup shortcut.")
    host.patch_error = None

    s.request("onboarding_done", {})
    assert host.patches[-1] == {"onboarded": True} and results[-1] == Outcome("onboarding_done", True, "Welcome done.")
    assert [r for r in host.rows if r[2] == "Finished the welcome"] == [("setup", 0, "Finished the welcome", "ran")]

    n_patches, n_results, n_rows = len(host.patches), len(results), len(host.rows)
    host.settings.onboarded = True
    assert s.request("onboarding_done", {}) is None
    assert (len(host.patches), len(results), len(host.rows)) == (n_patches, n_results, n_rows)


def test_a_hung_action_times_out(tmp_path):
    release = threading.Event()
    got, done = [], threading.Event()
    host = FakeHost()
    host.sample_hook = lambda: release.wait(5)  # "sleeps" 5 s unless the test lets it go

    def on_result(o):
        got.append(o)
        done.set()

    s = Setup(host, make_deps(tmp_path), on_result, timeout_s=0.2)
    try:
        t0 = time.monotonic()
        s.start("voice_sample", {"voice": "alba"})
        assert done.wait(1.0) and time.monotonic() - t0 < 1.0
        assert got == [Outcome("voice_sample", False, "No answer within 0.2 s.")]
        assert host.rows == [("setup", 0, "Played a voice sample", "failed")]
        assert host.kv.get(TESTS_KEY)["voice_sample"]["ok"] is False
    finally:
        release.set()
    time.sleep(0.2)
    assert got == [Outcome("voice_sample", False, "No answer within 0.2 s.")]  # the late result is dropped


def test_one_action_of_a_kind_at_a_time(tmp_path, monkeypatch):
    gate = threading.Event()
    host = FakeHost()
    host.sample_hook = lambda: gate.wait(5)
    got, first = [], threading.Event()

    def on_result(o):
        got.append(o)
        if o.ok:
            first.set()

    s = Setup(host, make_deps(tmp_path), on_result)
    performed = []
    real = s.perform
    monkeypatch.setattr(s, "perform", lambda a, args: performed.append(a) or real(a, args))
    try:
        s.start("voice_sample", {"voice": "alba"})
        deadline = time.monotonic() + 2
        while not host.samples and time.monotonic() < deadline:
            time.sleep(0.01)
        s.start("voice_sample", {"voice": "alba"})
        assert got == [Outcome("voice_sample", False, "Already running.")]
    finally:
        gate.set()
    assert first.wait(2)
    assert got[-1] == Outcome("voice_sample", True, "Played a sample of alba.")
    assert performed == ["voice_sample"] and len(host.rows) == 1 and len(host.samples) == 1
    # once it finished, the same action can run again
    s2_done = threading.Event()
    s.on_result = lambda o: s2_done.set()
    s.start("voice_sample", {"voice": "alba"})
    assert s2_done.wait(2) and performed == ["voice_sample", "voice_sample"]


def test_perform_never_raises_and_never_echoes(tmp_path):
    s, host, deps, results = make_setup(tmp_path)

    def boom(*_):
        raise RuntimeError(f"exploded with {CANARY}")

    host.probe_mic = boom
    assert s.perform("mic_check", {}) == Outcome("mic_check", False, "Something went wrong (RuntimeError).")


def test_outcome_details_are_scrubbed_and_clipped(tmp_path):
    deps = make_deps(tmp_path)
    deps.key.set(CANARY)
    o = Outcome("mic_check", False, f"line one\n{CANARY} " + "x" * 300)
    assert CANARY not in o.detail and len(o.detail) == 120 and "\n" not in o.detail


def test_a_failing_result_handler_or_log_never_breaks_the_next_action(tmp_path):
    host = FakeHost()

    def bad_log(*_):
        raise RuntimeError("db locked")

    host.log_action = bad_log
    calls = []

    def bad_result(o):
        calls.append(o)
        raise RuntimeError("socket gone")

    s = Setup(host, make_deps(tmp_path), bad_result, spawn=lambda fn: fn())
    s.start("mic_check", {})
    s.start("mic_check", {})
    assert [o.ok for o in calls] == [True, True]


# ---- the picture ----------------------------------------------------------------------------------------------------

def test_eyes_test_png():
    raw = EYES_IMAGE.read_bytes()
    assert raw.startswith(b"\x89PNG\r\n\x1a\n") and len(raw) < 20_000
    with Image.open(EYES_IMAGE) as im:
        assert im.size == (640, 200)
        gray = im.convert("L")
        assert gray.getextrema()[0] < 64 and gray.getpixel((0, 0)) == 255


def test_setup_imports_nothing_from_the_engine_or_protocol():
    src = setup_mod.__file__
    text = open(src, encoding="utf-8").read()
    assert "riya.core.engine" not in text and "riya.core.protocol" not in text and "from riya.core import engine" not in text


def test_constants():
    assert ARGS == {"online_key_set": {"key": str}, "voice_sample": {"voice": str}, "voice_choose": {"voice": str},
                    "autostart": {"on": bool}}
    assert (NONCE_TTL_S, MAX_PENDING, setup_mod.ACTION_TIMEOUT_S, setup_mod.ONLINE_TEST_TIMEOUT_S,
            setup_mod.ONLINE_TEST_TOKENS) == (60, 8, 30.0, 10.0, 200)
    assert setup_mod.NONCE_REFUSED == "That confirmation expired or was already used, so nothing ran."
    assert TESTS_KEY == "setup.last_tests"


def test_perform_outside_the_allow_list_runs_nothing(tmp_path):
    s, host, deps, _ = make_setup(tmp_path)
    assert s.perform("rm_rf", {}) == Outcome("rm_rf", False, "unknown setup action")
    assert s.perform("_rebuild", {}).ok is False and host.swaps == []


@pytest.mark.parametrize("first", ["set", "clear"])
def test_overlapping_key_set_and_clear_never_leave_a_removed_key_live(tmp_path, first):
    """One action pauses inside its model rebuild while the other runs: the models must end on the file's key."""
    other_key = "OTHER-k3y-abcdefghij"
    deps = make_deps(tmp_path)
    if first == "clear":
        deps.key.set(other_key)
    s, host, _, _ = make_setup(tmp_path, deps=deps)
    barrier, release = threading.Barrier(2, timeout=5), threading.Event()
    real = deps.make_model
    paused = []

    def make_model(env, name, timeout_s):
        if not paused:  # the first action's first build: meet the test, then wait to be let go
            paused.append(name)
            barrier.wait()
            release.wait(5)
        return real(env, name, timeout_s)

    deps.make_model = make_model
    act_first = (lambda: s.perform("online_key_set", {"key": CANARY})) if first == "set" else (
        lambda: s.perform("online_key_clear", {}))
    act_second = (lambda: s.perform("online_key_clear", {})) if first == "set" else (
        lambda: s.perform("online_key_set", {"key": CANARY}))
    a = threading.Thread(target=act_first, daemon=True)
    a.start()
    barrier.wait()  # the first action is now inside its rebuild
    b = threading.Thread(target=act_second, daemon=True)
    b.start()
    b.join(0.5)  # without the key lock the second action finishes here, inside the first one's window
    release.set()
    a.join(5)
    b.join(5)
    assert not a.is_alive() and not b.is_alive()
    saved = deps.key.current()[0]
    final = host.swaps[-1]
    assert [None if m is None else m.api_key for m in final] == [saved, saved]


@pytest.mark.parametrize("data, reason", [
    ({"disableAllHooks": True}, "disableAllHooks is true"),
    ({"allowedHttpHookUrls": ["https://example.com/*"]}, f"allowedHttpHookUrls doesn't include {ih.HOOK_URL}"),
    ({"httpHookAllowedEnvVars": ["OTHER"]}, f"httpHookAllowedEnvVars doesn't include {ih.ENV_VAR}"),
])
def test_connect_says_when_claude_code_will_block_the_hooks(tmp_path, data, reason):
    s, host, deps, results = make_setup(tmp_path)
    deps.claude_settings.parent.mkdir(parents=True)
    deps.claude_settings.write_text(json.dumps(data), encoding="utf-8")
    ask = s.request("hooks_connect", {})
    assert reason in ask.text and ask.text.endswith(ih.RESTART_LINE)
    assert s.answer(ask.nonce, True) is True
    assert results[-1] == Outcome("hooks_connect", True, f"Connected, but Claude Code will block it: {reason}")
    saved = json.loads(deps.claude_settings.read_text(encoding="utf-8"))
    assert ih.riya_events(saved) and all(saved[k] == v for k, v in data.items())  # Riya never edits those keys
    assert host.rows[-1] == ("setup", 2, "Connected Claude Code", "ran")


def test_a_hung_setx_is_reported_as_the_setx_step(tmp_path):
    import subprocess

    def hangs(cmd, **kw):
        assert kw["timeout"] == 10
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    deps = make_deps(tmp_path)
    deps.runner = hangs
    s, host, _, results = make_setup(tmp_path, deps=deps)
    s.answer(s.request("hooks_connect", {}).nonce, True)
    assert not results[-1].ok and results[-1].detail.startswith("Hooks added, but setx failed")
    s.start("hooks_connect", {})  # the slot was released: a second Connect isn't "Already running."
    assert results[-1].detail != "Already running."


def test_repr_and_str_never_hold_the_key(tmp_path):
    from riya.core.models import OpenAICompatModel
    saved = "SAVED-k3y-0123456789abcdef"
    key = OnlineKey(tmp_path / "secrets.env", CANARY, ".env")
    deps = make_deps(tmp_path, key=key, environ={KEY_VAR: CANARY, "PATH": "x"})
    deps.key.set(saved)
    s, host, _, _ = make_setup(tmp_path, deps=deps)
    model = OpenAICompatModel("online", "https://llm.example.test/v1", CANARY, "gemini")
    for obj in (deps, s, key, model):
        for text in (repr(obj), str(obj)):
            assert CANARY not in text and saved not in text, text
    assert mask(CANARY) in repr(key) and mask(CANARY) in repr(model) and mask(CANARY) in repr(deps)


def test_voice_sample_without_a_voice_samples_the_current_one(tmp_path):
    """The body sends args {} when it doesn't know her voice yet: the core samples the current voice."""
    s, host, deps, results = make_setup(tmp_path)
    host.settings.voice = "alba"
    assert s.request("voice_sample", {}) is None
    assert host.samples == [("alba", SAMPLE_TEXT)] and results[-1] == Outcome("voice_sample", True,
                                                                              "Played a sample of alba.")
    with pytest.raises(SetupError):
        s.request("voice_sample", {"voice": "alba", "x": 1})
    with pytest.raises(SetupError):
        s.request("voice_choose", {})  # choosing still needs a voice


def test_a_nonce_is_good_only_from_the_connection_it_was_issued_to(tmp_path):
    s, host, deps, results = make_setup(tmp_path)
    mine, theirs, got = [], [], []
    s.on_result = lambda o, to=None: got.append((o, to))
    to_a, to_b = mine.append, theirs.append
    ask = s.request("hooks_connect", {}, to=to_a)
    assert s.answer(ask.nonce, True, to=to_b) is False  # the other socket: refused, nothing runs
    assert s.answer(ask.nonce, True) is False  # nor in-process without the owner
    assert deps.runner.calls == [] and host.rows[-1][3] == "refused"
    assert s.answer(ask.nonce, True, to=to_a) is True  # still good for its own socket
    assert len(deps.runner.calls) == 1 and got[-1][1] is to_a and got[-1][0].action == "hooks_connect"
    s.request("mic_check", {}, to=to_b)  # a plain action's result goes back to whoever asked
    assert got[-1][1] is to_b and got[-1][0].action == "mic_check"
