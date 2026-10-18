import json
from types import SimpleNamespace

import pytest

from riya.agents import install_hooks as ih

HANDLER = {"type": "http", "url": "http://127.0.0.1:8766/hook", "timeout": 3,
           "headers": {"X-Riya-Token": "$RIYA_HOOK_TOKEN"}, "allowedEnvVars": ["RIYA_HOOK_TOKEN"]}
MATCHERS = {"Notification": "permission_prompt|idle_prompt", "PostToolUseFailure": "*",
            "PermissionRequest": "*", "SubagentStop": "*"}
EVENTS = {"UserPromptSubmit", "PermissionRequest", "Notification", "Stop", "StopFailure",
          "PostToolUseFailure", "SubagentStop", "SessionStart", "SessionEnd"}
MINE = {"type": "command", "command": "echo hi"}


class Runner:
    def __init__(self, rc=0):
        self.calls, self.rc = [], rc

    def __call__(self, cmd, **kw):
        self.calls.append((cmd, kw))
        return SimpleNamespace(returncode=self.rc)


@pytest.fixture
def env(tmp_path):
    return SimpleNamespace(settings=tmp_path / "claude" / "settings.json", token=tmp_path / "tok" / "hook_token",
                           runner=Runner(), out=[], tmp=tmp_path)


def run(env, *flags, runner=None):
    return ih.main([*flags, "--settings", str(env.settings)], runner=runner or env.runner,
                   token_path=env.token, out=env.out.append)


def load(env):
    return json.loads(env.settings.read_text(encoding="utf-8"))


def backups(env):
    return sorted(env.settings.parent.glob("settings.json.riya-backup-*"))


def test_apply_creates_missing_file_with_all_events(env):
    assert run(env, "--apply") == 0
    hooks = load(env)["hooks"]
    assert set(hooks) == EVENTS
    for event, groups in hooks.items():
        assert len(groups) == 1 and groups[0]["hooks"] == [HANDLER]
        if event in MATCHERS:
            assert groups[0]["matcher"] == MATCHERS[event]
        else:
            assert "matcher" not in groups[0]
    assert backups(env) == []  # nothing existed to back up


def test_apply_sets_env_var_via_runner_without_shell(env):
    run(env, "--apply")
    token = env.token.read_text(encoding="utf-8").strip()
    assert len(token) == 64
    assert env.runner.calls[0][0] == ["setx", "RIYA_HOOK_TOKEN", token]
    assert not env.runner.calls[0][1].get("shell")
    assert any("restart" in line.lower() and "terminal" in line.lower() for line in env.out)
    assert not any(token in line for line in env.out)


def test_apply_preserves_other_hooks_keys_and_order(env):
    original = {"model": "opus", "hooks": {"Stop": [{"hooks": [MINE]}], "PreToolUse": [{"matcher": "Bash", "hooks": [MINE]}]},
                "permissions": {"allow": ["Bash(ls)"]}}
    env.settings.parent.mkdir(parents=True)
    env.settings.write_text(json.dumps(original), encoding="utf-8")
    assert run(env, "--apply") == 0
    got = load(env)
    assert list(got) == ["model", "hooks", "permissions"]
    assert got["permissions"] == original["permissions"] and got["model"] == "opus"
    assert got["hooks"]["PreToolUse"] == original["hooks"]["PreToolUse"]
    assert got["hooks"]["Stop"][0] == {"hooks": [MINE]}
    assert got["hooks"]["Stop"][1] == {"hooks": [HANDLER]}
    assert list(got["hooks"])[:2] == ["Stop", "PreToolUse"]


def test_apply_is_idempotent(env):
    run(env, "--apply")
    first = env.settings.read_text(encoding="utf-8")
    assert run(env, "--apply") == 0
    assert env.settings.read_text(encoding="utf-8") == first


def test_apply_does_not_duplicate_an_existing_riya_handler_in_a_foreign_group(env):
    original = {"hooks": {"Stop": [{"hooks": [MINE, HANDLER]}]}}
    env.settings.parent.mkdir(parents=True)
    env.settings.write_text(json.dumps(original), encoding="utf-8")
    run(env, "--apply")
    assert load(env)["hooks"]["Stop"] == original["hooks"]["Stop"]


def test_backup_written_when_file_existed(env):
    env.settings.parent.mkdir(parents=True)
    raw = json.dumps({"model": "x"})
    env.settings.write_text(raw, encoding="utf-8")
    run(env, "--apply")
    (b,) = backups(env)
    assert b.read_text(encoding="utf-8") == raw
    assert len(b.name.rsplit("riya-backup-", 1)[1]) == len("20261018-120000")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '{"hooks": []}', '{"hooks": {"Stop": 5}}'])
def test_unusable_settings_abort_without_writing(env, content):
    env.settings.parent.mkdir(parents=True)
    env.settings.write_text(content, encoding="utf-8")
    assert run(env, "--apply") == 1
    assert env.settings.read_text(encoding="utf-8") == content
    assert backups(env) == [] and env.runner.calls == []
    assert any("nothing was changed" in line for line in env.out)
    assert run(env, "--remove") == (1 if content != '{"hooks": {"Stop": 5}}' else 0)
    assert env.settings.read_text(encoding="utf-8") == content


def test_remove_restores_original(env):
    original = {"model": "opus", "hooks": {"Stop": [{"hooks": [MINE]}], "PreToolUse": [{"matcher": "Bash", "hooks": [MINE]}]}}
    env.settings.parent.mkdir(parents=True)
    env.settings.write_text(json.dumps(original), encoding="utf-8")
    run(env, "--apply")
    assert run(env, "--remove") == 0
    assert load(env) == original
    assert list(load(env)["hooks"]) == ["Stop", "PreToolUse"]
    assert len(backups(env)) >= 1
    assert any("reg delete HKCU\\Environment /v RIYA_HOOK_TOKEN /f" in line for line in env.out)
    assert len(env.runner.calls) == 1  # remove never touches the env var


def test_remove_drops_empty_groups_events_and_hooks(env):
    run(env, "--apply")
    assert run(env, "--remove") == 0
    assert load(env) == {}


def test_remove_keeps_foreign_handler_in_shared_group(env):
    original = {"hooks": {"Stop": [{"hooks": [MINE, HANDLER]}]}}
    env.settings.parent.mkdir(parents=True)
    env.settings.write_text(json.dumps(original), encoding="utf-8")
    run(env, "--remove")
    assert load(env) == {"hooks": {"Stop": [{"hooks": [MINE]}]}}


def test_remove_leaves_foreign_empty_structures_alone(env):
    original = {"hooks": {"Stop": [], "Other": [{"hooks": []}]}}
    env.settings.parent.mkdir(parents=True)
    env.settings.write_text(json.dumps(original), encoding="utf-8")
    run(env, "--remove")
    assert load(env) == original
    assert backups(env) == []  # nothing to remove, no churn


def test_remove_on_missing_file_is_a_noop(env):
    assert run(env, "--remove") == 0
    assert not env.settings.exists()


def test_print_is_the_default_and_never_shows_token_or_writes(env):
    env.token.parent.mkdir(parents=True)
    env.token.write_text("s3cr3t-token-value", encoding="utf-8")
    assert run(env) == 0
    text = "\n".join(env.out)
    assert "s3cr3t-token-value" not in text
    assert str(env.token) in text and "http://127.0.0.1:8766/hook" in text
    assert json.loads(text[: text.index("\n\nToken file")]) == ih.snippet()
    assert not env.settings.exists() and env.runner.calls == []
    assert run(env, "--print") == 0


def test_setx_failure_is_reported(env):
    assert run(env, "--apply", runner=Runner(rc=1)) == 1
    assert set(load(env)["hooks"]) == EVENTS  # settings still written
    assert any("setx failed" in line for line in env.out)


def test_no_leftover_temp_files(env):
    run(env, "--apply")
    assert [p.name for p in env.settings.parent.iterdir()] == ["settings.json"]


def test_utf8_roundtrip(env):
    env.settings.parent.mkdir(parents=True)
    env.settings.write_bytes('{"note": "café ✓"}'.encode("utf-8"))
    run(env, "--apply")
    assert load(env)["note"] == "café ✓"


def put(env, data):
    env.settings.parent.mkdir(parents=True, exist_ok=True)
    env.settings.write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize("flag", ["--print", "--apply"])
@pytest.mark.parametrize("key,value,needle", [
    ("allowedHttpHookUrls", ["https://example.com/*"], "allowedHttpHookUrls"),
    ("httpHookAllowedEnvVars", ["OTHER"], "httpHookAllowedEnvVars"),
    ("disableAllHooks", True, "disableAllHooks"),
])
def test_blocking_settings_warn_and_are_left_alone(env, flag, key, value, needle):
    put(env, {key: value})
    run(env, flag)
    assert any(line.startswith("Warning") and needle in line for line in env.out)
    assert load(env)[key] == value


@pytest.mark.parametrize("data", [
    {"allowedHttpHookUrls": ["http://127.0.0.1:8766/hook"], "httpHookAllowedEnvVars": ["RIYA_HOOK_TOKEN"]},
    {"allowedHttpHookUrls": ["http://127.0.0.1:*"]}, {"disableAllHooks": False}, {}])
def test_no_warning_when_settings_allow_riya(env, data):
    put(env, data)
    run(env, "--apply")
    run(env, "--print")
    assert not any(line.startswith("Warning") for line in env.out)


def test_handler_never_has_on_failure(env):
    assert "onFailure" not in ih.HANDLER
    run(env, "--apply")
    assert all("onFailure" not in h for groups in load(env)["hooks"].values() for g in groups for h in g["hooks"])


def test_apply_mentions_hooks_command_in_restart_note(env):
    run(env, "--apply")
    assert any("/hooks" in line and "restart" in line.lower() for line in env.out)
