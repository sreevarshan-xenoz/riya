"""evaluation/pet_check.py and evaluation/pet_cpu.py: the pure parts and the fakes-only runs. The live runs are the user's
manual steps and are never part of pytest: nothing here starts the core or Electron, runs PowerShell or touches a window."""
import json
import queue
import subprocess
from pathlib import Path

import pytest

from evaluation import pet_check, pet_cpu
from evaluation.pet_check import CWD, EXPECTED, SCRIPT, SPACING_S, payloads, verdict
from riya.agents.hub import activity_of


def test_script_labels_match_the_hub():
    assert [(a.kind, a.label) for a in (activity_of(p, 0) for p in payloads("s1"))] == EXPECTED
    assert [p["hook_event_name"] for p in payloads("s1")] == [ev for ev, _ in SCRIPT]
    assert all(p["session_id"] == "s1" and p["cwd"] == CWD for p in payloads("s1"))
    assert SPACING_S == 2.0 and SPACING_S > 1.5  # past the per-session bubble gap, so none is held


def test_verdict():
    noisy = []
    for pair in EXPECTED:
        noisy += [("run", "running pytest"), pair]
    assert verdict(noisy) == "VERDICT: ok"
    assert verdict(list(EXPECTED)) == "VERDICT: ok"
    no_wave = [p for p in EXPECTED if p[0] != "wave"]
    assert verdict(no_wave) == "VERDICT: missing ('wave', 'needs you! running tests')"
    assert verdict(list(reversed(EXPECTED))).startswith("VERDICT: missing")  # order matters
    assert verdict([]) == f"VERDICT: missing {EXPECTED[0]}"
    assert verdict(EXPECTED + [("run", "x" * 41)]) == "VERDICT: a label over 40 chars"
    assert verdict(EXPECTED + [("run", "x" * 40)]) == "VERDICT: ok"


def test_pet_check_never_touches_claude_code():
    for name in ("pet_check.py", "pet_cpu.py"):
        src = (Path(pet_check.__file__).parent / name).read_text(encoding="utf-8")
        assert "install_hooks" not in src and "settings.json" not in src and "setx" not in src
    repo = Path(__file__).resolve().parents[1]
    assert CWD == "C:/riya-pet-check" and not (repo / CWD).exists() and not Path(CWD).exists()
    assert not CWD.lower().startswith(str(repo).replace("\\", "/").lower())


def test_cpu_percent_and_verdict():
    assert pet_cpu.cpu_percent({1: 10.0, 2: 5.0}, {1: 10.6, 2: 5.6, 3: 9.0}, 60, 8) == pytest.approx(0.25)
    assert pet_cpu.cpu_percent({1: 10.0, 2: 5.0}, {1: 9.0, 2: 5.6}, 60, 8) == pytest.approx(100 * 0.6 / 480)
    assert pet_cpu.cpu_percent({1: 1.0}, {2: 5.0}, 60, 8) == 0.0
    assert pet_cpu.BUDGET == {"idle": 1.0, "walk": 3.0}
    assert pet_cpu.cpu_verdict("walk", 2.9) == "VERDICT: walk 2.90% (budget 3.0%) ok"
    assert pet_cpu.cpu_verdict("idle", 1.2).endswith("OVER")
    assert pet_cpu.cpu_verdict("idle", 1.2) == "VERDICT: idle 1.20% (budget 1.0%) OVER"
    assert pet_cpu.parse_ps_json('{"Id":1,"CPU":2.5,"Path":"x"}') == [{"Id": 1, "CPU": 2.5, "Path": "x"}]
    assert pet_cpu.parse_ps_json('[{"Id":1},{"Id":2}]') == [{"Id": 1}, {"Id": 2}]
    assert pet_cpu.parse_ps_json("") == [] and pet_cpu.parse_ps_json("not json") == [] and pet_cpu.parse_ps_json("5") == []


def ps_rows():
    mine = str(pet_cpu.REPO_ROOT / "body" / "node_modules" / "electron" / "dist" / "electron.exe")
    other = r"C:\Users\someone\AppData\Local\Programs\Slack\electron.exe"
    sneaky = str(pet_cpu.REPO_ROOT / "body" / "node_modules_x" / "electron.exe")
    return [{"Id": 11, "CPU": 3.5, "Path": mine}, {"Id": 12, "CPU": 1.25, "Path": mine.upper()},
            {"Id": 20, "CPU": 99.0, "Path": other}, {"Id": 21, "CPU": 9.0, "Path": None},
            {"Id": 22, "CPU": 9.0, "Path": sneaky}, {"Id": 23, "CPU": None, "Path": mine}]


def test_body_cpu_seconds_keeps_only_the_bodys_electron():
    calls = []

    def fake(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(ps_rows()), stderr="")

    assert pet_cpu.body_cpu_seconds(run=fake) == {11: 3.5, 12: 1.25}
    cmd = calls[0][0]
    assert cmd[:3] == ["powershell", "-NoProfile", "-Command"] and "Get-Process electron" in cmd[3]
    assert "ConvertTo-Json" in cmd[3] and calls[0][1].get("capture_output") is True
    assert not calls[0][1].get("shell") and isinstance(cmd, list)


def test_body_cpu_seconds_survives_a_failing_shell():
    def boom(cmd, **kw):
        raise OSError("no powershell")

    assert pet_cpu.body_cpu_seconds(run=boom) == {}
    one = lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(ps_rows()[0]), stderr="")
    assert pet_cpu.body_cpu_seconds(run=one) == {11: 3.5}


def test_pet_cpu_main_samples_twice_and_always_exits_0(capsys):
    samples = iter([{11: 10.0, 12: 5.0}, {11: 10.6, 12: 5.6}])
    slept = []
    rc = pet_cpu.main(["--mode", "walk", "--seconds", "60"], sample=lambda: next(samples), sleep=slept.append, cores=8,
                      clock=iter([100.0, 160.0]).__next__)
    out = capsys.readouterr().out
    assert rc == 0 and slept == [60.0]
    assert "RIYA_PET_FORCE_WALK=1" in out and out.strip().splitlines()[-1] == "VERDICT: walk 0.25% (budget 3.0%) ok (2 of 2 pids counted)"
    # a process that ended mid-run isn't counted, and the line says so
    samples = iter([{11: 10.0, 12: 5.0, 13: 1.0}, {11: 10.6, 14: 2.0}])
    pet_cpu.main(["--mode", "idle", "--seconds", "60"], sample=lambda: next(samples), sleep=lambda s: None, cores=8,
                 clock=iter([0.0, 60.0]).__next__)
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith("(1 of 4 pids counted)")
    # no body running: say so, still exit 0, never a verdict that looks like a pass
    assert pet_cpu.main(["--mode", "idle", "--seconds", "1"], sample=lambda: {}, sleep=lambda s: None, cores=4,
                        clock=iter([0.0, 1.0]).__next__) == 0
    assert "no Riya body" in capsys.readouterr().out.strip().splitlines()[-1]


class FakeWS:
    """Replays core messages to the watcher's thread; close() ends the stream like a closed socket."""

    def __init__(self):
        self.queue, self.closed = queue.Queue(), False

    def feed(self, msg):
        self.queue.put(json.dumps(msg))

    def __iter__(self):
        while (raw := self.queue.get()) is not None:
            yield raw

    def close(self):
        self.closed = True
        self.queue.put(None)


TOKEN = "SECRET-TOKEN-123"


def run_with_fakes(monkeypatch, capsys, drop=(), extra_session=False, token=TOKEN):
    posted, ws = [], FakeWS()

    def fake_post(tok, payload, timeout_s=10.0):
        posted.append((tok, payload))
        act = activity_of(payload, 0.0)
        if act and (len(posted) - 1) not in drop:
            ws.feed({"type": "agent_activity", "session": act.session, "label": act.label, "kind": act.kind, "at": 1.0})
        if len(posted) == 1:
            ws.feed({"type": "presence", "state": "break"})
        if extra_session and len(posted) == 1:  # another session's bubbles: all five expected ones, and a too-long one
            for kind, label in EXPECTED + [("run", "x" * 41)]:
                ws.feed({"type": "agent_activity", "session": "other", "label": label, "kind": kind, "at": 1.0})
        return "200"

    slept = []
    rc = pet_check.main([], connect=lambda deadline_s, proc=None: ws, post=fake_post, token_for=lambda s: token,
                        sleep=slept.append, wait_s=1.0 if not drop else 0.5)
    out = capsys.readouterr().out
    assert token not in out  # the hook token is only ever sent as a header, never printed
    return rc, posted, slept, out, ws


def test_main_posts_the_script_and_prints_the_bubbles(monkeypatch, capsys):
    rc, posted, slept, out, ws = run_with_fakes(monkeypatch, capsys)
    assert rc == 0 and ws.closed
    assert [p["hook_event_name"] for _, p in posted] == [ev for ev, _ in SCRIPT]
    assert {tok for tok, _ in posted} == {TOKEN} and len({p["session_id"] for _, p in posted}) == 1
    assert posted[0][1]["session_id"].startswith("riya-pet-check-")
    assert slept.count(SPACING_S) == len(SCRIPT) - 1
    lines = out.strip().splitlines()
    assert lines[-1] == "VERDICT: ok"
    assert sum(" edit: editing engine.py" in ln for ln in lines) == 1 and any("wave: needs you! running tests" in ln for ln in lines)
    assert any("presence: break" in ln for ln in lines)


def test_main_names_the_first_missing_bubble(monkeypatch, capsys):
    rc, *_, out, _ws = run_with_fakes(monkeypatch, capsys, drop={3})  # the PermissionRequest causes no wave
    assert rc == 0 and out.strip().splitlines()[-1] == "VERDICT: missing ('wave', 'needs you! running tests')"


def test_main_ignores_other_sessions_and_reports_a_missing_core(monkeypatch, capsys):
    # the real session drops its wave: counting the other session's bubbles would turn this into ok (or over 40)
    rc, *_, out, _ = run_with_fakes(monkeypatch, capsys, drop={3}, extra_session=True)
    assert rc == 0 and out.strip().splitlines()[-1] == "VERDICT: missing ('wave', 'needs you! running tests')"
    assert "x" * 41 not in out and "other" not in out
    assert pet_check.main([], connect=lambda deadline_s, proc=None: None, sleep=lambda s: None) == 0
    assert "not run" in capsys.readouterr().out
    assert pet_check.main([], connect=lambda deadline_s, proc=None: FakeWS(), token_for=lambda s: None,
                          sleep=lambda s: None) == 0
    assert "no hook token" in capsys.readouterr().out
