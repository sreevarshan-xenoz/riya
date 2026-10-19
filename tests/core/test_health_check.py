"""evaluation/health_check.py: the pure parts. The live run is not part of pytest."""
import json
from pathlib import Path

from evaluation import health_check

FIXTURES = Path(__file__).resolve().parents[2] / "body" / "test" / "fixtures"
HELLO = ["hello", "settings", "state", "agents", "eyes", "learning", "reminders", "actions"]


def health_fixture() -> dict:
    return json.loads((FIXTURES / "core_health.json").read_text(encoding="utf-8"))


class FakeWs:
    """A sync connection that serves canned frames and records what is sent."""

    def __init__(self, frames):
        self.frames, self.sent, self.closed = list(frames), [], False

    def send(self, data):
        self.sent.append(json.loads(data))

    def recv(self, timeout=None):
        if not self.frames:
            raise TimeoutError("nothing more")
        return json.dumps(self.frames.pop(0))

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def test_fetch_sends_only_health_get():
    frames = [{"type": t} for t in HELLO] + [{"type": "voices", "items": []}, health_fixture()]
    ws = FakeWs(frames)
    seen = {}

    def connect(url, **kw):
        seen["url"], seen["kw"] = url, kw
        return ws

    assert health_check.fetch(health_check.WS_URL, 5.0, connect=connect) == health_fixture()
    assert ws.sent == [{"type": "health_get"}]
    assert "additional_headers" not in seen["kw"] and "origin" not in seen["kw"]
    assert ws.closed


def test_fetch_is_none_when_health_never_comes():
    ws = FakeWs([{"type": t} for t in HELLO])
    assert health_check.fetch(health_check.WS_URL, 0.5, connect=lambda url, **kw: ws) is None
    assert ws.sent == [{"type": "health_get"}]


def test_format_table():
    lines = health_check.format_table(health_fixture())
    assert len(lines) == 5
    assert lines[0].startswith(f"{'Online brain':<18} warn  Key ••••1a2b saved in Riya")
    assert lines[1] == f"{'Claude Code':<18} off   Not connected."
    assert lines[2].startswith(f"{'Look':<18} warn  Placeholder art")
    assert lines[3] == "Key: app"
    assert lines[4] == "VERDICT: 0 ok, 2 warn, 1 off"


def test_format_table_with_no_key_source():
    lines = health_check.format_table({"type": "health", "key_source": None, "items": []})
    assert lines == ["Key: none", "VERDICT: 0 ok, 0 warn, 0 off"]


def test_no_core_exits_zero(capsys):
    def connect(url, **kw):
        raise OSError("refused")

    assert health_check.main(["--timeout", "1"], connect=connect) == 0
    out = capsys.readouterr().out
    assert f"No core at {health_check.WS_URL}: start it with python -m riya.core" in out


def test_main_prints_the_table(capsys):
    frames = [{"type": t} for t in HELLO] + [health_fixture()]
    assert health_check.main([], connect=lambda url, **kw: FakeWs(frames)) == 0
    out = capsys.readouterr().out
    assert "VERDICT: 0 ok, 2 warn, 1 off" in out and "Key: app" in out


def test_the_check_never_sends_setup():
    text = Path(health_check.__file__).read_text(encoding="utf-8")
    for word in ("setup_request", "setup_action", "onboarding_done"):
        assert word not in text
