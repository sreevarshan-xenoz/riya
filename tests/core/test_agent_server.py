"""Kilo's free models through its local server: ask mode, tools off, a quiet retry caught as a limit."""
import http.server
import json
import threading

import pytest

from riya.core import agent_server, models, pool
from riya.core.agent_server import AgentServer, ServerAgentModel, flatten
from riya.core.models import ModelUnavailable


def test_flatten_keeps_the_system_and_folds_the_turns():
    msgs = [{"role": "system", "content": "be kind"}, {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hey"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {}}]}]
    system, prompt = flatten(msgs)
    assert system == "be kind" and prompt == "User: hi\nRiya: hey\nUser: look\nRiya:"
    assert flatten([{"role": "user", "content": "just this"}]) == ("", "just this")


class FakeServer(http.server.BaseHTTPRequestHandler):
    """Sessions, tool ids, a message that answers or retries forever, status, abort and delete."""
    mode = "answer"
    seen: list = []

    def _send(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/experimental/tool/ids":
            self._send(["bash", "edit", "write"])
        elif self.path == "/session/status":
            self._send({"ses_1": {"type": "retry", "attempt": 2, "message": "Free usage limit reached."}}
                       if FakeServer.mode == "retry" else {})
        else:
            self._send({})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        FakeServer.seen.append((self.path, body))
        if self.path == "/session":
            self._send({"id": "ses_1"})
        elif self.path.endswith("/message"):
            if FakeServer.mode == "retry":
                threading.Event().wait(3)  # a CLI retrying: no answer for a long time
            self._send({"info": {}, "parts": [{"type": "reasoning", "text": "hmm"}, {"type": "text", "text": "Hi!"}]})
        else:
            self._send(True)

    def do_DELETE(self):
        FakeServer.seen.append((self.path, None))
        self._send(True)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    s = AgentServer("kilo", which=lambda name: "/bin/true")
    s.base, s.tools_off = f"http://127.0.0.1:{srv.server_address[1]}", {"bash": False, "edit": False}

    class Alive:
        def poll(self):
            return None
    s.proc = Alive()
    FakeServer.mode, FakeServer.seen = "answer", []
    yield s
    srv.shutdown()


def test_a_reply_uses_ask_mode_with_tools_off_and_deletes_its_session(server):
    m = ServerAgentModel("online", server, "kilo", "kilo-auto/free")
    out = "".join(m.stream([{"role": "system", "content": "You are Riya."}, {"role": "user", "content": "hi"}]))
    assert out == "Hi!" and m.finish_reason == "stop"
    path, body = next(x for x in FakeServer.seen if x[0].endswith("/message"))
    assert body["agent"] == "ask" and body["tools"] == {"bash": False, "edit": False}
    assert body["system"] == "You are Riya." and body["model"] == {"providerID": "kilo", "modelID": "kilo-auto/free"}
    assert ("/session/ses_1", None) in FakeServer.seen


def test_a_quiet_retry_is_aborted_and_read_as_a_limit(server):
    FakeServer.mode = "retry"
    m = ServerAgentModel("online", server, "kilo", "kilo-auto/free")
    with pytest.raises(ModelUnavailable) as got:
        list(m.stream([{"role": "user", "content": "hi"}]))
    assert pool.kind_of(got.value) == "limit"
    assert any(p.endswith("/abort") for p, _ in FakeServer.seen)


def test_kilo_members_are_slow_toolless_and_only_when_installed(monkeypatch):
    monkeypatch.setattr(agent_server, "_servers", {})
    monkeypatch.setattr(agent_server.shutil, "which", lambda name: None)
    assert agent_server.kilo_members("online") == []
    monkeypatch.setattr(agent_server, "_servers", {})
    monkeypatch.setattr(agent_server.shutil, "which", lambda name: "/usr/bin/kilo")
    ms = agent_server.kilo_members("online")
    assert ms and all(m.slow and not m.tools and m.provider == "kilo" for m in ms)


def test_the_pool_puts_kilo_after_every_fast_member_and_never_in_her_eyes(monkeypatch):
    monkeypatch.setattr(agent_server, "_servers", {})
    monkeypatch.setattr(agent_server.shutil, "which", lambda name: "/usr/bin/kilo")
    online, vision = models.online_pair({})
    kinds = [m.slow for m in online.members]
    assert any(kinds) and kinds == sorted(kinds)
    assert not any(m.provider == "kilo" for m in vision.members)
    off, _ = models.online_pair({"RIYA_NO_AGENTS": "1"})
    assert not any(m.provider == "kilo" for m in off.members)
