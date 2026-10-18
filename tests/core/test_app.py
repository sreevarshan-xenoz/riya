import argparse
import asyncio
import threading

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from riya.core import __main__ as runner
from riya.core.__main__ import DEFAULT_HOST
from riya.core.app import Hub, create_app
from riya.core.clock import FakeClock
from riya.core.engine import Utterance
from riya.core.listen import ListenGate
from riya.core.presence import FakeIdleSource
from riya.core.protocol import ExpressionMsg
from tests.fakes import make_core


def client(replies=("Hi there.",), speak=None, delay=0.0):
    def factory(emit):
        core, _, _, local = make_core(local_replies=list(replies))
        core.emit, local.delay = emit, delay
        return core
    return TestClient(create_app(factory, speak=speak))


def receive_until(ws, type_):
    seen = []
    while True:
        m = ws.receive_json()
        seen.append(m["type"])
        if m["type"] == type_:
            return seen


def test_hello_then_user_text_round_trip():
    with client() as c, c.websocket_connect("/ws") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"type": "user_text", "text": "hi"})
        assert "say" in receive_until(ws, "say_end")


def test_bad_message_returns_error_and_socket_stays_open():
    with client() as c, c.websocket_connect("/ws") as ws:
        ws.send_text("garbage")
        assert "error" in receive_until(ws, "error")
        ws.send_json({"type": "user_text", "text": "still there?"})
        assert "say" in receive_until(ws, "say_end")


def test_typed_replies_go_through_the_speaker():
    heard, done = [], threading.Event()

    def speak(sentences):
        heard.append(list(sentences))
        done.set()

    with client(replies=("Hello you. Nice day.",), speak=speak) as c, c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "user_text", "text": "hi"})
        receive_until(ws, "say_end")
        assert done.wait(2)
    assert heard == [["Hello you.", "Nice day."]]


def test_other_messages_do_not_wait_behind_a_slow_reply():
    with client(replies=("One two three four five.",), delay=0.1) as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "state")
        ws.send_json({"type": "user_text", "text": "tell me something"})
        ws.send_json({"type": "get_memories"})
        seen = receive_until(ws, "say_end")
        assert "memories" in seen and seen.index("memories") < seen.index("say_end")


def test_typed_replies_on_one_socket_keep_their_order():
    with client(replies=("First.", "Second."), delay=0.05) as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "state")
        ws.send_json({"type": "user_text", "text": "one"})
        ws.send_json({"type": "user_text", "text": "two"})
        receive_until(ws, "say_end")
        receive_until(ws, "say_end")
        asked = [messages[-1]["content"] for messages in c.app.state.core.local.calls]
    assert asked == ["one", "two"]


def test_binds_localhost_only():
    assert DEFAULT_HOST == "127.0.0.1"


def test_non_text_messages_go_to_the_core_and_state_is_exposed():
    with client() as c, c.websocket_connect("/ws") as ws:
        assert receive_until(ws, "state") == ["hello", "settings", "state"]
        ws.send_json({"type": "get_memories"})
        assert "memories" in receive_until(ws, "memories")
        assert c.app.state.core.hello_messages()[0].type == "hello"


def test_every_connection_gets_hub_broadcasts_and_disconnect_is_cleaned_up():
    with client() as c:
        hub = c.app.state.hub
        with c.websocket_connect("/ws") as a, c.websocket_connect("/ws") as b:
            for ws in (a, b):
                receive_until(ws, "state")
            hub.publish(ExpressionMsg(name="happy"))
            assert receive_until(a, "expression") == ["expression"]
            assert receive_until(b, "expression") == ["expression"]
        c.portal.call(asyncio.sleep, 0.05)
        assert hub.connections == 0


def test_hub_publish_before_attach_is_silent_and_works_from_threads():
    hub = Hub()
    hub.publish(ExpressionMsg(name="happy"))  # no loop yet: dropped, never raises

    async def scenario():
        hub.attach(asyncio.get_running_loop())
        q = hub.connect()
        threading.Thread(target=hub.publish, args=(ExpressionMsg(name="sad"),)).start()
        assert (await asyncio.wait_for(q.get(), 2)).name == "sad"
        hub.disconnect(q)
        assert hub.connections == 0

    asyncio.run(scenario())
    hub.publish(ExpressionMsg(name="happy"))  # loop now closed: still silent


def test_build_core_imports_legacy_memory_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "make_idle_source", FakeIdleSource)
    legacy = tmp_path / "riya_memory.json"
    legacy.write_text('{"summary": "We talked about cats."}', encoding="utf-8")
    args = argparse.Namespace(db=str(tmp_path / "d" / "riya.db"), local_url="http://127.0.0.1:1", hash_embed=True,
                              legacy_memory=str(legacy))
    core = runner.build_core(args, lambda m: None)
    assert [e.summary for e in core.memory.recent_episodes()] == ["We talked about cats."]
    assert core.kv.get("legacy_imported") is True
    core.kv.conn.close()
    again = runner.build_core(args, lambda m: None)
    assert len(again.memory.recent_episodes()) == 1


def test_main_help_lists_options(capsys):
    try:
        runner.parse_args(["--help"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    for flag in ("--db", "--local-url", "--port", "--no-voice", "--hash-embed", "--legacy-memory"):
        assert flag in out


class FakeVoiceLoop:
    def __init__(self):
        self.calls, self.gate = [], ListenGate(FakeClock())

    def answer(self, sentences, reason="typed"):
        self.calls.append((list(sentences), reason))


def test_deliver_drains_without_voice_and_speaks_first_with_it():
    steps = []

    def greeting():
        steps.append("start")
        yield "Welcome back!"
        steps.append("end")

    runner.deliver(greeting(), None)
    assert steps == ["start", "end"]
    loop = FakeVoiceLoop()
    runner.deliver(Utterance(iter(["Welcome back!"]), spoken=True), loop)
    assert loop.calls == [(["Welcome back!"], "spoke_first")]


def test_deliver_never_voices_an_utterance_that_is_not_spoken():
    loop, silent = FakeVoiceLoop(), Utterance(iter(["psst"]), spoken=False)
    runner.deliver(silent, loop)
    assert loop.calls == [] and list(silent) == [] and loop.gate.is_open()


def test_speaker_drains_typed_replies_until_the_voice_loop_is_ready():
    speaker = runner.Speaker()
    early = iter(["a", "b"])
    speaker(early)
    assert list(early) == []
    speaker.loop = FakeVoiceLoop()
    speaker(iter(["c"]))
    assert speaker.loop.calls == [(["c"], "typed")]


# --- final review: the socket only talks to Riya's own body ---------------------------------------------------------

def test_websocket_from_a_web_page_origin_is_rejected_without_hello():
    with client() as c:
        with pytest.raises(WebSocketDisconnect) as e:
            with c.websocket_connect("/ws", headers={"origin": "https://evil.example"}) as ws:
                ws.receive_json()
        assert e.value.code == 1008


def test_websocket_without_origin_or_from_a_file_page_is_accepted():
    with client() as c:
        for headers in ({}, {"origin": "file://"}, {"origin": "null"}):
            with c.websocket_connect("/ws", headers=headers) as ws:
                assert ws.receive_json()["type"] == "hello"


def test_deliver_never_opens_the_mic_for_a_silent_utterance_while_gaming():
    loop, silent = FakeVoiceLoop(), Utterance(iter(["psst"]), spoken=False)
    runner.deliver(silent, loop, gaming=True)
    assert loop.calls == [] and list(silent) == [] and not loop.gate.is_open()
