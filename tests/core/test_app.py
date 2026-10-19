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


@pytest.fixture(autouse=True)
def no_leftover_redaction():
    """runner.main installs the key redaction for the whole process: take it off again after each test."""
    from riya.core import secrets as keystore
    yield
    keystore.uninstall_redaction()


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
        receive_until(ws, "actions")  # the end of hello
        ws.send_json({"type": "user_text", "text": "tell me something"})
        ws.send_json({"type": "get_memories"})
        seen = receive_until(ws, "say_end")
        assert "memories" in seen and seen.index("memories") < seen.index("say_end")


def test_typed_replies_on_one_socket_keep_their_order():
    with client(replies=("First.", "Second."), delay=0.05) as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "actions")  # the end of hello
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
        assert receive_until(ws, "actions") == ["hello", "settings", "state", "agents", "eyes", "learning",
                                                "reminders", "actions"]
        ws.send_json({"type": "get_memories"})
        assert "memories" in receive_until(ws, "memories")
        assert c.app.state.core.hello_messages()[0].type == "hello"


def test_every_connection_gets_hub_broadcasts_and_disconnect_is_cleaned_up():
    with client() as c:
        hub = c.app.state.hub
        with c.websocket_connect("/ws") as a, c.websocket_connect("/ws") as b:
            for ws in (a, b):
                receive_until(ws, "actions")  # the end of hello
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
    for flag in ("--db", "--local-url", "--port", "--no-voice", "--hash-embed", "--legacy-memory", "--secrets"):
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
        for origin in ("https://evil.example", "null"):  # "null": a sandboxed iframe on any web page
            with pytest.raises(WebSocketDisconnect) as e:
                with c.websocket_connect("/ws", headers={"origin": origin}) as ws:
                    ws.receive_json()
            assert e.value.code == 1008, origin


def test_websocket_without_origin_or_from_a_file_page_is_accepted():
    with client() as c:
        for headers in ({}, {"origin": "file://"}):
            with c.websocket_connect("/ws", headers=headers) as ws:
                assert ws.receive_json()["type"] == "hello"


def test_deliver_never_opens_the_mic_for_a_silent_utterance_while_gaming():
    loop, silent = FakeVoiceLoop(), Utterance(iter(["psst"]), spoken=False)
    runner.deliver(silent, loop, gaming=True)
    assert loop.calls == [] and list(silent) == [] and not loop.gate.is_open()


# --- final review: startup never depends on the network or the working directory ------------------------------------

def core_args(tmp_path, **kw):
    return argparse.Namespace(**{"db": str(tmp_path / "d" / "riya.db"), "local_url": "http://127.0.0.1:1",
                                 "hash_embed": False, "legacy_memory": "riya_memory.json", **kw})


def test_build_core_falls_back_to_the_hash_embedder_when_bge_cannot_load(tmp_path, monkeypatch, caplog):
    from riya.core import embed
    from riya.core.embed import HashEmbedder

    class Offline:
        def __init__(self, *a, **kw):
            raise OSError("no network and no cached model")
    monkeypatch.setattr(embed, "FastEmbedder", Offline)
    monkeypatch.setattr(runner, "make_idle_source", FakeIdleSource)
    monkeypatch.chdir(tmp_path)
    with caplog.at_level("WARNING"):
        core = runner.build_core(core_args(tmp_path), lambda m: None)
    assert isinstance(core.memory.embedder, HashEmbedder)
    assert any("embedder" in r.getMessage() for r in caplog.records)


def test_fast_embedder_keeps_its_model_in_riya_model_dir(tmp_path, monkeypatch):
    import sys
    import types

    from riya.core import embed
    seen = {}

    class TextEmbedding:
        def __init__(self, model, cache_dir=None):
            seen["cache_dir"] = cache_dir

        def embed(self, texts):
            return iter([[1.0, 0.0]] * len(texts))
    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=TextEmbedding))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    monkeypatch.delenv("RIYA_MODEL_DIR", raising=False)
    embed.FastEmbedder()
    assert seen["cache_dir"] == str(tmp_path / "local" / "Riya" / "models")
    monkeypatch.setenv("RIYA_MODEL_DIR", str(tmp_path / "mine"))
    embed.FastEmbedder()
    assert seen["cache_dir"] == str(tmp_path / "mine")


def test_legacy_memory_is_found_from_the_repo_root_and_only_marked_once_imported(tmp_path, monkeypatch):
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    repo.mkdir(); elsewhere.mkdir()
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "make_idle_source", FakeIdleSource)
    monkeypatch.chdir(elsewhere)
    args = core_args(tmp_path, hash_embed=True)
    core = runner.build_core(args, lambda m: None)
    assert core.memory.recent_episodes() == [] and not core.kv.get("legacy_imported")
    core.kv.conn.close()
    (repo / "riya_memory.json").write_text('{"summary": "We talked about cats."}', encoding="utf-8")
    again = runner.build_core(args, lambda m: None)
    assert [e.summary for e in again.memory.recent_episodes()] == ["We talked about cats."]
    assert again.kv.get("legacy_imported") is True


def test_quit_core_from_the_body_reaches_the_core():
    with client() as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "eyes")
        ws.send_json({"type": "quit_core"})
        ws.send_json({"type": "get_memories"})                # handled in order: quit_core went first
        receive_until(ws, "memories")
        assert c.app.state.core.quit_requested.is_set()


def test_run_ticks_stops_when_the_body_asks_the_core_to_quit():
    from riya.core.protocol import QuitCore
    core, *_ = make_core()
    ticks, waits = [], []
    core.tick = lambda: ticks.append(1)

    def wait(seconds):
        waits.append(seconds)
        if len(waits) == 3:
            core.handle(QuitCore())
        return core.quit_requested.is_set()

    runner.run_ticks(core, runner.Speaker(), alive=lambda: True, wait=wait)
    assert len(ticks) == 3 and waits == [1.0, 1.0, 1.0]


def test_run_ticks_stops_when_the_server_stops():
    core, *_ = make_core()
    alive = iter([True, True, False])
    runner.run_ticks(core, runner.Speaker(), alive=lambda: next(alive), wait=lambda s: False)
    assert not core.quit_requested.is_set()


def test_main_exits_cleanly_on_quit_core_and_gates_eyes_on_a_connected_body(monkeypatch):
    """The whole runner with a stand-in server: quit_core ends it normally (exit code 0), and the core's
    body_present is the hub's connection count."""
    import uvicorn

    from riya.core.protocol import QuitCore
    made = {}

    def build_core(args, emit, body_present):
        core, *_ = make_core()
        made["core"], made["body_present"] = core, body_present
        core.handle(QuitCore())                                # the body chose Quit before the first tick
        return core

    class Server:
        def __init__(self, config):
            self.config, self.should_exit = config, False
            made["server"] = self

        def run(self):
            with TestClient(self.config.app):
                while not self.should_exit:
                    threading.Event().wait(0.01)

    class HookReceiver:                                        # never the real one: it would bind 8766
        def __init__(self, enqueue):
            made["enqueue"], made["hooks"], self.stopped = enqueue, self, False

        def start(self):
            return self

        def stop(self):
            self.stopped = True

    def start_logwatch(core):                                  # never the real watcher: it would read G:
        made["logwatch"] = (threading.Event(), threading.Thread(target=lambda: None))
        assert core is made["core"]
        made["logwatch"][1].start()
        return made["logwatch"]

    monkeypatch.setattr(runner, "build_core", build_core)
    monkeypatch.setattr(runner, "HookReceiver", HookReceiver)
    monkeypatch.setattr(runner, "start_logwatch", start_logwatch)
    monkeypatch.setattr(uvicorn, "Server", Server)
    assert runner.main(["--no-voice", "--hash-embed"]) is None
    assert made["server"].should_exit and made["body_present"]() is False
    inbox = made["core"].agent_inbox                           # the core's inbox; the receiver feeds it
    made["enqueue"]({"hook_event_name": "Stop"})
    assert inbox.get_nowait() == {"hook_event_name": "Stop"} and made["hooks"].stopped
    assert made["core"].receiver is made["hooks"]               # the settings read its status through the core
    assert made["logwatch"][0].is_set()                        # the log watcher is stopped on shutdown


def test_main_runs_on_when_the_hook_receiver_cannot_start(monkeypatch):
    import uvicorn

    from riya.core.protocol import QuitCore
    made = {}

    def build_core(args, emit, body_present):
        core, *_ = make_core()
        made["core"] = core
        core.handle(QuitCore())
        return core

    class Server:
        def __init__(self, config):
            self.config, self.should_exit = config, False

        def run(self):
            with TestClient(self.config.app):
                while not self.should_exit:
                    threading.Event().wait(0.01)

    monkeypatch.setattr(runner, "build_core", build_core)
    from riya.agents.hooks import RejectCounter

    class Busy:                                                # 8766 taken: serving later, maybe never
        ok = False

        def __init__(self, enqueue):
            made["receiver"], self.rejected = self, RejectCounter()

        def start(self):
            return self

        def stop(self):
            made["stopped"] = True

    monkeypatch.setattr(runner, "HookReceiver", Busy)
    def start_logwatch(core):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        return threading.Event(), thread

    monkeypatch.setattr(runner, "start_logwatch", start_logwatch)
    monkeypatch.setattr(uvicorn, "Server", Server)
    assert runner.main(["--no-voice", "--hash-embed"]) is None
    assert made["core"].agent_inbox is not None and made["stopped"]
    assert made["core"]._agents_msg().receiver_ok is False


def test_a_typed_turn_while_she_streams_is_measured_but_still_queued():
    typed, cores, calls = threading.Event(), [], []

    def factory(emit):
        core, _, _, _ = make_core(local_replies=["One two. Three four.", "Second."])
        core.emit, real = emit, core.note_typed_turn

        def note():  # every typed turn is noted; the second one arrives while the first reply streams
            real()
            calls.append(1)
            if len(calls) == 2:
                typed.set()
        core.note_typed_turn = note
        cores.append(core)
        return core

    def speak(sentences):  # her first sentence is out; the rest streams only once the next turn was typed
        for i, _ in enumerate(sentences):
            if i == 0:
                typed.wait(5)

    with TestClient(create_app(factory, speak=speak)) as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "actions")  # the end of hello
        ws.send_json({"type": "user_text", "text": "tell me something"})
        receive_until(ws, "say")
        ws.send_json({"type": "user_text", "text": "wait"})
        receive_until(ws, "say_end")
        receive_until(ws, "say_end")  # still queued: answered after the first, in order
    core = cores[0]
    assert len(calls) == 2 and [(e.kind, e.outcome) for e in core.learn.events()] == [("barge", "barged")]
    assert [t["content"] for t in core.turns if t["role"] == "user"] == ["tell me something", "wait"]


# --- v0.7: the online key at start, the voice wiring, a bad setup_request over the socket ---------------------------

CANARY = "CANARY-k3y-0123456789abcdef"


def test_a_bad_setup_request_over_the_socket_never_echoes_the_key():
    raw = '{"type":"setup_request","action":"online_key_set","args":{"key":["' + CANARY + '"]}}'
    with client() as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "actions")  # the end of hello
        ws.send_text(raw)
        m = ws.receive_json()
        assert m == {"type": "error", "message": "invalid setup_request"} and CANARY not in str(m)


def test_connect_voice_wires_the_sampler_and_the_mic_probe(monkeypatch):
    import riya.voice
    from riya.core.protocol import VoicesMsg
    from tests.fakes import wait_for
    called = []

    def probe():
        called.append("probe_mic")
        raise AssertionError("connect_voice must not open the mic")

    class Loop:
        def __init__(self):
            self.gate = ListenGate(FakeClock())

        def sample(self, voice, text):
            called.append("sample")
            raise AssertionError("connect_voice must not speak")

    monkeypatch.setattr(riya.voice, "probe_mic", probe)
    loop = Loop()
    core, emitted, *_ = make_core()
    assert runner.connect_voice(core, loop) is loop
    assert core.voice_sampler == loop.sample and core.mic_prober is probe
    assert core.on_spoke_first == loop.gate.open
    assert wait_for(emitted, VoicesMsg).items == ["alba"]   # the sandboxed voices, read on the riya-voices thread
    assert called == []


def test_build_core_reads_secrets_after_dotenv(tmp_path, monkeypatch):
    from riya.core.secrets import KEY_VAR
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "make_idle_source", FakeIdleSource)
    (tmp_path / ".env").write_text(f"{KEY_VAR}=DOTENV-1234567890\n", encoding="utf-8")
    secrets = tmp_path / "app" / "secrets.env"
    secrets.parent.mkdir()
    secrets.write_text(f"{KEY_VAR}=APP-1234567890\n", encoding="utf-8")
    monkeypatch.setenv(KEY_VAR, "x")  # recorded, so teardown removes what load_dotenv exports
    monkeypatch.delenv(KEY_VAR)
    core = runner.build_core(core_args(tmp_path, hash_embed=True, secrets=str(secrets)), lambda m: None)
    assert core.online.api_key == "APP-1234567890" and core.vision.api_key == "APP-1234567890"
    assert core.online.first_token_timeout_s == 8 and core.vision.first_token_timeout_s == 20
    key = core.setup.deps.key
    assert key.info().source == "app" and (key.base, key.base_source) == ("DOTENV-1234567890", ".env")
    assert key.path == secrets and core.local_url == "http://127.0.0.1:1"


def test_no_parse_error_over_the_socket_echoes_the_key(caplog):
    caplog.set_level("DEBUG")
    sent = ['{"type":"setup_requesx","args":{"key":"' + CANARY + '"}}',
            '{"type":"set_settings","patch":"' + CANARY + '"}',
            '{"type":"user_text","text":["' + CANARY + '"]}']
    with client() as c, c.websocket_connect("/ws") as ws:
        receive_until(ws, "actions")  # the end of hello
        frames = []
        for raw in sent:
            ws.send_text(raw)
            frames.append(ws.receive_json())
    assert [f["message"] for f in frames] == ["invalid message", "invalid set_settings", "invalid user_text"]
    for text in [str(f) for f in frames] + [caplog.text]:
        assert not any(part in text for part in (CANARY, CANARY[:10], CANARY[-10:]))


# --- fix round 1: her voices without the voice loop -----------------------------------------------------------------

def test_voice_choices_fall_back_to_the_feminine_list(monkeypatch):
    from riya import voice_candidates as vc
    monkeypatch.setattr(vc, "builtin_voices", lambda: [])  # no pocket_tts
    assert vc.voice_choices() == list(vc.FEMININE)

    def broken():
        raise RuntimeError("a broken install")
    monkeypatch.setattr(vc, "builtin_voices", broken)
    assert vc.voice_choices() == list(vc.FEMININE)
    monkeypatch.setattr(vc, "builtin_voices", lambda: ["marius", "eve", "alba"])
    assert vc.voice_choices() == ["alba", "eve"]


def test_build_core_lists_her_voices_without_the_voice_stack(tmp_path, monkeypatch):
    from riya import voice_candidates as vc
    asked = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "make_idle_source", FakeIdleSource)
    monkeypatch.setattr(vc, "builtin_voices", lambda: asked.append(1) or [])
    core = runner.build_core(core_args(tmp_path, hash_embed=True), lambda m: None)
    core.setup.load_voices()
    assert core.setup.voices() == list(vc.FEMININE) and asked == [1]
    core.kv.conn.close()
    quiet = runner.build_core(core_args(tmp_path, hash_embed=True, no_voice=True), lambda m: None)
    quiet.setup.load_voices()                             # --no-voice never touches the TTS stack
    assert quiet.setup.voices() == list(vc.FEMININE) and asked == [1]


def test_main_lists_her_voices_when_text_only(monkeypatch):
    import uvicorn

    from riya.core.protocol import QuitCore, VoicesMsg
    from tests.fakes import wait_for
    made = {}

    def build_core(args, emit, body_present):
        core, emitted, *_ = make_core()
        made["emitted"] = emitted
        core.handle(QuitCore())
        return core

    class Server:
        def __init__(self, config):
            self.config, self.should_exit = config, False

        def run(self):
            with TestClient(self.config.app):
                while not self.should_exit:
                    threading.Event().wait(0.01)

    class Receiver:
        ok = True

        def __init__(self, enqueue):
            pass

        def start(self):
            return self

        def stop(self):
            pass

    def start_logwatch(core):
        thread = threading.Thread(target=lambda: None)
        thread.start()
        return threading.Event(), thread

    monkeypatch.setattr(runner, "build_core", build_core)
    monkeypatch.setattr(runner, "HookReceiver", Receiver)
    monkeypatch.setattr(runner, "start_logwatch", start_logwatch)
    monkeypatch.setattr(uvicorn, "Server", Server)
    assert runner.main(["--no-voice", "--hash-embed"]) is None
    assert wait_for(made["emitted"], VoicesMsg).items == ["alba"]  # the sandboxed list, read on riya-voices
