import http.server
import os
import threading

import pytest

from riya.core.models import USER_AGENT, ModelUnavailable, OpenAICompatModel, load_online_from_env
from riya.core.router import Router, needs_thinking
from tests.fakes import FakeModel

MSGS = [{"role": "user", "content": "x"}]


def test_needs_thinking_rules():
    for t in ["can you debug this traceback?", "explain how attention works", "write a python function to dedupe",
              "think hard: sqlite or postgres", "```py\nprint(1)\n```"]:
        assert needs_thinking(t), t
    for t in ["hiii", "I missed you too", "my day was rough", "good night riya", "haha you're cute"]:
        assert not needs_thinking(t), t


def test_small_talk_goes_local():
    local, online = FakeModel("local", ["hey you~"]), FakeModel("online", ["X"])
    r = Router(local, online)
    assert "".join(r.stream(MSGS, "hi riya")) == "hey you~" and r.last_source == "local" and online.calls == []


def test_local_streams_use_the_local_temperature_and_online_keeps_its_default():
    from riya.core.models import LOCAL_TEMPERATURE
    local, online = FakeModel("local", ["L."]), FakeModel("online", fail=ModelUnavailable("429"))
    r = Router(local, online)
    list(r.stream(MSGS, "hi riya")), list(r.stream(MSGS, "explain recursion"))  # the second falls back to local
    assert local.temperatures == [LOCAL_TEMPERATURE] * 2 and online.temperatures == [0.8]


def test_code_question_goes_online():
    r = Router(FakeModel("local", ["L"]), FakeModel("online", ["Use a set."]))
    assert "".join(r.stream(MSGS, "write a function to dedupe a list")) == "Use a set." and r.last_source == "online"


def test_online_failure_falls_back_to_local():
    r = Router(FakeModel("local", ["Local answer."]), FakeModel("online", fail=ModelUnavailable("429")))
    assert "".join(r.stream(MSGS, "explain recursion")) == "Local answer." and r.last_source == "local"


def test_online_missing_or_disabled_uses_local():
    local = FakeModel("local", ["L."])
    assert Router(local, None).route("explain recursion") == "local"
    assert Router(local, FakeModel("online"), online_enabled=False).route("explain recursion") == "local"


def test_online_refusal_falls_back_to_local():
    r = Router(FakeModel("local", ["Sure, here."]), FakeModel("online", ["I'm sorry, but I can't help with that."]))
    assert "".join(r.stream(MSGS, "explain this")) == "Sure, here." and r.last_source == "local"


def test_online_drop_mid_stream_keeps_partial_without_raising():
    r = Router(FakeModel("local", ["L."]), FakeModel("online", ["First sentence here. Second one"], fail_after=3))
    assert "".join(r.stream(MSGS, "explain this")) == "First sentence here. " and r.last_source == "online"


def test_online_drop_before_first_sentence_falls_back_to_local():
    r = Router(FakeModel("local", ["Local."]), FakeModel("online", ["No end in sight yet"], fail_after=2))
    assert "".join(r.stream(MSGS, "explain this")) == "Local." and r.last_source == "local"


def test_online_buffer_flushes_at_160_chars_without_sentence_end():
    long = " ".join(["word"] * 60)
    r = Router(FakeModel("local", ["L."]), FakeModel("online", [long]))
    assert "".join(r.stream(MSGS, "explain this")) == long and r.last_source == "online"


def _serve(handler_cls):
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_openai_compat_streams_sse():
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ["Hel", "lo"]:
                self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}\n\n'.encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass

    srv, url = _serve(H)
    assert "".join(OpenAICompatModel("t", url).stream(MSGS)) == "Hello"
    srv.shutdown()


def test_openai_compat_429_is_unavailable():
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(429)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv, url = _serve(H)
    with pytest.raises(ModelUnavailable):
        list(OpenAICompatModel("t", url).stream(MSGS))
    srv.shutdown()


def test_openai_compat_sends_body_and_auth_and_times_out_before_first_token():
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"], seen["path"] = self.headers.get("Authorization"), self.path
            seen["ua"] = self.headers.get("User-Agent")
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            threading.Event().wait(1.0)

        def log_message(self, *a):
            pass

    srv, url = _serve(H)
    with pytest.raises(ModelUnavailable):
        list(OpenAICompatModel("t", url, api_key="k", model="m", first_token_timeout_s=0.3).stream(MSGS))
    srv.shutdown()
    assert seen["auth"] == "Bearer k" and seen["path"] == "/chat/completions"
    assert seen["ua"] == USER_AGENT and not seen["ua"].startswith("Python-urllib")
    assert b'"model": "m"' in seen["body"] and b'"stream": true' in seen["body"]


def test_connection_refused_is_unavailable():
    with pytest.raises(ModelUnavailable):
        list(OpenAICompatModel("t", "http://127.0.0.1:1").stream(MSGS))


def test_load_online_from_env():
    from riya.core import models
    m = load_online_from_env({})  # OpenCode's free tier needs no key
    assert m.base_url == models.ONLINE_BASE_URL and m.model == models.ONLINE_MODEL and m.api_key is None
    assert load_online_from_env({"RIYA_ONLINE_API_KEY": ""}) is None  # an explicit blank turns it off
    assert load_online_from_env({"RIYA_ONLINE_OFFLINE": "1"}) is None
    assert load_online_from_env({"RIYA_ONLINE_OFFLINE": "1", "RIYA_ONLINE_API_KEY": "k"}) is None  # beats a saved key
    assert load_online_from_env({"RIYA_ONLINE_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai"}) is None
    m = load_online_from_env({"RIYA_ONLINE_API_KEY": "k"})
    assert m.api_key == "k" and m.base_url == models.ONLINE_BASE_URL
    m = load_online_from_env({"RIYA_ONLINE_API_KEY": "k", "RIYA_ONLINE_MODEL": "glm-5.3-flash-free",
                              "RIYA_ONLINE_BASE_URL": "https://llm.example.test/v1"})
    assert (m.base_url, m.model) == ("https://llm.example.test/v1", "glm-5.3-flash-free")
    m = load_online_from_env({"RIYA_ONLINE_KEYLESS": "1", "RIYA_ONLINE_BASE_URL": "http://127.0.0.1:8090/v1"})
    assert m.api_key is None
    vision = models.online_pair({"RIYA_ONLINE_VISION_MODEL": "eyes-model"})[1]
    assert vision.name == "vision" and vision.model == "eyes-model"


def test_load_dotenv_parses_and_does_not_override(tmp_path, monkeypatch):
    from riya.core.models import load_dotenv
    p = tmp_path / ".env"
    p.write_text('# comment\nRIYA_T_A=1\n\nRIYA_T_B="two words"\nRIYA_T_C=from_file\nbad line\n', encoding="utf-8")
    monkeypatch.setenv("RIYA_T_C", "already")
    for k in ("RIYA_T_A", "RIYA_T_B"):
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    got = load_dotenv(str(p))
    assert got == {"RIYA_T_A": "1", "RIYA_T_B": "two words", "RIYA_T_C": "from_file"}
    assert os.environ["RIYA_T_A"] == "1" and os.environ["RIYA_T_C"] == "already"
    assert load_dotenv(str(tmp_path / "missing.env")) == {}


# --- final review ---------------------------------------------------------------------------------------------------

def _sse(lines):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for line in lines:
                self.wfile.write(line.encode() + b"\n\n")

        def log_message(self, *a):
            pass
    return _serve(H)


@pytest.mark.parametrize("bad", ['data: {"choices": [{"delta": {"content": "Hi', 'data: {"choices":[{"delta":null}]}',
                                 'data: {"choices":"nope"}', 'data: [1, 2]'])
def test_malformed_sse_is_unavailable(bad):
    srv, url = _sse(['data: {"choices":[{"delta":{"content":"Hel"}}]}', bad, "data: [DONE]"])
    got = []
    with pytest.raises(ModelUnavailable):
        for piece in OpenAICompatModel("t", url).stream(MSGS):
            got.append(piece)
    srv.shutdown()
    assert got == ["Hel"]


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16", "utf-16-be"])
def test_load_dotenv_handles_a_bom(tmp_path, monkeypatch, encoding):
    from riya.core.models import load_dotenv
    p = tmp_path / ".env"
    bom = chr(0xFEFF) if encoding == "utf-16-be" else ""  # the other two codecs write their own BOM
    p.write_bytes((bom + "RIYA_T_KEY=secret\nRIYA_T_OTHER=2\n").encode(encoding))
    monkeypatch.setenv("RIYA_T_KEY", "x")
    monkeypatch.delenv("RIYA_T_KEY")
    monkeypatch.setenv("RIYA_T_OTHER", "x")
    monkeypatch.delenv("RIYA_T_OTHER")
    assert load_dotenv(str(p)) == {"RIYA_T_KEY": "secret", "RIYA_T_OTHER": "2"}
    assert os.environ["RIYA_T_KEY"] == "secret"


def test_load_dotenv_with_garbage_bytes_warns_and_returns_nothing(tmp_path, caplog):
    from riya.core.models import load_dotenv
    p = tmp_path / ".env"
    p.write_bytes(b"RIYA_T_BAD=\x80\x81\xc3\x28\n")
    with caplog.at_level("WARNING"):
        assert load_dotenv(str(p)) == {}
    assert "RIYA_T_BAD" not in os.environ and any(".env" in r.getMessage() for r in caplog.records)


def test_local_model_sends_the_user_turn_stop_sequences():
    import json
    from riya.core.models import LOCAL_STOP, local_model
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"], seen["path"] = json.loads(self.rfile.read(int(self.headers["Content-Length"]))), self.path
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"Hi."}}]}\n\ndata: [DONE]\n\n')

        def log_message(self, *a):
            pass

    srv, url = _serve(H)
    m = local_model(url + "/", first_token_timeout_s=5.0)
    assert m.name == "local" and "".join(m.stream(MSGS)) == "Hi."
    srv.shutdown()
    assert LOCAL_STOP == ("\nUser:",) and seen["body"]["stop"] == list(LOCAL_STOP)
    assert seen["path"] == "/v1/chat/completions" and OpenAICompatModel("online", url).stop is None


# --- final fix wave: M5, finish_reason and the online budget ------------------------------------------------------
def test_openai_compat_records_the_last_finish_reason():
    chunk = 'data: {"choices":[{"delta":{"content":"%s"},"finish_reason":%s}]}'
    srv, url = _sse([chunk % ("Hel", "null"), chunk % ("lo", '"length"'), "data: [DONE]"])
    m = OpenAICompatModel("t", url)
    assert m.finish_reason is None
    assert "".join(m.stream(MSGS)) == "Hello" and m.finish_reason == "length"
    srv.shutdown()
    srv, url = _sse([chunk % ("Hi", "null"), 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
                     "data: [DONE]"])
    m.base_url = f"{url}"
    assert "".join(m.stream(MSGS)) == "Hi" and m.finish_reason == "stop"
    srv.shutdown()


def test_router_reports_the_finish_reason_of_the_model_that_answered():
    local, online = FakeModel("local", ["L."]), FakeModel("online", ["O."], finish_reason="length")
    r = Router(local, online)
    list(r.stream(MSGS, "explain recursion"))
    assert r.last_finish_reason == "length"
    list(r.stream(MSGS, "hi riya"))
    assert r.last_finish_reason == "stop"


def test_online_turns_can_get_their_own_token_budget_and_the_fallback_keeps_the_local_one():
    seen = []

    class Spy(FakeModel):
        def stream(self, messages, max_tokens=300, temperature=0.8):
            seen.append((self.name, max_tokens))
            return super().stream(messages, max_tokens, temperature)
    r = Router(Spy("local", ["L."]), Spy("online", ["O."]))
    list(r.stream(MSGS, "explain recursion", 380, online_tokens=600))
    r.online = Spy("online", fail=ModelUnavailable("429"))
    list(r.stream(MSGS, "explain recursion", 380, online_tokens=600))
    assert seen == [("online", 600), ("online", 600), ("local", 380)]


def test_openai_compat_sends_list_content_parts_unchanged():
    seen = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')

        def log_message(self, *a):
            pass

    part = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}
    msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}, part]}]
    srv, url = _serve(H)
    assert "".join(OpenAICompatModel("t", url).stream(msgs)) == "ok"
    srv.shutdown()
    assert __import__("json").loads(seen["body"])["messages"] == msgs


def test_prefetched_keeps_order_errors_and_stops_when_left():
    import threading
    import time

    from riya.core.router import Prefetched

    def slow():
        time.sleep(0.2)
        yield "a"
        yield "b"
    p = Prefetched(slow())
    assert not p.ready(0.01) and p.ready(2.0)
    assert list(p) == ["a", "b"]

    def broken():
        yield "x"
        raise ModelUnavailable("drop")
    with pytest.raises(ModelUnavailable):
        assert list(Prefetched(broken())) == ["x"]
    assert Prefetched(iter(())).ready(1.0)  # an empty stream is "ready" (its end is in)

    pulled, gate = [], threading.Event()

    def endless():
        for i in range(1000):
            pulled.append(i)
            gate.wait(0.01)
            yield str(i)
    it = iter(Prefetched(endless()))
    assert next(it) == "0"
    it.close()
    time.sleep(0.1)
    n = len(pulled)
    time.sleep(0.1)
    assert len(pulled) == n < 1000  # left early: the worker stopped
