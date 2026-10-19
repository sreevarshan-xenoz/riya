"""Online function calling: the request body, the schemas, the defensive parse and `propose`'s single validated call."""
import json
import urllib.error

import pytest

from riya.core.hands.online import online_messages, parse_arguments, propose, tool_schemas
from riya.core.hands.tools import REGISTRY, TIERS, Call
from riya.core.models import ModelUnavailable, OpenAICompatModel, ToolReply
from riya.core.settings import Settings
from tests.fakes import FakeModel

NOW = 1_800_000_000.0
CMD = {"name": "backup", "argv": ["CANARY_ARGV"], "cwd": "CANARY_CWD", "timeout_s": 60}
APP = {"name": "Spotify", "target": "CANARY_TARGET"}


def reply_json(message: dict) -> bytes:
    return json.dumps({"choices": [{"message": message, "finish_reason": "stop"}]}).encode()


class FakeResp:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]


@pytest.fixture
def sent(monkeypatch):
    """Monkeypatch urlopen: records (request, timeout) and answers with `sent.body`."""
    box = type("Box", (), {})()
    box.reqs, box.body, box.raises = [], reply_json({"role": "assistant", "content": "hi"}), None

    def fake(req, timeout=None):
        box.reqs.append((req, timeout))
        if box.raises:
            raise box.raises
        return FakeResp(box.body)

    monkeypatch.setattr("riya.core.models.urllib.request.urlopen", fake)
    return box


def tc(name, args):
    return {"id": "c1", "type": "function", "function": {"name": name, "arguments": args}}


# ---- call_tools ----
def test_call_tools_request_and_parse(sent):
    m = OpenAICompatModel("online", "http://x/v1/", "KEY", "gem")
    sent.body = reply_json({"role": "assistant", "content": None, "tool_calls": [tc("open_url", '{"url": "a.com"}')]})
    tools = [{"type": "function", "function": {"name": "open_url"}}]
    r = m.call_tools([{"role": "user", "content": "open a.com"}], tools)
    assert r == ToolReply([("open_url", '{"url": "a.com"}')], None)
    req, timeout = sent.reqs[0]
    body = json.loads(req.data)
    assert timeout == 8.0 and body["stream"] is False and body["tools"] == tools and body["tool_choice"] == "auto"
    assert body["max_tokens"] == 200 and body["model"] == "gem"
    assert req.full_url == "http://x/v1/chat/completions" and req.get_header("Authorization") == "Bearer KEY"


def test_call_tools_text_reply_and_odd_shapes(sent):
    m = OpenAICompatModel("online", "http://x/v1")
    assert m.call_tools([], []) == ToolReply([], "hi")
    sent.body = reply_json({"role": "assistant", "content": None, "tool_calls": [tc("a", {"k": 1}), {"bad": 1}]})
    r = m.call_tools([], [])
    assert r.calls == [("a", '{"k": 1}'), ("", "")]  # a malformed entry still counts, so two calls can't pass as one


@pytest.mark.parametrize("body", [b"not json", b"[]", b'{"choices": []}', b'{"choices": [1]}',
                                  b'{"error": {"message": "quota"}}', b'{"choices": [{"message": 3}]}'])
def test_call_tools_bad_body_is_unavailable(sent, body):
    sent.body = body
    with pytest.raises(ModelUnavailable):
        OpenAICompatModel("online", "http://x/v1").call_tools([], [])


@pytest.mark.parametrize("exc", [urllib.error.HTTPError("u", 429, "x", {}, None), urllib.error.URLError("down"),
                                 TimeoutError("slow"), ConnectionResetError()])
def test_call_tools_errors_map_to_unavailable(sent, exc):
    sent.raises = exc
    with pytest.raises(ModelUnavailable):
        OpenAICompatModel("online", "http://x/v1").call_tools([], [])


# ---- online_messages ----
def test_messages_carry_only_turn_time_and_names():
    s = Settings(apps=[APP], commands=[CMD], weather_place="CANARY_PLACE")
    msgs = online_messages("open the thing", NOW, s, None)
    blob = json.dumps(msgs)
    assert "open the thing" in blob and "Spotify" in blob and "backup" in blob
    assert msgs[-1] == {"role": "user", "content": "open the thing"} and msgs[0]["role"] == "system"
    assert "2027" in blob or "2026" in blob or "2025" in blob  # the local time is there
    for canary in ("CANARY_TARGET", "CANARY_ARGV", "CANARY_CWD", "CANARY_PLACE"):
        assert canary not in blob


def test_messages_tolerate_junk_names():
    s = Settings(apps=[{"name": "A\nB"}, "x", {"name": 5}], commands=[{"name": "ok"}])
    blob = online_messages("hi", NOW, s, None)[0]["content"]
    assert "A B" in blob and "ok" in blob and "\nB" not in blob


# ---- tool_schemas ----
def names(schemas):
    return {t["function"]["name"] for t in schemas}


def test_schemas_skip_disabled_and_gate_t2():
    s = Settings(commands=[CMD])
    t01 = {n for n in TIERS if TIERS[n] < 2}
    assert names(tool_schemas(s, "open youtube")) == t01
    assert names(tool_schemas(s, "run backup and save a file")) == t01  # T2 is off by default, so never offered
    on = Settings(commands=[CMD], tools_enabled={n: True for n in TIERS})
    assert names(tool_schemas(on, "open youtube")) == t01
    assert names(tool_schemas(on, "run backup please")) == t01 | {"run_command", "write_file"}
    assert names(tool_schemas(on, "run BACKUP")) >= {"run_command"}  # the name matches ignoring case
    assert names(tool_schemas(on, "save that to a file")) == t01 | {"run_command", "write_file"}
    assert names(tool_schemas(Settings(tools_enabled={n: True for n in TIERS}), "open it")) == t01
    off = Settings(tools_enabled={**Settings().tools_enabled, "calc": False, "open_url": False})
    assert not {"calc", "open_url"} & names(tool_schemas(off, "open it"))


def test_schema_types_match_paramspecs():
    s = Settings(tools_enabled={n: True for n in TIERS})
    schemas = {t["function"]["name"]: t["function"] for t in tool_schemas(s, "write a file, run backup")}
    assert set(schemas) == set(REGISTRY)
    js = {"str": "string", "int": "integer", "float": "number", "duration": "string"}
    for name, fn in schemas.items():
        p = fn["parameters"]
        assert p["type"] == "object" and fn["description"] and fn["name"] == name
        assert set(p["properties"]) == set(REGISTRY[name].params)
        for k, spec in REGISTRY[name].params.items():
            prop = p["properties"][k]
            assert prop["type"] == js[spec.type]
            assert ("enum" in prop) == (spec.enum is not None)
            if spec.enum is not None:
                assert prop["enum"] == list(spec.enum)
        assert set(p["required"]) == {k for k, sp in REGISTRY[name].params.items() if sp.required}
    c = schemas["reminder_cancel"]["parameters"]
    assert c["properties"]["id"]["type"] == "integer" and c["properties"]["text"]["type"] == "string"
    assert "id" not in c["required"] and "text" not in c["required"]
    assert schemas["media"]["parameters"]["properties"]["action"]["enum"] == ["play_pause", "next", "previous"]
    assert schemas["timer_set"]["parameters"]["properties"]["duration"]["type"] == "string"
    json.dumps(list(schemas.values()))  # serializable


# ---- parse_arguments ----
BAD_ARGS = {"bad": "{bad", "list": "[1]", "str": '"x"', "int": "3", "null": "null", "empty": "", "none": None,
            "num": 5, "dict": {"a": 1}, "huge": '{"a": "' + "x" * 5000 + '"}', "deep_list": "[" * 100_000,
            "deep_obj": '{"a":' * 3000, "wide_utf8": '{"a": "' + "é" * 3000 + '"}'}


@pytest.mark.parametrize("raw", BAD_ARGS.values(), ids=BAD_ARGS)
def test_parse_arguments_rejects(raw):
    assert parse_arguments(raw) is None


def test_parse_arguments_accepts_a_dict():
    assert parse_arguments('{"url": "a.com", "n": 1}') == {"url": "a.com", "n": 1} and parse_arguments("{}") == {}


# ---- propose ----
def run(text, replies, settings=None, **kw):
    m = FakeModel("online", tool_replies=replies)
    settings = settings or Settings(apps=[APP], commands=[CMD])
    return propose(m, text, NOW, settings, **kw), m


def one(name, args):
    return [ToolReply([(name, args if isinstance(args, str) else json.dumps(args))], None)]


def test_valid_open_url_is_returned_as_sent():
    call, m = run("open youtube.com for me", one("open_url", {"url": "youtube.com"}))
    assert call == Call("open_url", {"url": "youtube.com"})
    (msgs, tools), = m.tool_calls
    assert msgs[-1]["content"] == "open youtube.com for me" and "open_url" in names(tools)


def test_propose_none_when_nothing_to_ask():
    ok = one("open_url", {"url": "youtube.com"})
    m = FakeModel("online", tool_replies=ok)
    assert propose(None, "open youtube.com", NOW, Settings()) is None
    assert propose(m, "what a nice day", NOW, Settings()) is None and m.tool_calls == []  # not a command form
    assert propose(m, "open youtube.com", NOW, Settings(hands_online=False)) is None and m.tool_calls == []
    assert propose(m, "open youtube.com", NOW, Settings(online_enabled=False)) is None and m.tool_calls == []


@pytest.mark.parametrize("replies", [
    one("open_url", "{bad json"),
    one("open_url", "[1, 2]"),
    one("open_url", {"url": 5}),
    one("open_url", {"url": "youtube.com", "extra": 1}),
    one("rm_rf", {}),
    one("open_url", {"url": "javascript:alert(1)"}),
    [ToolReply([("open_url", '{"url": "a.com"}'), ("open_url", '{"url": "b.com"}')], None)],
    [ToolReply([("time_now", "{}"), ("time_now", "{}")], None)],
    [ToolReply([], "Sure, opening it!")],
    [ModelUnavailable("down")],
    [RuntimeError("boom")],
    one("run_command", {"name": "backup"}),  # T2 isn't offered for this turn, so it's refused even though it's valid
    one("write_file", {"filename": "a.txt", "text": "x"}),
])
def test_propose_none(replies):
    call, _ = run("open youtube.com", replies, Settings(apps=[APP], commands=[CMD]))
    assert call is None


def test_model_that_fails_with_fail_raises_nothing():
    m = FakeModel("online", fail=ModelUnavailable("down"))
    assert propose(m, "open youtube.com", NOW, Settings()) is None


def test_t2_call_needs_its_tool_enabled_and_offered():
    s = Settings(commands=[CMD], tools_enabled={**Settings().tools_enabled, "run_command": True})
    call, _ = run("run backup", one("run_command", {"name": "backup"}), s)
    assert call == Call("run_command", {"name": "backup"})
    call, _ = run("run backup", one("run_command", {"name": "nope"}), s)
    assert call is None  # not one of their commands


def test_write_file_text_must_be_the_users_words():
    s = Settings(tools_enabled={**Settings().tools_enabled, "write_file": True})
    turn = "save a file called todo.txt saying buy   milk\nand eggs"
    good = one("write_file", {"filename": "todo.txt", "text": "buy milk and eggs"})
    assert run(turn, good, s)[0] == Call("write_file", {"filename": "todo.txt", "text": "buy milk and eggs"})
    bad = one("write_file", {"filename": "todo.txt", "text": "buy milk and eggs and rm -rf"})
    assert run(turn, bad, s)[0] is None
    assert run(turn, one("write_file", {"filename": "todo.txt", "text": "invented"}), s)[0] is None
    assert run(turn, one("write_file", {"filename": "todo.txt", "text": "   "}), s)[0] is None


def test_nothing_is_executed_and_request_has_no_canaries():
    call, m = run("open the app", one("open_app", {"name": "Spotify"}),
                  Settings(apps=[APP], commands=[CMD], weather_place="CANARY_PLACE"))
    assert call == Call("open_app", {"name": "Spotify"})  # unnormalized: no target is attached here
    blob = json.dumps(m.tool_calls)
    assert "Spotify" in blob and not any(c in blob for c in ("CANARY_TARGET", "CANARY_ARGV", "CANARY_CWD", "CANARY_PLACE"))


def test_propose_accepts_a_tz():
    from zoneinfo import ZoneInfo
    call, _ = run("open youtube.com", one("open_url", {"url": "youtube.com"}), tz=ZoneInfo("UTC"))
    assert call is not None
