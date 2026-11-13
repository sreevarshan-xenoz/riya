"""The model pool: failover before the first piece, benching on limits and errors, sticky order, the slow tier."""
import http.server
import threading

import pytest

from riya.core import models, pool
from riya.core.models import ModelUnavailable, OpenAICompatModel, ToolReply
from riya.core.pool import Member, ModelPool, kind_of

MSGS = [{"role": "user", "content": "hi"}]


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Fake:
    """Answers with `reply`, or raises `fail` (a ModelUnavailable) before its first piece."""

    def __init__(self, name, reply="ok", fail=None, tools=True):
        self.name, self.model, self.reply, self.fail, self.calls = name, name, reply, fail, 0
        self.finish_reason = None
        if not tools:
            self.call_tools = None

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        self.finish_reason = "stop"
        yield from self.reply.split(" ")

    def call_tools(self, messages, tools, timeout_s=8.0, max_tokens=200):
        self.calls += 1
        if self.fail is not None:
            raise self.fail
        return ToolReply([("t", "{}")])


def limit(retry=None):
    return ModelUnavailable("x: HTTP 429", status=429, retry_after_s=retry)


def make(*fakes, slow=(), clock=None):
    clock = clock or Clock()
    return ModelPool("online", [Member(f, slow=f.name in slow) for f in fakes], clock=clock), clock


def test_a_limited_member_hands_over_before_the_first_piece():
    a, b = Fake("a", fail=limit()), Fake("b", reply="hello there")
    p, _ = make(a, b)
    assert "".join(p.stream(MSGS)) == "hellothere" and p.finish_reason == "stop"
    assert [s.live for s in p.states()] == [False, True]


def test_limits_bench_with_retry_after_else_doubling_then_come_back():
    a, b = Fake("a", fail=limit(retry=30)), Fake("b")
    p, clock = make(a, b)
    list(p.stream(MSGS))
    assert p.states()[0].benched_for_s == 30
    clock.t += 31
    a.fail = limit()
    list(p.stream(MSGS))  # sticky on b, then a is due again but b answers first
    assert a.calls == 1
    p.last = None
    list(p.stream(MSGS))  # a is tried again: limited a second time, no Retry-After
    assert p.states()[0].benched_for_s == pool.LIMIT_BENCH_S * 2
    clock.t += pool.LIMIT_BENCH_S * 2 + 1
    a.fail = None
    p.last = None
    assert "".join(p.stream(MSGS)) == "ok" and p.states()[0].live and p.members[0].limits == 0


def test_the_limit_bench_is_capped():
    a = Fake("a", fail=limit())
    p, clock = make(a, Fake("b"))
    for _ in range(12):
        p.last = None
        list(p.stream(MSGS))
        clock.t += pool.LIMIT_BENCH_MAX_S + 1
    p.last = None
    list(p.stream(MSGS))
    assert p.states()[0].benched_for_s == pool.LIMIT_BENCH_MAX_S


def test_errors_bench_briefly_and_bad_requests_not_at_all():
    a = Fake("a", fail=ModelUnavailable("a: timed out"))
    p, _ = make(a, Fake("b"))
    list(p.stream(MSGS))
    assert p.states()[0].benched_for_s == pool.ERROR_BENCH_S
    c = Fake("c", fail=ModelUnavailable("c: HTTP 400", status=400))
    p, _ = make(c, Fake("d"))
    list(p.stream(MSGS))
    assert p.states()[0].live


def test_the_order_is_sticky_and_slow_members_come_last():
    a, b, s = Fake("a"), Fake("b"), Fake("s")
    p, _ = make(s, a, b, slow=("s",))
    list(p.stream(MSGS))
    assert (s.calls, a.calls) == (0, 1)  # the slow one waits behind every fast one
    p.last = p.members[2]
    list(p.stream(MSGS))
    assert b.calls == 1 and a.calls == 1
    a.fail = b.fail = limit()
    p.last = None
    assert "".join(p.stream(MSGS)) == "ok" and s.calls == 1


def test_everyone_down_is_unavailable_and_names_who_was_tried():
    p, _ = make(Fake("a", fail=limit()), Fake("b", fail=ModelUnavailable("b: reset")))
    with pytest.raises(ModelUnavailable, match="tried a, b"):
        list(p.stream(MSGS))
    with pytest.raises(ModelUnavailable):  # both benched now: nobody is tried
        list(p.stream(MSGS))


def test_a_failure_after_the_first_piece_is_the_reply_s():
    class Drops(Fake):
        def stream(self, *a, **k):
            yield "half"
            raise ModelUnavailable("drop")
    b = Fake("b")
    p, _ = make(Drops("a"), b)
    out = []
    with pytest.raises(ModelUnavailable):
        for piece in p.stream(MSGS):
            out.append(piece)
    assert out == ["half"] and b.calls == 0


def test_tools_skip_members_without_them_and_mark_those_that_refuse():
    a, b, c = Fake("a", tools=False), Fake("b", fail=ModelUnavailable("b: HTTP 400", status=400)), Fake("c")
    for f in (a,):
        del f.call_tools
    p = ModelPool("online", [Member(a, tools=False), Member(b), Member(c)], clock=Clock())
    assert p.call_tools(MSGS, []).calls == [("t", "{}")]
    assert p.members[1].tools is False and p.states()[1].live  # b still chats
    assert c.calls == 1


def test_kind_of():
    assert kind_of(ModelUnavailable("x", status=429)) == "limit"
    assert kind_of(ModelUnavailable("x", status=403, detail="Daily quota exceeded")) == "limit"
    assert kind_of(ModelUnavailable("x", status=403, detail="forbidden")) == "error"
    assert kind_of(ModelUnavailable("x: usage limit reached")) == "limit"
    assert kind_of(ModelUnavailable("x", status=400)) == "bad_request"
    assert kind_of(ModelUnavailable("x", status=503)) == "error"


def test_the_pool_reads_like_its_first_member():
    p, _ = make(Fake("a"), Fake("b"))
    assert p.model == "a" and p.name == "online"
    with pytest.raises(ValueError):
        ModelPool("online", [])


def test_the_member_list_refreshes_once_on_first_use():
    calls = []

    def refresh():
        calls.append(1)
        return [Member(Fake("a")), Member(Fake("new"))]
    p = ModelPool("online", [Member(Fake("a"))], clock=Clock(), refresh=refresh)
    assert calls == []  # never at construction
    list(p.stream(MSGS))
    list(p.stream(MSGS))
    assert calls == [1] and [m.name for m in p.members] == ["a", "new"]

    def broken():
        raise OSError("offline")
    p = ModelPool("online", [Member(Fake("a"))], clock=Clock(), refresh=broken)
    assert "".join(p.stream(MSGS)) == "ok"


def _serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_a_real_429_carries_retry_after_and_its_body():
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(429)
            self.send_header("Retry-After", "120")
            self.end_headers()
            self.wfile.write(b'{"error":"rate limit exceeded for this model"}')

        def log_message(self, *a):
            pass
    srv, url = _serve(H)
    try:
        with pytest.raises(ModelUnavailable) as got:
            list(OpenAICompatModel("t", url).stream(MSGS))
        e = got.value
        assert (e.status, e.retry_after_s) == (429, 120.0) and "rate limit" in e.detail
        with pytest.raises(ModelUnavailable) as got:
            OpenAICompatModel("t", url).call_tools(MSGS, [])
        assert got.value.status == 429
    finally:
        srv.shutdown()


def test_unavailable_still_takes_just_a_message():
    e = models.ModelUnavailable("plain")
    assert (e.status, e.retry_after_s, e.detail, str(e)) == (None, None, None, "plain")


def test_the_free_endpoint_and_a_key_both_lead_a_pool():
    online, vision = models.online_pair({})
    assert isinstance(online, ModelPool) and isinstance(vision, ModelPool)
    assert online.members[0].name == models.ONLINE_MODEL and online.api_key is None
    assert online.first_token_timeout_s == models.ONLINE_TIMEOUT_S and vision.first_token_timeout_s == 20
    assert len(online.members) == len(pool.OPENCODE_FREE)
    assert all(m.vision for m in vision.members) and len(vision.members) < len(online.members)
    assert {m.model.name for m in online.members} == {"online"} and {m.model.name for m in vision.members} == {"vision"}
    keyed, _ = models.online_pair({"RIYA_ONLINE_API_KEY": "ENVKEY-11111111"})  # the user's own key: it alone leads
    assert [m.provider for m in keyed.members] == ["online"] and keyed.api_key == "ENVKEY-11111111"
    other, _ = models.online_pair({"RIYA_ONLINE_BASE_URL": "https://example.com/v1", "RIYA_ONLINE_KEYLESS": "1"})
    assert [m.provider for m in other.members] == ["online"]
    assert models.online_pair({"RIYA_ONLINE_OFFLINE": "1"}) == (None, None)  # nothing online, no keys
    chosen, _ = models.online_pair({"RIYA_ONLINE_MODEL": "beam-free"})
    assert chosen.members[0].name == "beam-free"


def test_free_pool_puts_the_configured_model_first_and_refreshes_from_the_list():
    p = pool.free_pool("online", lambda i: Fake(i), "b-free", list_ids=lambda: {"a-free": False, "c-free": True})
    assert p.members[0].name == "b-free"
    list(p.stream(MSGS))
    assert {"a-free", "c-free"} <= {m.name for m in p.members}
    v = pool.free_pool("vision", lambda i: Fake(i), "glm-5.3-flash-free", vision_only=True,
                       list_ids=lambda: {"a-free": False, "c-free": True})
    list(v.stream(MSGS))
    assert "a-free" not in {m.name for m in v.members} and "c-free" in {m.name for m in v.members}


def test_the_live_free_list_is_read_from_models():
    body = (b'{"data":[{"id":"x-free","modalities":{"input":["text","image/png"]}},'
            b'{"id":"y-free","modalities":{"input":["text"]}},{"id":"paid-model"}, "junk"]}')

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body if self.path.endswith("/models") else b"{}")

        def log_message(self, *a):
            pass
    srv, url = _serve(H)
    try:
        assert pool.opencode_free_ids(url + "/v1") == {"x-free": True, "y-free": False}
    finally:
        srv.shutdown()


def test_a_shared_free_limit_benches_the_whole_group_at_once():
    shared = ModelUnavailable("x: HTTP 429", status=429, retry_after_s=47197.0,
                              detail='{"error":{"type":"FreeUsageLimitError","message":"Free usage limit reached."}}')
    a, b, c, other = Fake("a", fail=shared), Fake("b"), Fake("c"), Fake("other")
    p = ModelPool("online", [Member(a, group="opencode"), Member(b, group="opencode"), Member(c, group="opencode"),
                             Member(other, group="agy")], clock=Clock())
    assert "".join(p.stream(MSGS)) == "ok"
    assert (b.calls, c.calls, other.calls) == (0, 0, 1)  # one 429, not three
    assert [round(s.benched_for_s) for s in p.states()] == [47197, 47197, 47197, 0]
    one = ModelUnavailable("x: HTTP 429", status=429)
    d, e = Fake("d", fail=one), Fake("e")
    p = ModelPool("online", [Member(d, group="opencode"), Member(e, group="opencode")], clock=Clock())
    list(p.stream(MSGS))
    assert [s.live for s in p.states()] == [False, True]  # a plain 429 is that model's own
