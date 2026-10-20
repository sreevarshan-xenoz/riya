import json
import socket
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from riya.agents.hooks import (MAX_BODY, HookReceiver, RejectCounter, create_hook_app, load_or_create_token,
                               start_hook_receiver)
from riya.agents.peer import Peer, PeerMap

TOKEN = "t" * 64
OK = {"hook_event_name": "Stop", "session_id": "s1", "cwd": "E:/riya"}


def make(enqueue=None):
    got = []
    return TestClient(create_hook_app(TOKEN, enqueue or got.append)), got


def post(client, body=None, headers=None, **kw):
    h = {"X-Riya-Token": TOKEN, **(headers or {})}
    return client.post("/hook", content=json.dumps(OK if body is None else body), headers=h, **kw)


def test_valid_request_is_200_empty_and_enqueued_once():
    c, got = make()
    r = post(c)
    assert r.status_code == 200 and r.content == b""
    assert got == [OK]


def test_missing_or_wrong_token_is_401():
    c, got = make()
    assert c.post("/hook", content=json.dumps(OK)).status_code == 401
    assert post(c, headers={"X-Riya-Token": "nope"}).status_code == 401
    assert got == []


def test_any_origin_is_403_even_with_a_good_token():
    c, got = make()
    assert post(c, headers={"Origin": "http://evil.example"}).status_code == 403
    assert post(c, headers={"Origin": ""}).status_code == 403
    assert got == []


def test_origin_is_checked_before_the_token():
    c, _ = make()
    r = c.post("/hook", content="{}", headers={"Origin": "http://x", "X-Riya-Token": "bad"})
    assert r.status_code == 403


def test_oversize_body_is_413():
    c, got = make()
    r = c.post("/hook", content=b"{" + b" " * MAX_BODY + b"}", headers={"X-Riya-Token": TOKEN})
    assert r.status_code == 413 and got == []


def test_oversize_body_without_content_length_is_413():
    c, got = make()
    chunks = (b" " * 65536 for _ in range(5))  # a generator body is sent chunked, with no Content-Length
    r = c.post("/hook", content=chunks, headers={"X-Riya-Token": TOKEN})
    assert r.status_code == 413 and got == []


def test_a_body_at_the_limit_is_accepted():
    c, got = make()
    pad = b" " * (MAX_BODY - 2)
    r = c.post("/hook", content=b"{}" + pad, headers={"X-Riya-Token": TOKEN})
    assert r.status_code == 200 and got == [{}]


def test_the_token_is_checked_before_the_size():
    c, _ = make()
    r = c.post("/hook", content=b" " * (MAX_BODY + 1), headers={"X-Riya-Token": "bad"})
    assert r.status_code == 401


@pytest.mark.parametrize("body", [b"{not json", b"", b"\xff\xfe", b"[1, 2]", b'"hi"', b"42", b"null"])
def test_malformed_or_non_dict_json_is_200_and_not_enqueued(body):
    c, got = make()
    r = c.post("/hook", content=body, headers={"X-Riya-Token": TOKEN})
    assert r.status_code == 200 and r.content == b"" and got == []


def test_a_failing_enqueue_still_answers_200():
    def boom(_):
        raise RuntimeError("full")
    c, _ = make(boom)
    assert post(c).status_code == 200


def test_only_post_hook_exists():
    c, _ = make()
    assert c.get("/hook").status_code == 405
    assert c.post("/other", content="{}").status_code == 404


def test_response_is_fast():
    c, _ = make()
    post(c)
    t = time.perf_counter()
    for _ in range(10):
        post(c)
    assert (time.perf_counter() - t) / 10 < 0.1


def test_token_is_created_once_and_reused(tmp_path):
    p = tmp_path / "sub" / "hook_token"
    a = load_or_create_token(p)
    assert len(a) == 64 and int(a, 16) >= 0
    assert p.read_text().strip() == a
    assert load_or_create_token(p) == a


def test_an_empty_token_file_is_replaced(tmp_path):
    p = tmp_path / "hook_token"
    p.write_text("")
    assert len(load_or_create_token(p)) == 64


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_receiver_starts_serves_and_stops(tmp_path):
    got, port = [], free_port()
    started = start_hook_receiver(got.append, token_path=tmp_path / "tok", port=port)
    assert started is not None
    server, thread = started
    try:
        token = (tmp_path / "tok").read_text().strip()
        for _ in range(50):
            try:
                r = httpx.post(f"http://127.0.0.1:{port}/hook", content=json.dumps(OK),
                               headers={"X-Riya-Token": token}, timeout=2)
                break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("receiver never answered")
        assert r.status_code == 200 and got == [OK]
        assert thread.name == "riya-hooks" and thread.daemon
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_receiver_returns_none_when_the_port_is_taken(tmp_path, caplog):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = s.getsockname()[1]
        assert start_hook_receiver(lambda _: None, token_path=tmp_path / "tok", port=port) is None
    assert any("8766" in r.message or "hook" in r.message.lower() for r in caplog.records)


def test_receiver_returns_none_when_the_token_cannot_be_made(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    assert start_hook_receiver(lambda _: None, token_path=blocker / "tok", port=free_port()) is None


def test_a_deeply_nested_body_is_200_and_not_enqueued():
    c, got = make()
    r = c.post("/hook", content=b"[" * 200_000, headers={"X-Riya-Token": TOKEN})
    assert r.status_code == 200 and got == []


def test_a_malformed_content_length_never_500s():
    c, got = make()
    for bad in ("²", "٣", "-5", "1e3", " "):  # on the wire headers are bytes; the server decodes them as latin-1
        r = c.post("/hook", content=json.dumps(OK),
                   headers={"X-Riya-Token": TOKEN, "Content-Length": bad.encode("latin-1", "ignore") or b"?"})
        assert r.status_code != 500, bad


def test_a_non_ascii_digit_content_length_falls_back_to_the_counted_limit():
    c, got = make()
    r = c.post("/hook", content=b" " * (MAX_BODY + 1), headers={"X-Riya-Token": TOKEN, "Content-Length": "²".encode("latin-1")})
    assert r.status_code in (200, 400, 413) and got == []


@pytest.mark.parametrize("token", ["", "   ", "\n"])
def test_an_empty_token_cannot_build_the_app(token):
    with pytest.raises(ValueError):
        create_hook_app(token, lambda _: None)


def test_the_receiver_binds_loopback_only(tmp_path, monkeypatch):
    import uvicorn
    seen = []

    class Server:
        started = True

        def __init__(self, config):
            seen.append(config)
            self.should_exit = False

        def run(self):
            pass

    monkeypatch.setattr(uvicorn, "Server", Server)
    server, thread = start_hook_receiver(lambda _: None, token_path=tmp_path / "tok", port=free_port())
    thread.join(timeout=2)
    assert seen[0].host == "127.0.0.1"


def test_a_superscript_content_length_through_raw_asgi_is_not_a_500():
    import asyncio
    app = create_hook_app(TOKEN, lambda _: None)
    sent = []

    async def run():
        body = json.dumps(OK).encode()
        msgs = iter([{"type": "http.request", "body": body, "more_body": False}])

        async def receive():
            return next(msgs, {"type": "http.disconnect"})

        async def send(m):
            sent.append(m)
        scope = {"type": "http", "method": "POST", "path": "/hook", "query_string": b"", "headers": [
            (b"x-riya-token", TOKEN.encode()), (b"content-length", "²".encode("latin-1"))],
            "http_version": "1.1", "scheme": "http", "server": ("127.0.0.1", 8766), "client": ("127.0.0.1", 1)}
        await app(scope, receive, send)

    asyncio.run(run())
    assert sent[0]["status"] == 200


# --- wrong-token counter and receiver retry (final review M4, M5) ---------------------------------------------------

class Tick:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_wrong_tokens_are_counted_and_nothing_else_is():
    rejected = RejectCounter()
    c = TestClient(create_hook_app(TOKEN, lambda _: None, rejected=rejected))
    assert c.post("/hook", content=json.dumps(OK)).status_code == 401
    assert post(c, headers={"X-Riya-Token": "nope"}).status_code == 401
    assert post(c).status_code == 200
    assert post(c, headers={"Origin": "http://x"}).status_code == 403
    assert rejected.recent() == 2


def test_the_counter_forgets_after_24_hours_and_is_thread_safe():
    import threading
    tick = Tick()
    rejected = RejectCounter(clock=tick)
    threads = [threading.Thread(target=lambda: [rejected.hit() for _ in range(500)]) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rejected.recent() == 4000
    tick.t += 24 * 3600 - 1
    rejected.hit()
    assert rejected.recent() == 4001
    tick.t += 1
    assert rejected.recent() == 1


def wait_until(cond, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_a_busy_port_is_retried_until_it_binds(tmp_path):
    got = []
    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen()
    port = blocker.getsockname()[1]
    receiver = HookReceiver(got.append, token_path=tmp_path / "tok", port=port, retry_s=0.1).start()
    try:
        assert not receiver.ok
        time.sleep(0.3)
        assert not receiver.ok                                   # still busy: still retrying
        blocker.close()
        assert wait_until(lambda: receiver.ok)
        token = (tmp_path / "tok").read_text().strip()
        r = httpx.post(f"http://127.0.0.1:{port}/hook", content=json.dumps(OK), headers={"X-Riya-Token": token},
                       timeout=2)
        assert r.status_code == 200 and got == [OK]
        httpx.post(f"http://127.0.0.1:{port}/hook", content=json.dumps(OK), headers={"X-Riya-Token": "x"}, timeout=2)
        assert receiver.rejected.recent() == 1
    finally:
        blocker.close()
        receiver.stop()
    assert not receiver.ok


def test_stopping_ends_the_retry(tmp_path):
    with socket.socket() as blocker:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        receiver = HookReceiver(lambda _: None, token_path=tmp_path / "tok", port=blocker.getsockname()[1],
                                retry_s=0.05).start()
        receiver.stop()
        assert not receiver.ok and (receiver._retry is None or not receiver._retry.is_alive())


def test_a_free_port_serves_at_once_with_no_retry_thread(tmp_path):
    receiver = HookReceiver(lambda _: None, token_path=tmp_path / "tok", port=free_port()).start()
    try:
        assert receiver.ok and receiver._retry is None
    finally:
        receiver.stop()


def test_a_pre_tool_use_post_is_an_empty_fast_200_even_when_it_is_the_hot_path(tmp_path):
    """PreToolUse fires before every tool call: the receiver must still answer an empty 200 at once."""
    got, port = [], free_port()
    started = start_hook_receiver(got.append, token_path=tmp_path / "tok", port=port)
    assert started is not None
    server, thread = started
    payload = {"hook_event_name": "PreToolUse", "session_id": "s1", "cwd": "E:/riya", "tool_name": "Bash",
               "tool_input": {"command": "git status"}}
    try:
        token = (tmp_path / "tok").read_text().strip()
        url = f"http://127.0.0.1:{port}/hook"
        for _ in range(50):
            try:
                httpx.post(url, content=json.dumps(payload), headers={"X-Riya-Token": token}, timeout=2)
                break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("receiver never answered")
        with httpx.Client(timeout=2) as client:
            took = []
            for _ in range(10):
                t = time.perf_counter()
                r = client.post(url, content=json.dumps(payload), headers={"X-Riya-Token": token})
                took.append(time.perf_counter() - t)
                assert r.status_code == 200 and r.content == b""
        assert max(took) < 1.0 and sorted(took)[len(took) // 2] < 0.1
        assert got and all(g == payload for g in got) and len(got) >= 11
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_the_receiver_records_the_peer_and_stays_fast():
    ports, got = [], []

    def lookup(port):
        ports.append(port)
        return 4242

    peers = PeerMap(lookup)
    c = TestClient(create_hook_app(TOKEN, got.append, peers=peers))
    pre = {"hook_event_name": "PreToolUse", "session_id": "s1", "cwd": "E:/riya", "tool_name": "Read"}
    for _ in range(5):
        r = post(c, pre)
        assert r.status_code == 200 and r.content == b""
    assert len(got) == 5 and peers.get("s1") == Peer(4242, "riya")
    assert ports == [50000]  # the test client's side of the connection
    post(c, {**pre, "hook_event_name": "SessionStart"})
    assert len(ports) == 2 and len(got) == 6


def test_the_peer_is_not_recorded_for_a_bad_session_id():
    ports = []
    peers = PeerMap(lambda p: ports.append(p) or 4242)
    c = TestClient(create_hook_app(TOKEN, lambda p: None, peers=peers))
    for sid in (None, 7, "", "x" * 129):
        assert post(c, {"hook_event_name": "Stop", "session_id": sid, "cwd": "E:/riya"}).status_code == 200
    assert ports == []


def test_a_failing_peer_lookup_never_breaks_the_200():
    got = []

    class Boom:
        def note(self, *a, **k):
            raise RuntimeError("peer table exploded")

    c = TestClient(create_hook_app(TOKEN, got.append, peers=Boom()))
    r = post(c)
    assert r.status_code == 200 and r.content == b"" and got == [OK]


def test_the_peers_pass_through_the_receiver_classes():
    peers = PeerMap(lambda p: None)
    assert HookReceiver(lambda p: None, peers=peers).peers is peers
    assert HookReceiver(lambda p: None).peers is None
