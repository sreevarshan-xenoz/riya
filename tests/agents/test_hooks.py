import json
import socket
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from riya.agents.hooks import MAX_BODY, create_hook_app, load_or_create_token, start_hook_receiver

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
