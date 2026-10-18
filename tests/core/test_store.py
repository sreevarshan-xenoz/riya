from riya.core.clock import FakeClock
from riya.core.store import KV, SCHEMA_VERSION, open_db


def test_open_db_creates_parent_dir_and_tables(tmp_path):
    db = open_db(tmp_path / "nested" / "riya.db")
    names = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    assert {"facts", "episodes", "threads", "transcripts", "forgotten", "kv"} <= names
    assert db.execute("pragma journal_mode").fetchone()[0] == "wal"
    assert db.execute("pragma user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopen_keeps_data(tmp_path):
    p = tmp_path / "riya.db"
    KV(open_db(p)).set("x", {"a": 1})
    assert KV(open_db(p)).get("x") == {"a": 1}


def test_kv_default_and_overwrite():
    kv = KV(open_db(":memory:"))
    assert kv.get("missing", 5) == 5
    kv.set("k", [1, 2])
    kv.set("k", "v")
    assert kv.get("k") == "v"


def test_fake_clock():
    c = FakeClock(100.0)
    c.advance(5)
    assert c.now() == 105.0
