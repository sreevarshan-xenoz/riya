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


def _v1_db(path):
    """A database exactly as v0.1/v0.2 left it: schema 1 with a fact, a transcript and a setting."""
    import sqlite3

    from riya.core import store
    conn = sqlite3.connect(str(path))
    conn.executescript(store._V1 + "pragma user_version=1;")
    conn.execute("insert into facts(text, source, confidence, valid_from, status, created_at) "
                 "values('The user likes tea', 'user_said', 1.0, 1.0, 'active', 1.0)")
    conn.execute("insert into transcripts(at, role, text) values(2.0, 'user', 'hi')")
    conn.execute("""insert into kv(key, value) values('settings', '{"mode": "panel"}')""")
    conn.commit()
    conn.close()


def test_a_v1_database_migrates_to_v2_keeping_its_data(tmp_path):
    from riya.core.embed import HashEmbedder
    from riya.core.memory import Memory
    from riya.core.settings import load_settings
    p = tmp_path / "riya.db"
    _v1_db(p)
    db = open_db(p)
    assert SCHEMA_VERSION == 2 and db.execute("pragma user_version").fetchone()[0] == 2
    mem = Memory(db, HashEmbedder(), FakeClock())
    assert [f.text for f in mem.list_facts()] == ["The user likes tea"]
    assert [r["text"] for r in mem.unconsolidated()] == ["hi"]
    assert load_settings(KV(db)).mode == "panel" and load_settings(KV(db)).eyes_enabled is True
    g = mem.add_glance("VS Code", "main.py", "coding", "", None)
    assert mem.list_glances()[0].id == g.id
    db.close()
    again = open_db(p)                                  # opening again doesn't migrate twice
    assert again.execute("select count(*) from glances").fetchone()[0] == 1


def test_a_failed_migration_leaves_the_v1_database_untouched(tmp_path, monkeypatch):
    import sqlite3

    import pytest

    from riya.core import store
    p = tmp_path / "riya.db"
    _v1_db(p)
    monkeypatch.setattr(store, "_MIGRATIONS", [store._V1, "create table glances(id integer primary key);"
                                                         "create table broken(;"])
    with pytest.raises(sqlite3.Error):
        open_db(p)
    monkeypatch.undo()
    conn = sqlite3.connect(str(p))
    assert conn.execute("pragma user_version").fetchone()[0] == 1
    assert conn.execute("select count(*) from sqlite_master where name='glances'").fetchone()[0] == 0
    conn.close()
    assert open_db(p).execute("select count(*) from facts").fetchone()[0] == 1
