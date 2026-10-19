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


def test_a_v1_database_migrates_to_current_keeping_its_data(tmp_path):
    from riya.core.embed import HashEmbedder
    from riya.core.memory import Memory
    from riya.core.settings import load_settings
    p = tmp_path / "riya.db"
    _v1_db(p)
    db = open_db(p)
    assert SCHEMA_VERSION == 4 and db.execute("pragma user_version").fetchone()[0] == 4
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


def test_kv_keys_with_a_prefix_and_delete():
    kv = KV(open_db(":memory:"))
    for k in ("agents.logwatch.a", "agents.logwatch.b", "agents.logwatchX", "x%_.y", "other"):
        kv.set(k, 1)
    assert sorted(kv.keys_with("agents.logwatch.")) == ["agents.logwatch.a", "agents.logwatch.b"]
    assert kv.keys_with("x%_") == ["x%_.y"]                   # literal, not a LIKE pattern
    kv.delete("agents.logwatch.a")
    kv.delete("missing")
    assert kv.get("agents.logwatch.a") is None and kv.keys_with("agents.logwatch.") == ["agents.logwatch.b"]


def _v2_db(path):
    """A database exactly as v0.3 to v0.4 left it: schema 2 with a fact, a glance and a setting."""
    import sqlite3

    from riya.core import store
    conn = sqlite3.connect(str(path))
    conn.executescript(store._V1 + store._V2 + "pragma user_version=2;")
    conn.execute("insert into facts(text, source, confidence, valid_from, status, created_at) "
                 "values('The user likes tea', 'user_said', 1.0, 1.0, 'active', 1.0)")
    conn.execute("insert into glances(at, app, title, activity, details) values(3.0, 'Code', 't', 'coding', '')")
    conn.execute("""insert into kv(key, value) values('settings', '{"mode": "panel"}')""")
    conn.commit()
    conn.close()


def test_a_v2_database_migrates_to_current_keeping_its_rows(tmp_path):
    p = tmp_path / "riya.db"
    _v2_db(p)
    db = open_db(p)
    assert db.execute("pragma user_version").fetchone()[0] == 4
    assert db.execute("select text from facts").fetchone()[0] == "The user likes tea"
    assert db.execute("select count(*) from glances").fetchone()[0] == 1
    assert KV(db).get("settings") == {"mode": "panel"}
    names = {r[0] for r in db.execute("select name from sqlite_master")}
    assert {"reminders", "notes", "actions", "reminders_due", "actions_at"} <= names
    db.close()
    assert open_db(p).execute("pragma user_version").fetchone()[0] == 4        # a second open doesn't migrate again


def test_a_failed_v3_migration_leaves_the_v2_database_untouched(tmp_path, monkeypatch):
    import sqlite3

    import pytest

    from riya.core import store
    p = tmp_path / "riya.db"
    _v2_db(p)
    monkeypatch.setattr(store, "_MIGRATIONS", [store._V1, store._V2, "create table reminders(id integer primary key);"
                                                                    "create table broken(;"])
    with pytest.raises(sqlite3.Error):
        open_db(p)
    monkeypatch.undo()
    conn = sqlite3.connect(str(p))
    assert conn.execute("pragma user_version").fetchone()[0] == 2
    assert conn.execute("select count(*) from sqlite_master where name='reminders'").fetchone()[0] == 0
    conn.close()
    assert open_db(p).execute("select count(*) from facts").fetchone()[0] == 1


# ---- final review: a future schema is refused untouched; a migration keeps a backup first ---------------------------
def test_a_db_from_a_newer_riya_is_refused_and_never_written(tmp_path):
    import hashlib
    import sqlite3

    import pytest
    p = tmp_path / "riya.db"
    conn = sqlite3.connect(str(p))
    conn.executescript(f"create table future(x); insert into future values(1); pragma user_version={SCHEMA_VERSION + 1};")
    conn.close()
    before = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(RuntimeError, match=f"newer Riya.*schema {SCHEMA_VERSION + 1}"):
        open_db(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest() == before
    assert sorted(x.name for x in tmp_path.iterdir()) == ["riya.db"]   # no -wal, no backup
    conn = sqlite3.connect(str(p))
    assert conn.execute("pragma journal_mode").fetchone()[0] == "delete"
    conn.close()


def test_a_migration_backs_the_db_up_first_once_per_version(tmp_path):
    import sqlite3
    p = tmp_path / "riya.db"
    _v2_db(p)
    open_db(p).close()
    bak = tmp_path / f"riya.db.pre-v{SCHEMA_VERSION}.bak"
    old = sqlite3.connect(str(bak))
    assert old.execute("pragma user_version").fetchone()[0] == 2
    assert old.execute("select text from facts").fetchone()[0] == "The user likes tea"
    old.close()
    first = bak.read_bytes()
    open_db(p).close()                                    # already current: no new backup
    assert bak.read_bytes() == first
    assert sorted(x.name for x in tmp_path.iterdir() if x.suffix == ".bak") == [bak.name]


def test_a_new_db_has_no_backup(tmp_path):
    open_db(tmp_path / "riya.db").close()
    assert not [x for x in tmp_path.iterdir() if x.suffix == ".bak"]


def test_an_existing_backup_is_never_overwritten(tmp_path):
    p = tmp_path / "riya.db"
    _v2_db(p)
    bak = tmp_path / f"riya.db.pre-v{SCHEMA_VERSION}.bak"
    bak.write_bytes(b"the first backup")
    open_db(p).close()
    assert bak.read_bytes() == b"the first backup"


# ---- v0.6: schema v4 on a populated v3 database -------------------------------------------------------------------
def _v3_db(path):
    """A database exactly as v0.5 left it: schema 3 with a fact, a setting, a glance, a reminder, a note and an action."""
    import sqlite3

    from riya.core import store
    conn = sqlite3.connect(str(path))
    conn.executescript(store._V1 + store._V2 + store._V3 + "pragma user_version=3;")
    conn.execute("insert into facts(text, source, confidence, valid_from, status, created_at) "
                 "values('The user likes tea', 'user_said', 1.0, 1.0, 'active', 1.0)")
    conn.execute("""insert into kv(key, value) values('settings', '{"mode": "panel"}')""")
    conn.execute("insert into glances(at, app, title, activity, details) values(3.0, 'Code', 't', 'coding', '')")
    conn.execute("insert into reminders(kind, due_at, text, created_at) values('timer', 9.0, 'tea', 4.0)")
    conn.execute("insert into notes(at, text) values(5.0, 'buy milk')")
    conn.execute("insert into actions(at, tool, tier, summary, outcome) values(6.0, 'open_app', 1, 'opened x', 'ran')")
    conn.commit()
    conn.close()


_V3_ROWS = {"facts": 1, "glances": 1, "reminders": 1, "notes": 1, "actions": 1, "kv": 1}


def _counts(conn):
    return {t: conn.execute(f"select count(*) from {t}").fetchone()[0] for t in _V3_ROWS}


def test_a_populated_v3_db_migrates_to_v4(tmp_path):
    import sqlite3

    from riya.core.learn.records import LearnDB
    from riya.core.persona import load_traits
    p = tmp_path / "riya.db"
    _v3_db(p)
    db = open_db(p)
    assert db.execute("pragma user_version").fetchone()[0] == 4
    assert _counts(db) == _V3_ROWS
    assert db.execute("select text from facts").fetchone()[0] == "The user likes tea"
    assert KV(db).get("settings") == {"mode": "panel"}
    assert db.execute("select text from notes").fetchone()[0] == "buy milk"
    assert tuple(db.execute("select tool, outcome from actions").fetchone()) == ("open_app", "ran")
    assert tuple(db.execute("select text, due_at from reminders").fetchone()) == ("tea", 9.0)
    names = {r[0] for r in db.execute("select name from sqlite_master")}
    assert {"trait_history", "style_notes", "twin", "events", "events_at"} <= names
    assert LearnDB(db).dials() == load_traits()
    db.close()
    bak = tmp_path / "riya.db.pre-v4.bak"
    old = sqlite3.connect(str(bak))
    assert old.execute("pragma user_version").fetchone()[0] == 3
    assert _counts(old) == _V3_ROWS
    assert old.execute("select count(*) from sqlite_master where name='twin'").fetchone()[0] == 0
    old.close()
    mtime = bak.stat().st_mtime_ns
    again = open_db(p)                                    # already current: no second migration, no new backup
    assert again.execute("pragma user_version").fetchone()[0] == 4 and _counts(again) == _V3_ROWS
    assert bak.stat().st_mtime_ns == mtime
    again.close()


def test_a_failed_v4_migration_leaves_the_v3_database_untouched(tmp_path, monkeypatch):
    import sqlite3

    import pytest

    from riya.core import store
    p = tmp_path / "riya.db"
    _v3_db(p)
    monkeypatch.setattr(store, "_MIGRATIONS", [store._V1, store._V2, store._V3,
                                               "create table trait_history(id integer primary key);"
                                               "create table broken(;"])
    with pytest.raises(sqlite3.Error):
        open_db(p)
    monkeypatch.undo()
    conn = sqlite3.connect(str(p))
    assert conn.execute("pragma user_version").fetchone()[0] == 3
    assert conn.execute("select count(*) from sqlite_master where name='trait_history'").fetchone()[0] == 0
    assert _counts(conn) == _V3_ROWS
    conn.close()
    assert open_db(p).execute("select count(*) from facts").fetchone()[0] == 1
