import sqlite3

import pytest

from riya.core.clock import FakeClock
from riya.core.learn.dials import BANDS, DIALS
from riya.core.learn.records import DIALS_KEY, DRIFT_KEY, EVENTS_KEEP_S, MAX_NOTES, POLARITIES, LearnDB
from riya.core.learn.twin import Cell, Twin, evidence, mean
from riya.core.persona import load_traits
from riya.core.settings import Settings, apply_patch
from riya.core.store import KV, open_db

NOW = 1_800_000_000.0
DAY = 86400.0


def db():
    return LearnDB(open_db(":memory:"))


def test_defaults_are_the_traits_file():
    d = db().dials()
    assert d == load_traits() and d["emoji"] == 0.3 and set(d) == set(DIALS)


def test_nudge_returns_old_and_new_and_writes_history():
    d = db()
    reason = 'they said "stop teasing me"'
    assert d.nudge("teasing", -0.15, reason, "explicit", NOW) == (0.5, 0.35)
    assert d.dials()["teasing"] == 0.35
    h = d.history()[0]
    assert (h.dial, h.old, h.new, h.reason, h.source, h.at) == ("teasing", 0.5, 0.35, reason, "explicit", NOW)


def test_a_nudge_that_changes_nothing_writes_no_history():
    d = db()
    for _ in range(3):
        d.nudge("teasing", -0.15, "r", "explicit", NOW)
    assert d.nudge("teasing", -0.15, "r", "explicit", NOW) == (0.1, 0.1)
    assert len(d.history(100)) == 3


def test_nudges_never_leave_the_band():
    d = db()
    for _ in range(20):
        d.nudge("teasing", -0.15, "stop teasing", "explicit", NOW)
    assert d.dials()["teasing"] == 0.1
    assert [(h.old, h.new) for h in reversed(d.history(100))] == [(0.5, 0.35), (0.35, 0.2), (0.2, 0.1)]
    for dial in DIALS:
        for _ in range(20):
            d.nudge(dial, 0.15, "more", "explicit", NOW)
    assert d.dials() == {k: BANDS[k][1] for k in DIALS}


def test_nudge_rejects_bad_input():
    d = db()
    with pytest.raises(ValueError):
        d.nudge("bogus", 0.1, "r", "explicit", NOW)
    with pytest.raises(ValueError):
        d.nudge("teasing", 0.1, "r", "whim", NOW)
    assert d.history() == []


def test_corrupt_kv_falls_back_per_dial():
    conn = open_db(":memory:")
    KV(conn).set(DIALS_KEY, {"teasing": 5, "emoji": "x", "verbosity": True, "bogus": 1})
    d = LearnDB(conn).dials()
    assert d["teasing"] == 0.8 and d["emoji"] == 0.3 and d["verbosity"] == 0.35 and "bogus" not in d
    assert set(d) == set(DIALS) and d["sweetness"] == 0.85
    for junk in ([1, 2], "x", 7, None):                   # not even a mapping: every default
        KV(conn).set(DIALS_KEY, junk)
        assert LearnDB(conn).dials() == load_traits()
    conn.execute("insert or replace into kv(key, value) values(?, ?)",
                 (DIALS_KEY, '{"teasing": NaN, "emoji": Infinity, "clinginess": -Infinity}'))
    assert LearnDB(conn).dials() == {**load_traits(), "emoji": 0.3, "clinginess": 0.8}   # non-finite never counts


def test_reset_restores_the_defaults_and_logs_it():
    d = db()
    d.nudge("teasing", -0.15, "r", "explicit", NOW)
    d.nudge("verbosity", 0.15, "r", "explicit", NOW)
    d.reset_dials(NOW + 1)
    assert d.dials() == load_traits()
    h = d.history()
    assert {x.dial for x in h[:2]} == {"teasing", "verbosity"} and h[2].source == "explicit"
    assert all(x.source == "reset" and x.reason == "reset to her defaults" and x.at == NOW + 1 for x in h[:2])
    d.reset_dials(NOW + 2)                                # nothing changed, nothing logged
    assert len(d.history(100)) == 4


def test_set_dials_logs_one_row_per_changed_dial():
    d = db()
    d.set_dials({"teasing": 0.3, "sweetness": 0.85, "emoji": 0.0}, "why", "explicit", NOW)
    assert {h.dial for h in d.history()} == {"teasing", "emoji"}
    assert d.dials()["teasing"] == 0.3 and d.dials()["emoji"] == 0.0


def test_history_is_capped_and_newest_first():
    d = db()
    for i in range(30):
        d.set_dials({"verbosity": 0.2 if i % 2 else 0.6}, f"r{i}", "explicit", NOW + i)
    h = d.history()
    assert len(h) == 20 and h[0].reason == "r29" and h[-1].reason == "r10"
    assert len(d.history(5)) == 5


def test_notes_refresh_duplicates_and_cap_at_thirty():
    d = db()
    first = d.add_note("They like it when I ask about their day", "like", NOW)
    assert isinstance(first, int)
    assert d.add_note("They like it when I ask about their day", "like", NOW + 5) == first
    assert len(d.notes()) == 1 and d.notes()[0].at == NOW + 5 and d.notes()[0].active
    ids = [d.add_note(f"note {i}", "rule", NOW + 10 + i) for i in range(30)]   # 31 in all
    assert len(d.notes()) == MAX_NOTES
    assert d.conn.execute("select count(*) from style_notes").fetchone()[0] == 30
    assert d.conn.execute("select count(*) from style_notes where id=?", (first,)).fetchone()[0] == 0
    assert [n.id for n in d.notes()] == ids[::-1]
    d.add_note("note 0", "rule", NOW + 100)               # a refreshed note is no longer the oldest
    d.add_note("another", "rule", NOW + 101)
    texts = [n.text for n in d.notes()]
    assert "note 0" in texts and "note 1" not in texts and len(texts) == 30


def test_note_text_is_stored_as_data():
    d = db()
    evil = "x'); drop table style_notes; --"
    d.add_note(evil, "dislike", NOW)
    assert [n.text for n in d.notes()] == [evil] and d.notes()[0].polarity == "dislike"


def test_delete_and_clear_notes_are_real_deletes():
    d = db()
    a = d.add_note("a", "like", NOW)
    d.add_note("b", "rule", NOW + 1)
    assert d.delete_note(a) is True and d.delete_note(a) is False and d.delete_note(999) is False
    assert d.conn.execute("select count(*) from style_notes").fetchone()[0] == 1
    d.clear_notes()
    assert d.notes() == [] and d.conn.execute("select count(*) from style_notes").fetchone()[0] == 0


def test_bad_polarity_raises():
    d = db()
    assert POLARITIES == ("like", "dislike", "rule")
    with pytest.raises(ValueError):
        d.add_note("x", "love", NOW)
    assert d.notes() == []


def test_twin_cells_round_trip_and_reset():
    d = db()
    assert d.get_cell("bid", "morning/here/short") is None
    d.put_cell("bid", "morning/here/short", Cell(3.0, 2.0, NOW))
    d.put_cell("bid", "morning/here/short", Cell(4.0, 2.0, NOW + 1))     # upsert
    d.put_cell("bid", "evening/here/long", Cell(2.0, 5.0, NOW))
    d.put_cell("greeting", "evening/here/long", Cell(2.5, 2.0, NOW))
    assert d.get_cell("bid", "morning/here/short") == Cell(4.0, 2.0, NOW + 1)
    assert [(k, b) for k, b, _ in d.cells()] == [("bid", "evening/here/long"), ("bid", "morning/here/short"),
                                                 ("greeting", "evening/here/long")]
    d.reset_twin("bid")
    assert [k for k, _, _ in d.cells()] == ["greeting"]
    d.reset_twin(None)
    assert d.cells() == []


def test_twin_persists_across_reopening(tmp_path):
    p = tmp_path / "riya.db"
    clock = FakeClock(NOW)
    conn = open_db(p)
    Twin(LearnDB(conn), clock).record("bid", "evening/here/long", False)
    conn.close()
    again = Twin(LearnDB(open_db(p)), clock)
    assert again.p_engage("bid", "evening/here/long") == pytest.approx(2 / 5)


def test_events_round_trip_and_validate():
    d = db()
    d.log_event(NOW, "bid", "engaged", {"bucket": "evening/here/long", "n": 2, "u": "é"})
    d.log_event(NOW + 1, "mood", "up", {})
    ev = d.events()
    assert [e.kind for e in ev] == ["mood", "bid"]
    assert ev[1].context == {"bucket": "evening/here/long", "n": 2, "u": "é"}
    assert ev[1].outcome == "engaged" and ev[1].at == NOW
    assert len(d.events(1)) == 1
    with pytest.raises(ValueError):
        d.log_event(NOW, "nope", "engaged", {})
    with pytest.raises(ValueError):
        d.log_event(NOW, "bid", "nope", {})
    assert len(d.events()) == 2


def test_purge_keeps_29_days_and_drops_31():
    d = db()
    d.log_event(NOW - 29 * DAY, "bid", "ignored", {})
    d.log_event(NOW - 31 * DAY, "bid", "ignored", {})
    assert EVENTS_KEEP_S == 30 * DAY
    d.purge(NOW)
    assert [e.at for e in d.events()] == [NOW - 29 * DAY]


def test_drift_time():
    d = db()
    assert d.drift_at() is None
    d.set_drift_at(NOW)
    assert d.drift_at() == NOW and KV(d.conn).get(DRIFT_KEY) == NOW


def test_the_lock_can_be_shared():
    conn = open_db(":memory:")
    kv = KV(conn)
    assert LearnDB(conn, kv.lock).lock is kv.lock


def test_every_write_commits(tmp_path):
    p = tmp_path / "riya.db"
    d = LearnDB(open_db(p))
    d.nudge("teasing", -0.15, "r", "explicit", NOW)
    d.add_note("n", "like", NOW)
    d.log_event(NOW, "bid", "engaged", {})
    d.put_cell("bid", "b", Cell(3.0, 2.0, NOW))
    d.set_drift_at(NOW)
    other = sqlite3.connect(str(p))
    for t in ("trait_history", "style_notes", "events", "twin"):
        assert other.execute(f"select count(*) from {t}").fetchone()[0] == 1
    assert other.execute("select count(*) from kv where key in (?, ?)", (DIALS_KEY, DRIFT_KEY)).fetchone()[0] == 2
    other.close()


def test_learning_enabled_setting():
    s = Settings()
    assert s.learning_enabled is True
    assert apply_patch(s, {"learning_enabled": False}).learning_enabled is False
    with pytest.raises(ValueError):
        apply_patch(s, {"learning_enabled": "yes"})


@pytest.mark.parametrize("a,b", [(0.0, 2.0), (2.0, -1.0), ("abc", 2.0), (2.0, float("inf")), (-3.0, -3.0)])
def test_a_corrupt_twin_row_reads_as_the_prior(a, b):
    d = db()
    d.conn.execute("insert into twin(kind, bucket, alpha, beta, updated_at) values('bid', 'x', ?, ?, ?)", (a, b, NOW))
    d.conn.commit()
    assert d.get_cell("bid", "x") == Cell(2.0, 2.0, NOW)
    assert [c for _, _, c in d.cells()] == [Cell(2.0, 2.0, NOW)]
    assert Twin(d, FakeClock(NOW)).p_engage("bid", "x") == 0.5


def test_huge_or_non_numeric_kv_never_raises():
    conn = open_db(":memory:")
    big = "9" * 400
    conn.execute("insert or replace into kv(key, value) values(?, ?)", (DIALS_KEY, '{"teasing": %s, "emoji": -%s}' % (big, big)))
    conn.execute("insert or replace into kv(key, value) values(?, ?)", (DRIFT_KEY, big))
    d = LearnDB(conn)
    assert d.dials() == load_traits()
    assert d.drift_at() is None
    for junk in ("[1]", '"x"', "{", ""):
        conn.execute("insert or replace into kv(key, value) values(?, ?)", (DRIFT_KEY, junk))
        conn.execute("insert or replace into kv(key, value) values(?, ?)", (DIALS_KEY, junk))
        assert d.drift_at() is None and d.dials() == load_traits()


def test_a_v1_shaped_defaults_dict_gets_emoji_0_3():
    v1 = {"sweetness": 0.85, "teasing": 0.5, "clinginess": 0.8, "verbosity": 0.35}
    d = LearnDB(open_db(":memory:"), defaults=v1)
    assert d.dials() == load_traits() and d.dials()["emoji"] == 0.3
    d.nudge("emoji", 0.15, "r", "explicit", NOW)
    d.reset_dials(NOW + 1)
    assert d.dials()["emoji"] == 0.3
    assert LearnDB(open_db(":memory:"), defaults={}).dials() == load_traits()


def test_traits_file_ends_with_a_newline():
    from riya.core.persona import TRAITS_PATH
    assert TRAITS_PATH.read_bytes().endswith(b"\n")


@pytest.mark.parametrize("a,b,t", [(5.0, 2.0, "oops"), (5.0, 2.0, b"\x00"), (5.0, 2.0, float("inf")),
                                   (5.0, 2.0, float("-inf")), ("x", 2.0, "oops")])
def test_a_corrupt_twin_updated_at_reads_as_the_prior(a, b, t):
    d = db()
    d.conn.execute("insert into twin(kind, bucket, alpha, beta, updated_at) values('bid', 'x', ?, ?, ?)", (a, b, t))
    d.conn.commit()
    tw = Twin(d, FakeClock(NOW))
    assert tw.p_engage("bid", "x") == 0.5 and tw.chance("bid", "x") == 1.0
    assert mean(d.cells()[0][2], NOW) == 0.5 and evidence(d.get_cell("bid", "x"), NOW) == 0.0
    tw.record("bid", "x", True)
    assert d.get_cell("bid", "x") == Cell(3.0, 2.0, NOW)


def test_a_cell_with_a_bad_time_decays_as_the_prior():
    assert mean(Cell(5.0, 2.0, "oops"), NOW) == 0.5 and mean(Cell(5.0, 2.0, float("nan")), NOW) == 0.5
