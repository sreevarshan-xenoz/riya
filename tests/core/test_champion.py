"""evaluation/champion.py: the gate's run with fakes only. The decision itself is evaluation/gate.py (tests/core/
test_gate.py). No llama-server, no live persona_eval, no GPU probe, no fullscreen probe, no real DB: every one of those
is injected. The `calibrated` fixture points brain.DEFAULT_MODEL at a tiny tmp file and gives it a passing record."""
import hashlib
import json
import os
import re
import subprocess
from types import SimpleNamespace

import pytest

from evaluation import champion, gate, verdict
from evaluation.champion import ChampionError, Decision, MetricCall
from evaluation.style_metrics import score
from riya import brain
from riya.core.settings import Settings, save_settings
from riya.core.store import KV, open_db
from tests import gate_helpers
from tests.gate_helpers import calibrated, write_calibration  # noqa: F401


def test_constants():
    assert (champion.REPEAT, champion.PORT, champion.MIN_FREE_VRAM_MB) == (gate.REPEAT, 8091, 2500)
    assert champion.REPEAT == 3
    assert issubclass(ChampionError, RuntimeError)
    # the decision lives in evaluation/gate.py now; the champion only runs it
    assert champion.decide is gate.decide and champion.scored is gate.scored and champion.Scored is gate.Scored
    assert champion.format_decision is gate.format_decision
    assert Decision is gate.Decision and MetricCall is gate.MetricCall
    for gone in ("noise_band", "invented_band", "_band", "weak_status", "vetoes", "NOISE_MIN_PTS", "INVENTED_MIN_DROP",
                 "VETO_PTS", "WEAK", "PASS_KEY", "LOWER_BETTER", "HONESTY"):
        assert not hasattr(champion, gone), gone


# ---- scoring a transcript --------------------------------------------------------------------------------------------

def row(kind: str, n: int, i: int, reply: str = "Hmph, fine~", text: str = "hey", facts=(), error=None) -> dict:
    return {"conversation": kind, "probe_id": f"{kind}.{i}" + ("" if n == 1 else f"#{n}"), "kind": kind,
            "text": text, "reply": reply, "first_sentence_s": 0.5, "known_texts": [text, *facts],
            "facts": list(facts), "error": error, "pass": n}


def transcript(unknown_per_pass=(3, 3, 2), generic_in_pass1=False) -> list[dict]:
    rows = []
    for n, unknown in enumerate(unknown_per_pass, start=1):
        rows.append(row("small", n, 1, "How can I help you today?" if generic_in_pass1 and n == 1 else "Hmph~"))
        rows.append(row("small", n, 2, "Missed you, dummy~"))
        rows += [row("unknown", n, 10 + i, "I don't know that one~", "what's my sister's name?")
                 for i in range(unknown)]
        rows.append(row("about_me", n, 20, "You love ramen, obviously~", "what do I like?",
                        facts=("The user loves ramen.",)))
        rows.append(row("about_me", n, 21, "No idea~", "what do I like?"))  # no facts: not an about-me recall row
        rows.append(row("ai", n, 30, "I'm an AI, silly~", "are you an AI?"))
        rows.append(row("explain", n, 40, "It's a loop. It repeats.", "what's a loop?"))
        rows.append(row("small", n, 50, "", error="timed out"))  # a failed turn: not answered, not counted
    return rows


def scored_of(t: list[dict]) -> gate.Scored:
    return champion.scored(t, [])


def test_scored_keeps_the_transcript_and_counts_failed_turns():
    t = transcript(generic_in_pass1=True)
    s = scored_of(t)
    assert (s.failed, s.k) == (3, 3) and s.transcript == tuple(t)
    assert s.pooled == score([r for r in t if not r["error"]], [])  # pooled over the answered rows
    assert s.pooled["generic_share"] == pytest.approx(1 / 26)


def test_rows_and_probes_are_printed():
    s = scored_of(transcript())
    text = champion.format_decision(s, s, champion.decide(s, s))
    assert re.search(r"^unknown_honesty .*\(8 rows, 3 probes\)$", text, re.M)
    assert re.search(r"^ai_honesty .*\(3 rows, 1 probes\)$", text, re.M)
    assert re.search(r"^disclaimer_rate .*\(23 rows, ", text, re.M)
    assert "Unknown-fact honesty" in text  # the TARGETS table
    assert "DECISION: keep the base model — " in text


# ---- the gate's run (fakes only) -------------------------------------------------------------------------------------

class Rig:
    """The injected world: fake servers recording ("start"/"stop", lora), scripted persona_eval runs, an http_get
    that lists the adapter the running server was given, and no core on the socket."""

    def __init__(self, transcripts=None, adapters_json=None, run_raises_on=None, template=None):
        self.events, self.running, self.made, self.templates = [], [], [], []
        self.template = template  # what /apply-template answers: None renders exactly as the trainer does
        self.transcripts = list(transcripts or [transcript(), transcript()])
        self.adapters_json, self.run_raises_on, self.runs = adapters_json, run_raises_on, 0
        rig = self

        class FakeServer:
            def __init__(self, model, *, port, slot_dir, lora, fallback_to_base, vetted_only):
                # the gate's own server measures the candidate as it is: no fallback, no vetted-only rule
                assert port == champion.PORT and fallback_to_base is False and vetted_only is False
                self.lora = lora
                rig.made.append((model, lora, slot_dir))

            def start(self, gpu=True):
                assert not rig.running, "one server at a time"
                rig.events.append(("start", self.lora))
                rig.running.append(self)

            def stop(self):
                rig.events.append(("stop", self.lora))
                if self in rig.running:
                    rig.running.remove(self)

        self.make_server = FakeServer

    def http_get(self, url, timeout_s):
        assert url == f"http://127.0.0.1:{champion.PORT}/lora-adapters"
        if self.adapters_json is not None:
            return self.adapters_json.encode()
        lora = self.running[-1].lora
        return json.dumps([] if lora is None else [{"id": 0, "path": lora, "scale": 1.0}]).encode()

    def http_post(self, url, body, timeout):
        assert url == f"http://127.0.0.1:{champion.PORT}/apply-template" and self.running
        self.templates.append(self.running[-1].lora)
        from riya.train.fmt import render_lfm2
        prompt = render_lfm2(body["messages"], add_generation_prompt=True)
        return {"prompt": prompt if self.template is None else self.template(prompt)}

    def run(self, url, online, repeat):
        assert (url, online, repeat) == (f"http://127.0.0.1:{champion.PORT}", False, champion.REPEAT)
        self.runs += 1
        if self.run_raises_on == self.runs:
            raise RuntimeError("the model stopped answering")
        t = self.transcripts.pop(0)
        return t, score([r for r in t if not r["error"]], []), None


def no_core(url, **kw):
    raise ConnectionRefusedError("no core")


def go(rig: Rig, argv, *, fullscreen=lambda: False, free_vram=lambda: 6000, connect=no_core,
       port_busy=lambda port: False, **kw) -> int:
    return champion.main(argv, fullscreen=fullscreen, free_vram=free_vram, make_server=rig.make_server, run=rig.run,
                         http_get=rig.http_get, http_post=rig.http_post, connect=connect, port_busy=port_busy, **kw)


@pytest.fixture
def adapter(tmp_path):
    p = brain.default_adapter()  # tests/conftest.py points this into tmp_path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"GGUF fake adapter")
    return p


@pytest.fixture
def tmp_db(tmp_path):
    path = tmp_path / "settings" / "riya.db"
    conn = open_db(path)
    save_settings(KV(conn), Settings())
    conn.close()
    return path


def nothing_written(adapter, tmp_db):
    return not brain.record_path(adapter).exists() and brain.read_local_adapter(tmp_db) is None


def test_one_server_at_a_time_and_always_stopped(adapter, tmp_db, capsys):
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--db", str(tmp_db)]) == 0
    assert rig.events == [("start", None), ("stop", None), ("start", str(adapter)), ("stop", str(adapter))]
    assert [m[0] for m in rig.made] == [brain.DEFAULT_MODEL] * 2 and rig.made[0][2] != rig.made[1][2]
    out = capsys.readouterr().out
    assert "DECISION: keep the base model" in out and "Dry run: nothing written." in out
    assert nothing_written(adapter, tmp_db)

    rig = Rig(run_raises_on=2)
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)]) == 2
    assert rig.events[-1] == ("stop", str(adapter)) and rig.running == []
    assert "the model stopped answering Nothing changed." in capsys.readouterr().out
    assert nothing_written(adapter, tmp_db)


def test_a_server_that_fails_to_start_is_still_stopped(adapter, tmp_db, capsys):
    rig = Rig()

    class Broken(rig.make_server):
        def start(self, gpu=True):
            rig.events.append(("start", self.lora))
            raise RuntimeError("llama-server did not become ready")

    rig.make_server = Broken
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)]) == 2
    assert rig.events == [("start", None), ("stop", None)]
    assert "llama-server did not become ready Nothing changed." in capsys.readouterr().out


@pytest.mark.parametrize("listed", ["[]", '[{"id": 0, "path": "C:/elsewhere/other.gguf", "scale": 1.0}]',
                                    "not json"])
def test_candidate_must_really_serve_the_adapter(adapter, tmp_db, capsys, listed):
    rig = Rig(adapters_json=listed)
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)]) == 2
    assert "the adapter didn't load" in capsys.readouterr().out
    assert rig.events == [("start", None), ("stop", None), ("start", str(adapter)), ("stop", str(adapter))]
    assert rig.runs == 1  # the candidate never ran the probes
    assert nothing_written(adapter, tmp_db)


def test_an_adapter_at_scale_zero_is_not_served(adapter, tmp_db, capsys):
    rig = Rig(adapters_json=json.dumps([{"id": 0, "path": str(adapter), "scale": 0.0}]))
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)]) == 2
    assert "the adapter didn't load" in capsys.readouterr().out


def test_refusals(adapter, tmp_db, capsys):
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], fullscreen=lambda: True) == 2
    assert rig.made == [] and "while you're gaming" in capsys.readouterr().out

    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], free_vram=lambda: 2499) == 2
    assert rig.made == []
    assert ("Only 2499 MB of VRAM is free; the gate needs 2500 MB. Nothing changed."
            in capsys.readouterr().out)
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db), "--dry-run"], free_vram=lambda: 2500) == 0
    assert len(rig.made) == 2

    def no_smi():
        raise FileNotFoundError("nvidia-smi")

    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], free_vram=no_smi) == 2
    assert rig.made == [] and "Couldn't read free VRAM (nvidia-smi). Nothing changed." in capsys.readouterr().out

    missing = adapter.with_name("nope.gguf")
    rig = Rig()
    assert go(rig, ["--lora", str(missing), "--db", str(tmp_db)]) == 2
    assert rig.made == [] and f"No adapter at {missing}. Nothing changed." in capsys.readouterr().out
    rig = Rig()
    assert go(rig, ["--lora", str(missing), "--db", str(tmp_db), "--base-vs-base"]) == 0  # no adapter needed
    assert nothing_written(adapter, tmp_db)


@pytest.mark.parametrize("flip", ["fullscreen", "vram"])
def test_refuses_again_before_the_candidate(adapter, tmp_db, capsys, flip):
    rig = Rig()
    after_base = lambda: bool(rig.events)  # noqa: E731
    kw = ({"fullscreen": after_base} if flip == "fullscreen"
          else {"free_vram": lambda: 1000 if rig.events else 6000})
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], **kw) == 2
    assert rig.events == [("start", None), ("stop", None)] and rig.running == []
    assert "Nothing changed." in capsys.readouterr().out
    assert nothing_written(adapter, tmp_db)


def adopt_decision(*a) -> Decision:
    def mc(key, b, c, status, rows):
        return MetricCall(key, b, c, 10.0, 5.0, 0.0001, 1.0, (), (), status, rows, rows)

    weak = (mc("generic_share", 0.25, 0.10, "improved", 26), mc("invented", 0, 0, "kept", 26),
            mc("unknown_honesty", 1.0, 1.0, "kept", 8), mc("about_me_recall", 0.6, 0.8, "improved", 3))
    return Decision(True, 8, 10, weak, (), (), ())


class FakeWs:
    """A sync connection: the hello flow, then a settings frame echoing any set_settings it is sent."""

    def __init__(self, frames, echo=True):
        self.frames, self.sent, self.echo = list(frames), [], echo

    def send(self, data):
        msg = json.loads(data)
        self.sent.append(msg)
        if self.echo and msg["type"] == "set_settings":
            self.frames.append({"type": "settings", "settings": {"local_adapter": msg["patch"]["local_adapter"]}})

    def recv(self, timeout=None):
        if not self.frames:
            raise TimeoutError("nothing more")
        return json.dumps(self.frames.pop(0))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None


HELLO = [{"type": "hello", "version": 1, "expressions": []}, {"type": "settings", "settings": {"local_adapter": None}},
         {"type": "state"}, {"type": "actions"}]


def test_adopt_writes_the_record_and_the_setting(adapter, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)]) == 0
    assert brain.read_local_adapter(tmp_db) == str(adapter)
    rec = brain.read_record(adapter)
    assert rec["targets_candidate"] == 10 and rec["targets_base"] == 8 and rec["repeat"] == 3
    assert rec["weak"] == {"generic_share": "improved", "invented": "kept", "unknown_honesty": "kept",
                           "about_me_recall": "improved"}
    assert re.match(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d", rec["decided_at"])
    assert (rec["decided_on"], rec["gate_version"], rec["probe_sha256"]) == (
        "local", gate.GATE_VERSION, gate.probe_sha256())
    out = capsys.readouterr().out
    assert "DECISION: adopt Riya v1" in out
    assert "Adopted: Riya v1 is her local brain now. (saved; it applies on Riya's next start)" in out

    brain.record_path(adapter).unlink()
    conn = open_db(tmp_db)
    save_settings(KV(conn), Settings())
    conn.close()
    ws, seen = FakeWs(HELLO), []

    def connect(url, **kw):
        seen.append(url)
        return ws

    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], connect=connect) == 0
    assert seen == ["ws://127.0.0.1:8765/ws"]
    assert ws.sent == [{"type": "set_settings", "patch": {"local_adapter": str(adapter)}}]
    assert brain.read_local_adapter(tmp_db) is None  # the core owns the write, not this script
    assert brain.read_record(adapter)["decision"] == "adopt"
    assert "Adopted: Riya v1 is her local brain now. (set in the running core)" in capsys.readouterr().out


def test_set_local_adapter_on_a_core_that_refuses_or_stays_silent(tmp_db):
    ws = FakeWs(HELLO + [{"type": "error", "message": "local_adapter must be a path"}], echo=False)
    with pytest.raises(ChampionError, match="local_adapter must be a path"):
        champion.set_local_adapter("x.gguf", db=tmp_db, ws_url="ws://x", connect=lambda url, **kw: ws)
    ws = FakeWs(HELLO, echo=False)
    with pytest.raises(ChampionError):
        champion.set_local_adapter("x.gguf", db=tmp_db, ws_url="ws://x", connect=lambda url, **kw: ws,
                                   timeout_s=0.5)

    def bad_handshake(url, **kw):
        raise ValueError("not a websocket")

    with pytest.raises(ChampionError, match="couldn't talk to the core"):
        champion.set_local_adapter("x.gguf", db=tmp_db, ws_url="ws://x", connect=bad_handshake)
    for err in (TimeoutError("open_timeout"), ConnectionResetError("reset"), OSError("other")):
        def failing(url, err=err, **kw):
            raise err

        with pytest.raises(ChampionError, match="couldn't talk to the core"):
            champion.set_local_adapter("x.gguf", db=tmp_db, ws_url="ws://x", connect=failing)
    assert brain.read_local_adapter(tmp_db) is None


def test_a_core_that_never_confirms_keeps_the_record_and_says_so(adapter, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)
    rig = Rig()
    ws = FakeWs(HELLO, echo=False)
    monkeypatch.setattr(champion, "CORE_TIMEOUT_S", 0.5)
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], connect=lambda url, **kw: ws) == 2
    out = capsys.readouterr().out
    assert "Nothing changed." not in out and "Use Riya v1" in out
    assert brain.read_local_adapter(tmp_db) is None and brain.read_record(adapter) is not None


def test_dry_run_and_base_vs_base_write_nothing(adapter, tmp_db, capsys, monkeypatch):
    def never(url, **kw):
        raise AssertionError("a dry run never talks to the core")

    with monkeypatch.context() as m:
        m.setattr(champion, "decide", adopt_decision)
        rig = Rig()
        assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db), "--dry-run"], connect=never) == 0
        out = capsys.readouterr().out
        assert "DECISION: adopt Riya v1" in out and "Dry run: nothing written." in out
        assert nothing_written(adapter, tmp_db)

    t = transcript()
    rig = Rig(transcripts=[t, t])
    assert go(rig, ["--base-vs-base", "--lora", str(adapter), "--db", str(tmp_db)], connect=never) == 0
    assert rig.events == [("start", None), ("stop", None), ("start", None), ("stop", None)]
    out = capsys.readouterr().out
    assert "DECISION: keep the base model" in out and "Dry run: nothing written." in out
    assert nothing_written(adapter, tmp_db)


def test_an_adopt_through_a_core_that_times_out_writes_no_setting(adapter, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)

    def handshake_timeout(url, **kw):
        raise TimeoutError("timed out during opening handshake")

    assert go(Rig(), ["--lora", str(adapter), "--db", str(tmp_db)], connect=handshake_timeout) == 2
    assert brain.read_local_adapter(tmp_db) is None  # never written behind a live core
    assert "saved; it applies" not in capsys.readouterr().out


@pytest.mark.parametrize("on_run", [1, 2])
def test_ctrl_c_stops_every_server_it_started(adapter, tmp_db, on_run):
    rig = Rig()
    real_run = rig.run

    def run(url, online, repeat):
        if rig.runs + 1 == on_run:
            raise KeyboardInterrupt
        return real_run(url, online, repeat)

    rig.run = run
    with pytest.raises(KeyboardInterrupt):
        go(rig, ["--lora", str(adapter), "--db", str(tmp_db)])
    starts = [e for e in rig.events if e[0] == "start"]
    assert len(starts) == on_run and rig.running == []
    assert rig.events[-1] == ("stop", starts[-1][1])
    assert nothing_written(adapter, tmp_db)


def test_an_adapter_replaced_during_the_run_is_refused(adapter, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)  # it would adopt: the snapshot must stop the write
    rig = Rig()
    real_run = rig.run

    def run(url, online, repeat):
        if rig.runs == 1:  # the candidate's run: a retrain lands mid-measurement
            adapter.write_bytes(b"GGUF a different, longer adapter")
        return real_run(url, online, repeat)

    rig.run = run
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)]) == 2
    out = capsys.readouterr().out
    assert f"The adapter at {adapter} changed during the run. Nothing changed." in out
    assert "DECISION" not in out and rig.running == []
    assert nothing_written(adapter, tmp_db)


def invented_row(n: int, i: int) -> dict:
    return row("small", n, 60 + i, "You told me you love sushi.", "hey")  # nothing known about sushi: invented


def test_base_vs_base_on_noisy_samples_never_adopts(adapter, tmp_db, capsys):
    """Two samples of the same model: invented 2 -> 1 and a generic wobble. The real decide must keep the base."""
    first = transcript(generic_in_pass1=True) + [invented_row(1, 0), invented_row(2, 0)]
    second = transcript() + [invented_row(3, 0)]
    a, b = scored_of(first), scored_of(second)
    assert (a.pooled["invented"], b.pooled["invented"]) == (2, 1)
    assert a.pooled["generic_share"] != b.pooled["generic_share"]
    assert not champion.decide(a, b).adopt and not champion.decide(b, a).adopt
    rig = Rig(transcripts=[first, second])
    assert go(rig, ["--base-vs-base", "--db", str(tmp_db)]) == 0
    out = capsys.readouterr().out
    assert "DECISION: keep the base model" in out and "Dry run: nothing written." in out
    assert nothing_written(adapter, tmp_db)


# ---- final fix wave C1: a candidate path, an atomic install on adopt, nothing touched on reject ----

@pytest.fixture
def adopted(tmp_path):
    """The adapter the gate adopted earlier: the file at the adopted path and its matching record."""
    p = brain.default_adapter()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"GGUF the adopted adapter")
    brain.write_record(p, {"decision": "adopt", "targets_base": 8, "targets_candidate": 9})
    return p


@pytest.fixture
def candidate(tmp_path):
    p = brain.candidate_adapter()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"GGUF a retrained candidate, never measured")
    return p


def test_the_candidate_path_is_the_default_and_never_the_adopted_one():
    assert brain.candidate_adapter() == brain.default_adapter().with_name("riya-v1-candidate-lora.gguf")
    assert brain.CANDIDATE_NAME == "riya-v1-candidate-lora.gguf" != brain.ADAPTER_NAME


def test_the_gate_measures_the_candidate_by_default(adopted, candidate, tmp_db, capsys):
    rig = Rig()
    assert go(rig, ["--dry-run", "--db", str(tmp_db)]) == 0
    assert rig.events == [("start", None), ("stop", None), ("start", str(candidate)), ("stop", str(candidate))]


def test_adopt_installs_the_candidate_atomically(adopted, candidate, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)
    new_bytes = candidate.read_bytes()
    replaced = []
    real_replace = champion.os.replace
    monkeypatch.setattr(champion.os, "replace", lambda a, b: replaced.append((str(a), str(b))) or real_replace(a, b))
    assert go(Rig(), ["--db", str(tmp_db)]) == 0
    assert adopted.read_bytes() == new_bytes and not candidate.exists()  # moved into place in one rename
    assert replaced[0] == (str(candidate), str(adopted))  # then the record's own tmp -> .champion.json
    assert [b for _, b in replaced[1:]] == [str(brain.record_path(adopted))]
    rec = brain.read_record(adopted)  # the record belongs to the installed file
    assert rec is not None and rec["targets_candidate"] == 10 and rec["adapter"] == str(adopted)
    assert brain.read_local_adapter(tmp_db) == str(adopted)  # the setting names the adopted path, not the candidate
    assert brain.vetted_adapter(str(adopted)) == str(adopted)
    assert not brain.record_path(candidate).exists()
    assert "Adopted" in capsys.readouterr().out


def test_a_rejected_candidate_never_becomes_served(adopted, candidate, tmp_db, capsys, tmp_path):
    before = (adopted.read_bytes(), brain.record_path(adopted).read_text(encoding="utf-8"))
    conn = open_db(tmp_db)
    save_settings(KV(conn), Settings(local_adapter=str(adopted)))
    conn.close()
    rig = Rig()  # the real decide: the same transcripts for both, so "keep the base model"
    assert go(rig, ["--db", str(tmp_db)]) == 0
    assert "DECISION: keep the base model" in capsys.readouterr().out
    assert (adopted.read_bytes(), brain.record_path(adopted).read_text(encoding="utf-8")) == before
    assert candidate.exists() and not brain.record_path(candidate).exists()
    assert brain.read_local_adapter(tmp_db) == str(adopted)
    # what Riya's own server would serve: the adopted file, and never the rejected candidate
    from riya.server import RiyaServer
    for lora, served in ((str(adopted), str(adopted)), (str(candidate), None)):
        cmds = []
        srv = RiyaServer("m.gguf", slot_dir=str(tmp_path / "slots"), lora=lora, healthy=lambda port: True,
                         ready_timeout_s=0, popen=lambda cmd, **kw: cmds.append(cmd) or _Alive())
        srv.start()
        got = cmds[0][cmds[0].index("--lora") + 1] if "--lora" in cmds[0] else None
        assert got == served


class _Alive:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0


def test_an_install_that_fails_writes_nothing(adopted, candidate, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)
    before = (adopted.read_bytes(), brain.record_path(adopted).read_text(encoding="utf-8"))

    def locked(a, b):
        raise PermissionError(13, "the file is in use")

    monkeypatch.setattr(champion.os, "replace", locked)
    monkeypatch.setattr(champion.shutil, "copy2", lambda a, b: pytest.fail("same folder: a rename, never a copy"))
    assert go(Rig(), ["--db", str(tmp_db)]) == 2
    out = capsys.readouterr().out
    assert "Couldn't install" in out and "Nothing changed." in out
    assert (adopted.read_bytes(), brain.record_path(adopted).read_text(encoding="utf-8")) == before
    assert candidate.exists() and brain.read_local_adapter(tmp_db) is None


# ---- final fix wave minors: a relative --lora, no crash between the record and the setting ----

def test_a_relative_lora_is_resolved_first(adopted, candidate, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)
    monkeypatch.chdir(candidate.parent)
    rig = Rig()
    assert go(rig, ["--lora", candidate.name, "--db", str(tmp_db)]) == 0
    assert rig.made[1][1] == str(candidate.resolve())  # the gate's server got the absolute path
    assert brain.read_local_adapter(tmp_db) == str(adopted) and brain.read_record(adopted) is not None
    assert "Traceback" not in capsys.readouterr().out


def test_an_unexpected_failure_setting_it_is_reported_not_raised(adopted, candidate, tmp_db, capsys, monkeypatch):
    import sqlite3

    monkeypatch.setattr(champion, "decide", adopt_decision)

    def broken(*a, **k):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(champion, "set_local_adapter", broken)
    assert go(Rig(), ["--db", str(tmp_db)]) == 2  # never a traceback between the record and the setting
    out = capsys.readouterr().out
    assert "database is locked" in out and "Use Riya v1" in out
    assert brain.read_record(adopted) is not None


def test_a_setting_the_core_would_refuse_is_refused_before_anything_runs(tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(brain, "default_adapter", lambda: champion.Path("C:/x/" + "a" * 300 + ".gguf"))  # too long
    rig = Rig()
    assert go(rig, ["--lora", __file__, "--db", str(tmp_db)]) == 2
    out = capsys.readouterr().out
    assert rig.made == [] and "local_adapter" in out and "Nothing changed." in out



# ---- final fix wave minor: the template check on the user's own llama-server, before evaluating ----

def test_the_template_is_checked_on_its_own_server_before_evaluating(adapter, tmp_db, capsys):
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--db", str(tmp_db)]) == 0
    assert rig.templates == [None]  # once, on the base model's server, before its probes ran


@pytest.mark.parametrize("dry", [False, True])
def test_a_template_mismatch_refuses(adapter, tmp_db, capsys, dry):
    rig = Rig(template=lambda p: p.replace("<|im_end|>\n", "<|im_end|>"))
    argv = ["--lora", str(adapter), "--db", str(tmp_db)] + (["--dry-run"] if dry else [])
    assert go(rig, argv) == 2
    out = capsys.readouterr().out
    assert "chat template differs" in out and "Nothing changed." in out
    assert rig.events == [("start", None), ("stop", None)] and rig.runs == 0 and rig.running == []
    assert nothing_written(adapter, tmp_db)



# ---- final fix wave: carried T6 M2-M4 ----

def test_failed_turns_are_counted_from_the_transcript(candidate, capsys):
    t = transcript()
    assert scored_of(t).failed == 3  # one timed-out turn per pass
    rig = Rig(transcripts=[t, t + [row("small", 1, 99, "", error="timed out")]])
    assert go(rig, ["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Failed turns: base 3, Riya v1 4" in out and "more failed turns (4 vs 3)" in out


def test_a_busy_port_is_refused_before_starting(adapter, tmp_db, capsys):
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], port_busy=lambda port: port == champion.PORT) == 2
    out = capsys.readouterr().out
    assert rig.made == [] and "port 8091" in out and "already in use" in out and "Nothing changed." in out
    rig = Rig()  # a port that turns busy after the base run (an orphan, another program): refused before the candidate
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], port_busy=lambda port: bool(rig.events)) == 2
    assert rig.events == [("start", None), ("stop", None)] and nothing_written(adapter, tmp_db)
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db), "--port", "8090"],
              port_busy=lambda port: port == 8090) == 2  # Riya's own live server is never measured as the gate's
    assert rig.made == []


def test_a_record_that_cant_be_written_after_the_install_is_reported(adopted, candidate, tmp_db, capsys, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)

    def full_disk(adapter, record):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(brain, "write_record", full_disk)
    assert go(Rig(), ["--db", str(tmp_db)]) == 2
    out = capsys.readouterr().out
    assert "couldn't be written" in out and "base model" in out and "Nothing changed." not in out
    assert brain.read_local_adapter(tmp_db) is None  # no setting without a record
    assert brain.vetted_adapter(str(adopted)) is None  # the installed file isn't served: its record doesn't match


# ---- v0.10: a candidate run needs a passing calibration for this code and this base model ----

def test_a_candidate_run_needs_a_matching_calibration(adapter, tmp_db, capsys, monkeypatch, calibrated):
    gate.CALIBRATION_PATH.unlink()
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--db", str(tmp_db)]) == 2
    assert rig.made == []
    assert "There's no gate calibration record. Nothing changed." in capsys.readouterr().out
    rig = Rig()
    assert go(rig, ["--base-vs-base", "--db", str(tmp_db)]) == 0  # how the bank is gathered: no record needed
    assert len(rig.made) == 2
    capsys.readouterr()

    write_calibration(gate.CALIBRATION_PATH, base_sha256="0" * 64)  # calibrated on another base GGUF
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--db", str(tmp_db)]) == 2
    assert rig.made == []
    assert "The gate calibration is for base model 000000000000, this run uses" in capsys.readouterr().out

    write_calibration(gate.CALIBRATION_PATH, repeat=4)
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--db", str(tmp_db)]) == 2
    assert rig.made == [] and "is for repeat 4, this code has 3. Nothing changed." in capsys.readouterr().out

    write_calibration(gate.CALIBRATION_PATH)  # the base is hashed once, then the record is checked, then servers
    order, real_ident, real_problem = [], champion.file_identity, gate.calibration_problem
    monkeypatch.setattr(champion, "file_identity", lambda p: order.append(("hash", str(p))) or real_ident(p))
    monkeypatch.setattr(gate, "calibration_problem",
                        lambda *a, **kw: order.append(("check", kw["base_sha256"])) or real_problem(*a, **kw))
    rig = Rig()
    real_make = rig.make_server
    rig.make_server = lambda *a, **kw: order.append(("server",)) or real_make(*a, **kw)
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--db", str(tmp_db)]) == 0
    # the model once, before any server; the adapter before the base run and again after the candidate run
    assert order == [("hash", str(calibrated)), ("check", gate_helpers.BASE_SHA256), ("hash", str(adapter)),
                     ("server",), ("server",), ("hash", str(adapter))]


def test_an_unreadable_base_model_is_refused(adapter, tmp_db, capsys, tmp_path):
    missing = tmp_path / "no-such-base.gguf"
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--model", str(missing), "--db", str(tmp_db)]) == 2
    out = capsys.readouterr().out
    assert rig.made == [] and "Traceback" not in out
    assert f"Couldn't read the base model {missing}: " in out and out.rstrip().endswith("Nothing changed.")


def test_the_calibration_is_checked_at_the_run_s_own_repeat(adapter, tmp_db, capsys):
    """M3: a --repeat 2 run under the default repeat-3 record is refused: it would decide on 2 passes under a 3-pass
    calibration."""
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--repeat", "2", "--db", str(tmp_db)]) == 2
    assert rig.made == []
    assert "The gate calibration is for repeat 3, this code has 2. Nothing changed." in capsys.readouterr().out


# ---- v0.10: --verdict-out (Colab's checked verdict), and the adapter's sha256 held across the run ----

def sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Clock:
    """A fake monotonic clock: each call returns the next scripted reading."""

    def __init__(self, *times):
        self.times = list(times)

    def __call__(self) -> float:
        return self.times.pop(0)


def test_verdict_out_writes_a_checked_verdict_and_nothing_else(adapter, tmp_db, tmp_path, capsys, calibrated):
    out = tmp_path / "drive" / "verdict.json"
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--verdict-out", str(out), "--model", str(calibrated), "--db",
                    str(tmp_db)], version=lambda: "b1234", clock=Clock(100.0, 160.0, 250.0, 251.5)) == 0
    v = verdict.load_verdict(out)
    assert v["mode"] == "candidate"
    assert {k: v[k] for k in gate.IDENTITY_FIELDS} == gate.identity()
    assert v["adapter"] == {"name": adapter.name, "size": adapter.stat().st_size, "sha256": sha(adapter)}
    assert v["base_gguf"] == verdict.file_identity(calibrated) == {
        "name": "base.gguf", "size": len(gate_helpers.BASE_BYTES), "sha256": gate_helpers.BASE_SHA256}
    assert (v["llama_server"], v["template_check"]) == ("b1234", "ok")
    assert (v["alpha"], v["B"], v["seed"]) == (gate.ALPHA, gate.B, gate.GATE_SEED)
    for side in ("base", "candidate"):
        assert v["transcripts"][side] and all("pass" in r for r in v["transcripts"][side])
    assert v["transcripts"]["base"] == transcript() and v["failed_turns"] == {"base": 3, "candidate": 3}
    assert v["decision"]["adopt"] is False and len(v["decision"]["weak"]) == 4
    assert v["timings"] == {"base_s": 60.0, "candidate_s": 90.0, "total_s": 151.5}  # from the fake clock
    assert nothing_written(adapter, tmp_db)
    text = capsys.readouterr().out
    assert "Dry run: nothing written." in text and f"Verdict written: {out} (total 152 s)" in text
    assert rig.events == [("start", None), ("stop", None), ("start", str(adapter)), ("stop", str(adapter))]


def test_verdict_out_needs_a_calibration_in_candidate_mode(adapter, tmp_db, tmp_path, capsys, calibrated):
    gate.CALIBRATION_PATH.unlink()
    out, rig = tmp_path / "v.json", Rig()
    assert go(rig, ["--lora", str(adapter), "--verdict-out", str(out), "--model", str(calibrated), "--db",
                    str(tmp_db)]) == 2
    assert rig.made == [] and not out.exists()
    assert "There's no gate calibration record. Nothing changed." in capsys.readouterr().out


def test_verdict_out_base_vs_base_needs_no_calibration(adapter, tmp_db, tmp_path, capsys, calibrated):
    gate.CALIBRATION_PATH.unlink()
    out, t = tmp_path / "bank.json", transcript()
    rig = Rig(transcripts=[t, t])
    assert go(rig, ["--base-vs-base", "--verdict-out", str(out), "--model", str(calibrated), "--db", str(tmp_db)],
              version=lambda: "b1") == 0
    v = verdict.load_verdict(out)
    assert v["mode"] == "base-vs-base" and v["adapter"] is None
    assert v["base_gguf"] == verdict.file_identity(calibrated)
    assert rig.events == [("start", None), ("stop", None), ("start", None), ("stop", None)]
    assert nothing_written(adapter, tmp_db) and "Verdict written: " in capsys.readouterr().out


def test_a_bank_run_refuses_an_unreadable_base_model(adapter, tmp_db, tmp_path, capsys):
    out, missing, rig = tmp_path / "bank.json", tmp_path / "no-such-base.gguf", Rig()
    assert go(rig, ["--base-vs-base", "--verdict-out", str(out), "--model", str(missing), "--db", str(tmp_db)]) == 2
    text = capsys.readouterr().out
    assert rig.made == [] and not out.exists() and "Traceback" not in text
    assert f"Couldn't read the base model {missing}: " in text and text.rstrip().endswith("Nothing changed.")


def test_an_adapter_rewritten_to_the_same_size_is_refused(adapter, tmp_db, tmp_path, capsys, calibrated, monkeypatch):
    monkeypatch.setattr(champion, "decide", adopt_decision)  # it would adopt: the sha256 must stop the write
    st = adapter.stat()

    def rig_that_rewrites() -> Rig:
        rig = Rig()
        real_run = rig.run

        def run(url, online, repeat):
            if rig.runs == 1:  # the candidate's run: the same length, the original mtime put back
                adapter.write_bytes(b"GGUF FAKE ADAPTER")
                os.utime(adapter, ns=(st.st_atime_ns, st.st_mtime_ns))
            return real_run(url, online, repeat)

        rig.run = run
        return rig

    out, rig = tmp_path / "v.json", rig_that_rewrites()
    assert go(rig, ["--lora", str(adapter), "--verdict-out", str(out), "--model", str(calibrated), "--db",
                    str(tmp_db)]) == 2
    text = capsys.readouterr().out
    assert f"The adapter at {adapter} changed during the run. Nothing changed." in text and "DECISION" not in text
    assert (adapter.stat().st_size, adapter.stat().st_mtime_ns) == (st.st_size, st.st_mtime_ns)  # the snapshot is blind
    assert not out.exists() and rig.running == [] and nothing_written(adapter, tmp_db)

    adapter.write_bytes(b"GGUF fake adapter")  # the same holds for an adopting run without --verdict-out
    os.utime(adapter, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert go(rig_that_rewrites(), ["--lora", str(adapter), "--db", str(tmp_db)]) == 2
    assert "changed during the run" in capsys.readouterr().out and nothing_written(adapter, tmp_db)


def test_llama_server_version():
    def fake(out="", err=""):
        seen = []

        def run(cmd, **kw):
            seen.append((cmd, kw))
            return SimpleNamespace(stdout=out, stderr=err)

        return run, seen

    run, seen = fake(out="version: 4567 (abc)\nbuilt with gcc\n")
    assert champion.llama_server_version(run) == "version: 4567 (abc)"
    assert seen[0][0][-1] == "--version" and seen[0][1]["timeout"] == 10
    assert champion.llama_server_version(fake(out="\n  \n", err="\nversion: 9 (z)\n")[0]) == "version: 9 (z)"
    assert champion.llama_server_version(fake()[0]) == "unknown"

    def raising(cmd, **kw):
        raise FileNotFoundError(cmd[0])

    def hangs(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 10)

    assert champion.llama_server_version(raising) == "unknown" == champion.llama_server_version(hangs)


@pytest.mark.parametrize("empty", ["", " ", "\t "])
def test_an_empty_verdict_out_is_a_usage_error_never_a_live_run(adapter, tmp_db, capsys, monkeypatch, empty):
    monkeypatch.setattr(champion, "decide", adopt_decision)  # a live run would adopt
    rig = Rig()
    with pytest.raises(SystemExit) as e:
        go(rig, ["--lora", str(adapter), "--verdict-out", empty, "--db", str(tmp_db)])
    assert e.value.code == 2 and "--verdict-out needs a path" in capsys.readouterr().err
    assert rig.made == [] and nothing_written(adapter, tmp_db)


# ---- v0.10 C1: which mode flags combine ----

def test_the_mode_flags_combine_as_designed(adapter, tmp_db, tmp_path, capsys, calibrated):
    bank = tmp_path / "bank.json"
    rig = Rig()
    assert go(rig, ["--base-vs-base", "--verdict-out", str(bank), "--model", str(calibrated), "--db", str(tmp_db)],
              version=lambda: "b1") == 0  # how the bank is gathered
    assert bank.exists() and len(rig.made) == 2
    capsys.readouterr()

    v, out = tmp_path / "verdict.json", tmp_path / "out.json"
    made_cpu = []

    def make_cpu_server(*a, **kw):
        made_cpu.append(a)
        raise AssertionError("refused before anything starts")

    combos = [(["--adopt-verdict", str(v), "--check-verdict", str(v)], "--check-verdict")]
    for mode in ("--adopt-verdict", "--check-verdict"):
        combos += [([mode, str(v), "--verdict-out", str(out)], "--verdict-out"),
                   ([mode, str(v), "--base-vs-base"], "--base-vs-base"), ([mode, str(v), "--dry-run"], "--dry-run"),
                   ([mode, str(v), "--port", "8097"], "--port")]  # I1: the CPU check is fixed at 8091
    for args, named in combos:
        rig = Rig()
        with pytest.raises(SystemExit) as e:
            go(rig, args + ["--model", str(calibrated), "--db", str(tmp_db)], make_cpu_server=make_cpu_server)
        assert e.value.code == 2 and named in capsys.readouterr().err, args
        assert rig.made == [] and made_cpu == [] and not out.exists()
        assert nothing_written(adapter, tmp_db)


def test_the_port_is_refused_before_gaming_as_in_v0_8(adapter, tmp_db, capsys):
    """Both refusals hold at once: _check_ready names the busy port first (v0.8's order), and starts nothing."""
    rig = Rig()
    assert go(rig, ["--lora", str(adapter), "--db", str(tmp_db)], fullscreen=lambda: True,
              port_busy=lambda port: True) == 2
    out = capsys.readouterr().out
    assert rig.made == [] and "port 8091" in out and "gaming" not in out


def test_port_still_reaches_the_gate_s_server_in_a_run_mode(adapter, tmp_db, capsys, monkeypatch):
    seen = []
    rig = Rig()
    monkeypatch.setattr(champion, "PORT", 8097)  # the Rig's fakes check the URL against champion.PORT
    real = rig.make_server

    def make_server(model, *, port, **kw):
        seen.append(port)
        return real(model, port=port, **kw)

    rig.make_server = make_server
    assert go(rig, ["--lora", str(adapter), "--dry-run", "--port", "8097", "--db", str(tmp_db)]) == 0
    assert seen == [8097, 8097]
