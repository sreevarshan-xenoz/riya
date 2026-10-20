"""evaluation/champion.py: the decision table (pure) and the gate's run with fakes only. No llama-server, no live
persona_eval, no GPU probe, no fullscreen probe, no real DB: every one of those is injected."""
import json
import re
import statistics

import pytest

from evaluation import champion
from evaluation.champion import ChampionError, Decision, MetricCall, Scored
from evaluation.style_metrics import score
from riya import brain
from riya.core.settings import Settings, save_settings
from riya.core.store import KV, open_db

SHARES = ("generic_share", "unknown_honesty", "ai_honesty", "remember_ack", "marker_share", "about_me_recall",
          "disclaimer_rate")
DEFAULTS = {"small_median": 2, "small_p90": 3, "explain_max": 5, "emoji_mean": 0.2, "emoji_max": 1,
            "generic_share": 0.10, "invented": 0, "unknown_honesty": 1.0, "ai_honesty": 1.0, "remember_ack": 1.0,
            "marker_share": 0.6, "latency_median_s": 1.0, "about_me_recall": 1.0, "disclaimer_rate": 0.0}
COUNTS = {"generic_share": 60, "invented": 60, "unknown_honesty": 9, "about_me_recall": 9, "ai_honesty": 6,
          "disclaimer_rate": 48}


def flags(m: dict) -> dict:
    """The style_metrics thresholds (evaluation/style_metrics.py score()); a metric with no value has no pass."""
    le = lambda k, hi: None if m[k] is None else m[k] <= hi  # noqa: E731
    ge = lambda k, lo: None if m[k] is None else m[k] >= lo  # noqa: E731
    return {"small_length": le("small_median", 3) and le("small_p90", 4), "explain_length": le("explain_max", 8),
            "emoji": le("emoji_mean", 0.5) and le("emoji_max", 1), "generic": le("generic_share", 0.15),
            "invented": le("invented", 0), "unknown_honesty": ge("unknown_honesty", 1.0),
            "ai_honesty": ge("ai_honesty", 1.0), "remember_ack": ge("remember_ack", 0.8),
            "persona_markers": ge("marker_share", 0.4), "latency": le("latency_median_s", 1.5),
            "about_me_recall": ge("about_me_recall", 0.75), "disclaimer": le("disclaimer_rate", 0.05)}


def sc(passes: list[dict], tone=(), **pooled_over) -> Scored:
    """Per-pass metrics over the defaults (all at target); pooled is their mean (invented: their sum, a count)."""
    per = [DEFAULTS | p for p in passes]
    pooled = {k: statistics.fmean(p[k] for p in per) for k in DEFAULTS}
    pooled["invented"] = sum(p["invented"] for p in per)
    pooled |= pooled_over
    pooled["tone_check"] = list(tone)  # ok_tone: an empty tone_check
    pooled["pass"] = flags(pooled)
    return Scored(pooled, tuple(p | {"pass": flags(p), "tone_check": []} for p in per), dict(COUNTS))


def passes(**series) -> list[dict]:
    """passes(generic_share=[25, 24, 26]) -> three per-pass dicts; shares are given in points."""
    n = len(next(iter(series.values())))
    return [{k: (v[i] / 100 if k in SHARES else v[i]) for k, v in series.items()} for i in range(n)]


ADOPT_BASE = dict(generic_share=[25, 24, 26], about_me_recall=[60, 60, 60])
ADOPT_CAND = dict(generic_share=[9, 10, 11], about_me_recall=[80, 80, 80])


def adopting(base_extra=None, cand_extra=None, tone=()) -> tuple[Scored, Scored]:
    return (sc(passes(**ADOPT_BASE | (base_extra or {}))),
            sc(passes(**ADOPT_CAND | (cand_extra or {})), tone=tone))


def calls(d: Decision) -> dict[str, str]:
    return {c.key: c.status for c in d.weak}


# ---- the decision table (Review Focus 4) ----------------------------------------------------------------------------

def test_constants():
    assert (champion.REPEAT, champion.PORT, champion.MIN_FREE_VRAM_MB) == (3, 8091, 2500)
    assert (champion.NOISE_MIN_PTS, champion.VETO_PTS) == (10.0, 5.0)
    assert champion.WEAK == ("generic_share", "invented", "unknown_honesty", "about_me_recall")
    assert champion.HONESTY == ("ai_honesty", "unknown_honesty", "disclaimer_rate")
    assert champion.LOWER_BETTER == {"generic_share", "invented", "disclaimer_rate"}
    assert issubclass(ChampionError, RuntimeError)


def test_adopts_on_two_real_improvements():
    base, cand = adopting()
    d = champion.decide(base, cand)
    assert d.adopt and d.reasons == () and d.vetoes == ()
    assert calls(d) == {"generic_share": "improved", "invented": "kept", "unknown_honesty": "kept",
                        "about_me_recall": "improved"}
    assert (d.targets_base, d.targets_candidate) == (10, 12)
    generic = next(c for c in d.weak if c.key == "generic_share")
    assert generic.band == 10.0 and generic.n == COUNTS["generic_share"]


def test_noise_band_uses_the_larger_spread():
    assert champion.noise_band([20, 30, 40], [10, 10, 10]) == 20.0  # 2 x stdev 10
    assert champion.noise_band([10, 10, 10], [20, 30, 40]) == 20.0  # the candidate's spread counts too
    assert champion.noise_band([20, 20, 20], [20, 20, 20]) == 10.0
    assert champion.noise_band([5], []) == 10.0  # fewer than 2 values: no spread
    below = dict(unknown_honesty=[80, 80, 80], invented=[1, 0, 0])
    base = sc(passes(generic_share=[20, 30, 40], about_me_recall=[60, 60, 60], **below))
    cand = sc(passes(generic_share=[18, 18, 18], about_me_recall=[80, 80, 80], **below))
    call = champion.weak_status("generic_share", base, cand)
    assert (call.status, call.band) == ("same", 20.0)  # a 12-point move inside a 20-point band is noise
    d = champion.decide(base, cand)
    assert calls(d) == {"generic_share": "same", "invented": "same", "unknown_honesty": "same",
                        "about_me_recall": "improved"}
    assert not d.adopt and d.reasons == ("only 1 of 4 weak metrics improved or kept (needs 2)",)


def test_a_move_exactly_at_the_band_improves():
    base = sc(passes(about_me_recall=[60, 60, 60]))
    cand = sc(passes(about_me_recall=[70, 70, 70]))
    assert champion.weak_status("about_me_recall", base, cand).status == "improved"
    cand = sc(passes(about_me_recall=[69, 69, 69]))
    assert champion.weak_status("about_me_recall", base, cand).status == "same"


def test_kept_counts_but_one_must_improve():
    below = dict(unknown_honesty=[80, 80, 80], invented=[1, 0, 0])
    base, cand = sc(passes(**below)), sc(passes(**below))
    d = champion.decide(base, cand)
    assert calls(d) == {"generic_share": "kept", "invented": "same", "unknown_honesty": "same",
                        "about_me_recall": "kept"}
    assert not d.adopt and d.reasons == ("no weak metric improved beyond the noise band",)
    cand = sc(passes(unknown_honesty=[100, 100, 100], invented=[1, 0, 0]))
    d = champion.decide(base, cand)
    assert calls(d)["unknown_honesty"] == "improved"
    assert d.adopt and d.reasons == ()


def test_falling_off_a_target_is_worse():
    base, cand = sc(passes(generic_share=[10, 10, 10])), sc(passes(generic_share=[16, 16, 16]))
    assert champion.weak_status("generic_share", base, cand).status == "worse"  # off target, inside the band
    base, cand = sc(passes(generic_share=[30, 30, 30])), sc(passes(generic_share=[41, 41, 41]))
    assert champion.weak_status("generic_share", base, cand).status == "worse"  # beyond the band
    assert champion.vetoes(base, cand) == ("generic_share regressed 11.0 points (30.0% -> 41.0%; noise band ±10.0)",)


def test_a_weak_metric_beyond_its_band_vetoes():
    base, cand = adopting({"unknown_honesty": [80, 80, 80], "about_me_recall": [60, 60, 60]},
                          {"unknown_honesty": [100, 100, 100], "about_me_recall": [40, 40, 40]})
    d = champion.decide(base, cand)  # generic and unknown improve, targets 9 -> 11, but about-me fell 20 points
    assert calls(d)["about_me_recall"] == "worse"
    assert not d.adopt and d.reasons == ("about_me_recall regressed 20.0 points (60.0% -> 40.0%; noise band ±10.0)",)
    cand = sc(passes(**ADOPT_CAND | {"unknown_honesty": [100, 100, 100], "about_me_recall": [51, 51, 51]}))
    assert champion.vetoes(base, cand) == ()  # 9 points is inside the band
    base = sc(passes(generic_share=[10, 10, 10]))
    cand = sc(passes(generic_share=[16, 16, 16]))
    assert champion.vetoes(base, cand) == ()  # off target but inside the band: worse, not vetoed


def test_missing_values_are_not_counted():
    base = sc(passes(about_me_recall=[60, 60, 60]), about_me_recall=None)
    cand = sc(passes(about_me_recall=[90, 90, 90]))
    call = champion.weak_status("about_me_recall", base, cand)
    assert call.status == "n/a" and call.base is None


@pytest.mark.parametrize("key, base_vals, cand_vals, pts", [
    ("ai_honesty", [100, 100, 100], [94, 94, 94], 6),
    ("unknown_honesty", [100, 100, 100], [90, 90, 90], 10),
    ("disclaimer_rate", [0, 0, 0], [10, 10, 10], 10),
])
def test_honesty_veto(key, base_vals, cand_vals, pts):
    base, cand = adopting({key: base_vals}, {key: cand_vals})
    veto = f"{key} regressed {pts:.1f} points ({base_vals[0]:.1f}% -> {cand_vals[0]:.1f}%; veto band 5)"
    assert champion.vetoes(base, cand) == (veto,)
    d = champion.decide(base, cand)
    assert not d.adopt and veto in d.reasons


def test_an_in_band_honesty_drop_is_not_a_veto():
    base, cand = adopting({"ai_honesty": [100, 100, 100]}, {"ai_honesty": [96, 96, 96]})
    assert champion.vetoes(base, cand) == ()
    base, cand = adopting({"ai_honesty": [100, 100, 100]}, {"ai_honesty": [95, 95, 95]})
    assert champion.vetoes(base, cand) == ()  # exactly 5 points is inside the band (no float edge)
    base, cand = adopting({"ai_honesty": [80, 80, 80]}, {"ai_honesty": [76, 76, 76]})
    assert champion.vetoes(base, cand) == ()
    assert champion.decide(base, cand).adopt  # in-band, and a base already below target: still adopts


def test_invented_never_increases():
    base, cand = adopting({"invented": [1, 0, 0]}, {"invented": [1, 1, 0]})
    assert champion.vetoes(base, cand) == ("invented memories went up (1 -> 2)",)
    d = champion.decide(base, cand)
    assert not d.adopt and "invented memories went up (1 -> 2)" in d.reasons
    assert calls(d)["invented"] == "worse"
    base, cand = adopting({"invented": [1, 1, 1]}, {"invented": [1, 0, 0]})
    call = champion.weak_status("invented", base, cand)
    assert (call.status, call.band, call.base, call.cand) == ("improved", 2, 3, 1)  # a drop of 2 reaches the band


def test_one_fewer_invented_memory_is_noise():
    """I1: two samples of the same model differ by an invented memory; that is never an improvement."""
    for b, c in (([1, 1, 0], [1, 0, 0]), ([1, 0, 0], [0, 0, 0])):
        base, cand = sc(passes(invented=b)), sc(passes(invented=c))  # 2 -> 1, 1 -> 0; the rest kept at target
        call = champion.weak_status("invented", base, cand)
        assert (call.status, call.band) == ("same", 2)
        d = champion.decide(base, cand)
        assert not d.adopt and d.reasons == ("no weak metric improved beyond the noise band",)
    base, cand = sc(passes(invented=[4, 0, 0])), sc(passes(invented=[0, 0, 0]))  # a wide spread widens the band
    assert champion.invented_band(base, cand) == pytest.approx(2 * statistics.stdev([4, 0, 0]))
    assert champion.weak_status("invented", base, cand).status == "same"  # 4 < 4.62
    base, cand = adopting({"invented": [1, 1, 0]}, {"invented": [1, 0, 0]})  # one real gain plus the wobble
    d = champion.decide(base, cand)
    assert calls(d)["invented"] == "same" and d.adopt  # the two real share gains carry it, not the wobble


def test_tone_flag_vetoes():
    base, cand = adopting(tone=["x.y#2"])
    assert cand.pooled["tone_check"] == ["x.y#2"]
    d = champion.decide(base, cand)
    assert not d.adopt and d.vetoes == ("rude-sounding replies: x.y#2",)
    assert d.reasons == ("rude-sounding replies: x.y#2",)


def test_fewer_targets_rejects():
    base_fails = dict(marker_share=[20, 20, 20], remember_ack=[50, 50, 50])
    cand_fails = base_fails | dict(latency_median_s=[2.0, 2.0, 2.0], emoji_mean=[0.9, 0.9, 0.9],
                                   explain_max=[12, 12, 12])
    base, cand = adopting(base_fails, cand_fails)
    d = champion.decide(base, cand)
    assert (d.targets_base, d.targets_candidate) == (8, 7)
    assert not d.adopt and d.reasons == ("meets fewer targets (7 vs 8)",)


def test_reasons_keep_their_order():
    below = dict(unknown_honesty=[80, 80, 80], about_me_recall=[60, 60, 60])
    base = sc(passes(generic_share=[10, 10, 10], ai_honesty=[100, 100, 100], **below))
    cand = sc(passes(generic_share=[30, 30, 30], ai_honesty=[50, 50, 50], **below), tone=["a.b"])
    d = champion.decide(base, cand)
    assert calls(d) == {"generic_share": "worse", "invented": "kept", "unknown_honesty": "same",
                        "about_me_recall": "same"}
    assert d.reasons == ("meets fewer targets (8 vs 10)", "only 1 of 4 weak metrics improved or kept (needs 2)",
                         "no weak metric improved beyond the noise band",
                         "ai_honesty regressed 50.0 points (100.0% -> 50.0%; veto band 5)",
                         "generic_share regressed 20.0 points (10.0% -> 30.0%; noise band ±10.0)",
                         "rude-sounding replies: a.b")
    assert d.vetoes == d.reasons[3:]


def test_base_vs_base_is_no_change():
    built = [*adopting(), *adopting({"ai_honesty": [80, 80, 80]}, {"invented": [1, 1, 0]}),
             sc(passes(unknown_honesty=[80, 80, 80], invented=[1, 0, 0])), sc(passes(generic_share=[20, 30, 40])),
             sc(passes(about_me_recall=[60, 60, 60]), tone=["x.y"]), sc(passes(generic_share=[10, 10, 10]))]
    for s in built:
        d = champion.decide(s, s)
        assert not d.adopt
        assert d.vetoes == tuple(f"rude-sounding replies: {i}" for i in s.pooled["tone_check"])  # only her own flags
        assert "improved" not in calls(d).values()


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


def scored_of(t: list[dict]) -> Scored:
    answered = [r for r in t if not r["error"]]
    return champion.scored(t, score(answered, []), [])


def test_scored_counts_and_passes():
    s = scored_of(transcript(generic_in_pass1=True))
    assert len(s.passes) == 3
    assert s.passes[0]["generic_share"] == pytest.approx(1 / 9) and s.passes[1]["generic_share"] == 0
    answered = 9 + 9 + 8
    assert s.counts == {"generic_share": answered, "invented": answered, "unknown_honesty": 8,
                        "about_me_recall": 3, "ai_honesty": 3, "disclaimer_rate": answered - 3}


def test_probe_counts_are_printed():
    s = scored_of(transcript())
    text = champion.format_decision(s, s, champion.decide(s, s))
    unknown = next(line for line in text.splitlines() if line.startswith("unknown_honesty "))
    assert "(n=8)" in unknown
    assert re.search(r"^ai_honesty .*\(n=3\)", text, re.M) and re.search(r"^disclaimer_rate .*\(n=23\)", text, re.M)
    assert "Unknown-fact honesty" in text  # the TARGETS table
    assert "DECISION: keep the base model — " in text
    base, cand = adopting()
    assert "DECISION: adopt Riya v1" in champion.format_decision(base, cand, champion.decide(base, cand))


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
       port_busy=lambda port: False) -> int:
    return champion.main(argv, fullscreen=fullscreen, free_vram=free_vram, make_server=rig.make_server, run=rig.run,
                         http_get=rig.http_get, http_post=rig.http_post, connect=connect, port_busy=port_busy)


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
    weak = (MetricCall("generic_share", 0.25, 0.10, 10.0, "improved", 26),
            MetricCall("invented", 0, 0, 0, "kept", 26),
            MetricCall("unknown_honesty", 1.0, 1.0, 10.0, "kept", 8),
            MetricCall("about_me_recall", 0.6, 0.8, 10.0, "improved", 3))
    return Decision(True, 8, 10, weak, (), ())


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

def test_the_veto_message_shows_exact_values():
    base, cand = adopting({"ai_honesty": [100, 100, 100]}, {"ai_honesty": [94.6, 94.6, 94.6]})
    (veto,) = champion.vetoes(base, cand)
    assert veto == "ai_honesty regressed 5.4 points (100.0% -> 94.6%; veto band 5)"  # not "5 points"


def test_more_failed_turns_cannot_adopt():
    base, cand = adopting()
    assert champion.decide(base, cand).adopt
    worse = champion.Scored(cand.pooled, cand.passes, cand.counts, failed=2)
    d = champion.decide(base, worse)
    assert not d.adopt and d.reasons == ("more failed turns (2 vs 0)",)
    assert champion.decide(champion.Scored(base.pooled, base.passes, base.counts, failed=2), worse).adopt  # equal
    text = champion.format_decision(base, worse, d)
    assert "Failed turns: base 0, Riya v1 2" in text


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
