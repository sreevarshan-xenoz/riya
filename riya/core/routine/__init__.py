"""v0.12 "Morpheus in Riya": four small predictors of things she can check, a ported MORPHEUS appraisal, and
statements about the user's routine that are engine-made claims or hunches, logged when said and audited later.

The Routine facade is what the engine calls. With the switch off every method is a no-op (None or []), so she is
exactly v0.11. Learning off freezes it: nothing is observed and every claim falls back to its hunch.
"""
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable

from riya.core.hands.reminders import fmt_clock
from riya.core.memory import Glance
from riya.core.routine import appraisal as ap
from riya.core.routine import audit as au
from riya.core.routine import detect
from riya.core.routine import predictors as pr
from riya.core.routine import statements as st
from riya.core.routine.appraisal import Appraisal
from riya.core.routine.records import RoutineDB
from riya.core.routine.statements import SaidRow, Statement

CLAIMS_ENABLED = True
ABSENCE_REMARKS_ENABLED = True
PLACEHOLDER = "[a remark about your routine]"

AUDIT_EVERY_S = 3600.0
MAX_ABSENCE_S = 30 * 86400.0
MAX_RUN_S = 7 * 86400.0
ANSWERED_CLAUSE_S = 1800.0
UNSURE_KEY = "routine.unsure"
S8_KEY = "routine.s8_last"
UNSURE_OK_C = 0.5
UNSURE_OK_N = 2
P2_KINDS = ("bid", "game_offer", "greeting")
AGENT_NAME_MAX = 20  # an agent's name in her line, so a statement turn's added text stays within 280 characters
_BANDS = {"night": (22, 7), "morning": (5, 7), "afternoon": (12, 5), "evening": (17, 5)}  # (start hour, hours)
_PRESENCE = {"came_back": "back", "went_away": "away", "break_started": "break", "break_ended": "break_end"}


@dataclass(frozen=True)
class Sensed:
    now: float
    presence: str
    stretch_start: float
    stretch_censored: bool
    glances: tuple[Glance, ...]
    clinginess: float
    quiet: bool
    short_greeting: bool = False
    news: bool = False
    runs: tuple = ()  # RunInfo


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


class Routine:
    def __init__(self, conn, lock, kv, clock, tz, twin, settings: Callable, *, started_at: float,
                 glances: Callable[[float, float], list] = lambda a, b: [], evidence: bool = True,
                 on_scored: Callable | None = None):
        self.db = RoutineDB(conn, lock)
        self.kv, self.clock, self.tz, self.twin, self.settings = kv, clock, tz, twin, settings
        self.started_at, self.glances, self.evidence, self.on_scored = started_at, glances, evidence, on_scored
        self._said: set[str] = set()  # texts said by this process: the online scrub hides them
        self._dispute: tuple[int, float] | None = None  # the last statement, until the user's next turn
        self._answered_at: float | None = None  # the last P2 "answered" with p_engage < 0.4
        self._last_audit: float | None = None
        self._rot: dict[tuple[str, str], int] = {}
        self._return: dict | None = None  # the last P1 observation, for the greeting
        self._runs: dict[str, dict] = {}  # agent id -> its last P3 observation, for the report
        self._s8: dict | None = None  # a change on a day cell that may be told
        self._band: dict | None = None  # the P4 tracker: the band in progress
        self._prev_state: str | None = None
        self._last_presence: float | None = None

    # ---- switches ----------------------------------------------------------------------------------------------------
    def enabled(self) -> bool:
        return bool(getattr(self.settings(), "routine_enabled", False))

    def frozen(self) -> bool:
        return not getattr(self.settings(), "learning_enabled", True)

    def _learning(self) -> bool:
        return self.enabled() and not self.frozen()

    def _claims(self) -> bool:
        return CLAIMS_ENABLED and not self.frozen()

    # ---- time helpers ------------------------------------------------------------------------------------------------
    def _local(self, t: float) -> datetime:
        return datetime.fromtimestamp(t, self.tz)

    def _band_of(self, t: float) -> tuple[str, str, float, float]:
        """(band, daytype, band start, band end) for t; the night band belongs to the date it started."""
        d = self._local(t)
        h = d.hour
        band = "night" if h >= 22 or h < 5 else "morning" if h < 12 else "afternoon" if h < 17 else "evening"
        day = d.date() - timedelta(days=1) if band == "night" and h < 5 else d.date()
        start_h, hours = _BANDS[band]
        start = datetime(day.year, day.month, day.day, start_h, tzinfo=d.tzinfo).timestamp() if d.tzinfo else \
            datetime(day.year, day.month, day.day, start_h).timestamp()
        daytype = "weekend" if day.weekday() >= 5 else "weekday"
        return band, daytype, start, start + hours * 3600.0

    def _clock(self, t: float) -> str:
        return fmt_clock(t, self.tz)

    # ---- unsure (Resolution 15) --------------------------------------------------------------------------------------
    def _unsure(self, cell: str, step: pr.Step, now: float) -> bool:
        """Updates the cell's unsure state; True only when this observation made it enter unsure."""
        state = self.db.kv_get(UNSURE_KEY, {})
        state = state if isinstance(state, dict) else {}
        entered = False
        if step.change:
            entered = cell not in state
            state[cell] = {"since": now, "ok": 0}
        elif cell in state:
            c = step.cell.state if step.cell is not None else None
            n = self._n_after(step)
            ok = c is not None and ap.confidence(n, c.cusum) >= UNSURE_OK_C
            entry = state[cell] if isinstance(state[cell], dict) else {"since": now, "ok": 0}
            entry["ok"] = int(entry.get("ok", 0)) + 1 if ok else 0
            if entry["ok"] >= UNSURE_OK_N:
                del state[cell]
            else:
                state[cell] = entry
        else:
            return False
        self.db.kv_set(UNSURE_KEY, state)
        return entered

    @staticmethod
    def _n_after(step: pr.Step) -> float:
        c = step.cell
        if isinstance(c, pr.LogNormalCell):
            return c.n
        if isinstance(c, pr.BetaCell):
            return max(0.0, c.alpha + c.beta - sum(pr.P4_PRIOR))
        return step.n_eff_before + 1

    def _p2_unsure(self, cell: str, step: pr.Step, n_after: float, state: ap.AppraisalState, now: float) -> bool:
        s = self.db.kv_get(UNSURE_KEY, {})
        s = s if isinstance(s, dict) else {}
        if step.change:
            entered = cell not in s
            s[cell] = {"since": now, "ok": 0}
        elif cell in s:
            entered = False
            ok = ap.confidence(n_after, state.cusum) >= UNSURE_OK_C
            entry = s[cell] if isinstance(s[cell], dict) else {"since": now, "ok": 0}
            entry["ok"] = int(entry.get("ok", 0)) + 1 if ok else 0
            if entry["ok"] >= UNSURE_OK_N:
                del s[cell]
            else:
                s[cell] = entry
        else:
            return False
        self.db.kv_set(UNSURE_KEY, s)
        return entered

    # ---- observations ------------------------------------------------------------------------------------------------
    def _lognormal(self, predictor: str, prior: pr.LogNormalPrior, fine: str, coarse: str, x: float, now: float,
                   topic: str) -> tuple[Appraisal, int, str, pr.Predictive, float]:
        """Updates the fine and the coarse cell; the appraisal comes from the fine cell once it has BACKOFF_N of
        evidence. Returns (appraisal, its obs id, its bucket, the predictive it stood on, its n_eff before)."""
        fine_n = pr.lognormal_decayed(self.db.get_lognormal(predictor, fine), prior, now).n
        use = fine if fine_n >= pr.BACKOFF_N else coarse
        chosen = None
        for bucket in dict.fromkeys((fine, coarse)):
            before = pr.lognormal_decayed(self.db.get_lognormal(predictor, bucket), prior, now)
            pred = pr.predictive(before, prior)
            step = self.db.update_cell(predictor, bucket,
                                       lambda c: (lambda s: (s.cell, s))(pr.observe_lognormal(c, prior, x, now)))
            obs_id = self.db.add_obs(now, predictor, bucket, x)
            entered = self._unsure(f"{predictor}:{bucket}", step, now)
            if step.change and step.n_eff_before >= st.S8_N and predictor == "P1":
                q = (pred.quantile_s(0.25), pred.quantile_s(0.75))
                self._s8 = {"predictor": predictor, "bucket": bucket, "n_before": step.n_eff_before,
                            "lo": q[0], "hi": q[1], "at": now}
            if bucket == use:
                a = ap.of(step, topic=topic, predictor=predictor, bucket=bucket, value=x,
                          direction=pr.direction_for(topic, step.sign))
                chosen = (Appraisal(**{**a.__dict__, "change": entered}), obs_id, bucket, pred, step.n_eff_before)
        return chosen

    def observe_return(self, away_s: float, now: float) -> Appraisal | None:
        if not self._learning() or not _finite(away_s) or not _finite(now):
            return None
        left = now - away_s
        if away_s < pr.MIN_ABSENCE_S or away_s > MAX_ABSENCE_S or left < self.started_at:
            self._return = None
            return None
        band, daytype, _, _ = self._band_of(left)
        a, obs_id, bucket, pred, n_before = self._lognormal("P1", pr.P1_PRIOR, f"{band}/{daytype}", band, away_s,
                                                            now, "return")
        self._return = {"a": a, "obs_id": obs_id, "bucket": bucket, "pred": pred, "at": now, "away_s": away_s}
        return a

    def observe_run(self, ev, bucket: str | None, now: float) -> Appraisal | None:
        if (not self._learning() or bucket is None or not _finite(now)
                or getattr(ev, "kind", None) not in ("finished", "failed")):
            return None
        run_s = getattr(ev, "run_s", None)
        if not _finite(run_s) or run_s < pr.MIN_RUN_S or run_s > MAX_RUN_S:
            return None
        kind = "colab" if bucket.startswith("colab:") else "claude"
        a, obs_id, use, pred, _ = self._lognormal("P3", pr.P3_PRIORS[kind], bucket, f"{kind}:*", run_s, now, "run")
        self._runs[getattr(ev, "agent_id", "")] = {"a": a, "obs_id": obs_id, "bucket": use, "run_s": run_s}
        return a

    def observe_engagement(self, kind: str, bucket: str, engaged: bool) -> Appraisal | None:
        """P2: call before the twin records, so p_engage is what she expected."""
        if not self._learning() or kind not in P2_KINDS:
            return None
        from riya.core.learn import twin as tw
        now = self.clock.now()
        p = self.twin.p_engage(kind, bucket)
        n_eff = tw.evidence(self.twin.store.get_cell(kind, bucket), now)
        state, step = pr.appraise_p2(self.db.get_p2(kind, bucket), p, engaged, n_eff)
        self.db.put_p2(kind, bucket, state, now)
        entered = self._p2_unsure(f"P2:{kind}:{bucket}", step, n_eff + 1, state, now)
        if engaged and p < 0.4:
            self._answered_at = now
        a = ap.of(step, topic="answer", predictor="P2", bucket=f"{kind}:{bucket}", value=p,
                  direction=pr.direction_for("answer", step.sign))
        return Appraisal(**{**a.__dict__, "change": entered})

    def expected_ignore(self, kind: str, bucket: str) -> bool:
        return self.twin.p_engage(kind, bucket) < 0.5

    def observe_presence(self, now: float, state: str, events: list) -> list[Appraisal]:
        if not self._learning() or not _finite(now):
            return []
        for e in events or ():
            ev = _PRESENCE.get(getattr(e, "kind", None))
            if ev:
                self.db.log_presence(now, ev)
        prev, self._prev_state = self._prev_state, state
        if state == "gaming" and prev != "gaming":
            self.db.log_presence(now, "gaming")
        elif prev == "gaming" and state != "gaming":
            self.db.log_presence(now, "gaming_end")
        last, self._last_presence = self._last_presence, now
        return self._track_band(now, state, None if last is None else now - last)

    def _track_band(self, now: float, state: str, dt: float | None) -> list[Appraisal]:
        band, daytype, start, end = self._band_of(now)
        out = []
        b = self._band
        if b is not None and b["start"] != start:  # the band ended: observe it if it was covered and not yet seen
            if not b["observed"] and b["covered"] >= pr.BAND_COVER * (b["end"] - b["start"]):
                out.append(self._observe_band(b, False, now))
            b = None
        if b is None:
            b = self._band = {"band": band, "daytype": daytype, "start": start, "end": end, "covered": 0.0,
                              "observed": False, "span": None}
        if dt is not None and 0 < dt <= st.STRETCH_GAP_S and state != "away":
            b["covered"] += dt
        if state == "gaming":
            b["span"] = now if b["span"] is None else b["span"]
            if not b["observed"] and now - b["span"] >= pr.GAMING_SPAN_S:
                out.append(self._observe_band(b, True, now))
        else:
            b["span"] = None
        return out

    def _observe_band(self, b: dict, gamed: bool, now: float) -> Appraisal:
        b["observed"] = True
        fine, coarse = f"{b['band']}/{b['daytype']}", b["band"]
        fine_n = pr.beta_evidence(pr.beta_decayed(self.db.get_beta("P4", fine), pr.P4_PRIOR, now), pr.P4_PRIOR)
        use = fine if fine_n >= pr.BACKOFF_N else coarse
        chosen = None
        for bucket in (fine, coarse):
            before = pr.beta_decayed(self.db.get_beta("P4", bucket), pr.P4_PRIOR, now)
            step = self.db.update_cell("P4", bucket,
                                       lambda c: (lambda s: (s.cell, s))(pr.observe_beta(c, pr.P4_PRIOR, gamed, now)))
            self.db.add_obs(now, "P4", bucket, 1.0 if gamed else 0.0)
            entered = self._unsure(f"P4:{bucket}", step, now)
            if step.change and step.n_eff_before >= st.S8_N:
                usual = 1.0 if before.alpha / (before.alpha + before.beta) >= 0.5 else 0.0
                self._s8 = {"predictor": "P4", "bucket": bucket, "n_before": step.n_eff_before, "lo": usual,
                            "hi": usual, "at": now}
            if bucket == use:
                a = ap.of(step, topic="gaming", predictor="P4", bucket=bucket, value=1.0 if gamed else 0.0,
                          direction=pr.direction_for("gaming", step.sign))
                chosen = Appraisal(**{**a.__dict__, "change": entered})
        return chosen

    # ---- statements --------------------------------------------------------------------------------------------------
    def _recent(self, now: float) -> list[SaidRow]:
        return [SaidRow(r.at, r.kind, r.where, r.text, r.kind == "S2" and r.value.get("direction") == "late")
                for r in self.db.statements_since(now - 7 * 86400.0)]

    def _render(self, kind: str, variant: str, slots: dict, now: float) -> str:
        """The next rotation whose text she hasn't said in the last REPEAT_S (or the next one when all were)."""
        n = len(st.TEMPLATES[(kind, variant)])
        start = self._rot.get((kind, variant), 0)
        recent = {r.text.casefold() for r in self.db.statements_since(now - st.REPEAT_S)}
        for i in range(n):
            text = st.render(kind, variant, slots, start + i)
            if text.casefold() not in recent:
                self._rot[(kind, variant)] = start + i + 1
                return text
        self._rot[(kind, variant)] = start + 1
        return st.render(kind, variant, slots, start)

    def _make(self, kind: str, claim: bool, variant: str, slots: dict, value: dict, evidence: dict, where: str,
              now: float, due_at: float | None = None) -> Statement:
        text = self._render(kind, variant, slots, now)
        return Statement(kind=kind, claim=claim or not self.evidence, text=text, value=value, evidence=evidence,
                         where=where, due_at=due_at)

    def _capped(self, s: Statement | None, now: float) -> Statement | None:
        if s is None or not st.voluntary_ok(s.kind, s.where, s.text, self._recent(now), now):
            return None
        return s

    def _level(self, rule: str | None) -> str | None:
        """A rule's verdict after the flags: claims off (or learning off) turns a claim into its hunch."""
        if rule == "claim" and not self._claims():
            return "hunch"
        return rule

    def greeting(self, a: Appraisal | None, away_s: float, sensed: Sensed) -> Statement | None:
        if not self.enabled() or sensed.news:
            return None
        now = sensed.now
        r = self._return
        if a is not None and a.topic == "return" and r is not None and r["a"] is a:
            level = self._level(st.rule_s2(a))
            late = a.direction == "late"
            ok = level is not None
            if ok and late:
                pred = r["pred"]
                ok = ABSENCE_REMARKS_ENABLED and st.absence_ok(
                    away_s, pred.median_s(), pred.quantile_s(0.9), sensed.clinginess, sensed.quiet,
                    sensed.short_greeting, self._recent(now), now)
            if ok:
                s = self._make("S2", level == "claim", f"{a.direction}_{level}", {},
                               {"predictor": "P1", "bucket": r["bucket"], "obs_id": r["obs_id"], "x": away_s,
                                "direction": a.direction},
                               {"n_eff": a.n_eff, "confidence": a.confidence, "surprise": a.surprise}, "greeting", now)
                s = self._capped(s, now)
                if s is not None:
                    return s
        return self._capped(self._s8_statement("greeting", now), now)

    def _s8_statement(self, where: str, now: float) -> Statement | None:
        c = self._s8
        if c is None or now - c["at"] > 7 * 86400.0:
            return None
        last = self.db.kv_get(S8_KEY, {})
        last = last if isinstance(last, dict) else {}
        cell = f"{c['predictor']}:{c['bucket']}"
        if not st.rule_s8(c["n_before"], last.get(cell), now):
            return None
        return self._make("S8", False, "hunch", {},
                          {"predictor": c["predictor"], "bucket": c["bucket"], "lo": c["lo"], "hi": c["hi"]},
                          {"n_before": c["n_before"]}, where, now)

    def _s1(self, sensed: Sensed, where: str) -> Statement | None:
        now = sensed.now
        level = self._level(st.rule_s1(now - sensed.stretch_start, sensed.stretch_censored))
        if level is None:
            return None
        if level == "hunch":
            return self._make("S1", False, "hunch", {}, {"t0": sensed.stretch_start, "coding": False},
                              {"censored": sensed.stretch_censored}, where, now)
        coding = st.coding_ok(sensed.glances, sensed.stretch_start, now)
        base = "claim_bid" if where == "bid" else "claim"
        return self._make("S1", True, base + ("_coding" if coding else ""), {"since": self._clock(sensed.stretch_start)},
                          {"t0": sensed.stretch_start, "coding": coding},
                          {"stretch_s": now - sensed.stretch_start}, where, now)

    def bid(self, sensed: Sensed, long_stretch: bool) -> Statement | None:
        if not self.enabled():
            return None
        now = sensed.now
        for make in ((lambda: self._s1(sensed, "bid")) if long_stretch else (lambda: None),
                     lambda: self._s8_statement("bid", now), lambda: self._s7_bid(now)):
            s = self._capped(make(), now)
            if s is not None:
                return s
        return None

    def _s7_bid(self, now: float) -> Statement | None:
        """A gaming hunch within an hour before a band she thinks they game in."""
        nxt = now + st.S7_BID_LEAD_S
        band, daytype, start, end = self._band_of(nxt)
        if not st.s7_bid_ok(now, start):
            return None
        bucket = f"{band}/{daytype}"
        c = pr.beta_decayed(self.db.get_beta("P4", bucket), pr.P4_PRIOR, now)
        n = pr.beta_evidence(c, pr.P4_PRIOR)
        mean = c.alpha / (c.alpha + c.beta)
        if st.rule_s7(mean, n, pr.beta_sf(c.alpha, c.beta), ap.confidence(n, c.state.cusum)) is None:
            return None
        return self._make("S7", False, "bid_hunch", {},
                          {"predictor": "P4", "bucket": bucket, "band_start": start, "band_end": end},
                          {"mean": mean, "n_eff": n}, "bid", now, due_at=end)

    def report(self, ev, a: Appraisal | None, sensed: Sensed) -> Statement | None:
        if not self.enabled() or a is None or a.topic != "run":
            return None
        r = self._runs.get(getattr(ev, "agent_id", None))
        if r is None or r["a"] is not a:
            return None
        level = self._level(st.rule_s2(a))
        if level is None:
            return None
        now = sensed.now
        s = self._make("S4", level == "claim", f"{a.direction}_{level}", {"took": st.fmt_span(r["run_s"])},
                       {"predictor": "P3", "bucket": r["bucket"], "obs_id": r["obs_id"], "x": r["run_s"],
                        "direction": a.direction},
                       {"n_eff": a.n_eff, "confidence": a.confidence, "surprise": a.surprise}, "report", now)
        return self._capped(s, now)

    def question(self, text: str) -> str | None:
        return detect.routine_question(text) if self.enabled() else None

    def _unknown(self, now: float) -> Statement:
        return Statement(kind="unknown", claim=False, text=self._render("unknown", "line", {}, now), value={},
                         evidence={}, where="reply", due_at=None)

    def _lognormal_cell(self, predictor: str, prior: pr.LogNormalPrior, fine: str, coarse: str,
                        now: float) -> tuple[str, pr.LogNormalCell]:
        f = pr.lognormal_decayed(self.db.get_lognormal(predictor, fine), prior, now)
        if f.n >= pr.BACKOFF_N or fine == coarse:
            return fine, f
        return coarse, pr.lognormal_decayed(self.db.get_lognormal(predictor, coarse), prior, now)

    def _named_run_bucket(self, text: str, now: float) -> str | None:
        """The P3 bucket the question names, else the one most recently updated."""
        cells = [(b, c) for _, b, c in self.db.cells("P3") if not b.endswith(":*")]
        words = set(detect._norm(text).replace(":", " ").split())
        for b, _ in cells:
            if b.split(":", 1)[1].casefold() in words:
                return b
        if not cells:
            return None
        return max(cells, key=lambda bc: bc[1].updated_at)[0]

    def answer(self, topic: str, text: str, sensed: Sensed) -> Statement | None:
        if not self.enabled():
            return None
        now = sensed.now
        s = None
        if topic == "stretch":
            s = self._s1(sensed, "reply")
        elif topic == "runs":
            s = self._s5(text, now)
        elif topic == "run_left":
            s = self._s6(text, sensed)
        elif topic == "gaming":
            s = self._s7(now)
        return s or self._unknown(now)

    def _s5(self, text: str, now: float) -> Statement | None:
        bucket = self._named_run_bucket(text, now)
        if bucket is None:
            return None
        kind = "colab" if bucket.startswith("colab:") else "claude"
        prior = pr.P3_PRIORS[kind]
        use, c = self._lognormal_cell("P3", prior, bucket, f"{kind}:*", now)
        pred = pr.predictive(c, prior)
        conf = ap.confidence(c.n, c.state.cusum)
        level = self._level(st.rule_s5(c.n, conf, pr.within_band(pred, 2.0)))
        if level is None:
            return None
        name = bucket.split(":", 1)[1][:AGENT_NAME_MAX]
        median, lo, hi = pred.median_s(), pred.quantile_s(0.1), pred.quantile_s(0.9)
        value = {"predictor": "P3", "bucket": use, "median_s": median, "lo_s": lo, "hi_s": hi}
        ev = {"n_eff": c.n, "confidence": conf}
        if level == "claim":
            return self._make("S5", True, "claim", {"agent": name, "median": st.fmt_span(median)}, value, ev,
                              "reply", now)
        bare = lambda x: st.fmt_span(x).removeprefix("about ")  # noqa: E731  ("anywhere from" already hedges)
        return self._make("S5", False, "hunch", {"agent": name, "lo": bare(lo), "hi": bare(hi)},
                          value, ev, "reply", now)

    def _s6(self, text: str, sensed: Sensed) -> Statement | None:
        now = sensed.now
        runs = list(sensed.runs)
        if not runs:
            return None
        words = set(detect._norm(text).split())
        named = [r for r in runs if r.bucket.split(":", 1)[-1].casefold() in words]
        run = (named or sorted(runs, key=lambda r: r.started_at))[-1]
        kind = "colab" if run.bucket.startswith("colab:") else "claude"
        prior = pr.P3_PRIORS[kind]
        use, c = self._lognormal_cell("P3", prior, run.bucket, f"{kind}:*", now)
        if not st.rule_s6(c.n):
            return None
        elapsed = max(0.0, now - run.started_at - run.waiting_s)
        g = pr.predictive(c, prior).median_s() - elapsed
        value = {"predictor": "P3", "bucket": use, "left_s": max(0.0, g)}
        if g <= 60:
            return self._make("S6", False, "hunch_soon", {}, value, {"n_eff": c.n}, "reply", now)
        return self._make("S6", False, "hunch", {"left": st.fmt_span(g)}, value, {"n_eff": c.n}, "reply", now)

    def _s7(self, now: float) -> Statement | None:
        best = None
        for _, bucket, c in self.db.cells("P4"):
            if "/" not in bucket:
                continue
            c = pr.beta_decayed(c, pr.P4_PRIOR, now)
            n = pr.beta_evidence(c, pr.P4_PRIOR)
            mean = c.alpha / (c.alpha + c.beta)
            level = self._level(st.rule_s7(mean, n, pr.beta_sf(c.alpha, c.beta), ap.confidence(n, c.state.cusum)))
            if level is not None and (best is None or (level == "claim", mean) > (best[0] == "claim", best[3])):
                best = (level, bucket, n, mean)
        if best is None:
            return None
        level, bucket, n, mean = best
        band, daytype = bucket.split("/")
        phrase = f"{daytype} {band}s"
        return self._make("S7", level == "claim", level, {"band": phrase},
                          {"predictor": "P4", "bucket": bucket}, {"n_eff": n, "mean": mean}, "reply", now)

    def farewell(self, text: str, sensed: Sensed) -> Statement | None:
        if not self.enabled() or not detect.goodbye(text):
            return None
        now = sensed.now
        band, daytype, _, _ = self._band_of(now)
        use, c = self._lognormal_cell("P1", pr.P1_PRIOR, f"{band}/{daytype}", band, now)
        if not st.rule_s3(c.n, ap.confidence(c.n, c.state.cusum)):
            return None
        back_at = st.round_back(now + pr.predictive(c, pr.P1_PRIOR).median_s(), self.tz)
        return self._make("S3", False, "hunch", {"back": self._clock(back_at)}, {"back_at": back_at, "left_at": now},
                          {"n_eff": c.n}, "goodbye", now, due_at=back_at)

    # ---- speaking ----------------------------------------------------------------------------------------------------
    def said(self, s: Statement, now: float) -> int | None:
        """Logs a statement she just said (none for "unknown"), feeds the tally and remembers it for the scrub."""
        if not self.enabled():
            return None
        self._said.add(s.text)
        if s.kind == "unknown":
            return None
        sid = self.db.add_statement(s, now)
        row = self.db.statement(sid)
        if row is not None:
            au.tally_said(self.db, row)
        self._dispute = (sid, now)
        if s.kind == "S8" and isinstance(s.value.get("predictor"), str):
            last = self.db.kv_get(S8_KEY, {})
            last = last if isinstance(last, dict) else {}
            last[f"{s.value['predictor']}:{s.value.get('bucket')}"] = now
            self.db.kv_set(S8_KEY, last)
            self._s8 = None
        return sid

    def clause(self, s: Statement, now: float) -> str | None:
        if s.kind == "S2":
            return st.CLAUSES.get(s.value.get("direction"))
        if s.kind == "S8":
            return st.CLAUSES["unsure"]
        if s.kind != "unknown" and self._answered_at is not None and now - self._answered_at <= ANSWERED_CLAUSE_S:
            return st.CLAUSES["answered"]
        return None

    def on_user_turn(self, text: str, now: float) -> None:
        """A dispute counts only on the user's first turn after a statement, within DISPUTE_S."""
        d, self._dispute = self._dispute, None
        if d is not None and self.enabled() and now - d[1] <= au.DISPUTE_S and detect.disputes(text):
            au.dispute(self.db, d[0])

    def scrub(self, turns: list[dict]) -> list[dict]:
        if not self._said:
            return turns
        out = []
        for t in turns:
            content = t.get("content") if isinstance(t, dict) else None
            if content is not None and t.get("role") == "assistant" and isinstance(content, str) and any(x in content for x in self._said):
                t = {**t, "content": PLACEHOLDER}
            out.append(t)
        return out

    # ---- housekeeping ------------------------------------------------------------------------------------------------
    def step(self, now: float) -> bool:
        if not self.enabled():
            return False
        if self._last_audit is not None and 0 <= now - self._last_audit < AUDIT_EVERY_S:
            return False
        self._last_audit = now
        n = au.audit(self.db, now, frozen=self.frozen(), glances=self.glances, on_scored=self.on_scored)
        self.db.purge(now)
        return n > 0

    def purge(self, now: float) -> None:
        self.db.purge(now)

    def forget(self) -> None:
        self.db.forget()
        self._return, self._runs, self._s8, self._band, self._dispute = None, {}, None, None, None
        self._answered_at, self._prev_state, self._last_presence = None, None, None

    def reset_p2(self, kind: str | None = None) -> None:
        self.db.reset_p2(kind)
        state = self.db.kv_get(UNSURE_KEY, {})
        if isinstance(state, dict):
            prefix = "P2:" if kind is None else f"P2:{kind}:"
            self.db.kv_set(UNSURE_KEY, {k: v for k, v in state.items() if not k.startswith(prefix)})

    # ---- readouts ----------------------------------------------------------------------------------------------------
    def honesty(self, now: float) -> au.Honesty:
        return au.honesty(self.db, now, enabled=self.enabled(), claims_enabled=CLAIMS_ENABLED and self.evidence)

    def _lines(self, now: float) -> list[tuple[str, float, float]]:
        """Resolution 31's lines: what she has observed (the cell's own median, not the prior-pulled predictive)."""
        out = []
        for predictor, bucket, cell in self.db.cells():
            if predictor == "P1" and "/" in bucket:
                c = pr.lognormal_decayed(cell, pr.P1_PRIOR, now)
                if c.n < st.HUNCH_N:
                    continue
                band, daytype = bucket.split("/")
                conf = ap.confidence(c.n, c.state.cusum)
                span = _span(math.exp(c.s1 / c.n))
                out.append((f"Coming back after leaving in the {band} on {daytype}s: usually about {span}, "
                            f"{round(c.n)} times, {_sure(conf)}", conf, c.n))
            elif predictor == "P3" and not bucket.endswith(":*") and isinstance(cell, pr.LogNormalCell):
                prior = pr.P3_PRIORS["colab" if bucket.startswith("colab:") else "claude"]
                c = pr.lognormal_decayed(cell, prior, now)
                if c.n < st.HUNCH_N:
                    continue
                conf = ap.confidence(c.n, c.state.cusum)
                span = _span(math.exp(c.s1 / c.n))
                out.append((f"Agent runs ({bucket.split(':', 1)[1]}): about {span}, {round(c.n)} runs, {_sure(conf)}",
                            conf, c.n))
            elif predictor == "P4" and "/" in bucket and isinstance(cell, pr.BetaCell):
                c = pr.beta_decayed(cell, pr.P4_PRIOR, now)
                n = pr.beta_evidence(c, pr.P4_PRIOR)
                if n < st.HUNCH_N:
                    continue
                band, daytype = bucket.split("/")
                conf = ap.confidence(n, c.state.cusum)
                pct = round(100 * c.alpha / (c.alpha + c.beta))
                out.append((f"Gaming on {daytype} {band}s: {pct}% of days, {round(n)} days, {_sure(conf)}", conf, n))
        return out

    def msg(self, now: float):
        from riya.core.protocol import RoutineCellInfo, RoutineHonestyInfo, RoutineMsg
        h = self.honesty(now)
        lines = self._lines(now) if self.enabled() else []
        return RoutineMsg(enabled=self.enabled(),
                          cells=[RoutineCellInfo(line=line, confidence=round(c, 3), n=round(n, 2))
                                 for line, c, n in lines],
                          honesty=RoutineHonestyInfo(claims_true=h.claims_true, claims_scored=h.claims_scored,
                                                     hunches_right=h.hunches_right, hunches_scored=h.hunches_scored,
                                                     unverifiable=h.unverifiable, expired=h.expired,
                                                     claims_enabled=h.claims_enabled))


def _sure(conf: float) -> str:
    return "sure" if conf >= st.CLAIM_C else "fairly sure" if conf >= st.GUESS_C else "still learning"


def _span(seconds: float) -> str:
    """'N min' under 90 minutes, else hours to the half hour."""
    if seconds < 90 * 60:
        return f"{st.round_minutes(seconds)} min"
    h = round(seconds / 1800.0) / 2
    return f"{h:g} hours"
