"""RiyaCore: wires memory, feelings, presence, initiative, routing and consolidation into replies and speaking first."""
import logging
import queue
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from riya.agents.hub import AgentEvent, AgentHub
from riya.core.appraise import SENTENCE_END
from riya.core.consolidate import Consolidator, remember_intent, third_person
from riya.core.eyes.capture import FakeWindowSource, WindowInfo, WindowSource, clamp_rect, grab_jpeg, thumb
from riya.core.eyes.policy import GlancePolicy, window_refusal
from riya.core.eyes.see import describe, is_private, is_screen_question, vision_messages
from riya.core.feelings import Feelings
from riya.core.initiative import Action, Initiative
from riya.core.memory import Memory, about_me
from riya.core.models import LOCAL_TEMPERATURE, ChatModel
from riya.core.persona import load_traits, persona_prompt, style_examples
from riya.core.presence import Presence
from riya.core.prompt import (FEELINGS_SITUATION, IDENTITY_SITUATION, SEEING_SITUATION, agents_block, bid_situation,
                              build_messages, cant_see_situation, eyes_block, greeting_reports, greeting_situation,
                              remember_situation, report_bubble, report_line, report_situation)
from riya.core.protocol import (EXPRESSIONS, PROTOCOL_VERSION, AgentInfo, AgentsMsg, DontLook, EpisodeDelete, ErrorMsg,
                                ExpressionMsg, EyesMsg, GetMemories, GlanceDelete, GlanceInfo, Hello, Look, MemoriesMsg,
                                MemoryCorrect, MemoryDelete, QuitCore, Say, SayEnd, SetSettings, SettingsMsg, StateMsg,
                                TalkHotkey, ThreadResolve, TranscriptDelete)
from riya.core.router import Router, needs_thinking
from riya.core.settings import Settings, apply_patch, load_settings, save_settings, set_autostart
from riya.core.shaping import (BUDGETS, CAPS, CODE_BLOCK, ONLINE_BUDGETS, cap_sentences, count_emoji, drop_generic,
                               is_feelings_question, is_identity_question, limit_emoji, strip_label, turn_kind)
from riya.core.store import KV

log = logging.getLogger("riya.engine")
LAUNCHER = Path(__file__).resolve().parents[2] / "scripts" / "start_riya.ps1"
MODEL_DOWN = "Riya's local model isn't responding"
KEEP_TURNS = 24
BID_IGNORED_S = 300
GREET_DEFER_MAX_S = 1800
SAVE_EVERY_S = 30
STATE_EVERY_S = 30
PURGE_EVERY_S = 86400
CONSOLIDATE_RETRY_S = 600
REMEMBER_ACKS = ("Got it~ I'll remember that.", "Noted, I won't forget.", "Okay, saved it.",
                 "Mm, I'll remember that one.")  # templated, in her voice; the small model can't be trusted to confirm
GLANCE_BACKOFF_S = 1800  # after a failed look (an outage, a 429, the network): no unforced glances for 30 min
GLANCE_WAIT_S = 30  # a screen question waits this long for a glance in flight to finish
TODAY_KEY = "eyes.today"  # {"day", "count"}: the daily cap survives a restart
PEEK_EVERY_S = 30  # same window past the refresh time: a local-only thumbnail check, at most this often
LAST_WINDOW_S = 300  # a screen question typed into her own window looks at the window they were just in
BID_NOTE_S = 1800  # a note this fresh may be a bid's reason
SEE_TOKENS = 400  # a screen answer's online budget, at least (flash models may spend tokens thinking)
BUDDY_HALF_W = 150  # the buddy window is 300 px wide; buddy_position is its top-left corner
CENTER_BAND = 0.1  # the window's centre within 10% of its monitor's width of hers: "center"
TITLE_CHARS = 200
COULDNT_SEE = "you tried to look but couldn't see it (your eyes didn't answer)"
REPORT_GAP_S = 20  # at most one agent report this often; whatever arrives meanwhile merges into the next
REPORT_MAX_AGE_S = 6 * 3600  # a report still pending after this long is dropped
REPORT_FACE_S = 20
DRAIN_MAX = 200  # inbox items taken per tick: a flood waits for later ticks, the tick never stalls on it
URGENT = frozenset({"failed", "waiting", "lost", "stalled"})
REPORT_FACES = {"failed": "worried", "lost": "worried", "stalled": "worried", "waiting": "curious", "finished": "happy"}
HOOKS_SEEN_KEY = "agents.hooks_seen_at"
_EYES_KEYS = ("eyes_enabled", "dont_look", "online_enabled", "eyes_deny_titles", "eyes_deny_processes")
_TILDES = re.compile(r"[~\s]*")
_ENDED = re.compile(r"[.!?…~][\"')\]]?[\s~]*$")  # a sentence's end, at the end of the text


def _worded(text: str) -> bool:
    return any(c.isalnum() for c in text)


def _query(action: Action) -> str:
    return action.thread.text if action.thread else action.reason


def _threads(action: Action) -> list:
    return [action.thread] if action.thread else []


def _thumb(jpeg: bytes) -> bytes:
    try:
        return thumb(jpeg)
    except Exception:  # an image PIL can't read: "unknown", so the refresh check never fires on it
        return b""


def _key(info: WindowInfo) -> tuple[int, str]:
    return info.hwnd, info.title


class Utterance:
    """What tick() hands over when she speaks first: iterate it for the sentences; spoken says if it's meant aloud."""

    def __init__(self, sentences: Iterator[str], spoken: bool):
        self._sentences, self.spoken = sentences, spoken

    def __iter__(self) -> "Utterance":
        return self

    def __next__(self) -> str:
        return next(self._sentences)

    def close(self) -> None:
        self._sentences.close()


class RiyaCore:
    def __init__(self, *, db, embedder, clock, idle_source, local: ChatModel, online: ChatModel | None,
                 emit: Callable[[BaseModel], None], persona: str | None = None, vision: ChatModel | None = None,
                 window_source: WindowSource | None = None, grab: Callable[[WindowInfo], bytes] = grab_jpeg,
                 body_present: Callable[[], bool] = lambda: True):
        """vision: the online model instance her eyes use (glances and screen questions); None means she can't see.
        window_source and grab: the foreground window and its pixels (JPEG bytes, in memory only). body_present: a
        body is connected (her eyes stay shut without one)."""
        self.clock, self.local, self.online, self.emit = clock, local, online, emit
        self.body_present, self.quit_requested = body_present, threading.Event()  # quit_requested: the tray's Quit
        self.vision, self.windows, self._grab = vision, window_source or FakeWindowSource(None), grab
        # the small local model gets the short persona, the online one the full sheet; persona= overrides both
        self.personas = {k: persona if persona is not None else persona_prompt(k) for k in ("local", "online")}
        self.examples = style_examples()
        self.memory = Memory(db, embedder, clock)
        self.kv = KV(db, self.memory.lock)
        traits, saved = load_traits(), self.kv.get("feelings")
        self.feelings = Feelings.from_dict(saved, clock, traits) if saved else Feelings(clock, traits)
        self.presence = Presence(idle_source, clock, self.kv)
        self.initiative = Initiative(clock, lambda: self.settings)
        self.router = Router(local, online, self.settings.online_enabled)
        self.consolidator = Consolidator(self.memory, self._consolidation_model, clock)
        self.turns: list[dict] = []
        self.talk_requested = False
        self.on_spoke_first: Callable[[str], None] = lambda kind: None  # e.g. the voice loop opens the mic after a bid
        self._lock = threading.Lock()
        now = clock.now()
        self.last_turn_at = self.memory.last_transcript_at() or now
        self.pending_bid_at: float | None = None
        self._bid_thread: int | None = None  # the thread a pending bid brought up
        self._deferred_greeting: tuple[float, float] | None = None  # (away_s, came_back_at) while gaming
        self._last_save = self._last_state_at = now
        self._last_purge: float | None = None
        self._last_consolidation: float | None = None
        self._away_s = 0.0  # absence behind the greeting that speak_first is about to make
        self._acks = 0  # remember-that acks said so far, which picks the next one
        s = self.settings
        self.policy = GlancePolicy(clock, cap=s.glance_daily_cap, min_gap_s=s.glance_min_gap_s)
        if isinstance(today := self.kv.get(TODAY_KEY), dict):
            self.policy.restore(today.get("day"), today.get("count"))
        self._eyes_lock = threading.RLock()  # the policy and the eyes state: tick, reply, socket and glance threads
        self._expr_lock = threading.Lock()  # _sync_expression compares and sets under it
        self._eyes_gen = 0  # bumped whenever she must stop looking (don't-look, eyes off...): in-flight looks discard
        self._dont_look, self._eyes_on = s.dont_look, s.eyes_enabled
        self._glance_thread: threading.Thread | None = None
        self._glanced: tuple[tuple[int, str], float] | None = None  # (window, when) of her last look
        self._peeked_at: float | None = None
        self._behind: tuple[WindowInfo, float] | None = None  # the last foreground window that wasn't hers
        self._look: Look | None = None
        self._eyes_sent: EyesMsg | None = None
        # her agents: the hub is only touched under the core lock; the inboxes are filled from other threads (the
        # hook receiver, the logwatch thread) and drained by the tick
        self.agents = AgentHub(clock)
        self.agent_inbox: queue.SimpleQueue = queue.SimpleQueue()  # Claude Code hook payloads (dicts)
        self.colab_inbox: queue.SimpleQueue = queue.SimpleQueue()  # (run name, kind, line) from LogWatch.poll
        self._reports: dict[str, tuple[AgentEvent, float]] = {}  # pending, per agent only the newest: (event, at)
        self._greet_reports: list[tuple[AgentEvent, float]] = []  # folded into the greeting about to be said
        self._last_report_at: float | None = None
        seen = self.kv.get(HOOKS_SEEN_KEY)
        self._hooks_seen_at = float(seen) if isinstance(seen, (int, float)) else None
        self._agents_sent = self._agents_msg()
        self._expression = self._face()

    @property
    def settings(self) -> Settings:
        return load_settings(self.kv)

    def _consolidation_model(self) -> ChatModel:
        return self.online if self.settings.online_consolidation and self.online else self.local

    # ---- speaking -----------------------------------------------------------------------------------
    def reply(self, text: str, source: Literal["typed", "voice"] = "typed") -> Iterator[str]:
        with self._lock:
            saved = None
            try:
                saved = self._take_user_turn(text)
            except Exception:
                log.exception("couldn't record the user's turn")
            base = [remember_situation(saved) if saved else None,
                    IDENTITY_SITUATION if is_identity_question(text) else None,
                    FEELINGS_SITUATION if is_feelings_question(text) else None]
            seeing = None
            if not saved and is_screen_question(text):
                try:
                    seeing = self._look_now(base)
                except Exception:
                    log.exception("couldn't look for a screen question")
            situation = " ".join(filter(None, [*base, seeing.situation if seeing else None])) or None
            lead = None
            if saved:
                lead, self._acks = REMEMBER_ACKS[self._acks % len(REMEMBER_ACKS)], self._acks + 1
            yield from self._speak("reply", user_text=text, query=text, spoken=True, situation=situation, lead=lead,
                                   small=bool(saved), seeing=seeing if seeing and seeing.jpeg else None)

    def _take_user_turn(self, text: str) -> str | None:
        """Records the user's turn; returns the fact text when it was a remember-that turn (saved right away)."""
        now = self.clock.now()
        answered_bid = self.pending_bid_at is not None and now - self.pending_bid_at < BID_IGNORED_S
        if answered_bid and self._bid_thread is not None:
            self.memory.mark_thread(self._bid_thread, status="followed_up")
        self.last_turn_at, self.pending_bid_at, self._bid_thread = now, None, None
        self.memory.append_transcript("user", text)
        saved = None
        if (said := remember_intent(text)) is not None:
            saved = f"The user said: {third_person(said)}"
            self.memory.add_fact(saved, source="user_said")
        self.feelings.on_user_text(text)
        self.presence.note_turn()
        self._sync_expression()
        return saved

    def speak_first(self, action: Action) -> Iterator[str]:
        with self._lock:
            reports, self._greet_reports = self._greet_reports, []
            situation = greeting_situation(self._away_s, self.feelings.state.missing, action.thread)
            if reports:  # what her agents did while they were away
                situation += " " + greeting_reports([ev for ev, _ in reports])
            yield from self._speak("greet", user_text=None, query=_query(action), situation=situation,
                                   spoken=action.spoken, threads=_threads(action),
                                   after=lambda outcome: self._greeted(action, outcome, reports))

    def _greeted(self, action: Action, outcome: str, reports: list | None = None) -> None:
        """The thread counts as followed up only once she actually asked; a barge-in still stops a re-raise today.
        Reports folded into a greeting the model couldn't make go back to pending (unless newer news replaced them)."""
        if outcome == "failed":
            for ev, at in reports or []:
                self._reports.setdefault(ev.agent_id, (ev, at))
        if action.thread and outcome != "failed":
            self.memory.mark_thread(action.thread.id, status="followed_up" if outcome == "done" else None,
                                    raised_at=self.clock.now())

    def _speak(self, kind: str, *, user_text: str | None, query: str, situation: str | None, spoken: bool,
               threads: list | None = None, after: Callable[[str], None] | None = None,
               lead: str | None = None, small: bool = False,
               seeing: SimpleNamespace | None = None) -> Iterator[str]:
        """Streams one utterance. The outcome is "done", "closed" (the consumer stopped early) or "failed" (model).
        threads, when given, replaces the retrieved follow-ups (speaking first carries only its chosen thread).
        lead is a fixed sentence said first, before the model is even asked; the model may add only one more.
        small makes it small talk on the local model whatever the text (a remember-that turn just reacts).
        seeing (from _look_now, with an image) sends the image with her full prompt to the vision model; the same
        shaping applies. A short generic sentence of the model's (shaping.drop_generic) is dropped: not said, not
        recorded."""
        uid, parts, said, thinking, outcome = uuid4().hex, [], [], False, "failed"
        sentences, done, dropped = None, {"stream": False}, []
        try:
            self.router.online_enabled = self.settings.online_enabled
            spoken = spoken and not self.settings.mute_voice
            if lead:
                self.emit(Say(utterance_id=uid, text=lead, spoken=spoken, kind=kind))
                said.append(lead)
                yield lead
            route_text = "" if small else user_text or ""
            turn, budget, cap = self._shape(kind, route_text)
            # decided first, so the prompt carries the matching persona; a look at the screen is always online
            route = "online" if seeing else self.router.route(route_text)
            retrieved, eyes, agents = self._retrieve(query, threads), self._eyes_context(), self._agents_context()

            def build(persona: str, situation: str | None) -> list[dict]:
                return build_messages(persona=persona, examples=self.examples, feelings_tag=self.feelings.prompt_tag(),
                                      retrieved=retrieved, turns=self.turns, user_text=user_text, situation=situation,
                                      now=self.clock.now(), kind=turn, eyes=eyes, agents=agents)

            messages = build(self.personas[route], situation)
            if route == "online":
                thinking = True
                self.feelings.on_event("thinking")
                self._sync_expression()
            if seeing:
                messages[-1] = vision_messages(seeing.jpeg, user_text)[-1]
                stream = self._see_stream(messages, seeing,
                                          lambda why: build(self.personas["local"], seeing.fallback(why)),
                                          budget, max(ONLINE_BUDGETS.get(turn, budget), SEE_TOKENS))
                ran_out = lambda: getattr(self.local if seeing.fell_back else self.vision,  # noqa: E731
                                          "finish_reason", None) == "length"
            else:
                stream = self.router.stream(messages, route_text, budget, ONLINE_BUDGETS.get(turn))
                ran_out = lambda: self.router.last_finish_reason == "length"  # noqa: E731
            kept = drop_generic(self._sentences(stream, parts, done, ran_out, said=bool(lead)), others=bool(lead),
                                dropped=dropped)
            sentences = cap_sentences(kept, 1 if lead else cap)
            for sentence in sentences:
                thinking = self._end_thinking(thinking)
                self.emit(Say(utterance_id=uid, text=sentence, spoken=spoken, kind=kind))
                said.append(sentence)
                yield sentence
            outcome = "done"
        except GeneratorExit:
            outcome = "closed"
            raise
        except Exception:
            log.exception("model failed during a %s", kind)
            self.emit(ErrorMsg(message=MODEL_DOWN))
        finally:
            if sentences is not None:
                sentences.close()  # stops the model stream however the utterance ended
            # the model finished on its own and nothing of it was dropped: keep its own spacing
            whole = outcome == "done" and done["stream"] and not dropped and not done.get("cut")
            text = None if outcome == "failed" else (
                " ".join(filter(None, [lead, limit_emoji(strip_label("".join(parts)).strip())])) if whole
                else " ".join(said))
            try:
                self._end_thinking(thinking)
                self.emit(SayEnd(utterance_id=uid))
                self._finish_turn(user_text, text)
                if after:
                    after(outcome)
            except Exception:
                log.exception("couldn't record the %s", kind)

    @staticmethod
    def _shape(kind: str, text: str) -> tuple[str | None, int, int]:
        """(turn kind, token budget, sentence cap) for an utterance: a greeting has its own budget, cap and rules line;
        a reply's depend on the turn."""
        if kind == "greet":
            return "greeting", BUDGETS["greeting"], CAPS["greeting"]
        if kind == "report":
            return "report", BUDGETS["report"], CAPS["report"]
        k = turn_kind(text, about_me=about_me(text), needs_thinking=needs_thinking(text))
        return k, BUDGETS[k], CAPS[k]

    @staticmethod
    def _boundary(buf: str) -> int | None:
        """Where the first sentence in buf ends; a sentence end inside an open code fence doesn't count, so a fenced
        block (and its indentation) goes out whole, byte for byte."""
        return next((m.end() for m in SENTENCE_END.finditer(buf) if buf.count("```", 0, m.end()) % 2 == 0), None)

    @staticmethod
    def _cut(buf: str, final: bool = False) -> tuple[int, int] | None:
        """The first sentence in buf as (end of its words, end of what folds into it). A sentence with no letter or
        digit ("~", "…", "!!", an emoji) and a "~" opening the next one fold into the sentence before; one at the very
        start folds into the next. None while the next worded sentence hasn't started; final (the stream ended) takes
        the rest of buf."""
        head, pos = None, 0
        while (n := RiyaCore._boundary(buf[pos:])) is not None:
            worded = _worded(buf[pos:pos + n])
            if worded and head is not None:
                return head, _TILDES.match(buf, pos).end()
            pos += n
            if worded:
                head = pos
        if head is not None and _worded(buf[pos:]):
            return head, _TILDES.match(buf, pos).end()
        return (len(buf) if head is None else head, len(buf)) if final and buf.strip() else None

    @staticmethod
    def _unfinished(buf: str) -> bool:
        """buf ends in words with no sentence end after them: a reply cut off mid-sentence."""
        pos = 0
        while (n := RiyaCore._boundary(buf[pos:])) is not None:
            pos += n
        return _worded(buf[pos:]) and not _ENDED.search(buf[pos:])

    @staticmethod
    def _sentences(stream: Iterator[str], parts: list[str], done: dict, ran_out: Callable[[], bool] = lambda: False,
                   said: bool = False) -> Iterator[str]:
        """The model's pieces as whole sentences (_cut), each with only the reply's first emoji (an emoji's leftovers
        vanish) and the first without a leading "Riya:" label. When the model ran out of tokens (ran_out) after a
        complete sentence was said (here, or before it: said), its unfinished last fragment is dropped and
        done["cut"] set. Closing this closes the model stream; done["stream"] is set once the model ended by itself."""
        buf, used, first = "", 0, True

        def tidy(sentence: str) -> str:
            nonlocal used, first
            sentence = strip_label(sentence).strip() if first else sentence.strip()
            first = first and not sentence
            kept = limit_emoji(sentence, max(0, 1 - used))  # never touches fenced code
            used += count_emoji(CODE_BLOCK.sub("", kept))
            return kept if kept == sentence or any(c.isalnum() for c in kept) else ""  # an emoji's leftovers vanish

        def take(cut: tuple[int, int]) -> str:
            nonlocal buf
            head, end = cut
            words, tail, buf = tidy(buf[:head]), tidy(buf[head:end]), buf[end:]
            return f"{words} {tail}" if words and tail else words or tail

        try:
            for piece in stream:
                parts.append(piece)
                buf += piece
                while (cut := RiyaCore._cut(buf)) is not None:
                    if sentence := take(cut):
                        said = True
                        yield sentence
            if said and RiyaCore._unfinished(buf) and ran_out():
                buf, done["cut"] = "", True
            while (cut := RiyaCore._cut(buf, final=True)) is not None:
                if sentence := take(cut):
                    yield sentence
            done["stream"] = True
        finally:
            if hasattr(stream, "close"):
                stream.close()

    def _retrieve(self, query: str, threads: list | None):
        r = self.memory.retrieve(query)
        return r if threads is None else replace(r, threads=threads)

    def _finish_turn(self, user_text: str | None, said: str | None) -> None:
        """said is None when the model failed: the user's turn stays in the transcript, her answer isn't recorded."""
        if said is not None:
            if said:
                self.memory.append_transcript("assistant", said)
            new = ([] if user_text is None else [{"role": "user", "content": user_text}]) + (
                [{"role": "assistant", "content": said}] if said else [])
            self.turns = (self.turns + new)[-KEEP_TURNS:]
            self.last_turn_at = self.clock.now()
        self._save_feelings()
        self._sync_expression()

    def _end_thinking(self, thinking: bool) -> bool:
        if thinking:
            self.feelings.on_event("thinking_done")
            self._sync_expression()
        return False

    def _bid_reason(self, action: Action) -> str:
        """Without a thread, the latest fresh note may be the reason ("still on that bug?"); never a private one, and
        not while her eyes are off or covered."""
        s, g = self.settings, next(iter(self.memory.recent_glances(1)), None)
        if action.thread or g is None or not s.eyes_enabled or s.dont_look or self.clock.now() - g.at > BID_NOTE_S                 or is_private(g.activity):
            return action.reason
        return f"{action.reason}; last you saw, they were {g.activity} in {g.app}" + (
            f" ({g.notable})" if g.notable else "")

    def _bid(self, action: Action) -> None:
        try:
            messages = build_messages(persona=self.personas["local"], examples=self.examples,
                                      feelings_tag=self.feelings.prompt_tag(),
                                      retrieved=self._retrieve(_query(action), _threads(action)), turns=self.turns,
                                      user_text=None, situation=bid_situation(self._bid_reason(action), action.thread),
                                      now=self.clock.now(), eyes=self._eyes_context(), agents=self._agents_context())
            line = "".join(self.local.stream(messages, BUDGETS["bid"], LOCAL_TEMPERATURE))
            line = limit_emoji(strip_label(line).strip())
        except Exception:
            log.exception("model failed during a bid")
            return
        if not line:
            return
        if action.thread:
            self.memory.mark_thread(action.thread.id, raised_at=self.clock.now())
        uid = uuid4().hex
        self.emit(Say(utterance_id=uid, text=line, spoken=False, kind="bid"))
        self.emit(SayEnd(utterance_id=uid))
        self.memory.append_transcript("assistant", line)
        self.turns = (self.turns + [{"role": "assistant", "content": line}])[-KEEP_TURNS:]
        self.last_turn_at = self.pending_bid_at = self.clock.now()
        self._bid_thread = action.thread.id if action.thread else None
        if self.presence.state == "gaming":  # only the hotkey opens the mic during a game
            return
        try:
            self.on_spoke_first("bid")
        except Exception:
            log.exception("on_spoke_first failed after a bid")

    # ---- time ---------------------------------------------------------------------------------------
    def tick(self) -> Utterance | None:
        if not self._lock.acquire(blocking=False):  # a reply or greeting is running: catch up on the next tick
            return None
        try:
            return self._tick()
        except Exception:
            log.exception("tick failed")
            return None
        finally:
            self._lock.release()

    def _tick(self) -> Utterance | None:
        """Each step is guarded on its own, so one failing step can't starve the others."""
        now = self.clock.now()
        events = self._step("presence", self.presence.update) or []
        self._step("agents", lambda: self._agents_step(now))
        action = self._step("greeting", lambda: self._greeting(events, now))
        report = None if action else self._step("report", lambda: self._report_step(now))
        if action is None and report is None:
            self._step("bid", self._maybe_bid)
        self._step("bid ignored", lambda: self._check_bid_ignored(now))
        self._step("eyes", lambda: self._eyes_step(now))
        self._step("feelings", lambda: self.feelings.tick(present=self.presence.state != "away"))
        self._step("consolidation", lambda: self._maybe_consolidate(now))
        self._step("purge", lambda: self._maybe_purge(now))
        self._step("persistence", lambda: self._maybe_save(now))
        self._step("state", lambda: self._maybe_state(now))
        self._step("expression", self._sync_expression)
        if action:
            return Utterance(self.speak_first(action), action.spoken)
        return report if isinstance(report, Utterance) else None

    @staticmethod
    def _step(name: str, fn: Callable[[], object]):
        try:
            return fn()
        except Exception:
            log.exception("tick step %r failed", name)
            return None

    def _greeting(self, events: list, now: float) -> Action | None:
        action = None
        for ev in events:
            if ev.kind == "came_back":
                self.feelings.tick(present=False)  # catch up on the absence first, so it can't fade her missing them
                if self.presence.state == "gaming":  # never greet into a game: wait until it ends
                    self._deferred_greeting = (ev.away_s, now)
                else:
                    self._deferred_greeting = None
                    action = self._came_back(ev.away_s)
        if self._deferred_greeting and action is None:
            away_s, at = self._deferred_greeting
            if now - at > GREET_DEFER_MAX_S:
                self._deferred_greeting = None
            elif self.presence.state not in ("gaming", "away"):
                self._deferred_greeting = None
                action = self._came_back(away_s)
        return action

    def _maybe_bid(self) -> None:
        if self.presence.state == "break":
            bid = self.initiative.on_break("break", self.presence.work_stretch_s(), self.memory.due_threads())
            if bid:
                self._bid(bid)

    def _check_bid_ignored(self, now: float) -> None:
        if self.pending_bid_at is not None and now - self.pending_bid_at >= BID_IGNORED_S:
            self.feelings.on_event("bid_ignored")
            self.pending_bid_at = self._bid_thread = None

    def _maybe_consolidate(self, now: float) -> None:
        if self.presence.state == "gaming":  # never load the local model during a game
            return
        if self.consolidator.due(self.last_turn_at) and (
                self._last_consolidation is None or now - self._last_consolidation >= CONSOLIDATE_RETRY_S):
            self._last_consolidation = now
            self.consolidator.run()

    def _maybe_purge(self, now: float) -> None:
        if self._last_purge is None or now - self._last_purge >= PURGE_EVERY_S:
            self._last_purge = now
            self.memory.purge_transcripts()
            self.memory.purge_glances()
            self.memory.resolve_stale_threads()

    def _maybe_save(self, now: float) -> None:
        if now - self._last_save >= SAVE_EVERY_S:
            self._save_feelings()

    def _maybe_state(self, now: float) -> None:
        if now - self._last_state_at >= STATE_EVERY_S:
            self._last_state_at = now
            self.emit(self._state_msg())

    def _came_back(self, away_s: float) -> Action | None:
        self.feelings.on_return(away_s)
        self._sync_expression()
        try:  # what they said just before leaving may hold the thread to greet them with
            self.consolidator.run_pending(self.last_turn_at, model=self.local)
            self._last_consolidation = self.clock.now()
        except Exception:
            log.exception("consolidation before the greeting failed")
        action = self.initiative.on_came_back(away_s, self.memory.due_threads())
        if action is not None:
            self._away_s = away_s  # the thread is marked by speak_first, once the greeting was actually said
            self._greet_reports, self._reports = list(self._reports.values()), {}  # the greeting tells them
            if self._greet_reports:
                self._last_report_at = self.clock.now()
        return action

    # ---- agents -------------------------------------------------------------------------------------
    def _agents_step(self, now: float) -> None:
        """Drains both inboxes into the hub (at most DRAIN_MAX items a tick, Colab first: it's small), runs the hub's
        staleness rules, queues its events as pending reports (per agent only the newest; older than 6 h dropped)
        and sends the agents list when it changed. Any hook payload proves the hooks are installed: hooks_seen_at."""
        events, n, hooks = [], 0, False
        for inbox in (self.colab_inbox, self.agent_inbox):
            while n < DRAIN_MAX:
                try:
                    item = inbox.get_nowait()
                except queue.Empty:
                    break
                n += 1
                try:
                    if inbox is self.agent_inbox:
                        hooks = True
                        events += self.agents.on_claude_event(item)
                    else:
                        events += self.agents.on_colab(*item)
                except Exception:
                    log.exception("couldn't take an agent event")
        events += self.agents.tick()
        if hooks:
            self._hooks_seen_at = now
            self.kv.set(HOOKS_SEEN_KEY, now)
        for ev in events:
            self._reports.pop(ev.agent_id, None)
            self._reports[ev.agent_id] = (ev, now)
        self._reports = {k: v for k, v in self._reports.items() if now - v[1] < REPORT_MAX_AGE_S}
        self._emit_agents()

    def _report_step(self, now: float) -> "Utterance | bool | None":
        """Delivers the pending agent reports with manners, everything pending merged into one line, at most one
        every 20 s. Never while away (the came-back greeting carries them). An urgent kind goes out now (focused,
        talking, break); a finished one waits for a break or a talking turn. While gaming it's a templated bubble and
        the model is never called. Returns the spoken Utterance, True when a bubble was shown, else None."""
        state = self.presence.state
        if not self._reports or state == "away":
            return None
        if self._last_report_at is not None and 0 <= now - self._last_report_at < REPORT_GAP_S:
            return None
        events = [ev for ev, _ in self._reports.values()]
        kinds = {ev.kind for ev in events}
        if not (kinds & URGENT or state in ("break", "talking")):
            return None
        self._reports, self._last_report_at = {}, now
        self.feelings.show(REPORT_FACES[next(k for k in REPORT_FACES if k in kinds)], REPORT_FACE_S)
        self._sync_expression()
        if state == "gaming":
            self._report_bubble(events)
            return True
        return Utterance(self._report(events), True)

    def _report(self, events: list[AgentEvent]) -> Iterator[str]:
        """One spoken line from the local model (40 tokens, 1 sentence); if the model can't, the templated bubble."""
        def after(outcome: str) -> None:
            if outcome == "failed":
                self._report_bubble(events)

        with self._lock:
            yield from self._speak("report", user_text=None, query=report_line(events, detail=False),
                                   situation=report_situation(events), spoken=True, threads=[], after=after)

    def _report_bubble(self, events: list[AgentEvent]) -> None:
        """The templated report: a silent bubble, kept in the transcript like a bid."""
        line, uid = report_bubble(events), uuid4().hex
        self.emit(Say(utterance_id=uid, text=line, spoken=False, kind="report"))
        self.emit(SayEnd(utterance_id=uid))
        self.memory.append_transcript("assistant", line)
        self.turns = (self.turns + [{"role": "assistant", "content": line}])[-KEEP_TURNS:]
        self.last_turn_at = self.clock.now()

    def _agents_context(self) -> str | None:
        """The `### Your agents:` prompt line (outside the memory block, like the eyes block)."""
        return agents_block(self.agents.snapshot(), self.clock.now())

    def _agents_msg(self) -> AgentsMsg:
        return AgentsMsg(items=[AgentInfo(**a) for a in self.agents.snapshot()], hooks_seen_at=self._hooks_seen_at)

    def _emit_agents(self) -> None:
        """agents, only when it changed. hello_messages reads the last one sent (never the hub, from another thread)."""
        msg = self._agents_msg()
        if msg != self._agents_sent:
            self._agents_sent = msg
            self.emit(msg)

    # ---- eyes ---------------------------------------------------------------------------------------
    def _eyes_context(self) -> str | None:
        """The eyes block: none with her eyes off, and no "what they're doing now" line while she isn't looking."""
        s = self.settings
        if not s.eyes_enabled:
            return None
        return eyes_block(self.memory.recent_glances(3), self.clock.now(), now_line=not s.dont_look)

    def _why_not(self, info: WindowInfo | None, s: Settings, force: bool = False,
                 presence: str | None = None) -> str | None:
        """None when she may look at info now, else the policy's reason. Call with _eyes_lock held."""
        if not self.body_present():
            return "no body"
        return self.policy.allowed(info=info, presence=presence or self.presence.state, eyes_enabled=s.eyes_enabled,
                                   dont_look=s.dont_look, online_ok=self.vision is not None and s.online_enabled,
                                   extra_titles=s.eyes_deny_titles, extra_processes=s.eyes_deny_processes, force=force)

    def _may_look(self, gen: int) -> bool:
        """Nothing closed her eyes since gen was read (a look in flight is dropped, never sent or kept)."""
        return gen == self._eyes_gen and self._eyes_on and not self._dont_look

    def _record(self, info: WindowInfo, small: bytes) -> None:
        """A look happened (with _eyes_lock held); today's count is saved so the cap survives a restart."""
        self.policy.record(info, small)
        self._glanced = (_key(info), self.clock.now())
        self._save_today()

    def _sent(self) -> None:
        """A vision request went out with nothing kept (with _eyes_lock held): it counts all the same."""
        self.policy.sent()
        self._save_today()

    def _save_today(self) -> None:
        day, count = self.policy.today()
        self.kv.set(TODAY_KEY, {"day": day, "count": count})

    def _fresh(self, info: WindowInfo) -> WindowInfo | None:
        """info re-read by its handle right before a grab: still open, not minimized, not hers, same title and
        process (so the handle wasn't reused), with its current rect. None otherwise."""
        try:
            now = self.windows.window(info.hwnd)
        except Exception:
            return None
        if now is None or now.minimized or now.is_riya or (now.title, now.process) != (info.title, info.process):
            return None
        return now

    def _covered(self, info: WindowInfo) -> bool:
        """Another window sits above info in z-order over the part a grab would take (one of hers, a minimized or a
        cloaked one doesn't count). True as well when that can't be told."""
        try:
            above = self.windows.above(info.hwnd)
        except Exception:
            return True
        left, top, right, bottom = clamp_rect(info.rect, info.monitor)
        return any(max(left, w.rect[0]) < min(right, w.rect[2]) and max(top, w.rect[1]) < min(bottom, w.rect[3])
                   for w in above if not (w.is_riya or w.minimized or w.cloaked))

    def _eyes_step(self, now: float) -> None:
        """Each tick: tell the body where the window is, and start a glance in a worker thread when one is allowed
        and wanted. The tick itself never waits for a capture or the vision model."""
        s, info = self.settings, self.windows.foreground()
        if info is not None and not info.is_riya:
            self._behind = (info, now)
        self._emit_look(info, s)
        with self._eyes_lock:
            self._emit_eyes(s)
            if self._glance_thread is not None and self._glance_thread.is_alive():
                return
            wanted = self.policy.wanted(info, None)
            peek = not wanted and info is not None and self._glanced is not None and self._glanced[0] == _key(info) \
                and now - self._glanced[1] >= self.policy.refresh_s \
                and (self._peeked_at is None or now - self._peeked_at >= PEEK_EVERY_S)
            if not (wanted or peek) or self._why_not(info, s) is not None:
                return
            if peek:
                self._peeked_at = now
            self._glance_thread = threading.Thread(target=self._glance, args=(info, self._eyes_gen, peek),
                                                   name="riya-glance", daemon=True)
            self._glance_thread.start()

    def _glance(self, info: WindowInfo, gen: int, peek: bool) -> None:
        """The worker: re-read the window and re-check the policy (still in front, nothing over it), grab, (for a
        peek) check the screen changed, describe, keep the note. Every request sent counts. An outage backs off 30
        min; replies with no usable note in a row 5, 15, 30, then 60 min. Closing her eyes at any point drops the look
        (never sent once closed)."""
        try:
            with self._eyes_lock:
                if not self._may_look(gen):
                    return
                fresh = self._fresh(info)
                if fresh is None or self._why_not(fresh, self.settings) is not None:
                    return
                front = self.windows.foreground()  # still the window in front, with nothing over it
                if front is None or front.hwnd != fresh.hwnd or self._covered(fresh):
                    return
            info = fresh
            jpeg = self._grab(info)
            if not jpeg:
                return
            small = _thumb(jpeg)
            if peek:
                with self._eyes_lock:
                    if not self.policy.wanted(info, small):
                        return
            if not self._may_look(gen):
                return
            note, outcome = describe(self.vision, jpeg)
            del jpeg
            with self._eyes_lock:
                if not self._may_look(gen) or note is None:
                    self._sent()
                    self._emit_eyes()
                    if note is not None:
                        log.info("dropped a glance: her eyes closed while it was in flight")
                    elif outcome == "unparseable":
                        self.policy.unusable()
                    else:
                        self.policy.backoff(GLANCE_BACKOFF_S)
                    return
                self._record(info, small)
                self.policy.usable()
                title = "" if is_private(note.activity) else info.title[:TITLE_CHARS]
                self.memory.add_glance(note.app, title, note.activity, note.details, note.notable)
                self._emit_eyes()
        except Exception:
            log.exception("glance failed")

    def _look_now(self, base: list[str | None]) -> SimpleNamespace:
        """A fresh look for a screen question (force: no min gap or backoff; privacy, the cap and don't-look still
        hold). jpeg is None when she can't look, and situation says why; otherwise situation tells her to answer from
        the image, and fallback is what she's told if the vision model then fails."""
        s, now = self.settings, self.clock.now()
        info = self.windows.foreground()
        if info is not None and info.is_riya and self._behind and now - self._behind[1] <= LAST_WINDOW_S:
            info = self._behind[0]  # typed into her own window: the window they were just in
        presence = "gaming" if self.presence.state == "gaming" else "talking"  # they're talking to her right now
        # one look at a time: a glance in flight finishes first, so the cap and the vision model are never shared
        # (and no glance can start meanwhile: the tick can't run while a reply holds the core lock)
        busy = self._glance_thread
        if busy is not None and busy.is_alive():
            busy.join(GLANCE_WAIT_S)
        with self._eyes_lock:
            gen = self._eyes_gen
            why = self._why_not(info, s, force=True, presence=presence)
            if why is None and busy is not None and busy.is_alive():
                why = "you're still busy with another look at their screen"
            if why is None:
                fresh = self._fresh(info)
                why = "the window changed" if fresh is None else self._why_not(fresh, s, force=True,
                                                                                presence=presence)
                info = fresh or info
            if why is None and self._covered(info):
                why = "covered"
            if why is None and not self._may_look(gen):
                why = "don't look" if self._eyes_on else "eyes off"
        jpeg = b""
        if why is None:
            try:
                jpeg = self._grab(info)
                why = None if jpeg else "no window"
            except Exception:
                log.exception("screen capture failed")
                why = "the screen capture didn't work"
        if why is None and not self._may_look(gen):
            why = "don't look"
        guess = None
        if window_refusal(info, s.eyes_deny_titles, s.eyes_deny_processes) is None:
            guess = f'"{info.title}" ({info.process})'
        if why is not None:
            return SimpleNamespace(jpeg=None, situation=cant_see_situation(
                why, guess if why == "online model unavailable" else None))
        return SimpleNamespace(jpeg=jpeg, info=info, thumb=_thumb(jpeg), gen=gen, fell_back=False,
                               situation=SEEING_SITUATION.format(window=guess),
                               fallback=lambda why: " ".join(filter(None, [*base, cant_see_situation(
                                   why, guess if why == COULDNT_SEE else None)])))

    def _see_stream(self, messages: list[dict], seeing: SimpleNamespace, fallback: Callable[[str], list[dict]],
                    local_budget: int, budget: int) -> Iterator[str]:
        """The vision model's answer about the screen; its first piece counts the look and ends a run of unusable
        replies. If her eyes closed since the grab, the image is never sent; if the model fails before saying
        anything, the request still counts and glances back off (30 min, or for empty replies in a row 5, 15, 30,
        then 60 min). Either way the local model answers from fallback(why) instead (text only, no image)."""
        if not self._may_look(seeing.gen):
            seeing.fell_back = True
            yield from self.local.stream(fallback("don't look" if self._eyes_on else "eyes off"), local_budget,
                                         LOCAL_TEMPERATURE)
            return
        stream, said, failed = self.vision.stream(messages, budget), False, False
        try:
            for piece in stream:
                if not said:
                    said = True
                    with self._eyes_lock:
                        self._record(seeing.info, seeing.thumb)
                        self.policy.usable()  # the vision model answered: unusable replies start over
                    self._emit_eyes()
                yield piece
            if said:
                return
        except Exception as e:
            if said:
                log.warning("the vision stream ended early: %s", e)
                return
            log.warning("couldn't see for a screen question: %s", e)
            failed = True
        finally:
            stream.close()
        with self._eyes_lock:
            self._sent()
            if failed:
                self.policy.backoff(GLANCE_BACKOFF_S)
            else:
                self.policy.unusable()
        self._emit_eyes()
        seeing.fell_back = True
        yield from self.local.stream(fallback(COULDNT_SEE), local_budget, LOCAL_TEMPERATURE)

    @staticmethod
    def _side(info: WindowInfo, s: Settings) -> str:
        """Where the window's centre is from her: her buddy centre when known, else its monitor's centre."""
        left, _, right, _ = clamp_rect(info.rect, info.monitor)
        if right <= left:
            left, _, right, _ = info.rect
        mon_left, _, mon_right, _ = info.monitor
        pos = s.buddy_position if s.mode == "buddy" else None
        ref = pos[0] + BUDDY_HALF_W if pos else (mon_left + mon_right) / 2
        dx = (left + right) / 2 - ref
        return "center" if abs(dx) <= CENTER_BAND * (mon_right - mon_left) else "left" if dx < 0 else "right"

    def _emit_look(self, info: WindowInfo | None, s: Settings) -> None:
        """look, only when it changed: the side of her the window is on, and whether she's watching it."""
        why = window_refusal(info, s.eyes_deny_titles, s.eyes_deny_processes)
        side = None if why in ("no window", "minimized", "riya window") else self._side(info, s)
        watching = (why is None and s.eyes_enabled and not s.dont_look
                    and self.presence.state in ("focused", "talking"))
        look = Look(side=side, watching=watching)
        if look != self._look:
            self._look = look
            self.emit(look)

    def _eyes_msg(self, s: Settings | None = None) -> EyesMsg:
        s = s or self.settings
        with self._eyes_lock:
            return EyesMsg(enabled=s.eyes_enabled, dont_look=s.dont_look, glances_today=self.policy.glances_today(),
                           cap=s.glance_daily_cap)

    def _emit_eyes(self, s: Settings | None = None) -> None:
        """eyes, only when it changed."""
        with self._eyes_lock:
            msg = self._eyes_msg(s)
            if msg != self._eyes_sent:
                self._eyes_sent = msg
                self.emit(msg)

    # ---- state out ----------------------------------------------------------------------------------
    def _save_feelings(self) -> None:
        self._last_save = self.clock.now()
        self.kv.set("feelings", self.feelings.to_dict())

    def _face(self) -> str:
        """Her expression: covering her eyes while don't-look is on, otherwise what she feels."""
        return "covering_eyes" if self._dont_look else self.feelings.expression()

    def _sync_expression(self) -> None:
        with self._expr_lock:  # computed and sent together, so a stale face can't land after a newer one
            name = self._face()
            if name != self._expression:
                self._expression = name
                self.emit(ExpressionMsg(name=name))

    def _state_msg(self) -> StateMsg:
        state = asdict(self.feelings.state)
        return StateMsg(presence=self.presence.state, expression=self._face(),
                        feelings={k: float(v) for k, v in state.items() if isinstance(v, (int, float))})

    def voice_state(self) -> tuple[SimpleNamespace, float]:
        s = self.feelings.state
        return SimpleNamespace(arousal=s.arousal, valence=s.valence), s.energy

    def hello_messages(self) -> list[BaseModel]:
        return [Hello(version=PROTOCOL_VERSION, expressions=list(EXPRESSIONS)),
                SettingsMsg(settings=asdict(self.settings)), self._state_msg(), self._agents_sent, self._eyes_msg(),
                *([self._look] if self._look else [])]

    # ---- body messages ------------------------------------------------------------------------------
    def handle(self, msg: BaseModel) -> None:
        try:
            if isinstance(msg, GetMemories):
                pass
            elif isinstance(msg, MemoryDelete):
                self.memory.delete_fact(msg.id, delete_episodes=msg.delete_episodes)
            elif isinstance(msg, MemoryCorrect):
                self.memory.correct(msg.id, msg.text)
            elif isinstance(msg, TranscriptDelete):
                self.memory.delete_transcript_day(msg.day)
            elif isinstance(msg, EpisodeDelete):
                self.memory.delete_episode(msg.id)
            elif isinstance(msg, ThreadResolve):
                self.memory.mark_thread(msg.id, status="resolved")
            elif isinstance(msg, GlanceDelete):
                self.memory.delete_glance(msg.id)
            elif isinstance(msg, SetSettings):
                self._set_settings(msg.patch)
                return
            elif isinstance(msg, DontLook):
                self._set_settings({"dont_look": msg.on})
                return
            elif isinstance(msg, TalkHotkey):
                self.talk_requested = True
                return
            elif isinstance(msg, QuitCore):
                self.quit_requested.set()
                return
            else:  # BodyReady (the socket sends hello on connect) and UserText (goes through reply)
                return
        except (KeyError, ValueError) as e:
            self.emit(ErrorMsg(message=str(e)))
            return
        self.emit(self._memories_msg())

    def _memories_msg(self) -> MemoriesMsg:
        mem = self.memory
        return MemoriesMsg(facts=[asdict(f) for f in mem.list_facts()], transcript_days=mem.transcript_days(),
                           episodes=[{"id": e.id, "started_at": e.started_at, "ended_at": e.ended_at,
                                      "summary": e.summary} for e in mem.list_episodes()],
                           threads=[{"id": t.id, "text": t.text, "due_at": t.due_at} for t in mem.open_threads()],
                           glances=[GlanceInfo(**asdict(g)) for g in mem.list_glances()])

    def _set_settings(self, patch: dict) -> None:
        old = self.settings
        try:
            new = apply_patch(old, patch)
        except ValueError as e:
            self.emit(ErrorMsg(message=str(e)))
            return
        save_settings(self.kv, new)
        self.router.online_enabled = new.online_enabled
        with self._eyes_lock:
            self.policy.cap, self.policy.min_gap_s = new.glance_daily_cap, new.glance_min_gap_s
            if any(getattr(old, k) != getattr(new, k) for k in _EYES_KEYS):
                self._eyes_gen += 1  # a look in flight was allowed under the old settings: drop it
            self._dont_look, self._eyes_on = new.dont_look, new.eyes_enabled
        if new.autostart != old.autostart:
            try:
                set_autostart(new.autostart, LAUNCHER)
            except (OSError, KeyError) as e:
                log.warning("autostart change failed: %s", e)
                self.emit(ErrorMsg(message=f"couldn't change autostart: {e}"))
        self.emit(SettingsMsg(settings=asdict(new)))
        self._emit_eyes(new)
        self._sync_expression()
