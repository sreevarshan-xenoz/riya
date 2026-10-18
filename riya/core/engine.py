"""RiyaCore: wires memory, feelings, presence, initiative, routing and consolidation into replies and speaking first."""
import logging
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from riya.core.appraise import SENTENCE_END
from riya.core.consolidate import Consolidator, remember_intent, third_person
from riya.core.feelings import Feelings
from riya.core.initiative import Action, Initiative
from riya.core.memory import Memory, about_me
from riya.core.models import LOCAL_TEMPERATURE, ChatModel
from riya.core.persona import load_traits, persona_prompt, style_examples
from riya.core.presence import Presence
from riya.core.prompt import (IDENTITY_SITUATION, bid_situation, build_messages, greeting_situation,
                              remember_situation)
from riya.core.protocol import (EXPRESSIONS, PROTOCOL_VERSION, EpisodeDelete, ErrorMsg, ExpressionMsg, GetMemories,
                                Hello, MemoriesMsg, MemoryCorrect, MemoryDelete, Say, SayEnd, SetSettings,
                                SettingsMsg, StateMsg, TalkHotkey, ThreadResolve, TranscriptDelete)
from riya.core.router import Router, needs_thinking
from riya.core.settings import Settings, apply_patch, load_settings, save_settings, set_autostart
from riya.core.shaping import (BUDGETS, CAPS, CODE_BLOCK, cap_sentences, count_emoji, drop_generic,
                               is_identity_question, limit_emoji, strip_label, turn_kind)
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
_TILDES = re.compile(r"[~\s]*")


def _worded(text: str) -> bool:
    return any(c.isalnum() for c in text)


def _query(action: Action) -> str:
    return action.thread.text if action.thread else action.reason


def _threads(action: Action) -> list:
    return [action.thread] if action.thread else []


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
                 emit: Callable[[BaseModel], None], persona: str | None = None):
        self.clock, self.local, self.online, self.emit = clock, local, online, emit
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
        self._expression = self.feelings.expression()
        self._acks = 0  # remember-that acks said so far, which picks the next one

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
            situation = " ".join(filter(None, [remember_situation(saved) if saved else None,
                                               IDENTITY_SITUATION if is_identity_question(text) else None])) or None
            lead = None
            if saved:
                lead, self._acks = REMEMBER_ACKS[self._acks % len(REMEMBER_ACKS)], self._acks + 1
            yield from self._speak("reply", user_text=text, query=text, spoken=True, situation=situation, lead=lead,
                                   small=bool(saved))

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
            situation = greeting_situation(self._away_s, self.feelings.state.missing, action.thread)
            yield from self._speak("greet", user_text=None, query=_query(action), situation=situation,
                                   spoken=action.spoken, threads=_threads(action),
                                   after=lambda outcome: self._greeted(action, outcome))

    def _greeted(self, action: Action, outcome: str) -> None:
        """The thread counts as followed up only once she actually asked; a barge-in still stops a re-raise today."""
        if action.thread and outcome != "failed":
            self.memory.mark_thread(action.thread.id, status="followed_up" if outcome == "done" else None,
                                    raised_at=self.clock.now())

    def _speak(self, kind: str, *, user_text: str | None, query: str, situation: str | None, spoken: bool,
               threads: list | None = None, after: Callable[[str], None] | None = None,
               lead: str | None = None, small: bool = False) -> Iterator[str]:
        """Streams one utterance. The outcome is "done", "closed" (the consumer stopped early) or "failed" (model).
        threads, when given, replaces the retrieved follow-ups (speaking first carries only its chosen thread).
        lead is a fixed sentence said first, before the model is even asked; the model may add only one more.
        small makes it small talk on the local model whatever the text (a remember-that turn just reacts).
        A short generic sentence of the model's (shaping.drop_generic) is dropped: not said, not recorded."""
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
            route = self.router.route(route_text)  # decided first, so the prompt carries the matching persona
            messages = build_messages(persona=self.personas[route], examples=self.examples,
                                      feelings_tag=self.feelings.prompt_tag(),
                                      retrieved=self._retrieve(query, threads), turns=self.turns,
                                      user_text=user_text, situation=situation, now=self.clock.now(), kind=turn)
            if route == "online":
                thinking = True
                self.feelings.on_event("thinking")
                self._sync_expression()
            stream = self.router.stream(messages, route_text, budget)
            kept = drop_generic(self._sentences(stream, parts, done), others=bool(lead), dropped=dropped)
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
            whole = outcome == "done" and done["stream"] and not dropped
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
    def _sentences(stream: Iterator[str], parts: list[str], done: dict) -> Iterator[str]:
        """The model's pieces as whole sentences (_cut), each with only the reply's first emoji (an emoji's leftovers
        vanish) and the first without a leading "Riya:" label. Closing this closes the model stream; done["stream"]
        is set once the model ended by itself."""
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
                        yield sentence
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

    def _bid(self, action: Action) -> None:
        try:
            messages = build_messages(persona=self.personas["local"], examples=self.examples,
                                      feelings_tag=self.feelings.prompt_tag(),
                                      retrieved=self._retrieve(_query(action), _threads(action)), turns=self.turns,
                                      user_text=None, situation=bid_situation(action.reason, action.thread),
                                      now=self.clock.now())
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
        action = self._step("greeting", lambda: self._greeting(events, now))
        if action is None:
            self._step("bid", self._maybe_bid)
        self._step("bid ignored", lambda: self._check_bid_ignored(now))
        self._step("feelings", lambda: self.feelings.tick(present=self.presence.state != "away"))
        self._step("consolidation", lambda: self._maybe_consolidate(now))
        self._step("purge", lambda: self._maybe_purge(now))
        self._step("persistence", lambda: self._maybe_save(now))
        self._step("state", lambda: self._maybe_state(now))
        self._step("expression", self._sync_expression)
        return Utterance(self.speak_first(action), action.spoken) if action else None

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
        return action

    # ---- state out ----------------------------------------------------------------------------------
    def _save_feelings(self) -> None:
        self._last_save = self.clock.now()
        self.kv.set("feelings", self.feelings.to_dict())

    def _sync_expression(self) -> None:
        name = self.feelings.expression()
        if name != self._expression:
            self._expression = name
            self.emit(ExpressionMsg(name=name))

    def _state_msg(self) -> StateMsg:
        state = asdict(self.feelings.state)
        return StateMsg(presence=self.presence.state, expression=self.feelings.expression(),
                        feelings={k: float(v) for k, v in state.items() if isinstance(v, (int, float))})

    def voice_state(self) -> tuple[SimpleNamespace, float]:
        s = self.feelings.state
        return SimpleNamespace(arousal=s.arousal, valence=s.valence), s.energy

    def hello_messages(self) -> list[BaseModel]:
        return [Hello(version=PROTOCOL_VERSION, expressions=list(EXPRESSIONS)),
                SettingsMsg(settings=asdict(self.settings)), self._state_msg()]

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
            elif isinstance(msg, SetSettings):
                self._set_settings(msg.patch)
                return
            elif isinstance(msg, TalkHotkey):
                self.talk_requested = True
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
                           threads=[{"id": t.id, "text": t.text, "due_at": t.due_at} for t in mem.open_threads()])

    def _set_settings(self, patch: dict) -> None:
        old = self.settings
        try:
            new = apply_patch(old, patch)
        except ValueError as e:
            self.emit(ErrorMsg(message=str(e)))
            return
        save_settings(self.kv, new)
        self.router.online_enabled = new.online_enabled
        if new.autostart != old.autostart:
            try:
                set_autostart(new.autostart, LAUNCHER)
            except (OSError, KeyError) as e:
                log.warning("autostart change failed: %s", e)
                self.emit(ErrorMsg(message=f"couldn't change autostart: {e}"))
        self.emit(SettingsMsg(settings=asdict(new)))
