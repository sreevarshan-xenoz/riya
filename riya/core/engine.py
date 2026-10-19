"""RiyaCore: wires memory, feelings, presence, initiative, routing and consolidation into replies and speaking first."""
import logging
import os
import queue
import random
import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel

from riya.agents.hub import AgentEvent, AgentHub, summarize
from riya.core import games
from riya.core.appraise import SENTENCE_END
from riya.core.consolidate import Consolidator, remember_intent, third_person
from riya.core.eyes.capture import FakeWindowSource, WindowInfo, WindowSource, clamp_rect, grab_jpeg, thumb
from riya.core.eyes.policy import GlancePolicy, window_refusal
from riya.core.eyes.see import describe, is_private, is_screen_question, vision_messages
from riya.core.feelings import Feelings
from riya.core.hands import online as hands_online
from riya.core.hands.intent import is_command_form, is_forbidden, match
from riya.core.hands.policy import Confirmations, Pending, confirm_text, decide, yes_no
from riya.core.hands.records import HandsDB
from riya.core.hands.reminders import delivery, gap_ok, reminder_line
from riya.core.hands.tools import ONLINE_OFF, REGISTRY, TIERS, Call, ToolCtx, ToolResult, _q, validate_call
from riya.core.initiative import Action, Initiative
from riya.core.memory import Memory, about_me
from riya.core.models import LOCAL_TEMPERATURE, ChatModel
from riya.core.persona import load_traits, persona_prompt, style_examples
from riya.core.presence import Presence
from riya.core.prompt import (FEELINGS_SITUATION, IDENTITY_SITUATION, SEEING_SITUATION, agents_block, bid_situation,
                              build_messages, cant_see_situation, eyes_block, greeting_reports, greeting_situation,
                              hands_situation, remember_situation, reminders_block, report_bubble, report_line,
                              report_situation, tool_turn)
from riya.core.protocol import (EXPRESSIONS, PROTOCOL_VERSION, ActionInfo, ActionsClear, ActionsMsg, AgentInfo, AgentsMsg,
                                Confirm, ConfirmDone, ConfirmRequest, DontLook, EpisodeDelete, ErrorMsg, ExpressionMsg,
                                EyesMsg, GetMemories, GlanceDelete, GlanceInfo, Hello, Look, MemoriesMsg, MemoryCorrect,
                                MemoryDelete, NoteDelete, NoteInfo, QuitCore, ReminderCancel, ReminderInfo,
                                RemindersMsg, Say, SayEnd, SetSettings, SettingsMsg, StateMsg, TalkHotkey, ThreadResolve,
                                TranscriptDelete)
from riya.core.router import Router, needs_thinking
from riya.core.settings import Settings, apply_patch, in_quiet_hours, load_settings, save_settings, set_autostart
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
GAME_IDLE_S = 600  # a game with no turn this long ends, silently
BUSY = frozenset({"busy", "running"})  # an agent at work: her bids may offer a game meanwhile
HOOKS_SEEN_KEY = "agents.hooks_seen_at"
HOOKS_SEEN_EVERY_S = 60  # hooks_seen_at (KV and AgentsMsg) moves at most this often: a tool-call flood isn't news
REPORTS_KEY = "agents.reports"  # pending reports survive a restart
_EYES_KEYS = ("eyes_enabled", "dont_look", "online_enabled", "eyes_deny_titles", "eyes_deny_processes")
# her hands: every line here is a template (the model never words a refusal or a confirmation)
FORBIDDEN = "forbidden"  # a T3 request: there's no tool for it, only the kind refusal
FORBIDDEN_LINE = ("Sorry, that's one thing I never do: no deleting or moving files, no messages or email, no buying "
                  "things, nothing with passwords and no shell. I can help another way, though~")
INVALID_LINE = "I couldn't do that one: something about it didn't check out."
AFTER_GAME = "I'll ask you about that after your game~"
CONFIRM_CANCELLED = "Okay, I'll leave that one, then."
CONFIRM_DECLINED = "Okay, I won't."
CONFIRM_TIMED_OUT = "That one timed out, so I didn't do it. Ask me again?"
COMMAND_BUSY = "Something's still running, so I'll wait until it's done."
TOOL_FAILED = "Something went wrong when I tried that."
LOOKUP_MISS = ("You tried to look this up online and found nothing useful. Answer from what you know, and say so "
               "if you're not sure.")
REMINDER_TOOLS = frozenset({"timer_set", "reminder_set", "reminder_cancel"})
_BUTTON = {"no": "declined", "expired": "expired", "replaced": "replaced"}  # a yes reports what then happened
_NOT_RUN = frozenset({"declined", "expired", "cancelled", "replaced"})  # confirm outcomes logged as themselves
FRIENDLY = {"time_now": "telling the time", "timer_set": "timers", "reminder_set": "reminders",
            "reminder_list": "reminders", "reminder_cancel": "reminders", "note_add": "notes", "note_list": "notes",
            "calc": "the calculator", "weather": "the weather", "lookup": "looking things up",
            "open_url": "opening websites", "open_app": "opening apps", "media": "media keys", "volume": "volume",
            "clipboard_copy": "the clipboard", "run_command": "running your commands", "write_file": "writing files"}
_TILDES = re.compile(r"[~\s]*")
_ENDED = re.compile(r"[.!?…~][\"')\]]?[\s~]*$")  # a sentence's end, at the end of the text


def local_time(t: float, tz=None) -> datetime:
    """t as the wall-clock time in tz (quiet hours are local; None: the machine's zone)."""
    return datetime.fromtimestamp(t, tz)


def default_riya_dir() -> Path:
    """Her own folder, the only place write_file writes: Documents/Riya under %USERPROFILE%."""
    return Path(os.environ.get("USERPROFILE") or Path.home()) / "Documents" / "Riya"


def refusal(tool: str) -> str:
    return f"\"{FRIENDLY.get(tool, tool)}\" is switched off in my settings, so I can't do that."


def _worded(text: str) -> bool:
    return any(c.isalnum() for c in text)


def _query(action: Action) -> str:
    return action.thread.text if action.thread else action.reason


def _threads(action: Action) -> list:
    return [action.thread] if action.thread else []


def _priority(text: str) -> bool:
    """The spec's priority intents, which always pre-empt her hands: a screen question, a remember-that, an identity,
    about-me or feelings question."""
    return (is_screen_question(text) or remember_intent(text) is not None or is_identity_question(text)
            or about_me(text) or is_feelings_question(text))


def _chat_mid_game(text: str) -> bool:
    """Never a game move, in any game: a priority intent or a mention of their agents ("how's my agent doing?")."""
    return _priority(text) or games.about_agents(text)


def _thumb(jpeg: bytes) -> bytes:
    try:
        return thumb(jpeg)
    except Exception:  # an image PIL can't read: "unknown", so the refresh check never fires on it
        return b""


def _key(info: WindowInfo) -> tuple[int, str]:
    return info.hwnd, info.title


class Utterance:
    """What tick() hands over when she speaks first: iterate it for the sentences; spoken says if it's meant aloud.
    on_close runs once on close(), even for one closed before its first step (it releases what tick claimed)."""

    def __init__(self, sentences: Iterator[str], spoken: bool, on_close: Callable[[], None] | None = None):
        self._sentences, self.spoken, self._on_close = sentences, spoken, on_close

    def __iter__(self) -> "Utterance":
        return self

    def __next__(self) -> str:
        return next(self._sentences)

    def close(self) -> None:
        try:
            self._sentences.close()
        finally:
            done, self._on_close = self._on_close, None
            if done is not None:
                done()


class RiyaCore:
    def __init__(self, *, db, embedder, clock, idle_source, local: ChatModel, online: ChatModel | None,
                 emit: Callable[[BaseModel], None], persona: str | None = None, vision: ChatModel | None = None,
                 window_source: WindowSource | None = None, grab: Callable[[WindowInfo], bytes] = grab_jpeg,
                 body_present: Callable[[], bool] = lambda: True, effects=None, http=None,
                 riya_dir: str | Path | None = None, tz=None):
        """vision: the online model instance her eyes use (glances and screen questions); None means she can't see.
        window_source and grab: the foreground window and its pixels (JPEG bytes, in memory only). body_present: a
        body is connected (her eyes stay shut without one). effects and http: her hands on the OS and the network
        (None: she has none, and every such tool fails kindly; the runner passes the real ones). riya_dir: the only
        folder write_file writes into. tz: the zone for times she parses and for quiet hours (None: the machine's)."""
        self.clock, self.local, self.online, self.emit = clock, local, online, emit
        self.effects, self.http, self.tz = effects, http, tz
        self.riya_dir = Path(riya_dir) if riya_dir is not None else default_riya_dir()
        self.body_present, self.quit_requested = body_present, threading.Event()  # quit_requested: the tray's Quit
        self.vision, self.windows, self._grab = vision, window_source or FakeWindowSource(None), grab
        # the small local model gets the short persona, the online one the full sheet; persona= overrides both
        self.personas = {k: persona if persona is not None else persona_prompt(k) for k in ("local", "online")}
        self.examples = style_examples()
        self.memory = Memory(db, embedder, clock)
        self.kv = KV(db, self.memory.lock)
        self.hands = HandsDB(db, self.memory.lock)
        # one pending confirmation (and one parked call), taken by the user's turn or a body button: _hands_lock guards
        # them and the command worker's start; it's taken after the core lock and before _eyes_lock and memory
        self._hands_lock = threading.RLock()
        self.confirms = Confirmations(clock)
        self._results: queue.SimpleQueue = queue.SimpleQueue()  # (summary, ToolResult): _results_step says them
        self._command_thread: threading.Thread | None = None  # the riya-command worker: one command at a time
        # reminder ids an Utterance tick returned is about to say (skipped until it ends), and the ones the came-back
        # greeting about to be said carries in its tail; both under _hands_lock
        self._reminding: set[int] = set()
        self._greet_reminders: list = []
        self._last_remind_at: float | None = None  # the last spoken reminder line (REMIND_GAP_S apart)
        traits, saved = load_traits(), self.kv.get("feelings")
        self.feelings = Feelings.from_dict(saved, clock, traits) if saved else Feelings(clock, traits)
        self.presence = Presence(idle_source, clock, self.kv)
        self.initiative = Initiative(clock, lambda: self.settings, tz)
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
        self._reports = self._load_reports()  # pending, per agent only the newest: {agent_id: (event, at)}
        self._reports_saved = self._report_rows()
        self._greet_reports: list[tuple[AgentEvent, float]] = []  # folded into the greeting about to be said
        self._last_report_at: float | None = None
        seen = self.kv.get(HOOKS_SEEN_KEY)
        self._hooks_seen_at = float(seen) if isinstance(seen, (int, float)) else None
        self.receiver = None  # the HookReceiver (ok, rejected.recent()), set by the runner; read-only here
        self._agents_sent = self._agents_msg()
        # games: the state (the 20 questions secret included) lives here, never in the transcript or her prompt history
        self.game: games.GameState | None = None
        self._game_at = now  # the last game turn
        self._offer: tuple[str, float] | None = None  # (game, when) a bid offered while an agent was busy
        self.game_rng = random.Random()
        self._expression = self._face()

    @property
    def settings(self) -> Settings:
        return load_settings(self.kv)

    def _consolidation_model(self) -> ChatModel:
        return self.online if self.settings.online_consolidation and self.online else self.local

    # ---- speaking -----------------------------------------------------------------------------------
    def reply(self, text: str, source: Literal["typed", "voice"] = "typed") -> Iterator[str]:
        """The user's turn, in priority order: a pending confirmation takes it first (even mid-game); then the
        priority intents (screen, remember-that, identity, about-me, feelings; mid-game also a mention of their agents)
        never act; then the game; then a T3 refusal, a matched call, or the online model's proposal; else a normal
        reply. Only this turn's own text ever reaches match, propose or a confirmation."""
        with self._lock:
            taken = None
            try:
                taken = self._confirm_turn(text)
            except Exception:
                log.exception("couldn't take the confirmation turn")
            if taken is not None and not (yield from self._confirm_answer(text, *taken)):
                return
            chat, call = (_chat_mid_game if self.game is not None else _priority)(text), None
            try:
                if not chat:
                    call = FORBIDDEN if is_forbidden(text) else self._match(text)
                if (isinstance(call, Call) and call.tool == "lookup" and not self.settings.online_enabled
                        and not is_command_form(text)):
                    call = None  # online off: a question she may answer from what she knows (she can't look it up)
            except Exception:
                log.exception("couldn't read the turn for a tool")
            move = None
            try:
                move = self._game_move(text, acting=call is not None)
            except Exception:
                log.exception("couldn't play the game turn")
            if move is not None and move[1].kind != "off":
                yield from self._game_reply(text, *move)
                return
            if call is None and not chat:
                call = self._propose(text)
            note = None
            if call is not None:
                note = yield from self._hands_turn(text, call)
                if note is None:
                    return
            saved = None
            try:
                saved = self._take_user_turn(text)
            except Exception:
                log.exception("couldn't record the user's turn")
            base = [note, remember_situation(saved) if saved else None,
                    IDENTITY_SITUATION if is_identity_question(text) else None,
                    FEELINGS_SITUATION if is_feelings_question(text) else None,
                    games.mid_game_note(move[0]) if move else None]  # chat mid-game: the game stays on
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

    def _game_move(self, text: str, acting: bool = False) -> tuple[games.GameState, games.Verdict] | None:
        """The game side of a user turn, judged by the engine before the model is asked: a stop phrase ends the
        active game; a start request (or a yes to a fresh offer, with no game on) starts one, and mid-game only a
        request for a different game switches; otherwise an active game plays the turn (an "off" verdict: chat, a
        normal reply; _chat_mid_game turns are always off). acting (match or the T3 check recognised the turn): it's
        off too, so "pause the music" mid-game pauses the music, except a one-word turn in word chain, always a move.
        None when no game is involved. Any turn uses up a pending offer."""
        now = self.clock.now()
        offer, self._offer = self._offer, None
        offered = offer[0] if offer and now - offer[1] < BID_IGNORED_S else None
        g = self.game
        if g is not None and games.is_stop(text):
            self.game = None
            return g, games.stop(g, text)
        kind = games.detect_start(text, offered if g is None else None)
        if kind and (g is None or kind != g.kind):
            g = self.game = games.new_game(kind, self.game_rng)
            self._game_at = now
            return g, games.opening(g)
        if g is None:
            return None
        word = g.kind == "word_chain" and len(text.split()) == 1
        v = games.off(g) if _chat_mid_game(text) or (acting and not word) else games.play(g, text)
        self._game_at = now
        if v.over:
            self.game = None
        return g, v

    def _game_reply(self, text: str, g: games.GameState, v: games.Verdict) -> Iterator[str]:
        """A game turn: the local model voices the verdict in one short sentence (the situation is never kept), every
        sentence passes the secret guard, and the engine's own line (her word, the next riddle, the reveal) follows
        unless she already said it."""
        try:
            self._take_user_turn(text, remember=False, consolidate=False)
        except Exception:
            log.exception("couldn't record the user's turn")
        game = SimpleNamespace(guard=lambda sentence: games.guard(g, sentence), say=v.say, key=v.key, told=False)
        yield from self._speak("reply", user_text=text, query=text, spoken=True,
                               situation=games.game_situation(g, v), game=game)

    def _take_user_turn(self, text: str, remember: bool = True, consolidate: bool = True) -> str | None:
        """Records the user's turn; returns the fact text when it was a remember-that turn (saved right away). A game
        turn (remember and consolidate off) never saves a fact and never reaches consolidation."""
        now = self.clock.now()
        answered_bid = self.pending_bid_at is not None and now - self.pending_bid_at < BID_IGNORED_S
        if answered_bid and self._bid_thread is not None:
            self.memory.mark_thread(self._bid_thread, status="followed_up")
        self.last_turn_at, self.pending_bid_at, self._bid_thread = now, None, None
        self.memory.append_transcript("user", text, consolidate=consolidate)
        saved = None
        if remember and (said := remember_intent(text)) is not None:
            saved = f"The user said: {third_person(said)}"
            self.memory.add_fact(saved, source="user_said")
        self.feelings.on_user_text(text)
        self.presence.note_turn()
        self._sync_expression()
        return saved

    def speak_first(self, action: Action) -> Iterator[str]:
        """The greeting. Reminders that fell due while they were away (_greet_reminders) are its templated tail, said
        even when the model fails, and marked fired once it went out (due() and mark_fired() both here, under the
        core lock); the claim on them ends with the greeting."""
        with self._lock:
            with self._hands_lock:
                greet, self._greet_reminders = self._greet_reminders, []
            ids = {r.id for r in greet}
            try:
                reports, self._greet_reports = self._greet_reports, []
                situation = greeting_situation(self._away_s, self.feelings.state.missing, action.thread)
                if reports:  # what her agents did while they were away
                    situation += " " + greeting_reports([ev for ev, _ in reports])
                now = self.clock.now()
                due = [r for r in self.hands.due(now) if r.id in ids] if ids else []  # not cancelled meanwhile
                tail = reminder_line(due, now, self.tz) or None
                said: list[str] = []

                def after(outcome: str) -> None:
                    if due:  # fired first: no bookkeeping failure can make the tail go out again
                        self._last_remind_at = self.clock.now()  # the tail counts for the reminder spacing
                        self._fire([r.id for r in due], self._last_remind_at)
                    mine = [x for x in said if x != tail]  # the reminder tail isn't the greeting itself
                    self._greeted(action, outcome if mine or not tail else "failed", reports, bool(mine))
                yield from self._counted(self._speak(
                    "greet", user_text=None, query=_query(action), situation=situation, spoken=action.spoken,
                    threads=_threads(action), after=after, consolidate=not (reports or tail), tail=tail), said)
            finally:
                self._release(ids)

    @staticmethod
    def _counted(sentences: Iterator[str], said: list[str]) -> Iterator[str]:
        """sentences, each one also appended to said (so an after-callback knows whether anything was said)."""
        try:
            for sentence in sentences:
                said.append(sentence)
                yield sentence
        finally:
            sentences.close()

    def _greeted(self, action: Action, outcome: str, reports: list | None = None, said: bool = True) -> None:
        """The thread counts as followed up only once she actually asked; a barge-in still stops a re-raise today.
        Reports folded into a greeting that failed or said nothing go back to pending (unless newer news replaced
        them)."""
        if outcome == "failed" or not said:
            for ev, at in reports or []:
                self._reports.setdefault(ev.agent_id, (ev, at))
        if action.thread and outcome != "failed":
            self.memory.mark_thread(action.thread.id, status="followed_up" if outcome == "done" else None,
                                    raised_at=self.clock.now())

    def _speak(self, kind: str, *, user_text: str | None, query: str, situation: str | None, spoken: bool,
               threads: list | None = None, after: Callable[[str], None] | None = None,
               lead: str | None = None, small: bool = False, seeing: SimpleNamespace | None = None,
               game: SimpleNamespace | None = None, consolidate: bool = True, shape: str | None = None,
               tail: str | None = None, tail_if_silent: bool = False) -> Iterator[str]:
        """Streams one utterance. The outcome is "done", "closed" (the consumer stopped early) or "failed" (model).
        threads, when given, replaces the retrieved follow-ups (speaking first carries only its chosen thread).
        lead is a fixed sentence said first, before the model is even asked; the model may add only one more.
        small makes it small talk on the local model whatever the text (a remember-that turn just reacts).
        seeing (from _look_now, with an image) sends the image with her full prompt to the vision model; the same
        shaping applies. A short generic sentence of the model's (shaping.drop_generic) is dropped: not said, not
        recorded. game (a game turn): the local model, the small budget and one sentence; every sentence passes
        game.guard, and game.say (the engine's line: her word, the next riddle, the reveal) follows unless game.key
        was already said. The game state already moved, so that line goes out even when the model failed or said
        nothing, and on a barge-in it's still shown (not spoken) and recorded. A report, a game turn or consolidate
        off (a greeting carrying reports) is kept in the transcript but never reaches consolidation.
        shape="tool": her line about a tool's result, on the local model only, with the tool budget and cap and no
        reply-rules line. tail (never with game): one engine line after the model's sentences, said like the game's
        (even when the model failed; shown, not spoken, on a barge-in); tail_if_silent: only when the model said
        nothing, in place of MODEL_DOWN (a tool's ToolResult.text is the fallback for her line)."""
        uid, parts, said, thinking, outcome = uuid4().hex, [], [], False, "failed"
        sentences, done, dropped, model_said, told = None, {"stream": False}, [], [], {"tail": False}
        tool = shape == "tool"

        def engine_line() -> str | None:
            """The engine's line, at most once: the game's (unless she already said its key), or the tail."""
            if game:
                if not game.say or game.told or games.mentions(" ".join(said), game.key):
                    return None
                game.told = True
                return game.guard(game.say)
            if not tail or told["tail"] or (tail_if_silent and model_said):
                return None
            told["tail"] = True
            return tail
        try:
            self.router.online_enabled = self.settings.online_enabled
            spoken = spoken and not self.settings.mute_voice
            if lead:
                self.emit(Say(utterance_id=uid, text=lead, spoken=spoken, kind=kind))
                said.append(lead)
                yield lead
            route_text = "" if small or game or tool else user_text or ""
            turn, budget, cap = (("game", BUDGETS["small"], 1) if game else (None, BUDGETS["tool"], CAPS["tool"])
                                 if tool else self._shape(kind, route_text))
            # decided first, so the prompt carries the matching persona; a look at the screen is always online
            route = "online" if seeing else self.router.route(route_text)
            retrieved, eyes, agents = self._retrieve(query, threads), self._eyes_context(), self._agents_context()
            reminders = self._reminders_context()

            def build(persona: str, situation: str | None) -> list[dict]:
                return build_messages(persona=persona, examples=self.examples, feelings_tag=self.feelings.prompt_tag(),
                                      retrieved=retrieved, turns=self.turns, user_text=user_text, situation=situation,
                                      now=self.clock.now(), kind=turn, eyes=eyes, agents=agents, reminders=reminders)

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
                if game:
                    sentence = game.guard(sentence)
                self.emit(Say(utterance_id=uid, text=sentence, spoken=spoken, kind=kind))
                said.append(sentence)
                model_said.append(sentence)
                yield sentence
            if line := engine_line():
                self.emit(Say(utterance_id=uid, text=line, spoken=spoken, kind=kind))
                said.append(line)
                yield line
            outcome = "done"
        except GeneratorExit:
            outcome = "closed"
            raise
        except Exception:
            log.exception("model failed during a %s", kind)
            if not (tail_if_silent and tail and not model_said):  # else the tail stands in for her line
                self.emit(ErrorMsg(message=MODEL_DOWN))
            if line := engine_line():  # the game moved on (her word, the riddle, the reveal), or the tail
                self.emit(Say(utterance_id=uid, text=line, spoken=spoken, kind=kind))
                said.append(line)
                outcome = "done"
                yield line
        finally:
            if sentences is not None:
                sentences.close()  # stops the model stream however the utterance ended
            if outcome == "closed" and (line := engine_line()):  # barged in on: shown, not spoken, and kept
                self.emit(Say(utterance_id=uid, text=line, spoken=False, kind=kind))
                said.append(line)
            # the model finished on its own and nothing of it was dropped: keep its own spacing
            whole = (outcome == "done" and done["stream"] and not dropped and not done.get("cut") and not game
                     and not told["tail"])
            text = None if outcome == "failed" else (
                " ".join(filter(None, [lead, limit_emoji(strip_label("".join(parts)).strip())])) if whole
                else " ".join(said))
            try:
                self._end_thinking(thinking)
                self.emit(SayEnd(utterance_id=uid))
                self._finish_turn(user_text, text, consolidate=consolidate and not game and kind != "report",
                                  quoted=tool)
            except Exception:
                log.exception("couldn't record the %s", kind)
            if after:  # even when the recording failed (a greeting's after marks its reminders fired)
                try:
                    after(outcome)
                except Exception:
                    log.exception("couldn't finish the %s", kind)

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

    def _finish_turn(self, user_text: str | None, said: str | None, consolidate: bool = True,
                     quoted: bool = False) -> None:
        """said is None when the model failed: the user's turn stays in the transcript, her answer isn't recorded.
        quoted (her line about a tool's result, which may carry web or command text): later prompts see it only
        quoted, with the untrusted note (prompt.tool_turn); the transcript keeps what was shown."""
        if said is not None:
            if said:
                self.memory.append_transcript("assistant", said, consolidate=consolidate)
            new = ([] if user_text is None else [{"role": "user", "content": user_text}]) + (
                [{"role": "assistant", "content": tool_turn(said) if quoted else said}] if said else [])
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
        """A silent bid. While an agent is busy (and no game is on) it offers a random game, named by the engine if
        the model's line doesn't; a yes within BID_IGNORED_S starts it."""
        offer = None
        try:
            if self.game is None and any(a["status"] in BUSY for a in self.agents.snapshot()):
                offer = self.game_rng.choice(games.GAMES)
            situation = bid_situation(self._bid_reason(action), action.thread)
            if offer:
                situation += " " + games.offer_situation(offer)
            messages = build_messages(persona=self.personas["local"], examples=self.examples,
                                      feelings_tag=self.feelings.prompt_tag(),
                                      retrieved=self._retrieve(_query(action), _threads(action)), turns=self.turns,
                                      user_text=None, situation=situation,
                                      now=self.clock.now(), eyes=self._eyes_context(), agents=self._agents_context(),
                                      reminders=self._reminders_context())
            line = "".join(self.local.stream(messages, BUDGETS["bid"], LOCAL_TEMPERATURE))
            line = limit_emoji(strip_label(line).strip())
        except Exception:
            log.exception("model failed during a bid")
            return
        if not line:
            return
        if offer and not games.offers(line, offer):
            line = f"{line} {games.offer_line(offer)}"
        self._offer = (offer, self.clock.now()) if offer else None
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

    # ---- hands --------------------------------------------------------------------------------------
    def _match(self, text: str) -> Call | None:
        s = self.settings
        names = lambda items: [i["name"] for i in items if isinstance(i, dict) and isinstance(i.get("name"), str)]  # noqa: E731
        return match(text, self.clock.now(), apps=names(s.apps), commands=names(s.commands), tz=self.tz)

    def _propose(self, text: str) -> Call | None:
        """The online model's one call for a clear command (None when it, or her hands online, are off). Only the
        turn, the time and the names of their apps and commands are sent; propose never raises."""
        s = self.settings
        model = self.online if s.online_enabled and s.hands_online else None
        return hands_online.propose(model, text, self.clock.now(), s, self.tz)

    def _confirm_turn(self, text: str) -> tuple[str, Pending, bool] | None:
        """The user's turn against a pending confirmation: (answer, pending, it was still pending), or None."""
        with self._hands_lock:
            had = self.confirms.pending is not None
            taken = self.confirms.take_turn(text)
        return None if taken is None else (*taken, had)

    def _confirm_answer(self, text: str, answer: str, p: Pending, had: bool) -> Iterator[str]:
        """Answers a confirmation from the user's turn; returns True when the turn goes on to its normal reply (it
        wasn't an answer: cancelled with a one-line note, or a pending one that had expired)."""
        if answer == "cancelled" or (answer == "expired" and yes_no(text) is None):
            self._confirm_done(p, answer)
            if answer == "cancelled":
                yield from self._say_line(CONFIRM_CANCELLED)
            return True
        try:
            self._take_user_turn(text, remember=False, consolidate=False)
        except Exception:
            log.exception("couldn't record the user's turn")
        if answer == "yes":
            yield from self._run_call(text, p.call, p.summary, confirmed=p)
        elif answer == "no":
            self._confirm_done(p, "declined")
            yield from self._say_line(CONFIRM_DECLINED, user_text=text)
        else:  # a yes or no too late: expired (the tick already told the body if it had been expired there)
            if had:
                self._confirm_done(p, "expired")
            yield from self._say_line(CONFIRM_TIMED_OUT, user_text=text)
        return False

    def _confirm_done(self, p: Pending, outcome: str) -> None:
        """confirm_done for the body. An answer that ran nothing is logged here; a run (ran, started, failed) is
        logged where it happens."""
        self.emit(ConfirmDone(id=p.id, outcome=outcome))
        if outcome in _NOT_RUN:
            self._log_action(p.call.tool, TIERS.get(p.call.tool, 3), p.summary, outcome)

    def _confirm_button(self, id: str, yes: bool) -> None:
        """A body Yes/No button. Only the exact pending id counts; a stale one gets "replaced" (or "expired"). A yes
        runs the snapshot the user was asked about, outside _hands_lock (a slow effect stalls neither the socket loop
        nor the tick), and its result is queued for tick, never spoken from here."""
        with self._hands_lock:
            current = self.confirms.pending
            answer, p = self.confirms.take_button(id, yes)
            if p is None or answer != "yes":
                self.emit(ConfirmDone(id=p.id if p else str(id), outcome=_BUTTON[answer]))
                if p is not None and (answer == "no" or (current is not None and current.id == p.id)):
                    self._log_action(p.call.tool, TIERS.get(p.call.tool, 3), p.summary, _BUTTON[answer])
                return  # (an expiry the tick saw was logged there)
        _, result = self._execute(p.call, p.summary, confirmed=p)  # taken: no turn or tick can answer it now
        if result is not None:  # a command that started reports through its worker
            self._results.put((p.summary, result))

    def _hands_turn(self, text: str, call: Call | str) -> Iterator[str]:
        """A turn that asks her to act. Every call (matched or proposed) is validated against the settings first;
        then the policy refuses, asks or runs. The turn never reaches consolidation. With online access off a network
        tool makes no request: a plain templated line. A lookup that finds nothing (or fails) returns LOOKUP_MISS
        before the turn is recorded: the turn goes on to the normal reply with that situation."""
        s = self.settings
        checked = verdict = found = None
        if call != FORBIDDEN:
            checked = validate_call(call, s)
            verdict = decide(checked, s)
            if verdict == "run" and checked.tool == "lookup" and s.online_enabled:
                found = self._run_tool(checked, self._summary(checked))
                if not found.ok:
                    return LOOKUP_MISS
        try:
            self._take_user_turn(text, remember=False, consolidate=False)
        except Exception:
            log.exception("couldn't record the user's turn")
        if call == FORBIDDEN:
            self._log_action(FORBIDDEN, 3, summarize(text, 80), "refused")
            yield from self._say_line(FORBIDDEN_LINE, user_text=text)
            return
        if verdict == "refuse":
            self._log_action(call.tool, TIERS.get(call.tool, 3), self._summary(checked or call), "refused")
            yield from self._say_line(INVALID_LINE if checked is None else refusal(call.tool), user_text=text)
            return
        summary = self._summary(checked)
        if checked.tool in ONLINE_OFF and not s.online_enabled:
            self._log_action(checked.tool, TIERS[checked.tool], summary, "refused")
            yield from self._say_line(ONLINE_OFF[checked.tool], user_text=text)
            return
        if verdict == "run":
            yield from self._run_call(text, checked, summary, found=found)
            return
        gaming = self.presence.state == "gaming"
        if gaming and TIERS[checked.tool] >= 2:  # asked once the game ends (_park_step raises it)
            with self._hands_lock:
                self.confirms.park(checked, summary)
            yield from self._say_line(AFTER_GAME, user_text=text)
            return
        with self._hands_lock:
            p = self.confirms.ask(checked, summary)
        yield from self._raise_confirm(p, not (gaming or self._quiet(self.clock.now())), user_text=text)

    def _raise_confirm(self, p: Pending, spoken: bool, *, user_text: str | None = None,
                       kind: str = "reply") -> Iterator[str]:
        """confirm_request, then the templated question as a Say (spoken only when spoken; yielded only then)."""
        self.emit(ConfirmRequest(id=p.id, summary=p.summary, expires_at=p.expires_at))
        yield from self._say_line(confirm_text(p), user_text=user_text, spoken=spoken, kind=kind)

    def _run_call(self, text: str, call: Call, summary: str, confirmed: Pending | None = None,
                  found: ToolResult | None = None) -> Iterator[str]:
        """Runs a validated call for the user's turn: a command goes to the worker ("Running ..." at once, the result
        later through tick); anything else runs here and she says how it went (the result text is the fallback).
        found: the call already ran (a lookup that found something), so only her line about it is left."""
        how, result = ("ran", found) if found is not None else self._execute(call, summary, confirmed)
        if how == "started":
            yield from self._say_line(f"Running {_q(call.args['name'], 40)}~", user_text=text)
        elif how == "refused":
            yield from self._say_line(result.text, user_text=text)
        else:
            yield from self._speak("reply", user_text=text, query=text, spoken=True, consolidate=False, shape="tool",
                                   situation=hands_situation(summary, result), tail=result.text, tail_if_silent=True)

    def _execute(self, call: Call, summary: str,
                 confirmed: Pending | None = None) -> tuple[str, ToolResult | None]:
        """Executes a validated call: ("started", None) when a command went to the worker, ("refused", a templated
        result) when it can't run now (its tool was switched off since she asked, or a command is still running),
        else ("ran", the result). A confirmed call's confirm_done says what actually happened."""
        if confirmed is not None and not self.settings.tools_enabled.get(call.tool, False):
            how, result = "refused", ToolResult(False, refusal(call.tool))
        elif call.tool == "run_command":
            started = self._start_command(call, summary)
            how, result = ("started", None) if started else ("refused", ToolResult(False, COMMAND_BUSY))
        else:
            how, result = "ran", self._run_tool(call, summary)
        if how == "refused":
            self._log_action(call.tool, TIERS[call.tool], summary, "refused")
        if confirmed is not None:
            ok = how == "started" or (how == "ran" and result.ok)
            self._confirm_done(confirmed, ("started" if how == "started" else "ran") if ok else "failed")
        return how, result

    def _run_tool(self, call: Call, summary: str) -> ToolResult:
        """Runs one validated, non-command call now; logs it, saves a first weather place, sends the reminders."""
        try:
            result = REGISTRY[call.tool].run(call.args, self._ctx())
        except Exception:
            log.exception("the %s tool failed", call.tool)
            result = ToolResult(False, TOOL_FAILED)
        self._log_action(call.tool, TIERS[call.tool], summary, "ran" if result.ok else "failed")
        place = (result.data or {}).get("save_place")
        if isinstance(place, str) and place.strip():
            try:
                self._set_settings({"weather_place": place})
            except Exception:
                log.exception("couldn't save the weather place")
        if call.tool in REMINDER_TOOLS:
            self._emit_reminders()
        return result

    def _start_command(self, call: Call, summary: str) -> bool:
        """Starts the riya-command worker on the snapshot; False while another command is still running."""
        with self._hands_lock:
            if self._command_thread is not None and self._command_thread.is_alive():
                return False
            self._command_thread = threading.Thread(target=self._command_worker, args=(call, summary),
                                                    name="riya-command", daemon=True)
            self._command_thread.start()
            return True

    def _command_worker(self, call: Call, summary: str) -> None:
        """Off the core lock: the run owns effects.run and the redacted output tail; the result waits for tick."""
        try:
            result = REGISTRY["run_command"].run(call.args, self._ctx())
        except Exception:
            log.exception("the command worker failed")
            result = ToolResult(False, f"{_q(str(call.args.get('name', '')), 40)} couldn't start.")
        self._log_action("run_command", 2, summary, "ran" if result.ok else "failed")
        self._results.put((summary, result))

    @staticmethod
    def _summary(call: Call) -> str:
        try:
            return REGISTRY[call.tool].summary(call.args)
        except Exception:
            return str(call.tool)

    def _ctx(self) -> ToolCtx:
        return ToolCtx(settings=self.settings, db=self.hands, effects=self.effects, http=self.http, clock=self.clock,
                       riya_dir=self.riya_dir, tz=self.tz)

    def _quiet(self, now: float) -> bool:
        return in_quiet_hours(self.settings.quiet_hours, local_time(now, self.tz))

    def _say_line(self, line: str, *, user_text: str | None = None, spoken: bool = True,
                  kind: str = "reply") -> Iterator[str]:
        """One templated line of hers (the model never words it), kept out of consolidation; yielded only if
        spoken."""
        spoken = spoken and not self.settings.mute_voice
        uid = uuid4().hex
        self.emit(Say(utterance_id=uid, text=line, spoken=spoken, kind=kind))
        self.emit(SayEnd(utterance_id=uid))
        try:
            self._finish_turn(user_text, line, consolidate=False)
        except Exception:
            log.exception("couldn't record her line")
        if spoken:
            yield line

    def _log_action(self, tool: str, tier: int, summary: str, outcome: str) -> None:
        """Logs an action (outcome from records.OUTCOMES) and sends the log; never raises."""
        try:
            self.hands.log_action(self.clock.now(), tool, tier, summary, outcome)
            self.emit(self._actions_msg())
        except Exception:
            log.exception("couldn't log the action")

    def _actions_msg(self) -> ActionsMsg:
        return ActionsMsg(items=[ActionInfo(**asdict(a)) for a in self.hands.actions()])

    def _reminders_msg(self) -> RemindersMsg:
        return RemindersMsg(items=[ReminderInfo(id=r.id, kind=r.kind, due_at=r.due_at, text=r.text)
                                   for r in self.hands.active()])

    def _reminders_context(self) -> str | None:
        """The `### Your reminders:` prompt line (the next 3, outside the memory block)."""
        return reminders_block(self.hands.active(), self.clock.now(), self.tz)

    def _emit_reminders(self) -> None:
        try:
            self.emit(self._reminders_msg())
        except Exception:
            log.exception("couldn't send the reminders")

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
        self._step("confirm", self._confirm_step)
        action = self._step("greeting", lambda: self._greeting(events, now))
        out = None
        if action is None:  # a greeting tick runs none of these; otherwise the first that delivers ends the chain
            for name, fn in (("reminders", self._reminder_step), ("results", self._results_step),
                             ("park", self._park_step), ("report", self._report_step)):
                if (out := self._step(name, lambda: fn(now))) is not None:
                    break
            else:
                self._step("bid", self._maybe_bid)
        self._step("reports", self._save_reports)
        self._step("bid ignored", lambda: self._check_bid_ignored(now))
        self._step("game", lambda: self._end_idle_game(now))
        self._step("eyes", lambda: self._eyes_step(now))
        self._step("feelings", lambda: self.feelings.tick(present=self.presence.state != "away"))
        self._step("consolidation", lambda: self._maybe_consolidate(now))
        self._step("purge", lambda: self._maybe_purge(now))
        self._step("persistence", lambda: self._maybe_save(now))
        self._step("state", lambda: self._maybe_state(now))
        self._step("expression", self._sync_expression)
        if action:
            with self._hands_lock:  # the greeting's reminders are claimed until it ends
                ids = {r.id for r in self._greet_reminders}
                self._reminding |= ids
            return Utterance(self.speak_first(action), action.spoken, on_close=lambda: self._release(ids))
        return out if isinstance(out, Utterance) else None

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
        """No bids while a game is on, or while a confirmation waits (its mic would take the answer)."""
        if self.presence.state == "break" and self.game is None and not self._asking():
            bid = self.initiative.on_break("break", self.presence.work_stretch_s(), self.memory.due_threads())
            if bid:
                self._bid(bid)

    def _check_bid_ignored(self, now: float) -> None:
        if self.pending_bid_at is not None and now - self.pending_bid_at >= BID_IGNORED_S:
            self.feelings.on_event("bid_ignored")
            self.pending_bid_at = self._bid_thread = None

    def _end_idle_game(self, now: float) -> None:
        """A game with no turn for 10 min ends, silently."""
        if self.game is not None and now - self._game_at >= GAME_IDLE_S:
            self.game = None

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
            self.hands.purge(now)

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
            now = self.clock.now()
            if not self._restricted(now):  # else they stay pending: the report and reminder bubbles
                self._greet_reports, self._reports = list(self._reports.values()), {}  # the greeting tells them
                if self._greet_reports:
                    self._last_report_at = now
                with self._hands_lock:  # overdue reminders: the greeting's tail
                    self._greet_reminders = [r for r in self.hands.due(now) if r.id not in self._reminding]
        return action

    # ---- hands on the tick ----------------------------------------------------------------------------
    def _confirm_step(self) -> None:
        """A pending confirmation past its 60 s: confirm_done "expired" and the log (never an utterance)."""
        with self._hands_lock:
            p = self.confirms.expire()
        if p is not None:
            self._confirm_done(p, "expired")

    def _asking(self) -> bool:
        """A confirmation waits for its answer: the tick says nothing aloud meanwhile (bubbles only, never the mic),
        so a yes can only answer the question actually on the table."""
        with self._hands_lock:
            return self.confirms.pending is not None

    def _release(self, ids) -> None:
        """Ends tick's claim on reminder ids (and drops them from a greeting that never started)."""
        with self._hands_lock:
            self._reminding -= set(ids)
            self._greet_reminders = [r for r in self._greet_reminders if r.id not in ids]

    def _fire(self, ids, now: float) -> None:
        self.hands.mark_fired(ids, now)
        self._emit_reminders()

    def _reminder_step(self, now: float) -> "Utterance | bool | None":
        """Due reminders with manners, all merged into one line: spoken (focused, talking, break; at most one line
        every REMIND_GAP_S), a bubble at once while gaming, restricted (quiet hours, may not start conversations),
        muted or while a confirmation waits, and while away they wait for the came-back greeting. Never dropped: one
        is marked fired only once its line went out, and a bubble waits until a body is there to show it. Ids claimed
        by an Utterance still to be said are skipped."""
        with self._hands_lock:
            claimed = set(self._reminding)
        due = [r for r in self.hands.due(now) if r.id not in claimed]
        if not due:
            return None
        how = delivery(self.presence.state, self._restricted(now))
        if how == "wait":
            return None
        if how == "bubble" or self._asking() or self.settings.mute_voice:
            if not self.body_present():  # no one to show it to: it stays pending until a body connects
                return None
            line, uid = reminder_line(due, now, self.tz), uuid4().hex
            self.emit(Say(utterance_id=uid, text=line, spoken=False, kind="report"))
            self.emit(SayEnd(utterance_id=uid))
            self._fire([r.id for r in due], now)
            self._finish_turn(None, line, consolidate=False)
            return True
        if not gap_ok(self._last_remind_at, now):
            return None
        ids = {r.id for r in due}
        with self._hands_lock:
            self._reminding |= ids
        return Utterance(self._remind(ids), True, on_close=lambda: self._release(ids))

    def _remind(self, ids: set[int]) -> Iterator[str]:
        """The spoken reminder line (a template): due() is read again and mark_fired() called here, on this thread
        under the core lock, so a reminder cancelled meanwhile isn't said and none is said twice."""
        try:
            with self._lock:
                now = self.clock.now()
                items = [r for r in self.hands.due(now) if r.id in ids]
                if not items:
                    return
                line, uid = reminder_line(items, now, self.tz), uuid4().hex
                try:
                    self.emit(Say(utterance_id=uid, text=line, spoken=not self.settings.mute_voice, kind="report"))
                    self.emit(SayEnd(utterance_id=uid))
                finally:
                    self._last_remind_at = now
                    self._fire([r.id for r in items], now)
                    self._release(ids)
                try:
                    self._finish_turn(None, line, consolidate=False)
                except Exception:
                    log.exception("couldn't record the reminder")
                yield line
        finally:
            self._release(ids)

    def _results_step(self, now: float) -> "Utterance | bool | None":
        """A queued command or button result, one per tick, with the report manners (sharing REPORT_GAP_S with the
        agent reports): it waits while they're away; while gaming, restricted or a confirmation waits it's the
        ToolResult.text bubble (no model); otherwise she says it through the local model (the result text is the
        fallback). A bubble (or a muted line) waits until a body is there to show it."""
        if self._results.empty():
            return None
        how = delivery(self.presence.state, self._restricted(now))
        if how == "wait" or (self._last_report_at is not None and 0 <= now - self._last_report_at < REPORT_GAP_S):
            return None
        bubble = how == "bubble" or self._asking()
        if (bubble or self.settings.mute_voice) and not self.body_present():
            return None
        summary, result = self._results.get_nowait()
        self._last_report_at = now
        if bubble:
            uid = uuid4().hex
            self.emit(Say(utterance_id=uid, text=result.text, spoken=False, kind="report"))
            self.emit(SayEnd(utterance_id=uid))
            self.memory.append_transcript("assistant", result.text, consolidate=False)
            self.turns = (self.turns + [{"role": "assistant", "content": tool_turn(result.text)}])[-KEEP_TURNS:]
            self.last_turn_at = now
            return True
        started = {"yes": False}

        def say() -> Iterator[str]:
            started["yes"] = True
            with self._lock:
                yield from self._speak("report", user_text=None, query=summary, spoken=True, threads=[],
                                       consolidate=False, shape="tool", situation=hands_situation(summary, result),
                                       tail=result.text, tail_if_silent=True)

        def unsaid() -> None:  # closed before it began: back in the queue for a later tick
            if not started["yes"]:
                self._results.put((summary, result))
        return Utterance(say(), True, on_close=unsaid)

    def _park_step(self, now: float) -> "Utterance | bool | None":
        """A T2 call parked during a game: once they're neither gaming nor away it's asked, unless it waited over 10
        min (dropped, logged expired). The confirm_request and the question go out here, in the tick under the core
        lock, so no turn can answer it before it's shown; restricted (quiet hours, may not start conversations) it's
        only the bubble, with no Utterance (never the mic), else the Utterance just speaks the line. A pending
        confirmation it replaces gets confirm_done "replaced" and is logged so."""
        if self.presence.state in ("gaming", "away"):
            return None
        with self._hands_lock:
            old = self.confirms.pending
            got = self.confirms.unpark()
        if got is None:
            return None
        what, p = got
        if what == "raised" and old is not None:  # the raised call replaced a pending one: the body and the log know
            self._confirm_done(old, "replaced")
        if what == "expired":
            self._log_action(p.call.tool, TIERS.get(p.call.tool, 3), p.summary, "expired")
            return None
        restricted = self._restricted(now) or self.presence.state == "gaming"
        lines = list(self._raise_confirm(p, not restricted, kind="report"))  # empty unless it's to be spoken
        return Utterance((line for line in lines), True) if lines else True

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
        if hooks and (self._hooks_seen_at is None or not 0 <= now - self._hooks_seen_at < HOOKS_SEEN_EVERY_S):
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
        the model is never called. In quiet hours, or when she may not start conversations, it's always the bubble
        (no voice, no mic): urgent ones now, finished ones only on a talking turn. While a confirmation waits it's the
        bubble too. Returns the spoken Utterance, True
        when a bubble was shown, else None. A bubble (or a muted line) waits until a body is there to show it."""
        state = self.presence.state
        if not self._reports or state == "away":
            return None
        if self._last_report_at is not None and 0 <= now - self._last_report_at < REPORT_GAP_S:
            return None
        events = [ev for ev, _ in self._reports.values()]
        kinds = {ev.kind for ev in events}
        restricted = self._restricted(now)
        if not (kinds & URGENT or state in (("talking",) if restricted else ("break", "talking"))):
            return None
        bubble = state == "gaming" or restricted or self._asking()
        if (bubble or self.settings.mute_voice) and not self.body_present():
            return None  # no one to show it to: it stays pending until a body connects
        self._reports, self._last_report_at = {}, now
        self.feelings.show(REPORT_FACES[next(k for k in REPORT_FACES if k in kinds)], REPORT_FACE_S)
        self._sync_expression()
        if bubble:
            self._report_bubble(events)
            return True
        return Utterance(self._report(events), True)

    def _restricted(self, now: float) -> bool:
        """Quiet hours, or she may not start conversations: reports are never spoken, only the bubble."""
        s = self.settings
        return not s.may_start_conversations or in_quiet_hours(s.quiet_hours, local_time(now, self.tz))

    def _report(self, events: list[AgentEvent]) -> Iterator[str]:
        """One spoken line from the local model (40 tokens, 1 sentence). When nothing of it was said (the model
        failed, or said nothing usable), the templated bubble goes out instead, so the news is never lost."""
        said: list[str] = []

        def after(outcome: str) -> None:
            if not said:
                self._report_bubble(events)

        with self._lock:
            yield from self._counted(self._speak("report", user_text=None, query=report_line(events, detail=False),
                                                 situation=report_situation(events), spoken=True, threads=[],
                                                 after=after), said)

    def _load_reports(self) -> dict[str, tuple[AgentEvent, float]]:
        """The pending reports saved before a restart (the 6 h drop applies on the next tick); bad rows are skipped."""
        saved, out = self.kv.get(REPORTS_KEY), {}
        for r in saved if isinstance(saved, list) else []:
            try:
                strings = all(isinstance(r[k], str) for k in ("agent_id", "name", "text"))
                if r["kind"] in REPORT_FACES and strings and isinstance(r.get("error"), (str, type(None))):
                    out[r["agent_id"]] = (AgentEvent(r["kind"], r["agent_id"], r["name"], r["text"], r.get("error")),
                                          float(r["at"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _report_rows(self) -> list[dict]:
        return [{**asdict(ev), "at": at} for ev, at in self._reports.values()]

    def _save_reports(self) -> None:
        """Pending reports to KV, only when they changed."""
        rows = self._report_rows()
        if rows != self._reports_saved:
            self.kv.set(REPORTS_KEY, rows)
            self._reports_saved = rows

    def _report_bubble(self, events: list[AgentEvent]) -> None:
        """The templated report: a silent bubble, kept in the transcript like a bid (never consolidated)."""
        line, uid = report_bubble(events), uuid4().hex
        self.emit(Say(utterance_id=uid, text=line, spoken=False, kind="report"))
        self.emit(SayEnd(utterance_id=uid))
        self.memory.append_transcript("assistant", line, consolidate=False)
        self.turns = (self.turns + [{"role": "assistant", "content": line}])[-KEEP_TURNS:]
        self.last_turn_at = self.clock.now()

    def _agents_context(self) -> str | None:
        """The `### Your agents:` prompt line (outside the memory block, like the eyes block)."""
        return agents_block(self.agents.snapshot(), self.clock.now())

    def _agents_msg(self) -> AgentsMsg:
        r = self.receiver
        return AgentsMsg(items=[AgentInfo(**a) for a in self.agents.snapshot()], hooks_seen_at=self._hooks_seen_at,
                         receiver_ok=bool(r is not None and r.ok), rejected_recent=r.rejected.recent() if r else 0)

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
        """Everything a newly connected body needs: settings, state, agents, eyes, look, the reminders, the action
        log and a confirmation still pending."""
        with self._hands_lock:
            p = self.confirms.pending
        pending = p is not None and self.clock.now() < p.expires_at
        return [Hello(version=PROTOCOL_VERSION, expressions=list(EXPRESSIONS)),
                SettingsMsg(settings=asdict(self.settings)), self._state_msg(), self._agents_sent, self._eyes_msg(),
                *([self._look] if self._look else []), self._reminders_msg(), self._actions_msg(),
                *([ConfirmRequest(id=p.id, summary=p.summary, expires_at=p.expires_at)] if pending else [])]

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
            elif isinstance(msg, NoteDelete):
                self.hands.delete_note(msg.id)
            elif isinstance(msg, Confirm):
                self._confirm_button(msg.id, msg.yes)
                return
            elif isinstance(msg, ReminderCancel):
                self.hands.cancel(msg.id, self.clock.now())
                self._emit_reminders()
                return
            elif isinstance(msg, ActionsClear):
                self.hands.clear_actions()
                self.emit(self._actions_msg())
                return
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
                           glances=[GlanceInfo(**asdict(g)) for g in mem.list_glances()],
                           notes=[NoteInfo(id=n.id, at=n.at, text=n.text) for n in self.hands.notes()])

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
