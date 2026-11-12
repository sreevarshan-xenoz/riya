"""The Health page's one-click actions: a fixed allow-list, single-use confirm nonces for the outward-facing four, and
one worker thread per action, watched by a timeout.

Only a body click reaches this: `request` runs a plain action or returns a ConfirmAsk for a gated one, and only
`answer(nonce, yes=True)` with a fresh, unused nonce runs a gated action. The key never leaves here except as its mask:
every Outcome detail is scrubbed, action-log rows are fixed summaries, and args are never logged or echoed. Imports
nothing from the engine or the protocol."""
from __future__ import annotations

import logging
import secrets
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol
from urllib.parse import urlsplit

from riya import brain
from riya.agents import install_hooks
from riya.core import models
from riya.core.eyes.see import NOTE_MAX_TOKENS, vision_messages
from riya.core.health import DETAIL_MAX, clip, clip_left
from riya.core.models import VISION_TIMEOUT_S, ChatModel, ModelUnavailable, complete, online_pair
from riya.core.secrets import InvalidKey, KeyFileError, OnlineKey, mask, scrub
from riya.core.settings import Settings
from riya.core.store import KV
from riya.voice_candidates import TEXT

ACTIONS = ("online_key_set", "online_key_clear", "online_test", "voice_sample", "voice_choose", "mic_check",
           "eyes_test", "hooks_connect", "hooks_disconnect", "autostart", "onboarding_done", "use_base", "use_riya_v1")
CONFIRM_GATED = ("online_test", "eyes_test", "hooks_connect", "hooks_disconnect")
PLAIN = tuple(a for a in ACTIONS if a not in CONFIRM_GATED)
ARGS = {"online_key_set": {"key": str}, "voice_sample": {"voice": str}, "voice_choose": {"voice": str},
        "autostart": {"on": bool}}
OPTIONAL_ARGS = {"voice_sample": ("voice",)}  # left out: the current voice
NONCE_TTL_S = 60
MAX_PENDING = 8
ACTION_TIMEOUT_S = 30.0
ONLINE_TEST_TIMEOUT_S = 10.0
ONLINE_TEST_TOKENS = 200
ONLINE_TEST_PROMPT = "Reply with the single word: ok"
EYES_IMAGE = Path(__file__).parent / "assets" / "eyes_test.png"
EYES_QUESTION = "What number is written in this picture? Answer with the number only."
SAMPLE_TEXT = TEXT
TESTS_KEY = "setup.last_tests"
TESTED = ("online_test", "eyes_test", "mic_check", "voice_sample")  # their last result is kept under TESTS_KEY
CANCELLED = "Cancelled; nothing changed."
NONCE_REFUSED = "That confirmation expired or was already used, so nothing ran."
REFUSED_SUMMARY = "Refused a stale or reused confirmation"
SUMMARIES = {
    "online_key_set": "Saved a new online key", "online_key_clear": "Removed the saved online key",
    "online_test": "Tested the online key", "voice_sample": "Played a voice sample", "voice_choose": "Changed her voice",
    "mic_check": "Checked the mic", "eyes_test": "Ran the eye test", "hooks_connect": "Connected Claude Code",
    "hooks_disconnect": "Disconnected Claude Code", "autostart": "Changed start with Windows",
    "onboarding_done": "Finished the welcome", "use_base": "Switched to the base model",
    "use_riya_v1": "Switched to Riya v1",
}
USING_BASE = "Using the base model now; llama-server restarts with it in a few seconds."
USING_V1 = "Using Riya v1 now; llama-server restarts with it in a few seconds."
NOT_VETTED = "Riya v1 hasn't passed the champion gate yet (python -m evaluation.champion)."
_RESTART = "Restart Claude Code (and open a new terminal)."
log = logging.getLogger("riya.setup")


@dataclass(frozen=True)
class Outcome:
    action: str
    ok: bool
    detail: str

    def __post_init__(self):
        # scrub first: a remembered key inside an error text must not survive the clip
        object.__setattr__(self, "detail", clip(scrub(str(self.detail))))


@dataclass(frozen=True)
class ConfirmAsk:
    nonce: str
    action: str
    text: str


class SetupError(ValueError):
    """A request outside the allow-list. The message names the action, never the args."""


class Nonces:
    """Single-use confirmation tokens: each names one action and its args, and is good while
    clock.now() - issued_at < ttl_s, and only for the connection (`to`) it was issued to. At most MAX_PENDING wait at
    once; a new one drops the oldest."""

    def __init__(self, clock, ttl_s: float = NONCE_TTL_S, make: Callable[[], str] = lambda: secrets.token_urlsafe(16)):
        self.clock, self.ttl_s, self.make = clock, ttl_s, make
        self._pending: dict[str, tuple[str, dict, float, object]] = {}  # insertion order is issue order
        self._lock = threading.Lock()

    def issue(self, action: str, args: dict, to: object = None) -> str:
        with self._lock:
            now = self.clock.now()
            for n in [n for n, (_, _, at, _) in self._pending.items() if now - at >= self.ttl_s]:
                del self._pending[n]
            nonce = self.make()
            self._pending.pop(nonce, None)
            self._pending[nonce] = (action, dict(args), now, to)
            while len(self._pending) > MAX_PENDING:
                del self._pending[next(iter(self._pending))]
            return nonce

    def take(self, nonce, to: object = None) -> tuple[str, dict] | None:
        """(action, args), removing the nonce; None when it is unknown, already taken or expired. From another
        connection than the one it was issued to it is None too, and it stays good for its own."""
        if not isinstance(nonce, str):
            return None
        with self._lock:
            entry = self._pending.get(nonce)
            if entry is None or entry[3] is not to:
                return None
            del self._pending[nonce]
            action, args, at, _ = entry
            if self.clock.now() - at >= self.ttl_s:
                return None
            return action, args


class SetupHost(Protocol):
    """What Setup needs from the core (RiyaCore implements it)."""
    settings: Settings
    clock: object  # an object with now() -> float
    kv: KV

    def patch_settings(self, patch: dict) -> Settings:
        """ValueError for a bad patch; OSError when autostart couldn't change. A patch holding "autostart" always
        rewrites or removes the Startup shortcut."""

    def swap_online(self, online: ChatModel | None, vision: ChatModel | None) -> None: ...

    def sample_voice(self, voice: str, text: str) -> None:
        """RuntimeError(reason) when she can't speak it."""

    def probe_mic(self) -> tuple[bool, str]: ...

    def log_action(self, tool: str, tier: int, summary: str, outcome: str) -> None: ...


def _no_run(*_args, **_kw) -> SimpleNamespace:
    return SimpleNamespace(returncode=1)


@dataclass
class SetupDeps:
    key: OnlineKey  # its repr shows only the mask
    environ: Mapping[str, str] = field(repr=False)  # load_dotenv may have put the .env key in it
    claude_settings: Path
    token_path: Path
    runner: Callable
    voices: Callable[[], list[str]]  # may import the TTS stack (slow): only Setup.load_voices calls it
    eyes_image: Path = EYES_IMAGE
    make_model: Callable[[Mapping[str, str], str, float], ChatModel | None] = models.online_model
    adapter_path: Path = field(default_factory=lambda: brain.default_adapter())

    @classmethod
    def sandboxed(cls, root: Path) -> SetupDeps:
        """Everything under root, and a runner that runs nothing (setx "fails"): for a core nobody wired up."""
        root = Path(root)
        return cls(OnlineKey(root / "secrets.env", None, None), {}, root / ".claude" / "settings.json",
                   root / "hook_token", _no_run, lambda: ["alba"], adapter_path=root / "riya-v1-lora.gguf")


def _daemon(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, name="riya-setup-watch", daemon=True).start()


_SETX_FAILED = f"Hooks added, but setx failed: set {install_hooks.ENV_VAR} yourself to the token in "


def _installer_why(why: list[tuple[str, str]], token_path: Path) -> str:
    """The step the installer itself reported as failed (never a settings warning printed along the way), with the
    part the user acts on first; a path is clipped from the left so its tail survives."""
    if not why:
        return "The installer stopped without saying why; nothing is confirmed."
    step, reason = why[-1]
    if step == "setx":
        return _SETX_FAILED + clip_left(str(token_path), max(DETAIL_MAX - len(_SETX_FAILED), 2))
    return f"Nothing was changed: {reason}"


class Setup:
    def __init__(self, host: SetupHost, deps: SetupDeps, on_result: Callable[[Outcome], None], *,
                 timeout_s: float = ACTION_TIMEOUT_S, spawn: Callable[[Callable[[], None]], None] | None = None):
        self.host, self.deps, self.on_result = host, deps, on_result
        self.timeout_s = timeout_s
        self.nonces = Nonces(host.clock)
        self._spawn = spawn or _daemon
        self._running: set[str] = set()
        self._running_lock = threading.Lock()
        self._tests_lock = threading.Lock()
        self._key_lock = threading.Lock()  # a key save or clear and its model rebuild happen as one
        self._voices: list[str] = []
        self._voices_lock = threading.Lock()

    def __repr__(self) -> str:
        return f"Setup(deps={self.deps!r})"  # never the key: deps masks it

    # ---- voices ---------------------------------------------------------------------------------------------------
    def voices(self) -> list[str]:
        """The cached voice list ([] until load_voices ran); never calls deps.voices, so it can't stall."""
        with self._voices_lock:
            return list(self._voices)

    def load_voices(self) -> None:
        try:
            got = list(self.deps.voices())
        except Exception as e:
            log.warning("couldn't list her voices (%s); offering alba only", type(e).__name__)
            got = ["alba"]
        with self._voices_lock:
            self._voices = got

    # ---- the request / answer / perform split ---------------------------------------------------------------------
    def _validate(self, action, args) -> dict:
        if not isinstance(action, str) or action not in ACTIONS:
            raise SetupError("unknown setup action")
        spec = ARGS.get(action, {})
        required = set(spec) - set(OPTIONAL_ARGS.get(action, ()))
        if not isinstance(args, dict) or not required <= set(args) <= set(spec) or any(
                type(args[k]) is not t for k, t in spec.items() if k in args):  # exact types: True is no str, 1 no bool
            raise SetupError(f"bad arguments for {action}")
        return dict(args)

    def request(self, action: str, args: dict, to: object = None) -> ConfirmAsk | None:
        """A body click. Plain actions start at once; a gated one only gets a ConfirmAsk (its nonce) and runs nothing.
        `to` is the asking connection (None in-process): its nonce is good only from there, and every result of this
        request goes only there."""
        args = self._validate(action, args)
        if action == "onboarding_done" and getattr(self.host.settings, "onboarded", False) is True:
            return None  # idempotent: no thread, no Outcome, no second "Finished the welcome" row
        if action in CONFIRM_GATED:
            try:
                text = self.confirm_text(action)
            except SetupError as e:
                self._report(Outcome(action, False, str(e)), to)
                return None
            return ConfirmAsk(self.nonces.issue(action, args, to), action, text)
        self.start(action, args, to)
        return None

    def confirm_text(self, action: str) -> str:
        """What the user is asked to confirm; SetupError (with the reason) when there is nothing to confirm."""
        if action in ("hooks_connect", "hooks_disconnect"):
            try:
                return install_hooks.preview("apply" if action == "hooks_connect" else "remove",
                                             self.deps.claude_settings)
            except install_hooks.SettingsError:
                raise SetupError("Couldn't read Claude Code's settings.json; nothing was changed.") from None
        if action not in ("online_test", "eyes_test"):
            raise SetupError("unknown setup action")
        key_mask = self.deps.key.info().mask
        if not models.online_available(self._env()):
            raise SetupError("No online model to test.")
        base = models.online_base(self._env())
        try:
            host = urlsplit(base).hostname or "the online service"
        except ValueError:
            host = "the online service"
        with_key = f" with your key {key_mask}" if key_mask else ""
        if action == "online_test":
            return f"Riya will send one tiny request to {host}{with_key} to check that it works."
        return (f"Riya will send her built-in test picture (not your screen) to {host}{with_key} and "
                f"ask what number it shows.")

    def answer(self, nonce: str, yes: bool, to: object = None) -> bool:
        """The user's Yes or No to a ConfirmAsk, from connection `to`. False (and nothing runs) for a nonce that is
        unknown, reused, expired, issued to another connection or not for a gated action."""
        taken = self.nonces.take(nonce, to)
        if taken is None or taken[0] not in CONFIRM_GATED:
            self._log(REFUSED_SUMMARY, 2, "refused")
            return False
        action, args = taken
        if yes is not True:
            self._log(SUMMARIES[action], 2, "declined")
            self._report(Outcome(action, False, CANCELLED), to)
            return True
        self.start(action, args, to)
        return True

    def start(self, action: str, args: dict, to: object = None) -> None:
        """Run perform on a riya-setup thread, watched for timeout_s; one run of each action at a time. Its result goes
        to `to` (None: on_result's default)."""
        with self._running_lock:
            busy = action in self._running
            if not busy:
                self._running.add(action)
        if busy:
            self._report(Outcome(action, False, "Already running."), to)
            return
        try:
            self._spawn(lambda: self._watch(action, args, to))
        except Exception:
            self._release(action)
            raise

    def _release(self, action: str) -> None:
        with self._running_lock:
            self._running.discard(action)

    def _watch(self, action: str, args: dict, to: object = None) -> None:
        box: list[Outcome] = []
        done = threading.Event()

        def work():
            try:
                box.append(self.perform(action, args))
            finally:
                self._release(action)  # a hung run keeps its slot until it really ends: no pile of threads
                done.set()

        threading.Thread(target=work, name="riya-setup", daemon=True).start()
        if done.wait(self.timeout_s) and box:
            outcome = box[0]
        else:  # too slow (a late result is dropped) or perform broke its never-raises promise
            outcome = Outcome(action, False, f"No answer within {self.timeout_s:g} s." if not done.is_set()
                              else "Something went wrong.")
        self._finish(action, outcome, to)

    def _finish(self, action: str, outcome: Outcome, to: object = None) -> None:
        if action in TESTED:
            self._record_test(action, outcome)
        self._log(SUMMARIES[action], 2 if action in CONFIRM_GATED else 0, "ran" if outcome.ok else "failed")
        self._report(outcome, to)

    def _record_test(self, action: str, outcome: Outcome) -> None:
        try:
            entry = {"at": self.host.clock.now(), "ok": outcome.ok, "detail": outcome.detail,
                     "mask": self.deps.key.info().mask}
            with self._tests_lock:  # read-modify-write: two results finishing at once must both land
                tests = self.host.kv.get(TESTS_KEY, {})
                tests = tests if isinstance(tests, dict) else {}
                tests[action] = entry
                self.host.kv.set(TESTS_KEY, tests)
        except Exception as e:
            log.warning("couldn't record the %s result (%s)", action, type(e).__name__)

    def _log(self, summary: str, tier: int, outcome: str) -> None:
        try:
            self.host.log_action("setup", tier, summary, outcome)
        except Exception as e:
            log.warning("couldn't write the action log (%s)", type(e).__name__)

    def _report(self, outcome: Outcome, to: object = None) -> None:
        try:
            if to is None:
                self.on_result(outcome)
            else:
                self.on_result(outcome, to)  # only the connection that asked hears it
        except Exception as e:
            log.warning("couldn't report the %s result (%s)", outcome.action, type(e).__name__)

    # ---- the actions ----------------------------------------------------------------------------------------------
    def perform(self, action: str, args: dict) -> Outcome:
        """Run one action now, on this thread. Never raises; an unexpected error names only its type."""
        if action not in ACTIONS:
            return Outcome(str(action)[:40], False, "unknown setup action")
        try:
            return getattr(self, "_do_" + action)(args)
        except Exception as e:
            log.warning("setup action %s failed (%s)", action, type(e).__name__)
            return Outcome(action, False, f"Something went wrong ({type(e).__name__}).")

    def _env(self) -> dict[str, str]:
        return self.deps.key.env(self.deps.environ)

    def _rebuild(self) -> None:
        """Both online models from the current key, through models.online_pair (the one builder)."""
        self.host.swap_online(*online_pair(self._env(), self.deps.make_model))

    def _do_online_key_set(self, args) -> Outcome:
        with self._key_lock:  # an overlapping clear can't leave the models on a key the file no longer has
            try:
                key = self.deps.key.set(args["key"])
            except InvalidKey as e:
                return Outcome("online_key_set", False, str(e))
            except OSError:
                return Outcome("online_key_set", False, "Couldn't save the key.")
            self._rebuild()
        return Outcome("online_key_set", True, f"Saved {mask(key)}. Her online brain uses it now.")

    def _do_online_key_clear(self, args) -> Outcome:
        key = self.deps.key
        with self._key_lock:  # the check, the removal and the rebuild as one, against an overlapping save
            source = key.info().source
            if source != "app":
                why = {".env": "That key is set via .env; edit that file to remove it.",
                       "env": "That key comes from the RIYA_ONLINE_API_KEY environment variable."}
                return Outcome("online_key_clear", False, why.get(source, "There's no saved key."))
            try:
                key.clear()
            except KeyFileError as e:
                return Outcome("online_key_clear", False, str(e))
            except OSError:
                return Outcome("online_key_clear", False, "Couldn't remove the saved key.")
            self._rebuild()
        detail = "Removed the saved key."
        if key.base:
            if key.base_source == ".env":
                detail += " Using the one from .env."
            elif key.base_source == "env":
                detail += " Using the one from the RIYA_ONLINE_API_KEY environment variable."
        return Outcome("online_key_clear", True, detail)

    def _do_online_test(self, args) -> Outcome:
        model = self.deps.make_model(self._env(), "online-test", ONLINE_TEST_TIMEOUT_S)
        if model is None:
            return Outcome("online_test", False, "No online model is set up.")
        try:
            complete(model, [{"role": "user", "content": ONLINE_TEST_PROMPT}], ONLINE_TEST_TOKENS)
        except ModelUnavailable as e:
            return Outcome("online_test", False, f"It didn't answer: {e}")
        return Outcome("online_test", True, "The online brain answered.")  # a completed request passes, even empty

    def _do_eyes_test(self, args) -> Outcome:
        """The bundled picture, never the screen."""
        model = self.deps.make_model(self._env(), "eyes-test", VISION_TIMEOUT_S)
        if model is None:
            return Outcome("eyes_test", False, "Her eyes need an online model.")
        messages = vision_messages(Path(self.deps.eyes_image).read_bytes(), EYES_QUESTION, mime="image/png")
        try:
            text = complete(model, messages, NOTE_MAX_TOKENS)
        except ModelUnavailable as e:
            return Outcome("eyes_test", False, f"It didn't answer: {e}")
        if "42" in text:
            return Outcome("eyes_test", True, "She read the test picture: 42.")
        return Outcome("eyes_test", False, f"She couldn't read the test picture (she said {clip(text, 40)!r}).")

    def _known_voice(self, voice: str) -> bool:
        return voice in self.voices() or voice == self.host.settings.voice

    def _do_voice_sample(self, args) -> Outcome:
        voice = args.get("voice", self.host.settings.voice)
        if not self._known_voice(voice):
            return Outcome("voice_sample", False, "Unknown voice.")
        try:
            self.host.sample_voice(voice, SAMPLE_TEXT)
        except RuntimeError as e:
            return Outcome("voice_sample", False, str(e))
        return Outcome("voice_sample", True, f"Played a sample of {voice}.")

    def _do_voice_choose(self, args) -> Outcome:
        voice = args["voice"]
        if not self._known_voice(voice):
            return Outcome("voice_choose", False, "Unknown voice.")
        self.host.patch_settings({"voice": voice})
        return Outcome("voice_choose", True, f"Her voice is {voice} now.")

    def _do_mic_check(self, args) -> Outcome:
        ok, detail = self.host.probe_mic()
        return Outcome("mic_check", bool(ok), detail)

    def _hooks(self, mode: str) -> tuple[int, list[tuple[str, str]]]:
        """The installer in-process on deps' paths; (return code, the failed step it reported)."""
        lines: list[str] = []  # its printed text stays here: the detail comes from `why`, not from parsing lines
        why: list[tuple[str, str]] = []
        rc = install_hooks.main([mode, "--settings", str(self.deps.claude_settings)], runner=self.deps.runner,
                                token_path=self.deps.token_path, out=lines.append, why=why)
        return rc, why

    def _do_hooks_connect(self, args) -> Outcome:
        rc, why = self._hooks("--apply")
        if rc == 0:
            try:
                blocked = install_hooks.blockers(install_hooks.read_settings(Path(self.deps.claude_settings)))
            except install_hooks.SettingsError:
                blocked = []
            if blocked:  # written, but Claude Code's own settings stop them firing: say so, never edit those keys
                return Outcome("hooks_connect", True, f"Connected, but Claude Code will block it: {'; '.join(blocked)}")
            return Outcome("hooks_connect", True, f"Connected. {_RESTART}")
        return Outcome("hooks_connect", False, _installer_why(why, self.deps.token_path))

    def _do_hooks_disconnect(self, args) -> Outcome:
        rc, why = self._hooks("--remove")
        if rc == 0:
            return Outcome("hooks_disconnect", True, f"Disconnected. {_RESTART}")
        return Outcome("hooks_disconnect", False, _installer_why(why, self.deps.token_path))

    def _do_autostart(self, args) -> Outcome:
        on = args["on"]
        try:
            self.host.patch_settings({"autostart": on})
        except OSError:
            return Outcome("autostart", False, "Couldn't change the Startup shortcut.")
        return Outcome("autostart", True, "Starts with Windows now." if on else "Won't start with Windows.")

    def _do_onboarding_done(self, args) -> Outcome:
        self.host.patch_settings({"onboarded": True})
        return Outcome("onboarding_done", True, "Welcome done.")

    def _do_use_base(self, args) -> Outcome:
        self.host.patch_settings({"local_adapter": None})
        return Outcome("use_base", True, USING_BASE)

    def _do_use_riya_v1(self, args) -> Outcome:
        path = Path(self.deps.adapter_path)
        if not path.is_file():
            prefix = "There's no Riya v1 adapter at "
            return Outcome("use_riya_v1", False, prefix + clip_left(str(path), DETAIL_MAX - len(prefix)))
        if brain.read_record(path) is None:  # never adopt what the champion gate hasn't vetted
            return Outcome("use_riya_v1", False, NOT_VETTED)
        self.host.patch_settings({"local_adapter": str(path)})
        return Outcome("use_riya_v1", True, USING_V1)
