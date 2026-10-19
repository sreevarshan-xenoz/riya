"""Runs the Riya core: the localhost WebSocket server in a thread and the 1 s tick loop on the main thread."""
import argparse
import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import BaseModel

from riya.agents import install_hooks
from riya.agents.hooks import HookReceiver
from riya.agents.logwatch import LogWatch
from riya.core import secrets as keystore
from riya.core.app import create_app, drain
from riya.core.clock import SystemClock
from riya.core.embed import HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.eyes.capture import make_window_source, set_dpi_awareness
from riya.core.hands.effects import WindowsEffects
from riya.core.hands.net import UrlHttp
from riya.core.listen import ListenGate
from riya.core.models import VISION_TIMEOUT_S, load_dotenv, local_model, online_pair  # noqa: F401 (re-exported)
from riya.core.presence import make_idle_source
from riya.core.protocol import Mouth
from riya.core.secrets import KEY_VAR, OnlineKey
from riya.core.setup import SetupDeps
from riya.core.store import open_db
from riya.voice_candidates import builtin_voices, pick_voices

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
LOGWATCH_EVERY_S = 60.0
DEFAULT_DB = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Riya" / "riya.db"
REPO_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger("riya")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m riya.core", description="Run the Riya core (WebSocket on localhost).")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database (default: %%LOCALAPPDATA%%/Riya/riya.db)")
    ap.add_argument("--local-url", default="http://127.0.0.1:8090", help="llama-server base URL")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-voice", action="store_true", help="don't speak; drain spoken sentences silently")
    ap.add_argument("--hash-embed", action="store_true", help="use the dependency-free hash embedder")
    ap.add_argument("--legacy-memory", default="riya_memory.json",
                    help="legacy memory file, imported on first run (relative paths start at the repo root)")
    ap.add_argument("--secrets", default=None,
                    help="the online key file (default: %%LOCALAPPDATA%%/Riya/secrets.env)")
    return ap.parse_args(argv)


def make_embedder(hash_embed: bool):
    """bge-small, or the hash embedder when asked or when bge can't load (offline before its first download)."""
    if hash_embed:
        return HashEmbedder()
    try:
        from riya.core import embed
        return embed.FastEmbedder()
    except Exception as e:
        log.warning("the bge embedder couldn't load (%s); using the hash embedder, so memories saved with bge "
                    "won't be found until it loads again", e)
        return HashEmbedder()


def build_core(args: argparse.Namespace, emit: Callable[[BaseModel], None],
               body_present: Callable[[], bool] = lambda: True) -> RiyaCore:
    """Per-monitor DPI awareness comes first, before any window source or capture, so window rects and pixels agree.
    The online key: one saved in secrets.env beats the repo .env and the process environment (read after .env, so
    `had` tells the two apart). Her eyes get their own online model instance (a longer first-token timeout); on other
    platforms she has no window to look at. body_present gates her eyes on a connected body."""
    set_dpi_awareness()
    had = bool((os.environ.get(KEY_VAR) or "").strip())
    load_dotenv()
    key = OnlineKey.at_start(os.environ, had, Path(getattr(args, "secrets", None) or keystore.default_path()))
    online, vision = online_pair(key.env(os.environ))
    deps = SetupDeps(key, dict(os.environ), install_hooks.default_settings_path(), install_hooks.default_token_path(),
                     subprocess.run, lambda: pick_voices(builtin_voices()))
    core = RiyaCore(db=open_db(args.db), embedder=make_embedder(args.hash_embed), clock=SystemClock(),
                    idle_source=make_idle_source(), local=local_model(args.local_url),
                    online=online, emit=emit, vision=vision,
                    window_source=make_window_source(), body_present=body_present,
                    effects=WindowsEffects(), http=UrlHttp(),  # her hands: only the runner gives her real ones
                    setup_deps=deps, local_url=args.local_url)
    legacy = Path(args.legacy_memory)
    legacy = legacy if legacy.is_absolute() else REPO_ROOT / legacy
    if not core.kv.get("legacy_imported") and legacy.is_file():
        core.memory.import_legacy_json(legacy)
        core.kv.set("legacy_imported", True)
    return core


def build_voice(core: RiyaCore, publish: Callable[[BaseModel], None]):
    """Loads her voice (Pocket TTS), ears (Moonshine) and VAD (Silero); mouth levels go to every body."""
    import sounddevice  # noqa: F401  (fail here, not on the first reply, when there's no audio stack)

    from riya.stt_moonshine import MoonshineStreamSTT
    from riya.voice import PocketTTS, SileroVAD, VoiceLoop
    return connect_voice(core, VoiceLoop(core, MoonshineStreamSTT("small"), PocketTTS(core.settings.voice),
                                         SileroVAD(), ListenGate(core.clock),
                                         on_mouth=lambda level: publish(Mouth(level=level))))


def connect_voice(core: RiyaCore, loop):
    """After a silent bid the mic opens like after any turn, so a spoken answer needs no name. The Health page's Play
    sample and Mic check go through the loop and riya.voice.probe_mic; the TTS voice list is read once, on its own
    riya-voices thread (off the core lock and off the health thread)."""
    import riya.voice  # here, not at the top: importing the runner loads no audio stack

    core.on_spoke_first = loop.gate.open
    core.voice_sampler = loop.sample
    core.mic_prober = riya.voice.probe_mic
    threading.Thread(target=core.cache_voices, name="riya-voices", daemon=True).start()
    return loop


class Speaker:
    """Typed replies from the body: spoken by the voice loop once it's running, drained before that or without it."""

    def __init__(self):
        self.loop = None

    def __call__(self, sentences: Iterator[str]) -> None:
        if self.loop is None:
            drain(sentences)
        else:
            self.loop.answer(sentences, reason="typed")


def deliver(sentences: Iterator[str], loop, gaming: bool = False) -> None:
    """The one place a tick's utterance (her speaking first) is consumed: spoken if it's meant aloud, then the mic
    listens for an answer, except during a game (the voice loop's answer() checks that itself)."""
    if loop is None:
        drain(sentences)
    elif getattr(sentences, "spoken", True):
        loop.answer(sentences, reason="spoke_first")
    else:
        drain(sentences)
        if not gaming:
            loop.gate.open("spoke_first")


def run_ticks(core: RiyaCore, speaker: Speaker, alive: Callable[[], bool],
              wait: Callable[[float], bool] | None = None) -> None:
    """The 1 s tick loop, until the server stops or the body asks the core to quit (wait returns True then)."""
    wait = wait or core.quit_requested.wait
    while alive() and not core.quit_requested.is_set():
        if (sentences := core.tick()) is not None:
            deliver(sentences, speaker.loop, gaming=core.presence.state == "gaming")
        if wait(1.0):
            return


def run_logwatch(core: RiyaCore, stop: threading.Event, every: float = LOGWATCH_EVERY_S, make=LogWatch) -> None:
    """Polls the run logs into core.colab_inbox every 60 s until stop is set. The blocking Drive I/O happens here,
    never under the core lock; the entries are read from settings each cycle, so an edit applies on the next poll."""
    while not stop.is_set():
        try:
            for item in make(core.settings.agent_logs, core.kv, core.clock).poll():
                core.colab_inbox.put(item)
        except Exception:
            log.exception("the log watcher's poll failed")
        if stop.wait(every):
            return


def start_logwatch(core: RiyaCore, **kw) -> tuple[threading.Event, threading.Thread]:
    """The riya-logwatch daemon thread; set the returned event to stop it."""
    stop = threading.Event()
    thread = threading.Thread(target=run_logwatch, args=(core, stop), kwargs=kw, name="riya-logwatch", daemon=True)
    thread.start()
    return stop, thread


def start_receiver(core: RiyaCore, make=None):
    """The hook receiver, kept up (retried while 8766 is busy) and handed to the core for the settings' status."""
    receiver = (make or HookReceiver)(core.agent_inbox.put).start()
    core.receiver = receiver
    return receiver


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    keystore.install_redaction()  # before anything can log: every record, uvicorn's included, is scrubbed of the key
    speaker = Speaker()
    app = create_app(lambda emit: build_core(args, emit, body_present=lambda: app.state.hub.connections > 0),
                     speak=speaker)
    server = uvicorn.Server(uvicorn.Config(app, host=DEFAULT_HOST, port=args.port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="riya-ws", daemon=True)
    thread.start()
    hooks = logwatch = None
    try:
        while not hasattr(app.state, "core"):
            if not thread.is_alive():
                raise SystemExit("the server stopped before the core was ready")
            time.sleep(0.1)
        core = app.state.core
        log.info("Riya core listening on ws://%s:%d/ws", DEFAULT_HOST, args.port)
        # hook payloads and log lines cross from their own threads into the core's inboxes; the tick drains them
        hooks = start_receiver(core)
        logwatch = start_logwatch(core)
        if not args.no_voice:
            try:
                speaker.loop = build_voice(core, app.state.hub.publish)
                threading.Thread(target=speaker.loop.run, name="riya-voice", daemon=True).start()
            except Exception:
                log.exception("voice failed to start; running text-only")
        run_ticks(core, speaker, thread.is_alive)
        if core.quit_requested.is_set():
            log.info("Riya was quit from the tray; stopping the core")
    except KeyboardInterrupt:
        pass
    finally:
        if speaker.loop is not None:
            speaker.loop.stop()
        if hooks is not None:
            hooks.stop()
        if logwatch is not None:
            logwatch[0].set()
            logwatch[1].join(timeout=5)
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
