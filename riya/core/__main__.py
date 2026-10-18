"""Runs the Riya core: the localhost WebSocket server in a thread and the 1 s tick loop on the main thread."""
import argparse
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from pydantic import BaseModel

from riya.core.app import create_app, drain
from riya.core.clock import SystemClock
from riya.core.embed import HashEmbedder
from riya.core.engine import RiyaCore
from riya.core.listen import ListenGate
from riya.core.models import OpenAICompatModel, load_dotenv, load_online_from_env
from riya.core.presence import make_idle_source
from riya.core.protocol import Mouth
from riya.core.store import open_db

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_DB = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Riya" / "riya.db"
log = logging.getLogger("riya")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m riya.core", description="Run the Riya core (WebSocket on localhost).")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="SQLite database (default: %%LOCALAPPDATA%%/Riya/riya.db)")
    ap.add_argument("--local-url", default="http://127.0.0.1:8090", help="llama-server base URL")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-voice", action="store_true", help="don't speak; drain spoken sentences silently")
    ap.add_argument("--hash-embed", action="store_true", help="use the dependency-free hash embedder")
    ap.add_argument("--legacy-memory", default="riya_memory.json", help="legacy memory file, imported on first run")
    return ap.parse_args(argv)


def build_core(args: argparse.Namespace, emit: Callable[[BaseModel], None]) -> RiyaCore:
    load_dotenv()
    if args.hash_embed:
        embedder = HashEmbedder()
    else:
        from riya.core.embed import FastEmbedder
        embedder = FastEmbedder()
    core = RiyaCore(db=open_db(args.db), embedder=embedder, clock=SystemClock(), idle_source=make_idle_source(),
                    local=OpenAICompatModel("local", f"{args.local_url}/v1"), online=load_online_from_env(), emit=emit)
    if not core.kv.get("legacy_imported"):
        core.memory.import_legacy_json(args.legacy_memory)
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
    """After a silent bid the mic opens like after any turn, so a spoken answer needs no name."""
    core.on_spoke_first = loop.gate.open
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


def deliver(sentences: Iterator[str], loop) -> None:
    """The one place a tick's utterance (her speaking first) is consumed: spoken if it's meant aloud, then the mic
    listens for an answer."""
    if loop is None:
        drain(sentences)
    elif getattr(sentences, "spoken", True):
        loop.answer(sentences, reason="spoke_first")
    else:
        drain(sentences)
        loop.gate.open("spoke_first")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    speaker = Speaker()
    app = create_app(lambda emit: build_core(args, emit), speak=speaker)
    server = uvicorn.Server(uvicorn.Config(app, host=DEFAULT_HOST, port=args.port, log_level="warning"))
    thread = threading.Thread(target=server.run, name="riya-ws", daemon=True)
    thread.start()
    try:
        while not hasattr(app.state, "core"):
            if not thread.is_alive():
                raise SystemExit("the server stopped before the core was ready")
            time.sleep(0.1)
        core = app.state.core
        log.info("Riya core listening on ws://%s:%d/ws", DEFAULT_HOST, args.port)
        if not args.no_voice:
            try:
                speaker.loop = build_voice(core, app.state.hub.publish)
                threading.Thread(target=speaker.loop.run, name="riya-voice", daemon=True).start()
            except Exception:
                log.exception("voice failed to start; running text-only")
        while thread.is_alive():
            if (sentences := core.tick()) is not None:
                deliver(sentences, speaker.loop)
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if speaker.loop is not None:
            speaker.loop.stop()
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == "__main__":
    main()
