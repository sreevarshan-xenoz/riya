"""The localhost WebSocket app: a Hub fans core messages out to every connected body, /ws carries the protocol."""
import asyncio
import logging
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from riya.core.engine import RiyaCore
from riya.core.protocol import ErrorMsg, ProtocolError, UserText, dump, parse_body_message

log = logging.getLogger("riya.app")
BODY_ORIGINS = ("null", "file://")  # Electron loadFile pages; Node and Electron-main clients send no Origin


class Hub:
    """One queue per connected socket. publish() is safe from any thread and silent before a loop is attached."""

    def __init__(self):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: set[asyncio.Queue] = set()

    @property
    def connections(self) -> int:
        return len(self._queues)

    def attach(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.add(q)
        return q

    def disconnect(self, q: asyncio.Queue) -> None:
        self._queues.discard(q)

    def publish(self, msg: BaseModel) -> None:
        loop = self._loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(self._deliver, msg)
        except RuntimeError:  # the loop is closed (shutting down)
            pass

    def _deliver(self, msg: BaseModel) -> None:
        for q in list(self._queues):
            q.put_nowait(msg)


def drain(sentences: Iterator[str]) -> None:
    for _ in sentences:
        pass


def create_app(core_factory: Callable[[Callable[[BaseModel], None]], RiyaCore],
               speak: Callable[[Iterator[str]], None] | None = None) -> FastAPI:
    """speak consumes a typed reply's sentences (the voice loop speaks them); by default they are only drained."""
    hub = Hub()
    speak = speak or drain

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.attach(asyncio.get_running_loop())
        app.state.core = await run_in_threadpool(core_factory, hub.publish)
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.hub = hub

    async def receive(ws: WebSocket, core: RiyaCore, q: asyncio.Queue, texts: asyncio.Queue) -> None:
        while True:
            raw = await ws.receive_text()
            try:
                msg = parse_body_message(raw)
            except ProtocolError as e:
                q.put_nowait(ErrorMsg(message=str(e)))
                continue
            if isinstance(msg, UserText):
                texts.put_nowait(msg.text)  # answered in order by reply_worker; other messages don't wait behind it
            else:
                await run_in_threadpool(core.handle, msg)

    async def reply_worker(core: RiyaCore, texts: asyncio.Queue) -> None:
        while True:
            text = await texts.get()
            try:
                await run_in_threadpool(speak, core.reply(text, "typed"))
            except Exception:  # a broken speaker must not close the socket
                log.exception("couldn't speak a typed reply")

    async def send(ws: WebSocket, q: asyncio.Queue) -> None:
        while True:
            await ws.send_text(dump(await q.get()))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        core: RiyaCore = app.state.core
        origin = ws.headers.get("origin")
        if origin is not None and origin not in BODY_ORIGINS:  # a web page in some browser tab: never let it in
            log.warning("rejected a websocket from origin %r", origin)
            await ws.close(code=1008)
            return
        await ws.accept()
        q = hub.connect()  # before hello, so nothing published in between is lost
        tasks: list[asyncio.Task] = []
        try:
            for msg in core.hello_messages():
                await ws.send_text(dump(msg))
            texts: asyncio.Queue = asyncio.Queue()
            tasks = [asyncio.create_task(receive(ws, core, q, texts)), asyncio.create_task(send(ws, q)),
                     asyncio.create_task(reply_worker(core, texts))]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() and not isinstance(t.exception(), WebSocketDisconnect):
                    log.warning("websocket closed: %r", t.exception())
        except WebSocketDisconnect:
            pass
        finally:
            for t in tasks:
                t.cancel()
            hub.disconnect(q)

    return app
