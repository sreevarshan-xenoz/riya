"""Local-vs-online routing: small talk stays on her own model, hard thinking goes online with local as the fallback."""
import logging
import queue
import re
import threading
from typing import Iterator

from riya.bench import REFUSAL_PATTERNS
from riya.core.appraise import SENTENCE_END
from riya.core.models import LOCAL_TEMPERATURE, ChatModel

log = logging.getLogger("riya.router")
BUFFER_CHARS = 160
THINKING = re.compile(r"\b(debug|traceback|error|exception|explain|how (do|does|can|to)|why (is|does|do)|"
                      r"write (a|me|the)|code|function|script|plan|compare|calculate|translate|summari[sz]e)\b", re.I)


def needs_thinking(text: str) -> bool:
    return bool("`" in text or "think hard" in text.lower() or THINKING.search(text)
                or (len(text) > 200 and text.rstrip().endswith("?")))


class Router:
    def __init__(self, local: ChatModel, online: ChatModel | None = None, online_enabled: bool = True):
        self.local, self.online, self.online_enabled = local, online, online_enabled
        self.last_source = "local"

    def route(self, text: str) -> str:
        return "online" if self.online is not None and self.online_enabled and needs_thinking(text) else "local"

    @property
    def last_finish_reason(self) -> str | None:
        """Why the last answer's model stopped ("length": it ran out of tokens), when it says."""
        return getattr(self.online if self.last_source == "online" else self.local, "finish_reason", None)

    def stream(self, messages: list[dict], text: str, max_tokens: int = 300,
               online_tokens: int | None = None, route: str | None = None) -> Iterator[str]:
        """online_tokens, when given, is the online model's budget; a local fallback keeps max_tokens. route: the route
        the caller already decided (and built its prompt for); None decides here."""
        if (route or self.route(text)) == "online":
            yield from self._online(messages, max_tokens, online_tokens or max_tokens)
        else:
            self.last_source = "local"
            yield from self.local.stream(messages, max_tokens, LOCAL_TEMPERATURE)

    def _online(self, messages, max_tokens, online_tokens) -> Iterator[str]:
        buf, started = "", False
        try:
            for piece in self.online.stream(messages, online_tokens):
                if started:
                    yield piece
                    continue
                buf += piece
                if SENTENCE_END.search(buf) or len(buf) >= BUFFER_CHARS:
                    if REFUSAL_PATTERNS.search(buf):
                        log.info("online model refused; falling back to local")
                        break
                    started, self.last_source = True, "online"
                    yield buf
            else:  # stream ended
                if started:
                    return
                if buf and not REFUSAL_PATTERNS.search(buf):
                    self.last_source = "online"
                    yield buf
                    return
        except Exception as e:
            if started:
                log.warning("online stream ended early: %s", e)
                return
            log.warning("online model failed, falling back to local: %s", e)
        self.last_source = "local"
        yield from self.local.stream(messages, max_tokens, LOCAL_TEMPERATURE)


class Prefetched:
    """A stream run in a background thread, so the caller can wait for its first piece with a timeout (and say
    something meanwhile). Pieces and the stream's exception come out of iteration in order; leaving the iteration
    early stops the worker at its next piece."""

    def __init__(self, source: Iterator[str]):
        self._q: queue.Queue = queue.Queue()
        self._first, self._stop = threading.Event(), threading.Event()
        threading.Thread(target=self._run, args=(source,), daemon=True, name="riya-prefetch").start()

    def _run(self, source: Iterator[str]) -> None:
        try:
            for piece in source:
                if self._stop.is_set():
                    close = getattr(source, "close", None)
                    if close:
                        close()
                    return
                self._q.put(("piece", piece))
                self._first.set()
        except BaseException as e:  # handed to the reader, raised where the stream would have raised it
            self._q.put(("error", e))
        finally:
            self._q.put(("end", None))
            self._first.set()

    def ready(self, timeout_s: float) -> bool:
        """True once the first piece (or the end, or an error) is in; False after timeout_s without one."""
        return self._first.wait(timeout_s)

    def __iter__(self) -> Iterator[str]:
        try:
            while True:
                kind, value = self._q.get()
                if kind == "piece":
                    yield value
                elif kind == "error":
                    raise value
                else:
                    return
        finally:
            self._stop.set()
