"""Local-vs-online routing: small talk stays on her own model, hard thinking goes online with local as the fallback."""
import logging
import re
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

    def stream(self, messages: list[dict], text: str, max_tokens: int = 300) -> Iterator[str]:
        if self.route(text) == "online":
            yield from self._online(messages, max_tokens)
        else:
            self.last_source = "local"
            yield from self.local.stream(messages, max_tokens, LOCAL_TEMPERATURE)

    def _online(self, messages, max_tokens) -> Iterator[str]:
        buf, started = "", False
        try:
            for piece in self.online.stream(messages, max_tokens):
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
