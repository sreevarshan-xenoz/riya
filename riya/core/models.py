"""Chat-model adapters: any OpenAI-compatible server (local llama-server or an online free tier) behind one stream()."""
import codecs
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol
from urllib.parse import urlsplit

ONLINE_BASE_URL = "https://opencode.ai/inference/openai/v1"
ONLINE_MODEL = "glm-5.3-flash-free"  # free on OpenCode (zero-retention) and multimodal: one model can be her eyes
ONLINE_VISION_MODEL = ONLINE_MODEL
READ_TIMEOUT_S = 120.0
ONLINE_TIMEOUT_S = 12.0  # chat's first-token timeout (a free gateway can be slower than a first-party API)
VISION_TIMEOUT_S = 20.0  # an image request is slower to its first token than chat
MAX_REPLY_BYTES = 1 << 20  # a tool-call answer is tiny; never read more than this
# Cloudflare on opencode.ai answers urllib's default "Python-urllib/3.x" with 403; a normal client string is required.
USER_AGENT = "Riya/0.5 (personal desktop companion)"
LOCAL_STOP = ("\nUser:",)  # she must never write the user's next turn (the style examples' format)
LOCAL_TEMPERATURE = 0.6  # her voice on the small local model (tuned live in v0.2); online keeps the 0.8 default
log = logging.getLogger("riya.models")


class ModelUnavailable(Exception):
    """The model can't answer right now (network, rate limit, server error, or no first token in time)."""


@dataclass
class ToolReply:
    """A non-streaming tool-calling answer: each call is (tool name, JSON arguments string); text is a plain reply."""
    calls: list[tuple[str, str]]
    text: str | None = None


class ChatModel(Protocol):
    """messages are OpenAI chat dicts; a `content` may be a list of parts (text + image_url) for the online model,
    and is sent as-is (the local model is never sent images)."""
    name: str

    def stream(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.8) -> Iterator[str]: ...


class OpenAICompatModel:
    def __init__(self, name: str, base_url: str, api_key: str | None = None, model: str | None = None,
                 first_token_timeout_s: float = 8.0, stop: list[str] | None = None):
        self.name, self.base_url, self.api_key, self.model = name, base_url.rstrip("/"), api_key, model
        self.first_token_timeout_s, self.stop = first_token_timeout_s, stop
        self.finish_reason: str | None = None  # the last stream's ("length" means it ran out of tokens)

    def __repr__(self) -> str:  # never the key itself
        from riya.core.secrets import mask
        return f"OpenAICompatModel({self.name!r}, {self.base_url!r}, key={mask(self.api_key)!r}, model={self.model!r})"

    def stream(self, messages, max_tokens=300, temperature=0.8) -> Iterator[str]:
        self.finish_reason = None
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        if self.model is not None:
            body["model"] = self.model
        if self.stop:
            body["stop"] = self.stop
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=self.first_token_timeout_s) as r:
                started = False
                for raw in r:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:") or line.endswith("[DONE]"):
                        continue
                    piece, finish = _sse_event(line[5:])
                    self.finish_reason = finish or self.finish_reason
                    if piece:
                        if not started:
                            started = True
                            _set_read_timeout(r, READ_TIMEOUT_S)
                        yield piece
        except urllib.error.HTTPError as e:
            raise ModelUnavailable(f"{self.name}: HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:  # URLError, socket timeout, connection reset
            raise ModelUnavailable(f"{self.name}: {e}") from e
        except ValueError as e:  # a malformed event
            raise ModelUnavailable(f"{self.name}: bad stream event: {e}") from e


    def call_tools(self, messages, tools, timeout_s: float = 8.0, max_tokens: int = 200) -> ToolReply:
        """One non-streaming request offering `tools`; raises ModelUnavailable like stream()."""
        body = {"messages": messages, "tools": tools, "tool_choice": "auto", "max_tokens": max_tokens, "stream": False}
        if self.model is not None:
            body["model"] = self.model
        headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                return _tool_reply(json.loads(r.read(MAX_REPLY_BYTES)))
        except urllib.error.HTTPError as e:
            e.close()
            raise ModelUnavailable(f"{self.name}: HTTP {e.code}") from e
        except ModelUnavailable:
            raise
        except Exception as e:  # URLError, timeouts, IncompleteRead, bad JSON or shape, RecursionError: never leaks
            raise ModelUnavailable(f"{self.name}: {type(e).__name__}: {str(e)[:80]}") from e


def local_model(server_url: str, **kw) -> OpenAICompatModel:
    """Her own llama-server (the base URL, without /v1), with the stop sequences."""
    return OpenAICompatModel("local", f"{server_url.rstrip('/')}/v1", stop=list(LOCAL_STOP), **kw)


def _sse_event(data: str) -> tuple[str | None, str | None]:
    """(content, finish_reason) of one streamed chunk; ValueError when the event isn't the OpenAI chunk shape."""
    event = json.loads(data)
    if not isinstance(event, dict) or "error" in event:
        raise ValueError(f"not a chunk: {data[:80]!r}")
    choices = event.get("choices")
    if not choices:  # e.g. a usage-only chunk
        return None, None
    choice = choices[0] if isinstance(choices, list) and isinstance(choices[0], dict) else None
    delta = choice.get("delta") if choice else None
    if not isinstance(delta, dict):
        raise ValueError(f"no delta in {data[:80]!r}")
    piece, finish = delta.get("content"), choice.get("finish_reason")
    return piece if isinstance(piece, str) else None, finish if isinstance(finish, str) else None


def _tool_reply(data) -> ToolReply:
    """The ToolReply of one non-streaming chat completion; ValueError when it isn't that shape."""
    choices = data.get("choices") if isinstance(data, dict) else None
    msg = choices[0].get("message") if isinstance(choices, list) and choices and isinstance(choices[0], dict) else None
    if not isinstance(msg, dict):
        raise ValueError(f"no message in {str(data)[:80]!r}")
    calls = []
    for c in msg.get("tool_calls") or []:
        fn = c.get("function") if isinstance(c, dict) else None
        fn = fn if isinstance(fn, dict) else {}
        name, args = fn.get("name"), fn.get("arguments")
        # a malformed entry still counts as a call (empty name), so two calls can never pass as one
        calls.append((name if isinstance(name, str) else "", args if isinstance(args, str) else
                      "" if args is None else json.dumps(args)))
    text = msg.get("content")
    return ToolReply(calls, text if isinstance(text, str) else None)


def _set_read_timeout(resp, seconds: float) -> None:
    try:
        resp.fp.raw._sock.settimeout(seconds)
    except AttributeError:  # unusual response object: keep the connection timeout
        pass


def complete(model: ChatModel, messages: list[dict], max_tokens: int = 300) -> str:
    return "".join(model.stream(messages, max_tokens))


def _read_text(path: str) -> str:
    """UTF-8, with or without a BOM, or UTF-16 with its BOM (Notepad's "Unicode")."""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig")


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Parse KEY=VALUE lines and export them without overriding variables that are already set. Never raises: an
    unreadable file is logged and ignored."""
    found: dict[str, str] = {}
    try:
        lines = _read_text(path).splitlines()
    except OSError:
        return found
    except UnicodeDecodeError as e:
        log.warning("ignoring %s: it isn't UTF-8 or UTF-16 text (%s)", path, e)
        return found
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            found[key] = value
            os.environ.setdefault(key, value)
    return found


def online_base(env: Mapping[str, str]) -> str:
    """The online endpoint from RIYA_ONLINE_BASE_URL, else OpenCode's."""
    return (env.get("RIYA_ONLINE_BASE_URL") or ONLINE_BASE_URL).strip() or ONLINE_BASE_URL


def keyless_base(base_url: str) -> bool:
    """Whether the endpoint serves its free models without a key (any opencode.ai host)."""
    try:
        host = (urlsplit(base_url if "//" in base_url else "//" + base_url).hostname or "").lower()
    except ValueError:
        return False
    return host == "opencode.ai" or host.endswith(".opencode.ai")


def online_available(env: Mapping[str, str]) -> bool:
    """Whether an online model can be built: a key is set, the endpoint is keyless, or RIYA_ONLINE_KEYLESS=1. An
    explicitly blank RIYA_ONLINE_API_KEY or RIYA_ONLINE_OFFLINE=1 turns the online brain off (the check scripts use
    that to stay offline, even when a saved key exists)."""
    if env.get("RIYA_ONLINE_OFFLINE") == "1":
        return False
    if env.get("RIYA_ONLINE_API_KEY") is not None and not env["RIYA_ONLINE_API_KEY"].strip():
        return False
    return (bool((env.get("RIYA_ONLINE_API_KEY") or "").strip()) or env.get("RIYA_ONLINE_KEYLESS") == "1"
            or keyless_base(online_base(env)))


def load_online_from_env(env=os.environ, name: str = "online", model: str | None = None,
                         **kw) -> OpenAICompatModel | None:
    """The online model from RIYA_ONLINE_*; kw (e.g. first_token_timeout_s) go to OpenAICompatModel. `model` overrides
    RIYA_ONLINE_MODEL (her eyes have their own, RIYA_ONLINE_VISION_MODEL). No key is needed on a keyless endpoint."""
    if not online_available(env):
        return None
    key = (env.get("RIYA_ONLINE_API_KEY") or "").strip() or None
    chosen = model or env.get("RIYA_ONLINE_MODEL") or ONLINE_MODEL
    return OpenAICompatModel(name, online_base(env), key, chosen, **kw)


def online_model(env: Mapping[str, str], name: str, timeout_s: float) -> OpenAICompatModel | None:
    model = None
    if name == "vision":
        model = (env.get("RIYA_ONLINE_VISION_MODEL") or "").strip() or ONLINE_VISION_MODEL
    return load_online_from_env(env, name=name, model=model, first_token_timeout_s=timeout_s)


def online_pair(env: Mapping[str, str], make: Callable[[Mapping[str, str], str, float], ChatModel | None] = online_model
                ) -> tuple[ChatModel | None, ChatModel | None]:
    """The online chat model and her eyes' own vision instance (a longer first-token timeout), the one place both are
    built, at start and when the key changes."""
    return make(env, "online", ONLINE_TIMEOUT_S), make(env, "vision", VISION_TIMEOUT_S)


def vision_from_env(env=os.environ, timeout_s: float = VISION_TIMEOUT_S) -> OpenAICompatModel | None:
    """The vision model alone (for the eye-check script), with RIYA_ONLINE_VISION_MODEL as its model."""
    return online_model(env, "vision", timeout_s)
