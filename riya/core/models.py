"""Chat-model adapters: any OpenAI-compatible server (local llama-server or an online free tier) behind one stream()."""
import codecs
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Iterator, Protocol

ONLINE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ONLINE_MODEL = "gemini-2.5-flash"
READ_TIMEOUT_S = 120.0
LOCAL_STOP = ("\nUser:",)  # she must never write the user's next turn (the style examples' format)
LOCAL_TEMPERATURE = 0.6  # her voice on the small local model (tuned live in v0.2); online keeps the 0.8 default
log = logging.getLogger("riya.models")


class ModelUnavailable(Exception):
    """The model can't answer right now (network, rate limit, server error, or no first token in time)."""


class ChatModel(Protocol):
    name: str

    def stream(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.8) -> Iterator[str]: ...


class OpenAICompatModel:
    def __init__(self, name: str, base_url: str, api_key: str | None = None, model: str | None = None,
                 first_token_timeout_s: float = 8.0, stop: list[str] | None = None):
        self.name, self.base_url, self.api_key, self.model = name, base_url.rstrip("/"), api_key, model
        self.first_token_timeout_s, self.stop = first_token_timeout_s, stop

    def stream(self, messages, max_tokens=300, temperature=0.8) -> Iterator[str]:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        if self.model is not None:
            body["model"] = self.model
        if self.stop:
            body["stop"] = self.stop
        headers = {"Content-Type": "application/json"}
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
                    piece = _sse_piece(line[5:])
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


def local_model(server_url: str, **kw) -> OpenAICompatModel:
    """Her own llama-server (the base URL, without /v1), with the stop sequences."""
    return OpenAICompatModel("local", f"{server_url.rstrip('/')}/v1", stop=list(LOCAL_STOP), **kw)


def _sse_piece(data: str) -> str | None:
    """The content of one streamed chunk; ValueError when the event isn't the OpenAI chunk shape."""
    event = json.loads(data)
    if not isinstance(event, dict) or "error" in event:
        raise ValueError(f"not a chunk: {data[:80]!r}")
    choices = event.get("choices")
    if not choices:  # e.g. a usage-only chunk
        return None
    delta = choices[0].get("delta") if isinstance(choices, list) and isinstance(choices[0], dict) else None
    if not isinstance(delta, dict):
        raise ValueError(f"no delta in {data[:80]!r}")
    piece = delta.get("content")
    return piece if isinstance(piece, str) else None


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


def load_online_from_env(env=os.environ) -> OpenAICompatModel | None:
    key = (env.get("RIYA_ONLINE_API_KEY") or "").strip()
    if not key:
        return None
    return OpenAICompatModel("online", env.get("RIYA_ONLINE_BASE_URL") or ONLINE_BASE_URL, key,
                             env.get("RIYA_ONLINE_MODEL") or ONLINE_MODEL)
