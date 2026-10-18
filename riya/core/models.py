"""Chat-model adapters: any OpenAI-compatible server (local llama-server or an online free tier) behind one stream()."""
import json
import os
import urllib.error
import urllib.request
from typing import Iterator, Protocol

ONLINE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
ONLINE_MODEL = "gemini-2.5-flash"
READ_TIMEOUT_S = 120.0


class ModelUnavailable(Exception):
    """The model can't answer right now (network, rate limit, server error, or no first token in time)."""


class ChatModel(Protocol):
    name: str

    def stream(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.8) -> Iterator[str]: ...


class OpenAICompatModel:
    def __init__(self, name: str, base_url: str, api_key: str | None = None, model: str | None = None,
                 first_token_timeout_s: float = 8.0):
        self.name, self.base_url, self.api_key, self.model = name, base_url.rstrip("/"), api_key, model
        self.first_token_timeout_s = first_token_timeout_s

    def stream(self, messages, max_tokens=300, temperature=0.8) -> Iterator[str]:
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        if self.model is not None:
            body["model"] = self.model
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
                    choices = json.loads(line[5:]).get("choices") or [{}]
                    piece = choices[0].get("delta", {}).get("content")
                    if piece:
                        if not started:
                            started = True
                            _set_read_timeout(r, READ_TIMEOUT_S)
                        yield piece
        except urllib.error.HTTPError as e:
            raise ModelUnavailable(f"{self.name}: HTTP {e.code}") from e
        except (urllib.error.URLError, OSError) as e:  # URLError, socket timeout, connection reset
            raise ModelUnavailable(f"{self.name}: {e}") from e


def _set_read_timeout(resp, seconds: float) -> None:
    try:
        resp.fp.raw._sock.settimeout(seconds)
    except AttributeError:  # unusual response object: keep the connection timeout
        pass


def complete(model: ChatModel, messages: list[dict], max_tokens: int = 300) -> str:
    return "".join(model.stream(messages, max_tokens))


def load_dotenv(path: str = ".env") -> dict[str, str]:
    """Parse KEY=VALUE lines and export them without overriding variables that are already set."""
    found: dict[str, str] = {}
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
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
