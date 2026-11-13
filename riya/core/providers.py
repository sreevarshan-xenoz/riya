"""Free online model providers for the pool (docs/superpowers/specs/2026-11-13-riya-model-pool-design.md).

Each provider is an OpenAI-compatible endpoint that joins the pool once its key is in providers.env (next to
secrets.env, one NAME=value per line) or the process environment. Limits as published on 2026-11-13: Cerebras about
1M tokens a day per model, Groq 200K, Cloudflare 10,000 neurons a day for the account, OpenRouter 50 requests a day
for the account; the rest aren't published. `<ID>_MODELS` (e.g. GROQ_MODELS=a,b) replaces a provider's model list.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from riya.core.models import ChatModel, OpenAICompatModel
from riya.core.pool import Member
from riya.paths import data_dir

log = logging.getLogger("riya.providers")
KEYS_NAME = "providers.env"


@dataclass(frozen=True)
class Provider:
    id: str
    base_url: str  # may name {CLOUDFLARE_ACCOUNT_ID}-style variables from the keys
    key_var: str
    models: tuple[tuple[str, bool], ...]  # (model id, takes images), best first
    shared: bool = False  # one allowance for the whole account: any limit benches all its models
    needs: tuple[str, ...] = ()  # other variables the base_url names

    @property
    def models_var(self) -> str:
        return f"{self.id.upper()}_MODELS"


# The order is the pool's: biggest free budgets and fastest first.
PROVIDERS: tuple[Provider, ...] = (
    Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY",
             (("gpt-oss-120b", False), ("qwen-3.8-27b", True))),
    Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY",
             (("openai/gpt-oss-120b", False), ("qwen/qwen3.8-27b", False), ("openai/gpt-oss-20b", False))),
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY",
             (("gemini-flash-lite-latest", True), ("gemini-flash-latest", True))),
    Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY",
             (("mistral-medium-latest", False), ("mistral-small-latest", True))),
    Provider("sambanova", "https://api.sambanova.ai/v1", "SAMBANOVA_API_KEY",
             (("gpt-oss-120b", False), ("Meta-Llama-3.3-70B-Instruct", False))),
    Provider("nvidia", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY",
             (("openai/gpt-oss-120b", False), ("meta/llama-3.3-70b-instruct", False)), shared=True),
    Provider("cloudflare", "https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/v1",
             "CLOUDFLARE_API_TOKEN", (("@cf/openai/gpt-oss-120b", False), ("@cf/openai/gpt-oss-20b", False)),
             shared=True, needs=("CLOUDFLARE_ACCOUNT_ID",)),
    Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY",
             (("nvidia/nemotron-3-ultra-550b-a55b:free", False), ("google/gemma-4-31b-it:free", True),
              ("openrouter/free", True)), shared=True),
)
KEY_VARS = frozenset(v for p in PROVIDERS for v in (p.key_var, *p.needs, p.models_var))


def keys_path() -> Path:
    return data_dir() / KEYS_NAME


def load_keys(path: Path | None = None) -> dict[str, str]:
    """The known NAME=value lines of providers.env (last one wins; quotes stripped). Unknown names are ignored; a
    missing or unreadable file is no keys. Never logs a value."""
    path = Path(path or keys_path())
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return {}
    except (OSError, UnicodeDecodeError) as e:
        log.warning("can't read %s (%s)", path, type(e).__name__)
        return {}
    out = {}
    for line in lines:
        name, sep, value = line.strip().partition("=")
        name, value = name.strip().removeprefix("export ").strip(), value.strip()
        if not sep or name not in KEY_VARS or line.lstrip().startswith("#"):
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            out[name] = value
    return out


def keys_token(path: Path | None = None) -> float | None:
    """What changes when providers.env does (its mtime), for the pool's live reload."""
    try:
        return Path(path or keys_path()).stat().st_mtime
    except OSError:
        return None


def members(keys: Mapping[str, str], name: str, timeout_s: float, vision_only: bool = False,
            make: Callable[..., ChatModel] = OpenAICompatModel) -> list[Member]:
    """Pool members for every provider whose key (and any variable its URL needs) is present."""
    from riya.core.secrets import remember
    out = []
    for p in PROVIDERS:
        key = (keys.get(p.key_var) or "").strip()
        if not key or any(not (keys.get(v) or "").strip() for v in p.needs):
            continue
        remember(key)  # her logs never show it
        base = p.base_url.format(**{v: keys[v].strip() for v in p.needs})
        listed = p.models
        if (keys.get(p.models_var) or "").strip():
            listed = tuple((m.strip(), True) for m in keys[p.models_var].split(",") if m.strip())
        for model_id, vision in listed:
            if vision_only and not vision:
                continue
            out.append(Member(make(name, base, key, model_id, first_token_timeout_s=timeout_s), vision=vision,
                              group=p.id, shared=p.shared, provider=p.id))
    return out


def all_keys(environ: Mapping[str, str], path: Path | None = None) -> dict[str, str]:
    """providers.env over the process environment (the file is what the user edits)."""
    out = {k: v for k, v in environ.items() if k in KEY_VARS and v}
    out.update(load_keys(path))
    return out


def configured(environ: Mapping[str, str] = os.environ, path: Path | None = None) -> list[str]:
    """The ids of providers that would join the pool now (for Health and the setup check)."""
    keys = all_keys(environ, path)
    return [p.id for p in PROVIDERS if keys.get(p.key_var) and all(keys.get(v) for v in p.needs)]
