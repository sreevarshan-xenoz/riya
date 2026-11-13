"""Free online models that take over for each other (docs/superpowers/specs/2026-11-13-riya-model-pool-design.md).

A ModelPool looks like one online model to the rest of Riya. stream() tries its members in order until one yields
its first piece, then stays with it; a member that hits a limit is benched (Retry-After, else 10 minutes doubling
up to 6 hours), one that errors or times out for a minute, and a benched member is tried again the first time the
pool reaches it after its cooldown. The order is sticky: it starts from the member that last answered. Slow members
(CLI agents) come after every fast one.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterator

from riya.core.models import ChatModel, ModelUnavailable, ToolReply

LIMIT_BENCH_S = 600.0  # a limit without Retry-After: 10 minutes, doubling on each repeat
LIMIT_BENCH_MAX_S = 6 * 3600.0
ERROR_BENCH_S = 60.0  # a server error, a timeout, a dropped connection
GONE_BENCH_S = 6 * 3600.0  # a rejected key (the whole provider) or a model the provider doesn't have
RELOAD_EVERY_S = 5.0  # how often the pool looks for changed keys
_SHARED_LIMIT = re.compile(r"FreeUsageLimit|free usage limit", re.I)  # the whole tier, not one model
_LIMIT_WORDS = re.compile(r"rate.?limit|quota|too many requests|limit (exceeded|reached)|usage limit|insufficient",
                          re.I)
log = logging.getLogger("riya.pool")


@dataclass
class Member:
    model: ChatModel
    vision: bool = False
    tools: bool = True
    slow: bool = False
    group: str | None = None  # members sharing one allowance (opencode's free tier is one limit for all its models)
    shared: bool = False  # any limit is the whole group's (an account-wide allowance)
    provider: str | None = None  # a rejected key benches every member of the same provider
    benched_until: float = 0.0
    limits: int = 0  # consecutive limits (each doubles the next bench)
    last_error: str | None = None

    @property
    def name(self) -> str:
        return getattr(self.model, "model", None) or self.model.name

    @property
    def key(self) -> tuple[str | None, str]:
        return self.provider, self.name


@dataclass
class MemberState:
    """What Health shows for one member."""
    provider: str | None
    name: str
    live: bool
    benched_for_s: float
    slow: bool
    last_error: str | None


def kind_of(e: ModelUnavailable) -> str:
    """'limit', 'auth' (the key was rejected), 'missing' (no such model), 'bad_request' or 'error' from a
    ModelUnavailable's status and message."""
    status = getattr(e, "status", None)
    text = f"{e} {getattr(e, 'detail', '') or ''}"
    if status == 429 or (status in (402, 403) and _LIMIT_WORDS.search(text)) or (status is None and
                                                                                  _LIMIT_WORDS.search(text)):
        return "limit"
    if status == 401:
        return "auth"
    if status == 404:
        return "missing"
    if status in (400, 405, 413, 422):
        return "bad_request"
    return "error"


class ModelPool:
    def __init__(self, name: str, members: list[Member], clock: Callable[[], float] = time.monotonic,
                 refresh: Callable[[], list[Member]] | None = None,
                 extra: Callable[[], list[Member]] | None = None, token: Callable[[], object] | None = None):
        """refresh: the live member list, read once on first use. extra: members that come and go with the keys
        (providers.env), rebuilt whenever token() changes (checked at most every RELOAD_EVERY_S); a member that
        stays keeps its bench."""
        self.name, self.clock = name, clock
        self.finish_reason: str | None = None
        self.last: Member | None = None  # the member that answered last
        self._refresh, self._refreshed = refresh, refresh is None
        self._extra, self._token = extra, token
        self._seen_token, self._checked_at = object(), float("-inf")
        self._lock = threading.Lock()
        self._base = list(members)
        self.members = list(members)
        self._maybe_reload()
        if not self.members:
            raise ValueError("a pool needs at least one member")

    def __getattr__(self, attr):
        """api_key, base_url, model, first_token_timeout_s...: the first member's (what the configured model was)."""
        if attr.startswith("_") or attr in ("members",):
            raise AttributeError(attr)
        return getattr(self.members[0].model, attr)

    def _maybe_reload(self) -> None:
        if self._extra is None:
            return
        now = self.clock()
        if now - self._checked_at < RELOAD_EVERY_S:
            return
        self._checked_at = now
        token = self._token() if self._token else None
        if token == self._seen_token:
            return
        self._seen_token = token
        try:
            fresh = self._extra()
        except Exception as e:  # a broken keys file never stops a reply
            log.warning("the provider keys couldn't be read (%s)", type(e).__name__)
            return
        with self._lock:
            old = {m.key: m for m in self.members}
            extra = [old.get(m.key, m) for m in fresh]
            extra_keys = {m.key for m in extra}
            self.members = [m for m in self._base if m.key not in extra_keys] + extra
            if self.last not in self.members:
                self.last = None
        log.info("%s: %d models from %d providers", self.name, len(self.members),
                 len({m.provider for m in self.members}))

    def _maybe_refresh(self) -> None:
        self._maybe_reload()
        if self._refreshed:
            return
        self._refreshed = True
        try:
            fresh = self._refresh() if self._refresh else []
        except Exception as e:  # never stops a reply
            log.info("the model list couldn't be refreshed (%s); keeping %d members", type(e).__name__,
                     len(self.members))
            return
        with self._lock:
            known = {m.key for m in self.members}
            new = [m for m in fresh if m.key not in known]
            self._base += new
            first_extra = next((i for i, m in enumerate(self.members) if m not in self._base), len(self.members))
            self.members[first_extra:first_extra] = new  # the refreshed base stays ahead of the keyed providers

    def _order(self, need_tools: bool = False) -> list[Member]:
        with self._lock:
            members = [m for m in self.members if m.tools or not need_tools]
            if self.last in members:
                i = members.index(self.last)
                members = members[i:] + members[:i]
            return [m for m in members if not m.slow] + [m for m in members if m.slow]

    def _bench(self, m: Member, e: ModelUnavailable) -> None:
        kind = kind_of(e)
        with self._lock:
            m.last_error = str(e)[:120]
            if kind == "limit":
                retry = getattr(e, "retry_after_s", None)
                wait = retry if retry else min(LIMIT_BENCH_S * 2 ** m.limits, LIMIT_BENCH_MAX_S)
                m.limits += 1
                if m.group and (m.shared or _SHARED_LIMIT.search(f"{e} {getattr(e, 'detail', '') or ''}")):
                    until = self.clock() + wait  # one allowance for the group: bench them all at once
                    for other in self.members:
                        if other.group == m.group:
                            other.benched_until, other.last_error = until, m.last_error
                    log.info("%s's shared limit is used up; its %s members are benched for %.0f s", m.group,
                             sum(o.group == m.group for o in self.members), wait)
                    return
            elif kind == "auth" and m.provider:
                until = self.clock() + GONE_BENCH_S  # the key is the provider's: none of its models will answer
                for other in self.members:
                    if other.provider == m.provider:
                        other.benched_until, other.last_error = until, m.last_error
                log.warning("%s rejected its key; its models are benched for %.0f s", m.provider, GONE_BENCH_S)
                return
            elif kind in ("missing", "auth"):
                wait = GONE_BENCH_S
            elif kind == "error":
                wait = ERROR_BENCH_S
            else:
                return  # a bad request: this request's fault, not the member's
            m.benched_until = self.clock() + wait
        log.info("%s benched for %.0f s (%s)", m.name, wait, kind)

    def _ok(self, m: Member) -> None:
        with self._lock:
            m.benched_until, m.limits, m.last_error = 0.0, 0, None
            self.last = m

    def _live(self, m: Member) -> bool:
        return m.benched_until <= self.clock()

    def stream(self, messages: list[dict], max_tokens: int = 300, temperature: float = 0.8) -> Iterator[str]:
        self._maybe_refresh()
        self.finish_reason = None
        tried: list[str] = []
        for m in self._order():
            if not self._live(m):
                continue
            pieces = m.model.stream(messages, max_tokens, temperature)
            try:
                first = next(pieces)
            except StopIteration:
                self._ok(m)
                self.finish_reason = getattr(m.model, "finish_reason", None)
                return
            except ModelUnavailable as e:
                tried.append(m.name)
                self._bench(m, e)
                continue
            self._ok(m)
            yield first
            yield from pieces  # committed: a failure from here on is the reply's
            self.finish_reason = getattr(m.model, "finish_reason", None)
            return
        raise ModelUnavailable(f"{self.name}: every model is limited or down"
                               + (f" (tried {', '.join(tried)})" if tried else ""))

    def call_tools(self, messages, tools, timeout_s: float = 8.0, max_tokens: int = 200) -> ToolReply:
        self._maybe_refresh()
        for m in self._order(need_tools=True):
            if not self._live(m) or not hasattr(m.model, "call_tools"):
                continue
            try:
                reply = m.model.call_tools(messages, tools, timeout_s=timeout_s, max_tokens=max_tokens)
            except ModelUnavailable as e:
                if kind_of(e) == "bad_request":
                    with self._lock:
                        m.tools = False  # it doesn't take tools; it can still chat
                self._bench(m, e)
                continue
            self._ok(m)
            return reply
        raise ModelUnavailable(f"{self.name}: no model can call tools right now")

    def states(self) -> list[MemberState]:
        now = self.clock()
        with self._lock:
            return [MemberState(m.provider, m.name, m.benched_until <= now, max(0.0, m.benched_until - now), m.slow, m.last_error)
                    for m in self.members]


# opencode's free models (2026-11-13); the live /models list replaces this on first use. True: it takes images.
OPENCODE_FREE = {
    "glm-5.3-flash-free": True, "mimo-v2.6-flash-free": True, "deepseek-v4.1-flash-free": True,
    "minimax-m3.1-flash-free": True, "longcat-2.5-free": True, "inkling-2-small-free": True,
    "nemotron-3.5-super-vl-free": True, "nemotron-3.5-lightning-free": False, "iceman-free": False,
    "beam-free": False, "flint-40b-free": False,
}
MODELS_TIMEOUT_S = 8.0


def opencode_free_ids(base_url: str, timeout_s: float = MODELS_TIMEOUT_S) -> dict[str, bool]:
    """The live free list from the endpoint's /models: id -> takes images. Raises on any failure."""
    import json
    import urllib.request

    from riya.core.models import MAX_REPLY_BYTES, USER_AGENT
    req = urllib.request.Request(f"{base_url.rstrip('/')}/models", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read(MAX_REPLY_BYTES))
    out = {}
    for m in data.get("data", []) if isinstance(data, dict) else []:
        mid = m.get("id") if isinstance(m, dict) else None
        if isinstance(mid, str) and mid.endswith("-free"):
            inputs = (m.get("modalities") or {}).get("input") or []
            out[mid] = any(str(i).startswith("image") for i in inputs)
    if not out:
        raise ValueError("no free models listed")
    return out


def free_pool(name: str, make: Callable[[str], ChatModel], first: str, vision_only: bool = False,
              list_ids: Callable[[], dict[str, bool]] | None = None, clock: Callable[[], float] = time.monotonic,
              extra: Callable[[], list[Member]] | None = None, token: Callable[[], object] | None = None
              ) -> ModelPool:
    """A pool over opencode's free models, `first` (the configured one) leading. make(model_id) builds a member's
    client; list_ids reads the live list on first use (the built-in OPENCODE_FREE until then)."""
    def members(ids: dict[str, bool]) -> list[Member]:
        chosen = [first] + [i for i in ids if i != first]
        return [Member(make(i), vision=ids.get(i, True), group="opencode", provider="opencode") for i in chosen
                if ids.get(i, True) or not vision_only]
    refresh = (lambda: members(list_ids())) if list_ids else None
    return ModelPool(name, members(OPENCODE_FREE), clock=clock, refresh=refresh, extra=extra, token=token)
