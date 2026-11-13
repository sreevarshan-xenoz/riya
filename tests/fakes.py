import re
import tempfile
import time
import uuid
from pathlib import Path

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.models import ModelUnavailable, ToolReply
from riya.core.presence import FakeIdleSource
from riya.core.store import KV, open_db


class FakeModel:
    def __init__(self, name="fake", replies=("Okay.",), fail=None, fail_after=None, delay=0.0, finish_reason="stop",
                 tool_replies=()):
        self.name, self.replies, self.fail, self.fail_after = name, list(replies), fail, fail_after
        self.finish_after, self.finish_reason = finish_reason, None  # finish_reason is set once a reply ran out
        self.delay = delay  # seconds to wait before each token, to make a reply slow
        self.calls, self.temperatures, self.max_tokens = [], [], []
        self.tool_replies, self.tool_calls = list(tool_replies), []  # a reply may be an exception instance (raised)

    def call_tools(self, messages, tools, timeout_s=8.0, max_tokens=200):
        self.tool_calls.append((messages, tools))
        if self.fail:
            raise self.fail
        if not self.tool_replies:
            return ToolReply([], None)
        r = self.tool_replies[min(len(self.tool_calls) - 1, len(self.tool_replies) - 1)]
        if isinstance(r, Exception):
            raise r
        return r

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.calls.append(messages)
        self.temperatures.append(temperature)
        self.max_tokens.append(max_tokens)
        self.finish_reason = None
        if self.fail:
            raise self.fail
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        for i, tok in enumerate(re.findall(r"\S+\s*", text)):
            if self.fail_after is not None and i == self.fail_after:
                raise ModelUnavailable("dropped mid-stream")
            if self.delay:
                time.sleep(self.delay)
            yield tok
        self.finish_reason = self.finish_after


def make_core(clock=None, local_replies=("Okay.",), online=None, tmp_path=None, tz=None, routine=False, **eyes):
    """A RiyaCore on fakes: returns (core, emitted messages, idle source, local model). eyes: vision, window_source,
    grab (no window and no vision model by default, so she never looks). Her hands are fake too: core.effects is a
    FakeEffects, core.http a FakeHttp with no answers, and her folder a fresh temporary path (made only on a write).
    routine: "She notices your routines" (off unless a test asks, so every older test is on v0.11's path)."""
    from riya.core.engine import RiyaCore
    from riya.core.hands.effects import FakeEffects
    from riya.core.hands.net import FakeHttp

    clock = clock or FakeClock()
    db = open_db(tmp_path / "riya.db" if tmp_path is not None else ":memory:")
    kv = KV(db)
    kv.set("settings", {**kv.get("settings", {}), "routine_enabled": routine})  # a reopened db keeps its settings
    riya_dir = (tmp_path if tmp_path is not None else Path(tempfile.gettempdir())) / f"riya-test-{uuid.uuid4().hex}"
    emitted, src, local = [], FakeIdleSource(), FakeModel("local", local_replies)
    core = RiyaCore(db=db, embedder=HashEmbedder(), clock=clock, idle_source=src, local=local, online=online,
                    emit=emitted.append, effects=FakeEffects(), http=FakeHttp({}), riya_dir=riya_dir, tz=tz, **eyes)
    core.twin_rng = FixedRng(0.0)  # the twin never suppresses unless a test says so
    return core, emitted, src, local


def wait_for(emitted, cls, timeout_s=5.0, after=0):
    """The first instance of cls in emitted at index >= after, polled every 10 ms (other threads append to it);
    AssertionError after timeout_s."""
    deadline = time.monotonic() + timeout_s
    while True:
        for m in list(emitted[after:]):
            if isinstance(m, cls):
                return m
        if time.monotonic() >= deadline:
            raise AssertionError(f"no {cls.__name__} within {timeout_s:g} s")
        time.sleep(0.01)


class FixedRng:
    """An rng whose random() always returns the same value."""

    def __init__(self, value: float = 0.0):
        self.value = value

    def random(self) -> float:
        return self.value
