import re
import time

from riya.core.clock import FakeClock
from riya.core.embed import HashEmbedder
from riya.core.models import ModelUnavailable
from riya.core.presence import FakeIdleSource
from riya.core.store import open_db


class FakeModel:
    def __init__(self, name="fake", replies=("Okay.",), fail=None, fail_after=None, delay=0.0):
        self.name, self.replies, self.fail, self.fail_after = name, list(replies), fail, fail_after
        self.delay = delay  # seconds to wait before each token, to make a reply slow
        self.calls, self.temperatures = [], []

    def stream(self, messages, max_tokens=300, temperature=0.8):
        self.calls.append(messages)
        self.temperatures.append(temperature)
        if self.fail:
            raise self.fail
        text = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        for i, tok in enumerate(re.findall(r"\S+\s*", text)):
            if self.fail_after is not None and i == self.fail_after:
                raise ModelUnavailable("dropped mid-stream")
            if self.delay:
                time.sleep(self.delay)
            yield tok


def make_core(clock=None, local_replies=("Okay.",), online=None, tmp_path=None):
    """A RiyaCore on fakes: returns (core, emitted messages, idle source, local model)."""
    from riya.core.engine import RiyaCore

    clock = clock or FakeClock()
    db = open_db(tmp_path / "riya.db" if tmp_path is not None else ":memory:")
    emitted, src, local = [], FakeIdleSource(), FakeModel("local", local_replies)
    core = RiyaCore(db=db, embedder=HashEmbedder(), clock=clock, idle_source=src, local=local, online=online,
                    emit=emitted.append)
    return core, emitted, src, local
