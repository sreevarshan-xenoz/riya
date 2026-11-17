"""Training batches from a small train set: sampling must not raise when rows are fewer than batch_size."""
import importlib
import random
import sys
import types

import pytest


@pytest.fixture
def voice(monkeypatch):
    stub = types.ModuleType("tiktoken")
    stub.get_encoding = lambda name: object()
    monkeypatch.setitem(sys.modules, "tiktoken", stub)
    sys.modules.pop("train_voice", None)
    try:
        yield importlib.import_module("train_voice")
    finally:
        sys.modules.pop("train_voice", None)


def test_train_sample_handles_small_and_empty_pools(voice):
    rng = random.Random(0)
    picked = voice.train_sample(["a", "b", "c"], 3, rng)
    assert sorted(picked) == ["a", "b", "c"]  # enough rows: sample without replacement

    picked = voice.train_sample(["a", "b"], 5, rng)
    assert len(picked) == 5 and set(picked) <= {"a", "b"}  # too few: sample with replacement

    with pytest.raises(ValueError, match="no training rows"):
        voice.train_sample([], 5, rng)
