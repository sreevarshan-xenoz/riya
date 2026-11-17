"""The affect-grounded SFT loop's validation slices: a small val set must not produce an empty batch."""
import contextlib
import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def voice(monkeypatch):
    # The trainer imports tiktoken, which is a training extra the test environment does not install.
    stub = types.ModuleType("tiktoken")
    stub.get_encoding = lambda name: object()
    monkeypatch.setitem(sys.modules, "tiktoken", stub)
    sys.modules.pop("train_voice", None)
    try:
        yield importlib.import_module("train_voice")
    finally:
        sys.modules.pop("train_voice", None)


class FakeModel:
    def eval(self):
        pass

    def train(self):
        pass

    def __call__(self, X, Y, affect):
        return None, torch.tensor(0.5)


def evaluate(voice, monkeypatch, n_val):
    seen = []

    def make_batch(batch, enc, args, device, drop):
        seen.append(len(batch))
        return (torch.zeros(1, 1, dtype=torch.long), torch.zeros(1, 1, dtype=torch.long),
                torch.zeros(1, 8))

    monkeypatch.setattr(voice, "make_batch", make_batch)
    args = SimpleNamespace(batch_size=8)
    rows = {"val": [{} for _ in range(n_val)]}
    out = voice.evaluate(FakeModel(), rows, None, args, "cpu", contextlib.nullcontext())
    return seen, out


def test_eval_slices_a_small_val_set_without_an_empty_batch(voice, monkeypatch):
    seen, out = evaluate(voice, monkeypatch, 20)
    assert seen == [8, 8, 4] * 2  # 20 rows in batches of 8, twice (with and without affect)
    assert 0 not in seen
    assert out == {"val": pytest.approx(0.5), "val_noaffect": pytest.approx(0.5)}


def test_eval_of_an_empty_val_set_reports_nan_without_batches(voice, monkeypatch):
    seen, out = evaluate(voice, monkeypatch, 0)
    assert seen == []
    assert out.keys() == {"val", "val_noaffect"}
    assert all(torch.isnan(torch.tensor(v)) for v in out.values())
