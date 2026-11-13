"""The gate harness runs with routine off (v0.12 §4.12.2, Resolution 5)."""
from evaluation import persona_eval
from riya.core.engine import RiyaCore
from tests.fakes import FakeModel


def test_the_gate_harness_runs_with_routine_off(tmp_path, monkeypatch):
    seen = []

    class Recording(RiyaCore):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen.append(self.settings.routine_enabled)

    monkeypatch.setattr(persona_eval, "RiyaCore", Recording)
    conv = {"id": "t", "facts": [], "turns": [{"id": "hi", "kind": "small", "text": "hi riya"}]}
    rows = persona_eval.run_conversation(conv, FakeModel("local", ("Hi!",)), None, str(tmp_path))
    assert len(rows) == 1
    assert seen == [False]
    assert persona_eval.GATE_SETTINGS == {"routine_enabled": False}
