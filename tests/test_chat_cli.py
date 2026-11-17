"""The standalone text CLI's timing line: an empty reply must not crash it."""
import sys

import pytest

from riya import chat as chat_cli


class Done(Exception):
    pass


def run_cli(monkeypatch, capsys, sentences):
    class FakeRiya:
        def reply(self, user):
            return iter(sentences)

        def state_tag(self):
            return "state"

    monkeypatch.setattr(chat_cli, "Riya", lambda *a, **kw: FakeRiya())
    answers = iter(["hi", None])

    def fake_input(prompt=""):
        value = next(answers)
        if value is None:
            raise Done
        return value

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "argv", ["riya.chat"])
    with pytest.raises(Done):
        chat_cli.main()
    return capsys.readouterr().out


def test_empty_reply_reports_no_reply_instead_of_crashing(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, [])
    assert "(no reply)" in out
    assert "state" not in out


def test_a_reply_still_reports_the_first_sentence_latency(monkeypatch, capsys):
    out = run_cli(monkeypatch, capsys, ["Hello.", "How are you?"])
    assert "ms to first sentence" in out
    assert "Hello." in out
