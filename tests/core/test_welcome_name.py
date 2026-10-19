"""The welcome's name step goes through the existing remember path: she acks aloud and the fact is `user_said`."""
from tests.fakes import make_core


def test_the_welcome_name_is_saved_through_the_remember_path():
    core, *_ = make_core()
    list(core.reply("remember that my name is Sree"))
    facts = [f for f in core.memory.list_facts() if f.text == "The user said: their name is Sree"]
    assert len(facts) == 1 and facts[0].source == "user_said"


def test_a_key_pasted_as_a_name_is_refused_and_never_kept():
    """The welcome's name field is the step before the key: a key pasted there must not become a fact, a transcript
    row or part of the model's prompt. She says why instead."""
    from riya.core import secrets as keystore
    from riya.core.engine import KEY_NOT_REMEMBERED
    key = "AIzaSyD-0123456789abcdefghijklmnopqrstu"
    core, emitted, _, local = make_core()
    said = list(core.reply(f"remember that my name is {key}"))
    assert said[0] == KEY_NOT_REMEMBERED and "looks like a key" in KEY_NOT_REMEMBERED
    assert not [f for f in core.memory.list_facts() if f.source == "user_said"]
    assert key not in "\n".join(core.memory.db.iterdump()) and "AIza" not in "\n".join(core.memory.db.iterdump())
    assert key not in repr(local.calls) and key not in repr(emitted)
    for text in ("sk-abcdefghijklmnop", "ab12cd34ef56gh78ij90kl12mn34", "x" * 41):
        assert keystore.key_shaped(text), text
    for text in ("Sree", "Mary Jane", "R2-D2", "a" * 40, "my name is Sreevarshan"):
        assert not keystore.key_shaped(text), text
    list(core.reply("remember that my name is Sree"))  # a real name still goes through
    assert [f.text for f in core.memory.list_facts() if f.source == "user_said"] == ["The user said: their name is Sree"]
