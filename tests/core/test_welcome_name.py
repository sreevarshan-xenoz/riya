"""The welcome's name step goes through the existing remember path: she acks aloud and the fact is `user_said`."""
from tests.fakes import make_core


def test_the_welcome_name_is_saved_through_the_remember_path():
    core, *_ = make_core()
    list(core.reply("remember that my name is Sree"))
    facts = [f for f in core.memory.list_facts() if f.text == "The user said: their name is Sree"]
    assert len(facts) == 1 and facts[0].source == "user_said"
