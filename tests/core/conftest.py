import pytest


@pytest.fixture(autouse=True)
def no_real_hands(monkeypatch):
    """The runner's build_core gives the core WindowsEffects, UrlHttp and the Windows PeerOS: every core test gets the
    fakes instead."""
    from riya.core import __main__ as runner
    from riya.core.hands.effects import FakeEffects
    from riya.core.hands.net import FakeHttp
    from riya.agents.peer import FakePeerOS
    monkeypatch.setattr(runner, "WindowsEffects", FakeEffects)
    monkeypatch.setattr(runner, "UrlHttp", FakeHttp)
    monkeypatch.setattr(runner, "default_peer_os", lambda: FakePeerOS())
