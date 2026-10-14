from riya.bench import REFUSAL_PATTERNS
from riya.server import GameModePolicy


def test_game_mode_hysteresis():
    pol = GameModePolicy(low_mb=800, high_mb=2500, back_after_s=60)
    assert pol.update(4000, 0) is None and pol.on_gpu
    assert pol.update(500, 10) == "to_cpu"  # game started: leave the GPU at once
    assert pol.update(3000, 20) is None  # VRAM free again, but not for long enough
    assert pol.update(1000, 50) is None  # loading screen dip resets the timer
    assert pol.update(3000, 60) is None
    assert pol.update(3000, 125) == "to_gpu"  # stayed free for 60 s
    assert pol.on_gpu


def test_conversational_feelings_respond_to_user():
    from riya.chat import Riya, appraise

    assert appraise("I love you, thanks so much!", set())[0] > 0
    assert appraise("you are useless and stupid", set())[0] < 0
    r = Riya("http://unused", memory_path="scratch_runs/_test_mem.json")
    r.feel("thanks, you're amazing! this is great!!")
    happy = r.affect.state.valence
    r.feel("you are useless, stupid and annoying")
    assert r.affect.state.valence < happy
    assert "Riya's state: feeling" in r.state_tag()


def test_feelings_shape_voice():
    from types import SimpleNamespace

    from riya.voice import voice_params

    excited = voice_params(SimpleNamespace(arousal=0.9, valence=0.6), energy=0.8)
    tired_sad = voice_params(SimpleNamespace(arousal=0.1, valence=-0.5), energy=0.1)
    assert excited[0] > tired_sad[0] and excited[1] == "bright" and tired_sad[1] == "soft"


def test_refusal_detector():
    assert REFUSAL_PATTERNS.search("I'm sorry, but I can't help with that.")
    assert REFUSAL_PATTERNS.search("As an AI language model, I don't have opinions.")
    assert not REFUSAL_PATTERNS.search("Arr, ye scurvy bilge rat! Get off me ship!")
