import subprocess
import sys
import time

import numpy as np

from riya.core import __main__ as runner
from riya.core.clock import FakeClock
from riya.core.listen import ListenGate, rms
from riya.core.protocol import Say, SetSettings
from riya.voice import MouthPacer, VoiceLoop, voice_params
from tests.fakes import make_core

SR_OUT = 24000


def tone(amplitude, n=512, sr=16000):
    return (amplitude * np.sin(2 * np.pi * 220 * np.arange(n) / sr)).astype(np.float32)


class FakeVAD:
    def __init__(self, on_call=None):
        self.calls, self.on_call = 0, on_call

    def prob(self, chunk):
        if self.on_call:
            self.on_call(self.calls)
        self.calls += 1
        return 0.9 if rms(chunk) > 0.01 else 0.0


class FakeSTT:
    def transcribe(self, audio):
        return "unused"


class FakeTTS:
    sr = SR_OUT

    def __init__(self, chunks_per_sentence=2, on_chunk=None):
        self.calls, self.n, self.on_chunk = [], chunks_per_sentence, on_chunk

    def stream(self, text, speed, style):
        self.calls.append((text, speed, style))
        for i in range(self.n):
            yield tone(0.3, n=1600, sr=SR_OUT)
            if self.on_chunk:
                self.on_chunk(i)


class FakeOutput:
    def __init__(self):
        self.writes, self.events = [], []

    def start(self):
        self.events.append("start")

    def write(self, data):
        self.writes.append(data.copy())

    def stop(self):
        self.events.append("stop")

    def abort(self):
        self.events.append("abort")

    def close(self):
        self.events.append("close")


class BrokenOutput(FakeOutput):
    """A headset unplugged mid-reply: writing and stopping both fail."""

    def write(self, data):
        raise OSError("device unavailable")

    def stop(self):
        raise OSError("device unavailable")


class FakeMicStream:
    def __init__(self):
        self.active = True

    def stop(self):
        self.active = False

    def start(self):
        self.active = True


def make_loop(replies=("First one. Second one.",), tts=None, vad=None, output=FakeOutput, core_clock=None, **kw):
    core, emitted, src, local = make_core(clock=core_clock, local_replies=list(replies))
    clock = core_clock or FakeClock()
    outputs, mouth = [], []

    def open_output(sr):
        outputs.append(output())
        return outputs[-1]

    loop = VoiceLoop(core, FakeSTT(), tts or FakeTTS(), vad or FakeVAD(), ListenGate(clock), on_mouth=mouth.append,
                     open_output=open_output, end_silence_ms=0, settings_every_s=0, **kw)
    loop.src, loop.emitted = src, emitted
    return loop, core, clock, outputs, mouth, local


def test_say_streams_volume_scaled_audio_with_feelings_and_30hz_mouth():
    loop, core, _, outputs, mouth, _ = make_loop()
    t0 = time.monotonic()
    loop.say(core.reply("hi", "typed"))
    elapsed = time.monotonic() - t0
    out = outputs[0]
    assert len(out.writes) == 4 and out.events == ["start", "stop", "close"]
    assert np.isclose(np.abs(np.concatenate(out.writes)).max(), 0.3 * 0.8, atol=1e-3)  # settings.volume = 0.8
    speed, style = voice_params(*core.voice_state())
    assert [c[1:] for c in loop.tts.calls] == [(speed, style), (speed, style)]
    assert len(mouth) == 4 * 2 + 1 and min(mouth[:-1]) > 0.5 and mouth[-1] == 0.0
    assert elapsed >= 7 / 30  # 8 frames spaced at 30 Hz
    assert not loop.speaking.is_set() and loop.playback_rms == 0.0


def test_muted_voice_drains_without_playing():
    loop, core, _, outputs, mouth, _ = make_loop()
    core.handle(SetSettings(patch={"mute_voice": True}))
    loop.say(core.reply("hi", "typed"))
    assert outputs == [] and loop.tts.calls == []
    assert core.turns[-1] == {"role": "assistant", "content": "First one. Second one."}


def test_barge_in_stops_playback_and_records_the_partial_turn():
    holder = {}
    tts = FakeTTS(on_chunk=lambda i: holder["loop"].interrupt.set())
    loop, core, _, outputs, mouth, _ = make_loop(tts=tts)
    holder["loop"] = loop
    loop.say(core.reply("hi", "voice"))
    assert len(outputs[0].writes) == 1 and "abort" in outputs[0].events and "stop" not in outputs[0].events
    assert core.turns[-1] == {"role": "assistant", "content": "First one."}
    assert mouth[-1] == 0.0


def test_her_own_echo_does_not_barge_in_but_clear_user_speech_does():
    loop, *_ = make_loop()
    loop.speaking.set()
    loop.playback_rms = 0.04
    assert loop.classify(tone(0.07)) == "echo"  # rms 0.05: her voice through the speakers
    for c in [tone(0.07)] * 3 + [tone(0.4)] * 2 + [np.zeros(512, np.float32)]:
        loop.mic.put(c)
    audio, text = loop.listen()
    assert len(audio) == 3 * 512 and text is None and loop.interrupt.is_set()


def test_quiet_speech_counts_when_she_is_silent():
    loop, *_ = make_loop()
    for c in [tone(0.07)] * 2 + [np.zeros(512, np.float32)]:
        loop.mic.put(c)
    audio, _ = loop.listen()
    assert len(audio) == 3 * 512 and not loop.interrupt.is_set()


def test_heard_text_goes_through_the_gate():
    loop, core, clock, outputs, _, local = make_loop(replies=("Hi!",))
    assert loop.hear("what's the weather") is None and local.calls == []
    loop.hear("Riya, hi").join(5)
    assert core.turns[-2:] == [{"role": "user", "content": "Riya, hi"}, {"role": "assistant", "content": "Hi!"}]
    assert outputs and outputs[0].writes
    clock.advance(119)
    assert loop.gate.is_open()


def test_gaming_ignores_her_name_but_the_hotkey_opens_the_gate():
    loop, core, *_ = make_loop()
    core.presence.state = "gaming"
    assert loop.hear("Riya, hi") is None
    core.talk_requested = True
    loop.poll()
    assert core.talk_requested is False and loop.gate.is_open()


def test_hotkey_while_she_speaks_barges_in():
    loop, core, *_ = make_loop()
    loop.speaking.set()
    core.talk_requested = True
    loop.poll()
    assert loop.interrupt.is_set()


def test_mute_mic_pauses_the_mic_stream():
    loop, core, *_ = make_loop()
    loop.mic_stream = FakeMicStream()
    core.handle(SetSettings(patch={"mute_mic": True}))
    loop.poll()
    assert not loop.mic_stream.active
    core.handle(SetSettings(patch={"mute_mic": False}))
    loop.poll()
    assert loop.mic_stream.active


def test_typed_answer_is_spoken_and_opens_the_mic():
    loop, core, _, outputs, _, _ = make_loop(replies=("Hello!",))
    loop.answer(core.reply("hello", "typed"))
    assert outputs[0].writes and loop.gate.is_open()


def test_mouth_pacer_carries_partial_frames_across_chunks():
    got = []
    pacer = MouthPacer(got.append, SR_OUT)
    for _ in range(3):
        pacer.push(tone(0.3, n=1200, sr=SR_OUT))  # 1.5 frames each
    pacer.finish()
    assert len(got) == 5 and min(got) > 0.5  # 3600 samples = 4.5 frames: the half frame is flushed at the end


def test_mouth_pacer_cancel_drops_pending_frames():
    got = []
    pacer = MouthPacer(got.append, SR_OUT)
    pacer.push(tone(0.3, n=SR_OUT, sr=SR_OUT))  # 30 frames, a full second
    pacer.cancel()
    assert len(got) < 30


def test_voice_module_imports_no_audio_stack():
    code = ("import sys, riya.voice; "
            "print([m for m in ('torch', 'sounddevice', 'pocket_tts', 'moonshine_voice') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"



def test_her_echo_mid_utterance_neither_interrupts_nor_is_transcribed():
    holder = {}

    def she_starts_talking(call):
        if call == 2:  # the user has said two chunks; now her reply starts playing
            holder["loop"].speaking.set()
            holder["loop"].playback_rms = 0.04

    loop, *_ = make_loop(vad=FakeVAD(on_call=she_starts_talking))
    holder["loop"] = loop
    user, echo = tone(0.4), tone(0.07)
    for c in [user, user, echo, echo, echo, user, np.zeros(512, np.float32)]:
        loop.mic.put(c)
    audio, _ = loop.listen()
    assert len(audio) == 2 * 512 and np.array_equal(audio, np.concatenate([user, user]))
    assert not loop.interrupt.is_set()  # her own echo never cut her off
    audio, _ = loop.listen()  # the remaining echo is ignored; clearly louder speech still barges in
    assert loop.interrupt.is_set() and len(audio) == 2 * 512 and np.array_equal(audio[:512], user)


def test_device_failure_mid_reply_leaves_the_loop_usable():
    loop, core, _, outputs, mouth, _ = make_loop(output=BrokenOutput)
    loop.answer(core.reply("hi", "typed"))  # must not raise
    assert not loop.speaking.is_set() and loop.playback_rms == 0.0 and mouth[-1] == 0.0
    assert core.turns[-1] == {"role": "assistant", "content": "First one. Second one."}  # the bubble still completes
    assert loop.gate.is_open()  # note_turn still ran


def test_a_silent_bid_opens_the_mic_for_a_nameless_answer():
    clock = FakeClock()
    loop, core, *_ = make_loop(replies=("Hey... water break?",), core_clock=clock)
    runner.connect_voice(core, loop)
    core.handle(SetSettings(patch={"quiet_hours": None}))
    for _ in range(int((2 * 3600 + 600) / 300)):  # a long focused stretch, then a break
        clock.advance(300)
        assert core.tick() is None
    loop.src.idle = 150
    clock.advance(150)
    assert core.tick() is None
    assert any(isinstance(m, Say) and m.kind == "bid" and not m.spoken for m in loop.emitted)
    clock.advance(100)
    assert loop.gate.accept("yeah okay, five minutes", gaming=False) == "yeah okay, five minutes"
    clock.advance(121)
    assert loop.gate.accept("random tv noise", gaming=False) is None


def test_say_speaks_tts_text_and_skips_sentences_that_become_empty():
    loop, *_ = make_loop()
    loop.say(iter(["Hmph \U0001F624 *giggles* you're late!", "*sighs*", "\U0001F600", "(laughs) Fine."]))
    assert [c[0] for c in loop.tts.calls] == ["Hmph you're late!", "Fine."]
