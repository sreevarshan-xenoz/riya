"""Riya's live voice, driven by the core (all speech on CPU, so the GPU only holds her small LLM).

  python -m riya.core                 # runs the core with voice; --no-voice for text only
  python -m riya.voice_candidates     # writes sample WAVs to pick her voice (settings.voice)

mic -> VAD (Silero) -> end of turn -> STT -> listen gate -> RiyaCore.reply() sentence by sentence -> TTS -> speaker
- Sentence streaming: sentence 1 is spoken while sentence 2 is still being generated (low time-to-first-audio).
- She listens after her name, the talk hotkey, typed text or her speaking first, until ~2 min pass without a turn.
- Barge-in: your voice stops her only when it is clearly louder than her own playback in the mic; so does the hotkey.
- Affect -> voice: arousal/energy set speaking rate; valence picks a warmer or softer voice blend.
- Mouth levels for the body's lip flap are reported at 30 Hz while she speaks.
Engines are small adapters so the winner of colab/bench_voice.py can be swapped in.
"""
import logging
import queue
import threading
import time
from collections.abc import Callable, Iterator

import numpy as np

from riya.core.listen import MOUTH_FPS, ListenGate, barge_in_ok, mouth_level, mouth_levels, rms
from riya.core.shaping import tts_text

SR = 16000
log = logging.getLogger("riya.voice")


def voice_params(state, energy):
    """Map Riya's feelings to speaking style. Returns (speed, voice_style)."""
    speed = 0.92 + 0.18 * state.arousal - 0.12 * max(0.0, 0.5 - energy)
    style = "bright" if state.valence > 0.3 else "soft" if state.valence < -0.3 else "warm"
    return float(np.clip(speed, 0.8, 1.2)), style


class PocketTTS:
    """Kyutai Pocket TTS (100M, CPU): streams audio chunks as they are generated. Winner of colab/bench_voice.py
    (318 ms to first audio, 0.89x real time on a 2-core Colab CPU; faster on the laptop's 8 cores).
    One identity voice; emotion comes from reference clips per style once Riya's own voice is designed (page 26)."""

    def __init__(self, voice="alba", style_clips=None):
        from pocket_tts import TTSModel

        self.model = TTSModel.load_model()
        self.sr = self.model.sample_rate
        self.states = {"warm": self.model.get_state_for_audio_prompt(voice)}
        for style, wav in (style_clips or {}).items():  # e.g. {"bright": "riya_bright.wav", "soft": "riya_soft.wav"}
            self.states[style] = self.model.get_state_for_audio_prompt(wav)

    def stream(self, text, speed, style):
        state = self.states.get(style, self.states["warm"])
        for chunk in self.model.generate_audio_stream(state, text):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1)


class KokoroTTS:
    VOICES = {"bright": "af_bella", "warm": "af_heart", "soft": "af_sky"}

    def __init__(self, model_dir):
        from kokoro_onnx import Kokoro

        self.k = Kokoro(f"{model_dir}/kokoro-v1.0.onnx", f"{model_dir}/voices-v1.0.bin")

    def speak(self, text, speed, style):
        audio, sr = self.k.create(text, voice=self.VOICES[style], speed=speed, lang="en-us")
        return np.asarray(audio, dtype=np.float32), sr


class WhisperSTT:
    def __init__(self, size="base.en", threads=4):
        from faster_whisper import WhisperModel

        self.m = WhisperModel(size, device="cpu", compute_type="int8", cpu_threads=threads)

    def transcribe(self, audio):
        return " ".join(s.text for s in self.m.transcribe(audio, beam_size=1, language="en")[0]).strip()


class SileroVAD:
    def __init__(self):
        import torch

        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
        self.torch = torch

    def prob(self, chunk512):
        return float(self.model(self.torch.from_numpy(chunk512), SR).item())


class MouthPacer:
    """Reports mouth levels for the audio being played, one per 1/fps, from its own thread so TTS never waits on it.
    Frames carry over chunk boundaries, and the timeline slips forward when playback starves."""

    def __init__(self, on_mouth: Callable[[float], None], sr: int, fps: int = MOUTH_FPS):
        self.on_mouth, self.sr, self.fps, self.win = on_mouth, sr, fps, sr // fps
        self.rest = np.zeros(0, np.float32)
        self.t0: float | None = None
        self.frames = 0
        self.q: queue.Queue = queue.Queue()
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._run, name="riya-mouth", daemon=True)
        self.thread.start()

    def push(self, chunk: np.ndarray) -> None:
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        pushed_until = self.t0 + (self.frames * self.win + len(self.rest)) / self.sr
        if now > pushed_until:  # playback ran dry: this chunk starts now
            self.t0 += now - pushed_until
        audio = np.concatenate([self.rest, chunk])
        full = len(audio) // self.win * self.win
        for level in mouth_levels(audio[:full], self.sr, self.fps):
            self._queue(level)
        self.rest = audio[full:]

    def _queue(self, level: float) -> None:
        self.q.put((self.t0 + self.frames / self.fps, level))
        self.frames += 1

    def finish(self) -> None:
        """Flush the last partial frame and wait until every level has been reported."""
        if len(self.rest) and self.t0 is not None:
            self._queue(mouth_level(self.rest))
        self.q.put(None)
        self.thread.join()

    def cancel(self) -> None:
        self.cancelled.set()
        self.q.put(None)
        self.thread.join()

    def _run(self) -> None:
        while (item := self.q.get()) is not None:
            at, level = item
            if self.cancelled.wait(max(0.0, at - time.monotonic())):
                return
            self.on_mouth(level)


class VoiceLoop:
    """Listens for the core and speaks for it. run() owns the mic; say() may be called from any thread and speaks one
    utterance at a time, so the mic keeps listening (for barge-in) while she talks."""

    def __init__(self, core, stt, tts, vad, gate: ListenGate, on_mouth: Callable[[float], None],
                 end_silence_ms: int = 350, open_output: Callable[[int], object] | None = None,
                 settings_every_s: float = 0.5):
        self.core, self.stt, self.tts, self.vad, self.gate, self.on_mouth = core, stt, tts, vad, gate, on_mouth
        self.end_silence = end_silence_ms / 1000
        self.open_output = open_output or _sounddevice_output
        self.settings_every_s = settings_every_s
        self.speaking, self.interrupt, self.stopped = threading.Event(), threading.Event(), threading.Event()
        self.mic: queue.Queue = queue.Queue()
        self.mic_stream = None
        self.playback_rms = 0.0
        self._say_lock = threading.Lock()
        self._mic_muted = False
        self._next_settings_check = 0.0

    # ---- listening ----------------------------------------------------------------------------------
    def _on_mic(self, indata, frames, t, status):
        self.mic.put(indata[:, 0].copy())

    def classify(self, chunk: np.ndarray) -> str:
        """"user", "echo" or "silence". While she talks, speech counts as the user's only when it is clearly louder
        than her own voice coming back, even in the middle of an utterance."""
        if self.vad.prob(chunk) <= 0.5:
            return "silence"
        if self.speaking.is_set() and not barge_in_ok(rms(chunk), self.playback_rms):
            return "echo"
        return "user"

    def poll(self) -> None:
        """The talk hotkey opens the gate and cuts her off; mute_mic pauses the mic stream."""
        if self.core.talk_requested:
            self.core.talk_requested = False
            self.gate.open("hotkey")
            if self.speaking.is_set():
                self.interrupt.set()
        now = time.monotonic()
        if now >= self._next_settings_check:
            self._next_settings_check = now + self.settings_every_s
            self._set_mic_muted(self.core.settings.mute_mic)

    def _set_mic_muted(self, muted: bool) -> None:
        if muted == self._mic_muted:
            return
        self._mic_muted = muted
        if self.mic_stream is not None:
            (self.mic_stream.stop if muted else self.mic_stream.start)()
        while muted and not self.mic.empty():
            self.mic.get_nowait()

    def listen(self) -> tuple[np.ndarray, str | None] | None:
        """Collect one user utterance: starts on speech, ends after end_silence of non-speech. With a streaming STT
        (feed/finish) it's transcribed while you talk. Returns (audio, text_or_None), or None when stopped or muted."""
        buf, speech, silent_since = [], False, None
        streaming = hasattr(self.stt, "feed")
        while not self.stopped.is_set():
            self.poll()
            if self._mic_muted and speech:
                if streaming and hasattr(self.stt, "reset"):
                    self.stt.reset()
                return None
            try:
                chunk = self.mic.get(timeout=0.1)
            except queue.Empty:
                continue
            kind = self.classify(chunk)
            if kind == "user":
                if self.speaking.is_set():
                    self.interrupt.set()  # you started talking: she stops
                speech, silent_since = True, None
            elif speech and silent_since is None:
                silent_since = time.monotonic()
            if speech:
                if kind != "echo":  # her echo is neither transcribed nor keeps your utterance open
                    buf.append(chunk)
                    if streaming:
                        self.stt.feed(chunk)
                if silent_since is not None and time.monotonic() - silent_since >= self.end_silence:
                    return np.concatenate(buf), (self.stt.finish() if streaming else None)
        return None

    def hear(self, text: str) -> threading.Thread | None:
        """A finished transcript goes through the gate; her reply is spoken in the background so the mic stays live."""
        text = text.strip()
        accepted = self.gate.accept(text, gaming=self.core.presence.state == "gaming") if text else None
        if accepted is None:
            return None
        t = threading.Thread(target=self._turn, args=(self.core.reply(accepted, source="voice"),), name="riya-say",
                             daemon=True)
        t.start()
        return t

    def _turn(self, sentences: Iterator[str]) -> None:
        self.say(sentences)
        self.gate.note_turn()

    def run(self) -> None:
        import sounddevice as sd

        with sd.InputStream(samplerate=SR, channels=1, blocksize=512, dtype="float32", callback=self._on_mic) as mic:
            self.mic_stream = mic
            log.info("Riya is listening for her name or the talk hotkey")
            while not self.stopped.is_set():
                try:
                    if (heard := self.listen()) is not None:
                        audio, text = heard
                        self.hear(text if text is not None else self.stt.transcribe(audio))
                except Exception:
                    log.exception("couldn't handle what the mic heard")

    def stop(self) -> None:
        self.stopped.set()

    # ---- speaking -----------------------------------------------------------------------------------
    def answer(self, sentences: Iterator[str], reason: str = "typed") -> None:
        """Speak a reply to typed text, or her speaking first; the mic then stays open for a follow-up without her
        name. While gaming only the hotkey opens it."""
        listen = self.core.presence.state != "gaming"
        if listen:
            self.gate.open(reason)
        self.say(sentences)
        if listen:
            self.gate.note_turn()

    def say(self, sentences: Iterator[str]) -> None:
        """Speak sentences as they arrive; playback starts at the first audio chunk. A barge-in stops playback and
        closes the reply, so the core records only what was said. With mute_voice the sentences are only drained.
        The voice speaks tts_text(sentence): no emoji, no *stage directions*; a sentence with nothing left is skipped.
        Never raises: an audio failure is logged, the rest of the reply is drained, and her state is reset."""
        with self._say_lock:
            settings = self.core.settings
            out = pacer = None
            self.interrupt.clear()
            try:
                for sentence in sentences:
                    if settings.mute_voice or not (words := tts_text(sentence)):  # nothing left to say aloud
                        continue
                    speed, style = voice_params(*self.core.voice_state())
                    for chunk in self._synth(words, speed, style):
                        if self.interrupt.is_set():
                            break
                        if out is None:
                            sr = getattr(self.tts, "sr", 24000)
                            out, pacer = self.open_output(sr), MouthPacer(self.on_mouth, sr)
                            out.start()
                            self.speaking.set()  # from her first sound on, barge-in needs the echo margin
                        chunk = np.asarray(chunk, dtype=np.float32).reshape(-1) * np.float32(settings.volume)
                        self.playback_rms = rms(chunk)
                        out.write(chunk.reshape(-1, 1))
                        pacer.push(chunk)
                    if self.interrupt.is_set():
                        break
            except Exception:
                log.exception("voice playback failed; the rest of the reply is shown, not spoken")
                _quietly(lambda: _drain(sentences), "draining the reply")
            finally:
                cut = self.interrupt.is_set()
                try:
                    if hasattr(sentences, "close"):
                        _quietly(sentences.close, "closing the reply")
                    if out is not None:
                        _quietly(out.abort if cut else out.stop, "stopping playback")
                        _quietly(out.close, "closing the audio output")
                    if pacer is not None:
                        _quietly(pacer.cancel if cut else pacer.finish, "stopping the mouth")
                finally:
                    self.playback_rms = 0.0
                    self.speaking.clear()
                    _quietly(lambda: self.on_mouth(0.0), "closing the mouth")

    def _synth(self, sentence: str, speed: float, style: str):
        if hasattr(self.tts, "stream"):
            return self.tts.stream(sentence, speed, style)
        return [self.tts.speak(sentence, speed, style)[0]]


def _drain(sentences: Iterator[str]) -> None:
    for _ in sentences:
        pass


def _quietly(step: Callable[[], object], what: str) -> None:
    try:
        step()
    except Exception:
        log.exception("voice: %s failed", what)


def _sounddevice_output(sr: int):
    import sounddevice as sd

    return sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
