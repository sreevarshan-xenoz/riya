"""Riya's live voice loop (all speech on CPU, so the GPU only holds her small LLM).

  python -m riya.voice --url http://127.0.0.1:8090

mic -> VAD (Silero) -> end of turn -> STT -> Riya.reply() streamed sentence by sentence -> TTS -> speaker
- Sentence streaming: sentence 1 is spoken while sentence 2 is still being generated (low time-to-first-audio).
- Barge-in: if you start speaking while Riya talks, playback stops and she listens.
- Affect -> voice: arousal/energy set speaking rate; valence picks a warmer or softer voice blend.
Engines are small adapters so the winner of colab/bench_voice.py can be swapped in.
"""
import argparse
import queue
import threading
import time

import numpy as np

from riya.chat import Riya

SR = 16000


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


class VoiceLoop:
    def __init__(self, riya, stt, tts, vad, end_silence_ms=350, barge_in=True):
        import sounddevice as sd

        self.sd, self.riya, self.stt, self.tts, self.vad = sd, riya, stt, tts, vad
        self.end_silence = end_silence_ms / 1000
        self.barge_in, self.speaking = barge_in, threading.Event()
        self.interrupt = threading.Event()
        self.mic = queue.Queue()

    def _on_mic(self, indata, frames, t, status):
        self.mic.put(indata[:, 0].copy())

    def listen(self):
        """Collect one user utterance: starts on speech, ends after end_silence of non-speech.
        With a streaming STT (feed/finish), audio is transcribed while you talk; returns (audio, text_or_None)."""
        buf, speech, silent_since = [], False, None
        streaming = hasattr(self.stt, "feed")
        while True:
            chunk = self.mic.get()
            p = self.vad.prob(chunk)
            if p > 0.5:
                if self.speaking.is_set() and self.barge_in:
                    self.interrupt.set()  # you started talking: Riya stops
                speech, silent_since = True, None
            elif speech:
                silent_since = silent_since or time.time()
            if speech:
                buf.append(chunk)
                if streaming:
                    self.stt.feed(chunk)
                if silent_since and time.time() - silent_since >= self.end_silence:
                    return np.concatenate(buf), (self.stt.finish() if streaming else None)

    def say(self, sentences):
        """Speak sentences as they arrive; with a streaming TTS, playback starts at the first audio chunk."""
        self.speaking.set()
        self.interrupt.clear()
        try:
            sr = getattr(self.tts, "sr", 24000)
            with self.sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as out:
                for sentence in sentences:
                    if self.interrupt.is_set():
                        break
                    speed, style = voice_params(self.riya.affect.state, float(self.riya.body[0]))
                    chunks = (self.tts.stream(sentence, speed, style) if hasattr(self.tts, "stream")
                              else [self.tts.speak(sentence, speed, style)[0]])
                    for chunk in chunks:
                        if self.interrupt.is_set():
                            break
                        out.write(chunk.reshape(-1, 1))
        finally:
            self.speaking.clear()

    def run(self):
        with self.sd.InputStream(samplerate=SR, channels=1, blocksize=512, dtype="float32", callback=self._on_mic):
            print("Riya is listening... (Ctrl+C to quit)")
            while True:
                audio, text = self.listen()
                t0 = time.time()
                text = text if text is not None else self.stt.transcribe(audio)
                if not text:
                    continue
                print(f"you> {text}   (STT {1000 * (time.time() - t0):.0f} ms)")
                threading.Thread(target=self.say, args=(self.riya.reply(text),), daemon=True).start()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8090")
    p.add_argument("--tts", choices=["pocket", "kokoro"], default="pocket")
    p.add_argument("--voice", default="alba", help="Pocket TTS voice name or a reference WAV of Riya's voice")
    p.add_argument("--kokoro_dir", default="voice_models")
    p.add_argument("--stt", default="moonshine-small",
                   help="moonshine-tiny|moonshine-small|moonshine-medium (streaming) or a faster-whisper size")
    a = p.parse_args()
    tts = PocketTTS(a.voice) if a.tts == "pocket" else KokoroTTS(a.kokoro_dir)
    if a.stt.startswith("moonshine-"):
        from riya.stt_moonshine import MoonshineStreamSTT

        stt = MoonshineStreamSTT(a.stt.split("-", 1)[1])
    else:
        stt = WhisperSTT(a.stt)
    VoiceLoop(Riya(a.url), stt, tts, SileroVAD()).run()


if __name__ == "__main__":
    main()
