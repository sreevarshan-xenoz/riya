"""Streaming Moonshine v2 STT adapter for riya/voice.py.

    stt = MoonshineStreamSTT("small")       # tiny | small | medium (streaming), or tiny-batch / base-batch
    for chunk in mic_chunks_while_user_speaks:  # float32 mono 16 kHz, any length (e.g. 512 samples)
        stt.feed(chunk)                      # transcribes incrementally in the background of the utterance
    text = stt.finish()                      # flush + final transcript (fast: most work was done while speaking)
    stt.reset()                              # optional; finish() already resets for the next utterance

Also exposes transcribe(audio) so it is a drop-in for WhisperSTT in VoiceLoop (non-streaming path).
Needs `pip install moonshine-voice` (English models are MIT; downloaded/cached on first use).
"""
import numpy as np

SR = 16000


class MoonshineStreamSTT:
    ARCHS = {
        "tiny": "TINY_STREAMING",
        "small": "SMALL_STREAMING",
        "medium": "MEDIUM_STREAMING",
        "tiny-batch": "TINY",
        "base-batch": "BASE",
    }

    def __init__(self, size="small", language="en", update_interval=0.5, options=None):
        import moonshine_voice as mv

        arch = getattr(mv.ModelArch, self.ARCHS.get(size, size))
        path, arch = mv.get_model_for_language(language, arch)
        self.tr = mv.Transcriber(path, arch, update_interval=update_interval, options=options)
        self.update_interval = update_interval
        self.stream = None
        self.reset()

    def reset(self):
        """Drop any partial utterance and open a fresh stream."""
        if self.stream is not None:
            try:
                self.stream.stop()
            except Exception:
                pass
            if hasattr(self.stream, "close"):
                self.stream.close()
        self.stream = self.tr.create_stream(update_interval=self.update_interval)
        self.stream.start()
        self.partial = ""

    def feed(self, chunk):
        """Add float32 mono 16 kHz samples. May run a transcription pass (every update_interval s of audio)."""
        self.stream.add_audio(np.asarray(chunk, dtype=np.float32).ravel().tolist(), SR)

    def finish(self):
        """End of speech: flush remaining audio, return the final transcript, and reset for the next turn."""
        t = self.stream.stop()
        text = " ".join(l.text.strip() for l in (t.lines if t else []) if l.text.strip())
        if hasattr(self.stream, "close"):
            self.stream.close()
        self.stream = None
        self.reset()
        return text

    def transcribe(self, audio):
        """Non-streaming fallback with the same signature as voice.WhisperSTT.transcribe."""
        t = self.tr.transcribe_without_streaming(np.asarray(audio, dtype=np.float32).ravel().tolist(), SR)
        return " ".join(l.text.strip() for l in t.lines if l.text.strip())

    def close(self):
        if self.stream is not None and hasattr(self.stream, "close"):
            self.stream.close()
        self.tr.close()
