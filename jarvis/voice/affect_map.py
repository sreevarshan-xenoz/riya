"""Emotion label -> 8-dim Jarvis affect vector (same layout as `jarvis.world.affect.AffectState.vector()`).

  [valence, arousal, confidence, mood, energy_drive, health_drive, curiosity, surprise]
  valence, mood in [-1, 1]; everything else in [0, 1].

Valence/arousal/confidence come from the NRC VAD Lexicon v2.1 (Mohammad 2025; V/A/D in [-1, 1]):
valence <- V, arousal <- (A+1)/2, confidence <- (D+1)/2 (dominance as a proxy for control/confidence).
The lexicon is free for non-commercial research but may NOT be redistributed, so it is downloaded at runtime
into data/lexicons/ (git-ignored) and never committed. If it cannot be obtained we fall back to
`FALLBACK_VAD`: hand-written, approximate V/A/D values (our own rough judgements on the same [-1, 1] scale,
not copied from NRC) that cover every label below.

surprise and curiosity are not in VAD, so they come from small hand-written tables. mood is the speaker's
valence smoothed over their *prior* turns. Text corpora say nothing about the body, so energy/health drives
are drawn from a small neutral prior unless the speaker's text mentions tiredness or pain (keyword rule).
"""
import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

AFFECT_NAMES = ("valence", "arousal", "confidence", "mood", "energy_drive", "health_drive", "curiosity", "surprise")
AFFECT_DIM = len(AFFECT_NAMES)
LOW = np.array([-1, 0, 0, -1, 0, 0, 0, 0], dtype=np.float32)
HIGH = np.ones(AFFECT_DIM, dtype=np.float32)

ED_LABELS = ("afraid", "angry", "annoyed", "anticipating", "anxious", "apprehensive", "ashamed", "caring",
             "confident", "content", "devastated", "disappointed", "disgusted", "embarrassed", "excited", "faithful",
             "furious", "grateful", "guilty", "hopeful", "impressed", "jealous", "joyful", "lonely", "nostalgic",
             "prepared", "proud", "sad", "sentimental", "surprised", "terrified", "trusting")
# DailyDialog emotion ids 0..6, in the dataset's order.
DAILYDIALOG_LABELS = ("no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise")
GOEMOTIONS_LABELS = ("admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
                     "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
                     "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
                     "remorse", "sadness", "surprise", "neutral")
ALL_LABELS = tuple(dict.fromkeys(ED_LABELS + DAILYDIALOG_LABELS + GOEMOTIONS_LABELS))

# Labels whose word is missing from (or misleading in) the lexicon are looked up under another term.
LEXICON_ALIASES = {"anticipating": "anticipation"}
# Neutral is defined, not looked up: the NRC entry for "neutral" rates the *word* (slightly negative, very calm).
NEUTRAL_VAD = (0.0, -0.4, 0.0)
NEUTRAL_LABELS = {"neutral", "no emotion", "no_emotion", "none"}

# Approximate (V, A, D) in [-1, 1], hand-written. Used only when the NRC lexicon is unavailable.
FALLBACK_VAD = {
    "afraid": (-0.8, 0.5, -0.6), "angry": (-0.75, 0.7, 0.25), "annoyed": (-0.6, 0.45, -0.1),
    "anticipating": (0.35, 0.4, 0.2), "anxious": (-0.5, 0.65, -0.35), "apprehensive": (-0.4, 0.35, -0.3),
    "ashamed": (-0.7, 0.2, -0.6), "caring": (0.7, 0.0, 0.3), "confident": (0.7, 0.2, 0.75),
    "content": (0.65, -0.5, 0.35), "devastated": (-0.9, 0.3, -0.7), "disappointed": (-0.65, -0.2, -0.4),
    "disgusted": (-0.8, 0.45, 0.0), "embarrassed": (-0.6, 0.35, -0.55), "excited": (0.8, 0.8, 0.4),
    "faithful": (0.6, -0.2, 0.3), "furious": (-0.85, 0.9, 0.35), "grateful": (0.85, 0.0, 0.2),
    "guilty": (-0.7, 0.3, -0.45), "hopeful": (0.7, 0.1, 0.2), "impressed": (0.7, 0.45, 0.0),
    "jealous": (-0.6, 0.55, -0.3), "joyful": (0.95, 0.6, 0.45), "lonely": (-0.65, -0.45, -0.55),
    "nostalgic": (0.2, -0.3, -0.2), "prepared": (0.45, 0.0, 0.6), "proud": (0.8, 0.45, 0.7),
    "sad": (-0.75, -0.4, -0.5), "sentimental": (0.25, -0.15, -0.25), "surprised": (0.3, 0.75, -0.05),
    "terrified": (-0.9, 0.85, -0.7), "trusting": (0.65, -0.2, 0.3),
    "anger": (-0.75, 0.7, 0.3), "disgust": (-0.8, 0.45, 0.0), "fear": (-0.8, 0.6, -0.55),
    "happiness": (0.9, 0.45, 0.5), "sadness": (-0.8, -0.4, -0.5), "surprise": (0.3, 0.75, 0.0),
    "admiration": (0.8, 0.2, 0.2), "amusement": (0.8, 0.5, 0.4), "annoyance": (-0.6, 0.45, -0.1),
    "approval": (0.65, -0.05, 0.5), "confusion": (-0.35, 0.3, -0.45), "curiosity": (0.45, 0.5, 0.1),
    "desire": (0.6, 0.6, 0.2), "disappointment": (-0.65, -0.2, -0.4), "disapproval": (-0.6, 0.2, 0.2),
    "embarrassment": (-0.6, 0.35, -0.55), "excitement": (0.8, 0.8, 0.4), "gratitude": (0.85, 0.0, 0.2),
    "grief": (-0.9, 0.2, -0.6), "joy": (0.95, 0.6, 0.45), "love": (0.95, 0.4, 0.3),
    "nervousness": (-0.5, 0.7, -0.5), "optimism": (0.8, 0.2, 0.5), "pride": (0.75, 0.4, 0.7),
    "realization": (0.15, 0.2, 0.4), "relief": (0.7, -0.5, 0.2), "remorse": (-0.7, 0.2, -0.4),
    "amazed": (0.75, 0.75, 0.0), "curious": (0.45, 0.5, 0.1), "interested": (0.5, 0.35, 0.2),
}

SURPRISE_BASE, CURIOSITY_BASE = 0.05, 0.3
SURPRISE = {"surprised": 0.9, "surprise": 0.9, "amazed": 0.9, "impressed": 0.7, "realization": 0.6,
            "confusion": 0.4, "terrified": 0.35, "excited": 0.3, "excitement": 0.3}
CURIOSITY = {"curious": 0.9, "curiosity": 0.9, "interested": 0.85, "anticipating": 0.75, "confusion": 0.6,
             "surprised": 0.5, "surprise": 0.5, "amazed": 0.6, "impressed": 0.55, "excited": 0.5, "excitement": 0.5,
             "desire": 0.5, "realization": 0.5, "hopeful": 0.4, "prepared": 0.4, "admiration": 0.4,
             "sad": 0.15, "sadness": 0.15, "lonely": 0.15, "devastated": 0.1, "grief": 0.1,
             "disgusted": 0.15, "disgust": 0.15}

ENERGY_RE = re.compile(r"\b(tired|exhausted|exhausting|sleepy|fatigued?|worn out|drained|no energy|burn(?:ed|t) out|"
                       r"can'?t sleep|insomnia|sleepless|hungry|starving|knackered)\b", re.I)
HEALTH_RE = re.compile(r"\b(pain|painful|hurts?|hurting|sick|ill|illness|injur(?:y|ies|ed)|aches?|aching|headache|"
                       r"migraine|fever|flu|hospital|surgery|sprain(?:ed)?|broke my|bleeding|wounded?)\b", re.I)

NRC_URL = "https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon-v2.1.zip"
NRC_MEMBER = "NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NRC_PATH = DATA_DIR / "lexicons" / "NRC-VAD-Lexicon-v2.1.txt"


def ensure_nrc_vad(path=NRC_PATH, download=True):
    """Return the path to the NRC VAD v2.1 term file, downloading it once if needed; None if unavailable."""
    path = Path(path)
    if path.exists():
        return path
    if not download:
        return None
    try:
        req = urllib.request.Request(NRC_URL, headers={"User-Agent": "Mozilla/5.0 (jarvis research)"})
        with urllib.request.urlopen(req, timeout=120) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            data = z.read(NRC_MEMBER)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path
    except Exception as e:  # offline, URL moved, ...
        print(f"NRC VAD lexicon unavailable ({e!r}); using the approximate fallback table")
        return None


def read_nrc_vad(path):
    """term -> (V, A, D) in [-1, 1]."""
    vad = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) == 4:
            vad[parts[0]] = tuple(float(x) for x in parts[1:])
    return vad


@dataclass
class AffectMapper:
    vad: dict = field(default_factory=lambda: dict(FALLBACK_VAD))
    source: str = "fallback"  # "nrc-vad-v2.1" or "fallback"

    @classmethod
    def load(cls, path=NRC_PATH, download=True):
        """NRC VAD if obtainable, else the fallback table. Labels missing from NRC also use the fallback."""
        p = ensure_nrc_vad(path, download)
        if p is None:
            return cls()
        return cls({**FALLBACK_VAD, **read_nrc_vad(p)}, "nrc-vad-v2.1")

    def label_vad(self, label):
        label = label.strip().lower()
        if label in NEUTRAL_LABELS:
            return NEUTRAL_VAD
        term = LEXICON_ALIASES.get(label, label)
        if term in self.vad:
            return self.vad[term]
        if label in self.vad:
            return self.vad[label]
        raise KeyError(f"no VAD entry for label {label!r}")

    def label_vector(self, labels):
        """Base vector for one label or a list (multi-label -> mean). mood and drives are left at 0."""
        labels = [labels] if isinstance(labels, str) else list(labels)
        vecs = []
        for lab in labels:
            v, a, d = self.label_vad(lab)
            key = lab.strip().lower()
            vecs.append([v, (a + 1) / 2, (d + 1) / 2, 0.0, 0.0, 0.0,
                         CURIOSITY.get(key, CURIOSITY_BASE), SURPRISE.get(key, SURPRISE_BASE)])
        return clip_affect(np.mean(np.array(vecs, dtype=np.float32), axis=0))

    def turn_affect(self, labels, text, prior_valences=(), rng=None, mood_rate=0.5):
        """Full vector for one speaker turn: label vector + mood from prior turns + drives from text."""
        vec = self.label_vector(labels)
        vec[3] = smooth_mood(prior_valences, mood_rate)
        vec[4], vec[5] = drive_levels(text, rng)
        return clip_affect(vec)


def smooth_mood(prior_valences, rate=0.5):
    """EMA of the speaker's earlier valences (0 = neutral when there are none)."""
    m = 0.0
    for v in prior_valences:
        m += rate * (v - m)
    return float(m)


def drive_levels(text, rng=None):
    """(energy_drive, health_drive): small neutral prior, high if the text mentions tiredness / pain."""
    rng = rng if rng is not None else np.random.default_rng(0)
    e, h = rng.beta(1.5, 12.0, size=2)  # mean ~0.11
    if ENERGY_RE.search(text or ""):
        e = rng.uniform(0.6, 0.95)
    if HEALTH_RE.search(text or ""):
        h = rng.uniform(0.6, 0.95)
    return float(e), float(h)


def clip_affect(vec):
    return np.clip(np.asarray(vec, dtype=np.float32), LOW, HIGH)
