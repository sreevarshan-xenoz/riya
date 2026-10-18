"""Appraisal of what the user says: valence, arousal and novelty from a lexicon (moved unchanged from riya/chat.py)."""
import re

import numpy as np

POS = set("love thanks thank great awesome amazing good nice happy glad wonderful beautiful cool brilliant proud "
          "yay haha lol fun excited best perfect sweet kind".split())
NEG = set("hate stupid idiot useless bad awful terrible sad angry annoying worst boring shut dumb wrong failed "
          "tired sick hurt scared worried alone lonely".split())
INTENSE = set("!!! urgent now help emergency wow omg seriously really very".split())
SENTENCE_END = re.compile(r"([.!?…]+[\"')\]]?\s)")


def appraise(text, prev_words):
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0, 0.0, 0.0
    pos, neg = sum(w in POS for w in words), sum(w in NEG for w in words)
    valence = (pos - neg) / max(3, len(words) ** 0.5)
    arousal = min(1.0, 0.2 * text.count("!") + 0.3 * sum(w in INTENSE for w in words) + 0.3 * text.isupper())
    novelty = 1.0 - len(set(words) & prev_words) / max(1, len(set(words)))
    return float(np.clip(valence, -1, 1)), arousal, novelty
