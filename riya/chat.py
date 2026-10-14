"""Riya's conversational runtime: feelings + long-session memory + streaming, on any OpenAI-compatible server
(llama-server from riya/server.py). The voice layer consumes `Riya.reply()` sentence by sentence.

  python -m riya.chat --url http://127.0.0.1:8090

Feelings: the same AffectCore as the world-model agent, with a conversational 'body':
  energy  = session fatigue (drains with continuous talk, recovers during breaks)
  health  = system comfort (drops when game mode squeezes her onto the CPU)
plus appraisal of what the user says (warmth/hostility -> valence, intensity -> arousal, novelty -> surprise).
Memory: last turns verbatim + a rolling summary of older turns, saved to disk between sessions.
"""
import argparse
import json
import os
import re
import time
import urllib.request

import numpy as np

from jarvis.voice.persona import NAME, system_prompt
from jarvis.world.affect import AffectCore

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


class Riya:
    def __init__(self, url, memory_path="riya_memory.json", keep_turns=12, summarize_every=8):
        self.url, self.memory_path = url.rstrip("/"), memory_path
        self.keep_turns, self.summarize_every = keep_turns, summarize_every
        self.affect = AffectCore()
        self.body = np.array([0.8, 1.0], dtype=np.float32)  # [energy, comfort]
        self.turns, self.summary, self.since_summary = [], "", 0
        self.last_t, self.prev_words = time.time(), set()
        self.on_cpu = False
        if os.path.exists(memory_path):
            m = json.load(open(memory_path, encoding="utf-8"))
            self.summary, self.turns = m.get("summary", ""), m.get("turns", [])[-keep_turns:]

    # ---- feelings ----------------------------------------------------------------------------------
    def feel(self, user_text):
        now = time.time()
        gap_min = (now - self.last_t) / 60
        body = self.body.copy()
        body[0] = min(0.9, body[0] + 0.02 * gap_min) if gap_min > 5 else body[0] - 0.01  # rest vs continuous talk
        body[1] = 0.7 if self.on_cpu else min(1.0, body[1] + 0.05)
        valence, arousal, novelty = appraise(user_text, self.prev_words)
        # Social appraisal acts like a body signal: warmth soothes, hostility hurts a little.
        body[1] = float(np.clip(body[1] + 0.1 * valence, 0, 1))
        self.affect.update(self.body, body, prediction_error=0.6 * novelty + 0.4 * arousal)
        s = self.affect.state
        s.valence = float(np.clip(0.6 * s.valence + 0.4 * valence, -1, 1))
        s.arousal = float(np.clip(0.7 * s.arousal + 0.3 * arousal, 0, 1))
        self.body, self.last_t = body, now
        self.prev_words = set(re.findall(r"[a-z']+", user_text.lower()))

    def state_tag(self):
        s = self.affect.state
        return (f"[{NAME}'s state: feeling {self.affect.label()} | valence {s.valence:+.2f} arousal {s.arousal:.2f} "
                f"confidence {s.confidence:.2f} mood {s.mood:+.2f} energy {self.body[0]:.2f} health {self.body[1]:.2f} "
                f"curiosity {s.curiosity:.2f} surprise {s.surprise:.2f}]")

    # ---- model calls ---------------------------------------------------------------------------------
    def _stream(self, messages, max_tokens=200):
        body = {"messages": messages, "max_tokens": max_tokens, "stream": True, "temperature": 0.8, "cache_prompt": True}
        req = urllib.request.Request(f"{self.url}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if line.startswith("data:") and not line.endswith("[DONE]"):
                    piece = json.loads(line[5:])["choices"][0].get("delta", {}).get("content")
                    if piece:
                        yield piece

    def messages(self, user_text):
        system = system_prompt() + "\n" + self.state_tag()
        if self.summary:
            system += f"\nWhat you remember from earlier: {self.summary}"
        msgs = [{"role": "system", "content": system}] + self.turns + [{"role": "user", "content": user_text}]
        return msgs

    def reply(self, user_text):
        """Yield Riya's reply sentence by sentence (for streaming TTS)."""
        self.feel(user_text)
        buf, full = "", []
        for piece in self._stream(self.messages(user_text)):
            buf += piece
            full.append(piece)
            while (m := SENTENCE_END.search(buf)):
                yield buf[: m.end()].strip()
                buf = buf[m.end():]
        if buf.strip():
            yield buf.strip()
        self.turns += [{"role": "user", "content": user_text}, {"role": "assistant", "content": "".join(full).strip()}]
        self.since_summary += 1
        if len(self.turns) > 2 * self.keep_turns and self.since_summary >= self.summarize_every:
            self._summarize()
        self.save()

    def _summarize(self):
        old, self.turns = self.turns[: -self.keep_turns], self.turns[-self.keep_turns:]
        convo = "\n".join(f"{t['role']}: {t['content']}" for t in old)
        prompt = (f"Previous summary: {self.summary or '(none)'}\n\nNew conversation:\n{convo}\n\n"
                  "Update the summary of what matters about the user and this relationship (facts, plans, feelings, "
                  "preferences) in under 120 words.")
        self.summary = "".join(self._stream([{"role": "user", "content": prompt}], 220)).strip()
        self.since_summary = 0

    def save(self):
        with open(self.memory_path, "w", encoding="utf-8") as f:
            json.dump({"summary": self.summary, "turns": self.turns}, f, ensure_ascii=False, indent=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8090")
    p.add_argument("--memory", default="riya_memory.json")
    p.add_argument("--show_state", action="store_true")
    a = p.parse_args()
    riya = Riya(a.url, a.memory)
    print(f"{NAME} is here. (Ctrl+C to quit)")
    while True:
        user = input("you> ").strip()
        if not user:
            continue
        t0 = time.time()
        for i, sentence in enumerate(riya.reply(user)):
            if i == 0:
                first = time.time() - t0
            print(sentence, end=" ", flush=True)
        print(f"\n   ({first * 1000:.0f} ms to first sentence)" + (f" {riya.state_tag()}" if a.show_state else ""))


if __name__ == "__main__":
    main()
