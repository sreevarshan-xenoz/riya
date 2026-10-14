"""Teacher requests for synthetic butler replies (no API calls; this only writes the request file).

  python -m jarvis.voice.synth_prompts --n 20000      # -> data/voice/synth_requests.jsonl

Each request pairs a sampled affect vector (uniform, plus oversampled extremes) with a user turn. The teacher
gets the vector *in words* (humans/LLMs follow verbal emotion levels better than numbers, Notion page 23);
the numeric vector is stored with the row and is what the student LM will receive.
Row: {"id", "affect": [8], "affect_description", "user", "user_source", "explicit_emotion_ok", "system", "prompt"}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

if not __package__:  # allow `python jarvis/voice/synth_prompts.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jarvis.voice.affect_map import AFFECT_NAMES, DATA_DIR, HIGH, LOW, clip_affect
from jarvis.voice.persona import NAME, PERSONA_PATH, persona_text

OUT_PATH = DATA_DIR / "voice" / "synth_requests.jsonl"

SEED_USER_TURNS = (
    f"Good morning, {NAME}. What's on the agenda today?",
    "Go through my last 200 emails and tell me what actually needs me.",
    "Plan my whole week around the gym, two deadlines and my sister's birthday.",
    "Remember when I told you about my old school? What was my teacher's name?",
    "Can you work out whether this investment beats paying off the loan early?",
    "Debug this for me while I make tea.",
    "Can you remind me to call my mother at six?",
    "I just failed my driving test again.",
    "I got the job!",
    "What's the weather looking like this afternoon?",
    "Tell me something interesting.",
    "Are you actually alive?",
    "I can't sleep. Talk to me for a bit?",
    "My code won't compile and I've been at it for hours.",
    "How are you feeling today?",
    "Could you summarise what we talked about yesterday?",
    "I think I broke the coffee machine.",
    "Do you ever get bored?",
    "My dog has been sick all week.",
    "What should I cook for dinner?",
    "Can you help me write an apology to my friend?",
    "Turn off the lights, please.",
    "Why is the sky blue?",
    "I'm nervous about my presentation tomorrow.",
    "You made a mistake on that calculation earlier.",
    "Let's play a game.",
    "Are you a person?",
    "I finally finished the marathon!",
    "What do you think about when nobody is talking to you?",
    "My flight got cancelled and I'm stuck at the airport.",
    "Can you recommend a book?",
    "I'm bored.",
    f"Thanks for your help today, {NAME}.",
    "Something smells like it's burning in the kitchen!",
    "How much battery do you have left?",
    "Explain black holes like I'm five.",
    "I had a fight with my brother.",
    "Should I take the new job offer or stay where I am?",
    "Sing me a song.",
    "I'm worried I'm not good enough.",
    "Wake me up at seven tomorrow.",
    "What's two hundred and fifty times four?",
    "Did you miss me?",
    "I'm going to be late for the meeting. What do I do?",
    "Tell me a joke.",
)


def sample_affect(n, rng, extreme_frac=0.35):
    """Uniform over the valid ranges; a fraction gets 1-3 dimensions pushed to an end of their range."""
    vecs = rng.uniform(LOW, HIGH, size=(n, len(AFFECT_NAMES))).astype(np.float32)
    calm = rng.random(n) < 0.6  # drives and surprise are usually low in practice; keep them low unless pushed
    vecs[np.ix_(calm, [4, 5, 7])] *= 0.3
    for i in np.flatnonzero(rng.random(n) < extreme_frac):
        for d in rng.choice(len(AFFECT_NAMES), size=rng.integers(1, 4), replace=False):
            end = HIGH[d] if rng.random() < 0.5 else LOW[d]
            vecs[i, d] = end - np.sign(end - (LOW[d] + HIGH[d]) / 2) * rng.uniform(0, 0.1) * (HIGH[d] - LOW[d])
    return clip_affect(vecs)


def _degree(x):
    """|x| in [0, 1] -> adverb, or None if negligible."""
    x = abs(x)
    return None if x < 0.2 else "slightly" if x < 0.45 else "fairly" if x < 0.7 else "very"


def describe_affect(vec):
    """8-dim affect vector -> short natural-language description, e.g. 'fairly anxious, low on energy, very curious'."""
    v, a, c, m, e, h, cur, s = (float(x) for x in vec)
    parts = []
    if abs(v) >= 0.2:
        if v > 0:
            word = "excited" if a > 0.65 else "content and relaxed" if a < 0.35 else "cheerful"
        elif a > 0.65:
            word = "irritated" if c > 0.6 else "anxious"
        else:
            word = "glum" if a < 0.35 else "uneasy"
        parts.append(f"{_degree(v)} {word}")
    elif a > 0.75:
        parts.append("alert and keyed up")
    elif a < 0.2:
        parts.append("calm and a little drowsy")
    if c < 0.2:
        parts.append("very unsure of itself")
    elif c < 0.35:
        parts.append("somewhat hesitant")
    elif c > 0.8:
        parts.append("very self-assured")
    elif c > 0.65:
        parts.append("fairly confident")
    if m <= -0.5:
        parts.append("in a persistently low mood")
    elif m <= -0.25:
        parts.append("a bit out of sorts lately")
    elif m >= 0.5:
        parts.append("in excellent spirits lately")
    elif m >= 0.25:
        parts.append("in good spirits overall")
    if e > 0.7:
        parts.append("very low on energy (needs to recharge soon)")
    elif e > 0.45:
        parts.append("running low on energy")
    if h > 0.7:
        parts.append("damaged and in need of repair")
    elif h > 0.45:
        parts.append("with some minor system aches")
    if cur > 0.75:
        parts.append("very curious")
    elif cur > 0.6:
        parts.append("curious")
    elif cur < 0.15:
        parts.append("uninterested")
    if s > 0.75:
        parts.append("startled")
    elif s > 0.5:
        parts.append("surprised")
    return ", ".join(parts) if parts else "calm and even-keeled"


def build_prompt(desc, vec, user, explicit_ok):
    style = ("You may name the feeling if it is natural." if explicit_ok else
             "Do not name any emotion or state explicitly; let it show only through word choice, pacing, "
             "hedging and energy.")
    drives = []
    if vec[4] > 0.6:
        drives.append(f"{NAME}'s energy is low, so she may briefly mention needing to recharge")
    if vec[5] > 0.6:
        drives.append(f"{NAME} is damaged, so she may briefly mention needing maintenance")
    raw = ", ".join(f"{k}={x:+.2f}" for k, x in zip(AFFECT_NAMES, vec))
    return (f"{NAME}'s current internal state: {desc}.\n(Raw affect vector, for reference: {raw}.)\n\n"
            f"Write {NAME}'s next reply to the user, in character, under 60 words. {style} "
            + (". ".join(drives) + ". " if drives else "")
            + f"Output only the reply.\n\nUser: {user}")


def load_persona(path=PERSONA_PATH):
    """The persona sheet minus its markdown title/meta header, with the butler's name filled in."""
    return persona_text(path=Path(path))


def load_user_turns(data_dir=DATA_DIR / "voice", limit=5000, rng=None):
    """Seed prompts plus conversation openers from the built ED/DailyDialog files, if present."""
    turns = [(t, "seed") for t in SEED_USER_TURNS]
    found = []
    for name in ("empathetic_dialogues", "daily_dialog"):
        path = Path(data_dir) / f"{name}.jsonl"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                found += [(r["response"], name) for r in map(json.loads, f)
                          if not r["context"] and r["split"] == "train" and 3 <= len(r["response"].split()) <= 60]
    if found:
        rng = rng or np.random.default_rng(0)
        turns += [found[i] for i in rng.permutation(len(found))[:limit]]
    return turns


def make_requests(n, rng, user_turns, persona, explicit_frac=0.3, seed_frac=0.15):
    """seed_frac of requests use the butler-specific seed prompts; the rest sample corpus openers."""
    seeds = [u for u in user_turns if u[1] == "seed"]
    corpus = [u for u in user_turns if u[1] != "seed"] or seeds
    vecs = sample_affect(n, rng)
    for i, vec in enumerate(vecs):
        pool = seeds if rng.random() < seed_frac else corpus
        user, src = pool[rng.integers(len(pool))]
        explicit_ok = bool(rng.random() < explicit_frac)
        desc = describe_affect(vec)
        yield {"id": f"synth-{i:06d}", "affect": [round(float(x), 4) for x in vec], "affect_description": desc,
               "user": user, "user_source": src, "explicit_emotion_ok": explicit_ok, "system": persona,
               "prompt": build_prompt(desc, vec, user, explicit_ok)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT_PATH))
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    persona = load_persona()
    users = load_user_turns(rng=rng)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(make_requests(a.n, rng, users, persona))
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    srcs = {}
    for r in rows:
        srcs[r["user_source"]] = srcs.get(r["user_source"], 0) + 1
    print(f"{len(rows)} requests -> {out}  (user turns: {srcs})")
    for r in rows[:5]:
        print(f"- [{r['affect_description']}] {r['user']}")


if __name__ == "__main__":
    main()
