"""Build the affect-labelled dialogue set (CPU only): one JSONL row per speaker turn.

  python -m jarvis.voice.build_dataset                 # all sources -> data/voice/*.jsonl + stats.json
  python -m jarvis.voice.build_dataset --max_dialogs 200

Row: {"source", "split", "dialog_id", "context": [{"role","text"}], "response", "affect": [8 floats], "label"}.
Roles are from the responder's point of view: the responder's own earlier turns are "assistant", the other
person's are "user". Sources (the original script-based HF datasets no longer load with `datasets>=3`, so we use
parquet re-uploads of the same data):
- EmpatheticDialogues: `Estwld/empathetic_dialogues_llm` (Rashkin et al. 2019, regrouped per conversation,
  "_comma_" restored). Its "user" role is the ED *speaker*, whose emotion the label describes, so only those turns
  become responses, all with the conversation label.
- DailyDialog: `pixelsandpointers/better_daily_dialog` (Li et al. 2017, one row per utterance with its emotion id
  0..6 and dialogue act). Every utterance becomes a response with its own label.
"""
import argparse
import ast
import json
import re
import sys
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

if not __package__:  # allow `python jarvis/voice/build_dataset.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jarvis.voice.affect_map import AFFECT_NAMES, DAILYDIALOG_LABELS, DATA_DIR, AffectMapper

ED_ID = "Estwld/empathetic_dialogues_llm"
DD_ID = "pixelsandpointers/better_daily_dialog"
OUT_DIR = DATA_DIR / "voice"


PUNCT = str.maketrans({"’": "'", "‘": "'", "′": "'", "“": '"', "”": '"',
                       "。": ".", "、": ","})


def detok(text):
    """Undo DailyDialog-style tokenised spacing and curly quotes ("can ’ t , ok ?" -> "can't, ok?")."""
    t = re.sub(r"\s+([,.!?;:%)])", r"\1", " ".join(text.translate(PUNCT).split()))
    t = re.sub(r"([(])\s+", r"\1", t)
    t = re.sub(r"\s*'\s*(s|t|re|ve|ll|d|m)\b", r"'\1", t)
    t = re.sub(r"\s+n't\b", "n't", t)
    return t.strip()


def turn_rng(*key):
    return np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode()))


def dialog_rows(source, split, dialog_id, turns, speakers, labels, mapper, keep):
    """turns: texts; speakers: speaker id per turn; labels: label per turn; keep(i) -> emit turn i as a response."""
    rows, valences = [], defaultdict(list)
    for i, (text, spk, lab) in enumerate(zip(turns, speakers, labels)):
        if keep(i) and text:
            own = " ".join(t for t, s in zip(turns[:i + 1], speakers[:i + 1]) if s == spk)
            vec = mapper.turn_affect(lab, own, valences[spk], turn_rng(source, split, dialog_id, i))
            ctx = [{"role": "assistant" if s == spk else "user", "text": t} for t, s in zip(turns[:i], speakers[:i])]
            rows.append({"source": source, "split": split, "dialog_id": str(dialog_id), "context": ctx,
                         "response": text, "affect": [round(float(x), 4) for x in vec], "label": lab})
        valences[spk].append(mapper.label_vector(lab)[0])
    return rows


def empathetic_dialogues(mapper, max_dialogs=None):
    from datasets import load_dataset
    for split, ds in load_dataset(ED_ID).items():
        split = {"valid": "validation"}.get(split, split)
        for n, ex in enumerate(ds):
            if max_dialogs and n >= max_dialogs:
                break
            conv = ex["conversations"]
            conv = ast.literal_eval(conv) if isinstance(conv, str) else conv
            turns = [" ".join(c["content"].split()) for c in conv]
            speakers = [c["role"] for c in conv]  # "user" = ED speaker (labelled), "assistant" = listener
            labels = [ex["emotion"] if s == "user" else "neutral" for s in speakers]
            yield from dialog_rows("empathetic_dialogues", split, ex["conv_id"], turns, speakers, labels, mapper,
                                   keep=lambda i: speakers[i] == "user")


def daily_dialog(mapper, max_dialogs=None):
    from datasets import load_dataset
    for split, ds in load_dataset(DD_ID).items():
        dialogs = defaultdict(list)
        for ex in ds:  # rows are in utterance order within each dialog_id
            dialogs[ex["dialog_id"]].append(ex)
        for n, (did, utts) in enumerate(dialogs.items()):
            if max_dialogs and n >= max_dialogs:
                break
            turns = [detok(u["utterance"]) for u in utts]
            labels = [DAILYDIALOG_LABELS[int(u["emotion"])] for u in utts]
            yield from dialog_rows("daily_dialog", split, did, turns, [i % 2 for i in range(len(turns))], labels,
                                   mapper, keep=lambda i: True)


SOURCES = {"empathetic_dialogues": empathetic_dialogues, "daily_dialog": daily_dialog}


def stats(rows):
    if not rows:
        return {"rows": 0}
    aff = np.array([r["affect"] for r in rows], dtype=np.float32)
    return {"rows": len(rows), "splits": dict(Counter(r["split"] for r in rows)),
            "labels": dict(Counter(r["label"] for r in rows).most_common()),
            "affect_mean": {k: round(float(v), 3) for k, v in zip(AFFECT_NAMES, aff.mean(0))},
            "affect_std": {k: round(float(v), 3) for k, v in zip(AFFECT_NAMES, aff.std(0))},
            "empty_context": sum(not r["context"] for r in rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--out_dir", default=str(OUT_DIR))
    p.add_argument("--max_dialogs", type=int, default=None, help="per split, for quick runs")
    p.add_argument("--no_download_lexicon", action="store_true")
    a = p.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapper = AffectMapper.load(download=not a.no_download_lexicon)
    print(f"affect map: {mapper.source}")
    all_stats = {"lexicon": mapper.source, "datasets": {"empathetic_dialogues": ED_ID, "daily_dialog": DD_ID}}
    for name in a.sources:
        rows = list(SOURCES[name](mapper, a.max_dialogs))
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        s = all_stats[name] = stats(rows)
        print(f"\n== {name}: {s['rows']} rows {s.get('splits')} -> {out / (name + '.jsonl')}")
        print("labels:", ", ".join(f"{k} {v}" for k, v in s.get("labels", {}).items()))
        print("mean affect:", "  ".join(f"{k} {v:+.3f}" for k, v in s.get("affect_mean", {}).items()))
    (out / "stats.json").write_text(json.dumps(all_stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
