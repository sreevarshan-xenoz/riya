"""The authoring hold-out check: no training row may overlap a gate probe. CPU only; reads, never writes.

  python -m evaluation.probe_holdout --old-probes scratch_runs/riya_v1/gate/probes_v0.9.json [--files F ...]
                                     [--seeds 0 1 2]

It checks, against the current probes (riya.train.holdout.PROBES_PATH):
1. that every new (g_*) probe has >= HOLDOUT_N normalised words, so the 6-word window holds it out;
2. the real round-2 files (requests_all.jsonl and sft.jsonl by default), row by row;
3. build(seed) under the old (v0.9) probes: a build under the new ones would re-draw a colliding row and hide the hit;
4. build(seed) under the new probes, which must raise nothing.
A hit is fixed by rewriting the probe before any model answers it, never by editing a training row (round 2 trained on
them). Prints one `HIT <source> <row id> <probe>` line per hit, `MISSING <file>`, then the verdict.
Exit 0 on 0 hits, 1 on hits, 2 when a file is missing (the check can't be claimed).
See docs/superpowers/specs/2026-10-21-riya-v0.10-gate-that-sees-design.md §4.1."""
import argparse
import json
from collections.abc import Callable
from pathlib import Path

from riya.train import build_sft, holdout
from riya.train.fmt import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
FILES = (ROOT / "scratch_runs" / "riya_v1" / "requests_all.jsonl", ROOT / "scratch_runs" / "riya_v1" / "sft.jsonl")
SEEDS = (0, 1, 2)


def hits(rows: list[dict], texts: list[str], facts: list[str]) -> list[tuple[str, str]]:
    """(row id, the probe text or fact it hit) for every row with a holdout.first_hit, in order."""
    return [(row["id"], hit) for row in rows if (hit := holdout.first_hit(row, texts, facts)) is not None]


def short_probes(probes: dict) -> list[str]:
    """The ids of g_* conversations whose turn has under HOLDOUT_N normalised words."""
    return [c["id"] for c in probes["conversations"] if c["id"].startswith("g_")
            and any(len(holdout.normalize(t["text"]).split()) < holdout.HOLDOUT_N for t in c["turns"])]


def main(argv: list[str] | None = None, *, build: Callable[[int], list[dict]] = build_sft.build) -> int:
    p = argparse.ArgumentParser(prog="python -m evaluation.probe_holdout",
                                description="Check that no training row overlaps a gate probe (writes nothing).")
    p.add_argument("--old-probes", type=Path, required=True, help="the v0.9 probe file the seeds were drawn under")
    p.add_argument("--files", type=Path, nargs="+", default=list(FILES))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = p.parse_args(argv)
    current = holdout.PROBES_PATH
    texts, facts = holdout.probe_texts(current), holdout.probe_facts(current)
    found: list[tuple[str, str, str]] = []  # (source, row id, probe)
    missing: list[Path] = []
    for cid in short_probes(json.loads(current.read_text(encoding="utf-8"))):
        found.append(("short", cid, f"under {holdout.HOLDOUT_N} normalised words"))
    for path in args.files:
        if not path.exists():
            missing.append(path)
            print(f"MISSING {path}")
            continue
        rows = read_jsonl(path)
        print(f"Checked {len(rows)} rows of {path} ({path.stat().st_size} bytes)")
        found += [(str(path), rid, hit) for rid, hit in hits(rows, texts, facts)]
    for seed in args.seeds:
        with holdout.probes_at(args.old_probes):
            rows = build(seed)
        print(f"Checked {len(rows)} rows of build(seed={seed}) under {args.old_probes}")
        found += [(f"build-old:seed={seed}", rid, hit) for rid, hit in hits(rows, texts, facts)]
    for seed in args.seeds:
        try:
            build(seed)
        except ValueError as e:
            found.append((f"build:seed={seed}", "-", str(e)))
    for source, rid, hit in found:
        print(f"HIT {source} {rid} {hit}")
    if missing:
        print(f"VERDICT: incomplete ({len(missing)} file(s) missing, {len(found)} hits)")
        return 2
    print(f"VERDICT: {len(found)} hits")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
