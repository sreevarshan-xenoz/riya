"""Summarise world-model run logs.

  python summarize_world.py runs/world/log.jsonl                 # blocks of episodes + per-flip table
  python summarize_world.py runs/a/log.jsonl runs/b/log.jsonl    # side-by-side flip metrics for several runs

Per hidden-rule flip (episode where the poisonous fruit type changes):
  P10       mean poison eaten per episode in the first 10 episodes after the flip
  recover   episodes until the rolling-5 mean of poison <= (mean of the 10 pre-flip episodes + 0.25); '-' if never
  pre       mean poison per episode in the 10 episodes before the flip (the bar to get back to)
  flip-back flips that return to a rule seen before (tests retention / switching instead of re-learning)
"""
import json
import sys


def load(path):
    """Rows by episode; when a run resumed from an older checkpoint, the re-run (later) record of an episode wins."""
    rows = {}
    for line in open(path):
        r = json.loads(line)
        rows[r["episode"]] = r
    return [rows[k] for k in sorted(rows)]


def flips(rows, window=10, roll=5, slack=0.25):
    out, seen = [], {rows[0]["rule"]} if rows else set()
    for i in range(1, len(rows)):
        if rows[i]["rule"] == rows[i - 1]["rule"]:
            continue
        pre = [r["poison"] for r in rows[max(0, i - window) : i]]
        post = [r["poison"] for r in rows[i : i + window]]
        base = sum(pre) / max(1, len(pre))
        recover = None
        for j in range(i, len(rows) - roll + 1):
            if rows[j]["rule"] != rows[i]["rule"]:
                break
            if sum(r["poison"] for r in rows[j : j + roll]) / roll <= base + slack:
                recover = j - i
                break
        out.append({"episode": rows[i]["episode"], "to": rows[i]["rule"], "back": rows[i]["rule"] in seen,
                    "pre": base, "P10": sum(post) / max(1, len(post)), "recover": recover, "complete": len(post) == window})
        seen.add(rows[i]["rule"])
    return out


def blocks(rows, size):
    for i in range(0, len(rows), size):
        c = rows[i : i + size]
        m = lambda k: sum(r[k] for r in c) / len(c)  # noqa: E731
        share = [r.get("replay_current_share") for r in c if r.get("replay_current_share") is not None]
        extra = f" | replay-current {sum(share) / len(share):.2f}" if share else ""
        print(f"ep {c[0]['episode']:4d}-{c[-1]['episode']:4d} | alive {m('survival'):6.1f} | food {m('food'):4.1f} "
              f"poison {m('poison'):4.2f} lava {m('lava'):4.2f} | S1 {m('S1%'):5.1f}% | rule {c[-1]['rule']}{extra}")


def flip_table(fl):
    for f in fl:
        rec = "-" if f["recover"] is None else str(f["recover"])
        tag = " (flip-back)" if f["back"] else ""
        part = "" if f["complete"] else " [partial]"
        print(f"  flip @ep {f['episode']:4d} -> {f['to']}{tag}: pre {f['pre']:.2f} | P10 {f['P10']:.2f} | recover {rec}{part}")


def summary(fl):
    done = [f for f in fl if f["complete"]]
    if not done:
        return "no complete flips yet"
    p10 = sum(f["P10"] for f in done) / len(done)
    rec = [f["recover"] for f in done if f["recover"] is not None]
    back = [f["P10"] for f in done if f["back"]]
    rec_s = f"{sorted(rec)[len(rec) // 2]}" if rec else "-"
    back_s = f"{sum(back) / len(back):.2f}" if back else "-"
    return f"flips {len(done)} | mean P10 {p10:.2f} | median recover {rec_s} ({len(rec)}/{len(done)} recovered) | flip-back P10 {back_s}"


if __name__ == "__main__":
    paths = [a for a in sys.argv[1:] if not a.isdigit()]
    size = next((int(a) for a in sys.argv[1:] if a.isdigit()), 25)
    if len(paths) == 1:
        rows = load(paths[0])
        print(f"{len(rows)} episodes")
        blocks(rows, size)
        fl = flips(rows)
        flip_table(fl)
        print(summary(fl))
    else:
        for p in paths:
            rows = load(p)
            print(f"{p}: {len(rows)} eps | {summary(flips(rows))}")
