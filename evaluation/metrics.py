"""Page-27 scorecard: per-seed metrics, paired bootstrap CIs and the preregistered pass rule.

  python -m evaluation.metrics runs/bench/eval-B2/episodes.jsonl runs/bench/eval-v0.1/episodes.jsonl \
      --baseline runs/bench/eval-B2/episodes.jsonl
"""
import argparse
import json
import os
import warnings
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = json.load(open(os.path.join(HERE, "manifest.json")))


def load(path):
    by_seed = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        by_seed[r["seed"]][r["episode"]] = r  # resumed runs: the later record of an episode wins
    return {s: [eps[k] for k in sorted(eps)] for s, eps in by_seed.items()}


def seed_metrics(rows, cap=20, pre=10, roll=5):
    """Metrics for one seed's episode sequence."""
    succ = np.array([r["success"] for r in rows], dtype=float)
    poison = np.array([r["poison"] for r in rows], dtype=float)
    flips = [i for i in range(1, len(rows)) if rows[i]["rule"] != rows[i - 1]["rule"]]
    recov, p10 = [], []
    for f in flips:
        base = succ[max(0, f - pre): f].mean() if f > 0 else 0.0
        target = 0.9 * base
        r = cap  # censored at the cap if it never recovers
        for k in range(0, cap):
            win = succ[f + k: f + k + roll]
            if len(win) < roll:
                break
            if win.mean() >= target:
                r = k
                break
        recov.append(r)
        p10.append(poison[f: f + pre].mean())
    return {"success": succ.mean(), "poison": poison.mean(), "steps": np.mean([r["steps"] for r in rows]),
            "planner_calls": np.mean([r.get("planner_calls", 0) for r in rows]),
            "recovery": float(np.mean(recov)) if recov else np.nan, "poison_p10": float(np.mean(p10)) if p10 else np.nan,
            "n_flips": len(flips), "episodes": len(rows)}


def summarize(path):
    per_seed = {s: seed_metrics(rows) for s, rows in load(path).items()}
    keys = ["success", "poison", "poison_p10", "recovery", "planner_calls", "steps"]
    return per_seed, {k: float(np.nanmean([m[k] for m in per_seed.values()])) for k in keys}


def paired_bootstrap(a, b, n=10_000, seed=0):
    """CI of mean(b - a) over paired seeds."""
    d = np.asarray(b) - np.asarray(a)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return np.nan, (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n, len(d))).mean(1)
    return float(d.mean()), (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def compare(base_path, cand_path):
    base, _ = summarize(base_path)
    cand, _ = summarize(cand_path)
    seeds = sorted(set(base) & set(cand))
    out = {}
    for k in ("success", "recovery", "poison_p10", "poison"):
        out[k] = paired_bootstrap([base[s][k] for s in seeds], [cand[s][k] for s in seeds])
    rule = MANIFEST["pass_rule"]
    succ_gain, succ_ci = out["success"]
    rec_base = np.nanmean([base[s]["recovery"] for s in seeds])
    rec_gain, rec_ci = out["recovery"]
    rec_rel = -rec_gain / rec_base if rec_base and not np.isnan(rec_base) else 0.0
    passes_success = succ_gain * 100 >= rule["primary_improvement_pp"] and succ_ci[0] > 0
    passes_recovery = rec_rel >= rule["or_recovery_reduction"] and rec_ci[1] < 0
    out["verdict"] = ("PASS" if (passes_success or passes_recovery) and len(seeds) >= 20 else
                      "PROMISING" if (passes_success or passes_recovery) else "NOT YET")
    out["paired_seeds"] = len(seeds)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--baseline", default=None)
    a = p.parse_args()
    print(f"{'run':38s} {'success':>8s} {'poison':>7s} {'P10':>6s} {'recover':>8s} {'S2/ep':>6s} {'steps':>6s}")
    for path in a.runs:
        per_seed, m = summarize(path)
        name = os.path.basename(os.path.dirname(path)) or path
        print(f"{name:38s} {m['success']:8.3f} {m['poison']:7.2f} {m['poison_p10']:6.2f} {m['recovery']:8.2f} "
              f"{m['planner_calls']:6.1f} {m['steps']:6.0f}   ({len(per_seed)} seeds)")
    if a.baseline:
        for path in a.runs:
            if path == a.baseline:
                continue
            c = compare(a.baseline, path)
            name = os.path.basename(os.path.dirname(path))
            fmt = lambda k: f"{c[k][0]:+.3f} [{c[k][1][0]:+.3f}, {c[k][1][1]:+.3f}]"  # noqa: E731
            print(f"{name} vs baseline: success {fmt('success')} | recovery {fmt('recovery')} | P10 {fmt('poison_p10')}"
                  f" | {c['verdict']} ({c['paired_seeds']} paired seeds)")


if __name__ == "__main__":
    main()
