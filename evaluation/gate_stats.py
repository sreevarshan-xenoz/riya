"""The champion gate's paired test: a permutation test stratified by conversation, with min-p over S1 (the pooled
change, in points or in bad rows) and S2 (the net solid flips). Pure: numpy and stdlib, no I/O. See
docs/superpowers/specs/2026-10-21-riya-v0.10-gate-that-sees-design.md §4.2 and the plan's Resolutions 5, 6 and 8."""
import itertools
import math
import zlib
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

TOL = 1e-9  # every comparison of a statistic or a p-value


@dataclass(frozen=True)
class Obs:
    """One answered row. A failed turn has no Obs, so its slot is simply empty."""
    stratum: str  # the conversation
    probe: str    # the probe_id without "#n"
    cand: bool
    run: int      # the pass number, 1..k
    good: int     # 1 good, 0 bad: callers flip the bad-is-1 metrics


@dataclass(frozen=True)
class Paired:
    delta: float  # S1 observed: "+" is toward better
    band: float   # the null's 99th percentile of |S1|, for display only
    p_better: float
    p_worse: float
    flips_better: tuple[str, ...]
    flips_worse: tuple[str, ...]
    rows: int     # answered candidate-side rows in the shared strata
    probes: int   # distinct probe keys among them
    exact: bool


NO_SHARED = Paired(0.0, 0.0, 1.0, 1.0, (), (), 0, 0, True)


def split_masks(k: int) -> np.ndarray:
    """Every way to call k of a stratum's 2k slots "candidate": shape (C(2k, k), 2k), True for a candidate slot. Base
    pass n is slot n-1 and candidate pass n is slot k+n-1, so the observed split is [False]*k + [True]*k (the last)."""
    masks = np.zeros((math.comb(2 * k, k), 2 * k), dtype=bool)
    for i, cand in enumerate(itertools.combinations(range(2 * k), k)):
        masks[i, list(cand)] = True
    return masks


def _upper(pool: np.ndarray, x: np.ndarray) -> np.ndarray:
    """#{pool >= x} / len(pool) for each x: a one-sided p ranked within the pool."""
    ranked = np.sort(pool)
    return (len(ranked) - np.searchsorted(ranked, x - TOL, side="left")) / len(ranked)


def _min_p(t1: np.ndarray, t2: np.ndarray, o1: float, o2: float) -> float:
    """Min-p over (t1, t2): each statistic's p ranked within the same assignments, then the observed min-p ranked
    among the assignments' min-p's. The pool holds the observed assignment, so this is #{m <= m_obs} / len."""
    m = np.minimum(_upper(t1, t1), _upper(t2, t2))
    m_obs = min(_upper(t1, np.array([o1]))[0], _upper(t2, np.array([o2]))[0])
    return float(np.count_nonzero(m <= m_obs + TOL) / len(m))


def paired_test(obs: Iterable[Obs], *, k: int, count: bool = False, B: int, seed: int, key: str = "") -> Paired:
    """Base against candidate on one metric's rows. Each shared stratum's 2k slots (k passes a side) are reshuffled
    k + k with all their rows; exact over every assignment when prod C(2k, k) <= B, else B seeded draws (the observed
    assignment joins them, so p = (1 + #) / (B + 1)). `count` makes S1 the drop in bad rows (invented)."""
    obs = list(obs)
    sides: dict[str, set[bool]] = {}
    for o in obs:
        if not 1 <= o.run <= k or o.good not in (0, 1):
            raise ValueError(f"bad Obs for k={k}: {o}")
        sides.setdefault(o.stratum, set()).add(bool(o.cand))
    strata = sorted(s for s, c in sides.items() if len(c) == 2)
    if not strata:
        return NO_SHARED
    s_index = {s: i for i, s in enumerate(strata)}
    obs = [o for o in obs if o.stratum in s_index]
    cand_rows = [o for o in obs if o.cand]

    # per-slot sums: slot_good/slot_n (S, 2k) for S1; per (stratum, probe) all-good / all-bad slots for S2
    S, K2 = len(strata), 2 * k
    slot_good = np.zeros((S, K2), dtype=np.int64)
    slot_n = np.zeros((S, K2), dtype=np.int64)
    pkeys = sorted({(o.stratum, o.probe) for o in obs})
    p_index = {pk: i for i, pk in enumerate(pkeys)}
    p_good = np.zeros((len(pkeys), K2), dtype=np.int64)
    p_n = np.zeros((len(pkeys), K2), dtype=np.int64)
    for o in obs:
        s, slot, p = s_index[o.stratum], (k if o.cand else 0) + o.run - 1, p_index[(o.stratum, o.probe)]
        slot_good[s, slot] += o.good
        slot_n[s, slot] += 1
        p_good[p, slot] += o.good
        p_n[p, slot] += 1

    masks = split_masks(k)
    M = len(masks)
    cm = masks.T.astype(np.int64)        # (2k, M): candidate slots
    bm = (~masks).T.astype(np.int64)     # (2k, M): base slots
    observed = int(np.flatnonzero((masks == np.r_[np.zeros(k, bool), np.ones(k, bool)]).all(axis=1))[0])

    # per stratum per split (S, M)
    cg, cn = slot_good @ cm, slot_n @ cm
    bg, bn = slot_good @ bm, slot_n @ bm
    all_good = ((p_n > 0) & (p_good == p_n)).astype(np.int64)  # (P, 2k)
    all_bad = ((p_n > 0) & (p_good == 0)).astype(np.int64)
    better = ((all_bad @ bm) == k) & ((all_good @ cm) == k)     # (P, M)
    worse = ((all_good @ bm) == k) & ((all_bad @ cm) == k)
    net = np.zeros((S, M), dtype=np.int64)
    np.add.at(net, [s_index[s] for s, _ in pkeys], better.astype(np.int64) - worse.astype(np.int64))

    def stats(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(S1, S2) for each assignment column of idx (S, R)."""
        def gather(a: np.ndarray) -> np.ndarray:
            return np.take_along_axis(a, idx, axis=1).sum(axis=0)
        g_c, n_c, g_b, n_b = gather(cg), gather(cn), gather(bg), gather(bn)
        if count:
            s1 = ((n_b - g_b) - (n_c - g_c)).astype(float)
        else:
            rate_c = np.divide(g_c, n_c, out=np.zeros(len(n_c)), where=n_c > 0)
            rate_b = np.divide(g_b, n_b, out=np.zeros(len(n_b)), where=n_b > 0)
            s1 = 100 * (rate_c - rate_b)
        return s1, gather(net).astype(float)

    o1, o2 = (float(v[0]) for v in stats(np.full((S, 1), observed)))
    exact = M ** S <= B
    if exact:
        idx = np.indices((M,) * S).reshape(S, -1)
        t1, t2 = stats(idx)
        band = float(np.percentile(np.abs(t1), 99))
    else:
        draws = np.random.default_rng([seed, zlib.crc32(key.encode())]).integers(0, M, size=(B, S))
        d1, d2 = stats(draws.T)
        band = float(np.percentile(np.abs(d1), 99))
        t1, t2 = np.r_[o1, d1], np.r_[o2, d2]

    p_better = _min_p(t1, t2, o1, o2)
    p_worse = _min_p(-t1, -t2, -o1, -o2)
    flips_better = tuple(sorted(p for (_, p), f in zip(pkeys, better[:, observed]) if f))
    flips_worse = tuple(sorted(p for (_, p), f in zip(pkeys, worse[:, observed]) if f))
    return Paired(o1, band, p_better, p_worse, flips_better, flips_worse, len(cand_rows),
                  len({o.probe for o in cand_rows}), exact)
