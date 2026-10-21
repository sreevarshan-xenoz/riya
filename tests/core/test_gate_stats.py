"""v0.10 Task 4: gate_stats, the stratified paired permutation test with min-p over S1 (the pooled change) and S2 (the
solid flips). Exact p-values on hand-enumerable cases, the (1 + #)/(B + 1) formula on sampled ones, and a seeded null."""
import itertools
import math
import time
import zlib

import numpy as np

from evaluation.gate_stats import Obs, Paired, paired_test, split_masks

TOL = 1e-9


def side(stratum, probe, cand, outcomes):
    """Obs for one side of one probe; None in `outcomes` is a failed (missing) pass."""
    return [Obs(stratum, probe, cand, n, g) for n, g in enumerate(outcomes, 1) if g is not None]


def probe_rows(stratum, probe, base, cand):
    return side(stratum, probe, False, base) + side(stratum, probe, True, cand)


# --- a pure-Python reference, written straight from Resolution 5 ---------------------------------------------------

def ref_stats(obs, k, count, strata, assignment, masks):
    """(S1, S2) when stratum strata[i] takes split masks[assignment[i]]."""
    mask_of = {s: masks[a] for s, a in zip(strata, assignment)}
    gc = nc = gb = nb = 0
    slots = {}
    for o in obs:
        if o.stratum not in mask_of:
            continue
        slot = (k if o.cand else 0) + o.run - 1
        if mask_of[o.stratum][slot]:
            gc, nc = gc + o.good, nc + 1
        else:
            gb, nb = gb + o.good, nb + 1
        slots[(o.stratum, o.probe, slot)] = o.good
    if count:
        s1 = (nb - gb) - (nc - gc)
    else:
        s1 = 100 * ((gc / nc if nc else 0.0) - (gb / nb if nb else 0.0))
    s2 = 0
    for stratum, probe in {(o.stratum, o.probe) for o in obs if o.stratum in mask_of}:
        mask = mask_of[stratum]
        cand = [slots.get((stratum, probe, j)) for j in range(2 * k) if mask[j]]
        base = [slots.get((stratum, probe, j)) for j in range(2 * k) if not mask[j]]
        if base == [0] * k and cand == [1] * k:
            s2 += 1
        if base == [1] * k and cand == [0] * k:
            s2 -= 1
    return s1, s2


def ref_min_p(pool, observed):
    """pool: (S1, S2) over every assignment of the reference set (the observed one included). Returns (p_better,
    p_worse) by min-p, each statistic's p ranked within the pool."""
    n = len(pool)

    def upper(values, x):
        return sum(v >= x - TOL for v in values) / n

    out = []
    for sign in (1, -1):
        t1 = [sign * a for a, _ in pool]
        t2 = [sign * b for _, b in pool]
        m = [min(upper(t1, a), upper(t2, b)) for a, b in zip(t1, t2)]
        m_obs = min(upper(t1, sign * observed[0]), upper(t2, sign * observed[1]))
        out.append(sum(x <= m_obs + TOL for x in m) / n)
    return tuple(out)


def shared_strata(obs):
    sides = {}
    for o in obs:
        sides.setdefault(o.stratum, set()).add(o.cand)
    return sorted(s for s, c in sides.items() if c == {False, True})


def random_table(rng, strata, k, probes_per=2, missing=0.15):
    obs = []
    for s in strata:
        for q in range(probes_per):
            rate = rng.uniform(0.1, 0.9)
            for cand in (False, True):
                outs = [None if rng.random() < missing else int(rng.random() < rate) for _ in range(k)]
                obs += side(s, f"{s}.q{q}", cand, outs)
    return obs


# --- split_masks ----------------------------------------------------------------------------------------------------

def test_split_masks_cover_every_k_of_2k_split_once():
    for k in (1, 2, 3, 4):
        masks = split_masks(k)
        assert masks.shape == (math.comb(2 * k, k), 2 * k) and masks.dtype == bool
        assert (masks.sum(axis=1) == k).all()
        assert len({tuple(m) for m in masks}) == len(masks)
        assert tuple([False] * k + [True] * k) in {tuple(m) for m in masks}


# --- exact cases ----------------------------------------------------------------------------------------------------

def test_exact_p_on_one_stratum():
    # 20 splits of 6 slots: the candidate side gets j of the base's three 1s with counts 1, 9, 9, 1
    r = paired_test(probe_rows("s", "s.q", [1, 1, 1], [0, 0, 0]), k=3, B=10_000, seed=0)
    assert r.exact is True
    assert r.delta == -100.0
    assert r.p_worse == 0.05 and r.p_better == 1.0
    assert r.band == 100.0
    assert r.flips_worse == ("s.q",) and r.flips_better == ()
    assert (r.rows, r.probes) == (3, 1)
    # one-sided: the mirror image swaps the two p-values
    m = paired_test(probe_rows("s", "s.q", [0, 0, 0], [1, 1, 1]), k=3, B=10_000, seed=0)
    assert (m.delta, m.p_better, m.p_worse) == (100.0, 0.05, 1.0)
    assert m.flips_better == ("s.q",) and m.flips_worse == ()


def test_exact_p_on_one_stratum_at_k_4():
    """v0.10.1 runs at k = 4: 70 splits of 8 slots, the observed one the single most extreme, so p is 1/70."""
    r = paired_test(probe_rows("s", "s.q", [1] * 4, [0] * 4), k=4, B=10_000, seed=0)
    assert r.exact is True and r.delta == -100.0
    assert abs(r.p_worse - 1 / 70) < TOL and r.p_better == 1.0
    assert r.flips_worse == ("s.q",) and r.flips_better == () and (r.rows, r.probes) == (4, 1)


def test_identical_outcomes_give_p_one():
    obs = []
    for i in range(30):
        g = i % 2
        obs += probe_rows(f"c{i:02d}", f"c{i:02d}.q", [g] * 3, [g] * 3)
    r = paired_test(obs, k=3, B=2_000, seed=1)
    assert r.exact is False
    assert r.p_better == r.p_worse == 1.0 and r.delta == 0.0 and r.band == 0.0
    assert r.flips_better == r.flips_worse == ()
    # identical but mixed within each stratum: about 0.50, never a small p (spec 4.3)
    mixed = []
    for i in range(30):
        mixed += probe_rows(f"c{i:02d}", f"c{i:02d}.q", [1, 1, 0], [1, 1, 0])
    r = paired_test(mixed, k=3, B=2_000, seed=1)
    assert r.delta == 0.0
    assert r.p_worse >= 0.5 and r.p_better >= 0.5


def test_rows_and_probes_count_the_candidate_side():
    obs = probe_rows("a", "a.q", [1, 0, 1], [1, 1, 1]) + probe_rows("b", "b.q", [0, 1, 1], [1, None, 0])
    r = paired_test(obs, k=3, B=10_000, seed=0)
    assert (r.rows, r.probes) == (5, 2)
    # a stratum with base rows only adds nothing
    r2 = paired_test(obs + side("c", "c.q", False, [1, 1, 1]), k=3, B=10_000, seed=0)
    assert (r2.rows, r2.probes) == (5, 2)
    assert r2 == r


def test_min_p_matches_a_brute_force_reference():
    k, masks = 3, [tuple(m) for m in split_masks(3)]
    observed_split = masks.index(tuple([False] * 3 + [True] * 3))
    for t in range(5):
        rng = np.random.default_rng(100 + t)
        obs = random_table(rng, ["x", "y"], k)
        strata = shared_strata(obs)
        assert len(strata) == 2
        for count in (False, True):
            pool = [ref_stats(obs, k, count, strata, a, masks) for a in itertools.product(range(len(masks)), repeat=2)]
            observed = ref_stats(obs, k, count, strata, (observed_split,) * 2, masks)
            p_better, p_worse = ref_min_p(pool, observed)
            r = paired_test(obs, k=k, count=count, B=400, seed=t)
            assert r.exact is True
            assert abs(r.delta - observed[0]) <= 1e-12
            assert abs(r.p_better - p_better) <= 1e-12 and abs(r.p_worse - p_worse) <= 1e-12
            assert abs(r.band - float(np.percentile([abs(a) for a, _ in pool], 99))) <= 1e-12


def test_min_p_uses_the_solid_flips_not_s1_alone():
    # x.q0 flips solidly to worse while the noisier probes drift up: S1 alone gives (p_better, p_worse) = (0.50, 0.90),
    # and the solid flip pulls p_worse to 0.05 (review I1)
    obs = (probe_rows("x", "x.q0", [1, 1, 1], [0, 0, 0]) + probe_rows("x", "x.q1", [0, 0, 1], [1, 1, 1])
           + probe_rows("y", "y.q0", [0, 1, 0], [1, 1, 1]))
    r = paired_test(obs, k=3, B=10_000, seed=0)
    assert r.exact is True and r.delta > 0
    assert r.p_worse == 0.05 and r.p_better == 0.51
    assert r.flips_worse == ("x.q0",) and r.flips_better == ()
    masks = [tuple(m) for m in split_masks(3)]
    observed_split = masks.index(tuple([False] * 3 + [True] * 3))
    pool = [ref_stats(obs, 3, False, ["x", "y"], a, masks) for a in itertools.product(range(len(masks)), repeat=2)]
    observed = ref_stats(obs, 3, False, ["x", "y"], (observed_split,) * 2, masks)
    assert ref_min_p(pool, observed) == (r.p_better, r.p_worse)
    s1_only = ref_min_p([(a, a) for a, _ in pool], (observed[0], observed[0]))
    assert s1_only == (0.5, 0.9)


def test_the_tolerance_counts_float_ties_as_ties():
    # pooled rates over different denominators reach equal values through different fractions, which differ in the
    # last ulp; with TOL = 0 this table's p_better drops from 0.7075 to 0.6375 (review M1)
    obs = random_table(np.random.default_rng(23), ["x", "y"], 3, probes_per=3, missing=0.35)
    strata = shared_strata(obs)
    assert strata == ["x", "y"]
    r = paired_test(obs, k=3, B=400, seed=0)
    assert r.exact is True
    assert (r.p_better, r.p_worse) == (0.7075, 0.4525)
    masks = [tuple(m) for m in split_masks(3)]
    observed_split = masks.index(tuple([False] * 3 + [True] * 3))
    pool = [ref_stats(obs, 3, False, strata, a, masks) for a in itertools.product(range(len(masks)), repeat=2)]
    observed = ref_stats(obs, 3, False, strata, (observed_split,) * 2, masks)
    p_better, p_worse = ref_min_p(pool, observed)
    assert abs(r.p_better - p_better) <= 1e-12 and abs(r.p_worse - p_worse) <= 1e-12


def test_an_emptied_side_counts_as_a_zero_rate():
    # base [1, -, -] and candidate [1, 0, -]: the splits that put every answered row on one side give the empty side
    # a 0 rate (review M2; a 0.5 fill gives 0.45 / 0.85)
    obs = side("s", "s.q", False, [1, None, None]) + side("s", "s.q", True, [1, 0, None])
    r = paired_test(obs, k=3, B=10_000, seed=0)
    assert r.exact is True and r.delta == -50.0
    assert r.p_worse == 0.5 and r.p_better == 0.8
    assert (r.rows, r.probes) == (2, 1)
    # the mirror pins an emptied base side the same way
    obs = side("s", "s.q", False, [1, 0, None]) + side("s", "s.q", True, [1, None, None])
    r = paired_test(obs, k=3, B=10_000, seed=0)
    assert r.delta == 50.0 and r.p_better == 0.5 and r.p_worse == 0.8
    assert (r.rows, r.probes) == (1, 1)


# --- sampled cases --------------------------------------------------------------------------------------------------

def test_sampled_p_follows_the_formula():
    # 6 ** 4 = 1,296 assignments > B: B seeded draws, the observed split joins them, p = (1 + #) / (B + 1)
    k, B, seed, key = 2, 300, 5, "unknown_honesty"
    masks = [tuple(m) for m in split_masks(k)]
    observed_split = masks.index(tuple([False] * k + [True] * k))
    rng = np.random.default_rng(42)
    obs = random_table(rng, ["d", "a", "c", "b"], k)
    strata = shared_strata(obs)
    assert len(strata) == 4
    draws = np.random.default_rng([seed, zlib.crc32(key.encode())]).integers(0, len(masks), size=(B, len(strata)))
    observed = ref_stats(obs, k, False, strata, (observed_split,) * 4, masks)
    pool = [observed] + [ref_stats(obs, k, False, strata, tuple(d), masks) for d in draws]
    p_better, p_worse = ref_min_p(pool, observed)
    r = paired_test(obs, k=k, B=B, seed=seed, key=key)
    assert r.exact is False
    assert abs(r.p_better - p_better) <= 1e-12 and abs(r.p_worse - p_worse) <= 1e-12
    for p in (r.p_better, r.p_worse):
        assert abs(p * (B + 1) - round(p * (B + 1))) <= 1e-9 and p >= 1 / (B + 1)
    assert abs(r.band - float(np.percentile([abs(a) for a, _ in pool[1:]], 99))) <= 1e-12


def test_a_sampled_clear_regression_hits_the_floor_one_sided():
    obs = []
    for i in range(12):
        obs += probe_rows(f"c{i:02d}", f"c{i:02d}.q", [1, 1, 1], [0, 0, 0])
    B = 999
    r = paired_test(obs, k=3, B=B, seed=3)
    assert r.exact is False and r.delta == -100.0
    assert r.p_worse == 1 / (B + 1)
    assert r.p_better == 1.0
    assert len(r.flips_worse) == 12 and r.flips_better == ()


def test_seed_determinism():
    rng = np.random.default_rng(9)
    obs = random_table(rng, [f"s{i:02d}" for i in range(40)], 3, probes_per=1)
    a = paired_test(obs, k=3, B=2_000, seed=11, key="ai_honesty")
    b = paired_test(list(reversed(obs)), k=3, B=2_000, seed=11, key="ai_honesty")
    assert a.exact is False
    assert a == b
    c = paired_test(obs, k=3, B=2_000, seed=12, key="ai_honesty")
    d = paired_test(obs, k=3, B=2_000, seed=11, key="about_me_recall")
    for other in (c, d):
        assert (other.delta, other.flips_better, other.flips_worse) == (a.delta, a.flips_better, a.flips_worse)
        assert (other.rows, other.probes) == (a.rows, a.probes)


# --- failed turns and empty sides -----------------------------------------------------------------------------------

def test_a_failed_side_makes_no_solid_flip_and_no_crash():
    r = paired_test(probe_rows("s", "s.q", [1, 1, 1], [0, None, 0]), k=3, B=10_000, seed=0)
    assert r.flips_worse == () and r.flips_better == ()
    assert r.delta == -100.0 and (r.rows, r.probes) == (2, 1)
    r = paired_test(probe_rows("s", "s.q", [1, 1, 1], [0, 0, 0]), k=3, B=10_000, seed=0)
    assert r.flips_worse == ("s.q",)
    # a failed base pass breaks the base's solidity the same way
    r = paired_test(probe_rows("s", "s.q", [1, None, 1], [0, 0, 0]), k=3, B=10_000, seed=0)
    assert r.flips_worse == ()
    empty = Paired(0.0, 0.0, 1.0, 1.0, (), (), 0, 0, True)
    assert paired_test(side("s", "s.q", False, [1, 1, 1]), k=3, B=10_000, seed=0) == empty
    assert paired_test(side("s", "s.q", True, [1, 1, 1]), k=3, B=10_000, seed=0) == empty
    assert paired_test([], k=3, B=10_000, seed=0) == empty


def test_count_mode():
    # invented: good is "not flagged"; the base is flagged 0/0/0 and the candidate 1/1/1 on 6 probes
    obs = []
    for i in range(6):
        obs += probe_rows(f"c{i}", f"c{i}.q", [1, 1, 1], [0, 0, 0])
    r = paired_test(obs, k=3, count=True, B=10_000, seed=0)
    assert r.delta == -18.0
    assert r.p_worse < 0.01 and r.p_better == 1.0
    assert len(r.flips_worse) == 6
    # the rate view of the same table
    assert paired_test(obs, k=3, B=10_000, seed=0).delta == -100.0


# --- validity under a heterogeneous null ----------------------------------------------------------------------------

def test_heterogeneous_null_is_valid():
    rates = np.random.default_rng(7).uniform(0.05, 0.95, 30)
    strata = [f"c{i:02d}" for i in range(30)]
    worse = better = 0
    start = time.perf_counter()
    for i in range(2_000):
        outcomes = (np.random.default_rng(i).random((30, 6)) < rates[:, None]).astype(int).tolist()
        obs = []
        for s, row in zip(strata, outcomes):
            obs += probe_rows(s, f"{s}.q", row[:3], row[3:])
        r = paired_test(obs, k=3, B=1_000, seed=i)
        worse += r.p_worse <= 0.01
        better += r.p_better <= 0.01
    elapsed = time.perf_counter() - start
    assert worse / 2_000 <= 0.015 and better / 2_000 <= 0.015
    assert elapsed <= 30, f"the null test took {elapsed:.1f}s"
