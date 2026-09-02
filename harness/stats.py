# SPDX-License-Identifier: MIT
"""Statistics cross-checks for the lictor experiment (WP-9).

Every function here is a CROSS-CHECK of a number that `lictor curve`, `lictor sweep`
or `lictor calibrate` (crates/lictor-calib/src/metrics.rs) computed from signed
receipts. The Rust side is the source; this module recomputes the same quantities
from the CSV / JSONL numbers so that `harness/analyze.py` can print
`stats cross-check: ok` (or the diffs). Nothing here is ever the origin of a
headline number.

scipy-free by design: the regularised incomplete beta is a Lentz continued fraction
on `math.lgamma`, its quantile is a bisection, McNemar is an exact binomial on
`math.comb`. numpy is used only by the paired bootstrap.

Conventions (mirroring ARCHITECTURE 10.7 and the frozen `lictor_calib::metrics`):

- `clopper_pearson(k, n, conf)` -> `(lo, hi)`, the exact two-sided interval.
- `mcnemar_exact(b, c)` -> two-sided exact p on the discordant pairs (b, c).
- `ponr_tick(coverage, eps_prog)` -> `t_fail = min{t : c*_T - c*_t < eps_prog}` with
  `c*_t = max_{u <= t} coverage_u`; an episode with no progress at all has
  `t_fail = 0` (the artefact disclosed on figure F3).
- `roc_auc(pos, neg)` -> rank-based AUC with ties averaged (Mann-Whitney).
- `aucpdt(leads, horizon)` -> area under the detection-rate-vs-lead-time curve,
  `(1/H) * sum_{L=0}^{H-1} |{lead >= L}| / N` (VLA-FAIL); `None` = never detected.
- `paired_bootstrap_diff(baseline, arm, n_resamples, seed)` -> `(diff, lo, hi)` with
  `diff = mean(arm) - mean(baseline)` (the sign of `delta_vs_baseline`), percentile
  bootstrap over pairs, `numpy.random.default_rng(20260830)` and 10 000 resamples
  by default. The Rust implementation draws its resamples from splitmix64, so the
  two CIs agree only up to Monte-Carlo noise; `analyze.py` uses BOOT_TOL for them.
- `f1_timeliness_hypervolume(points)` -> the area of the (F1, timeliness) plane
  dominated by a set of operating points (ActProbe's F1-timeliness hypervolume,
  reference point (0, 0)); `timeliness(leads, horizon)` = mean of
  `clip(lead / horizon, 0, 1)` over detected failures.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

BOOT_SEED = 20260830
BOOT_RESAMPLES = 10_000
# Two bootstrap CIs drawn from different RNG streams (numpy here, splitmix64 in Rust)
# agree only up to Monte-Carlo noise; 10 000 resamples on n >= 60 pairs keeps the
# 2.5 % / 97.5 % quantiles within about 0.01, so 0.02 is a generous cross-check band.
BOOT_TOL = 0.02
EXACT_TOL = 1e-9

# --------------------------------------------------------------------------- beta


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta (modified Lentz)."""
    max_it = 500
    eps = 3e-16
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b) for a, b > 0 and 0 <= x <= 1."""
    if a <= 0.0 or b <= 0.0:
        raise ValueError("betainc: a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_quantile(p: float, a: float, b: float) -> float:
    """x such that I_x(a, b) = p, by bisection (monotone, 80 halvings)."""
    if not 0.0 <= p <= 1.0:
        raise ValueError("beta_quantile: p outside [0, 1]")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Exact two-sided binomial interval for k successes in n trials."""
    if n < 0 or k < 0 or k > n:
        raise ValueError("clopper_pearson: need 0 <= k <= n")
    if n == 0:
        return (0.0, 1.0)
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else beta_quantile(alpha / 2.0, k, n - k + 1)
    hi = 1.0 if k == n else beta_quantile(1.0 - alpha / 2.0, k + 1, n - k)
    return (lo, hi)


# ------------------------------------------------------------------------ McNemar


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant counts (b, c)."""
    if b < 0 or c < 0:
        raise ValueError("mcnemar_exact: counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    p = 2.0 * tail / float(2**n)
    return min(1.0, p)


# --------------------------------------------------------------------------- PoNR


def running_max(xs: Sequence[float]) -> list[float]:
    out: list[float] = []
    m = -math.inf
    for x in xs:
        if x > m:
            m = x
        out.append(m)
    return out


def ponr_tick(coverage: Sequence[float], eps_prog: float) -> int:
    """Point-of-no-return PROXY: t_fail = min{t : c*_T - c*_t < eps_prog}.

    c*_t is the running maximum of coverage. An empty trace or a trace with no
    progress at all yields 0 (every detector then shows a negative lead on it).
    """
    if len(coverage) == 0:
        return 0
    cstar = running_max(coverage)
    c_final = cstar[-1]
    for t, c in enumerate(cstar):
        if c_final - c < eps_prog:
            return t
    return len(cstar) - 1


def lead_ticks(t_fail: int, first_stop_tick: int | None) -> int | None:
    """lead = t_fail - first_stop_tick; None when the arm never stopped."""
    if first_stop_tick is None:
        return None
    return int(t_fail) - int(first_stop_tick)


# ---------------------------------------------------------------------------- ROC


def roc_auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    """Rank-based AUC (Mann-Whitney) with ties averaged; 0.5 when a class is empty."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    scored = [(float(v), 1) for v in pos] + [(float(v), 0) for v in neg]
    scored.sort(key=lambda t: t[0])
    ranks = [0.0] * len(scored)
    i = 0
    while i < len(scored):
        j = i
        while j + 1 < len(scored) and scored[j + 1][0] == scored[i][0]:
            j += 1
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_pos = sum(r for r, (_, lab) in zip(ranks, scored) if lab == 1)
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    return (rank_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def bacc(tpr: float, fpr: float) -> float:
    """Balanced accuracy at a fixed operating point."""
    return 0.5 * (tpr + (1.0 - fpr))


# ------------------------------------------------------------------------- AUCPDT


def aucpdt(leads: Iterable[int | None], horizon: int) -> float:
    """Area under detection-rate-vs-lead-time, normalised by the horizon.

    D(L) = |{lead >= L}| / N for L = 0 .. horizon-1; `None` entries never count as
    detected. Returns 0.0 for an empty list or a zero horizon.
    """
    ls = list(leads)
    if horizon <= 0 or len(ls) == 0:
        return 0.0
    n = float(len(ls))
    detected = [int(l) for l in ls if l is not None]
    total = 0.0
    for big_l in range(horizon):
        total += sum(1 for l in detected if l >= big_l) / n
    return total / float(horizon)


# ----------------------------------------------------------- F1-timeliness (ActProbe)


def f1_score(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2.0 * tp / float(denom)


def timeliness(leads: Iterable[int | None], horizon: int) -> float:
    """Mean of clip(lead / horizon, 0, 1) over detected failures (0 when none)."""
    det = [int(l) for l in leads if l is not None]
    if horizon <= 0 or len(det) == 0:
        return 0.0
    return sum(min(1.0, max(0.0, l / float(horizon))) for l in det) / float(len(det))


def hypervolume_2d(points: Iterable[tuple[float, float]]) -> float:
    """Area of the union of rectangles [0, x] x [0, y] over the points (ref (0, 0))."""
    pts = sorted(((float(x), float(y)) for x, y in points), key=lambda p: (-p[0], -p[1]))
    area = 0.0
    y_max = 0.0
    for x, y in pts:
        if y > y_max:
            area += x * (y - y_max)
            y_max = y
    return area


def f1_timeliness_hypervolume(points: Iterable[tuple[float, float]]) -> float:
    """ActProbe-style hypervolume of (F1, timeliness) operating points."""
    return hypervolume_2d(points)


# ------------------------------------------------------------------------ quantiles


def quantile_nearest_rank(xs: Sequence[float], q: float) -> float:
    """Nearest-rank quantile on a sorted copy (q in [0, 1]); NaN for empty input."""
    if len(xs) == 0:
        return math.nan
    s = sorted(float(x) for x in xs)
    if q <= 0.0:
        return s[0]
    k = int(math.ceil(q * len(s)))
    return s[min(len(s), max(1, k)) - 1]


def lead_summary(leads: Iterable[int | None]) -> dict[str, float]:
    """mean / p50 / p10 over detected leads (NaN when nothing was detected)."""
    det = [float(l) for l in leads if l is not None]
    if len(det) == 0:
        return {"lead_mean": math.nan, "lead_p50": math.nan, "lead_p10": math.nan}
    return {
        "lead_mean": sum(det) / len(det),
        "lead_p50": quantile_nearest_rank(det, 0.5),
        "lead_p10": quantile_nearest_rank(det, 0.1),
    }


# ------------------------------------------------------------------------ bootstrap


def pairs_from_contingency(
    n: int, both: int, baseline_only: int, arm_only: int
) -> tuple[list[bool], list[bool]]:
    """Canonical paired arrays from a 2x2 table (order: both, baseline-only, arm-only, neither).

    The bootstrap distribution of a paired difference depends only on the table,
    so this canonical ordering makes `paired_bootstrap_diff` reproducible from the
    counts alone (used by the cross-check and by the figure fixture).
    """
    neither = n - both - baseline_only - arm_only
    if min(n, both, baseline_only, arm_only, neither) < 0:
        raise ValueError("pairs_from_contingency: inconsistent table")
    baseline = [True] * both + [True] * baseline_only + [False] * arm_only + [False] * neither
    arm = [True] * both + [False] * baseline_only + [True] * arm_only + [False] * neither
    return baseline, arm


def paired_bootstrap_diff(
    baseline: Sequence[bool],
    arm: Sequence[bool],
    n_resamples: int = BOOT_RESAMPLES,
    seed: int = BOOT_SEED,
) -> tuple[float, float, float]:
    """(diff, lo, hi): diff = mean(arm) - mean(baseline); 95 % percentile bootstrap over pairs."""
    import numpy as np

    if len(baseline) != len(arm):
        raise ValueError("paired_bootstrap_diff: arrays must pair up")
    n = len(baseline)
    if n == 0:
        return (math.nan, math.nan, math.nan)
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(arm, dtype=np.float64)
    d = b - a
    diff = float(d.mean())
    if n_resamples <= 0:
        return (diff, diff, diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_resamples), n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return (diff, float(lo), float(hi))


def orient_contingency(
    n: int, n_succ_arm: int, n_succ_baseline: int, b: int, c: int
) -> tuple[int, int, int] | None:
    """Recover (both, baseline_only, arm_only) from the CSV counts, or None if inconsistent.

    `lictor curve` writes the discordant pair as (mcnemar_b, mcnemar_c) without saying
    which of the two is "baseline succeeded, arm failed"; the success totals decide.
    """
    delta_count = n_succ_arm - n_succ_baseline
    if delta_count == c - b:
        baseline_only, arm_only = b, c
    elif delta_count == b - c:
        baseline_only, arm_only = c, b
    else:
        return None
    both = n_succ_baseline - baseline_only
    neither = n - both - baseline_only - arm_only
    if both < 0 or neither < 0:
        return None
    return (both, baseline_only, arm_only)


def clopper_pearson_halfwidth(n: int, conf: float = 0.95) -> float:
    """Half-width of the CP interval at p = 0.5 (the 'about +-X pp' sentence in the docs)."""
    if n <= 0:
        return 1.0
    lo, hi = clopper_pearson(n // 2, n, conf)
    return 0.5 * (hi - lo)
