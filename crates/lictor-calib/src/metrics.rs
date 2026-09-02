// SPDX-License-Identifier: MIT
//! Statistics for the offline lane: Clopper-Pearson, exact McNemar, the point-of-no-return proxy, ROC-AUC,
//! AUCPDT and the paired bootstrap (splitmix64, in-crate).
//!
//! Off the decision path: `ln`/`exp` and `f64::total_cmp` sorting are allowed here (docs/ARCHITECTURE.md sec 4,
//! float discipline scope). Nothing here allocates on a hot path or touches a clock or an OS RNG: the bootstrap
//! resampler is the in-crate splitmix64 generator seeded by the caller, so every number is reproducible from the
//! inputs alone (`harness/stats.py` re-derives them from the CSV counts with the same conventions).
//!
//! Conventions (documented because the frozen signatures do not pin them):
//! - `paired_bootstrap_diff(a, b, ..)`: `diff = mean(a) - mean(b)` (the caller passes `(arm, baseline)`, so the
//!   sign is the sign of `delta_vs_baseline`); the interval is the 2.5 / 97.5 percentile of the resampled
//!   differences with linear interpolation between order statistics (numpy's default), index draws are
//!   `splitmix64() % n` in a fixed order.
//! - `aucpdt(leads, horizon)`: `(1 / H) * sum_{L = 0}^{H - 1} |{lead >= L}| / N` -- the area under the
//!   detection-rate-vs-lead-time curve normalised by the horizon; an undetected failure is passed as
//!   `i64::MIN` so it counts in `N` and never in the numerator.
//! - `roc_auc(pos, neg)`: rank-based (Mann-Whitney) with ties counted 1/2; `NaN` when either class is empty.
//! - `mcnemar_exact(b, c)`: two-sided exact binomial on the discordant pairs, `p = min(1, 2 P(X <= min(b, c)))`
//!   with `X ~ Bin(b + c, 1/2)`; `p = 1` when `b + c = 0`.
//! - `ponr_tick(coverage, eps)`: `t_fail = min{t : c*_T - c*_t < eps}` with `c*_t = max_{u <= t} coverage_u`
//!   (non-finite samples do not move the running maximum); an episode with no progress has `t_fail = 0`.

/// Exact two-sided Clopper-Pearson interval for `k` successes in `n` trials at confidence `conf` (e.g. 0.95).
/// `(0, 1)` when `n == 0`. Beta quantiles come from an in-crate regularised incomplete beta with bisection.
pub fn clopper_pearson(k: u32, n: u32, conf: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let k = k.min(n);
    let alpha = (1.0 - conf).clamp(0.0, 1.0);
    let half = alpha / 2.0;
    let lo = if k == 0 { 0.0 } else { beta_quantile(half, f64::from(k), f64::from(n - k + 1)) };
    let hi = if k == n { 1.0 } else { beta_quantile(1.0 - half, f64::from(k + 1), f64::from(n - k)) };
    (lo, hi)
}

/// Exact two-sided McNemar p-value on the discordant counts `b`, `c`.
pub fn mcnemar_exact(b: u32, c: u32) -> f64 {
    let n = b + c;
    if n == 0 {
        return 1.0;
    }
    let m = b.min(c);
    // Binomial(n, 1/2) pmf by the ratio recurrence; n is at most a few thousand episodes so 0.5^n stays normal.
    let mut pmf = 0.5f64.powi(n as i32);
    let mut cdf = 0.0;
    for i in 0..=m {
        cdf += pmf;
        pmf *= f64::from(n - i) / f64::from(i + 1);
    }
    (2.0 * cdf).min(1.0)
}

/// t_fail = min{t : c*_T - c*_t < eps}
pub fn ponr_tick(coverage: &[f64], eps_prog: f64) -> u32 {
    let mut running = Vec::with_capacity(coverage.len());
    let mut best = f64::NEG_INFINITY;
    for &c in coverage {
        if c.is_finite() && c > best {
            best = c;
        }
        running.push(best);
    }
    if running.is_empty() || !best.is_finite() {
        return 0;
    }
    for (t, &r) in running.iter().enumerate() {
        if best - r < eps_prog {
            return t as u32;
        }
    }
    (running.len() - 1) as u32
}

/// Rank-based ROC-AUC (Mann-Whitney) of `pos` scores against `neg` scores; ties count 1/2. NaN when a class is empty.
pub fn roc_auc(pos: &[f64], neg: &[f64]) -> f64 {
    if pos.is_empty() || neg.is_empty() {
        return f64::NAN;
    }
    let mut neg_sorted: Vec<f64> = neg.to_vec();
    neg_sorted.sort_by(f64::total_cmp);
    let mut acc = 0.0f64;
    for &p in pos {
        // number of negatives strictly below p, and the number equal to p
        let below = neg_sorted.partition_point(|&x| x.total_cmp(&p) == std::cmp::Ordering::Less);
        let not_above = neg_sorted.partition_point(|&x| x.total_cmp(&p) != std::cmp::Ordering::Greater);
        acc += below as f64 + 0.5 * (not_above - below) as f64;
    }
    acc / (pos.len() as f64 * neg.len() as f64)
}

/// Area under the detection-rate-vs-lead-time curve normalised by the horizon (VLA-FAIL):
/// `(1/H) sum_{L=0}^{H-1} |{lead >= L}| / N`. Undetected failures are passed as `i64::MIN`.
pub fn aucpdt(leads: &[i64], horizon: u32) -> f64 {
    if leads.is_empty() || horizon == 0 {
        return 0.0;
    }
    let mut sorted: Vec<i64> = leads.to_vec();
    sorted.sort_unstable();
    let n = sorted.len() as f64;
    let h = horizon as usize;
    let mut total = 0.0f64;
    for l in 0..h {
        let l = l as i64;
        let below = sorted.partition_point(|&x| x < l);
        total += (sorted.len() - below) as f64 / n;
    }
    total / horizon as f64
}

/// splitmix64 (Steele, Lea, Flood 2014): the only generator in the crate; seeded by the caller.
#[derive(Clone, Copy, Debug)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// `(diff, lo, hi)` with `diff = mean(a) - mean(b)` over the paired samples and a percentile bootstrap CI
/// (2.5 / 97.5, linear interpolation) from `n_resamples` index draws of the in-crate splitmix64 seeded with `seed`.
/// The pairs are `(a[i], b[i])`; a length mismatch uses the common prefix. Empty input -> `(0, 0, 0)`.
pub fn paired_bootstrap_diff(a: &[bool], b: &[bool], n_resamples: u32, seed: u64) -> (f64, f64, f64) {
    let n = a.len().min(b.len());
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    // Per-pair contribution: +1 (a only), -1 (b only), 0 otherwise.
    let d: Vec<i32> = (0..n).map(|i| i32::from(a[i]) - i32::from(b[i])).collect();
    let diff = d.iter().map(|&x| f64::from(x)).sum::<f64>() / n as f64;
    if n_resamples == 0 {
        return (diff, diff, diff);
    }
    let mut rng = SplitMix64::new(seed);
    let mut samples = Vec::with_capacity(n_resamples as usize);
    for _ in 0..n_resamples {
        let mut acc: i64 = 0;
        for _ in 0..n {
            let idx = (rng.next_u64() % n as u64) as usize;
            acc += i64::from(d[idx]);
        }
        samples.push(acc as f64 / n as f64);
    }
    samples.sort_by(f64::total_cmp);
    (diff, percentile_linear(&samples, 0.025), percentile_linear(&samples, 0.975))
}

/// numpy-style linear-interpolation percentile of an ascending slice (`p` in [0, 1]).
pub fn percentile_linear(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let pos = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let w = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * w
}

/// Nearest-rank quantile of an ascending slice: element `ceil(q * n)` (1-based), clamped to the slice.
pub fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let rank = (q * n as f64).ceil() as usize;
    sorted[rank.clamp(1, n) - 1]
}

// ---- regularised incomplete beta ---------------------------------------------------------------------------

/// Lanczos ln-gamma (g = 7, 9 coefficients); relative error ~1e-15 for x > 0.
fn ln_gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const COEF: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection
        let s = (std::f64::consts::PI * x).sin();
        return (std::f64::consts::PI / s).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = COEF[0];
    let t = x + G + 0.5;
    for (i, c) in COEF.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Continued fraction for the incomplete beta (modified Lentz).
fn betacf(a: f64, b: f64, x: f64) -> f64 {
    const MAX_IT: usize = 300;
    const EPS: f64 = 3e-16;
    const FPMIN: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_IT {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularised incomplete beta I_x(a, b) for a, b > 0 and x in [0, 1].
pub fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * betacf(a, b, x) / a
    } else {
        1.0 - bt * betacf(b, a, 1.0 - x) / b
    }
}

/// Quantile of Beta(a, b) at probability `p` by bisection on the regularised incomplete beta.
pub fn beta_quantile(p: f64, a: f64, b: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if reg_inc_beta(a, b, mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-15 {
            break;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_gamma_matches_factorials() {
        for n in 1..15u32 {
            let fact: f64 = (1..n).map(f64::from).product();
            assert!((ln_gamma(f64::from(n)) - fact.ln()).abs() < 1e-10, "n = {n}");
        }
    }

    #[test]
    fn inc_beta_symmetry_and_uniform() {
        // I_x(1, 1) = x
        for i in 0..=10 {
            let x = i as f64 / 10.0;
            assert!((reg_inc_beta(1.0, 1.0, x) - x).abs() < 1e-12);
        }
        // I_x(a, b) = 1 - I_{1-x}(b, a)
        let v = reg_inc_beta(3.0, 7.0, 0.3);
        let w = 1.0 - reg_inc_beta(7.0, 3.0, 0.7);
        assert!((v - w).abs() < 1e-12);
    }

    #[test]
    fn splitmix_reference_vector() {
        // splitmix64 with seed 0: the first output is 0xe220a8397b1dcdaf (the ARCHITECTURE hash64(0, 0) vector).
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
    }

    #[test]
    fn percentile_endpoints() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile_linear(&v, 0.0), 1.0);
        assert_eq!(percentile_linear(&v, 1.0), 4.0);
        assert_eq!(percentile_linear(&v, 0.5), 2.5);
        assert_eq!(nearest_rank(&v, 0.5), 2.0);
        assert_eq!(nearest_rank(&v, 0.999), 4.0);
        assert_eq!(nearest_rank(&v, 0.0), 1.0);
    }
}
