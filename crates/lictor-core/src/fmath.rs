// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; the bodies below are the frozen text (they are the specification),
//! keep the signatures.
//!
//! The only float helpers allowed on the decision path. Identical results under std and no_std.
//! `libm` is an UNCONDITIONAL dependency of lictor-core (there is no `libm` feature): under `std` the hardware sqrt is
//! used, under no_std `libm::sqrt`; both are correctly rounded, hence bit-identical.

#[inline]
pub fn sqrt(x: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        x.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrt(x)
    }
}

#[inline]
pub fn abs(x: f64) -> f64 {
    f64::from_bits(x.to_bits() & !(1u64 << 63))
}

/// `if a < b { a } else { b }` -- deliberately NOT f64::min (NaN never reaches here; fail-closed earlier).
#[inline]
pub fn min(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline]
pub fn max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

#[inline]
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    max(lo, min(x, hi))
}

/// Euclidean norm with fixed left-to-right accumulation over `v[..n]`.
#[inline]
pub fn norm(v: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    let mut i = 0;
    while i < n {
        s += v[i] * v[i];
        i += 1;
    }
    sqrt(s)
}

/// ||a[..n] - b[..n]|| with fixed order.
#[inline]
pub fn dist(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut s = 0.0;
    let mut i = 0;
    while i < n {
        let d = a[i] - b[i];
        s += d * d;
        i += 1;
    }
    sqrt(s)
}

#[inline]
pub fn all_finite(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abs_clears_only_the_sign_bit() {
        assert_eq!(abs(-3.5), 3.5);
        assert_eq!(abs(3.5), 3.5);
        assert_eq!(abs(-0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(abs(f64::NEG_INFINITY), f64::INFINITY);
    }

    #[test]
    fn min_max_clamp_are_strict_comparisons() {
        assert_eq!(min(1.0, 2.0), 1.0);
        assert_eq!(min(2.0, 1.0), 1.0);
        assert_eq!(max(1.0, 2.0), 2.0);
        assert_eq!(max(2.0, 1.0), 2.0);
        // ties return the second argument (a < b is false), so signed zeros are deterministic
        assert_eq!(min(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(clamp(5.0, 0.0, 1.0), 1.0);
        assert_eq!(clamp(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(clamp(0.5, 0.0, 1.0), 0.5);
        assert_eq!(min(f64::INFINITY, 3.0), 3.0);
    }

    #[test]
    fn norm_and_dist_accumulate_left_to_right() {
        assert_eq!(sqrt(4.0), 2.0);
        assert_eq!(norm(&[3.0, 4.0], 2), 5.0);
        assert_eq!(norm(&[3.0, 4.0, 100.0], 2), 5.0, "only the first n entries count");
        assert_eq!(norm(&[], 0), 0.0);
        assert_eq!(dist(&[1.0, 1.0], &[4.0, 5.0], 2), 5.0);
        // fixed order: (1e16^2 + 1) + (-1e16)^2, not a reordered sum
        let v = [1e16, 1.0, -1e16];
        let expected = sqrt(((1e16 * 1e16) + 1.0) + (1e16 * 1e16));
        assert_eq!(norm(&v, 3).to_bits(), expected.to_bits());
    }

    #[test]
    fn all_finite_rejects_nan_and_inf() {
        assert!(all_finite(&[0.0, -1.5, 1e300]));
        assert!(all_finite(&[]));
        assert!(!all_finite(&[0.0, f64::NAN]));
        assert!(!all_finite(&[f64::INFINITY]));
        assert!(!all_finite(&[f64::NEG_INFINITY, 1.0]));
    }
}
