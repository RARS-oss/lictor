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
