// SPDX-License-Identifier: MIT
//! The deterministic runtime half of split-conformal: standardise, aggregate over the gate, strict trip.
//!
//! Everything here is a handful of `+ - /` and comparisons over the fixed `Scores` arrays plus integer bit tests;
//! no allocation, no clock, no panic path on data. The offline half (`lictor calibrate`, which produces the
//! `CalibrationC` table) lives in `lictor-calib`; the two share `lictor_core::bin_of` so the time bin a runtime
//! tick lands in is exactly the bin the calibrator assigned it.

use lictor_core::{fmath, CalibrationC, Scores, MAX_TERMS, NFEAT, T_GRID};

/// z_j = (f_j - center[b][j]) / scale[b][j] for masked & valid j; b = cal.bin(t). Others 0.0.
pub fn standardise(cal: &CalibrationC, t: u32, sc: &mut Scores) {
    let mut b = cal.bin(t);
    // `bin_of` returns < t_grid; a table with a malformed t_grid must still not index out of range.
    if b >= T_GRID {
        b = T_GRID - 1;
    }
    let live = cal.mask & sc.valid;
    let mut j = 0;
    while j < NFEAT {
        if live & (1u32 << j) != 0 {
            sc.z[j] = (sc.f[j] - cal.center[b][j]) / cal.scale[b][j];
        } else {
            sc.z[j] = 0.0;
        }
        j += 1;
    }
}

/// s = max over terms of min over the term's channels (all masked & valid), NEG_INFINITY if no term is fully valid.
/// Also sets sc.fired (z_j > tau per channel, informational). Returns s.
///
/// Fixed-order DNF: terms are visited in `0..n_terms` (capped at `MAX_TERMS`), channels inside a term in ascending
/// index order. A term with no channels is skipped (an empty AND would be vacuously +inf). The maximum and the
/// minimum go through `fmath` so the result is bit-identical under std and no_std.
pub fn aggregate(cal: &CalibrationC, sc: &mut Scores) -> f64 {
    let live = cal.mask & sc.valid;
    let n_terms = {
        let n = cal.gate.n_terms as usize;
        if n > MAX_TERMS {
            MAX_TERMS
        } else {
            n
        }
    };
    let mut s = f64::NEG_INFINITY;
    let mut term = 0;
    while term < n_terms {
        let tm = cal.gate.terms[term];
        if tm != 0 && (tm & live) == tm {
            let mut v = f64::INFINITY;
            let mut j = 0;
            while j < NFEAT {
                if tm & (1u32 << j) != 0 {
                    v = fmath::min(v, sc.z[j]);
                }
                j += 1;
            }
            s = fmath::max(s, v);
        }
        term += 1;
    }
    let mut fired = 0u32;
    let mut j = 0;
    while j < NFEAT {
        let bit = 1u32 << j;
        if live & bit != 0 && sc.z[j] > cal.tau {
            fired |= bit;
        }
        j += 1;
    }
    sc.fired = fired;
    sc.s = s;
    s
}

/// STRICT: s > tau.
#[inline]
pub fn trip(cal: &CalibrationC, sc: &Scores) -> bool {
    sc.s > cal.tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use lictor_core::GateSpec;

    fn cal_with(mask: u32, terms: &[u32], tau: f64) -> CalibrationC {
        let mut c = CalibrationC::DISARMED;
        c.mask = mask;
        let mut g = GateSpec::DISARMED;
        for (i, t) in terms.iter().enumerate() {
            g.terms[i] = *t;
        }
        g.n_terms = terms.len() as u8;
        c.gate = g;
        c.tau = tau;
        c
    }

    #[test]
    fn disarmed_never_trips() {
        let cal = CalibrationC::DISARMED;
        let mut sc = Scores { valid: 0xfff, ..Default::default() };
        standardise(&cal, 0, &mut sc);
        let s = aggregate(&cal, &mut sc);
        assert!(s == f64::NEG_INFINITY);
        assert!(!trip(&cal, &sc));
        assert_eq!(sc.fired, 0);
    }

    #[test]
    fn singleton_terms_are_a_plain_max() {
        let cal = cal_with(0b111, &[0b001, 0b010, 0b100], 1.0);
        let mut sc = Scores { valid: 0b111, ..Default::default() };
        sc.f = [0.5, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        standardise(&cal, 5, &mut sc);
        let s = aggregate(&cal, &mut sc);
        assert_eq!(s, 3.0);
        assert_eq!(sc.fired, 0b100);
        assert!(trip(&cal, &sc));
    }

    #[test]
    fn and_pair_is_a_min_and_boundary_does_not_trip() {
        let cal = cal_with(0b11, &[0b11], 2.0);
        let mut sc = Scores { valid: 0b11, ..Default::default() };
        sc.f = [2.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        standardise(&cal, 0, &mut sc);
        let s = aggregate(&cal, &mut sc);
        assert_eq!(s, 2.0);
        assert!(!trip(&cal, &sc), "s == tau must not trip");
        assert_eq!(sc.fired, 0b10);
        // One invalid channel disqualifies the whole term.
        sc.valid = 0b01;
        standardise(&cal, 0, &mut sc);
        assert_eq!(aggregate(&cal, &mut sc), f64::NEG_INFINITY);
    }
}
