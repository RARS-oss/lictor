// SPDX-License-Identifier: MIT
//! Conformal runtime rule tests: `fixtures/conformal/cases.json` (written by `fixtures/tier1/gen.py`, an
//! independent Python reimplementation of standardise / aggregate / strict trip / `bin_of`) plus the hand cases
//! the plan lists explicitly.

use lictor_core::{bin_of, CalibrationC, GateSpec, Scores, MAX_TERMS, NFEAT, T_GRID};
use lictor_detect::conformal::{aggregate, standardise, trip};
use serde_json::Value;

const CASES: &str = include_str!("fixtures/conformal/cases.json");
const TOL: f64 = 1e-12;

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn u32_of(v: &Value) -> u32 {
    u32::try_from(v.as_u64().unwrap()).unwrap()
}

fn cal_with(mask: u32, terms: &[u32], tau: f64) -> CalibrationC {
    let mut c = CalibrationC::DISARMED;
    c.mask = mask;
    let mut g = GateSpec::DISARMED;
    assert!(terms.len() <= MAX_TERMS);
    for (i, t) in terms.iter().enumerate() {
        g.terms[i] = *t;
    }
    g.n_terms = u8::try_from(terms.len()).unwrap();
    c.gate = g;
    c.tau = tau;
    c
}

fn cal_from(case: &Value) -> CalibrationC {
    let terms: Vec<u32> = case["terms"].as_array().unwrap().iter().map(u32_of).collect();
    let tau = if case["tau"].is_null() { f64::INFINITY } else { case["tau"].as_f64().unwrap() };
    let mut c = cal_with(u32_of(&case["mask"]), &terms, tau);
    c.horizon_ticks = u32_of(&case["horizon_ticks"]);
    c.t_grid = u16::try_from(case["t_grid"].as_u64().unwrap()).unwrap();
    if let Some(bins) = case["bins"].as_object() {
        for (b, row) in bins {
            let b: usize = b.parse().unwrap();
            assert!(b < T_GRID);
            for (j, x) in f64s(&row["center"]).iter().enumerate() {
                c.center[b][j] = *x;
            }
            for (j, x) in f64s(&row["scale"]).iter().enumerate() {
                c.scale[b][j] = *x;
            }
        }
    }
    c
}

fn close(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= TOL
}

#[test]
fn fixture_cases() {
    let root: Value = serde_json::from_str(CASES).unwrap();
    let cases = root["cases"].as_array().unwrap();
    assert!(cases.len() >= 12, "fixture has {} cases", cases.len());
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let cal = cal_from(case);
        let mut sc = Scores::default();
        let f = f64s(&case["f"]);
        sc.f.copy_from_slice(&f);
        sc.valid = u32_of(&case["valid"]);
        let t = u32_of(&case["t"]);
        standardise(&cal, t, &mut sc);
        let s = aggregate(&cal, &mut sc);
        let exp = &case["expect"];
        let z = f64s(&exp["z"]);
        for (j, zj) in z.iter().enumerate() {
            assert!(close(sc.z[j], *zj), "{name}: z[{j}] = {} expected {zj}", sc.z[j]);
        }
        let s_exp = if exp["s"].is_null() { f64::NEG_INFINITY } else { exp["s"].as_f64().unwrap() };
        assert!(close(s, s_exp), "{name}: s = {s} expected {s_exp}");
        assert_eq!(sc.s, s, "{name}: aggregate must store s");
        assert_eq!(sc.fired, u32_of(&exp["fired"]), "{name}: fired");
        assert_eq!(trip(&cal, &sc), exp["trip"].as_bool().unwrap(), "{name}: trip");
        // Invariants independent of the fixture: unmasked or invalid channels are exactly 0 and never fire.
        let dead = !(cal.mask & sc.valid);
        for j in 0..NFEAT {
            if dead & (1 << j) != 0 {
                assert_eq!(sc.z[j], 0.0, "{name}: z[{j}] must be 0.0 when unmasked/invalid");
                assert_eq!(sc.fired & (1 << j), 0, "{name}: fired[{j}] must be clear when unmasked/invalid");
            }
        }
    }
}

#[test]
fn fixture_bin_of_table() {
    let root: Value = serde_json::from_str(CASES).unwrap();
    let rows = root["bin_of"].as_array().unwrap();
    assert!(rows.len() >= 8);
    for row in rows {
        let t = u32_of(&row["t"]);
        let h = u32_of(&row["horizon_ticks"]);
        let g = u16::try_from(row["t_grid"].as_u64().unwrap()).unwrap();
        let exp = usize::try_from(row["expect"].as_u64().unwrap()).unwrap();
        assert_eq!(bin_of(t, h, g), exp, "bin_of({t}, {h}, {g})");
        let mut c = CalibrationC::DISARMED;
        c.horizon_ticks = h;
        c.t_grid = g;
        assert_eq!(c.bin(t), exp, "CalibrationC::bin({t}) with ({h}, {g})");
    }
}

#[test]
fn empty_mask_is_neg_infinity_and_never_trips() {
    let cal = CalibrationC::DISARMED;
    let mut sc = Scores { valid: 0xfff, ..Default::default() };
    sc.f = [5.0; NFEAT];
    standardise(&cal, 17, &mut sc);
    assert_eq!(sc.z, [0.0; NFEAT]);
    let s = aggregate(&cal, &mut sc);
    assert_eq!(s, f64::NEG_INFINITY);
    assert_eq!(sc.s, f64::NEG_INFINITY);
    assert_eq!(sc.fired, 0);
    assert!(!trip(&cal, &sc));
    assert!(!cal.armed());
}

#[test]
fn all_invalid_is_neg_infinity() {
    let cal = cal_with(0xff, &[1, 2, 4, 8, 16, 32, 64, 128], 0.0);
    let mut sc = Scores { valid: 0, ..Default::default() };
    sc.f = [9.0; NFEAT];
    standardise(&cal, 0, &mut sc);
    assert_eq!(sc.z, [0.0; NFEAT]);
    assert_eq!(aggregate(&cal, &mut sc), f64::NEG_INFINITY);
    assert_eq!(sc.fired, 0);
    assert!(!trip(&cal, &sc));
}

#[test]
fn singleton_gate_is_a_plain_max() {
    let cal = cal_with(0xff, &[1, 2, 4, 8, 16, 32, 64, 128], 1.5);
    let mut sc = Scores { valid: 0xff, ..Default::default() };
    sc.f = [0.5, -2.0, 3.0, 0.25, 1.0, 0.75, -0.5, 2.5, 100.0, 100.0, 100.0, 100.0];
    standardise(&cal, 0, &mut sc);
    let s = aggregate(&cal, &mut sc);
    assert_eq!(s, 3.0);
    assert_eq!(sc.fired, (1 << 2) | (1 << 7));
    assert!(trip(&cal, &sc));
    // The ext channels are valid-but-unmasked: zero, silent.
    assert_eq!(sc.z[8], 0.0);
    assert_eq!(sc.fired >> 8, 0);
}

#[test]
fn and_pair_gate_is_a_min() {
    let cal = cal_with(0b11, &[0b11], 0.0);
    let mut sc = Scores { valid: 0b11, ..Default::default() };
    sc.f[0] = 4.0;
    sc.f[1] = -1.0;
    standardise(&cal, 0, &mut sc);
    assert_eq!(aggregate(&cal, &mut sc), -1.0);
    assert!(!trip(&cal, &sc));
    assert_eq!(sc.fired, 0b01);
    sc.f[1] = 2.0;
    standardise(&cal, 0, &mut sc);
    assert_eq!(aggregate(&cal, &mut sc), 2.0);
    assert!(trip(&cal, &sc));
    assert_eq!(sc.fired, 0b11);
    // One invalid channel disqualifies the whole term.
    sc.valid = 0b01;
    standardise(&cal, 0, &mut sc);
    assert_eq!(aggregate(&cal, &mut sc), f64::NEG_INFINITY);
    assert!(!trip(&cal, &sc));
}

#[test]
fn boundary_s_equal_tau_does_not_trip() {
    let tau = 2.0;
    let cal = cal_with(0b1, &[0b1], tau);
    let mut sc = Scores { valid: 0b1, ..Default::default() };
    sc.f[0] = tau;
    standardise(&cal, 0, &mut sc);
    let s = aggregate(&cal, &mut sc);
    assert_eq!(s, tau);
    assert!(!trip(&cal, &sc), "s == tau must NOT trip (strict)");
    assert_eq!(sc.fired, 0);
    sc.f[0] = f64::from_bits(tau.to_bits() + 1);
    standardise(&cal, 0, &mut sc);
    aggregate(&cal, &mut sc);
    assert!(trip(&cal, &sc), "one ulp above tau trips");
    assert_eq!(sc.fired, 0b1);
}

#[test]
fn standardise_uses_the_bin_of_t() {
    let mut cal = cal_with(0b1, &[0b1], 0.0);
    cal.horizon_ticks = 300;
    cal.t_grid = 100;
    cal.center[0][0] = 1.0;
    cal.scale[0][0] = 2.0;
    cal.center[99][0] = -1.0;
    cal.scale[99][0] = 0.5;
    let mut sc = Scores { valid: 0b1, ..Default::default() };
    sc.f[0] = 3.0;
    standardise(&cal, 0, &mut sc);
    assert_eq!(sc.z[0], 1.0);
    standardise(&cal, 299, &mut sc);
    assert_eq!(sc.z[0], 8.0);
    standardise(&cal, 1_000_000, &mut sc);
    assert_eq!(sc.z[0], 8.0);
}

#[test]
fn bin_of_hand_values() {
    assert_eq!(bin_of(0, 300, 100), 0);
    assert_eq!(bin_of(299, 300, 100), 99);
    assert_eq!(bin_of(300, 300, 100), 99);
    assert_eq!(bin_of(1_000_000, 300, 100), 99);
    assert_eq!(bin_of(2, 300, 100), 0);
    assert_eq!(bin_of(3, 300, 100), 1);
    assert_eq!(bin_of(150, 300, 100), 50);
    assert_eq!(bin_of(299, 300, 1), 0);
    assert_eq!(bin_of(0, 300, 1), 0);
}
