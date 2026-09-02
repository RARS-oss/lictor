// SPDX-License-Identifier: MIT
//! Brake feasibility against the Python-generated fixture (`fixtures/brake/gen.py` is the reference
//! reimplementation of the gym-pusht PD loop and the closed forms), plus the monotonicity property and the
//! brake/hold actions.

use lictor_core::{ActionKind, BrakeKind, ChunkBuf, FuseConfig, FuseMode, SafetyEnvelope};
use lictor_detect::brake::{brake_action, brake_feasible, hold_action, BrakeOut};
use serde_json::Value;

const BASE: &str = include_str!("../../../envelopes/pusht.base.toml");
const CASES: &str = include_str!("fixtures/brake/cases.json");

fn f(v: &Value, k: &str) -> f64 {
    v[k].as_f64().unwrap_or_else(|| panic!("missing number `{k}`"))
}

fn u(v: &Value, k: &str) -> u64 {
    v[k].as_u64().unwrap_or_else(|| panic!("missing integer `{k}`"))
}

fn arr(v: &Value, k: &str) -> Vec<f64> {
    v[k].as_array()
        .unwrap_or_else(|| panic!("missing array `{k}`"))
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn close(a: f64, b: f64) -> bool {
    let scale = if b.abs() > 1.0 { b.abs() } else { 1.0 };
    (a - b).abs() <= 1e-12 * scale
}

fn base() -> SafetyEnvelope {
    SafetyEnvelope::from_toml(BASE).unwrap()
}

/// The envelope of a fixture case: the PushT base with the case's brake model, action kind, box and limits.
fn envelope_for(c: &Value) -> SafetyEnvelope {
    let mut e = base();
    e.brake.kind = serde_json::from_value(c["kind"].clone()).unwrap();
    e.embodiment.action_kind = serde_json::from_value(c["action_kind"].clone()).unwrap();
    e.brake.k_p = f(c, "k_p");
    e.brake.k_v = f(c, "k_v");
    e.brake.substeps = u(c, "substeps") as u16;
    e.brake.dt = f(c, "dt");
    e.brake.commit_steps = u(c, "commit_steps") as u16;
    e.brake.brake_steps = u(c, "brake_steps") as u16;
    e.brake.react_ticks = u(c, "react_ticks") as u16;
    e.embodiment.control_hz_num = u(c, "control_hz_num") as u32;
    e.embodiment.control_hz_den = u(c, "control_hz_den") as u32;
    e.embodiment.horizon = u(c, "horizon") as u16;
    e.embodiment.exec_steps = u(c, "exec_steps") as u16;
    e.a_max = f(c, "a_max");
    e.j_max = f(c, "j_max");
    e.box_lo = arr(c, "box_lo");
    e.box_hi = arr(c, "box_hi");
    e.margin = f(c, "margin");
    e
}

fn chunk_of(c: &Value, horizon: u16, exec: u16) -> ChunkBuf {
    let rows: Vec<f64> = c["chunk"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
        .collect();
    let mut b = ChunkBuf::new();
    b.fill(0, 0, horizon, 2, exec, &rows).unwrap();
    b
}

fn run_case(c: &Value) -> (BrakeOut, Value) {
    let e = envelope_for(c);
    let cfg = e.compile(FuseMode::Enforce, None).unwrap();
    let buf = chunk_of(c, e.embodiment.horizon, e.embodiment.exec_steps);
    let p0 = arr(c, "p0");
    let v0 = arr(c, "v0");
    let from = u(c, "from") as usize;
    (brake_feasible(&cfg, &p0, &v0, buf.view().unwrap(), from), c["expect"].clone())
}

#[test]
fn fixture_parity_to_1e12() {
    let cases: Vec<Value> = serde_json::from_str(CASES).unwrap();
    assert!(cases.len() >= 8, "gen.py must emit at least 8 cases");
    let mut kinds_seen = Vec::new();
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let (out, expect) = run_case(c);
        let want_margin = f(&expect, "margin");
        let want_stop = f(&expect, "stop_dist");
        assert!(close(out.margin, want_margin), "{name}: margin {} != {want_margin}", out.margin);
        assert!(close(out.stop_dist, want_stop), "{name}: stop_dist {} != {want_stop}", out.stop_dist);
        assert_eq!(out.feasible, expect["feasible"].as_bool().unwrap(), "{name}: feasible");
        assert_eq!(out.feasible, out.margin >= 0.0, "{name}: feasible must be margin >= 0");
        let kind = c["kind"].as_str().unwrap().to_string();
        if !kinds_seen.contains(&kind) {
            kinds_seen.push(kind);
        }
    }
    for k in ["pd_second_order", "first_order_decay", "bounded_accel", "jerk_limited", "zero_velocity_hold"] {
        assert!(kinds_seen.iter().any(|s| s == k), "fixture lacks a `{k}` case");
    }
    let froms: Vec<u64> = cases.iter().map(|c| u(c, "from")).collect();
    for want in [0, 3, 7] {
        assert!(froms.contains(&want), "fixture lacks a from={want} case");
    }
    assert!(cases.iter().any(|c| arr(c, "v0").iter().any(|x| *x != 0.0)), "fixture lacks a v0 != 0 case");
    assert!(cases.iter().any(|c| c["expect"]["feasible"] == false), "fixture lacks an infeasible case");
    assert!(cases.iter().any(|c| c["expect"]["feasible"] == true), "fixture lacks a feasible case");
}

#[test]
fn fixture_is_bit_exact_not_just_close() {
    // Python and Rust evaluate the same IEEE-754 operations in the same order: the doubles are identical.
    let cases: Vec<Value> = serde_json::from_str(CASES).unwrap();
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        let (out, expect) = run_case(c);
        assert_eq!(out.margin.to_bits(), f(&expect, "margin").to_bits(), "{name}: margin bits");
        assert_eq!(out.stop_dist.to_bits(), f(&expect, "stop_dist").to_bits(), "{name}: stop_dist bits");
    }
}

fn config_with_horizon(h: u16) -> FuseConfig {
    let mut e = base();
    e.embodiment.horizon = h;
    e.compile(FuseMode::Enforce, None).unwrap()
}

#[test]
fn a_64_row_chunk_completes_and_margin_is_monotone_in_v0_magnitude() {
    let cfg = config_with_horizon(64);
    let mut buf = ChunkBuf::new();
    // every row holds the setpoint 5 px inside the +x wall (box_hi 495); the agent starts 30 px short of it
    let rows: Vec<f64> = (0..64).flat_map(|_| [490.0, 256.0]).collect();
    buf.fill(0, 0, 64, 2, 8, &rows).unwrap();
    let ch = buf.view().unwrap();
    let p0 = [460.0, 256.0];
    let mut prev = f64::INFINITY;
    let mut margins = Vec::new();
    for k in 0..=16 {
        let v0 = [100.0 * k as f64, 0.0];
        let out = brake_feasible(&cfg, &p0, &v0, ch, 0);
        assert!(out.margin.is_finite() && out.stop_dist.is_finite());
        assert!(out.margin <= prev, "margin rose from {prev} to {} at |v0| = {}", out.margin, v0[0]);
        prev = out.margin;
        margins.push(out.margin);
    }
    assert!(
        margins[0] > 0.0,
        "at rest the critically damped agent does not overshoot: feasible ({margins:?})"
    );
    assert!(*margins.last().unwrap() < 0.0, "at 1600 px/s toward the wall it is not ({margins:?})");
    assert!(margins.first().unwrap() > margins.last().unwrap());
}

#[test]
fn from_at_or_past_commit_steps_simulates_only_the_brake_tail() {
    let cfg = config_with_horizon(15);
    let mut buf = ChunkBuf::new();
    let rows: Vec<f64> = (0..15).flat_map(|i| [600.0 + 50.0 * i as f64, 256.0]).collect();
    buf.fill(0, 0, 15, 2, 8, &rows).unwrap();
    let ch = buf.view().unwrap();
    let p0 = [300.0, 256.0];
    let v0 = [0.0, 0.0];
    let at_rest = brake_feasible(&cfg, &p0, &v0, ch, 8);
    assert!(at_rest.feasible);
    assert!(close(at_rest.stop_dist, 0.0), "the tail from rest does not move: {}", at_rest.stop_dist);
    // the agent does not move: the slack is the smallest wall distance over both dims, 495 - 300 in x
    assert_eq!(at_rest.margin, 495.0 - 300.0);
    let same = brake_feasible(&cfg, &p0, &v0, ch, 14);
    assert_eq!(same.margin.to_bits(), at_rest.margin.to_bits());
    let prefix = brake_feasible(&cfg, &p0, &v0, ch, 0);
    assert!(!prefix.feasible, "rows commanding x >= 600 drive the agent through the wall");
}

#[test]
fn closed_form_uses_the_passed_velocity() {
    let mut e = base();
    e.brake.kind = BrakeKind::BoundedAccel;
    e.a_max = 2000.0;
    let cfg = e.compile(FuseMode::Enforce, None).unwrap();
    let mut buf = ChunkBuf::new();
    buf.fill(0, 0, 15, 2, 8, &(0..15).flat_map(|_| [256.0, 256.0]).collect::<Vec<_>>()).unwrap();
    let ch = buf.view().unwrap();
    let p0 = [256.0, 256.0];
    let slow = brake_feasible(&cfg, &p0, &[0.0, 0.0], ch, 0);
    let fast = brake_feasible(&cfg, &p0, &[900.0, 0.0], ch, 0);
    assert!(slow.feasible);
    assert!(fast.margin < slow.margin, "a larger v0 shrinks the closed-form margin");
    // the rows command the current position, so the modelled velocity is zero after the first step and both
    // runs end where they started with an empty stopping ball: stop_dist == 0 for both
    assert_eq!(fast.stop_dist, 0.0);
    assert_eq!(slow.stop_dist, 0.0);
    // the binding constraint is the ball from the CURRENT state: 256 - 17 - (||v0||^2/(2 a_max) + ||v0|| dt)
    let expect = 256.0 - 17.0 - (900.0 * 900.0 / (2.0 * 2000.0) + 900.0 * 0.1);
    assert!(close(fast.margin, expect), "{} vs {expect}", fast.margin);
    assert_eq!(slow.margin, 256.0 - 17.0);
}

#[test]
fn brake_and_hold_actions_position_kind() {
    let cfg = config_with_horizon(15);
    let mut out = [f64::NAN; 4];
    brake_action(&cfg, &[500.0, -3.0], &[10.0, 10.0], &mut out);
    assert_eq!(&out[..2], &[495.0, 17.0]);
    assert!(out[2].is_nan() && out[3].is_nan(), "only [..dim] is written");
    brake_action(&cfg, &[100.0, 200.0], &[0.0, 0.0], &mut out);
    assert_eq!(&out[..2], &[100.0, 200.0]);
    hold_action(&cfg, &[600.0, 300.0], &mut out);
    assert_eq!(&out[..2], &[495.0, 300.0]);
    let mut short = [0.0; 1];
    hold_action(&cfg, &[100.0, 100.0], &mut short);
    assert_eq!(short, [100.0]);
}

#[test]
fn brake_and_hold_actions_velocity_kinds() {
    let mut e = base();
    e.brake.kind = BrakeKind::ZeroVelocityHold;
    e.embodiment.action_kind = ActionKind::JointVelocity;
    e.a_max = 500.0;
    let cfg = e.compile(FuseMode::Enforce, None).unwrap();
    let mut out = [0.0; 2];
    brake_action(&cfg, &[256.0, 256.0], &[100.0, 0.0], &mut out);
    let expect = 100.0 * (1.0 - 500.0 * 0.1 / (100.0 + 1e-9));
    assert!(close(out[0], expect) && out[1] == 0.0, "{out:?}");
    brake_action(&cfg, &[256.0, 256.0], &[10.0, 0.0], &mut out);
    assert_eq!(out, [0.0, 0.0], "below a_max*dt the ramp reaches zero at once");
    hold_action(&cfg, &[256.0, 256.0], &mut out);
    assert_eq!(out, [0.0, 0.0]);

    e.embodiment.action_kind = ActionKind::EeDelta;
    let cfg = e.compile(FuseMode::Enforce, None).unwrap();
    brake_action(&cfg, &[256.0, 256.0], &[100.0, 0.0], &mut out);
    assert!(close(out[0], expect * 0.1), "ee_delta rows are per-step displacements: {out:?}");
}
