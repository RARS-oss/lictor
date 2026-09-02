// SPDX-License-Identifier: MIT
//! Tier-0 fixture and property tests. `fixtures/tier0/cases.json` is written by `fixtures/tier1/gen.py` (an
//! independent Python reimplementation of the ARCHITECTURE 5.1 table and of the sequential leash projection).
//!
//! The `FuseConfig` is built by hand from the fixture's config block (every field of `FuseConfig` is public and
//! the derived constants are `1/dt`, `inv_dt^2`, `inv_dt^3`, `v_max*dt`), so the fixture tests do not depend on
//! `SafetyEnvelope::compile`; the box limits in the fixture are the compiled (margin-adjusted) values. One test
//! cross-checks that hand-built config against the real `compile()` of `envelopes/pusht.base.toml` and replays
//! the base-limit cases under it.

use lictor_core::{
    ActionKind, BrakeKind, BrakeModel, CalibrationC, ChunkBuf, ChunkView, ClampMode, ContactLimit,
    FuseConfig, FuseMode, Hysteresis, ObsView, RearmPolicy, SafetyEnvelope, TripMask, MAX_D, MAX_POS,
};
use lictor_detect::tier0::{check_action, check_chunk, Tier0Out};
use serde_json::Value;

const CASES: &str = include_str!("fixtures/tier0/cases.json");
/// Absolute tolerance on projected coordinates (the Python mirror uses the same leash factor; this only absorbs
/// last-bit differences of the sequential accumulation).
const TOL: f64 = 1e-9;
/// Tolerance for the unit-vector direction checks.
const DIR_TOL: f64 = 1e-12;

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn trip_bit(name: &str) -> u32 {
    match name {
        "workspace" => TripMask::WORKSPACE,
        "speed" => TripMask::SPEED,
        "accel" => TripMask::ACCEL,
        "jerk" => TripMask::JERK,
        "reach" => TripMask::REACH,
        "contact" => TripMask::CONTACT,
        "brake" => TripMask::BRAKE,
        other => panic!("unknown trip name {other}"),
    }
}

fn trips_of(v: &Value) -> u32 {
    v.as_array().unwrap().iter().fold(0, |m, n| m | trip_bit(n.as_str().unwrap()))
}

fn trip_names(m: u32) -> Vec<&'static str> {
    TripMask::NAMES.iter().enumerate().filter(|(i, _)| m & (1 << i) != 0).map(|(_, n)| *n).collect()
}

fn cfg_from(v: &Value) -> FuseConfig {
    let dim = usize::try_from(v["dim"].as_u64().unwrap()).unwrap();
    let pos_dim = usize::try_from(v["pos_dim"].as_u64().unwrap()).unwrap();
    let dt = v["dt"].as_f64().unwrap();
    let inv_dt = 1.0 / dt;
    let inv_dt2 = inv_dt * inv_dt;
    let inv_dt3 = inv_dt2 * inv_dt;
    let mut box_lo = [0.0; MAX_POS];
    let mut box_hi = [0.0; MAX_POS];
    for (i, x) in f64s(&v["box_lo"]).iter().enumerate() {
        box_lo[i] = *x;
    }
    for (i, x) in f64s(&v["box_hi"]).iter().enumerate() {
        box_hi[i] = *x;
    }
    let v_max = v["v_max"].as_f64().unwrap();
    let clamp = match v["clamp"].as_str().unwrap() {
        "project" => ClampMode::Project,
        "off" => ClampMode::Off,
        other => panic!("unknown clamp mode {other}"),
    };
    let contact = v["contact"].as_object().map(|c| ContactLimit {
        radius: c["radius"].as_f64().unwrap(),
        v_max: c["v_max"].as_f64().unwrap(),
        aux_center: [
            u8::try_from(c["aux_center"][0].as_u64().unwrap()).unwrap(),
            u8::try_from(c["aux_center"][1].as_u64().unwrap()).unwrap(),
        ],
    });
    let mut norm_center = [0.0; MAX_D];
    let mut inv_norm_scale = [1.0; MAX_D];
    for c in 0..dim {
        norm_center[c] = 256.0;
        inv_norm_scale[c] = 1.0 / 256.0;
    }
    FuseConfig {
        envelope_digest: [0; 32],
        embodiment_digest: [0; 32],
        calib_digest: None,
        mode: FuseMode::Enforce,
        kind: ActionKind::EePosition,
        dim,
        pos_dim,
        horizon: 15,
        exec: 8,
        overlap: 7,
        dt,
        inv_dt,
        inv_dt2,
        inv_dt3,
        norm_center,
        inv_norm_scale,
        norm_scale_iso: 256.0,
        box_lo,
        box_hi,
        v_max,
        a_max: v["a_max"].as_f64().unwrap(),
        j_max: v["j_max"].as_f64().unwrap(),
        reach_max: v["reach_max"].as_f64().unwrap(),
        step_max: v_max * dt,
        contact,
        brake: BrakeModel {
            kind: BrakeKind::PdSecondOrder,
            k_p: 100.0,
            k_v: 20.0,
            substeps: 10,
            dt: 0.01,
            commit_steps: 8,
            brake_steps: 8,
            react_ticks: 1,
        },
        clamp,
        hyst: Hysteresis {
            k: 3,
            n: 5,
            warn_margin: 0.5,
            clear_ticks: 5,
            clamp_streak_to_brake: 3,
            max_clamps_per_episode: 60,
            stop_confirm_ticks: 2,
            v_stop_eps: 5.0,
            brake_timeout_ticks: 30,
            rearm: RearmPolicy::Auto,
            rearm_hold: 10,
            max_rearms: 2,
            escalate_after_hold_ticks: 30,
            handoff_timeout_ticks: 200,
            watchdog_ticks: 2,
        },
        window_mask: (1u64 << 5) - 1,
        tier0_enabled: trips_of(&v["tier0_enabled"]),
        n_operators: 0,
        calib: CalibrationC::DISARMED,
    }
}

struct Chunk {
    seq: u32,
    t_emit: u32,
    horizon: u16,
    dim: u16,
    exec_steps: u16,
    data: Vec<f64>,
}

impl Chunk {
    fn from_json(v: &Value) -> Self {
        Self {
            seq: u32::try_from(v["seq"].as_u64().unwrap()).unwrap(),
            t_emit: u32::try_from(v["t_emit"].as_u64().unwrap()).unwrap(),
            horizon: u16::try_from(v["horizon"].as_u64().unwrap()).unwrap(),
            dim: u16::try_from(v["dim"].as_u64().unwrap()).unwrap(),
            exec_steps: u16::try_from(v["exec_steps"].as_u64().unwrap()).unwrap(),
            data: f64s(&v["data"]),
        }
    }

    fn view(&self) -> ChunkView<'_> {
        ChunkView {
            seq: self.seq,
            t_emit: self.t_emit,
            horizon: self.horizon,
            dim: self.dim,
            exec_steps: self.exec_steps,
            data: &self.data,
        }
    }
}

fn close(a: f64, b: f64, tol: f64) -> bool {
    a == b || (a - b).abs() <= tol
}

fn assert_out(name: &str, got: &[f64], exp: &[f64]) {
    assert_eq!(got.len(), exp.len(), "{name}: projected length");
    for (i, (g, e)) in got.iter().zip(exp).enumerate() {
        assert!(close(*g, *e, TOL), "{name}: out[{i}] = {g} expected {e}");
    }
}

fn assert_res(name: &str, got: Tier0Out, exp: &Value) {
    let trips = trips_of(&exp["trips"]);
    assert_eq!(
        got.trips,
        trips,
        "{name}: trips {:?} expected {:?}",
        trip_names(got.trips),
        trip_names(trips)
    );
    assert_eq!(
        got.clamped_dims,
        u32::try_from(exp["clamped_dims"].as_u64().unwrap()).unwrap(),
        "{name}: clamped_dims"
    );
    let peak = exp["peak_speed"].as_f64().unwrap();
    assert!(
        close(got.peak_speed, peak, 1e-9 * peak.abs().max(1.0)),
        "{name}: peak_speed {} expected {peak}",
        got.peak_speed
    );
}

fn root() -> Value {
    serde_json::from_str(CASES).unwrap()
}

#[test]
fn fixture_shape_and_coverage() {
    let root = root();
    let chunks = root["chunk_cases"].as_array().unwrap();
    let actions = root["action_cases"].as_array().unwrap();
    assert!(chunks.len() >= 10, "{} chunk cases", chunks.len());
    assert!(actions.len() >= 6, "{} action cases", actions.len());
    let mut seen = 0u32;
    let mut off = false;
    let mut contact_armed = false;
    let mut contact_disarmed = false;
    let mut disabled_no_trip = false;
    for c in chunks {
        seen |= trips_of(&c["expect"]["trips"]);
        let cfg = &c["config"];
        if cfg["clamp"] == "off" {
            off = true;
        }
        let enabled = trips_of(&cfg["tier0_enabled"]);
        if cfg["contact"].is_object() {
            if enabled & TripMask::CONTACT != 0 {
                contact_armed = true;
            } else {
                contact_disarmed = true;
            }
        }
        if enabled & TripMask::SPEED == 0 && trips_of(&c["expect"]["trips"]) == 0 {
            disabled_no_trip = true;
        }
    }
    assert_eq!(seen, TripMask::TIER0_SOFT, "every soft trip appears: {:?}", trip_names(seen));
    assert!(off && contact_armed && contact_disarmed && disabled_no_trip);
}

#[test]
fn chunk_fixtures() {
    let root = root();
    for case in root["chunk_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cfg = cfg_from(&case["config"]);
        let pos = f64s(&case["pos"]);
        let aux = f64s(&case["aux"]);
        let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &aux, ext: &[] };
        let ch = Chunk::from_json(&case["chunk"]);
        let mut out = ChunkBuf::new();
        let res = check_chunk(&cfg, obs, ch.view(), &mut out);
        assert_res(name, res, &case["expect"]);
        let view = out.view().expect("out is filled");
        assert_eq!(view.horizon, ch.horizon, "{name}: horizon");
        assert_eq!(view.dim, ch.dim, "{name}: dim");
        assert_eq!(view.seq, ch.seq, "{name}: seq");
        assert_eq!(view.t_emit, ch.t_emit, "{name}: t_emit");
        assert_out(name, view.data, &f64s(&case["expect"]["out"]));
        if res.clamped_dims == 0 {
            assert_eq!(view.data, &ch.data[..], "{name}: nothing clamped -> out is the chunk, bit for bit");
        } else {
            assert_ne!(view.data, &ch.data[..], "{name}: clamped_dims set -> out differs");
        }
        // clamped_dims is exactly the set of coordinates that moved anywhere in the chunk.
        let d = usize::from(ch.dim);
        let mut moved = 0u32;
        for (i, (g, r)) in view.data.iter().zip(&ch.data).enumerate() {
            if g != r {
                moved |= 1 << (i % d);
            }
        }
        assert_eq!(res.clamped_dims, moved, "{name}: clamped_dims vs moved coordinates");
    }
}

#[test]
fn action_fixtures() {
    let root = root();
    for case in root["action_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cfg = cfg_from(&case["config"]);
        let pos = f64s(&case["pos"]);
        let aux = f64s(&case["aux"]);
        let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &aux, ext: &[] };
        let prev = f64s(&case["prev"]);
        let a = f64s(&case["a"]);
        let mut out = [0.0; MAX_D];
        let res = check_action(&cfg, obs, &prev, &a, &mut out);
        assert_res(name, res, &case["expect"]);
        assert_out(name, &out[..cfg.dim], &f64s(&case["expect"]["out"]));
        if res.clamped_dims == 0 {
            assert_eq!(&out[..cfg.dim], &a[..], "{name}: nothing clamped -> out == a");
        }
    }
}

fn base_cfg() -> FuseConfig {
    let root = root();
    let case = &root["chunk_cases"][0];
    assert_eq!(case["name"], "inside_box_smooth");
    cfg_from(&case["config"])
}

fn unit(v: &[f64]) -> Vec<f64> {
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| x / n).collect()
}

#[test]
fn action_leash_preserves_direction_and_is_idempotent() {
    let cfg = base_cfg();
    let pos = [256.0, 256.0];
    let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &[], ext: &[] };
    let prev = [256.0, 256.0];
    for k in 0..24 {
        let theta = f64::from(k) * 0.2617993877991494 + 0.05;
        let a = [prev[0] + 140.0 * theta.cos(), prev[1] + 140.0 * theta.sin()];
        let mut out = [0.0; MAX_D];
        let res = check_action(&cfg, obs, &prev, &a, &mut out);
        assert_eq!(res.trips, TripMask::SPEED, "theta {theta}");
        assert_eq!(res.clamped_dims, 0b11);
        let raw_dir = unit(&[a[0] - prev[0], a[1] - prev[1]]);
        let out_dir = unit(&[out[0] - prev[0], out[1] - prev[1]]);
        for c in 0..2 {
            assert!(
                (raw_dir[c] - out_dir[c]).abs() <= DIR_TOL,
                "direction {c}: {} vs {}",
                raw_dir[c],
                out_dir[c]
            );
        }
        let n = ((out[0] - prev[0]).powi(2) + (out[1] - prev[1]).powi(2)).sqrt();
        assert!(n <= cfg.step_max, "leashed step {n} must not exceed step_max {}", cfg.step_max);
        assert!(n > cfg.step_max * (1.0 - 1e-9), "leashed step {n} lands on the limit");
        // Idempotence: the projection of the projection is the projection, and it re-passes the strict check.
        let mut again = [0.0; MAX_D];
        let res2 = check_action(&cfg, obs, &prev, &out[..2], &mut again);
        assert_eq!(res2.trips, 0);
        assert_eq!(res2.clamped_dims, 0);
        assert_eq!(&again[..2], &out[..2]);
    }
}

#[test]
fn action_reach_pull_preserves_the_ray() {
    let mut cfg = base_cfg();
    cfg.v_max = 3000.0;
    cfg.step_max = cfg.v_max * cfg.dt;
    let pos = [200.0, 300.0];
    let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &[], ext: &[] };
    let prev = [200.0, 300.0];
    let a = [200.0 + 128.0, 300.0 - 96.0];
    let mut out = [0.0; MAX_D];
    let res = check_action(&cfg, obs, &prev, &a, &mut out);
    assert_eq!(res.trips, TripMask::REACH);
    let raw_dir = unit(&[a[0] - pos[0], a[1] - pos[1]]);
    let out_dir = unit(&[out[0] - pos[0], out[1] - pos[1]]);
    for c in 0..2 {
        assert!((raw_dir[c] - out_dir[c]).abs() <= DIR_TOL);
    }
    let r = ((out[0] - pos[0]).powi(2) + (out[1] - pos[1]).powi(2)).sqrt();
    assert!(r <= cfg.reach_max && r > cfg.reach_max * (1.0 - 1e-9), "r = {r}");
    let mut again = [0.0; MAX_D];
    let res2 = check_action(&cfg, obs, &prev, &out[..2], &mut again);
    assert_eq!(res2.trips, 0);
    assert_eq!(&again[..2], &out[..2]);
}

#[test]
fn off_mode_and_disabled_checks_pass_through() {
    let mut cfg = base_cfg();
    cfg.clamp = ClampMode::Off;
    let pos = [256.0, 256.0];
    let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &[], ext: &[] };
    let prev = [256.0, 256.0];
    let a = [406.0, 500.0];
    let mut out = [0.0; MAX_D];
    let res = check_action(&cfg, obs, &prev, &a, &mut out);
    assert_eq!(res.trips, TripMask::SPEED | TripMask::WORKSPACE | TripMask::REACH);
    assert_eq!(res.clamped_dims, 0);
    assert_eq!(&out[..2], &a[..]);

    // Disabled checks neither trip nor clamp, whatever the clamp mode.
    cfg.clamp = ClampMode::Project;
    cfg.tier0_enabled = TripMask::BRAKE;
    let res = check_action(&cfg, obs, &prev, &a, &mut out);
    assert_eq!(res.trips, 0);
    assert_eq!(res.clamped_dims, 0);
    assert_eq!(&out[..2], &a[..]);
    let expect_peak = (150.0f64 * 150.0 + 244.0 * 244.0).sqrt() * cfg.inv_dt;
    assert!((res.peak_speed - expect_peak).abs() < 1e-9, "{} vs {expect_peak}", res.peak_speed);
}

#[test]
fn chunk_leash_preserves_direction_and_is_idempotent() {
    let cfg = base_cfg();
    let pos = [256.0, 256.0];
    let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &[], ext: &[] };
    // Every row 140 px from p_t on a heading that jumps by 2 rad per row: consecutive rows are ~235 px apart,
    // so SPEED trips on every row while the box (max excursion 140 px) and reach (140 < 150) stay clear.
    let mut data = Vec::new();
    for i in 0..15 {
        let theta = 0.4 + 2.0 * f64::from(i);
        data.push(pos[0] + 140.0 * theta.cos());
        data.push(pos[1] + 140.0 * theta.sin());
    }
    let raw = Chunk { seq: 0, t_emit: 0, horizon: 15, dim: 2, exec_steps: 8, data };
    let mut out = ChunkBuf::new();
    let res = check_chunk(&cfg, obs, raw.view(), &mut out);
    assert!(res.trips & TripMask::SPEED != 0);
    let proj = out.view().unwrap().data.to_vec();
    // Direction of every projected step equals the direction of the raw row relative to the PREVIOUS PROJECTED row.
    let mut q = pos.to_vec();
    for i in 0..15 {
        let r = &raw.data[2 * i..2 * i + 2];
        let o = &proj[2 * i..2 * i + 2];
        let raw_dir = unit(&[r[0] - q[0], r[1] - q[1]]);
        let out_dir = unit(&[o[0] - q[0], o[1] - q[1]]);
        for c in 0..2 {
            assert!((raw_dir[c] - out_dir[c]).abs() <= DIR_TOL, "row {i} coord {c}");
        }
        let n = ((o[0] - q[0]).powi(2) + (o[1] - q[1]).powi(2)).sqrt();
        assert!(n <= cfg.step_max, "row {i}: step {n}");
        q = o.to_vec();
    }
    // Idempotence: projecting the projection changes nothing and raises no speed/box/reach trip.
    let again = Chunk { seq: 0, t_emit: 0, horizon: 15, dim: 2, exec_steps: 8, data: proj.clone() };
    let mut out2 = ChunkBuf::new();
    let res2 = check_chunk(&cfg, obs, again.view(), &mut out2);
    assert_eq!(res2.trips & (TripMask::SPEED | TripMask::WORKSPACE | TripMask::REACH), 0);
    assert_eq!(res2.clamped_dims, 0);
    assert_eq!(out2.view().unwrap().data, &proj[..]);
}

#[test]
fn compiled_base_envelope_matches_the_fixture_config() {
    // The fixture configs are hand-built from the manifest; the real `compile()` of envelopes/pusht.base.toml
    // must agree with the base fixture config on every field Tier 0 reads, and the base-limit fixture cases must
    // replay unchanged under the compiled config.
    let env = SafetyEnvelope::from_toml(include_str!("../../../envelopes/pusht.base.toml"))
        .expect("base envelope parses");
    let cfg = env.compile(FuseMode::Enforce, None).expect("base envelope compiles");
    let fx = base_cfg();
    assert_eq!(cfg.dim, fx.dim);
    assert_eq!(cfg.pos_dim, fx.pos_dim);
    assert_eq!(cfg.clamp, fx.clamp);
    assert_eq!(cfg.tier0_enabled, fx.tier0_enabled, "{:?}", trip_names(cfg.tier0_enabled));
    assert_eq!(cfg.contact, fx.contact);
    assert_eq!(cfg.box_lo[..2], fx.box_lo[..2]);
    assert_eq!(cfg.box_hi[..2], fx.box_hi[..2]);
    let pairs = [
        (cfg.dt, fx.dt, "dt"),
        (cfg.inv_dt, fx.inv_dt, "inv_dt"),
        (cfg.inv_dt2, fx.inv_dt2, "inv_dt2"),
        (cfg.inv_dt3, fx.inv_dt3, "inv_dt3"),
        (cfg.v_max, fx.v_max, "v_max"),
        (cfg.a_max, fx.a_max, "a_max"),
        (cfg.j_max, fx.j_max, "j_max"),
        (cfg.reach_max, fx.reach_max, "reach_max"),
        (cfg.step_max, fx.step_max, "step_max"),
        (cfg.norm_scale_iso, fx.norm_scale_iso, "norm_scale_iso"),
    ];
    for (a, b, what) in pairs {
        assert!(close(a, b, 1e-12 * b.abs().max(1.0)), "{what}: compiled {a} vs fixture {b}");
    }
    let root = root();
    let mut replayed = 0;
    for case in root["chunk_cases"].as_array().unwrap() {
        let c = &case["config"];
        let is_base = c["v_max"] == 1000.0
            && c["a_max"] == 20000.0
            && c["j_max"] == 400000.0
            && c["reach_max"] == 150.0
            && c["clamp"] == "project"
            && c["contact"].is_null()
            && c["dim"] == 2
            && trips_of(&c["tier0_enabled"]) == cfg.tier0_enabled;
        if !is_base {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let pos = f64s(&case["pos"]);
        let aux = f64s(&case["aux"]);
        let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &aux, ext: &[] };
        let ch = Chunk::from_json(&case["chunk"]);
        let mut out = ChunkBuf::new();
        let res = check_chunk(&cfg, obs, ch.view(), &mut out);
        assert_res(name, res, &case["expect"]);
        assert_out(name, out.view().unwrap().data, &f64s(&case["expect"]["out"]));
        replayed += 1;
    }
    assert!(replayed >= 4, "{replayed} base-limit cases replayed under the compiled config");
}
