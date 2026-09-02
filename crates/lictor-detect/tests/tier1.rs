// SPDX-License-Identifier: MIT
//! Tier-1 feature tests. `fixtures/tier1/cases.json` is written by `fixtures/tier1/gen.py`, an independent
//! Python reimplementation of the eight ARCHITECTURE 5.3 formulas (with the manifest-only `speed_peak` / `stall`
//! definitions and the t_emit-derived overlap); every raw feature is asserted to 1e-12.
//!
//! The `FuseConfig` is built by hand from the fixture's config block (manifest values only: dim, pos_dim,
//! horizon, exec, dt, norm_center, norm_scale, norm_scale_iso), so these tests do not depend on
//! `SafetyEnvelope::compile`.

use std::collections::HashMap;

use lictor_core::{
    ActionKind, BrakeKind, BrakeModel, CalibrationC, ChunkView, ClampMode, Feat, FuseConfig, FuseMode,
    Hysteresis, ObsView, RearmPolicy, Scores, TripMask, MAX_D, MAX_POS, NFEAT,
};
use lictor_detect::tier1::{features, Tier1Rt, PE_WINDOW, STALL_VREF_FRAC};
use serde_json::Value;

const CASES: &str = include_str!("fixtures/tier1/cases.json");
const SRC: &str = include_str!("../src/tier1.rs");
const TOL: f64 = 1e-12;

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cfg_from(v: &Value) -> FuseConfig {
    let dim = usize::try_from(v["dim"].as_u64().unwrap()).unwrap();
    let pos_dim = usize::try_from(v["pos_dim"].as_u64().unwrap()).unwrap();
    let horizon = usize::try_from(v["horizon"].as_u64().unwrap()).unwrap();
    let exec = usize::try_from(v["exec"].as_u64().unwrap()).unwrap();
    let dt = v["dt"].as_f64().unwrap();
    let inv_dt = 1.0 / dt;
    let inv_dt2 = inv_dt * inv_dt;
    let inv_dt3 = inv_dt2 * inv_dt;
    let mut norm_center = [0.0; MAX_D];
    let mut inv_norm_scale = [1.0; MAX_D];
    for (c, x) in f64s(&v["norm_center"]).iter().enumerate() {
        norm_center[c] = *x;
    }
    for (c, x) in f64s(&v["norm_scale"]).iter().enumerate() {
        inv_norm_scale[c] = 1.0 / *x;
    }
    let mut box_lo = [0.0; MAX_POS];
    let mut box_hi = [0.0; MAX_POS];
    for c in 0..pos_dim {
        box_lo[c] = 17.0;
        box_hi[c] = 495.0;
    }
    FuseConfig {
        envelope_digest: [0; 32],
        embodiment_digest: [0; 32],
        calib_digest: None,
        mode: FuseMode::Observe,
        kind: ActionKind::EePosition,
        dim,
        pos_dim,
        horizon,
        exec,
        overlap: horizon - exec,
        dt,
        inv_dt,
        inv_dt2,
        inv_dt3,
        norm_center,
        inv_norm_scale,
        norm_scale_iso: v["norm_scale_iso"].as_f64().unwrap(),
        box_lo,
        box_hi,
        v_max: 1000.0,
        a_max: 20000.0,
        j_max: 400000.0,
        reach_max: 150.0,
        step_max: 100.0,
        contact: None,
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
        clamp: ClampMode::Project,
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
        tier0_enabled: TripMask::TIER0_SOFT | TripMask::BRAKE,
        n_operators: 0,
        calib: CalibrationC::DISARMED,
    }
}

fn base_cfg() -> FuseConfig {
    let root: Value = serde_json::from_str(CASES).unwrap();
    cfg_from(&root[0]["config"])
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
    fn from_json(v: &Value) -> Option<Self> {
        if v.is_null() {
            return None;
        }
        Some(Self {
            seq: u32::try_from(v["seq"].as_u64().unwrap()).unwrap(),
            t_emit: u32::try_from(v["t_emit"].as_u64().unwrap()).unwrap(),
            horizon: u16::try_from(v["horizon"].as_u64().unwrap()).unwrap(),
            dim: u16::try_from(v["dim"].as_u64().unwrap()).unwrap(),
            exec_steps: u16::try_from(v["exec_steps"].as_u64().unwrap()).unwrap(),
            data: f64s(&v["data"]),
        })
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

fn close(a: f64, b: f64) -> bool {
    a == b || (a - b).abs() <= TOL
}

fn valid_names(m: u32) -> Vec<&'static str> {
    Feat::NAMES.iter().enumerate().filter(|(j, _)| m & (1 << j) != 0).map(|(_, n)| *n).collect()
}

#[test]
fn fixtures_replay() {
    let root: Value = serde_json::from_str(CASES).unwrap();
    let cases = root.as_array().unwrap();
    let mut last_tce: HashMap<String, f64> = HashMap::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let cfg = cfg_from(&case["config"]);
        let mut rt = Tier1Rt::new();
        let mut sc = Scores::default();
        for (k, tick) in case["ticks"].as_array().unwrap().iter().enumerate() {
            let t = u32::try_from(tick["t"].as_u64().unwrap()).unwrap();
            let idx = u16::try_from(tick["idx"].as_u64().unwrap()).unwrap();
            let pos = f64s(&tick["pos"]);
            let ext = f64s(&tick["ext"]);
            let chunk = Chunk::from_json(&tick["chunk"]);
            let obs = ObsView { t, pos: &pos, vel: None, aux: &[], ext: &ext };
            // decide() pushes the trail before calling features().
            rt.trail.push(&pos, cfg.pos_dim);
            features(&cfg, &mut rt, obs, chunk.as_ref().map(Chunk::view), idx, &mut sc);
            let exp = &tick["expect"];
            let f = f64s(&exp["f"]);
            let valid = u32::try_from(exp["valid"].as_u64().unwrap()).unwrap();
            assert_eq!(
                sc.valid,
                valid,
                "{name} tick {k} (t = {t}): valid {:?} expected {:?}",
                valid_names(sc.valid),
                valid_names(valid)
            );
            for (j, fj) in f.iter().enumerate() {
                assert!(
                    close(sc.f[j], *fj),
                    "{name} tick {k} (t = {t}): f[{}] = {} expected {fj}",
                    Feat::NAMES[j],
                    sc.f[j]
                );
                if valid & (1 << j) == 0 {
                    assert_eq!(
                        sc.f[j],
                        0.0,
                        "{name} tick {k}: invalid feature {} must read 0.0",
                        Feat::NAMES[j]
                    );
                }
            }
            if let Some(ch) = &chunk {
                // The raw-vs-raw contract: rt.prev is the chunk as delivered, bit for bit.
                assert!(rt.have_prev, "{name} tick {k}: have_prev after a chunk");
                let prev = rt.prev.view().expect("prev filled");
                assert_eq!(prev.data, &ch.data[..], "{name} tick {k}: rt.prev must hold the RAW chunk");
                assert_eq!(prev.t_emit, ch.t_emit);
                assert_eq!(prev.horizon, ch.horizon);
                assert_eq!(prev.dim, ch.dim);
                assert_eq!(prev.seq, ch.seq);
                assert_eq!(rt.held_valid, sc.valid & Feat::CHUNK_BOUNDARY_MASK);
                if sc.valid & Feat::Tce.bit() != 0 {
                    last_tce.insert(name.to_string(), sc.f[Feat::Tce as usize]);
                }
            } else {
                // Between boundaries the held features are copied unchanged.
                for j in 0..NFEAT {
                    if Feat::CHUNK_BOUNDARY_MASK & (1 << j) != 0 {
                        let expect = if rt.held_valid & (1 << j) != 0 { rt.held[j] } else { 0.0 };
                        assert_eq!(sc.f[j], expect, "{name} tick {k}: held copy of {}", Feat::NAMES[j]);
                    }
                }
            }
        }
        // The chunk-boundary bits of `valid` never change between deliveries: they are held.
        let ticks = case["ticks"].as_array().unwrap();
        let mut held_bits: Option<u32> = None;
        for tick in ticks {
            let valid =
                u32::try_from(tick["expect"]["valid"].as_u64().unwrap()).unwrap() & Feat::CHUNK_BOUNDARY_MASK;
            if !tick["chunk"].is_null() {
                held_bits = Some(valid);
            } else if let Some(h) = held_bits {
                assert_eq!(valid, h, "{name}: boundary validity changed between deliveries");
            }
        }
    }
    // The raw-vs-raw pair: the projected previous chunk gives a different tce.
    let mut pairs = 0;
    for case in cases {
        if let Some(other) = case["tce_differs_from"].as_str() {
            let name = case["name"].as_str().unwrap();
            let a = last_tce.get(name).unwrap_or_else(|| panic!("{name}: no valid tce"));
            let b = last_tce.get(other).unwrap_or_else(|| panic!("{other}: no valid tce"));
            assert!((a - b).abs() > 1e-6, "{name} vs {other}: tce {a} vs {b} should differ");
            pairs += 1;
        }
    }
    assert!(pairs >= 1, "the fixture must carry a raw-vs-projected pair");
}

#[test]
fn fixture_carries_the_required_cases() {
    let root: Value = serde_json::from_str(CASES).unwrap();
    let cases = root.as_array().unwrap();
    let by_name =
        |n: &str| cases.iter().find(|c| c["name"] == n).unwrap_or_else(|| panic!("case {n} missing"));
    let tce = Feat::Tce.bit();
    let acc = Feat::Acc.bit();
    let njr = Feat::Njr.bit();
    let per_tick = Feat::PathIneff.bit() | Feat::Stall.bit();

    // Validity bits of the LAST delivered chunk of a case.
    let last_chunk_valid = |n: &str| -> u32 {
        let c = by_name(n);
        let t = c["ticks"].as_array().unwrap().iter().rfind(|t| !t["chunk"].is_null()).unwrap();
        u32::try_from(t["expect"]["valid"].as_u64().unwrap()).unwrap()
    };
    let last_tick = |n: &str| -> &Value { by_name(n)["ticks"].as_array().unwrap().last().unwrap() };

    // At least three chunk pairs with a valid overlap.
    let pairs = cases
        .iter()
        .filter(|c| {
            c["ticks"].as_array().unwrap().iter().any(|t| {
                !t["chunk"].is_null()
                    && (u32::try_from(t["expect"]["valid"].as_u64().unwrap()).unwrap() & tce) != 0
            })
        })
        .count();
    assert!(pairs >= 3, "{pairs} chunk pairs with a valid tce");

    // Frozen policy: acm_neg at its maximum (0), njr 0, speed_peak 0, and tce/acc 0 on the second chunk.
    let frozen = by_name("frozen_policy");
    let second = &frozen["ticks"][8];
    assert!(!second["chunk"].is_null());
    let f = f64s(&second["expect"]["f"]);
    assert_eq!(f[Feat::AcmNeg as usize], 0.0);
    assert_eq!(f[Feat::Njr as usize], 0.0);
    assert_eq!(f[Feat::SpeedPeak as usize], 0.0);
    assert_eq!(f[Feat::Tce as usize], 0.0);
    assert_eq!(f[Feat::Acc as usize], 0.0);
    let lt = f64s(&last_tick("frozen_policy")["expect"]["f"]);
    assert_eq!(lt[Feat::PathIneff as usize], 1.0);
    assert_eq!(lt[Feat::Stall as usize], 1.0);

    // Dithering trail: path_ineff dominates the per-tick channels.
    let lt = f64s(&last_tick("dithering_trail")["expect"]["f"]);
    let lv = u32::try_from(last_tick("dithering_trail")["expect"]["valid"].as_u64().unwrap()).unwrap();
    assert_eq!(lv & per_tick, per_tick);
    assert!(lt[Feat::PathIneff as usize] > 0.75, "path_ineff {}", lt[Feat::PathIneff as usize]);
    assert!(lt[Feat::PathIneff as usize] > lt[Feat::Stall as usize]);

    // Sync delay: d = 7 absent, d = 6 gives L = 1 (invalid), d = 5 gives L = 2 (valid).
    assert_eq!(last_chunk_valid("sync_d7_overlap_absent") & (tce | acc), 0);
    assert_eq!(last_chunk_valid("sync_d6_overlap_l1") & (tce | acc), 0);
    assert_eq!(last_chunk_valid("sync_d5_overlap_l2") & (tce | acc), tce | acc);
    let d7 = by_name("sync_d7_overlap_absent");
    let chunks: Vec<&Value> =
        d7["ticks"].as_array().unwrap().iter().filter(|t| !t["chunk"].is_null()).collect();
    assert_eq!(
        chunks[1]["chunk"]["t_emit"].as_u64().unwrap() - chunks[0]["chunk"]["t_emit"].as_u64().unwrap(),
        15
    );

    // Async drop keeps the overlap.
    assert_eq!(last_chunk_valid("async_drop_d3") & (tce | acc), tce | acc);
    let ad = by_name("async_drop_d3");
    let first = ad["ticks"].as_array().unwrap().iter().find(|t| !t["chunk"].is_null()).unwrap();
    assert_eq!(first["idx"], 3);
    assert_eq!(first["t"].as_u64().unwrap() - first["chunk"]["t_emit"].as_u64().unwrap(), 3);

    // Short horizon: njr invalid, acm_neg valid.
    let sh = last_chunk_valid("short_horizon_h3");
    assert_eq!(sh & njr, 0);
    assert_ne!(sh & Feat::AcmNeg.bit(), 0);
    assert_eq!(sh & (tce | acc), 0);

    // The raw-vs-projected pair is declared and its expected tce values differ.
    let proj = by_name("projected_prev_spike");
    assert_eq!(proj["tce_differs_from"], "raw_prev_spike");
    let tce_of = |n: &str| -> f64 {
        let c = by_name(n);
        let t = c["ticks"].as_array().unwrap().iter().rfind(|t| !t["chunk"].is_null()).unwrap();
        f64s(&t["expect"]["f"])[Feat::Tce as usize]
    };
    assert!((tce_of("raw_prev_spike") - tce_of("projected_prev_spike")).abs() > 1e-6);
    // Both first chunks share seq/t_emit; only the data differ (the leashed spike).
    let raw0 = &by_name("raw_prev_spike")["ticks"][0]["chunk"];
    let proj0 = &proj["ticks"][0]["chunk"];
    assert_eq!(raw0["t_emit"], proj0["t_emit"]);
    assert_ne!(f64s(&raw0["data"]), f64s(&proj0["data"]));

    // The hold behaviour is in the fixture: a non-boundary tick repeats the boundary features of its chunk.
    let s = by_name("sync_d0_pair_then_hold");
    let ticks = s["ticks"].as_array().unwrap();
    let f8 = f64s(&ticks[8]["expect"]["f"]);
    let f9 = f64s(&ticks[9]["expect"]["f"]);
    for j in 0..NFEAT {
        if Feat::CHUNK_BOUNDARY_MASK & (1 << j) != 0 {
            assert_eq!(f8[j], f9[j], "hold of {}", Feat::NAMES[j]);
        }
    }
    // ... and the ext channels come and go with the observation.
    assert_eq!(u32::try_from(ticks[0]["expect"]["valid"].as_u64().unwrap()).unwrap() & 0xf00, 0);
    assert_eq!(u32::try_from(ticks[5]["expect"]["valid"].as_u64().unwrap()).unwrap() & 0xf00, 0x300);
    assert_eq!(u32::try_from(ticks[20]["expect"]["valid"].as_u64().unwrap()).unwrap() & 0xf00, 0xf00);
}

#[test]
fn no_fitted_limit_identifier_in_code() {
    // The invariant of ARCHITECTURE 5.3: no Tier-1 feature reads the four fitted Tier-0 limits (nor the operator
    // list). Comments may mention them (the frozen doc comment does); code may not.
    let banned = ["v_max", "a_max", "j_max", "reach_max", "operators", "step_max"];
    let is_ident = |c: char| c.is_ascii_alphanumeric() || c == '_';
    for (ln, line) in SRC.lines().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        for word in banned {
            let mut from = 0;
            while let Some(p) = code[from..].find(word) {
                let start = from + p;
                let end = start + word.len();
                let before = code[..start].chars().next_back().map(is_ident).unwrap_or(false);
                let after = code[end..].chars().next().map(is_ident).unwrap_or(false);
                assert!(before || after, "tier1.rs line {}: code reads `{word}`: {}", ln + 1, line.trim());
                from = end;
            }
        }
    }
}

#[test]
fn default_and_reset() {
    let mut rt = Tier1Rt::default();
    assert!(!rt.have_prev);
    assert!(rt.trail.is_empty());
    assert_eq!(rt.held_valid, 0);
    assert_eq!(rt.held, [0.0; NFEAT]);
    assert!(rt.prev.view().is_none());
    for i in 0..5 {
        rt.trail.push(&[f64::from(i), 0.0], 2);
    }
    rt.held_valid = Feat::CHUNK_BOUNDARY_MASK;
    rt.held[Feat::Reach as usize] = 0.5;
    rt.have_prev = true;
    rt.reset();
    assert!(!rt.have_prev);
    assert!(rt.trail.is_empty());
    assert_eq!(rt.held_valid, 0);
    assert_eq!(rt.held, [0.0; NFEAT]);
    assert!(rt.prev.view().is_none());
}

#[test]
fn per_tick_features_without_any_chunk() {
    // A straight-line trail: after 20 samples path_ineff is exactly 0 (net == path == 190) and
    // stall = 1 - min(1, 190 / (v_ref * W * dt)) with v_ref * W * dt = 0.05 * 256 * 20 = 256.
    let cfg = base_cfg();
    let mut rt = Tier1Rt::new();
    let mut sc = Scores::default();
    for t in 0..25u32 {
        let pos = [100.0 + 10.0 * f64::from(t), 200.0];
        let obs = ObsView { t, pos: &pos, vel: None, aux: &[], ext: &[] };
        rt.trail.push(&pos, cfg.pos_dim);
        features(&cfg, &mut rt, obs, None, 0, &mut sc);
        assert!(!rt.have_prev);
        assert_eq!(rt.held_valid, 0);
        if (t as usize) + 1 < PE_WINDOW {
            assert_eq!(sc.valid, 0, "t = {t}");
            assert_eq!(sc.f, [0.0; NFEAT]);
        } else {
            assert_eq!(sc.valid, Feat::PathIneff.bit() | Feat::Stall.bit(), "t = {t}");
            assert_eq!(sc.f[Feat::PathIneff as usize], 0.0);
            let v_ref = STALL_VREF_FRAC * cfg.norm_scale_iso / cfg.dt;
            let expect = 1.0 - 190.0 / (v_ref * 20.0 * cfg.dt);
            assert!(
                close(sc.f[Feat::Stall as usize], expect),
                "stall {} vs {expect}",
                sc.f[Feat::Stall as usize]
            );
            assert!(close(sc.f[Feat::Stall as usize], 1.0 - 190.0 / 256.0));
            for j in 0..NFEAT {
                if j != Feat::PathIneff as usize && j != Feat::Stall as usize {
                    assert_eq!(sc.f[j], 0.0);
                }
            }
        }
    }
    // A motionless trail: path 0 -> path_ineff = 1 - 0/eps = 1, stall = 1.
    let mut rt = Tier1Rt::new();
    for t in 0..PE_WINDOW {
        let pos = [300.0, 300.0];
        let obs = ObsView { t: t as u32, pos: &pos, vel: None, aux: &[], ext: &[] };
        rt.trail.push(&pos, cfg.pos_dim);
        features(&cfg, &mut rt, obs, None, 0, &mut sc);
    }
    assert_eq!(sc.f[Feat::PathIneff as usize], 1.0);
    assert_eq!(sc.f[Feat::Stall as usize], 1.0);
    // A fast straight trail saturates stall at 0: 19 steps of 20 px = 380 > 256.
    let mut rt = Tier1Rt::new();
    for t in 0..PE_WINDOW {
        let pos = [20.0 * t as f64, 50.0];
        let obs = ObsView { t: t as u32, pos: &pos, vel: None, aux: &[], ext: &[] };
        rt.trail.push(&pos, cfg.pos_dim);
        features(&cfg, &mut rt, obs, None, 0, &mut sc);
    }
    assert_eq!(sc.f[Feat::Stall as usize], 0.0);
    assert_eq!(sc.f[Feat::PathIneff as usize], 0.0);
}

#[test]
fn ext_validity_follows_the_slice_length() {
    let cfg = base_cfg();
    let mut rt = Tier1Rt::new();
    let mut sc = Scores::default();
    let pos = [256.0, 256.0];
    let ext_all = [0.5, -0.25, 3.0, -7.5];
    for n in 0..=4usize {
        let obs = ObsView { t: 0, pos: &pos, vel: None, aux: &[], ext: &ext_all[..n] };
        rt.trail.push(&pos, cfg.pos_dim);
        features(&cfg, &mut rt, obs, None, 0, &mut sc);
        let expect_valid = ((1u32 << n) - 1) << Feat::Ext0 as u32;
        assert_eq!(sc.valid, expect_valid, "ext len {n}");
        for (k, x) in ext_all.iter().enumerate() {
            let j = Feat::Ext0 as usize + k;
            if k < n {
                assert_eq!(sc.f[j], *x);
            } else {
                assert_eq!(sc.f[j], 0.0);
            }
        }
    }
}
