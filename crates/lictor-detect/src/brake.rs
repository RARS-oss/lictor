// SPDX-License-Identifier: MIT
//! Brake feasibility -- the checked stopping condition of ARCHITECTURE sec 5.2 -- and the brake/hold actions.
//!
//! `PdSecondOrder` is the exact forward rollout of the gym-pusht agent (`acc = k_p (a - p) - k_v v; v += acc dt;
//! p += v dt`, `substeps` per control step) over the committed rows `from..commit_steps` followed by
//! `brake_steps` control steps with the setpoint held at `clamp(p)`. Exactly
//! `(commit_steps - from + brake_steps) * substeps` iterations (none of the prefix when `from >= commit_steps`).
//! The closed-form kinds run `commit_steps - from` Euler steps of the commanded velocity and evaluate a stopping
//! ball (`d_stop(v) + ||v|| * react_ticks * dt`) at the current plant state and at the start of every committed
//! step. Everything is `+ - * / sqrt min max` through `lictor_core::fmath`, fixed accumulation order, fixed
//! arrays, no allocation. `tests/fixtures/brake/gen.py` is the reference reimplementation of both paths.

use lictor_core::{fmath, ActionKind, BrakeKind, ChunkView, FuseConfig, MAX_POS};

/// Denominator guard of the velocity ramp (`||v|| + EPS`).
const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, Default)]
pub struct BrakeOut {
    pub feasible: bool,
    pub margin: f64,
    pub stop_dist: f64,
}

/// The rolled plant: position, velocity and the number of live dims.
struct Plant {
    p: [f64; MAX_POS],
    v: [f64; MAX_POS],
    n: usize,
}

#[inline]
fn min_usize(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

/// Live dims: the configured plant dim, bounded by every slice actually handed in (schema violations are
/// raised by `decide()` before this runs; the bound only keeps a malformed call from panicking).
fn plant_dim(cfg: &FuseConfig, p0: &[f64], v0: &[f64], ch: ChunkView<'_>) -> usize {
    let n = min_usize(min_usize(cfg.pos_dim, cfg.dim), min_usize(p0.len(), v0.len()));
    min_usize(min_usize(n, ch.dim as usize), MAX_POS)
}

/// Braking feasibility of the committed prefix of `ch` starting at row `from`, from plant state (p0, v0).
/// `from` = the row executed THIS tick (`idx`): at a chunk boundary `from == idx` (0 for sync/freeze delivery, `d` for async drop).
/// `v0` = obs.vel when the manifest provides it (PushT), else the finite-difference v_hat.
/// PdSecondOrder: EXACTLY (commit_steps + brake_steps) * substeps iterations of the PD map.
/// Closed-form kinds: EXACTLY commit_steps iterations of Euler + stopping ball. No allocation.
pub fn brake_feasible(cfg: &FuseConfig, p0: &[f64], v0: &[f64], ch: ChunkView<'_>, from: usize) -> BrakeOut {
    let n = plant_dim(cfg, p0, v0, ch);
    let mut pl = Plant { p: [0.0; MAX_POS], v: [0.0; MAX_POS], n };
    pl.p[..n].copy_from_slice(&p0[..n]);
    pl.v[..n].copy_from_slice(&v0[..n]);
    let (margin, beyond) = match cfg.brake.kind {
        BrakeKind::PdSecondOrder => pd_rollout(cfg, &mut pl, ch, from),
        _ => closed_form(cfg, &mut pl, ch, from),
    };
    BrakeOut { feasible: margin >= 0.0, margin, stop_dist: fmath::dist(&pl.p, p0, n) + beyond }
}

/// One physics substep of the PD agent on every live dim (`a` may be shorter: the zip stops there).
#[inline]
fn pd_substep(k_p: f64, k_v: f64, dt: f64, a: &[f64], pl: &mut Plant) {
    let n = pl.n;
    for ((pc, vc), ac) in pl.p[..n].iter_mut().zip(pl.v[..n].iter_mut()).zip(a) {
        let acc = k_p * (*ac - *pc) - k_v * *vc;
        *vc += acc * dt;
        *pc += *vc * dt;
    }
}

/// min over dims of the distance inside the (margin-adjusted) box.
#[inline]
fn slack(cfg: &FuseConfig, pl: &Plant) -> f64 {
    let mut m = f64::INFINITY;
    for (c, pc) in pl.p[..pl.n].iter().enumerate() {
        m = fmath::min(m, fmath::min(*pc - cfg.box_lo[c], cfg.box_hi[c] - *pc));
    }
    m
}

/// min over dims of the distance inside the box after shrinking it by the stopping radius `r`.
#[inline]
fn slack_r(cfg: &FuseConfig, pl: &Plant, r: f64) -> f64 {
    let mut m = f64::INFINITY;
    for (c, pc) in pl.p[..pl.n].iter().enumerate() {
        m = fmath::min(m, fmath::min(*pc - r - cfg.box_lo[c], cfg.box_hi[c] - *pc - r));
    }
    m
}

/// Returns `(margin, 0.0)`; the plant ends where the rollout ends.
fn pd_rollout(cfg: &FuseConfig, pl: &mut Plant, ch: ChunkView<'_>, from: usize) -> (f64, f64) {
    let b = cfg.brake;
    let (k_p, k_v, dt) = (b.k_p, b.k_v, b.dt);
    let sub = b.substeps as usize;
    let commit = b.commit_steps as usize;
    let mut margin = f64::INFINITY;
    let mut i = from;
    while i < commit {
        let a = ch.action(i);
        for _ in 0..sub {
            pd_substep(k_p, k_v, dt, a, pl);
            margin = fmath::min(margin, slack(cfg, pl));
        }
        i += 1;
    }
    let mut a_hold = [0.0; MAX_POS];
    for (c, h) in a_hold[..pl.n].iter_mut().enumerate() {
        *h = fmath::clamp(pl.p[c], cfg.box_lo[c], cfg.box_hi[c]);
    }
    let tail = b.brake_steps as usize * sub;
    for _ in 0..tail {
        pd_substep(k_p, k_v, dt, &a_hold, pl);
        margin = fmath::min(margin, slack(cfg, pl));
    }
    (margin, 0.0)
}

/// Stopping radius from velocity `v`: `d_stop(v) + ||v|| * react_ticks * dt` (dt = the control period).
fn radius(cfg: &FuseConfig, v: &[f64], n: usize, react: f64) -> f64 {
    let nv = fmath::norm(v, n);
    let d_stop = match cfg.brake.kind {
        BrakeKind::FirstOrderDecay => nv / cfg.brake.k_v,
        BrakeKind::BoundedAccel | BrakeKind::ZeroVelocityHold => nv * nv / (2.0 * cfg.a_max),
        BrakeKind::JerkLimited => nv * nv / (2.0 * cfg.a_max) + nv * cfg.a_max / (2.0 * cfg.j_max),
        BrakeKind::PdSecondOrder => 0.0,
    };
    d_stop + nv * react
}

/// The velocity the plant is commanded to hold during a step with action row `a` (writes `pl.v`).
#[inline]
fn commanded_velocity(cfg: &FuseConfig, a: &[f64], pl: &mut Plant, g: f64) {
    let n = pl.n;
    match cfg.kind {
        ActionKind::JointVelocity => {
            for (vc, ac) in pl.v[..n].iter_mut().zip(a) {
                *vc = *ac;
            }
        }
        ActionKind::EeDelta => {
            for (vc, ac) in pl.v[..n].iter_mut().zip(a) {
                *vc = *ac / cfg.dt;
            }
        }
        ActionKind::EePosition | ActionKind::JointPosition | ActionKind::Other => {
            for ((vc, pc), ac) in pl.v[..n].iter_mut().zip(pl.p[..n].iter()).zip(a) {
                *vc = g * (*ac - *pc);
            }
        }
    }
}

/// Returns `(margin, R_end)`: the plant ends after the last Euler step and may still travel `R_end` while stopping.
fn closed_form(cfg: &FuseConfig, pl: &mut Plant, ch: ChunkView<'_>, from: usize) -> (f64, f64) {
    let b = cfg.brake;
    let dt = cfg.dt;
    let react = (b.react_ticks as f64) * dt;
    let g = b.k_p / b.k_v;
    let commit = b.commit_steps as usize;
    let n = pl.n;
    let mut margin = slack_r(cfg, pl, radius(cfg, &pl.v, n, react));
    let mut i = from;
    while i < commit {
        commanded_velocity(cfg, ch.action(i), pl, g);
        let r = radius(cfg, &pl.v, n, react);
        margin = fmath::min(margin, slack_r(cfg, pl, r));
        for (pc, vc) in pl.p[..n].iter_mut().zip(pl.v[..n].iter()) {
            *pc += *vc * dt;
        }
        i += 1;
    }
    (margin, radius(cfg, &pl.v, n, react))
}

/// The Braking-phase action: setpoint := clamp_box(p) (position kinds) or ramp-to-zero (velocity kinds).
///
/// Position kinds (`ee_position`, `joint_position`, `other`): `out[c] = clamp(p[c], lo[c], hi[c])`. Velocity
/// kinds: `v * max(0, 1 - a_max dt / (||v|| + eps))`; for `ee_delta` the ramped velocity is multiplied by `dt`
/// so the row stays a per-step displacement. Only `out[..dim]` is written.
pub fn brake_action(cfg: &FuseConfig, p: &[f64], v: &[f64], out: &mut [f64]) {
    let n = min_usize(min_usize(cfg.dim, out.len()), MAX_POS);
    match cfg.kind {
        ActionKind::JointVelocity | ActionKind::EeDelta => {
            let nv = fmath::norm(v, min_usize(n, v.len()));
            let f = fmath::max(0.0, 1.0 - cfg.a_max * cfg.dt / (nv + EPS));
            let scale = if cfg.kind == ActionKind::EeDelta { f * cfg.dt } else { f };
            for (c, o) in out[..n].iter_mut().enumerate() {
                *o = if c < v.len() { v[c] * scale } else { 0.0 };
            }
        }
        ActionKind::EePosition | ActionKind::JointPosition | ActionKind::Other => {
            for (c, o) in out[..n].iter_mut().enumerate() {
                let x = if c < p.len() { p[c] } else { 0.0 };
                *o = fmath::clamp(x, cfg.box_lo[c], cfg.box_hi[c]);
            }
        }
    }
}

/// The Held action: the latched hold setpoint (already clamped) copied into out.
///
/// Position kinds copy `p_latch` through a defensive `clamp`; velocity kinds hold by commanding zero.
pub fn hold_action(cfg: &FuseConfig, p_latch: &[f64], out: &mut [f64]) {
    let n = min_usize(min_usize(cfg.dim, out.len()), MAX_POS);
    match cfg.kind {
        ActionKind::JointVelocity | ActionKind::EeDelta => {
            for o in out[..n].iter_mut() {
                *o = 0.0;
            }
        }
        ActionKind::EePosition | ActionKind::JointPosition | ActionKind::Other => {
            for (c, o) in out[..n].iter_mut().enumerate() {
                let x = if c < p_latch.len() { p_latch[c] } else { 0.0 };
                *o = fmath::clamp(x, cfg.box_lo[c], cfg.box_hi[c]);
            }
        }
    }
}
