// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; replace the bodies, keep the signatures.
//! Brake feasibility (exact PD rollout for `PdSecondOrder`; closed forms otherwise) and the brake/hold actions.

use lictor_core::{ChunkView, FuseConfig};

#[derive(Clone, Copy, Debug, Default)]
pub struct BrakeOut {
    pub feasible: bool,
    pub margin: f64,
    pub stop_dist: f64,
}

/// Braking feasibility of the committed prefix of `ch` starting at row `from`, from plant state (p0, v0).
/// `from` = the row executed THIS tick (`idx`): at a chunk boundary `from == idx` (0 for sync/freeze delivery, `d` for async drop).
/// `v0` = obs.vel when the manifest provides it (PushT), else the finite-difference v_hat.
/// PdSecondOrder: EXACTLY (commit_steps + brake_steps) * substeps iterations of the PD map.
/// Closed-form kinds: EXACTLY commit_steps iterations of Euler + stopping ball. No allocation.
pub fn brake_feasible(
    _cfg: &FuseConfig,
    _p0: &[f64],
    _v0: &[f64],
    _ch: ChunkView<'_>,
    _from: usize,
) -> BrakeOut {
    todo!("WP-1")
}

/// The Braking-phase action: setpoint := clamp_box(p) (position kinds) or ramp-to-zero (velocity kinds).
pub fn brake_action(_cfg: &FuseConfig, _p: &[f64], _v: &[f64], _out: &mut [f64]) {
    todo!("WP-1")
}

/// The Held action: the latched hold setpoint (already clamped) copied into out.
pub fn hold_action(_cfg: &FuseConfig, _p_latch: &[f64], _out: &mut [f64]) {
    todo!("WP-1")
}
