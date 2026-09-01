// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! Split-conformal calibration (docs/ARCHITECTURE.md sec 7 steps 2-9).

use lictor_core::{CalMethod, CalibrationC, SafetyEnvelope, NFEAT, T_GRID};

use crate::file::CalibrationFile;
use crate::traces::Trace;

pub struct CalibrateOpts {
    pub alpha_num: u32,
    pub alpha_den: u32,
    pub method: CalMethod,
    pub gate: Vec<String>,
    pub holdout_num: u32,
    pub holdout_den: u32,
    pub kn: [u8; 2],
    pub warn_margin: f64,
    pub n_calib_cap: Option<u32>,
    pub horizon_ticks: u32,
    /// 2: center/scale AND tau on the fit subset (bound approximate); 3: center/scale on A (40 %), tau on B (30 %),
    /// holdout C (30 %) (bound exact at K = 1)
    pub split: u8,
}

pub fn robust_center_scale(
    _fit: &[Trace],
    _method: CalMethod,
    _horizon_ticks: u32,
) -> ([[f64; NFEAT]; T_GRID], [[f64; NFEAT]; T_GRID], u16) {
    todo!("WP-5")
}

pub fn episode_max_s(_tr: &Trace, _cal: &CalibrationC) -> f64 {
    todo!("WP-5")
}

/// (tau, note); +INF when k > n
pub fn split_quantile(_sorted_max_s: &[f64], _alpha_num: u32, _alpha_den: u32) -> (f64, Option<String>) {
    todo!("WP-5")
}

/// binds envelope.digest_hex() AND envelope.embodiment_digest()
pub fn calibrate(
    _traces: &[Trace],
    _opts: &CalibrateOpts,
    _envelope: &SafetyEnvelope,
    _policy_digest: &str,
    _source_run: &str,
) -> anyhow::Result<CalibrationFile> {
    todo!("WP-5")
}
