// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! Layer-A sweep: one point per (alpha, detector), scored offline on the calibration/eval traces.

use lictor_core::CalMethod;

use crate::file::CalibrationFile;
use crate::traces::Trace;

pub struct DetectorSpec {
    pub name: String,
    pub gate: Vec<String>,
    pub kn: [u8; 2],
    pub tier0: bool,
}

/// Exactly the sweep.jsonl keys in FILE FORMATS (fields are WP-5's to add).
pub struct SweepPoint {}

/// The scalar options of a sweep (WP-0 freeze amendment: bundled so `sweep` stays under clippy's argument limit).
pub struct SweepOpts {
    pub method: CalMethod,
    pub horizon_ticks: u32,
    pub eps_prog: f64,
}

/// `artefacts`: calibration.<alpha>.json files already produced by `lictor calibrate`; when an (alpha, gate, kn, method) matches one,
/// its center/scale/tau are used VERBATIM (`tau_source: "artefact"`) so the Layer-A point is comparable to the Layer-B arm;
/// otherwise a full fit with no holdout is done (`tau_source: "fit"`).
pub fn sweep(
    _calib: &[Trace],
    _eval: &[Trace],
    _alphas: &[(u32, u32)],
    _dets: &[DetectorSpec],
    _opts: &SweepOpts,
    _artefacts: &[CalibrationFile],
) -> Vec<SweepPoint> {
    todo!("WP-5")
}
