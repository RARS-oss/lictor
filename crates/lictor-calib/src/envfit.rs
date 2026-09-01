// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! Envelope fit: Tier-0 limits from the calibration arm's successful traces (quantile x slack); embodiment untouched.

use lictor_core::SafetyEnvelope;

use crate::traces::Trace;

pub struct FitOpts {
    pub quantile: f64,
    pub slack: f64,
}

/// `operators` replaces base.operators (the oracle envelope); embodiment untouched. Returns (envelope, report md).
pub fn fit_envelope(
    _base: &SafetyEnvelope,
    _traces: &[Trace],
    _opts: &FitOpts,
    _operators: &[String],
) -> anyhow::Result<(SafetyEnvelope, String)> {
    todo!("WP-5")
}
