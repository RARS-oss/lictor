// SPDX-License-Identifier: MIT
//! OWNER: WP-2. Stub written by WP-0; replace the bodies, keep the signatures.
//! The deterministic runtime half of split-conformal: standardise, aggregate over the gate, strict trip.

use lictor_core::{CalibrationC, Scores};

/// z_j = (f_j - center[b][j]) / scale[b][j] for masked & valid j; b = cal.bin(t). Others 0.0.
pub fn standardise(_cal: &CalibrationC, _t: u32, _sc: &mut Scores) {
    todo!("WP-2")
}

/// s = max over terms of min over the term's channels (all masked & valid), NEG_INFINITY if no term is fully valid.
/// Also sets sc.fired (z_j > tau per channel, informational). Returns s.
pub fn aggregate(_cal: &CalibrationC, _sc: &mut Scores) -> f64 {
    todo!("WP-2")
}

/// STRICT: s > tau.
#[inline]
pub fn trip(_cal: &CalibrationC, _sc: &Scores) -> bool {
    todo!("WP-2")
}
