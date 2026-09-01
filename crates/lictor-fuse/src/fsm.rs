// SPDX-License-Identifier: MIT
//! OWNER: WP-3. Stub written by WP-0; replace the bodies, keep the signatures.
//! The pure escalation state machine (docs/ARCHITECTURE.md sec 6 table; rows evaluated top-down, first match wins).

use lictor_core::{FuseConfig, FuseState, ReasonCode, VerifiedAck};

use crate::fuse::FuseRt;

#[derive(Clone, Copy, Debug)]
pub struct FsmInput {
    pub trips: u32,
    pub predictive: bool,
    pub warn: bool,
    pub soft_clampable: bool,
    pub stopped: bool,
    pub chunk_boundary: bool,
    pub chunk_ok: bool,
    pub ack: Option<VerifiedAck>,
}

/// Pure transition per the ARCHITECTURE table (rows evaluated top-down, first match wins). Updates counters in `rt`,
/// returns (new_state, extra_trips, reason).
pub fn next(_cfg: &FuseConfig, _rt: &mut FuseRt, _inp: FsmInput) -> (FuseState, u32, ReasonCode) {
    todo!("WP-3")
}
