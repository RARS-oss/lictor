// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! The VERDICT chain (replayable) and the TIMING chain (honestly not).

use lictor_canon::{F64Array, F64Hex};
use lictor_core::{ActionSource, FuseState, ReasonCode, SafetyVerdict, Status};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TickEvent {
    pub seq: u32,
    pub t: u32,
    pub state: FuseState,
    pub prev_state: FuseState,
    pub status: Status,
    pub trips: u32,
    pub action_src: ActionSource,
    pub substituted: bool,
    pub clamped_dims: u32,
    /// [action_dim]
    pub action: F64Array,
    /// [NFEAT] raw features (what `lictor sweep` re-scores)
    pub f: F64Array,
    /// [NFEAT]
    pub z: F64Array,
    pub valid: u32,
    pub fired: u32,
    pub s: F64Hex,
    pub tau: F64Hex,
    pub window_hits: u8,
    pub brake_margin: F64Hex,
    pub reason: ReasonCode,
    pub handoff_seq: Option<u32>,
    pub violation_reached_env: bool,
    pub prev: String,
    /// hash = sha256(canon(self with hash = "")) -- see docs/receipt-schema.md
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimingEvent {
    pub seq: u32,
    pub decide_ns: u64,
    pub io_ns: u64,
    pub prev: String,
    pub hash: String,
}

pub fn tick_event(_prev: &str, _v: &SafetyVerdict) -> TickEvent {
    todo!("WP-4")
}

pub fn timing_event(_prev: &str, _seq: u32, _decide_ns: u64, _io_ns: u64) -> TimingEvent {
    todo!("WP-4")
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChainReport {
    pub ok: bool,
    pub break_at: Option<u32>,
    pub head: String,
    pub n: u32,
}

pub fn verify_tick_chain(_events: &[TickEvent], _genesis_prev: &str) -> ChainReport {
    todo!("WP-4")
}

pub fn verify_timing_chain(_events: &[TimingEvent]) -> ChainReport {
    todo!("WP-4")
}
