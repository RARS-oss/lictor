// SPDX-License-Identifier: MIT
//! OWNER: WP-3. Stub written by WP-0; the hand-written `Default` below is the frozen specification.
//! Per-episode counters: Copy, fixed arrays; converted to receipt `VerdictCounts` by lictor-receipt.

use lictor_core::{FuseState, ReasonCode, NFEAT};

#[derive(Clone, Copy, Debug)]
pub struct Tally {
    pub ticks: u32,
    pub nominal: u32,
    pub watching: u32,
    pub clamped: u32,
    pub braking: u32,
    pub held: u32,
    pub escalated: u32,
    pub fault: u32,
    pub terminated: u32,
    pub substituted: u32,
    pub chunks_seen: u32,
    pub chunks_rejected: u32,
    pub clamps: u32,
    pub holds: u32,
    pub rearms: u32,
    pub escalations: u32,
    pub trips_by_bit: [u32; 16],
    pub fired_by_feat: [u32; NFEAT],
    pub first_trip_tick: Option<u32>,
    pub first_trip_reason: Option<ReasonCode>,
    pub first_stop_tick: Option<u32>,
    pub handoff_tick: Option<u32>,
    pub violations_reached_env: u32,
    pub max_s: f64,
    pub max_z: [f64; NFEAT],
    pub terminal_state: FuseState,
}

/// Hand-written (NOT derived): zeros/None everywhere, `terminal_state = Idle`, `max_s = f64::NEG_INFINITY`,
/// `max_z = [f64::NEG_INFINITY; NFEAT]` -- `s` is NEG_INFINITY while nothing is valid or Tier 1 is disarmed, and 0.0 would
/// bias the per-episode `max_s` that feeds ROC-AUC. F64Hex encodes +-inf exactly; the wire encodes it as `null`.
impl Default for Tally {
    fn default() -> Self {
        Self {
            ticks: 0,
            nominal: 0,
            watching: 0,
            clamped: 0,
            braking: 0,
            held: 0,
            escalated: 0,
            fault: 0,
            terminated: 0,
            substituted: 0,
            chunks_seen: 0,
            chunks_rejected: 0,
            clamps: 0,
            holds: 0,
            rearms: 0,
            escalations: 0,
            trips_by_bit: [0; 16],
            fired_by_feat: [0; NFEAT],
            first_trip_tick: None,
            first_trip_reason: None,
            first_stop_tick: None,
            handoff_tick: None,
            violations_reached_env: 0,
            max_s: f64::NEG_INFINITY,
            max_z: [f64::NEG_INFINITY; NFEAT],
            terminal_state: FuseState::Idle,
        }
    }
}
