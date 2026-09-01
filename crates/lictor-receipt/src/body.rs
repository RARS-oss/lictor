// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! The receipt body and its bindings.

use std::collections::BTreeMap;

use lictor_canon::{F64Array, F64Hex};
use lictor_core::{ExecMode, FuseMode, FuseState};

use crate::handoff::HandoffRecord;
use crate::tick::TickEvent;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunBinding {
    pub run_id: String,
    pub arm_id: String,
    pub episode_index: u32,
    pub seed: u64,
    pub seed_pool: String,
    pub init_state_digest: String,
    pub env: BTreeMap<String, String>,
    pub policy: BTreeMap<String, String>,
    pub host: BTreeMap<String, String>,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BudgetBinding {
    pub mode: FuseMode,
    pub delay_steps: u16,
    pub tick_ms: u32,
    pub exec_mode: ExecMode,
    pub stitch: String,
    pub on_escalate: String,
    pub tier0_armed: Vec<String>,
    pub tier1_armed: bool,
    pub gate: Vec<String>,
    pub alpha_num: u32,
    pub alpha_den: u32,
    pub kn: [u8; 2],
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FaultBinding {
    pub kind: String,
    pub params: BTreeMap<String, String>,
    pub stream_seed: u64,
}

#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct VerdictCounts {
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
    pub trips_by_bit: Vec<u32>,
    pub fired_by_feat: Vec<u32>,
    pub first_trip_tick: Option<u32>,
    pub first_trip_reason: Option<String>,
    pub first_stop_tick: Option<u32>,
    pub handoff_tick: Option<u32>,
    pub violations_reached_env: u32,
    pub terminal_state: FuseState,
}

impl From<&lictor_fuse::Tally> for VerdictCounts {
    fn from(_t: &lictor_fuse::Tally) -> Self {
        todo!("WP-4")
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeOutcome {
    pub steps: u32,
    pub success: bool,
    pub terminated: bool,
    pub truncated: bool,
    pub max_coverage: F64Hex,
    pub final_coverage: F64Hex,
    pub reward_sum: F64Hex,
    /// "success"|"truncated"|"escalation_terminate"|"fault"|"fuse_crash"|"abort"|"retune"
    pub ended_by: String,
    pub max_s: F64Hex,
    pub max_z: F64Array,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LatencySummary {
    pub n: u32,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
    pub p999_ns: u64,
    pub max_ns: u64,
    pub label: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReceiptBody {
    pub schema: String,
    pub canonical: String,
    pub created_epoch: u64,
    pub lictor_version: String,
    pub lictor_git: String,
    pub lictor_sha256: String,
    /// the `hello.client` string (host-declared)
    pub client: String,
    pub run: RunBinding,
    pub budget: BudgetBinding,
    pub fault_injection: Option<FaultBinding>,
    /// floatify(envelope) -- the full config, float-free
    pub envelope: serde_json::Value,
    pub envelope_digest: String,
    pub calibration_digest: Option<String>,
    /// content-addressed: key -> sha256. Host-declared entries (from episode_begin.inputs) use repo-relative paths;
    /// fuse-computed entries use the keys "lictor:bin", "lictor:envelope", "lictor:calibration" (absent when no calibration)
    pub inputs: BTreeMap<String, String>,
    pub counts: VerdictCounts,
    pub outcome: EpisodeOutcome,
    pub handoffs: Vec<HandoffRecord>,
    pub verdict_events: u32,
    pub verdict_chain_head: String,
    pub timing_events: u32,
    pub timing_chain_head: String,
    pub latency: LatencySummary,
    /// "tail32"|"all"|"none"
    pub ticks_policy: String,
    /// embedded per ticks_policy; the full stream is in the ticks file
    #[serde(default)]
    pub ticks: Vec<TickEvent>,
    pub fuse_ok: bool,
    pub fuse_notes: Vec<String>,
    pub ledger_prev: Option<String>,
}

impl ReceiptBody {
    pub fn canonical(&self) -> Result<Vec<u8>, lictor_canon::CanonError> {
        todo!("WP-4")
    }

    pub fn digest_hex(&self) -> Result<String, lictor_canon::CanonError> {
        todo!("WP-4")
    }
}

/// The honest verdict on the RECORD (not on the run). intact != fuse_ok. `ephemeral_key`: the receipt was signed by a key generated
/// at `lictor serve` startup (no `--key`); `verify` recomputes it as `body.fuse_notes.iter().any(|n| n == "ephemeral signing key")`.
pub fn evaluate_fuse(
    _budget: &BudgetBinding,
    _counts: &VerdictCounts,
    _calibration_digest: Option<&str>,
    _ephemeral_key: bool,
) -> (bool, Vec<String>) {
    todo!("WP-4")
}
