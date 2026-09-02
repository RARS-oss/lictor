// SPDX-License-Identifier: MIT
//! The receipt body and its bindings.
//!
//! Everything under `run`, `budget.delay_steps`/`exec_mode`/`stitch`, `fault_injection`, the host part of `inputs`
//! and `outcome.success` is the HOST'S DECLARATION, signed by proxy; the fuse verifies mode, dims, digests, its
//! own binary hash and every tick (docs/ARCHITECTURE.md sec 8, trust model). `evaluate_fuse` is the honest verdict
//! on the RECORD: `intact != fuse_ok`.

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

/// The serde (snake_case) name of a ReasonCode, e.g. `brake_tier1_cp`.
pub fn reason_name(r: lictor_core::ReasonCode) -> String {
    match serde_json::to_value(r) {
        Ok(serde_json::Value::String(s)) => s,
        _ => String::new(),
    }
}

impl From<&lictor_fuse::Tally> for VerdictCounts {
    fn from(t: &lictor_fuse::Tally) -> Self {
        Self {
            ticks: t.ticks,
            nominal: t.nominal,
            watching: t.watching,
            clamped: t.clamped,
            braking: t.braking,
            held: t.held,
            escalated: t.escalated,
            fault: t.fault,
            terminated: t.terminated,
            substituted: t.substituted,
            chunks_seen: t.chunks_seen,
            chunks_rejected: t.chunks_rejected,
            clamps: t.clamps,
            holds: t.holds,
            rearms: t.rearms,
            escalations: t.escalations,
            trips_by_bit: t.trips_by_bit.to_vec(),
            fired_by_feat: t.fired_by_feat.to_vec(),
            first_trip_tick: t.first_trip_tick,
            first_trip_reason: t.first_trip_reason.map(reason_name),
            first_stop_tick: t.first_stop_tick,
            handoff_tick: t.handoff_tick,
            violations_reached_env: t.violations_reached_env,
            terminal_state: t.terminal_state,
        }
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
        lictor_canon::canon_of(self)
    }

    pub fn digest_hex(&self) -> Result<String, lictor_canon::CanonError> {
        Ok(lictor_canon::sha256_hex(&self.canonical()?))
    }
}

/// The `fuse_notes` entry a `lictor serve` started without `--key` writes; `verify` recomputes `ephemeral_key` from it.
pub const NOTE_EPHEMERAL_KEY: &str = "ephemeral signing key";
pub const NOTE_OBSERVE: &str =
    "the fuse observed but did not enforce -- this receipt does not attest protection";
pub const NOTE_TIER1_UNCALIBRATED: &str = "tier 1 armed without a calibration digest";
pub const NOTE_FAULT: &str = "fuse latched Fault";
/// `BudgetBinding.exec_mode` is not optional in the frozen type, so this note is unreachable from a well-typed
/// receipt; it is kept as the frozen grep target (see docs/receipt-schema.md, freeze notes).
pub const NOTE_DELAY_WITHOUT_EXEC: &str = "delay injected without an exec mode";

/// The honest verdict on the RECORD (not on the run). intact != fuse_ok. `ephemeral_key`: the receipt was signed by a key generated
/// at `lictor serve` startup (no `--key`); `verify` recomputes it as `body.fuse_notes.iter().any(|n| n == "ephemeral signing key")`.
pub fn evaluate_fuse(
    budget: &BudgetBinding,
    counts: &VerdictCounts,
    calibration_digest: Option<&str>,
    ephemeral_key: bool,
) -> (bool, Vec<String>) {
    let mut notes = Vec::new();
    if budget.mode == FuseMode::Observe {
        notes.push(NOTE_OBSERVE.to_string());
    }
    if counts.violations_reached_env > 0 {
        notes.push(format!("{} tier-0 violations reached the environment", counts.violations_reached_env));
    }
    if budget.tier1_armed && calibration_digest.map(str::is_empty).unwrap_or(true) {
        notes.push(NOTE_TIER1_UNCALIBRATED.to_string());
    }
    if counts.terminal_state == FuseState::Fault {
        notes.push(NOTE_FAULT.to_string());
    }
    if budget.delay_steps > 0 && exec_mode_missing(budget) {
        notes.push(NOTE_DELAY_WITHOUT_EXEC.to_string());
    }
    if ephemeral_key {
        notes.push(NOTE_EPHEMERAL_KEY.to_string());
    }
    (notes.is_empty(), notes)
}

/// The frozen `BudgetBinding.exec_mode: ExecMode` cannot be missing; a well-typed receipt always carries one.
fn exec_mode_missing(budget: &BudgetBinding) -> bool {
    !matches!(budget.exec_mode, ExecMode::Sync | ExecMode::Async)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> BudgetBinding {
        BudgetBinding {
            mode: FuseMode::Enforce,
            delay_steps: 0,
            tick_ms: 100,
            exec_mode: ExecMode::Sync,
            stitch: "drop".into(),
            on_escalate: "terminate_fail".into(),
            tier0_armed: vec!["workspace".into()],
            tier1_armed: true,
            gate: vec!["tce".into()],
            alpha_num: 5,
            alpha_den: 100,
            kn: [3, 5],
        }
    }

    #[test]
    fn evaluate_fuse_notes() {
        let counts = VerdictCounts { terminal_state: FuseState::Armed, ..Default::default() };
        assert_eq!(evaluate_fuse(&budget(), &counts, Some("e1b8"), false), (true, vec![]));
        let mut b = budget();
        b.mode = FuseMode::Observe;
        let (ok, notes) = evaluate_fuse(&b, &counts, Some("e1b8"), false);
        assert!(!ok);
        assert_eq!(notes, vec![NOTE_OBSERVE.to_string()]);
        let c = VerdictCounts {
            violations_reached_env: 137,
            terminal_state: FuseState::Fault,
            ..Default::default()
        };
        let (ok, notes) = evaluate_fuse(&budget(), &c, None, true);
        assert!(!ok);
        assert_eq!(
            notes,
            vec![
                "137 tier-0 violations reached the environment".to_string(),
                NOTE_TIER1_UNCALIBRATED.to_string(),
                NOTE_FAULT.to_string(),
                NOTE_EPHEMERAL_KEY.to_string(),
            ]
        );
        let mut b = budget();
        b.tier1_armed = false;
        assert!(evaluate_fuse(&b, &counts, None, false).0);
    }

    #[test]
    fn tally_conversion() {
        let mut trips_by_bit = [0u32; 16];
        trips_by_bit[10] = 1;
        let t = lictor_fuse::Tally {
            ticks: 300,
            first_trip_reason: Some(lictor_core::ReasonCode::BrakeTier1Cp),
            trips_by_bit,
            ..Default::default()
        };
        let c = VerdictCounts::from(&t);
        assert_eq!(c.ticks, 300);
        assert_eq!(c.first_trip_reason.as_deref(), Some("brake_tier1_cp"));
        assert_eq!(c.trips_by_bit.len(), 16);
        assert_eq!(c.fired_by_feat.len(), lictor_core::NFEAT);
        assert_eq!(c.terminal_state, FuseState::Idle);
    }
}
