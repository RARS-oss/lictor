// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; the payload structs are frozen (fields and `deny_unknown_fields`).
//! serde types for lictor-wire/v1 (see the WIRE PROTOCOL section of docs/ARCHITECTURE.md).
//!
//! `deny_unknown_fields` on the internally-tagged `Request` enum does NOT reject unknown keys inside the variant payloads,
//! so it is on EVERY payload struct below as well (fail-closed: an unknown key anywhere in a request is a schema fault).
//! Every real-valued REQUEST field is Option<f64> (JSON null == non-finite -> NaN -> GUARD NONFINITE). Real-valued RESPONSE
//! fields are f64; serde_json emits `null` for a non-finite f64, and exactly two response fields can be non-finite BY DESIGN:
//! `scores.s` (NEG_INFINITY while no gate term is fully valid or Tier 1 is disarmed) and `tau` (+INFINITY when disarmed or
//! k > n), same for `hello_ok.calibration.tau`. The Python client maps null -> -inf / +inf for those keys ONLY.

use std::collections::BTreeMap;

use lictor_core::{ActionSource, CalMethod, ExecMode, FuseMode, FuseState, ReasonCode, Status};
use lictor_receipt::{AckToken, HandoffRecord, VerdictCounts};

pub const PROTO: &str = "lictor-wire/v1";

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Hello(HelloReq),
    EpisodeBegin(EpisodeBeginReq),
    Tick(TickReq),
    EpisodeEnd(EpisodeEndReq),
    Bye(ByeReq),
}

#[derive(Debug, Clone, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response {
    HelloOk(HelloOk),
    EpisodeOk(EpisodeOk),
    Verdict(Box<VerdictMsg>),
    EpisodeReceipt(EpisodeReceiptMsg),
    ByeOk { id: u64 },
    Error(ErrorMsg),
}

// ---- request payloads (keys == the WIRE PROTOCOL JSON)

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HelloReq {
    pub id: u64,
    pub proto: String,
    pub client: String,
    pub mode: FuseMode,
    pub embodiment_id: String,
    pub action_dim: u16,
    pub pos_dim: u16,
    pub horizon: u16,
    pub exec_steps: u16,
    pub envelope_digest: String,
    pub calibration_digest: Option<String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunMsg {
    pub run_id: String,
    pub arm_id: String,
    pub episode_index: u32,
    pub seed: u64,
    pub seed_pool: String,
    pub init_state_digest: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BudgetMsg {
    pub delay_steps: u16,
    pub tick_ms: u32,
    pub exec_mode: ExecMode,
    pub stitch: String,
    pub on_escalate: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BindingMsg {
    pub env: BTreeMap<String, String>,
    pub policy: BTreeMap<String, String>,
    pub host: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FaultInjectionMsg {
    pub kind: String,
    pub params: BTreeMap<String, String>,
    pub stream_seed: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeBeginReq {
    pub id: u64,
    pub run: RunMsg,
    pub budget: BudgetMsg,
    pub binding: BindingMsg,
    pub fault_injection: Option<FaultInjectionMsg>,
    /// host-declared content hashes: repo-relative path -> sha256 (absent == empty)
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ObsMsg {
    pub pos: Vec<Option<f64>>,
    pub vel: Option<Vec<Option<f64>>>,
    pub aux: Vec<Option<f64>>,
    pub ext: Vec<Option<f64>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkMsg {
    pub seq: u32,
    pub t_emit: u32,
    pub h: u16,
    pub d: u16,
    pub exec: u16,
    pub a: Vec<Vec<Option<f64>>>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TickReq {
    pub id: u64,
    pub t: u32,
    pub idx: u16,
    pub missed_ticks: u8,
    pub obs: ObsMsg,
    pub chunk: Option<ChunkMsg>,
    pub ack: Option<AckToken>,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeMsg {
    pub steps: u32,
    pub success: bool,
    pub terminated: bool,
    pub truncated: bool,
    pub max_coverage: Option<f64>,
    pub final_coverage: Option<f64>,
    pub reward_sum: Option<f64>,
    pub ended_by: String,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EpisodeEndReq {
    pub id: u64,
    pub t: u32,
    pub outcome: OutcomeMsg,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ByeReq {
    pub id: u64,
}

// ---- response payloads

#[derive(Debug, Clone, serde::Serialize)]
pub struct CalibrationInfo {
    pub method: CalMethod,
    pub alpha_num: u32,
    pub alpha_den: u32,
    pub n_calib: u32,
    pub tau: f64,
    pub kn: [u8; 2],
    pub gate: Vec<String>,
    pub digest: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct HelloOk {
    pub id: u64,
    pub proto: String,
    pub lictor: String,
    pub git: String,
    pub lictor_sha256: String,
    pub envelope_digest: String,
    pub embodiment_digest: String,
    pub calibration_digest: Option<String>,
    pub pubkey: String,
    pub ephemeral_key: bool,
    pub mode: FuseMode,
    pub tier0_armed: Vec<String>,
    pub tier1_armed: bool,
    pub calibration: Option<CalibrationInfo>,
    pub features: Vec<String>,
    pub trip_names: Vec<String>,
    pub latency_label: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EpisodeOk {
    pub id: u64,
    pub state: FuseState,
    pub seq: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ScoresMsg {
    pub f: Vec<f64>,
    pub z: Vec<f64>,
    pub s: f64,
    pub valid: u32,
    pub fired: u32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct VerdictMsg {
    pub id: u64,
    pub t: u32,
    pub seq: u32,
    pub status: Status,
    pub state: FuseState,
    pub prev_state: FuseState,
    pub trips: Vec<String>,
    pub trip_mask: u32,
    pub action: Vec<f64>,
    pub action_src: ActionSource,
    pub substituted: bool,
    pub clamped_dims: u32,
    pub scores: ScoresMsg,
    pub tau: f64,
    pub window_hits: u8,
    pub brake_margin: f64,
    pub reason: ReasonCode,
    pub reason_text: String,
    pub handoff: Option<HandoffRecord>,
    pub ack_result: Option<String>,
    pub violation_reached_env: bool,
    pub verdict_ns: u64,
    pub chain: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EpisodeReceiptMsg {
    pub id: u64,
    pub receipt_path: String,
    pub ticks_path: String,
    pub body_digest: String,
    pub verdict_chain_head: String,
    pub timing_chain_head: String,
    pub fuse_ok: bool,
    pub fuse_notes: Vec<String>,
    pub counts: VerdictCounts,
    pub ledger_seq: u32,
    pub ledger_head: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ErrorMsg {
    pub id: u64,
    pub code: String,
    pub message: String,
    pub fatal: bool,
}
