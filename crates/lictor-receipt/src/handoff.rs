// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! The human-escalation surface: HandoffRecord, AckToken, ack signing and pure verification, persisted nonces.

use std::collections::BTreeMap;

use lictor_canon::F64Hex;
use lictor_core::{AckDecision, ReasonCode, VerifiedAck};

use crate::sign::ReceiptError;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AckToken {
    pub schema: String,
    pub handoff_digest: String,
    pub decision: AckDecision,
    pub operator: String,
    pub nonce: u64,
    pub note: String,
    pub sig: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HandoffRecord {
    pub schema: String,
    /// replay-binding: a captured ack from another run/arm/episode cannot verify
    pub run_id: String,
    pub arm_id: String,
    pub episode_index: u32,
    pub seq: u32,
    pub tick: u32,
    pub reason: ReasonCode,
    pub reason_text: String,
    pub reasons: Vec<String>,
    pub trips: u32,
    pub fired: u32,
    pub window_hits: u8,
    pub top_z: Vec<(String, F64Hex)>,
    /// == the `hash` of the TickEvent of the tick on which Escalated was entered (the verdict-chain head after that tick)
    pub chain_at: String,
    pub envelope_digest: String,
    pub calibration_digest: Option<String>,
    /// sha256(canon(self with digest="", ack=None, resolved_tick=None, outcome="")) -- WHAT THE OPERATOR SIGNS
    pub digest: String,
    pub ack: Option<AckToken>,
    pub resolved_tick: Option<u32>,
    /// "resumed"|"aborted"|"retune"|"timeout"|"unacked"
    pub outcome: String,
}

/// canon({handoff_digest,decision,operator,nonce,note,schema})
pub fn ack_signing_bytes(_t: &AckToken) -> Result<Vec<u8>, ReceiptError> {
    todo!("WP-4")
}

pub fn sign_ack(
    _handoff_digest: &str,
    _decision: AckDecision,
    _nonce: u64,
    _note: &str,
    _seed: &[u8; 32],
) -> Result<AckToken, ReceiptError> {
    todo!("WP-4")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AckError {
    UnknownOperator,
    BadSignature,
    WrongHandoff,
    NonceReplay,
    NoPendingHandoff,
    TooLongNote,
}

/// Pure verification: operator listed in `operators` (slot = index), Ed25519 verify_strict, digest == pending, nonce > last_nonce[slot].
/// `last_nonce` lives in the Session ACROSS episodes (never reset per episode) and is persisted per pubkey in
/// `<out>/.lictor/verifier_nonce.json` when `--out` is set; without that file acks are replayable across processes (SECURITY.md).
pub fn verify_ack(
    _t: &AckToken,
    _operators: &[String],
    _pending_digest: Option<&str>,
    _pending_seq: u32,
    _last_nonce: &[u64],
) -> Result<VerifiedAck, AckError> {
    todo!("WP-4")
}

/// pubkey -> last nonce; missing file == empty
pub fn load_nonces(_path: &std::path::Path) -> Result<BTreeMap<String, u64>, ReceiptError> {
    todo!("WP-4")
}

pub fn save_nonces(_path: &std::path::Path, _m: &BTreeMap<String, u64>) -> Result<(), ReceiptError> {
    todo!("WP-4")
}
