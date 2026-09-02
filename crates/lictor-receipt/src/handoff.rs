// SPDX-License-Identifier: MIT
//! The human-escalation surface: the `HandoffRecord` the fuse emits when it enters Escalated and the
//! `AckToken` an operator signs to resolve it.
//!
//! `run_id` / `arm_id` / `episode_index` are digested, so an ack captured on one run/arm/episode does not verify
//! on a paired re-run of the same seed; `chain_at` is the verdict-chain head after the escalation tick, so the
//! operator signs the exact fuse history they were shown. Per-operator `last_nonce` lives in the Session across
//! episodes and is persisted per pubkey in `<out>/.lictor/verifier_nonce.json`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use lictor_canon::{canon_of, sha256_hex, F64Hex};
use lictor_core::{AckDecision, ReasonCode, VerifiedAck};

use crate::sign::{sign_bytes, verify_bytes, ReceiptError};
use crate::HANDOFF_SCHEMA;

/// The longest `note` an ack may carry (it is rendered on a terminal and bound into the receipt).
pub const MAX_NOTE_CHARS: usize = 200;

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

impl HandoffRecord {
    /// sha256(canon(self with digest = "", ack = None, resolved_tick = None, outcome = "")): the resolution
    /// fields are excluded so the digest the operator signed is stable after the ack is attached.
    pub fn compute_digest(&self) -> Result<String, ReceiptError> {
        let mut h = self.clone();
        h.digest = String::new();
        h.ack = None;
        h.resolved_tick = None;
        h.outcome = String::new();
        Ok(sha256_hex(&canon_of(&h)?))
    }

    /// Fill `digest` from the content (the constructor step after the fuse assembled the record).
    pub fn with_digest(mut self) -> Result<Self, ReceiptError> {
        self.digest = self.compute_digest()?;
        Ok(self)
    }
}

/// The signed fields of an ack, in the profile's canonical form.
#[derive(serde::Serialize)]
struct AckCore<'a> {
    schema: &'a str,
    handoff_digest: &'a str,
    decision: AckDecision,
    operator: &'a str,
    nonce: u64,
    note: &'a str,
}

/// canon({handoff_digest,decision,operator,nonce,note,schema})
pub fn ack_signing_bytes(t: &AckToken) -> Result<Vec<u8>, ReceiptError> {
    let core = AckCore {
        schema: &t.schema,
        handoff_digest: &t.handoff_digest,
        decision: t.decision,
        operator: &t.operator,
        nonce: t.nonce,
        note: &t.note,
    };
    Ok(canon_of(&core)?)
}

pub fn sign_ack(
    handoff_digest: &str,
    decision: AckDecision,
    nonce: u64,
    note: &str,
    seed: &[u8; 32],
) -> Result<AckToken, ReceiptError> {
    if note.chars().count() > MAX_NOTE_CHARS {
        return Err(ReceiptError::Key(format!("ack note longer than {MAX_NOTE_CHARS} characters")));
    }
    let mut t = AckToken {
        schema: HANDOFF_SCHEMA.to_string(),
        handoff_digest: handoff_digest.to_string(),
        decision,
        operator: crate::keys::pubkey_hex(seed),
        nonce,
        note: note.to_string(),
        sig: String::new(),
    };
    let bytes = ack_signing_bytes(&t)?;
    let (_, sig) = sign_bytes(seed, &bytes);
    t.sig = sig;
    Ok(t)
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

impl std::fmt::Display for AckError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            AckError::UnknownOperator => "operator pubkey is not in the envelope's operator list",
            AckError::BadSignature => "ack signature does not verify",
            AckError::WrongHandoff => "ack is for a different handoff digest",
            AckError::NonceReplay => "ack nonce is not greater than the operator's last nonce",
            AckError::NoPendingHandoff => "no handoff is pending",
            AckError::TooLongNote => "ack note is longer than 200 characters",
        })
    }
}

impl std::error::Error for AckError {}

/// Pure verification: operator listed in `operators` (slot = index), Ed25519 verify_strict, digest == pending, nonce > last_nonce[slot].
/// `last_nonce` lives in the Session ACROSS episodes (never reset per episode) and is persisted per pubkey in
/// `<out>/.lictor/verifier_nonce.json` when `--out` is set; without that file acks are replayable across processes (SECURITY.md).
pub fn verify_ack(
    t: &AckToken,
    operators: &[String],
    pending_digest: Option<&str>,
    pending_seq: u32,
    last_nonce: &[u64],
) -> Result<VerifiedAck, AckError> {
    let slot = operators.iter().position(|o| o == &t.operator).ok_or(AckError::UnknownOperator)?;
    if t.note.chars().count() > MAX_NOTE_CHARS {
        return Err(AckError::TooLongNote);
    }
    if t.schema != HANDOFF_SCHEMA {
        return Err(AckError::BadSignature);
    }
    let bytes = ack_signing_bytes(t).map_err(|_| AckError::BadSignature)?;
    verify_bytes(&t.operator, &t.sig, &bytes).map_err(|_| AckError::BadSignature)?;
    let pending = pending_digest.ok_or(AckError::NoPendingHandoff)?;
    if t.handoff_digest != pending {
        return Err(AckError::WrongHandoff);
    }
    let last = last_nonce.get(slot).copied().unwrap_or(0);
    if t.nonce <= last {
        return Err(AckError::NonceReplay);
    }
    let operator_slot = u8::try_from(slot).map_err(|_| AckError::UnknownOperator)?;
    Ok(VerifiedAck { decision: t.decision, operator_slot, nonce: t.nonce, handoff_seq: pending_seq })
}

/// pubkey -> last nonce; missing file == empty
pub fn load_nonces(path: &Path) -> Result<BTreeMap<String, u64>, ReceiptError> {
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let text =
        std::fs::read_to_string(path).map_err(|e| ReceiptError::Io(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text)
        .map_err(|e| ReceiptError::Io(format!("{}: not a nonce map: {e}", path.display())))
}

/// Pretty JSON with sorted keys (a BTreeMap serialises sorted); the parent directory is created.
pub fn save_nonces(path: &Path, m: &BTreeMap<String, u64>) -> Result<(), ReceiptError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let text = serde_json::to_string_pretty(m).map_err(|e| ReceiptError::Io(e.to_string()))?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(text.as_bytes())?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(())
}
