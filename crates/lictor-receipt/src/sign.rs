// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! Ed25519 signing and verification of receipts; ticks-file verification.

use crate::body::ReceiptBody;
use crate::tick::{ChainReport, TickEvent};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedReceipt {
    pub body: ReceiptBody,
    pub body_digest: String,
    pub pubkey: String,
    pub sig: String,
}

pub fn sign(_body: ReceiptBody, _seed: &[u8; 32]) -> Result<SignedReceipt, ReceiptError> {
    todo!("WP-4")
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct VerifyReport {
    pub schema_ok: bool,
    pub sig_ok: bool,
    pub digest_ok: bool,
    pub ticks_ok: bool,
    pub pubkey_ok: bool,
    pub envelope_digest_ok: bool,
    pub counts_ok: bool,
    pub fuse_ok: bool,
    pub break_at: Option<u32>,
    pub notes: Vec<String>,
}

/// pubkey_ok = (expect_pubkey.is_none() || sr.pubkey == expected). envelope_digest_ok = recomputed sha256(canon(body.envelope)) == body.envelope_digest.
/// counts_ok = counts.ticks == verdict_events == timing_events == outcome.steps.
impl VerifyReport {
    pub fn intact(&self) -> bool {
        self.schema_ok
            && self.sig_ok
            && self.digest_ok
            && self.ticks_ok
            && self.pubkey_ok
            && self.envelope_digest_ok
            && self.counts_ok
    }
}

pub fn verify(_sr: &SignedReceipt, _expect_pubkey: Option<&str>) -> VerifyReport {
    todo!("WP-4")
}

/// Recompute the verdict chain from a full ticks file. `ChainReport.ok` is CHAIN INTEGRITY ONLY (a re-chained file passes it);
/// the caller MUST compare `report.head` with `sr.body.verdict_chain_head` and report `HEAD MISMATCH` when they differ, and MUST
/// cross-check the ticks header (run_id, arm_id, episode_index, genesis) against the body and the embedded tail's first `prev`.
pub fn verify_ticks_file(_sr: &SignedReceipt, _ticks: &[TickEvent]) -> ChainReport {
    todo!("WP-4")
}

/// The two committed TEST keys (their pubkeys), so `lictor verify` can warn "signed with the committed test key".
/// Placeholder until WP-4 fills it from tests/fixtures/receipt/key.hex and bench/fixtures/key.hex.
pub const TEST_PUBKEYS: [&str; 2] = ["", ""];

#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("canon: {0}")]
    Canon(#[from] lictor_canon::CanonError),
    #[error("key: {0}")]
    Key(String),
    #[error("io: {0}")]
    Io(String),
}
