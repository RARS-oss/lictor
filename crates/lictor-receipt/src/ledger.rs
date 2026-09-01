// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! A hash chain per arm: dropping, reordering or editing an entry WITHOUT the signing key breaks the chain at a reported seq.
//! It does not defend against the key-holder (who can rebuild and re-sign); tail truncation of a DECLARED pool is detected
//! by `lictor curve`.

use crate::sign::{ReceiptError, SignedReceipt};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry {
    pub seq: u32,
    pub run_id: String,
    pub arm_id: String,
    pub episode_index: u32,
    pub seed: u64,
    pub init_state_digest: String,
    pub receipt_digest: String,
    pub verdict_chain_head: String,
    pub success: bool,
    pub fuse_ok: bool,
    pub tripped: bool,
    pub stopped: bool,
    pub escalated: bool,
    pub prev: String,
    pub hash: String,
}

pub fn ledger_entry(_prev: &str, _seq: u32, _sr: &SignedReceipt) -> LedgerEntry {
    todo!("WP-4")
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LedgerReport {
    pub chain_ok: bool,
    pub break_at: Option<u32>,
    pub episodes: u32,
    pub successes: u32,
    pub trips: u32,
    pub stops: u32,
    pub escalations: u32,
    pub head: String,
}

pub fn verify_ledger(_entries: &[LedgerEntry]) -> LedgerReport {
    todo!("WP-4")
}

pub fn read_ledger(_path: &std::path::Path) -> Result<Vec<LedgerEntry>, ReceiptError> {
    todo!("WP-4")
}

/// reads head, appends, single writer
pub fn append_ledger(_path: &std::path::Path, _sr: &SignedReceipt) -> Result<LedgerEntry, ReceiptError> {
    todo!("WP-4")
}
