// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; replace the bodies, keep the signatures.
//! Assemble ReceiptBody from bindings + Tally + chains; sign; write receipt/ticks/timing; append ledger LAST.

use lictor_receipt::{BudgetBinding, LedgerEntry, RunBinding, SignedReceipt, TickEvent, TimingEvent};

use crate::session::SessionConfig;

pub struct EpisodePaths {
    pub receipt: std::path::PathBuf,
    pub ticks: std::path::PathBuf,
    pub timing: std::path::PathBuf,
    pub ledger: std::path::PathBuf,
}

pub fn episode_paths(
    _out_dir: &std::path::Path,
    _run_id: &str,
    _arm_id: &str,
    _episode_index: u32,
) -> EpisodePaths {
    todo!("WP-6")
}

pub fn write_episode(
    _paths: &EpisodePaths,
    _sr: &SignedReceipt,
    _ticks: &[TickEvent],
    _timing: &[TimingEvent],
) -> anyhow::Result<LedgerEntry> {
    todo!("WP-6")
}

/// Host-side crash accounting (`lictor crash-receipt`): when the serve child died before `episode_end`, write a minimal SIGNED receipt
/// with zero counts, `outcome = {steps: 0, success: false, ended_by: "fuse_crash"}`, `terminal_state: fault`, `fuse_ok: false`,
/// `fuse_notes: ["fuse process died before episode_end; host-written crash receipt"]`, empty ticks/timing files, and append the ledger.
pub fn write_crash_episode(
    _cfg: &SessionConfig,
    _run: RunBinding,
    _budget: BudgetBinding,
    _client: &str,
    _note: &str,
) -> anyhow::Result<LedgerEntry> {
    todo!("WP-6")
}
