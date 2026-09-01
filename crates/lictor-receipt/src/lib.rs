// SPDX-License-Identifier: MIT
#![forbid(unsafe_code)]
//! lictor-receipt: the verdict and timing hash chains, the receipt body, Ed25519 signing and verification,
//! the per-arm ledger, curve receipts, the human-escalation handoff/ack surface, run history and key files.
//! Off the decision path (std; may use `ln`/`exp`-style functions). See docs/ARCHITECTURE.md sec 8.

pub mod body;
pub mod curve;
pub mod handoff;
pub mod history;
pub mod keys;
pub mod ledger;
pub mod sign;
pub mod tick;

pub use body::{
    evaluate_fuse, BudgetBinding, EpisodeOutcome, FaultBinding, LatencySummary, ReceiptBody, RunBinding,
    VerdictCounts,
};
pub use curve::{sign_curve, verify_curve, CurveMetrics, CurveReceiptBody, DeltaCi, SignedCurve};
pub use handoff::{
    ack_signing_bytes, load_nonces, save_nonces, sign_ack, verify_ack, AckError, AckToken, HandoffRecord,
};
pub use history::{record_episode, summarize, EpisodeRecord, LoopSignals};
pub use keys::{default_key_dir, keygen, load_seed, pubkey_hex, save_seed};
pub use ledger::{append_ledger, ledger_entry, read_ledger, verify_ledger, LedgerEntry, LedgerReport};
pub use sign::{sign, verify, verify_ticks_file, ReceiptError, SignedReceipt, VerifyReport, TEST_PUBKEYS};
pub use tick::{
    tick_event, timing_event, verify_tick_chain, verify_timing_chain, ChainReport, TickEvent, TimingEvent,
};

pub const RECEIPT_SCHEMA: &str = "lictor-receipt/v1";
pub const TICKS_SCHEMA: &str = "lictor-ticks/v1";
pub const LEDGER_SCHEMA: &str = "lictor-ledger/v1";
pub const CURVE_SCHEMA: &str = "lictor-curve/v1";
pub const HANDOFF_SCHEMA: &str = "lictor-handoff/v1";
pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";
