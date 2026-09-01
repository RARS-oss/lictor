// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; this file is frozen data types only (no bodies to fill).
//! Crypto-free ack types (verification lives in lictor-receipt).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AckDecision {
    Resume,
    Abort,
    Retune,
}

/// An AckToken whose signature, operator membership and handoff digest were ALREADY verified by the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedAck {
    pub decision: AckDecision,
    pub operator_slot: u8,
    pub nonce: u64,
    pub handoff_seq: u32,
}
