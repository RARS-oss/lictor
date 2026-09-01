// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; replace the bodies, keep the signatures.
//! The human-escalation glossary (sbx `hint_for` lineage).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    Ok,
    ObserveOnly,
    WatchBand,
    ClampWorkspace,
    ClampSpeed,
    ClampAccel,
    ClampJerk,
    ClampReach,
    ClampContact,
    BrakeInfeasible,
    BrakeTier1Cp,
    BrakeClampBudget,
    Stopping,
    HeldStandstill,
    EscalateHoldTimeout,
    EscalateRearmBudget,
    EscalateTier1Persistent,
    FaultNonFinite,
    FaultSchema,
    FaultWatchdog,
    FaultBrakeTimeout,
    FaultInternal,
    RearmedAuto,
    RearmedAck,
    TerminatedAbort,
    TerminatedTimeout,
    TerminatedRetune,
    EpisodeEnd,
}

/// One sentence per code, <= 120 chars, plain English, explaining WHY the fuse acted.
pub fn reason_text(_r: ReasonCode) -> &'static str {
    todo!("WP-1")
}
