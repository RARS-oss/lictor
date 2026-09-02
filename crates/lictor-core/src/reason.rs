// SPDX-License-Identifier: MIT
//! The human-escalation glossary (sbx `hint_for` lineage): one plain-English sentence per reason code, so a
//! `HandoffRecord` or a receipt tells a person WHY the fuse acted without a manual.

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

impl ReasonCode {
    /// Every variant, in declaration order (for glossaries and exhaustive tests).
    pub const ALL: [ReasonCode; 28] = [
        ReasonCode::Ok,
        ReasonCode::ObserveOnly,
        ReasonCode::WatchBand,
        ReasonCode::ClampWorkspace,
        ReasonCode::ClampSpeed,
        ReasonCode::ClampAccel,
        ReasonCode::ClampJerk,
        ReasonCode::ClampReach,
        ReasonCode::ClampContact,
        ReasonCode::BrakeInfeasible,
        ReasonCode::BrakeTier1Cp,
        ReasonCode::BrakeClampBudget,
        ReasonCode::Stopping,
        ReasonCode::HeldStandstill,
        ReasonCode::EscalateHoldTimeout,
        ReasonCode::EscalateRearmBudget,
        ReasonCode::EscalateTier1Persistent,
        ReasonCode::FaultNonFinite,
        ReasonCode::FaultSchema,
        ReasonCode::FaultWatchdog,
        ReasonCode::FaultBrakeTimeout,
        ReasonCode::FaultInternal,
        ReasonCode::RearmedAuto,
        ReasonCode::RearmedAck,
        ReasonCode::TerminatedAbort,
        ReasonCode::TerminatedTimeout,
        ReasonCode::TerminatedRetune,
        ReasonCode::EpisodeEnd,
    ];
}

/// One sentence per code, <= 120 chars, plain English, explaining WHY the fuse acted.
pub fn reason_text(r: ReasonCode) -> &'static str {
    match r {
        ReasonCode::Ok => "No limit was reached; the policy action was passed through unchanged.",
        ReasonCode::ObserveOnly => {
            "The fuse is observing only: it scored this tick but did not change the action."
        }
        ReasonCode::WatchBand => {
            "A predictive score is close to its threshold; the fuse is watching but has not intervened."
        }
        ReasonCode::ClampWorkspace => {
            "A commanded position left the allowed workspace and was pulled back inside it."
        }
        ReasonCode::ClampSpeed => {
            "A commanded step was faster than the speed limit and was shortened to fit."
        }
        ReasonCode::ClampAccel => {
            "The commanded motion accelerated harder than allowed and was smoothed to the limit."
        }
        ReasonCode::ClampJerk => {
            "The commanded motion changed acceleration faster than allowed and was smoothed."
        }
        ReasonCode::ClampReach => {
            "The first commanded position was too far from the current position and was pulled closer."
        }
        ReasonCode::ClampContact => {
            "The commanded speed close to the object exceeded the contact limit and was reduced."
        }
        ReasonCode::BrakeInfeasible => {
            "The committed motion could not be stopped inside the workspace; the fuse is braking."
        }
        ReasonCode::BrakeTier1Cp => {
            "The predictive detector fired often enough in a row that the fuse is braking."
        }
        ReasonCode::BrakeClampBudget => {
            "Commands had to be corrected too many times in a row; the fuse is braking."
        }
        ReasonCode::Stopping => {
            "A controlled stop is in progress; the setpoint is held at the current position."
        }
        ReasonCode::HeldStandstill => {
            "The robot has stopped and is being held still at its current position."
        }
        ReasonCode::EscalateHoldTimeout => {
            "The robot has been held still for too long without a clean re-arm; a person must decide."
        }
        ReasonCode::EscalateRearmBudget => {
            "The fuse has already re-armed as often as allowed this episode; a person must decide."
        }
        ReasonCode::EscalateTier1Persistent => {
            "The predictive detector kept firing while the robot was held; a person must decide."
        }
        ReasonCode::FaultNonFinite => {
            "An input contained a non-finite number; the fuse holds position until the episode ends."
        }
        ReasonCode::FaultSchema => {
            "A message broke the agreed format or ordering; the fuse holds position until the episode ends."
        }
        ReasonCode::FaultWatchdog => {
            "Too many ticks were missed in a row; the fuse holds position until the episode ends."
        }
        ReasonCode::FaultBrakeTimeout => {
            "The robot did not come to a stop within the braking time limit; the fuse holds position."
        }
        ReasonCode::FaultInternal => {
            "The fuse found an internal inconsistency and holds position until the episode ends."
        }
        ReasonCode::RearmedAuto => {
            "The robot stood still cleanly for the required time and control returned to the policy."
        }
        ReasonCode::RearmedAck => {
            "A signed operator acknowledgement resumed the episode and control returned to the policy."
        }
        ReasonCode::TerminatedAbort => {
            "A signed operator acknowledgement aborted the episode; the robot stays held."
        }
        ReasonCode::TerminatedTimeout => {
            "No operator decision arrived in time; the episode ended with the robot held."
        }
        ReasonCode::TerminatedRetune => {
            "A signed operator acknowledgement ended the episode so the envelope can be retuned."
        }
        ReasonCode::EpisodeEnd => "The episode ended normally.",
    }
}
