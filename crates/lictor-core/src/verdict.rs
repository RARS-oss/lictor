// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; replace the bodies, keep the signatures.
//! The per-tick verdict, the severity lattice and the trip-mask bit names.

use serde::{Deserialize, Serialize};

use crate::{chunk::MAX_D, reason::ReasonCode, scores::Scores, state::FuseState};

/// Severity lattice (sbx feedback lineage). Ord derives from declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Nominal,
    Watching,
    Clamped,
    Braking,
    Held,
    Escalated,
    Fault,
    Terminated,
}

pub struct TripMask;

impl TripMask {
    pub const NONE: u32 = 0;
    pub const WORKSPACE: u32 = 1 << 0;
    pub const SPEED: u32 = 1 << 1;
    pub const ACCEL: u32 = 1 << 2;
    pub const JERK: u32 = 1 << 3;
    pub const REACH: u32 = 1 << 4;
    pub const CONTACT: u32 = 1 << 5;
    /// braking infeasible
    pub const BRAKE: u32 = 1 << 6;
    pub const NONFINITE: u32 = 1 << 7;
    pub const SCHEMA: u32 = 1 << 8;
    pub const WATCHDOG: u32 = 1 << 9;
    /// K-of-N predictive fire
    pub const TIER1_CP: u32 = 1 << 10;
    pub const CLAMP_BUDGET: u32 = 1 << 11;
    pub const HANDOFF_TIMEOUT: u32 = 1 << 12;
    pub const OPERATOR_ABORT: u32 = 1 << 13;
    pub const BRAKE_TIMEOUT: u32 = 1 << 14;
    pub const REARM_BUDGET: u32 = 1 << 15;
    pub const TIER0_SOFT: u32 =
        Self::WORKSPACE | Self::SPEED | Self::ACCEL | Self::JERK | Self::REACH | Self::CONTACT;
    pub const TIER0_HARD: u32 = Self::BRAKE | Self::NONFINITE | Self::SCHEMA | Self::WATCHDOG;
    pub const NAMES: [&'static str; 16] = [
        "workspace",
        "speed",
        "accel",
        "jerk",
        "reach",
        "contact",
        "brake",
        "nonfinite",
        "schema",
        "watchdog",
        "tier1_cp",
        "clamp_budget",
        "handoff_timeout",
        "operator_abort",
        "brake_timeout",
        "rearm_budget",
    ];

    /// The names of the bits set in `m`, in bit order (trivially derived from `NAMES`).
    pub fn names(m: u32) -> impl Iterator<Item = &'static str> {
        Self::NAMES.iter().enumerate().filter(move |(i, _)| m & (1u32 << i) != 0).map(|(_, n)| *n)
    }

    pub fn from_name(s: &str) -> Option<u32> {
        Self::NAMES.iter().position(|n| *n == s).map(|i| 1u32 << i)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSource {
    Policy,
    Clamped,
    Brake,
    Hold,
}

/// Copy, no String, no allocation. ~600 bytes.
#[derive(Clone, Copy, Debug)]
pub struct SafetyVerdict {
    /// tick sequence within the episode, from 0
    pub seq: u32,
    /// absolute env step
    pub t: u32,
    pub status: Status,
    /// state AFTER this tick
    pub state: FuseState,
    pub prev_state: FuseState,
    /// TripMask bits raised THIS tick
    pub trips: u32,
    /// the action the executor MUST apply (only [..action_dim] meaningful)
    pub action: [f64; MAX_D],
    pub action_dim: u16,
    pub action_src: ActionSource,
    /// action != raw policy action (always false in Observe mode)
    pub substituted: bool,
    /// dims moved by the Tier-0 projection
    pub clamped_dims: u32,
    pub scores: Scores,
    pub tau: f64,
    /// K-of-N ring AFTER this tick
    pub window: u64,
    /// popcount(window)
    pub window_hits: u8,
    /// units of slack in the braking check; < 0 == infeasible
    pub brake_margin: f64,
    pub reason: ReasonCode,
    /// Some on the tick Escalated is entered
    pub handoff_seq: Option<u32>,
    /// a VerifiedAck was applied this tick
    pub ack_consumed: bool,
    /// Observe mode: the would-be action_src was != Policy
    pub violation_reached_env: bool,
}
