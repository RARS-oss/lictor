// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; the bodies below are the frozen text, keep the signatures.
//! Fuse states and modes.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuseState {
    #[default]
    Idle,
    Armed,
    Watching,
    Clamped,
    Braking,
    Held,
    Escalated,
    Fault,
    Terminated,
}

impl FuseState {
    pub fn is_stop(self) -> bool {
        matches!(self, Self::Braking | Self::Held | Self::Escalated | Self::Fault | Self::Terminated)
    }

    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Fault | Self::Terminated)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuseMode {
    Observe,
    Enforce,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecMode {
    Sync,
    Async,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeInit {
    pub episode_index: u32,
    pub seed: u64,
    pub delay_steps: u16,
    pub exec: ExecMode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_idle_and_predicates_partition_the_states() {
        assert_eq!(FuseState::default(), FuseState::Idle);
        let stop = [
            FuseState::Braking,
            FuseState::Held,
            FuseState::Escalated,
            FuseState::Fault,
            FuseState::Terminated,
        ];
        let run = [FuseState::Idle, FuseState::Armed, FuseState::Watching, FuseState::Clamped];
        for s in stop {
            assert!(s.is_stop(), "{s:?}");
        }
        for s in run {
            assert!(!s.is_stop() && !s.is_terminal(), "{s:?}");
        }
        assert!(FuseState::Fault.is_terminal() && FuseState::Terminated.is_terminal());
        assert!(!FuseState::Held.is_terminal() && !FuseState::Escalated.is_terminal());
        assert!(FuseState::Idle < FuseState::Armed && FuseState::Fault < FuseState::Terminated);
    }
}
