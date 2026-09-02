// SPDX-License-Identifier: MIT
//! Per-episode counters: Copy, fixed arrays; converted to receipt `VerdictCounts` by lictor-receipt.
//!
//! The hand-written `Default` below is the frozen specification. `record` is the single write path used by
//! `decide()` (one call per tick, after the verdict is final); the chunk counters have their own helper because
//! they are known before the verdict exists. Every counter saturates instead of wrapping, so a pathological
//! host cannot make a count panic in debug builds or roll over in release builds.

use lictor_core::{fmath, FuseState, ReasonCode, SafetyVerdict, NFEAT};

#[derive(Clone, Copy, Debug)]
pub struct Tally {
    pub ticks: u32,
    pub nominal: u32,
    pub watching: u32,
    pub clamped: u32,
    pub braking: u32,
    pub held: u32,
    pub escalated: u32,
    pub fault: u32,
    pub terminated: u32,
    pub substituted: u32,
    pub chunks_seen: u32,
    pub chunks_rejected: u32,
    pub clamps: u32,
    pub holds: u32,
    pub rearms: u32,
    pub escalations: u32,
    pub trips_by_bit: [u32; 16],
    pub fired_by_feat: [u32; NFEAT],
    pub first_trip_tick: Option<u32>,
    pub first_trip_reason: Option<ReasonCode>,
    pub first_stop_tick: Option<u32>,
    pub handoff_tick: Option<u32>,
    pub violations_reached_env: u32,
    pub max_s: f64,
    pub max_z: [f64; NFEAT],
    pub terminal_state: FuseState,
}

/// Hand-written (NOT derived): zeros/None everywhere, `terminal_state = Idle`, `max_s = f64::NEG_INFINITY`,
/// `max_z = [f64::NEG_INFINITY; NFEAT]` -- `s` is NEG_INFINITY while nothing is valid or Tier 1 is disarmed, and 0.0 would
/// bias the per-episode `max_s` that feeds ROC-AUC. F64Hex encodes +-inf exactly; the wire encodes it as `null`.
impl Default for Tally {
    fn default() -> Self {
        Self {
            ticks: 0,
            nominal: 0,
            watching: 0,
            clamped: 0,
            braking: 0,
            held: 0,
            escalated: 0,
            fault: 0,
            terminated: 0,
            substituted: 0,
            chunks_seen: 0,
            chunks_rejected: 0,
            clamps: 0,
            holds: 0,
            rearms: 0,
            escalations: 0,
            trips_by_bit: [0; 16],
            fired_by_feat: [0; NFEAT],
            first_trip_tick: None,
            first_trip_reason: None,
            first_stop_tick: None,
            handoff_tick: None,
            violations_reached_env: 0,
            max_s: f64::NEG_INFINITY,
            max_z: [f64::NEG_INFINITY; NFEAT],
            terminal_state: FuseState::Idle,
        }
    }
}

/// What the state machine did this tick beyond what the verdict itself shows (derived by `decide()` from the
/// counters in `FuseRt` before and after `fsm::next`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TickEvents {
    /// `Held` was entered this tick (row 15).
    pub entered_hold: bool,
    /// `Escalated` was entered this tick (row 12): a handoff was emitted.
    pub entered_escalated: bool,
    /// `FuseRt::rearms` advanced (rows 6, 9, 11).
    pub rearmed: bool,
    /// `FuseRt::clamps` advanced (row 20).
    pub clamped: bool,
    /// `calib.mask & scores.valid`: the channels whose `z` is a real standardised value this tick.
    pub live_z: u32,
}

impl Tally {
    /// Folds one final verdict into the counters. Called exactly once per tick by `decide()`.
    ///
    /// `first_stop_tick` is the first tick whose post-state `is_stop()` (Braking, Held, Escalated, Fault or
    /// Terminated) -- rows 17/17b/18/19 of the table say "first_stop" on entering Braking, and a direct Fault
    /// (row 1) stops the robot just the same. `max_s` folds every tick (NEG_INFINITY stays NEG_INFINITY while
    /// Tier 1 never produces a valid aggregate); `max_z[j]` folds only the channels in `live_z`, because an
    /// invalid or unmasked channel carries `z_j = 0.0`, which is a placeholder, not a score.
    pub(crate) fn record(&mut self, v: &SafetyVerdict, ev: TickEvents) {
        self.ticks = self.ticks.saturating_add(1);
        match v.state {
            FuseState::Idle | FuseState::Armed => self.nominal = self.nominal.saturating_add(1),
            FuseState::Watching => self.watching = self.watching.saturating_add(1),
            FuseState::Clamped => self.clamped = self.clamped.saturating_add(1),
            FuseState::Braking => self.braking = self.braking.saturating_add(1),
            FuseState::Held => self.held = self.held.saturating_add(1),
            FuseState::Escalated => self.escalated = self.escalated.saturating_add(1),
            FuseState::Fault => self.fault = self.fault.saturating_add(1),
            FuseState::Terminated => self.terminated = self.terminated.saturating_add(1),
        }
        if v.substituted {
            self.substituted = self.substituted.saturating_add(1);
        }
        let mut b = 0;
        while b < 16 {
            if v.trips & (1u32 << b) != 0 {
                self.trips_by_bit[b] = self.trips_by_bit[b].saturating_add(1);
            }
            b += 1;
        }
        let mut j = 0;
        while j < NFEAT {
            let bit = 1u32 << j;
            if v.scores.fired & bit != 0 {
                self.fired_by_feat[j] = self.fired_by_feat[j].saturating_add(1);
            }
            if ev.live_z & bit != 0 {
                self.max_z[j] = fmath::max(self.max_z[j], v.scores.z[j]);
            }
            j += 1;
        }
        if v.trips != 0 && self.first_trip_tick.is_none() {
            self.first_trip_tick = Some(v.t);
            self.first_trip_reason = Some(v.reason);
        }
        if v.state.is_stop() && self.first_stop_tick.is_none() {
            self.first_stop_tick = Some(v.t);
        }
        if ev.entered_escalated {
            self.escalations = self.escalations.saturating_add(1);
            if self.handoff_tick.is_none() {
                self.handoff_tick = Some(v.t);
            }
        }
        if ev.entered_hold {
            self.holds = self.holds.saturating_add(1);
        }
        if ev.rearmed {
            self.rearms = self.rearms.saturating_add(1);
        }
        if ev.clamped {
            self.clamps = self.clamps.saturating_add(1);
        }
        if v.violation_reached_env {
            self.violations_reached_env = self.violations_reached_env.saturating_add(1);
        }
        self.max_s = fmath::max(self.max_s, v.scores.s);
        self.terminal_state = v.state;
    }

    /// A chunk was delivered this tick (`chunks_seen`); `rejected` when a `TIER0_HARD` bit kept it out of `cur`.
    pub(crate) fn record_chunk(&mut self, rejected: bool) {
        self.chunks_seen = self.chunks_seen.saturating_add(1);
        if rejected {
            self.chunks_rejected = self.chunks_rejected.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lictor_core::{ActionSource, Scores, Status, MAX_D};

    fn verdict(state: FuseState, trips: u32, s: f64) -> SafetyVerdict {
        SafetyVerdict {
            seq: 0,
            t: 7,
            status: Status::Nominal,
            state,
            prev_state: FuseState::Armed,
            trips,
            action: [0.0; MAX_D],
            action_dim: 2,
            action_src: ActionSource::Policy,
            substituted: state != FuseState::Armed,
            clamped_dims: 0,
            scores: Scores { f: [0.0; NFEAT], z: [0.0; NFEAT], s, valid: 0, fired: 0 },
            tau: f64::INFINITY,
            window: 0,
            window_hits: 0,
            brake_margin: 1.0,
            reason: ReasonCode::Ok,
            handoff_seq: None,
            ack_consumed: false,
            violation_reached_env: false,
        }
    }

    #[test]
    fn default_starts_at_negative_infinity_not_zero() {
        let t = Tally::default();
        assert_eq!(t.max_s, f64::NEG_INFINITY);
        assert!(t.max_z.iter().all(|z| *z == f64::NEG_INFINITY));
        assert_eq!(t.terminal_state, FuseState::Idle);
        assert_eq!(t.ticks + t.nominal + t.clamps + t.violations_reached_env, 0);
        assert!(t.first_trip_tick.is_none() && t.handoff_tick.is_none());
    }

    #[test]
    fn record_counts_states_trips_and_firsts() {
        let mut t = Tally::default();
        t.record(&verdict(FuseState::Armed, 0, f64::NEG_INFINITY), TickEvents::default());
        assert_eq!((t.ticks, t.nominal, t.substituted), (1, 1, 0));
        assert_eq!(t.max_s, f64::NEG_INFINITY, "a NEG_INFINITY aggregate leaves max_s untouched");
        let mut v = verdict(FuseState::Braking, 1 << 6, 0.25);
        v.reason = ReasonCode::BrakeInfeasible;
        t.record(&v, TickEvents::default());
        assert_eq!((t.braking, t.substituted, t.trips_by_bit[6]), (1, 1, 1));
        assert_eq!(t.first_trip_tick, Some(7));
        assert_eq!(t.first_trip_reason, Some(ReasonCode::BrakeInfeasible));
        assert_eq!(t.first_stop_tick, Some(7));
        assert_eq!(t.max_s, 0.25);
        assert_eq!(t.terminal_state, FuseState::Braking);
        let ev = TickEvents {
            entered_hold: true,
            rearmed: true,
            clamped: true,
            entered_escalated: true,
            live_z: 0,
        };
        t.record(&verdict(FuseState::Held, 0, -1.0), ev);
        assert_eq!((t.holds, t.rearms, t.clamps, t.escalations), (1, 1, 1, 1));
        assert_eq!(t.handoff_tick, Some(7));
        assert_eq!(t.max_s, 0.25, "max_s is a running maximum");
        t.record_chunk(false);
        t.record_chunk(true);
        assert_eq!((t.chunks_seen, t.chunks_rejected), (2, 1));
    }

    #[test]
    fn max_z_folds_only_live_channels() {
        let mut t = Tally::default();
        let mut v = verdict(FuseState::Armed, 0, 0.0);
        v.scores.z[0] = 2.5;
        v.scores.z[1] = 9.0;
        t.record(&v, TickEvents { live_z: 1, ..TickEvents::default() });
        assert_eq!(t.max_z[0], 2.5);
        assert_eq!(t.max_z[1], f64::NEG_INFINITY, "an unmasked channel keeps NEG_INFINITY");
    }
}
