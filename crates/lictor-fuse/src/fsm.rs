// SPDX-License-Identifier: MIT
//! The pure escalation state machine (docs/ARCHITECTURE.md sec 6 table; rows evaluated top-down, first match wins).
//!
//! `next` implements rows 1-23 plus 17b exactly as written, with every counter side effect the table lists.
//! Two things the table leaves implicit are made explicit here and covered by `tests/fixtures/fsm/transitions.json`:
//!
//! - Per-visit counters are zeroed on ENTRY: `brake_ticks`/`stopped_ticks` when Braking is entered (rows 17,
//!   17b, 18, 19), `held_ticks`/`held_clean` when Held is entered (row 15) and `escalated_ticks` when Escalated
//!   is entered (row 12). Without that a second visit to Held would escalate immediately.
//! - Rows 17-23 have no "stay" row for Watching or Clamped: a clean tick in Watching with an empty window and
//!   `clean_run < clear_ticks` (the clear-down in progress), or the same in Clamped, matches nothing. The last
//!   row below keeps the state (Watching -> Watching with `WatchBand`; Clamped -> Clamped with `Ok`, since no
//!   limit was reached on that tick and the already-projected chunk continues); an `Idle` fuse (no episode) is
//!   a host error and fails closed into Fault with `FaultInternal`.
//!
//! The rows whose reason the table does not name use `HeldStandstill` for the absorbing "stay" rows 2, 3, 8
//! and 13 (the robot is held still at the latched setpoint on every one of them) and `Stopping` for row 16.
//!
//! Ack handling: the runtime has already verified signature, operator membership and the handoff digest; this
//! function re-checks only the two integers (`ack.handoff_seq == rt.handoff_seq`, `ack.nonce > last_nonce[slot]`)
//! and ignores the ack otherwise. An ignored ack falls through to the counter rows of the current state. Ticks in
//! Held only accept `Resume`/`Abort` (rows 9/10); `Retune` in Held is not a row and is ignored.
//!
//! Float discipline: the only floating-point work here is the hold latch (`fmath::clamp` per coordinate).
//! No allocation, no clock, no panic path; every counter saturates.

use lictor_core::{
    fmath, AckDecision, FuseConfig, FuseMode, FuseState, RearmPolicy, ReasonCode, TripMask, VerifiedAck,
    MAX_OPERATORS, MAX_POS,
};

use crate::fuse::FuseRt;

#[derive(Clone, Copy, Debug)]
pub struct FsmInput {
    pub trips: u32,
    pub predictive: bool,
    pub warn: bool,
    pub soft_clampable: bool,
    pub stopped: bool,
    pub chunk_boundary: bool,
    pub chunk_ok: bool,
    pub ack: Option<VerifiedAck>,
}

/// The three fail-closed bits of row 1.
const FAULT_BITS: u32 = TripMask::NONFINITE | TripMask::SCHEMA | TripMask::WATCHDOG;

/// Pure transition per the ARCHITECTURE table (rows evaluated top-down, first match wins). Updates counters in `rt`,
/// returns (new_state, extra_trips, reason).
///
/// Also writes `rt.state` (the returned state), so a caller that only wants the transition needs nothing else.
pub fn next(cfg: &FuseConfig, rt: &mut FuseRt, inp: FsmInput) -> (FuseState, u32, ReasonCode) {
    // "clean" = no trip AND !warn AND !predictive this tick; clean_run runs in every state.
    let clean = inp.trips == 0 && !inp.warn && !inp.predictive;
    rt.clean_run = if clean { rt.clean_run.saturating_add(1) } else { 0 };

    let from = rt.state;
    let ack = accepted_ack(rt, inp.ack);
    let out = transition(cfg, rt, from, clean, ack, inp);
    rt.state = out.0;
    out
}

/// The ack, if it passes the two integer checks; `None` otherwise (ignored).
fn accepted_ack(rt: &FuseRt, ack: Option<VerifiedAck>) -> Option<VerifiedAck> {
    let a = ack?;
    let slot = a.operator_slot as usize;
    if slot >= MAX_OPERATORS {
        return None;
    }
    if a.handoff_seq != rt.handoff_seq || a.nonce <= rt.last_nonce[slot] {
        return None;
    }
    Some(a)
}

/// Latches the hold setpoint: `hold = clamp(pos)` over the live position dims (`rt.pos` holds the last finite
/// position; `decide()` never writes a non-finite one).
fn latch_hold(cfg: &FuseConfig, rt: &mut FuseRt) {
    let n = if cfg.pos_dim < MAX_POS { cfg.pos_dim } else { MAX_POS };
    let mut c = 0;
    while c < n {
        rt.hold[c] = fmath::clamp(rt.pos[c], cfg.box_lo[c], cfg.box_hi[c]);
        c += 1;
    }
}

fn enter_braking(rt: &mut FuseRt) {
    rt.brake_ticks = 0;
    rt.stopped_ticks = 0;
}

/// Rows 6 / 9: an accepted `Resume` ack re-arms the fuse.
fn rearm_ack(rt: &mut FuseRt, a: VerifiedAck) {
    consume_nonce(rt, a);
    rt.rearms = rt.rearms.saturating_add(1);
    rt.window.clear();
    rt.clean_run = 0;
    rt.clamp_streak = 0;
    rt.held_ticks = 0;
    rt.held_clean = 0;
    rt.escalated_ticks = 0;
    rt.handoff_pending = false;
}

fn consume_nonce(rt: &mut FuseRt, a: VerifiedAck) {
    let slot = a.operator_slot as usize;
    if slot < MAX_OPERATORS {
        rt.last_nonce[slot] = a.nonce;
    }
}

fn fault_reason(trips: u32) -> ReasonCode {
    if trips & TripMask::NONFINITE != 0 {
        ReasonCode::FaultNonFinite
    } else if trips & TripMask::SCHEMA != 0 {
        ReasonCode::FaultSchema
    } else {
        ReasonCode::FaultWatchdog
    }
}

/// `Clamp<first soft bit>` in bit order (workspace, speed, accel, jerk, reach, contact).
fn clamp_reason(trips: u32) -> ReasonCode {
    if trips & TripMask::WORKSPACE != 0 {
        ReasonCode::ClampWorkspace
    } else if trips & TripMask::SPEED != 0 {
        ReasonCode::ClampSpeed
    } else if trips & TripMask::ACCEL != 0 {
        ReasonCode::ClampAccel
    } else if trips & TripMask::JERK != 0 {
        ReasonCode::ClampJerk
    } else if trips & TripMask::REACH != 0 {
        ReasonCode::ClampReach
    } else {
        ReasonCode::ClampContact
    }
}

fn transition(
    cfg: &FuseConfig,
    rt: &mut FuseRt,
    from: FuseState,
    clean: bool,
    ack: Option<VerifiedAck>,
    inp: FsmInput,
) -> (FuseState, u32, ReasonCode) {
    use FuseState::*;
    let h = &cfg.hyst;

    // Row 1: any state, a fail-closed bit -> Fault (latched; the hold is latched on entry only, so a Fault
    // that keeps faulting does not chase a drifting position).
    if inp.trips & FAULT_BITS != 0 {
        if from != Fault {
            latch_hold(cfg, rt);
        }
        return (Fault, 0, fault_reason(inp.trips));
    }

    match from {
        // Row 2.
        Terminated => (Terminated, 0, ReasonCode::HeldStandstill),
        // Row 3: never re-armable in-episode.
        Fault => (Fault, 0, ReasonCode::HeldStandstill),
        Escalated => {
            if let Some(a) = ack {
                match a.decision {
                    // Row 4.
                    AckDecision::Abort => {
                        consume_nonce(rt, a);
                        rt.handoff_pending = false;
                        (Terminated, TripMask::OPERATOR_ABORT, ReasonCode::TerminatedAbort)
                    }
                    // Row 5.
                    AckDecision::Retune => {
                        consume_nonce(rt, a);
                        rt.handoff_pending = false;
                        (Terminated, 0, ReasonCode::TerminatedRetune)
                    }
                    // Row 6.
                    AckDecision::Resume => {
                        rearm_ack(rt, a);
                        (Armed, 0, ReasonCode::RearmedAck)
                    }
                }
            } else if rt.escalated_ticks >= h.handoff_timeout_ticks {
                // Row 7.
                rt.handoff_pending = false;
                (Terminated, TripMask::HANDOFF_TIMEOUT, ReasonCode::TerminatedTimeout)
            } else {
                // Row 8.
                rt.escalated_ticks = rt.escalated_ticks.saturating_add(1);
                (Escalated, 0, ReasonCode::HeldStandstill)
            }
        }
        Held => {
            if let Some(a) = ack {
                match a.decision {
                    // Row 9 (as row 6).
                    AckDecision::Resume => {
                        rearm_ack(rt, a);
                        return (Armed, 0, ReasonCode::RearmedAck);
                    }
                    // Row 10 (as row 4).
                    AckDecision::Abort => {
                        consume_nonce(rt, a);
                        rt.handoff_pending = false;
                        return (Terminated, TripMask::OPERATOR_ABORT, ReasonCode::TerminatedAbort);
                    }
                    // Not a row in Held: ignored, fall through to rows 11-13.
                    AckDecision::Retune => {}
                }
            }
            // Row 11: auto re-arm (a fresh chunk passed the full Tier-0 suite incl. brake THIS tick).
            if h.rearm == RearmPolicy::Auto
                && rt.held_clean >= h.rearm_hold
                && inp.chunk_ok
                && rt.rearms < h.max_rearms
            {
                rt.rearms = rt.rearms.saturating_add(1);
                rt.window.clear();
                return (Armed, 0, ReasonCode::RearmedAuto);
            }
            // Row 12: escalate (handoff).
            if rt.held_ticks >= h.escalate_after_hold_ticks {
                rt.handoff_seq = rt.handoff_seq.wrapping_add(1);
                rt.handoff_pending = true;
                rt.escalated_ticks = 0;
                return if rt.rearms >= h.max_rearms {
                    (Escalated, TripMask::REARM_BUDGET, ReasonCode::EscalateRearmBudget)
                } else if inp.predictive {
                    (Escalated, 0, ReasonCode::EscalateTier1Persistent)
                } else {
                    (Escalated, 0, ReasonCode::EscalateHoldTimeout)
                };
            }
            // Row 13.
            rt.held_ticks = rt.held_ticks.saturating_add(1);
            rt.held_clean = if clean { rt.held_clean.saturating_add(1) } else { 0 };
            (Held, 0, ReasonCode::HeldStandstill)
        }
        Braking => {
            // Row 14: a stop that never confirms is a plant anomaly -> Fault.
            if rt.brake_ticks >= h.brake_timeout_ticks {
                latch_hold(cfg, rt);
                return (Fault, TripMask::BRAKE_TIMEOUT, ReasonCode::FaultBrakeTimeout);
            }
            // Row 15: `stopped` for `stop_confirm_ticks` consecutive ticks (this one included).
            if inp.stopped && u16::from(rt.stopped_ticks).saturating_add(1) >= u16::from(h.stop_confirm_ticks)
            {
                latch_hold(cfg, rt);
                rt.held_ticks = 0;
                rt.held_clean = 0;
                return (Held, 0, ReasonCode::HeldStandstill);
            }
            // Row 16.
            rt.brake_ticks = rt.brake_ticks.saturating_add(1);
            rt.stopped_ticks = if inp.stopped { rt.stopped_ticks.saturating_add(1) } else { 0 };
            (Braking, 0, ReasonCode::Stopping)
        }
        Armed | Watching | Clamped => {
            let soft = inp.trips & TripMask::TIER0_SOFT;
            // Row 17.
            if inp.trips & TripMask::BRAKE != 0 {
                enter_braking(rt);
                return (Braking, 0, ReasonCode::BrakeInfeasible);
            }
            // Row 17b: a soft trip that cannot be clamped (clamp_mode == Off) brakes with the clamp reason.
            if soft != 0 && !inp.soft_clampable {
                enter_braking(rt);
                return (Braking, 0, clamp_reason(soft));
            }
            // Row 18.
            if inp.predictive {
                enter_braking(rt);
                return (Braking, TripMask::TIER1_CP, ReasonCode::BrakeTier1Cp);
            }
            // Row 19.
            if rt.clamp_streak >= h.clamp_streak_to_brake || rt.clamps > h.max_clamps_per_episode {
                enter_braking(rt);
                return (Braking, TripMask::CLAMP_BUDGET, ReasonCode::BrakeClampBudget);
            }
            // Row 20.
            if soft != 0 {
                rt.clamps = rt.clamps.saturating_add(1);
                if inp.chunk_boundary {
                    rt.clamp_streak = rt.clamp_streak.saturating_add(1);
                }
                return (Clamped, 0, clamp_reason(soft));
            }
            // Row 21: clear-down.
            if from != Armed && rt.clean_run >= u16::from(h.clear_ticks) {
                rt.window.clear();
                rt.clamp_streak = 0;
                return (Armed, 0, ReasonCode::Ok);
            }
            // Row 22.
            if inp.warn || rt.window.hits() >= 1 {
                return (Watching, 0, ReasonCode::WatchBand);
            }
            // Row 23.
            if from == Armed {
                let r = if cfg.mode == FuseMode::Observe { ReasonCode::ObserveOnly } else { ReasonCode::Ok };
                return (Armed, 0, r);
            }
            // Stay rows (see the module doc): the clear-down is in progress.
            if from == Watching {
                (Watching, 0, ReasonCode::WatchBand)
            } else {
                (Clamped, 0, ReasonCode::Ok)
            }
        }
        // No episode: a host error, fail closed.
        Idle => {
            latch_hold(cfg, rt);
            (Fault, 0, ReasonCode::FaultInternal)
        }
    }
}
