// SPDX-License-Identifier: MIT
//! The GUARD stage and the order of the stages inside `decide()`: fail-closed on non-finite input (including
//! `aux`, which nothing consumes -- deliberate), schema and continuity violations, the watchdog, hard-trip
//! chunk rejection, the latched hold, the disarmed Tier-1 aggregate, and the async-drop `from = idx` brake check.

mod common;

use common::*;
use lictor_core::{ActionSource, FuseMode, FuseState, ReasonCode, Status, TripMask};
use lictor_detect::brake::brake_feasible;
use lictor_fuse::Fuse;

/// A resting agent at `pos` whose policy walks +2 px/step in x.
struct Sim {
    fuse: Fuse,
    pos: [f64; 2],
    vel: [f64; 2],
    /// The per-row step of the policy's straight chunks.
    step: [f64; 2],
}

impl Sim {
    fn new(mode: FuseMode) -> Self {
        Self::with(cfg(mode))
    }

    fn with(cfg: lictor_core::FuseConfig) -> Self {
        let mut fuse = Fuse::new(cfg);
        fuse.reset(init());
        Self { fuse, pos: [300.0, 300.0], vel: [0.0, 0.0], step: [2.0, 0.0] }
    }

    fn chunk(&self, seq: u32, t_emit: u32) -> Chunk {
        Chunk::straight(seq, t_emit, self.pos, self.step)
    }

    fn obs(&self, t: u32) -> Obs {
        Obs::at(t, self.pos, self.vel)
    }

    /// One tick; a chunk is delivered on multiples of 8.
    fn step(&mut self, t: u32) -> lictor_core::SafetyVerdict {
        let seq = t / 8;
        let ch = self.chunk(seq, seq * 8);
        let delivered = if t.is_multiple_of(8) { Some(&ch) } else { None };
        self.tick(t, delivered, (t % 8) as u16, 0)
    }

    /// One tick from the current position and velocity.
    fn tick(&mut self, t: u32, chunk: Option<&Chunk>, idx: u16, missed: u8) -> lictor_core::SafetyVerdict {
        let o = self.obs(t);
        tick(&mut self.fuse, &o, chunk, idx, missed)
    }

    fn run(&mut self, from: u32, to: u32) {
        for t in from..to {
            let v = self.step(t);
            assert_eq!(v.state, FuseState::Armed, "t={t} {:?} {:?}", v.reason, trip_names(v.trips));
        }
    }
}

fn assert_fault(v: &lictor_core::SafetyVerdict, bit: u32, reason: ReasonCode) {
    assert_eq!(v.state, FuseState::Fault, "{:?}", v.reason);
    assert_eq!(v.status, Status::Fault);
    assert_eq!(v.trips & bit, bit, "trips {:?}", trip_names(v.trips));
    assert_eq!(v.reason, reason);
    assert_eq!(v.action_src, ActionSource::Hold);
    assert!(v.substituted);
    assert!(v.brake_margin.is_nan(), "no brake check ran on a GUARD tick");
    assert_eq!(v.scores.s, f64::NEG_INFINITY);
}

#[test]
fn nan_in_obs_faults_with_the_hold_and_the_fault_persists() {
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 5);
    // t = 5: NaN position -> Fault; the hold is the previous finite position (300, 300).
    let mut bad = sim.obs(5);
    bad.pos[0] = f64::NAN;
    let ch = sim.chunk(0, 0);
    let v = tick(&mut sim.fuse, &bad, None, 5, 0);
    assert_fault(&v, TripMask::NONFINITE, ReasonCode::FaultNonFinite);
    assert_eq!(action2(&v), [300.0, 300.0]);
    assert_eq!(v.prev_state, FuseState::Armed);
    assert_eq!(v.seq, 5);
    // Finite inputs afterwards: still Fault, same hold, no new trip, seq keeps counting.
    for t in 6..8 {
        let v = sim.tick(t, None, t as u16, 0);
        assert_eq!(v.state, FuseState::Fault);
        assert_eq!(v.prev_state, FuseState::Fault);
        assert_eq!(v.trips, 0);
        assert_eq!(v.reason, ReasonCode::HeldStandstill);
        assert_eq!(action2(&v), [300.0, 300.0]);
        assert_eq!(v.action_src, ActionSource::Hold);
        assert_eq!(v.seq, t);
    }
    // A perfectly good chunk does not re-arm a Fault.
    let ch1 = Chunk::straight(1, 8, sim.pos, [2.0, 0.0]);
    let v = sim.tick(8, Some(&ch1), 0, 0);
    assert_eq!(v.state, FuseState::Fault);
    assert_eq!(v.trips, 0);
    drop(ch);
    let t = sim.fuse.finish();
    assert_eq!(t.fault, 4);
    assert_eq!(t.first_trip_tick, Some(5));
    assert_eq!(t.first_trip_reason, Some(ReasonCode::FaultNonFinite));
    assert_eq!(t.first_stop_tick, Some(5));
    assert_eq!(t.terminal_state, FuseState::Fault);
    assert_eq!(t.substituted, 4);
}

#[test]
fn nan_in_aux_faults_even_though_nothing_consumes_aux() {
    // Deliberate fail-closed: aux is host-declared privileged state; a NaN there is a broken host, full stop.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 3);
    let mut bad = sim.obs(3);
    bad.aux[2] = f64::NAN;
    let v = tick(&mut sim.fuse, &bad, None, 3, 0);
    assert_fault(&v, TripMask::NONFINITE, ReasonCode::FaultNonFinite);
    assert_eq!(action2(&v), [300.0, 300.0]);
    // The same for a NaN in ext and in the chunk.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 2);
    let mut bad = sim.obs(2);
    bad.ext = vec![f64::INFINITY];
    let v = tick(&mut sim.fuse, &bad, None, 2, 0);
    assert_fault(&v, TripMask::NONFINITE, ReasonCode::FaultNonFinite);
    let mut sim = Sim::new(FuseMode::Enforce);
    let mut ch = sim.chunk(0, 0);
    ch.a[7] = f64::NAN;
    let v = sim.tick(0, Some(&ch), 0, 0);
    assert_fault(&v, TripMask::NONFINITE, ReasonCode::FaultNonFinite);
    assert_eq!(sim.fuse.finish().chunks_seen, 0, "a GUARD tick never ingests the chunk");
}

#[test]
fn fault_holds_in_observe_mode_too() {
    let mut sim = Sim::new(FuseMode::Observe);
    sim.run(0, 2);
    let mut bad = sim.obs(2);
    bad.pos[1] = f64::NEG_INFINITY;
    let v = tick(&mut sim.fuse, &bad, None, 2, 0);
    assert_fault(&v, TripMask::NONFINITE, ReasonCode::FaultNonFinite);
    assert!(!v.violation_reached_env, "nothing reached the environment unchecked");
    assert_eq!(action2(&v), [300.0, 300.0]);
}

#[test]
fn idx_at_or_beyond_horizon_is_schema() {
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 2);
    let v = sim.tick(2, None, 15, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);
    assert_eq!(v.trips & TripMask::NONFINITE, 0);
}

#[test]
fn time_jump_needs_matching_missed_ticks() {
    // 24 -> 30 with missed_ticks = 0: SCHEMA.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 25);
    let v = sim.tick(30, None, 6, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);

    // 24 -> 30 with missed_ticks = 5 under a watchdog of 8: accepted.
    let mut sim = Sim::with(cfg_with(FuseMode::Enforce, |e| e.hysteresis.watchdog_ticks = 8));
    sim.run(0, 25);
    let v = sim.tick(30, None, 6, 5);
    assert_eq!(v.state, FuseState::Armed, "{:?}", trip_names(v.trips));
    assert_eq!(v.trips, 0);
    assert_eq!(v.action_src, ActionSource::Policy);
    assert_eq!(v.t, 30);
    // ... and the continuity guard now counts from 30.
    let v = sim.tick(31, None, 7, 0);
    assert_eq!(v.state, FuseState::Armed);

    // 24 -> 30 with missed_ticks = 5 under the base watchdog of 2: continuity holds, the watchdog trips.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 25);
    let v = sim.tick(30, None, 6, 5);
    assert_fault(&v, TripMask::WATCHDOG, ReasonCode::FaultWatchdog);
    assert_eq!(v.trips & TripMask::SCHEMA, 0, "continuity was satisfied");
}

#[test]
fn first_tick_must_be_t_zero_and_a_delivery() {
    let mut sim = Sim::new(FuseMode::Enforce);
    let ch = Chunk::straight(0, 5, sim.pos, [2.0, 0.0]);
    let v = sim.tick(5, Some(&ch), 0, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);
    // The hold latches the (finite) position of the faulting tick itself.
    assert_eq!(action2(&v), [300.0, 300.0]);

    // The very first tick faulting with a non-finite position: the hold is clamp(zeros) = the box corner
    // (documented; harmless on PushT).
    let mut sim = Sim::new(FuseMode::Enforce);
    let mut bad = sim.obs(0);
    bad.pos[0] = f64::NAN;
    let ch = Chunk::straight(0, 0, [300.0, 300.0], [2.0, 0.0]);
    let v = tick(&mut sim.fuse, &bad, Some(&ch), 0, 0);
    assert_fault(&v, TripMask::NONFINITE, ReasonCode::FaultNonFinite);
    let cfg = cfg(FuseMode::Enforce);
    assert_eq!(action2(&v), [cfg.box_lo[0], cfg.box_lo[1]]);

    let mut sim = Sim::new(FuseMode::Enforce);
    let v = sim.tick(0, None, 0, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);
}

#[test]
fn chunk_continuity_is_guarded() {
    // seq 3 after 1.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 16);
    let ch = Chunk::straight(3, 16, sim.pos, [2.0, 0.0]);
    let v = sim.tick(16, Some(&ch), 0, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);
    assert_eq!(sim.fuse.rt().next_chunk_seq, 2, "the rejected delivery does not advance next_chunk_seq");

    // t_emit in the future.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 8);
    let ch = Chunk::straight(1, 9, sim.pos, [2.0, 0.0]);
    let v = sim.tick(8, Some(&ch), 0, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);

    // idx inconsistent with t - t_emit.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 8);
    let ch = Chunk::straight(1, 6, sim.pos, [2.0, 0.0]);
    let v = sim.tick(8, Some(&ch), 1, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);

    // Wrong dims.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 8);
    let mut ch = Chunk::straight(1, 8, sim.pos, [2.0, 0.0]);
    ch.exec = 7;
    let v = sim.tick(8, Some(&ch), 0, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);

    // The host's own schema flag.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 3);
    let o = sim.obs(3);
    let v = tick_full(&mut sim.fuse, &o, None, 3, 0, None, true);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);

    // A position of the wrong width.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 3);
    let mut bad = sim.obs(3);
    bad.pos.push(1.0);
    let v = tick(&mut sim.fuse, &bad, None, 3, 0);
    assert_fault(&v, TripMask::SCHEMA, ReasonCode::FaultSchema);
}

#[test]
fn missed_ticks_over_the_watchdog_fault() {
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 1);
    // 0 -> 4 with missed 3 (> watchdog 2): continuity ok, WATCHDOG.
    let v = sim.tick(4, None, 4, 3);
    assert_fault(&v, TripMask::WATCHDOG, ReasonCode::FaultWatchdog);
    assert_eq!(v.trips, TripMask::WATCHDOG);
    // missed == watchdog is fine.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 1);
    let v = sim.tick(3, None, 3, 2);
    assert_eq!(v.state, FuseState::Armed, "{:?}", trip_names(v.trips));
}

/// The wall scenario: the agent 5 px from the margin-adjusted wall at 250 px/s toward it. A chunk that parks
/// the setpoint on the wall makes the PD rollout overshoot (brake infeasible); one that pulls back is feasible.
fn wall_sim(mode: FuseMode) -> Sim {
    let mut sim = Sim::new(mode);
    sim.pos = [490.0, 300.0];
    sim.vel = [250.0, 0.0];
    sim
}

fn wall_chunk(seq: u32, t_emit: u32) -> Chunk {
    Chunk::rows(seq, t_emit, &[[495.0, 300.0]; 15])
}

#[test]
fn a_hard_trip_rejects_the_chunk_and_brakes() {
    let cfg = cfg(FuseMode::Enforce);
    let mut sim = wall_sim(FuseMode::Enforce);
    let ch = wall_chunk(0, 0);
    let b = brake_feasible(&cfg, &sim.pos, &sim.vel, ch.view(), 0);
    assert!(!b.feasible && b.margin < 0.0, "margin {}", b.margin);
    let v = sim.tick(0, Some(&ch), 0, 0);
    assert_eq!(v.state, FuseState::Braking, "{:?}", trip_names(v.trips));
    assert_eq!(v.trips, TripMask::BRAKE, "{:?}", trip_names(v.trips));
    assert_eq!(v.reason, ReasonCode::BrakeInfeasible);
    assert_eq!(v.action_src, ActionSource::Brake);
    assert_eq!(action2(&v), clamp2(&cfg, sim.pos));
    assert!(v.substituted);
    assert_eq!(v.brake_margin.to_bits(), b.margin.to_bits());
    let t = sim.fuse.finish();
    assert_eq!((t.chunks_seen, t.chunks_rejected), (1, 1));
    assert_eq!(t.first_stop_tick, Some(0));
    assert_eq!(t.trips_by_bit[6], 1);
    // The next tick still has a chunk to index (the projection of the rejected one) and keeps braking.
    sim.vel = [100.0, 0.0];
    let v = sim.tick(1, None, 1, 0);
    assert_eq!(v.state, FuseState::Braking);
    assert_eq!(v.reason, ReasonCode::Stopping);
    assert_eq!(v.trips & TripMask::SCHEMA, 0);
}

#[test]
fn the_hold_is_latched_and_does_not_chase_the_position() {
    let cfg = cfg(FuseMode::Enforce);
    let mut sim = wall_sim(FuseMode::Enforce);
    let ch = wall_chunk(0, 0);
    let v = sim.tick(0, Some(&ch), 0, 0);
    assert_eq!(v.state, FuseState::Braking);
    // The robot stops at (493, 300): two confirmed stopped ticks -> Held with hold = clamp(493, 300).
    sim.pos = [492.0, 300.0];
    sim.vel = [0.0, 0.0];
    let v = sim.tick(1, None, 1, 0);
    assert_eq!(v.state, FuseState::Braking);
    assert_eq!(action2(&v), [492.0, 300.0], "the brake setpoint follows the position while braking");
    sim.pos = [493.0, 300.0];
    let v = sim.tick(2, None, 2, 0);
    assert_eq!(v.state, FuseState::Held);
    assert_eq!(v.reason, ReasonCode::HeldStandstill);
    assert_eq!(v.action_src, ActionSource::Hold);
    assert_eq!(action2(&v), clamp2(&cfg, [493.0, 300.0]));
    // Synthetic drift: the hold must not move.
    for (k, t) in (3u32..8).enumerate() {
        sim.pos = [480.0 - 5.0 * k as f64, 310.0 + 3.0 * k as f64];
        let v = sim.tick(t, None, t as u16, 0);
        assert_eq!(v.state, FuseState::Held, "t={t}");
        assert_eq!(action2(&v), [493.0, 300.0], "t={t}");
        assert_eq!(v.action_src, ActionSource::Hold);
    }
    let t = sim.fuse.finish();
    assert_eq!(t.holds, 1);
    assert_eq!(t.held, 6);
    assert_eq!(t.braking, 2);
}

#[test]
fn a_disarmed_tier1_leaves_max_s_at_negative_infinity() {
    let cfg = cfg_with(FuseMode::Enforce, |e| e.gate.clear());
    assert!(!cfg.calib.armed());
    let mut sim = Sim::with(cfg);
    sim.run(0, 24);
    let t = sim.fuse.finish();
    assert_eq!(t.max_s, f64::NEG_INFINITY, "not 0.0");
    assert!(t.max_z.iter().all(|z| *z == f64::NEG_INFINITY));
    assert_eq!(t.ticks, 24);
    assert_eq!(t.nominal, 24);
    assert_eq!(t.chunks_seen, 3);
    // The base gate (no calibration) computes features but has tau = +inf: s is finite once features are valid,
    // and still nothing fires.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.run(0, 24);
    let t = sim.fuse.finish();
    assert!(t.max_s.is_finite(), "max_s {}", t.max_s);
    assert_eq!(t.fired_by_feat, [0; lictor_core::NFEAT]);
}

#[test]
fn async_drop_delivery_is_checked_from_row_idx() {
    // Rows 0..2 park the setpoint on the wall (infeasible from 250 px/s); rows 3.. pull back to x = 430.
    let mut rows = [[430.0, 300.0]; 15];
    rows[0] = [495.0, 300.0];
    rows[1] = [495.0, 300.0];
    rows[2] = [495.0, 300.0];
    let cfg = cfg(FuseMode::Enforce);
    let pos = [490.0, 300.0];
    let vel = [250.0, 0.0];
    let sync = Chunk::rows(1, 8, &rows);
    let asyn = Chunk::rows(1, 5, &rows);
    let from0 = brake_feasible(&cfg, &pos, &vel, sync.view(), 0);
    let from3 = brake_feasible(&cfg, &pos, &vel, asyn.view(), 3);
    assert!(!from0.feasible, "rows 0..2 alone: margin {}", from0.margin);
    assert!(from3.feasible, "from row 3: margin {}", from3.margin);

    // Sync delivery at t = 8 (t_emit = 8, idx = 0): rejected, Braking.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.pos = pos;
    sim.step = [-2.0, 0.0];
    sim.run(0, 8);
    sim.vel = vel;
    let v = sim.tick(8, Some(&sync), 0, 0);
    assert_eq!(v.state, FuseState::Braking, "{:?}", trip_names(v.trips));
    assert_eq!(v.brake_margin.to_bits(), from0.margin.to_bits());
    assert_eq!(sim.fuse.finish().chunks_rejected, 1);

    // Async-drop delivery at t = 8 (t_emit = 5, idx = 3): accepted, row 3 is the action, brake sees from = 3.
    let mut sim = Sim::new(FuseMode::Enforce);
    sim.pos = pos;
    sim.step = [-2.0, 0.0];
    sim.run(0, 8);
    sim.vel = vel;
    let v = sim.tick(8, Some(&asyn), 3, 0);
    assert_eq!(v.state, FuseState::Armed, "{:?} {:?}", v.reason, trip_names(v.trips));
    assert_eq!(v.trips, 0);
    assert_eq!(action2(&v), [430.0, 300.0]);
    assert_eq!(v.action_src, ActionSource::Policy);
    assert_eq!(v.brake_margin.to_bits(), from3.margin.to_bits());
    assert_eq!(sim.fuse.finish().chunks_rejected, 0);
    // The following intra tick indexes row 4 of the same chunk.
    sim.vel = [0.0, 0.0];
    let v = sim.tick(9, None, 4, 0);
    assert_eq!(v.state, FuseState::Armed, "{:?}", trip_names(v.trips));
    assert_eq!(action2(&v), [430.0, 300.0]);
}

#[test]
fn reset_clears_everything_between_episodes() {
    let mut sim = wall_sim(FuseMode::Enforce);
    let ch = wall_chunk(0, 0);
    let v = sim.tick(0, Some(&ch), 0, 0);
    assert_eq!(v.state, FuseState::Braking);
    sim.fuse.reset(init());
    let rt = sim.fuse.rt();
    assert_eq!(rt.state, FuseState::Armed);
    assert_eq!((rt.seq, rt.last_t, rt.next_chunk_seq), (0, 0, 0));
    assert!(!rt.cur.is_filled() && !rt.scratch.is_filled());
    assert_eq!(rt.tally.max_s, f64::NEG_INFINITY);
    assert_eq!(rt.tally.chunks_seen, 0);
    assert_eq!(rt.window.bits, 0);
    assert_eq!(rt.last_cmd, [0.0; lictor_core::MAX_D]);
    let mut sim2 = Sim {
        fuse: Fuse::new(cfg(FuseMode::Enforce)),
        pos: [300.0, 300.0],
        vel: [0.0, 0.0],
        step: [2.0, 0.0],
    };
    sim2.fuse.reset(init());
    sim2.run(0, 9);
    let t = sim2.fuse.finish();
    assert_eq!((t.ticks, t.nominal, t.chunks_seen, t.terminal_state), (9, 9, 2, FuseState::Armed));
}

#[test]
fn without_a_velocity_the_fuse_finite_differences_the_position() {
    let mut sim = Sim::new(FuseMode::Enforce);
    let ch = sim.chunk(0, 0);
    let mut o = sim.obs(0);
    o.vel = None;
    let v = tick(&mut sim.fuse, &o, Some(&ch), 0, 0);
    assert_eq!(v.state, FuseState::Armed, "{:?}", trip_names(v.trips));
    assert_eq!(&sim.fuse.rt().vel[..2], &[0.0, 0.0], "zero on the first tick");
    // 10 px in one control period of 0.1 s = 100 px/s.
    let mut o = Obs::at(1, [310.0, 300.0], [0.0, 0.0]);
    o.vel = None;
    let v = tick(&mut sim.fuse, &o, None, 1, 0);
    assert_eq!(v.state, FuseState::Armed, "{:?}", trip_names(v.trips));
    let rt = sim.fuse.rt();
    assert_eq!(&rt.vel[..2], &[100.0, 0.0]);
    assert_eq!(&rt.prev_pos[..2], &[300.0, 300.0]);
    assert_eq!(&rt.pos[..2], &[310.0, 300.0]);
    assert!(rt.have_prev_pos);
}
