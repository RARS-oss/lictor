// SPDX-License-Identifier: MIT
//! `fixtures/episode/synthetic_300.json` through `Fuse` in both modes: Observe passes the raw policy row through
//! on EVERY tick while counting the violations that reached the environment and still walking the state
//! machine (Clamped, Braking, Held, re-arm); Enforce substitutes and lets nothing through.

mod common;

use common::*;
use lictor_core::{ActionSource, FuseMode, FuseState, ReasonCode, TripMask};
use lictor_fuse::Fuse;

fn visited(vs: &[lictor_core::SafetyVerdict], s: FuseState) -> bool {
    vs.iter().any(|v| v.state == s)
}

#[test]
fn observe_mode_passes_the_policy_action_through_on_every_tick() {
    let ep = Episode::load();
    assert_eq!(ep.obs.len(), 300);
    assert_eq!(ep.chunks.len(), 38);
    let mut fuse = Fuse::new(cfg(FuseMode::Observe));
    fuse.reset(init());
    let vs = run_episode(&mut fuse, &ep);
    let tl = timeline(&vs);

    for v in &vs {
        assert_eq!(action2(v), ep.policy_row(v.t), "t={} action != policy row ({tl})", v.t);
        assert!(!v.substituted, "t={} substituted in Observe mode ({tl})", v.t);
        assert_eq!(v.action_src, ActionSource::Policy, "t={} src ({tl})", v.t);
        assert_eq!(v.trips & TripMask::TIER0_HARD & !TripMask::BRAKE, 0, "t={} guard bit ({tl})", v.t);
        assert_eq!(v.violation_reached_env, v.state != FuseState::Armed && v.state != FuseState::Watching);
    }
    let tally = fuse.finish();
    assert!(tally.violations_reached_env > 0, "no violation counted ({tl})");
    assert_eq!(tally.substituted, 0);
    assert!(visited(&vs, FuseState::Clamped), "Clamped never visited ({tl})");
    assert!(visited(&vs, FuseState::Braking), "Braking never visited ({tl})");
    assert!(visited(&vs, FuseState::Held), "Held never visited ({tl})");
    assert_eq!(tally.ticks, 300);
    assert_eq!(tally.chunks_seen, 38);
    assert_eq!(tally.chunks_rejected, 1, "only chunk 12 carries a hard (brake) trip ({tl})");
    assert_eq!(tally.terminal_state, FuseState::Armed, "{tl}");
    assert_eq!(tally.rearms, 1, "{tl}");
    assert_eq!(tally.holds, 1, "{tl}");
    assert_eq!(tally.first_trip_tick, Some(40));
    assert_eq!(tally.first_trip_reason, Some(ReasonCode::ClampWorkspace));
    assert_eq!(tally.first_stop_tick, Some(96));
    // Observe mode assumes the brake stopped the robot: Held follows Braking after stop_confirm_ticks = 2.
    assert_eq!(vs[96].state, FuseState::Braking, "{tl}");
    assert_eq!(vs[98].state, FuseState::Held, "{tl}");
    // No calibration: the base gate scores the 8 manifest features against tau = +inf, so `s` is finite once
    // the boundary features are valid and nothing ever fires or enters the window.
    assert!(tally.max_s.is_finite(), "max_s {}", tally.max_s);
    assert_ne!(vs[0].scores.valid & lictor_core::Feat::CHUNK_BOUNDARY_MASK, 0, "boundary features at t = 0");
    assert_eq!(vs[0].scores.valid & lictor_core::Feat::Tce.bit(), 0, "tce needs a previous chunk");
    assert!(vs.iter().all(|v| v.scores.fired == 0 && v.window_hits == 0 && v.tau == f64::INFINITY));
    // Every Armed -> Armed tick (row 23) in Observe mode says so; the clear-down (row 21) and the auto re-arm
    // (row 11) keep their own reasons.
    let armed_stays = vs.iter().filter(|v| v.state == FuseState::Armed && v.prev_state == FuseState::Armed);
    assert!(armed_stays.clone().count() > 250);
    assert!(armed_stays.clone().all(|v| v.reason == ReasonCode::ObserveOnly));
    assert!(vs.iter().any(|v| v.reason == ReasonCode::RearmedAuto), "{tl}");
    assert!(
        vs.iter().any(|v| v.state == FuseState::Armed && v.reason == ReasonCode::Ok),
        "clear-down ({tl})"
    );
}

#[test]
fn enforce_mode_lets_no_violation_through_and_brakes_to_the_clamped_position() {
    let ep = Episode::load();
    let cfg = cfg(FuseMode::Enforce);
    let mut fuse = Fuse::new(cfg.clone());
    fuse.reset(init());
    let vs = run_episode(&mut fuse, &ep);
    let tl = timeline(&vs);
    let tally = fuse.finish();

    assert_eq!(tally.violations_reached_env, 0, "{tl}");
    assert!(vs.iter().all(|v| !v.violation_reached_env));
    assert!(tally.substituted > 0, "{tl}");
    assert_eq!(tally.substituted, vs.iter().filter(|v| v.action_src != ActionSource::Policy).count() as u32);

    let first_brake =
        vs.iter().find(|v| v.state == FuseState::Braking).unwrap_or_else(|| panic!("no Braking ({tl})"));
    let t = first_brake.t;
    let pos = [ep.obs[t as usize].pos[0], ep.obs[t as usize].pos[1]];
    assert_eq!(action2(first_brake), clamp2(&cfg, pos), "first Braking tick action (t={t}, {tl})");
    assert_eq!(first_brake.action_src, ActionSource::Brake);
    assert!(first_brake.substituted);
    assert_eq!(first_brake.reason, ReasonCode::BrakeInfeasible);
    assert_ne!(first_brake.trips & TripMask::BRAKE, 0);
    assert!(first_brake.brake_margin < 0.0, "margin {}", first_brake.brake_margin);
    assert_eq!(t, 96, "{tl}");

    // The clamped chunk 5: Clamped at t = 40 with the projected row, then cleared after clear_ticks clean ticks.
    let c40 = &vs[40];
    assert_eq!(c40.state, FuseState::Clamped, "{tl}");
    assert_eq!(c40.action_src, ActionSource::Clamped);
    assert_eq!(c40.clamped_dims, 0b10, "only y moved");
    assert_eq!(c40.reason, ReasonCode::ClampWorkspace);
    // Row 0 of chunk 5 is inside the box, so the projected row equals the raw row even though the chunk tripped.
    assert_eq!(action2(c40), ep.policy_row(40));
    assert_eq!(vs[45].state, FuseState::Armed, "{tl}");

    // The hold latches the position the robot stopped at and stays put; a clean chunk re-arms the fuse.
    let held: Vec<_> = vs.iter().filter(|v| v.state == FuseState::Held).collect();
    assert!(!held.is_empty(), "{tl}");
    let h0 = action2(held[0]);
    assert!(
        held.iter().all(|v| action2(v) == h0 && v.action_src == ActionSource::Hold),
        "hold drifted ({tl})"
    );
    assert_eq!(tally.holds, 1, "{tl}");
    assert_eq!(tally.rearms, 1, "{tl}");
    assert!(vs.iter().any(|v| v.reason == ReasonCode::RearmedAuto), "{tl}");
    assert_eq!(tally.terminal_state, FuseState::Armed, "{tl}");
    assert_eq!(tally.chunks_rejected, 1);

    // Policy rows in Enforce mode are bit-for-bit the raw rows.
    for v in vs.iter().filter(|v| v.action_src == ActionSource::Policy) {
        assert_eq!(action2(v), ep.policy_row(v.t), "t={}", v.t);
        assert!(!v.substituted);
    }
    // Nothing after the re-arm trips again (the stall is Tier-1 territory, disarmed here).
    assert!(vs.iter().filter(|v| v.t >= 130).all(|v| v.state == FuseState::Armed), "{tl}");
}

#[test]
fn both_modes_compute_identical_scores_and_trips_until_the_first_intervention() {
    let ep = Episode::load();
    let mut obs = Fuse::new(cfg(FuseMode::Observe));
    obs.reset(init());
    let vo = run_episode(&mut obs, &ep);
    let mut enf = Fuse::new(cfg(FuseMode::Enforce));
    enf.reset(init());
    let ve = run_episode(&mut enf, &ep);
    for (a, b) in vo.iter().zip(&ve).take(40) {
        assert_eq!(a.trips, b.trips, "t={}", a.t);
        assert_eq!(a.scores.f, b.scores.f, "t={}", a.t);
        assert_eq!(a.scores.valid, b.scores.valid, "t={}", a.t);
        assert_eq!(a.brake_margin.to_bits(), b.brake_margin.to_bits(), "t={}", a.t);
        assert_eq!(a.state, b.state, "t={}", a.t);
    }
    // Same detections at the two rogue boundaries in both modes.
    for t in [40usize, 96, 104] {
        assert_eq!(vo[t].trips, ve[t].trips, "t={t}");
        assert_eq!(vo[t].clamped_dims, ve[t].clamped_dims, "t={t}");
    }
}
