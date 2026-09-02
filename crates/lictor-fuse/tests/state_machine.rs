// SPDX-License-Identifier: MIT
//! Drives `lictor_fuse::fsm::next` through `fixtures/fsm/transitions.json`: every row of the ARCHITECTURE sec 6
//! table (1-23 plus 17b) at least once, with the expected `(to, extra_trips, reason)`, the counter side effects,
//! the hold latch, and the ignored-ack / unreachable-combination guards.

mod common;

use common::*;
use lictor_core::{ClampMode, FuseMode, FuseState, RearmPolicy, ReasonCode, TripMask};
use lictor_fuse::{fsm, FsmInput, FuseRt};
use serde_json::Value;
use std::collections::BTreeSet;

const CASES: &str = include_str!("fixtures/fsm/transitions.json");

/// The test position: y is outside the box so a latched hold is visibly clamped.
const POS: [f64; 2] = [300.0, 600.0];

fn apply_rt(rt: &mut FuseRt, v: &Value) {
    let Some(m) = v.as_object() else { return };
    for (k, x) in m {
        let u = |x: &Value| x.as_u64().unwrap_or_else(|| panic!("integer expected for {k}"));
        match k.as_str() {
            "clean_run" => rt.clean_run = u(x) as u16,
            "clamp_streak" => rt.clamp_streak = u(x) as u8,
            "clamps" => rt.clamps = u(x) as u16,
            "brake_ticks" => rt.brake_ticks = u(x) as u16,
            "stopped_ticks" => rt.stopped_ticks = u(x) as u8,
            "held_ticks" => rt.held_ticks = u(x) as u16,
            "held_clean" => rt.held_clean = u(x) as u16,
            "escalated_ticks" => rt.escalated_ticks = u(x) as u32,
            "rearms" => rt.rearms = u(x) as u8,
            "handoff_seq" => rt.handoff_seq = u(x) as u32,
            "handoff_pending" => rt.handoff_pending = x.as_bool().unwrap(),
            "window_bits" => rt.window.bits = u(x),
            "last_nonce0" => rt.last_nonce[0] = u(x),
            "last_nonce1" => rt.last_nonce[1] = u(x),
            other => panic!("unknown rt field {other}"),
        }
    }
}

fn check_rt(name: &str, rt: &FuseRt, v: &Value) {
    let Some(m) = v.as_object() else { return };
    for (k, x) in m {
        let got: u64 = match k.as_str() {
            "clean_run" => rt.clean_run.into(),
            "clamp_streak" => rt.clamp_streak.into(),
            "clamps" => rt.clamps.into(),
            "brake_ticks" => rt.brake_ticks.into(),
            "stopped_ticks" => rt.stopped_ticks.into(),
            "held_ticks" => rt.held_ticks.into(),
            "held_clean" => rt.held_clean.into(),
            "escalated_ticks" => rt.escalated_ticks.into(),
            "rearms" => rt.rearms.into(),
            "handoff_seq" => rt.handoff_seq.into(),
            "handoff_pending" => {
                assert_eq!(rt.handoff_pending, x.as_bool().unwrap(), "{name}: handoff_pending");
                continue;
            }
            "window_bits" => rt.window.bits,
            "last_nonce0" => rt.last_nonce[0],
            "last_nonce1" => rt.last_nonce[1],
            other => panic!("unknown expected rt field {other}"),
        };
        assert_eq!(got, x.as_u64().unwrap(), "{name}: rt.{k}");
    }
}

fn input_of(v: &Value) -> FsmInput {
    let b = |k: &str| v[k].as_bool().unwrap_or_else(|| panic!("bool {k}"));
    let ack = v["ack"].as_object().map(|a| {
        common::ack(
            decision_of(a["decision"].as_str().unwrap()),
            a["slot"].as_u64().unwrap() as u8,
            a["nonce"].as_u64().unwrap(),
            a["handoff_seq"].as_u64().unwrap() as u32,
        )
    });
    FsmInput {
        trips: trips_of(&v["trips"]),
        predictive: b("predictive"),
        warn: b("warn"),
        soft_clampable: b("soft_clampable"),
        stopped: b("stopped"),
        chunk_boundary: b("chunk_boundary"),
        chunk_ok: b("chunk_ok"),
        ack,
    }
}

#[test]
fn every_row_of_the_table_matches_the_fixture() {
    let doc: Value = serde_json::from_str(CASES).expect("fixture parses");
    let cases = doc["cases"].as_array().expect("cases");
    let mut rows_seen = BTreeSet::new();
    for c in cases {
        let name = c["name"].as_str().unwrap();
        rows_seen.insert(c["row"].as_str().unwrap().to_string());

        let mode = match c["cfg"]["mode"].as_str() {
            Some("observe") => FuseMode::Observe,
            _ => FuseMode::Enforce,
        };
        let cfg = cfg_with(mode, |e| {
            if let Some(cm) = c["cfg"]["clamp_mode"].as_str() {
                e.clamp_mode = if cm == "off" { ClampMode::Off } else { ClampMode::Project };
            }
            if let Some(r) = c["cfg"]["rearm"].as_str() {
                e.hysteresis.rearm = if r == "ack_only" { RearmPolicy::AckOnly } else { RearmPolicy::Auto };
            }
        });

        let mut rt = FuseRt::new();
        rt.reset(init());
        rt.pos[0] = POS[0];
        rt.pos[1] = POS[1];
        rt.state = state_of(c["from"].as_str().unwrap());
        apply_rt(&mut rt, &c["rt"]);
        let inp = input_of(&c["input"]);

        let (to, extra, reason) = fsm::next(&cfg, &mut rt, inp);

        let exp = &c["expect"];
        assert_eq!(to, state_of(exp["to"].as_str().unwrap()), "{name}: to");
        assert_eq!(rt.state, to, "{name}: rt.state is the returned state");
        assert_eq!(trip_names(extra), trip_names(trips_of(&exp["extra_trips"])), "{name}: extra trips");
        assert_eq!(reason_name(reason), exp["reason"].as_str().unwrap(), "{name}: reason");
        check_rt(name, &rt, &exp["rt"]);
        if let Some(h) = exp["hold"].as_array() {
            let want = [h[0].as_f64().unwrap(), h[1].as_f64().unwrap()];
            assert_eq!([rt.hold[0], rt.hold[1]], want, "{name}: hold");
        }
    }
    let required: BTreeSet<String> = (1..=23).map(|r| r.to_string()).chain(["17b".to_string()]).collect();
    let missing: Vec<&String> = required.difference(&rows_seen).collect();
    assert!(missing.is_empty(), "rows without a fixture case: {missing:?}");
    assert!(cases.len() >= 24, "{} cases", cases.len());
}

#[test]
fn extra_trips_are_only_the_six_fsm_bits() {
    let doc: Value = serde_json::from_str(CASES).unwrap();
    let fsm_bits = TripMask::TIER1_CP
        | TripMask::CLAMP_BUDGET
        | TripMask::HANDOFF_TIMEOUT
        | TripMask::OPERATOR_ABORT
        | TripMask::BRAKE_TIMEOUT
        | TripMask::REARM_BUDGET;
    let mut raised = 0u32;
    for c in doc["cases"].as_array().unwrap() {
        let extra = trips_of(&c["expect"]["extra_trips"]);
        assert_eq!(extra & !fsm_bits, 0, "{}: extra trips outside the FSM set", c["name"]);
        raised |= extra;
    }
    assert_eq!(raised, fsm_bits, "the fixture raises every FSM bit at least once");
}

#[test]
fn clean_run_counts_in_every_state_and_ack_needs_both_integers() {
    let cfg = cfg(FuseMode::Enforce);
    let mut rt = FuseRt::new();
    rt.reset(init());
    let clean = FsmInput {
        trips: 0,
        predictive: false,
        warn: false,
        soft_clampable: true,
        stopped: true,
        chunk_boundary: false,
        chunk_ok: false,
        ack: None,
    };
    for s in
        [FuseState::Braking, FuseState::Held, FuseState::Escalated, FuseState::Fault, FuseState::Terminated]
    {
        rt.state = s;
        let before = rt.clean_run;
        fsm::next(&cfg, &mut rt, clean);
        assert_eq!(rt.clean_run, before + 1, "clean_run in {s:?}");
    }
    let dirty = FsmInput { trips: TripMask::WORKSPACE, ..clean };
    rt.state = FuseState::Held;
    fsm::next(&cfg, &mut rt, dirty);
    assert_eq!(rt.clean_run, 0);

    // Held + Resume with the right handoff_seq but a stale nonce: ignored (row 13), then accepted once fresh.
    rt.state = FuseState::Held;
    rt.handoff_seq = 4;
    rt.last_nonce[2] = 10;
    let stale = FsmInput { ack: Some(ack(lictor_core::AckDecision::Resume, 2, 10, 4)), ..clean };
    let (to, _, reason) = fsm::next(&cfg, &mut rt, stale);
    assert_eq!((to, reason), (FuseState::Held, ReasonCode::HeldStandstill));
    let fresh = FsmInput { ack: Some(ack(lictor_core::AckDecision::Resume, 2, 11, 4)), ..clean };
    let (to, _, reason) = fsm::next(&cfg, &mut rt, fresh);
    assert_eq!((to, reason), (FuseState::Armed, ReasonCode::RearmedAck));
    assert_eq!(rt.last_nonce[2], 11);
    assert_eq!(rt.rearms, 1);
}
