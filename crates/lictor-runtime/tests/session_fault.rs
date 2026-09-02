// SPDX-License-Identifier: MIT
//! Fail-closed session behaviour: every protocol/schema violation is an `error{fatal:true}`, the fuse latches
//! Fault, later ticks come back as `fault`/`hold` verdicts, and `episode_end` after a fatal error still yields a
//! receipt (terminal_state fault, fuse_ok false) plus a ledger entry.
//!
//! These tests drive the real `decide()` (WP-3) through the Session.

use lictor_core::{FuseMode, FuseState, SafetyEnvelope, Status};
use lictor_runtime::codec::parse_request;
use lictor_runtime::session::{Session, SessionConfig};
use lictor_runtime::wire::{Request, Response};

const ENVELOPE: &str = include_str!("../../../envelopes/pusht.base.toml");
const KEY_HEX: &str = include_str!("../../lictor-receipt/tests/fixtures/receipt/key.hex");

fn config(mode: FuseMode, out_dir: Option<std::path::PathBuf>) -> SessionConfig {
    let env = SafetyEnvelope::from_toml(ENVELOPE).unwrap();
    let mut cfg = SessionConfig::minimal(env, ENVELOPE, mode);
    cfg.key = Some(lictor_receipt::keys::parse_seed(KEY_HEX).unwrap());
    cfg.out_dir = out_dir;
    cfg
}

fn req(v: serde_json::Value) -> Request {
    parse_request(&v.to_string()).unwrap()
}

fn hello(id: u64, mode: &str, digest: &str) -> Request {
    req(
        serde_json::json!({"id": id, "kind": "hello", "proto": "lictor-wire/v1", "client": "test/0", "mode": mode,
        "embodiment_id": "gym_pusht/PushT-v0", "action_dim": 2, "pos_dim": 2, "horizon": 15, "exec_steps": 8,
        "envelope_digest": digest, "calibration_digest": null}),
    )
}

fn begin(id: u64, idx: u32) -> Request {
    req(serde_json::json!({"id": id, "kind": "episode_begin",
        "run": {"run_id": "run-fault", "arm_id": "t0-d0", "episode_index": idx, "seed": idx, "seed_pool": "eval",
                "init_state_digest": "3f".repeat(32)},
        "budget": {"delay_steps": 0, "tick_ms": 100, "exec_mode": "sync", "stitch": "drop", "on_escalate": "terminate_fail"},
        "binding": {"env": {"env_id": "gym_pusht/PushT-v0"}, "policy": {"weights_sha256": "ab".repeat(32)}, "host": {}},
        "fault_injection": null}))
}

fn rows(pos: [f64; 2]) -> Vec<[f64; 2]> {
    (1..=15).map(|i| [pos[0] + i as f64, pos[1] + 0.5 * i as f64]).collect()
}

fn tick_v(id: u64, t: u32, pos: serde_json::Value, chunk: Option<(u32, Vec<[f64; 2]>)>) -> serde_json::Value {
    let idx = t % 8;
    let chunk = match chunk {
        Some((seq, a)) => serde_json::json!({"seq": seq, "t_emit": t, "h": 15, "d": 2, "exec": 8, "a": a}),
        None => serde_json::Value::Null,
    };
    serde_json::json!({"id": id, "kind": "tick", "t": t, "idx": idx, "missed_ticks": 0,
        "obs": {"pos": pos, "vel": [10.0, 5.0], "aux": [256.0, 256.0, 0.0, 0.1], "ext": []},
        "chunk": chunk, "ack": null})
}

/// A nominal tick: the agent drifts slowly; a fresh chunk every 8 ticks.
fn tick(id: u64, t: u32) -> Request {
    let pos = [200.0 + t as f64, 200.0 + 0.5 * t as f64];
    let chunk = if t.is_multiple_of(8) { Some((t / 8, rows(pos))) } else { None };
    req(tick_v(id, t, serde_json::json!(pos), chunk))
}

fn end(id: u64, t: u32, ended_by: &str) -> Request {
    req(serde_json::json!({"id": id, "kind": "episode_end", "t": t,
        "outcome": {"steps": t, "success": false, "terminated": false, "truncated": false,
                    "max_coverage": 0.1, "final_coverage": 0.1, "reward_sum": 1.0, "ended_by": ended_by}}))
}

fn assert_error(r: &Response, code: &str) {
    match r {
        Response::Error(e) => {
            assert_eq!(e.code, code, "{}", e.message);
            assert!(e.fatal);
        }
        other => panic!("expected error {code}, got {}", serde_json::to_string(other).unwrap()),
    }
}

fn assert_fault_hold(r: &Response) {
    match r {
        Response::Verdict(v) => {
            assert_eq!(v.status, Status::Fault, "{}", serde_json::to_string(v).unwrap());
            assert_eq!(v.state, FuseState::Fault);
            assert_eq!(v.action_src, lictor_core::ActionSource::Hold);
        }
        other => panic!("expected a fault verdict, got {}", serde_json::to_string(other).unwrap()),
    }
}

fn started(dir: &tempfile::TempDir) -> (Session, u64) {
    let cfg = config(FuseMode::Enforce, Some(dir.path().to_path_buf()));
    let digest = cfg.envelope.digest_hex();
    let mut s = Session::new(cfg).unwrap();
    assert!(matches!(s.handle(hello(1, "enforce", &digest)), Response::HelloOk(_)));
    assert!(matches!(s.handle(begin(2, 0)), Response::EpisodeOk(_)));
    (s, 3)
}

#[test]
fn tick_before_episode_begin_is_fatal_then_hold() {
    let cfg = config(FuseMode::Enforce, None);
    let digest = cfg.envelope.digest_hex();
    let mut s = Session::new(cfg).unwrap();
    assert!(matches!(s.handle(hello(1, "enforce", &digest)), Response::HelloOk(_)));
    assert_error(&s.handle(tick(2, 0)), "state");
    assert!(s.fault_latched());
    assert_fault_hold(&s.handle(tick(3, 1)));
    assert_fault_hold(&s.handle(tick(4, 2)));
    // a real episode afterwards starts clean
    assert!(matches!(s.handle(begin(5, 0)), Response::EpisodeOk(_)));
    assert!(!s.fault_latched());
    match s.handle(tick(6, 0)) {
        Response::Verdict(v) => {
            assert_eq!(v.status, Status::Nominal, "{}", serde_json::to_string(&v).unwrap())
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
}

#[test]
fn unknown_nested_field_is_a_schema_error_and_latches() {
    let dir = tempfile::tempdir().unwrap();
    let (mut s, id) = started(&dir);
    assert!(matches!(s.handle(tick(id, 0)), Response::Verdict(_)));
    let bad = "{\"id\":4,\"kind\":\"tick\",\"t\":1,\"idx\":1,\"missed_ticks\":0,\"obs\":{\"pos\":[1.0,2.0],\"vel\":null,\"aux\":[],\"ext\":[],\"extra\":1},\"chunk\":null,\"ack\":null}";
    let e = parse_request(bad).unwrap_err();
    assert!(e.to_string().contains("unknown field"));
    let r = s.handle_bad_line(lictor_runtime::codec::peek_id(bad), &e.to_string());
    assert_error(&r, "schema");
    if let Response::Error(e) = &r {
        assert_eq!(e.id, 4);
    }
    assert!(s.fault_latched());
    assert_fault_hold(&s.handle(tick(5, 1)));
    assert_fault_hold(&s.handle(tick(6, 2)));
}

#[test]
fn h_mismatch_is_a_schema_error() {
    let dir = tempfile::tempdir().unwrap();
    let (mut s, id) = started(&dir);
    let pos = [200.0, 200.0];
    let short: Vec<[f64; 2]> = rows(pos).into_iter().take(14).collect();
    let r = s.handle(req(tick_v(id, 0, serde_json::json!(pos), Some((0, short)))));
    assert_error(&r, "schema");
    if let Response::Error(e) = &r {
        assert!(e.message.contains("14 rows"), "{}", e.message);
    }
    assert_fault_hold(&s.handle(tick(id + 1, 1)));
}

#[test]
fn non_monotonic_id_is_a_protocol_error() {
    let dir = tempfile::tempdir().unwrap();
    let (mut s, id) = started(&dir);
    assert!(matches!(s.handle(tick(id, 0)), Response::Verdict(_)));
    assert_error(&s.handle(tick(id, 1)), "protocol");
    assert_error(&s.handle(tick(id - 1, 1)), "protocol");
    assert_fault_hold(&s.handle(tick(id + 5, 1)));
}

#[test]
fn time_jump_without_missed_ticks_is_a_schema_trip() {
    let dir = tempfile::tempdir().unwrap();
    let (mut s, id) = started(&dir);
    assert!(matches!(s.handle(tick(id, 0)), Response::Verdict(_)));
    match s.handle(tick(id + 1, 2)) {
        Response::Verdict(v) => {
            assert!(v.trips.iter().any(|t| t == "schema"), "{:?}", v.trips);
            assert_eq!(v.status, Status::Fault);
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    assert_fault_hold(&s.handle(tick(id + 2, 3)));
}

#[test]
fn null_in_pos_is_a_nonfinite_fault() {
    let dir = tempfile::tempdir().unwrap();
    let (mut s, id) = started(&dir);
    assert!(matches!(s.handle(tick(id, 0)), Response::Verdict(_)));
    match s.handle(req(tick_v(id + 1, 1, serde_json::json!([null, 200.0]), None))) {
        Response::Verdict(v) => {
            assert_eq!(v.status, Status::Fault);
            assert!(v.trips.iter().any(|t| t == "nonfinite"), "{:?}", v.trips);
            assert_eq!(v.action_src, lictor_core::ActionSource::Hold);
            assert!(v.action.iter().all(|x| x.is_finite()));
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
}

#[test]
fn episode_end_after_fatal_error_still_writes_a_receipt() {
    let dir = tempfile::tempdir().unwrap();
    let (mut s, id) = started(&dir);
    assert!(matches!(s.handle(tick(id, 0)), Response::Verdict(_)));
    assert_error(&s.handle(tick(id, 1)), "protocol");
    assert_fault_hold(&s.handle(tick(id + 1, 1)));
    assert_fault_hold(&s.handle(tick(id + 2, 2)));
    let r = s.handle(end(id + 3, 3, "fault"));
    let m = match r {
        Response::EpisodeReceipt(m) => m,
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    };
    assert_eq!(m.counts.terminal_state, FuseState::Fault);
    assert!(!m.fuse_ok);
    assert!(m.fuse_notes.iter().any(|n| n.contains("Fault")), "{:?}", m.fuse_notes);
    assert_eq!(m.counts.ticks, 3);
    assert_eq!(m.ledger_seq, 0);
    let ledger =
        lictor_receipt::read_ledger(dir.path().join("run-fault/t0-d0/ledger.jsonl").as_path()).unwrap();
    assert_eq!(ledger.len(), 1);
    assert!(!ledger[0].fuse_ok && !ledger[0].success);
    let text = std::fs::read_to_string(dir.path().join(&m.receipt_path)).unwrap();
    let sr: lictor_receipt::SignedReceipt = serde_json::from_str(&text).unwrap();
    assert_eq!(sr.body.outcome.ended_by, "fault");
    let report = lictor_receipt::verify(&sr, None);
    assert!(report.intact(), "{:?}", report.notes);
    assert!(!report.fuse_ok);
    // the session is clean again for the next episode
    assert!(!s.fault_latched());
    assert!(matches!(s.handle(begin(id + 4, 1)), Response::EpisodeOk(_)));
    assert!(matches!(s.handle(tick(id + 5, 0)), Response::Verdict(v) if v.status == Status::Nominal));
}

#[test]
fn episode_end_without_episode_and_double_begin_are_state_errors() {
    let cfg = config(FuseMode::Observe, None);
    let digest = cfg.envelope.digest_hex();
    let mut s = Session::new(cfg).unwrap();
    assert!(matches!(s.handle(hello(1, "observe", &digest)), Response::HelloOk(_)));
    assert_error(&s.handle(end(2, 0, "fault")), "state");
    assert!(matches!(s.handle(begin(3, 0)), Response::EpisodeOk(_)));
    assert_error(&s.handle(begin(4, 1)), "state");
    assert_error(&s.handle(hello(5, "observe", &digest)), "state");
    assert!(s.fault_latched());
    assert!(matches!(
        s.handle(Request::Bye(lictor_runtime::wire::ByeReq { id: 6 })),
        Response::ByeOk { id: 6 }
    ));
}
