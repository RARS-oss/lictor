// SPDX-License-Identifier: MIT
//! A full synthetic 300-tick episode through `Session` in a tempdir, plus the startup refusals, the crash receipt,
//! determinism of the verdict chain, and the persisted ack nonce.
//!
//! Every test drives the real `decide()` (WP-3) through the Session.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lictor_core::{
    AckDecision, CalMethod, CalibrationC, ExecMode, FuseMode, FuseState, GateSpec, RearmPolicy,
    SafetyEnvelope, Status, NFEAT,
};
use lictor_receipt::{
    read_ledger, sign_ack, verify, verify_ledger, verify_ticks_file, BudgetBinding, RunBinding,
    SignedReceipt, TickEvent, TimingEvent,
};
use lictor_runtime::codec::parse_request;
use lictor_runtime::episode::{episode_paths, write_crash_episode, NOTE_CRASH};
use lictor_runtime::session::{CalibrationLoaded, Session, SessionConfig, NONCE_FILE};
use lictor_runtime::wire::{Request, Response, VerdictMsg};

const ENVELOPE: &str = include_str!("../../../envelopes/pusht.base.toml");
const KEY_HEX: &str = include_str!("../../lictor-receipt/tests/fixtures/receipt/key.hex");
const OPERATOR_HEX: &str = include_str!("../../lictor-receipt/tests/fixtures/receipt/operator.hex");

fn envelope() -> SafetyEnvelope {
    SafetyEnvelope::from_toml(ENVELOPE).unwrap()
}

fn config(env: SafetyEnvelope, mode: FuseMode, out_dir: Option<PathBuf>) -> SessionConfig {
    let mut cfg = SessionConfig::minimal(env, ENVELOPE, mode);
    cfg.key = Some(lictor_receipt::keys::parse_seed(KEY_HEX).unwrap());
    cfg.out_dir = out_dir;
    cfg.lictor_sha256 = "5d".repeat(32);
    cfg
}

fn req(v: serde_json::Value) -> Request {
    parse_request(&v.to_string()).unwrap()
}

fn hello(id: u64, mode: &str, digest: &str, calibration: Option<&str>) -> Request {
    req(
        serde_json::json!({"id": id, "kind": "hello", "proto": "lictor-wire/v1", "client": "lictor_client/0.1.0",
        "mode": mode, "embodiment_id": "gym_pusht/PushT-v0", "action_dim": 2, "pos_dim": 2, "horizon": 15,
        "exec_steps": 8, "envelope_digest": digest, "calibration_digest": calibration}),
    )
}

fn begin(id: u64, arm: &str, idx: u32, weights: &str) -> Request {
    req(serde_json::json!({"id": id, "kind": "episode_begin",
        "run": {"run_id": "run-ep", "arm_id": arm, "episode_index": idx, "seed": idx, "seed_pool": "eval",
                "init_state_digest": "3f".repeat(32)},
        "budget": {"delay_steps": 0, "tick_ms": 100, "exec_mode": "sync", "stitch": "drop", "on_escalate": "terminate_fail"},
        "binding": {"env": {"env_id": "gym_pusht/PushT-v0", "max_episode_steps": "300"},
                    "policy": {"repo_id": "lerobot/diffusion_pusht", "weights_sha256": weights},
                    "host": {"OMP_NUM_THREADS": "1"}},
        "fault_injection": null,
        "inputs": {"harness/pusht_rollout.py": "22".repeat(32)}}))
}

fn end(id: u64, steps: u32, ended_by: &str) -> Request {
    req(serde_json::json!({"id": id, "kind": "episode_end", "t": steps,
        "outcome": {"steps": steps, "success": true, "terminated": true, "truncated": false,
                    "max_coverage": 0.97, "final_coverage": 0.97, "reward_sum": 210.5, "ended_by": ended_by}}))
}

fn chunk_rows(pos: [f64; 2], step: [f64; 2]) -> Vec<[f64; 2]> {
    (1..=15).map(|i| [pos[0] + step[0] * i as f64, pos[1] + step[1] * i as f64]).collect()
}

fn tick_json(
    id: u64,
    t: u32,
    pos: [f64; 2],
    vel: [f64; 2],
    chunk: Option<Vec<[f64; 2]>>,
    ack: serde_json::Value,
) -> serde_json::Value {
    let chunk = match chunk {
        Some(a) => serde_json::json!({"seq": t / 8, "t_emit": t, "h": 15, "d": 2, "exec": 8, "a": a}),
        None => serde_json::Value::Null,
    };
    serde_json::json!({"id": id, "kind": "tick", "t": t, "idx": t % 8, "missed_ticks": 0,
        "obs": {"pos": pos, "vel": vel, "aux": [256.0, 256.0, 0.0, 0.1 + 0.002 * t as f64], "ext": []},
        "chunk": chunk, "ack": ack})
}

/// Nominal synthetic motion: the agent drifts at (10, 5) px/s from (150, 150); chunks continue the drift.
fn nominal_tick(id: u64, t: u32) -> Request {
    let pos = [150.0 + t as f64, 150.0 + 0.5 * t as f64];
    let chunk = if t.is_multiple_of(8) { Some(chunk_rows(pos, [1.0, 0.5])) } else { None };
    req(tick_json(id, t, pos, [10.0, 5.0], chunk, serde_json::Value::Null))
}

fn verdict(r: Response) -> Box<VerdictMsg> {
    match r {
        Response::Verdict(v) => v,
        other => panic!("expected a verdict, got {}", serde_json::to_string(&other).unwrap()),
    }
}

/// hello + episode_begin + `n` nominal ticks + episode_end; returns (first verdict JSON, receipt message).
fn run_nominal(
    s: &mut Session,
    mode: &str,
    arm: &str,
    idx: u32,
    n: u32,
) -> (String, lictor_runtime::wire::EpisodeReceiptMsg) {
    let digest = envelope().digest_hex();
    assert!(matches!(s.handle(hello(1, mode, &digest, None)), Response::HelloOk(_)));
    assert!(matches!(s.handle(begin(2, arm, idx, "ab")), Response::EpisodeOk(_)));
    let mut first = String::new();
    for t in 0..n {
        let r = s.handle(nominal_tick(3 + u64::from(t), t));
        if t == 0 {
            first = serde_json::to_string(&r).unwrap();
        }
        let v = verdict(r);
        assert_eq!(v.seq, t);
        assert_eq!(v.t, t);
        assert_eq!(v.status, Status::Nominal, "tick {t}: {}", serde_json::to_string(&v).unwrap());
        assert_eq!(v.state, FuseState::Armed);
        s.note_io_ns(v.seq, 40_000 + u64::from(t));
    }
    match s.handle(end(3 + u64::from(n), n, "success")) {
        Response::EpisodeReceipt(m) => (first, m),
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
}

fn read_events<T: serde::de::DeserializeOwned>(p: &Path) -> (serde_json::Value, Vec<T>) {
    let text = std::fs::read_to_string(p).unwrap();
    let mut lines = text.lines();
    let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    let events = lines.map(|l| serde_json::from_str(l).unwrap()).collect();
    (header, events)
}

#[test]
fn full_300_tick_observe_episode_verifies() {
    let dir = tempfile::tempdir().unwrap();
    let mut s = Session::new(config(envelope(), FuseMode::Observe, Some(dir.path().to_path_buf()))).unwrap();
    let (first, m) = run_nominal(&mut s, "observe", "obs-d0", 7, 300);
    // Tier 1 disarmed: tau = +inf is `null` on the wire and maps back; `s` is finite here because compile()
    // embeds the envelope gate even without a calibration (features are recorded for calibration traces) --
    // see `empty_gate_first_tick_has_null_s` for the s = -inf case.
    assert!(first.contains("\"tau\":null"), "{first}");
    let v: serde_json::Value = serde_json::from_str(&first).unwrap();
    assert_eq!(v["tau"].as_f64().unwrap_or(f64::INFINITY), f64::INFINITY);
    assert!(v["scores"]["s"].is_f64() || v["scores"]["s"].is_null(), "{first}");
    assert!(v["action"].as_array().unwrap().iter().all(|x| x.is_f64()));
    assert!(v["brake_margin"].is_f64());

    assert_eq!(m.counts.ticks, 300);
    assert!(!m.fuse_ok, "observe receipts do not attest protection");
    assert!(m.fuse_notes.iter().any(|n| n.contains("observed")));
    assert_eq!(m.ledger_seq, 0);
    assert!(m.receipt_path.ends_with("run-ep/obs-d0/receipts/000007.json"), "{}", m.receipt_path);
    assert!(m.ticks_path.ends_with("run-ep/obs-d0/ticks/000007.jsonl"));

    let paths = episode_paths(dir.path(), "run-ep", "obs-d0", 7);
    let sr: SignedReceipt = serde_json::from_str(&std::fs::read_to_string(&paths.receipt).unwrap()).unwrap();
    let pubkey = lictor_receipt::pubkey_hex(&lictor_receipt::keys::parse_seed(KEY_HEX).unwrap());
    let report = verify(&sr, Some(&pubkey));
    assert!(report.intact(), "{:?}", report.notes);
    assert!(report.counts_ok && report.envelope_digest_ok && report.pubkey_ok);
    assert_eq!(sr.body.client, "lictor_client/0.1.0");
    assert_eq!(sr.body.counts.ticks, 300);
    assert_eq!(sr.body.verdict_events, 300);
    assert_eq!(sr.body.timing_events, 300);
    assert_eq!(sr.body.outcome.steps, 300);
    assert_eq!(sr.body.ticks.len(), 32);
    assert_eq!(sr.body.ticks[0].seq, 268);
    assert_eq!(sr.body.latency.n, 300);
    assert!(sr.body.latency.label.contains("not a real-time environment"));
    assert_eq!(sr.body.inputs.get("lictor:bin").map(String::as_str), Some("5d".repeat(32).as_str()));
    assert!(sr.body.inputs.contains_key("lictor:envelope"));
    assert!(sr.body.inputs.contains_key("harness/pusht_rollout.py"));
    assert!(!sr.body.inputs.contains_key("lictor:calibration"));
    assert_eq!(sr.body.budget.mode, FuseMode::Observe);
    assert!(!sr.body.budget.tier1_armed);
    assert_eq!(sr.body.ledger_prev, None);
    assert_eq!(sr.body_digest, m.body_digest);

    // ticks file: header + 300 canonical events chaining to the signed head
    let (header, ticks) = read_events::<TickEvent>(&paths.ticks);
    assert_eq!(header["schema"], "lictor-ticks/v1");
    assert_eq!(header["episode_index"], 7);
    assert_eq!(ticks.len(), 300);
    let chain = verify_ticks_file(&sr, &ticks);
    assert!(chain.ok);
    assert_eq!(chain.head, sr.body.verdict_chain_head);
    assert_eq!(chain.head, m.verdict_chain_head);
    assert_eq!(ticks[268].prev, sr.body.ticks[0].prev);
    let (theader, timing) = read_events::<TimingEvent>(&paths.timing);
    assert_eq!(theader["stream"], "timing");
    assert_eq!(timing.len(), 300);
    let tchain = lictor_receipt::verify_timing_chain(&timing);
    assert!(tchain.ok);
    assert_eq!(tchain.head, sr.body.timing_chain_head);
    assert_eq!(timing[5].io_ns, 40_005, "note_io_ns lands in the timing entry of that seq");

    let ledger = read_ledger(&paths.ledger).unwrap();
    assert_eq!(ledger.len(), 1);
    let lr = verify_ledger(&ledger);
    assert!(lr.chain_ok);
    assert_eq!(lr.head, m.ledger_head);
    assert_eq!(ledger[0].receipt_digest, m.body_digest);

    // a second episode in the same session links to the first ledger entry
    let (_, m2) = run_second(&mut s, 8);
    assert_eq!(m2.ledger_seq, 1);
    let sr2: SignedReceipt = serde_json::from_str(
        &std::fs::read_to_string(episode_paths(dir.path(), "run-ep", "obs-d0", 8).receipt).unwrap(),
    )
    .unwrap();
    assert_eq!(sr2.body.ledger_prev.as_deref(), Some(m.ledger_head.as_str()));
}

#[test]
fn empty_gate_first_tick_has_null_s() {
    let mut env = envelope();
    env.gate = Vec::new();
    let digest = env.digest_hex();
    let mut s = Session::new(config(env, FuseMode::Observe, None)).unwrap();
    assert!(matches!(s.handle(hello(1, "observe", &digest, None)), Response::HelloOk(_)));
    assert!(matches!(s.handle(begin(2, "obs-d0", 0, "ab")), Response::EpisodeOk(_)));
    let r = s.handle(nominal_tick(3, 0));
    let text = serde_json::to_string(&r).unwrap();
    assert!(text.contains("\"s\":null"), "{text}");
    assert!(text.contains("\"tau\":null"), "{text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["scores"]["s"].as_f64().unwrap_or(f64::NEG_INFINITY), f64::NEG_INFINITY);
    assert_eq!(v["tau"].as_f64().unwrap_or(f64::INFINITY), f64::INFINITY);
    let vm = verdict(r);
    assert_eq!(vm.scores.s, f64::NEG_INFINITY);
    assert_eq!(vm.tau, f64::INFINITY);
    assert!(vm.scores.f.iter().all(|x| x.is_finite()));
}

fn run_second(s: &mut Session, idx: u32) -> (String, lictor_runtime::wire::EpisodeReceiptMsg) {
    let base = 1000u64;
    assert!(matches!(s.handle(begin(base, "obs-d0", idx, "ab")), Response::EpisodeOk(_)));
    for t in 0..20 {
        verdict(s.handle(nominal_tick(base + 1 + u64::from(t), t)));
    }
    match s.handle(end(base + 100, 20, "truncated")) {
        Response::EpisodeReceipt(m) => (String::new(), m),
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
}

#[test]
fn hello_digest_mismatch_is_fatal() {
    let mut s = Session::new(config(envelope(), FuseMode::Observe, None)).unwrap();
    match s.handle(hello(1, "observe", &"00".repeat(32), None)) {
        Response::Error(e) => {
            assert_eq!(e.code, "envelope");
            assert!(e.fatal);
            assert!(e.message.contains("envelope_digest"));
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    let digest = envelope().digest_hex();
    match s.handle(hello(2, "enforce", &digest, None)) {
        Response::Error(e) => assert!(e.message.contains("mode"), "{}", e.message),
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    match s.handle(hello(3, "observe", &digest, Some("e1b8"))) {
        Response::Error(e) => assert!(e.message.contains("calibration_digest"), "{}", e.message),
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    let mut wrong: serde_json::Value = serde_json::from_str(&serde_json::json!({"id": 4, "kind": "hello", "proto": "lictor-wire/v0",
        "client": "t", "mode": "observe", "embodiment_id": "gym_pusht/PushT-v0", "action_dim": 2, "pos_dim": 2,
        "horizon": 15, "exec_steps": 8, "envelope_digest": digest, "calibration_digest": null}).to_string()).unwrap();
    match s.handle(req(wrong.clone())) {
        Response::Error(e) => assert_eq!(e.code, "protocol"),
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    wrong["proto"] = serde_json::json!("lictor-wire/v1");
    wrong["horizon"] = serde_json::json!(16);
    wrong["id"] = serde_json::json!(5);
    match s.handle(req(wrong)) {
        Response::Error(e) => assert!(e.message.contains("horizon"), "{}", e.message),
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    match s.handle(hello(6, "observe", &digest, None)) {
        Response::HelloOk(h) => {
            assert_eq!(h.envelope_digest, digest);
            assert_eq!(h.embodiment_digest, envelope().embodiment_digest());
            assert!(!h.ephemeral_key);
            assert!(!h.tier1_armed);
            assert!(h.calibration.is_none());
            assert_eq!(h.tier0_armed, ["workspace", "speed", "accel", "jerk", "reach", "brake"]);
            assert_eq!(h.features.len(), NFEAT);
            assert_eq!(h.trip_names.len(), 16);
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
}

fn calibration_c(env: &SafetyEnvelope) -> CalibrationC {
    let gate = GateSpec::parse(&env.gate).unwrap();
    let mut c = CalibrationC::DISARMED;
    c.method = CalMethod::Static;
    c.alpha_num = 5;
    c.alpha_den = 100;
    c.n_calib = 137;
    c.horizon_ticks = env.embodiment.horizon_ticks;
    c.t_grid = 1;
    c.scale = [[1.0; NFEAT]; lictor_core::T_GRID];
    c.mask = gate.mask();
    c.gate = gate;
    c.tau = 3.0;
    c.digest = [0xe1; 32];
    c
}

#[test]
fn embodiment_digest_mismatch_is_refused_at_startup() {
    let env = envelope();
    let mut cfg = config(env.clone(), FuseMode::Enforce, None);
    cfg.calibration = Some(CalibrationLoaded {
        c: calibration_c(&env),
        digest: "e1".repeat(32),
        embodiment_digest: "00".repeat(32),
        file_sha256: "11".repeat(32),
        path: String::new(),
    });
    let err = Session::new(cfg).err().expect("refused");
    assert!(err.to_string().contains("embodiment digest"), "{err}");
}

#[test]
fn weights_digest_mismatch_is_refused_at_episode_begin() {
    let dir = tempfile::tempdir().unwrap();
    let env = envelope();
    let cal_path = dir.path().join("calibration.a05.json");
    std::fs::write(&cal_path, "{\"schema\":\"lictor-calibration/v1\",\"policy_digest\":\"abab\"}\n").unwrap();
    let mut cfg = config(env.clone(), FuseMode::Enforce, None);
    cfg.calibration = Some(CalibrationLoaded {
        c: calibration_c(&env),
        digest: "e1".repeat(32),
        embodiment_digest: env.embodiment_digest(),
        file_sha256: "11".repeat(32),
        path: cal_path.to_string_lossy().to_string(),
    });
    let mut s = Session::new(cfg).unwrap();
    match s.handle(hello(1, "enforce", &env.digest_hex(), Some(&"e1".repeat(32)))) {
        Response::HelloOk(h) => {
            assert!(h.tier1_armed);
            let c = h.calibration.expect("calibration info");
            assert_eq!((c.alpha_num, c.alpha_den, c.n_calib), (5, 100, 137));
            assert_eq!(c.tau, 3.0);
            assert_eq!(c.kn, [3, 5]);
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    match s.handle(begin(2, "t01-a05-d0", 0, "cdcd")) {
        Response::Error(e) => {
            assert_eq!(e.code, "envelope");
            assert!(e.fatal);
            assert!(e.message.contains("weights_sha256"), "{}", e.message);
        }
        other => panic!("{}", serde_json::to_string(&other).unwrap()),
    }
    assert!(!s.in_episode());
    // the matching digest is accepted (episode_begin does not touch decide())
    assert!(matches!(s.handle(begin(3, "t01-a05-d0", 0, "abab")), Response::EpisodeOk(_)));
}

fn crash_budget() -> BudgetBinding {
    BudgetBinding {
        mode: FuseMode::Enforce,
        delay_steps: 0,
        tick_ms: 100,
        exec_mode: ExecMode::Sync,
        stitch: "drop".into(),
        on_escalate: "terminate_fail".into(),
        tier0_armed: vec!["workspace".into(), "brake".into()],
        tier1_armed: false,
        gate: vec![],
        alpha_num: 0,
        alpha_den: 1,
        kn: [3, 5],
    }
}

#[test]
fn crash_receipt_verifies_with_fuse_ok_false() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = config(envelope(), FuseMode::Enforce, Some(dir.path().to_path_buf()));
    let run = RunBinding {
        run_id: "run-crash".into(),
        arm_id: "t01-a05-d0".into(),
        episode_index: 3,
        seed: 3,
        seed_pool: "eval".into(),
        init_state_digest: "3f".repeat(32),
        env: BTreeMap::new(),
        policy: BTreeMap::new(),
        host: BTreeMap::new(),
    };
    let entry =
        write_crash_episode(&cfg, run, crash_budget(), "lictor crash-receipt", "child exit 137").unwrap();
    assert_eq!(entry.seq, 0);
    assert!(!entry.fuse_ok && !entry.success && !entry.tripped);
    let paths = episode_paths(dir.path(), "run-crash", "t01-a05-d0", 3);
    let sr: SignedReceipt = serde_json::from_str(&std::fs::read_to_string(&paths.receipt).unwrap()).unwrap();
    let report = verify(&sr, None);
    assert!(report.intact(), "{:?}", report.notes);
    assert!(!report.fuse_ok);
    assert!(!sr.body.fuse_ok);
    assert_eq!(sr.body.outcome.ended_by, "fuse_crash");
    assert_eq!(sr.body.outcome.steps, 0);
    assert!(!sr.body.outcome.success);
    assert_eq!(sr.body.counts.terminal_state, FuseState::Fault);
    assert!(sr.body.fuse_notes.iter().any(|n| n == NOTE_CRASH), "{:?}", sr.body.fuse_notes);
    assert!(sr.body.fuse_notes.iter().any(|n| n == "child exit 137"));
    assert!(sr.body.fuse_notes.iter().any(|n| n.contains("Fault")));
    assert_eq!(sr.body.verdict_chain_head, lictor_receipt::ZERO_HASH);
    let (_, ticks) = read_events::<TickEvent>(&paths.ticks);
    assert!(ticks.is_empty());
    let (h, timing) = read_events::<TimingEvent>(&paths.timing);
    assert_eq!(h["stream"], "timing");
    assert!(timing.is_empty());
    let ledger = read_ledger(&paths.ledger).unwrap();
    assert_eq!(ledger.len(), 1);
    assert!(verify_ledger(&ledger).chain_ok);
    // a second crash receipt for the next index chains onto the first
    let run2 = RunBinding {
        run_id: "run-crash".into(),
        arm_id: "t01-a05-d0".into(),
        episode_index: 4,
        seed: 4,
        seed_pool: "eval".into(),
        init_state_digest: "3f".repeat(32),
        env: BTreeMap::new(),
        policy: BTreeMap::new(),
        host: BTreeMap::new(),
    };
    let e2 = write_crash_episode(&cfg, run2, crash_budget(), "lictor crash-receipt", "").unwrap();
    assert_eq!(e2.seq, 1);
    assert_eq!(e2.prev, entry.hash);
    let sr2: SignedReceipt = serde_json::from_str(
        &std::fs::read_to_string(episode_paths(dir.path(), "run-crash", "t01-a05-d0", 4).receipt).unwrap(),
    )
    .unwrap();
    assert_eq!(sr2.body.ledger_prev.as_deref(), Some(entry.hash.as_str()));
    assert_eq!(sr2.body.fuse_notes.len(), sr.body.fuse_notes.len() - 1);
    // without a signing key the note says so
    let mut cfg2 = config(envelope(), FuseMode::Enforce, Some(dir.path().to_path_buf()));
    cfg2.key = None;
    let run3 = RunBinding {
        run_id: "run-crash".into(),
        arm_id: "t0-d0".into(),
        episode_index: 0,
        seed: 0,
        seed_pool: "eval".into(),
        init_state_digest: "3f".repeat(32),
        env: BTreeMap::new(),
        policy: BTreeMap::new(),
        host: BTreeMap::new(),
    };
    write_crash_episode(&cfg2, run3, crash_budget(), "c", "").unwrap();
    let sr3: SignedReceipt = serde_json::from_str(
        &std::fs::read_to_string(episode_paths(dir.path(), "run-crash", "t0-d0", 0).receipt).unwrap(),
    )
    .unwrap();
    assert!(sr3.body.fuse_notes.iter().any(|n| n == "ephemeral signing key"));
    assert!(verify(&sr3, None).intact());
}

#[test]
fn two_runs_share_the_verdict_chain_but_not_the_timing_chain() {
    let d1 = tempfile::tempdir().unwrap();
    let d2 = tempfile::tempdir().unwrap();
    let mut s1 = Session::new(config(envelope(), FuseMode::Enforce, Some(d1.path().to_path_buf()))).unwrap();
    let mut s2 = Session::new(config(envelope(), FuseMode::Enforce, Some(d2.path().to_path_buf()))).unwrap();
    let (_, m1) = run_nominal(&mut s1, "enforce", "t0-d0", 1, 120);
    let (_, m2) = run_nominal(&mut s2, "enforce", "t0-d0", 1, 120);
    assert_eq!(m1.verdict_chain_head, m2.verdict_chain_head);
    assert_ne!(m1.timing_chain_head, m2.timing_chain_head);
    assert!(m1.fuse_ok, "{:?}", m1.fuse_notes);
    assert_eq!(m1.counts.substituted, 0);
    assert_eq!(s1.verdict_chain_head(), lictor_receipt::ZERO_HASH, "no episode open after episode_end");
}

fn ack_json(handoff_digest: &str, nonce: u64) -> serde_json::Value {
    let seed = lictor_receipt::keys::parse_seed(OPERATOR_HEX).unwrap();
    let tok = sign_ack(handoff_digest, AckDecision::Resume, nonce, "oracle resume", &seed).unwrap();
    serde_json::to_value(tok).unwrap()
}

/// Enforce + rearm=ack_only + the operator listed: three violating chunks -> Clamped x3 -> Braking -> Held ->
/// Escalated (handoff) -> a signed Resume ack re-arms. Returns (ack_result of the ack tick, receipt).
fn drive_escalation(
    s: &mut Session,
    replay_ack: Option<&serde_json::Value>,
) -> (String, Option<String>, lictor_runtime::wire::EpisodeReceiptMsg) {
    let env = ack_envelope();
    assert!(matches!(s.handle(hello(1, "enforce", &env.digest_hex(), None)), Response::HelloOk(_)));
    assert!(matches!(s.handle(begin(2, "t01-a05-d0-oracle", 7, "ab")), Response::EpisodeOk(_)));
    let pos = [256.0, 256.0];
    let mut id = 3u64;
    let mut handoff: Option<(String, u32)> = None;
    for t in 0..200u32 {
        let violating = (8..=24).contains(&t);
        let chunk = if t.is_multiple_of(8) {
            Some(if violating { vec![[456.0, 256.0]; 15] } else { chunk_rows(pos, [0.5, 0.0]) })
        } else {
            None
        };
        let ack = match &handoff {
            Some((digest, _)) => match replay_ack {
                Some(a) => a.clone(),
                None => ack_json(digest, 1),
            },
            None => serde_json::Value::Null,
        };
        let sending_ack = !ack.is_null();
        let v = verdict(s.handle(req(tick_json(id, t, pos, [0.0, 0.0], chunk, ack))));
        id += 1;
        if let Some(h) = &v.handoff {
            assert!(handoff.is_none(), "one handoff expected");
            assert_eq!(v.state, FuseState::Escalated);
            assert_eq!(h.tick, t);
            assert_eq!(h.chain_at, v.chain);
            assert_eq!(h.digest, h.clone().with_digest().unwrap().digest);
            handoff = Some((h.digest.clone(), t));
        }
        if sending_ack {
            let ack_result = v.ack_result.clone().unwrap_or_default();
            let ack_sent_digest = handoff.as_ref().map(|h| h.0.clone());
            if ack_result == "accepted" {
                assert_eq!(v.state, FuseState::Armed, "{}", serde_json::to_string(&v).unwrap());
            }
            // a few more nominal ticks, then stop
            for k in 1..=5 {
                let chunk = if (t + k).is_multiple_of(8) { Some(chunk_rows(pos, [0.5, 0.0])) } else { None };
                verdict(s.handle(req(tick_json(id, t + k, pos, [0.0, 0.0], chunk, serde_json::Value::Null))));
                id += 1;
            }
            let m = match s.handle(end(id, t + 6, "escalation_terminate")) {
                Response::EpisodeReceipt(m) => m,
                other => panic!("{}", serde_json::to_string(&other).unwrap()),
            };
            return (ack_result, ack_sent_digest, m);
        }
    }
    panic!("the fuse never escalated within 200 ticks (last handoff {handoff:?})");
}

fn ack_envelope() -> SafetyEnvelope {
    let mut env = envelope();
    env.hysteresis.rearm = RearmPolicy::AckOnly;
    let seed = lictor_receipt::keys::parse_seed(OPERATOR_HEX).unwrap();
    env.operators = vec![lictor_receipt::pubkey_hex(&seed)];
    env
}

#[test]
fn accepted_ack_persists_its_nonce_and_replays_are_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let mut s =
        Session::new(config(ack_envelope(), FuseMode::Enforce, Some(dir.path().to_path_buf()))).unwrap();
    let (result, digest, m) = drive_escalation(&mut s, None);
    assert_eq!(result, "accepted");
    assert_eq!(m.counts.escalations, 1);
    assert_eq!(m.counts.rearms, 1);
    let nonces: BTreeMap<String, u64> =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join(NONCE_FILE)).unwrap()).unwrap();
    let op = lictor_receipt::pubkey_hex(&lictor_receipt::keys::parse_seed(OPERATOR_HEX).unwrap());
    assert_eq!(nonces.get(&op), Some(&1));
    let paths = episode_paths(dir.path(), "run-ep", "t01-a05-d0-oracle", 7);
    let sr: SignedReceipt = serde_json::from_str(&std::fs::read_to_string(&paths.receipt).unwrap()).unwrap();
    assert!(verify(&sr, None).intact());
    assert_eq!(sr.body.handoffs.len(), 1);
    assert_eq!(sr.body.handoffs[0].outcome, "resumed");
    assert!(sr.body.handoffs[0].ack.is_some());
    assert_eq!(sr.body.handoffs[0].digest, digest.unwrap());

    // a fresh Session in the same out_dir loads the nonce table: the identical ack is a replay
    let replay = ack_json(&sr.body.handoffs[0].digest, 1);
    let mut s2 =
        Session::new(config(ack_envelope(), FuseMode::Enforce, Some(dir.path().to_path_buf()))).unwrap();
    let (result2, _, m2) = drive_escalation(&mut s2, Some(&replay));
    assert_eq!(result2, "rejected:nonce_replay");
    assert_eq!(m2.counts.rearms, 0);
    assert_eq!(m2.ledger_seq, 1);
}
