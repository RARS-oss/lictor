// SPDX-License-Identifier: MIT
//! Build a deterministic synthetic episode, verify it, and check that two builds agree byte-for-byte.
//! With `LICTOR_WRITE_FIXTURES=1` this test REGENERATES every committed fixture under tests/fixtures/receipt
//! from the fixed TEST seed keys (key.hex / operator.hex); without it, it asserts the committed fixtures equal
//! a fresh regeneration (so the fixtures are reproducible, not merely stored).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;

use lictor_canon::{canon_of, floatify, sha256_hex, F64Array, F64Hex, CANONICAL_ID};
use lictor_core::{
    AckDecision, ActionSource, ExecMode, FuseMode, FuseState, ReasonCode, SafetyVerdict, Scores, Status,
    TripMask, MAX_D, NFEAT,
};
use lictor_receipt::{
    append_ledger, evaluate_fuse, pubkey_hex, read_ledger, sign, sign_ack, tick_event, timing_event, verify,
    verify_ledger, verify_tick_chain, verify_ticks_file, verify_timing_chain, BudgetBinding, EpisodeOutcome,
    HandoffRecord, LatencySummary, ReceiptBody, RunBinding, SignedReceipt, TickEvent, TimingEvent,
    VerdictCounts, HANDOFF_SCHEMA, LEDGER_SCHEMA, RECEIPT_SCHEMA, TICKS_SCHEMA, ZERO_HASH,
};

/// The committed TEST signing key: bytes 00..1f. Not a secret; it exists so the fixtures are reproducible.
const FIXTURE_SEED: [u8; 32] = [
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
];
/// The committed TEST operator key: bytes 20..3f.
const OPERATOR_SEED: [u8; 32] = [
    0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f, 0x30,
    0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
];
const RUN_ID: &str = "2026-09-01T09-14Z-pilot";
const ARM_ID: &str = "t01-a05-d0";
const CREATED_EPOCH: u64 = 1788327242;
const WSL_LABEL: &str =
    "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment";

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("receipt")
}

fn digest_str(s: &str) -> String {
    sha256_hex(s.as_bytes())
}

/// The synthetic envelope (a small config tree with real numbers, floatified like the TOML would be).
fn envelope(operator_pubkey: &str) -> serde_json::Value {
    floatify(serde_json::json!({
        "schema": "lictor-envelope/v1",
        "embodiment": {"name": "pusht", "action_dim": 2, "control_hz": 10},
        "workspace": {"min": [0.0, 0.0], "max": [512.0, 512.0]},
        "limits": {"speed": 540.0, "accel": 9000.0, "jerk": 180000.0, "reach": 600.0},
        "brake": {"kind": "pd_rollout", "k_p": 100.0, "k_v": 20.0, "dt": 0.01, "horizon_s": 0.5},
        "tier1": {"alpha": 0.05, "k": 3, "n": 5, "gate": ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak"]},
        "escalation": {"hold_ticks": 30, "on_escalate": "terminate_fail"},
        "operators": [operator_pubkey],
    }))
}

/// Per-tick phase of the 300-tick narrative for the long episode: 281 nominal, 12 watching (120..131),
/// 4 braking (181..184), 3 held (185..187), re-arm at 188.
fn phase(seq: u32) -> (Status, FuseState, ActionSource, ReasonCode, u32) {
    match seq {
        120..=131 => (Status::Watching, FuseState::Watching, ActionSource::Policy, ReasonCode::WatchBand, 0),
        181 => (
            Status::Braking,
            FuseState::Braking,
            ActionSource::Brake,
            ReasonCode::BrakeTier1Cp,
            TripMask::TIER1_CP,
        ),
        182..=184 => (Status::Braking, FuseState::Braking, ActionSource::Brake, ReasonCode::Stopping, 0),
        185..=187 => (Status::Held, FuseState::Held, ActionSource::Hold, ReasonCode::HeldStandstill, 0),
        _ => (Status::Nominal, FuseState::Armed, ActionSource::Policy, ReasonCode::Ok, 0),
    }
}

fn synth_verdict(seq: u32, long: bool) -> SafetyVerdict {
    let (status, state, src, reason, trips) = if long { phase(seq) } else { phase(0) };
    let prev_state = if long && seq > 0 {
        phase(seq - 1).1
    } else if seq == 0 {
        FuseState::Idle
    } else {
        state
    };
    let reason = if long && seq == 188 { ReasonCode::RearmedAuto } else { reason };
    let mut action = [0.0; MAX_D];
    action[0] = 13.375 + f64::from(seq) * 0.5;
    action[1] = 300.4 - f64::from(seq) * 0.25;
    if src != ActionSource::Policy {
        action[0] = 256.0;
        action[1] = 256.0;
    }
    let mut scores = Scores::default();
    for j in 0..NFEAT {
        let raw = f64::from((seq * (j as u32 + 1)) % 17) / 8.0;
        scores.f[j] = raw;
        scores.z[j] = if j < 8 { raw - 0.5 } else { 0.0 };
    }
    scores.valid = 0xff;
    if long && (181..=184).contains(&seq) {
        scores.z[6] = 4.16;
        scores.fired = 1 << 6;
    }
    scores.s = scores.z[..8].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    SafetyVerdict {
        seq,
        t: seq,
        status,
        state,
        prev_state,
        trips,
        action,
        action_dim: 2,
        action_src: src,
        substituted: src != ActionSource::Policy,
        clamped_dims: 0,
        scores,
        tau: 26.0 / 7.0,
        window: 0,
        window_hits: if long && (181..=187).contains(&seq) { 4 } else { 0 },
        brake_margin: 41.2,
        reason,
        handoff_seq: None,
        ack_consumed: false,
        violation_reached_env: false,
    }
}

fn percentile(sorted: &[u64], q: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

struct Episode {
    receipt: SignedReceipt,
    ticks: Vec<TickEvent>,
    timing: Vec<TimingEvent>,
}

/// Build one synthetic episode. `episode_index == 7` is the long 300-tick narrative; the others are short
/// nominal episodes so the ledger has five entries.
fn build_episode(episode_index: u32, ledger_prev: Option<String>) -> Episode {
    let long = episode_index == 7;
    let n = if long { 300 } else { 120 + 20 * episode_index };
    let mut ticks = Vec::with_capacity(n as usize);
    let mut timing = Vec::with_capacity(n as usize);
    let mut counts =
        VerdictCounts { trips_by_bit: vec![0; 16], fired_by_feat: vec![0; NFEAT], ..Default::default() };
    let mut max_s = f64::NEG_INFINITY;
    let mut max_z = [f64::NEG_INFINITY; NFEAT];
    let (mut vprev, mut tprev) = (ZERO_HASH.to_string(), ZERO_HASH.to_string());
    for seq in 0..n {
        let v = synth_verdict(seq, long);
        let e = tick_event(&vprev, &v);
        vprev = e.hash.clone();
        ticks.push(e);
        let t = timing_event(
            &tprev,
            seq,
            1000 + u64::from(seq * 37 % 900),
            30_000 + u64::from(seq * 101 % 20_000),
        );
        tprev = t.hash.clone();
        timing.push(t);
        counts.ticks += 1;
        match v.status {
            Status::Nominal => counts.nominal += 1,
            Status::Watching => counts.watching += 1,
            Status::Braking => counts.braking += 1,
            Status::Held => counts.held += 1,
            _ => {}
        }
        counts.substituted += u32::from(v.substituted);
        for bit in 0..16 {
            if v.trips & (1 << bit) != 0 {
                counts.trips_by_bit[bit] += 1;
                if counts.first_trip_tick.is_none() {
                    counts.first_trip_tick = Some(seq);
                    counts.first_trip_reason = Some("brake_tier1_cp".to_string());
                }
            }
        }
        for (j, mz) in max_z.iter_mut().enumerate() {
            if v.scores.fired & (1 << j) != 0 {
                counts.fired_by_feat[j] += 1;
            }
            *mz = mz.max(v.scores.z[j]);
        }
        max_s = max_s.max(v.scores.s);
        if v.state.is_stop() && counts.first_stop_tick.is_none() {
            counts.first_stop_tick = Some(seq);
        }
        if v.state == FuseState::Held && v.prev_state != FuseState::Held {
            counts.holds += 1;
        }
        if v.reason == ReasonCode::RearmedAuto {
            counts.rearms += 1;
        }
        if seq.is_multiple_of(8) {
            counts.chunks_seen += 1;
        }
        counts.terminal_state = v.state;
    }
    let mut decide: Vec<u64> = timing.iter().map(|t| t.decide_ns).collect();
    decide.sort_unstable();
    let latency = LatencySummary {
        n,
        p50_ns: percentile(&decide, 0.5),
        p90_ns: percentile(&decide, 0.9),
        p99_ns: percentile(&decide, 0.99),
        p999_ns: percentile(&decide, 0.999),
        max_ns: *decide.last().unwrap_or(&0),
        label: WSL_LABEL.to_string(),
    };
    let operator_pk = pubkey_hex(&OPERATOR_SEED);
    let envelope = envelope(&operator_pk);
    let envelope_digest = sha256_hex(&canon_of(&envelope).unwrap());
    let calibration_digest = digest_str("synthetic calibration alpha=5/100 gate=8 kn=3/5 n_calib=137");
    let inputs: BTreeMap<String, String> = [
        "envelopes/pusht.toml",
        "harness/pusht_rollout.py",
        "harness/executor.py",
        "harness/compat.py",
        "lictor:bin",
        "lictor:envelope",
        "lictor:calibration",
    ]
    .iter()
    .map(|k| (k.to_string(), digest_str(k)))
    .collect();
    let success = !long && episode_index % 2 == 1;
    let budget = BudgetBinding {
        mode: FuseMode::Enforce,
        delay_steps: 0,
        tick_ms: 100,
        exec_mode: ExecMode::Sync,
        stitch: "drop".into(),
        on_escalate: "terminate_fail".into(),
        tier0_armed: ["workspace", "speed", "accel", "jerk", "reach", "brake"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        tier1_armed: true,
        gate: ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        alpha_num: 5,
        alpha_den: 100,
        kn: [3, 5],
    };
    let (fuse_ok, fuse_notes) = evaluate_fuse(&budget, &counts, Some(&calibration_digest), false);
    let tail_from = ticks.len().saturating_sub(32);
    let body = ReceiptBody {
        schema: RECEIPT_SCHEMA.into(),
        canonical: CANONICAL_ID.into(),
        created_epoch: CREATED_EPOCH,
        lictor_version: "0.1.0".into(),
        lictor_git: "fixture".into(),
        lictor_sha256: digest_str("lictor binary (synthetic fixture)"),
        client: "lictor_client/0.1.0".into(),
        run: RunBinding {
            run_id: RUN_ID.into(),
            arm_id: ARM_ID.into(),
            episode_index,
            seed: u64::from(episode_index),
            seed_pool: "eval".into(),
            init_state_digest: digest_str(&format!("init state {episode_index}")),
            env: [
                ("env_id", "gym_pusht/PushT-v0"),
                ("gym_pusht", "0.1.5"),
                ("gymnasium", "1.3.0"),
                ("obs_type", "pixels_agent_pos"),
                ("control_hz", "10"),
                ("max_episode_steps", "300"),
                ("vel_source", "info.vel_agent"),
            ]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
            policy: [
                ("repo_id", "lerobot/diffusion_pusht"),
                ("horizon", "16"),
                ("n_action_steps", "15"),
                ("n_obs_steps", "2"),
                ("device", "cuda"),
                ("dtype", "float32"),
                ("normalization_migrated", "true"),
            ]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
            host: [
                ("os", "synthetic"),
                ("torch", "2.7.1+cu126"),
                ("lerobot", "0.6.1"),
                ("OMP_NUM_THREADS", "1"),
                ("MKL_NUM_THREADS", "1"),
                ("PYTHONHASHSEED", "0"),
                ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
            ]
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        },
        budget,
        fault_injection: None,
        envelope,
        envelope_digest,
        calibration_digest: Some(calibration_digest),
        inputs,
        counts,
        outcome: EpisodeOutcome {
            steps: n,
            success,
            terminated: success,
            truncated: !success,
            max_coverage: F64Hex(if success { 0.96 } else { 0.71 }),
            final_coverage: F64Hex(if success { 0.96 } else { 0.68 }),
            reward_sum: F64Hex(f64::from(n) * 0.595),
            ended_by: if success { "success".into() } else { "truncated".into() },
            max_s: F64Hex(max_s),
            max_z: F64Array::from_slice(&max_z, &[NFEAT as u32]),
        },
        handoffs: vec![],
        verdict_events: n,
        verdict_chain_head: vprev,
        timing_events: n,
        timing_chain_head: tprev,
        latency,
        ticks_policy: "tail32".into(),
        ticks: ticks[tail_from..].to_vec(),
        fuse_ok,
        fuse_notes,
        ledger_prev,
    };
    let receipt = sign(body, &FIXTURE_SEED).unwrap();
    Episode { receipt, ticks, timing }
}

fn handoff_for(ep: &Episode) -> HandoffRecord {
    let b = &ep.receipt.body;
    HandoffRecord {
        schema: HANDOFF_SCHEMA.into(),
        run_id: RUN_ID.into(),
        arm_id: format!("{ARM_ID}-oracle"),
        episode_index: b.run.episode_index,
        seq: 0,
        tick: 97,
        reason: ReasonCode::EscalateHoldTimeout,
        reason_text: "Held 30 ticks without a clean re-arm window; a human must decide.".into(),
        reasons: vec!["tier1_cp".into(), "rearm_budget".into()],
        trips: TripMask::TIER1_CP | TripMask::REARM_BUDGET,
        fired: 1 << 6,
        window_hits: 4,
        top_z: vec![
            ("stall".into(), F64Hex(4.16)),
            ("path_ineff".into(), F64Hex(3.3)),
            ("tce".into(), F64Hex(1.5)),
        ],
        chain_at: ep.ticks[97].hash.clone(),
        envelope_digest: b.envelope_digest.clone(),
        calibration_digest: b.calibration_digest.clone(),
        digest: String::new(),
        ack: None,
        resolved_tick: None,
        outcome: "unacked".into(),
    }
    .with_digest()
    .unwrap()
}

fn pretty(v: &impl serde::Serialize) -> String {
    // serde_json's Map is BTreeMap-backed (no preserve_order feature): pretty output has sorted keys.
    let value = serde_json::to_value(v).unwrap();
    let mut s = serde_json::to_string_pretty(&value).unwrap();
    s.push('\n');
    s
}

fn write_jsonl<T: serde::Serialize>(path: &std::path::Path, header: &serde_json::Value, rows: &[T]) {
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&canon_of(header).unwrap()).unwrap();
    f.write_all(b"\n").unwrap();
    for r in rows {
        f.write_all(&canon_of(r).unwrap()).unwrap();
        f.write_all(b"\n").unwrap();
    }
}

/// Build the five-episode arm (3..7) through a ledger in `dir`; returns the episodes in order.
fn build_arm(dir: &std::path::Path) -> Vec<Episode> {
    let ledger = dir.join("ledger.jsonl");
    let mut out = Vec::new();
    let mut prev: Option<String> = None;
    for ep in 3..=7 {
        let e = build_episode(ep, prev.clone());
        let entry = append_ledger(&ledger, &e.receipt).unwrap();
        prev = Some(entry.hash);
        out.push(e);
    }
    out
}

fn read_ticks(path: &std::path::Path) -> (serde_json::Value, Vec<TickEvent>) {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: serde_json::Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    (header, lines.map(|l| serde_json::from_str(l).unwrap()).collect())
}

#[test]
fn two_builds_agree_and_verify() {
    let a = build_episode(7, None);
    let b = build_episode(7, None);
    assert_eq!(a.receipt, b.receipt);
    assert_eq!(a.ticks, b.ticks);
    assert_eq!(a.timing, b.timing);
    let body = &a.receipt.body;
    assert_eq!(body.counts.ticks, 300);
    assert_eq!(
        (body.counts.nominal, body.counts.watching, body.counts.braking, body.counts.held),
        (281, 12, 4, 3)
    );
    assert_eq!(body.counts.first_trip_tick, Some(181));
    assert_eq!(body.counts.first_trip_reason.as_deref(), Some("brake_tier1_cp"));
    assert_eq!(body.ticks.len(), 32);
    assert_eq!(body.ticks[0].seq, 268);
    let r = verify(&a.receipt, Some(&pubkey_hex(&FIXTURE_SEED)));
    assert!(r.intact(), "{r:?}");
    assert!(r.fuse_ok, "{r:?}");
    assert_eq!(r.notes, Vec::<String>::new());
    let chain = verify_tick_chain(&a.ticks, ZERO_HASH);
    assert!(chain.ok && chain.head == body.verdict_chain_head && chain.n == 300);
    let t = verify_timing_chain(&a.timing);
    assert!(t.ok && t.head == body.timing_chain_head);
    assert_eq!(verify_ticks_file(&a.receipt, &a.ticks).head, body.verdict_chain_head);
    assert_eq!(a.receipt.pubkey, lictor_receipt::TEST_PUBKEYS[0]);
}

#[test]
fn write_or_check_fixtures() {
    let dir = fixture_dir();
    let tmp = tempfile::tempdir().unwrap();
    let arm = build_arm(tmp.path());
    let ep7 = arm.last().unwrap();
    let handoff = handoff_for(ep7);
    let ack = sign_ack(&handoff.digest, AckDecision::Resume, 7, "oracle resume", &OPERATOR_SEED).unwrap();
    let ledger_text = std::fs::read_to_string(tmp.path().join("ledger.jsonl")).unwrap();

    if std::env::var("LICTOR_WRITE_FIXTURES").as_deref() == Ok("1") {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("receipt_000007.json"), pretty(&ep7.receipt)).unwrap();
        let header = serde_json::json!({"schema": TICKS_SCHEMA, "run_id": RUN_ID, "arm_id": ARM_ID,
            "episode_index": 7, "genesis": ZERO_HASH});
        write_jsonl(&dir.join("ticks_000007.jsonl"), &header, &ep7.ticks);
        let theader = serde_json::json!({"schema": TICKS_SCHEMA, "stream": "timing", "run_id": RUN_ID,
            "arm_id": ARM_ID, "episode_index": 7, "genesis": ZERO_HASH});
        write_jsonl(&dir.join("timing_000007.jsonl"), &theader, &ep7.timing);
        std::fs::write(dir.join("ledger.jsonl"), &ledger_text).unwrap();
        std::fs::write(
            dir.join("key.hex"),
            format!("# TEST KEY (committed on purpose; not a secret): the fixture signing seed, bytes 00..1f\n{}\n", hex::encode(FIXTURE_SEED)),
        )
        .unwrap();
        std::fs::write(
            dir.join("operator.hex"),
            format!("# TEST KEY (committed on purpose; not a secret): the fixture operator seed, bytes 20..3f\n{}\n", hex::encode(OPERATOR_SEED)),
        )
        .unwrap();
        std::fs::write(dir.join("pubkey.txt"), format!("{}\n", pubkey_hex(&FIXTURE_SEED))).unwrap();
        std::fs::write(dir.join("handoff.json"), pretty(&handoff)).unwrap();
        std::fs::write(dir.join("ack.json"), pretty(&ack)).unwrap();
        eprintln!("fixtures written to {}", dir.display());
    }

    // The committed fixtures must equal a fresh regeneration.
    let committed: SignedReceipt =
        serde_json::from_str(&std::fs::read_to_string(dir.join("receipt_000007.json")).unwrap()).unwrap();
    assert_eq!(committed, ep7.receipt, "receipt_000007.json is not what the generator produces");
    let (header, ticks) = read_ticks(&dir.join("ticks_000007.jsonl"));
    assert_eq!(header["schema"], TICKS_SCHEMA);
    assert_eq!(header["genesis"], ZERO_HASH);
    assert_eq!(header["episode_index"], 7);
    assert_eq!(ticks, ep7.ticks);
    assert_eq!(&ticks[268..], &committed.body.ticks[..]);
    let timing: Vec<TimingEvent> = std::fs::read_to_string(dir.join("timing_000007.jsonl"))
        .unwrap()
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(timing, ep7.timing);
    assert_eq!(std::fs::read_to_string(dir.join("ledger.jsonl")).unwrap(), ledger_text);
    let entries = read_ledger(&dir.join("ledger.jsonl")).unwrap();
    let lr = verify_ledger(&entries);
    assert!(lr.chain_ok && lr.episodes == 5 && lr.trips == 1 && lr.stops == 1 && lr.successes == 2, "{lr:?}");
    assert_eq!(entries[4].receipt_digest, committed.body_digest);
    assert_eq!(committed.body.ledger_prev.as_deref(), Some(entries[3].hash.as_str()));
    let first = std::fs::read_to_string(dir.join("ledger.jsonl")).unwrap();
    assert!(first.lines().next().unwrap().contains(LEDGER_SCHEMA));
    assert_eq!(std::fs::read_to_string(dir.join("pubkey.txt")).unwrap().trim(), pubkey_hex(&FIXTURE_SEED));
    assert_eq!(lictor_receipt::load_seed(&dir.join("key.hex")).unwrap(), FIXTURE_SEED);
    assert_eq!(lictor_receipt::load_seed(&dir.join("operator.hex")).unwrap(), OPERATOR_SEED);
    let h: HandoffRecord =
        serde_json::from_str(&std::fs::read_to_string(dir.join("handoff.json")).unwrap()).unwrap();
    assert_eq!(h, handoff);
    assert_eq!(h.compute_digest().unwrap(), h.digest);
    let a: lictor_receipt::AckToken =
        serde_json::from_str(&std::fs::read_to_string(dir.join("ack.json")).unwrap()).unwrap();
    assert_eq!(a, ack);
}

#[test]
fn short_episodes_verify_and_chain_heads_differ_per_episode() {
    let a = build_episode(3, None);
    let b = build_episode(4, Some("abc".into()));
    assert!(verify(&a.receipt, None).intact());
    assert!(verify(&b.receipt, None).intact());
    assert_ne!(a.receipt.body.verdict_chain_head, b.receipt.body.verdict_chain_head);
    assert_eq!(a.receipt.body.ticks.len(), 32);
    assert_eq!(a.receipt.body.verdict_events, 180);
}
