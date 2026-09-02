// SPDX-License-Identifier: MIT
//! Wire fixtures: every golden request parses and survives a semantic round trip, every golden response line is
//! exactly what the frozen response structs serialise to, and every `bad_lines.ndjson` entry is rejected -- at the
//! codec (serde) or at `Staging::stage` -- with the expected reason.

use std::io::Cursor;

use lictor_core::{ActionSource, CalMethod, FuseMode, FuseState, ReasonCode, SafetyEnvelope, Status, NFEAT};
use lictor_receipt::VerdictCounts;
use lictor_runtime::codec::{parse_request, read_request, write_response};
use lictor_runtime::session::Staging;
use lictor_runtime::wire::*;

const GOLDEN_REQUESTS: &str = include_str!("fixtures/wire/golden_requests.ndjson");
const GOLDEN_RESPONSES: &str = include_str!("fixtures/wire/golden_responses.ndjson");
const BAD_LINES: &str = include_str!("fixtures/wire/bad_lines.ndjson");
const ENVELOPE: &str = include_str!("../../../envelopes/pusht.base.toml");

fn lines(s: &str) -> impl Iterator<Item = &str> {
    s.lines().map(str::trim).filter(|l| !l.is_empty())
}

#[test]
fn golden_requests_parse_and_round_trip() {
    let mut n = 0;
    for line in lines(GOLDEN_REQUESTS) {
        let req = parse_request(line).unwrap_or_else(|e| panic!("{line}: {e}"));
        // semantic round trip: re-serialise the JSON value (different key order / whitespace) and parse again
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let pretty = serde_json::to_string_pretty(&v).unwrap();
        let again = parse_request(&pretty.replace('\n', " ")).unwrap();
        assert_eq!(format!("{req:?}"), format!("{again:?}"), "{line}");
        let kind = v["kind"].as_str().unwrap();
        let id = v["id"].as_u64().unwrap();
        match (&req, kind) {
            (Request::Hello(h), "hello") => assert_eq!(h.id, id),
            (Request::EpisodeBegin(b), "episode_begin") => {
                assert_eq!(b.id, id);
                assert_eq!(b.inputs.len(), v["inputs"].as_object().map(|o| o.len()).unwrap_or(0));
            }
            (Request::Tick(t), "tick") => {
                assert_eq!(t.id, id);
                assert_eq!(t.obs.pos.len(), 2);
                if v["obs"]["pos"][0].is_null() {
                    assert!(t.obs.pos[0].is_none());
                }
                assert_eq!(t.chunk.is_some(), !v["chunk"].is_null());
                assert_eq!(t.ack.is_some(), !v["ack"].is_null());
            }
            (Request::EpisodeEnd(e), "episode_end") => {
                assert_eq!(e.id, id);
                if v["outcome"]["max_coverage"].is_null() {
                    assert!(e.outcome.max_coverage.is_none());
                }
            }
            (Request::Bye(b), "bye") => assert_eq!(b.id, id),
            other => panic!("kind mismatch: {other:?}"),
        }
        // and through the codec, which also validates framing
        let mut r = Cursor::new(format!("{line}\n").into_bytes());
        let mut buf = String::new();
        assert!(matches!(read_request(&mut r, &mut buf), Ok(Some(_))));
        assert_eq!(buf, line);
        n += 1;
    }
    assert!(n >= 10, "fixture has {n} lines");
}

fn zeros() -> Vec<f64> {
    vec![0.0; NFEAT]
}

fn hello_ok(git: &str, calibration: Option<CalibrationInfo>, ephemeral: bool, mode: FuseMode) -> Response {
    Response::HelloOk(HelloOk {
        id: 1,
        proto: PROTO.to_string(),
        lictor: "0.1.0".to_string(),
        git: git.to_string(),
        lictor_sha256: "5d5b09f6dcb2d53a5fffc60c4ac0d55fabdf556069d6631545f42aa6e3500f2e".to_string(),
        envelope_digest: format!("7c4a{}", "0".repeat(60)),
        embodiment_digest: format!("91d0{}", "0".repeat(60)),
        calibration_digest: calibration.as_ref().map(|c| c.digest.clone()),
        pubkey: format!("64e8{}", "0".repeat(60)),
        ephemeral_key: ephemeral,
        mode,
        tier0_armed: ["workspace", "speed", "accel", "jerk", "reach", "brake"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        tier1_armed: calibration.is_some(),
        calibration,
        features: lictor_core::Feat::NAMES.iter().map(|s| s.to_string()).collect(),
        trip_names: lictor_core::TripMask::NAMES.iter().map(|s| s.to_string()).collect(),
        latency_label: lictor_runtime::latency::WSL2_LABEL.to_string(),
    })
}

/// The responses the golden fixture lines were written from, in file order.
fn golden_responses() -> Vec<Response> {
    let calib = CalibrationInfo {
        method: CalMethod::Binned,
        alpha_num: 5,
        alpha_den: 100,
        n_calib: 137,
        tau: 3.7142857142857144,
        kn: [3, 5],
        gate: ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak"]
            .iter()
            .map(|s| s.to_string())
            .collect(),
        digest: format!("e1b8{}", "0".repeat(60)),
    };
    let mut trips_by_bit = vec![0u32; 16];
    trips_by_bit[10] = 1;
    vec![
        hello_ok("a3f1c9e", None, false, FuseMode::Observe),
        hello_ok("nogit", Some(calib), true, FuseMode::Enforce),
        Response::EpisodeOk(EpisodeOk { id: 2, state: FuseState::Armed, seq: 0 }),
        Response::Verdict(Box::new(VerdictMsg {
            id: 31,
            t: 24,
            seq: 24,
            status: Status::Nominal,
            state: FuseState::Armed,
            prev_state: FuseState::Armed,
            trips: vec![],
            trip_mask: 0,
            action: vec![214.0, 300.2],
            action_src: ActionSource::Policy,
            substituted: false,
            clamped_dims: 0,
            scores: ScoresMsg {
                f: vec![0.012, 0.31, -0.021, 1.9, 0.11, 0.08, 0.0, 0.34, 0.0, 0.0, 0.0, 0.0],
                z: vec![0.4, 0.9, -0.2, 0.1, 0.7, 0.3, 0.0, 1.1, 0.0, 0.0, 0.0, 0.0],
                s: 1.1,
                valid: 255,
                fired: 0,
            },
            tau: 3.7142857142857144,
            window_hits: 0,
            brake_margin: 41.2,
            reason: ReasonCode::Ok,
            reason_text: "All checks passed; the policy action is applied unchanged.".to_string(),
            handoff: None,
            ack_result: None,
            violation_reached_env: false,
            verdict_ns: 1180,
            chain: format!("5b7e{}", "0".repeat(60)),
        })),
        Response::Verdict(Box::new(VerdictMsg {
            id: 32,
            t: 25,
            seq: 25,
            status: Status::Fault,
            state: FuseState::Fault,
            prev_state: FuseState::Armed,
            trips: vec!["nonfinite".to_string()],
            trip_mask: 128,
            action: vec![213.5, 301.0],
            action_src: ActionSource::Hold,
            substituted: true,
            clamped_dims: 0,
            scores: ScoresMsg { f: zeros(), z: zeros(), s: f64::NEG_INFINITY, valid: 0, fired: 0 },
            tau: f64::INFINITY,
            window_hits: 0,
            brake_margin: 0.0,
            reason: ReasonCode::FaultNonFinite,
            reason_text: "A non-finite value reached the fuse; the fuse latched Fault and holds position."
                .to_string(),
            handoff: None,
            ack_result: Some("rejected:nonce_replay".to_string()),
            violation_reached_env: false,
            verdict_ns: 900,
            chain: "1".repeat(64),
        })),
        Response::EpisodeReceipt(EpisodeReceiptMsg {
            id: 331,
            receipt_path: "results/2026-09-01T09-14Z-pilot/t01-a05-d0/receipts/000007.json".to_string(),
            ticks_path: "results/2026-09-01T09-14Z-pilot/t01-a05-d0/ticks/000007.jsonl".to_string(),
            body_digest: format!("be21{}", "0".repeat(60)),
            verdict_chain_head: format!("5b7e{}", "0".repeat(60)),
            timing_chain_head: format!("22a0{}", "0".repeat(60)),
            fuse_ok: true,
            fuse_notes: vec![],
            counts: VerdictCounts {
                ticks: 300,
                nominal: 281,
                watching: 12,
                clamped: 0,
                braking: 4,
                held: 3,
                escalated: 0,
                fault: 0,
                terminated: 0,
                substituted: 7,
                chunks_seen: 38,
                chunks_rejected: 0,
                clamps: 0,
                holds: 1,
                rearms: 1,
                escalations: 0,
                trips_by_bit,
                fired_by_feat: vec![3, 1, 0, 0, 0, 9, 11, 0, 0, 0, 0, 0],
                first_trip_tick: Some(181),
                first_trip_reason: Some("brake_tier1_cp".to_string()),
                first_stop_tick: Some(181),
                handoff_tick: None,
                violations_reached_env: 0,
                terminal_state: FuseState::Armed,
            },
            ledger_seq: 7,
            ledger_head: format!("aa31{}", "0".repeat(60)),
        }),
        Response::ByeOk { id: 332 },
        Response::Error(ErrorMsg {
            id: 31,
            code: "schema".to_string(),
            message: "tick.chunk.a has 14 rows, expected h=15".to_string(),
            fatal: true,
        }),
    ]
}

#[test]
fn golden_responses_match_the_structs() {
    let expected = golden_responses();
    let fixture: Vec<&str> = lines(GOLDEN_RESPONSES).collect();
    assert_eq!(fixture.len(), expected.len(), "fixture line count");
    for (line, resp) in fixture.iter().zip(&expected) {
        let want: serde_json::Value = serde_json::from_str(line).unwrap();
        let mut out = Vec::new();
        write_response(&mut out, resp).unwrap();
        assert!(out.ends_with(b"\n") && out.iter().filter(|b| **b == b'\n').count() == 1);
        let got: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(got, want, "line: {line}");
    }
    // the two by-design non-finite fields serialise as null and map back to -inf / +inf
    let mut out = Vec::new();
    write_response(&mut out, &expected[4]).unwrap();
    let text = String::from_utf8(out).unwrap();
    assert!(text.contains("\"s\":null"), "{text}");
    assert!(text.contains("\"tau\":null"), "{text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let s = v["scores"]["s"].as_f64().unwrap_or(f64::NEG_INFINITY);
    let tau = v["tau"].as_f64().unwrap_or(f64::INFINITY);
    assert_eq!(s, f64::NEG_INFINITY);
    assert_eq!(tau, f64::INFINITY);
}

#[test]
fn bad_lines_are_rejected_with_the_expected_reason() {
    let env = SafetyEnvelope::from_toml(ENVELOPE).unwrap();
    let cfg = env.compile(FuseMode::Observe, None).unwrap();
    let mut seen_codec = 0;
    let mut seen_staging = 0;
    let mut notes: Vec<String> = Vec::new();
    for entry in lines(BAD_LINES) {
        let e: serde_json::Value = serde_json::from_str(entry).unwrap();
        let stage = e["stage"].as_str().unwrap();
        let reason = e["reason"].as_str().unwrap();
        let line = e["line"].as_str().unwrap();
        if let Some(n) = e["note"].as_str() {
            notes.push(n.to_string());
        }
        match stage {
            "codec" => {
                let mut r = Cursor::new(format!("{line}\n").into_bytes());
                let mut buf = String::new();
                let err = match read_request(&mut r, &mut buf) {
                    Err(err) => err,
                    Ok(_) => panic!("accepted a bad line: {line}"),
                };
                assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{line}");
                assert!(err.to_string().contains(reason), "{line}: got {err}, wanted {reason:?}");
                seen_codec += 1;
            }
            "staging" => {
                let mut st = Staging::new();
                if let Some(pre) = e["prelude"].as_array() {
                    for p in pre {
                        let Request::Tick(t) = parse_request(p.as_str().unwrap()).unwrap() else {
                            panic!("prelude must be ticks")
                        };
                        st.stage(&cfg, &t).unwrap_or_else(|m| panic!("prelude rejected: {m}"));
                    }
                }
                let Request::Tick(t) = parse_request(line).unwrap() else {
                    panic!("staging lines are ticks")
                };
                let msg = st.stage(&cfg, &t).expect_err(line);
                assert!(msg.contains(reason), "{line}: got {msg:?}, wanted {reason:?}");
                // after a rejection the staged input never carries a chunk
                assert!(st.input(&t, None, true).chunk.is_none());
                seen_staging += 1;
            }
            other => panic!("unknown stage {other}"),
        }
    }
    assert!(seen_codec >= 12 && seen_staging >= 8, "codec {seen_codec} staging {seen_staging}");
    // the spec's mandatory cases are all present (by their fixture notes)
    for must in [
        "unknown key inside obs",
        "unknown key inside chunk",
        "unknown key inside run",
        "unknown key inside ack",
        "a chunk with 14 rows",
        "t_emit > t",
        "idx != t - t_emit",
        "a second chunk with seq 0",
        "the non-JSON literal NaN",
    ] {
        assert!(notes.iter().any(|n| n.starts_with(must)), "missing mandatory bad line: {must}");
    }
}

#[test]
fn staging_null_becomes_nan_and_chunk_is_staged() {
    let env = SafetyEnvelope::from_toml(ENVELOPE).unwrap();
    let cfg = env.compile(FuseMode::Observe, None).unwrap();
    let line = lines(GOLDEN_REQUESTS).nth(5).unwrap(); // the tick with nulls (no chunk)
    let Request::Tick(t) = parse_request(line).unwrap() else { panic!() };
    let mut st = Staging::new();
    // t == 25 without a chunk: stage succeeds (continuity is decide()'s business)
    st.stage(&cfg, &t).unwrap();
    let inp = st.input(&t, None, false);
    assert!(inp.obs.pos[0].is_nan() && inp.obs.pos[1] == 301.0);
    assert!(inp.obs.vel.is_none());
    assert!(inp.obs.aux[1].is_nan());
    assert_eq!(inp.obs.ext.len(), 2);
    assert!(inp.obs.ext[1].is_nan());
    assert!(inp.chunk.is_none());
    assert!(!inp.schema_fault);
    // the full chunk tick from the fixture (t=24, seq 3): stage it as the 4th chunk of an episode
    let line = lines(GOLDEN_REQUESTS).nth(4).unwrap();
    let Request::Tick(t) = parse_request(line).unwrap() else { panic!() };
    let mut st = Staging::new();
    for (i, t0) in [0u32, 8, 16].iter().enumerate() {
        let mut v: serde_json::Value = serde_json::from_str(line).unwrap();
        v["t"] = serde_json::json!(t0);
        v["chunk"]["t_emit"] = serde_json::json!(t0);
        v["chunk"]["seq"] = serde_json::json!(i);
        let Request::Tick(tt) = serde_json::from_value(v).unwrap() else { panic!() };
        st.stage(&cfg, &tt).unwrap();
    }
    st.stage(&cfg, &t).unwrap();
    let inp = st.input(&t, None, false);
    let ch = inp.chunk.expect("chunk staged");
    assert_eq!((ch.seq, ch.t_emit, ch.horizon, ch.dim, ch.exec_steps), (3, 24, 15, 2, 8));
    assert_eq!(ch.action(0), &[214.0, 300.2]);
    assert_eq!(ch.action(14), &[228.5, 286.4]);
    assert_eq!(inp.obs.vel, Some(&[41.2, -12.7][..]));
}
