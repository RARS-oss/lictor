// SPDX-License-Identifier: MIT
//! history.jsonl: the 200-line cap and the two loop signals.

use lictor_receipt::{record_episode, summarize, EpisodeRecord, LoopSignals};

fn rec(i: u32, arm: &str, reason: Option<&str>, trips: &[&str]) -> EpisodeRecord {
    EpisodeRecord {
        ts: format!("2026-09-01T09:{:02}:00Z", i % 60),
        run_id: "run".into(),
        arm_id: arm.into(),
        seed: u64::from(i),
        success: reason.is_none(),
        first_trip_reason: reason.map(str::to_string),
        trips: trips.iter().map(|s| s.to_string()).collect(),
        stopped: reason.is_some(),
        escalated: false,
    }
}

#[test]
fn empty_history_has_no_signals() {
    let dir = tempfile::tempdir().unwrap();
    let s = summarize(dir.path(), 10).unwrap();
    assert_eq!(s, LoopSignals { tail_streak: None, identical_run: 0, last_n: 0 });
}

#[test]
fn cap_at_200_lines() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..205 {
        record_episode(dir.path(), &rec(i, "a", None, &[])).unwrap();
    }
    let text = std::fs::read_to_string(dir.path().join("history.jsonl")).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 200);
    let first: EpisodeRecord = serde_json::from_str(lines[0]).unwrap();
    let last: EpisodeRecord = serde_json::from_str(lines[199]).unwrap();
    assert_eq!((first.seed, last.seed), (5, 204));
    assert!(!dir.path().join("history.jsonl.tmp").exists());
    assert_eq!(summarize(dir.path(), 500).unwrap().last_n, 200);
}

#[test]
fn tail_streak_and_identical_run() {
    let dir = tempfile::tempdir().unwrap();
    record_episode(dir.path(), &rec(0, "a", Some("brake_tier1_cp"), &["tier1_cp"])).unwrap();
    record_episode(dir.path(), &rec(1, "a", None, &[])).unwrap();
    record_episode(dir.path(), &rec(2, "b", Some("brake_tier1_cp"), &["tier1_cp"])).unwrap();
    record_episode(dir.path(), &rec(3, "b", Some("clamp_speed"), &["speed"])).unwrap();
    record_episode(dir.path(), &rec(4, "b", Some("brake_tier1_cp"), &["tier1_cp"])).unwrap();
    let s = summarize(dir.path(), 5).unwrap();
    assert_eq!(s.last_n, 5);
    assert_eq!(s.tail_streak, Some(("brake_tier1_cp".to_string(), 3)));
    assert_eq!(s.identical_run, 1, "the last (arm, trips) differs from the one before it");
    // Only the last 2 considered: the streak needs >= 2 sharing the reason.
    let s = summarize(dir.path(), 2).unwrap();
    assert_eq!(s.tail_streak, None);
    assert_eq!(s.last_n, 2);
    // Three identical (arm, trips) tails in a row.
    record_episode(dir.path(), &rec(5, "b", Some("brake_tier1_cp"), &["tier1_cp"])).unwrap();
    record_episode(dir.path(), &rec(6, "b", Some("brake_tier1_cp"), &["tier1_cp"])).unwrap();
    let s = summarize(dir.path(), 10).unwrap();
    assert_eq!(s.identical_run, 3);
    assert_eq!(s.tail_streak, Some(("brake_tier1_cp".to_string(), 5)));
    // A trailing success (no reason) yields no streak; identical_run counts the success alone.
    record_episode(dir.path(), &rec(7, "b", None, &[])).unwrap();
    let s = summarize(dir.path(), 10).unwrap();
    assert_eq!(s.tail_streak, None);
    assert_eq!(s.identical_run, 1);
}

#[test]
fn corrupt_lines_are_skipped() {
    let dir = tempfile::tempdir().unwrap();
    record_episode(dir.path(), &rec(0, "a", None, &[])).unwrap();
    let path = dir.path().join("history.jsonl");
    let mut text = std::fs::read_to_string(&path).unwrap();
    text.push_str("{\"partial\": tru");
    std::fs::write(&path, text).unwrap();
    record_episode(dir.path(), &rec(1, "a", None, &[])).unwrap();
    assert_eq!(summarize(dir.path(), 10).unwrap().last_n, 2);
}
