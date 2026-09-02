// SPDX-License-Identifier: MIT
//! Ledger append/read round trip in a tempdir; header handling; canonical line format.

use std::path::PathBuf;

use lictor_receipt::{
    append_ledger, ledger_entry, load_seed, read_ledger, sign, verify_ledger, LedgerEntry, SignedReceipt,
    LEDGER_SCHEMA, ZERO_HASH,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("receipt")
}

fn receipt_for(episode_index: u32) -> SignedReceipt {
    let sr: SignedReceipt =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("receipt_000007.json")).unwrap())
            .unwrap();
    let seed = load_seed(&fixture_dir().join("key.hex")).unwrap();
    let mut body = sr.body;
    body.run.episode_index = episode_index;
    body.run.seed = u64::from(episode_index);
    body.outcome.success = episode_index.is_multiple_of(2);
    sign(body, &seed).unwrap()
}

#[test]
fn append_and_read_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("arm").join("ledger.jsonl");
    let r0 = receipt_for(0);
    let e0 = append_ledger(&path, &r0).unwrap();
    assert_eq!(e0.seq, 0);
    assert_eq!(e0.prev, ZERO_HASH);
    assert_eq!(e0.receipt_digest, r0.body_digest);
    assert_eq!(e0.hash, e0.recompute_hash());
    let e1 = append_ledger(&path, &receipt_for(1)).unwrap();
    let e2 = append_ledger(&path, &receipt_for(2)).unwrap();
    assert_eq!((e1.seq, e2.seq), (1, 2));
    assert_eq!(e1.prev, e0.hash);
    assert_eq!(e2.prev, e1.hash);

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 4, "header + 3 entries");
    let header: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(header["schema"], LEDGER_SCHEMA);
    assert_eq!(header["genesis"], ZERO_HASH);
    assert_eq!(header["run_id"], r0.body.run.run_id);
    assert_eq!(header["arm_id"], r0.body.run.arm_id);
    // Every line is canonical (compact, sorted keys): it equals its own canon.
    for l in &lines {
        let v: serde_json::Value = serde_json::from_str(l).unwrap();
        assert_eq!(lictor_canon::canon(&v).unwrap(), l.as_bytes());
    }

    let entries = read_ledger(&path).unwrap();
    assert_eq!(entries, vec![e0.clone(), e1.clone(), e2.clone()]);
    let report = verify_ledger(&entries);
    assert!(report.chain_ok);
    assert_eq!(report.head, e2.hash);
    assert_eq!(
        (report.episodes, report.successes, report.trips, report.stops, report.escalations),
        (3, 2, 3, 3, 0)
    );

    // The entry builder is deterministic given (prev, seq, receipt).
    assert_eq!(ledger_entry(&e0.hash, 1, &receipt_for(1)), e1);
}

#[test]
fn header_is_optional_on_read_and_blank_lines_are_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ledger.jsonl");
    append_ledger(&path, &receipt_for(0)).unwrap();
    append_ledger(&path, &receipt_for(1)).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let without_header: String = text.lines().skip(1).map(|l| format!("{l}\n\n")).collect();
    std::fs::write(&path, without_header).unwrap();
    let entries = read_ledger(&path).unwrap();
    assert_eq!(entries.len(), 2);
    assert!(verify_ledger(&entries).chain_ok);
    // A foreign header schema is refused.
    std::fs::write(&path, "{\"schema\":\"other/v1\"}\n").unwrap();
    assert!(read_ledger(&path).is_err());
    // A non-entry line is refused with its line number.
    std::fs::write(&path, "{\"seq\":0}\n").unwrap();
    let err = read_ledger(&path).unwrap_err().to_string();
    assert!(err.contains(":1:"), "{err}");
    assert!(read_ledger(&dir.path().join("missing.jsonl")).is_err());
}

#[test]
fn empty_ledger_verifies_to_genesis() {
    let r = verify_ledger(&[]);
    assert!(r.chain_ok && r.episodes == 0 && r.head == ZERO_HASH && r.break_at.is_none());
    let e: LedgerEntry = read_ledger(&fixture_dir().join("ledger.jsonl")).unwrap().remove(0);
    assert_eq!(e.seq, 0);
    assert_eq!(e.prev, ZERO_HASH);
}
