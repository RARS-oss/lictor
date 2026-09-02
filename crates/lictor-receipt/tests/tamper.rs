// SPDX-License-Identifier: MIT
//! Tamper tests over the committed fixtures: every scalar of the body mutated one at a time, one tick edited,
//! a re-chained ticks file, whitespace-only edits, re-signing with another key, a deleted ledger entry, and the
//! two honest-but-not-protective receipts (Observe, ephemeral key).

use std::path::PathBuf;

use lictor_core::FuseMode;
use lictor_receipt::{
    evaluate_fuse, load_seed, pubkey_hex, read_ledger, sign, verify, verify_ledger, verify_ticks_file,
    SignedReceipt, TickEvent, ZERO_HASH,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("receipt")
}

fn receipt() -> SignedReceipt {
    serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("receipt_000007.json")).unwrap())
        .unwrap()
}

fn ticks() -> Vec<TickEvent> {
    std::fs::read_to_string(fixture_dir().join("ticks_000007.jsonl"))
        .unwrap()
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn expected_pubkey() -> String {
    std::fs::read_to_string(fixture_dir().join("pubkey.txt")).unwrap().trim().to_string()
}

/// Collect the JSON pointer of every scalar leaf (numbers, strings, booleans, nulls) of `v`.
fn leaves(v: &serde_json::Value, path: String, out: &mut Vec<String>) {
    match v {
        serde_json::Value::Object(m) => {
            for (k, c) in m {
                leaves(c, format!("{path}/{}", k.replace('~', "~0").replace('/', "~1")), out);
            }
        }
        serde_json::Value::Array(a) => {
            for (i, c) in a.iter().enumerate() {
                leaves(c, format!("{path}/{i}"), out);
            }
        }
        _ => out.push(path),
    }
}

fn mutate(v: &mut serde_json::Value) {
    *v = match v.take() {
        serde_json::Value::Null => serde_json::Value::from(0),
        serde_json::Value::Bool(b) => serde_json::Value::Bool(!b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                serde_json::Value::from(i + 1)
            } else {
                serde_json::Value::from(n.as_u64().unwrap_or(0).wrapping_add(1))
            }
        }
        serde_json::Value::String(s) => {
            serde_json::Value::String(if s.is_empty() { "x".into() } else { s[1..].to_string() })
        }
        other => other,
    };
}

#[test]
fn fixture_is_intact_and_fuse_ok() {
    let sr = receipt();
    let r = verify(&sr, Some(&expected_pubkey()));
    assert!(r.intact(), "{r:?}");
    assert!(r.fuse_ok);
    assert!(r.notes.is_empty(), "{:?}", r.notes);
    assert_eq!(sr.pubkey, lictor_receipt::TEST_PUBKEYS[0]);
}

#[test]
fn every_scalar_of_the_body_is_bound() {
    let sr = receipt();
    let mut value = serde_json::to_value(&sr).unwrap();
    let mut paths = Vec::new();
    leaves(&value["body"], "/body".into(), &mut paths);
    assert!(paths.len() > 500, "expected hundreds of scalar leaves, found {}", paths.len());
    let (mut by_sig, mut by_parse) = (0usize, 0usize);
    for p in &paths {
        let original = value.pointer(p).unwrap().clone();
        mutate(value.pointer_mut(p).unwrap());
        match serde_json::from_value::<SignedReceipt>(value.clone()) {
            Ok(mutated) => {
                let r = verify(&mutated, Some(&expected_pubkey()));
                assert!(!r.sig_ok, "mutating {p} left sig_ok == true");
                assert!(!r.digest_ok, "mutating {p} left digest_ok == true");
                by_sig += 1;
            }
            Err(_) => by_parse += 1, // an enum / hex / base64 field: the typed parser rejects the edit outright
        }
        *value.pointer_mut(p).unwrap() = original;
    }
    eprintln!("{} leaves: {by_sig} caught by the signature, {by_parse} rejected at parse", paths.len());
    assert!(by_sig > 300);
    // The untouched value still verifies (the restore step is faithful).
    let back: SignedReceipt = serde_json::from_value(value).unwrap();
    assert!(verify(&back, Some(&expected_pubkey())).intact());
}

#[test]
fn one_tick_edited_breaks_the_ticks_file_at_its_seq() {
    let sr = receipt();
    let mut t = ticks();
    let ok = verify_ticks_file(&sr, &t);
    assert!(ok.ok && ok.head == sr.body.verdict_chain_head && ok.n == 300);
    t[17].trips ^= 1;
    let r = verify_ticks_file(&sr, &t);
    assert!(!r.ok);
    assert_eq!(r.break_at, Some(17));
}

#[test]
fn a_rechained_ticks_file_passes_integrity_but_not_the_head() {
    let sr = receipt();
    let mut t = ticks();
    t[17].trips ^= 1;
    let mut prev = t[16].hash.clone();
    for e in t.iter_mut().skip(17) {
        e.prev = prev;
        e.hash = e.recompute_hash();
        prev = e.hash.clone();
    }
    let r = verify_ticks_file(&sr, &t);
    assert!(r.ok, "chain integrity holds on a re-chained file: {r:?}");
    assert_eq!(r.n, 300);
    assert_ne!(r.head, sr.body.verdict_chain_head, "HEAD MISMATCH is what the caller must report");
    // The embedded tail (signed) no longer matches the file's tail.
    assert_ne!(t[299], sr.body.ticks[31]);
}

#[test]
fn whitespace_only_edits_do_not_matter() {
    let pretty = std::fs::read_to_string(fixture_dir().join("receipt_000007.json")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&pretty).unwrap();
    let compact = serde_json::to_string(&v).unwrap();
    assert_ne!(compact, pretty);
    let sr: SignedReceipt = serde_json::from_str(&compact).unwrap();
    assert!(verify(&sr, Some(&expected_pubkey())).intact());
    let spaced = pretty.replace('\n', "\r\n").replace("  ", "\t");
    let sr2: SignedReceipt = serde_json::from_str(&spaced).unwrap();
    assert!(verify(&sr2, Some(&expected_pubkey())).intact());
    assert_eq!(sr2.body_digest, sr.body.digest_hex().unwrap());
}

#[test]
fn resigned_with_another_key_is_valid_but_not_the_expected_key() {
    let sr = receipt();
    let other = [9u8; 32];
    let re = sign(sr.body.clone(), &other).unwrap();
    assert_ne!(re.pubkey, sr.pubkey);
    assert_eq!(re.body_digest, sr.body_digest);
    let r = verify(&re, Some(&expected_pubkey()));
    assert!(r.sig_ok && r.digest_ok);
    assert!(!r.pubkey_ok);
    assert!(!r.intact());
    assert!(r.notes.iter().any(|n| n.contains("pubkey mismatch")));
    assert!(verify(&re, None).intact());
    assert!(verify(&re, Some(&pubkey_hex(&other))).intact());
}

#[test]
fn deleting_ledger_entry_three_breaks_at_three() {
    let entries = read_ledger(&fixture_dir().join("ledger.jsonl")).unwrap();
    assert_eq!(entries.len(), 5);
    assert!(verify_ledger(&entries).chain_ok);
    let mut dropped = entries.clone();
    dropped.remove(3);
    let r = verify_ledger(&dropped);
    assert!(!r.chain_ok);
    assert_eq!(r.break_at, Some(3));
    assert_eq!(r.episodes, 4);
    // Reordering breaks too; an edited flag breaks at the edited seq.
    let mut swapped = entries.clone();
    swapped.swap(1, 2);
    assert_eq!(verify_ledger(&swapped).break_at, Some(1));
    let mut edited = entries.clone();
    edited[2].success = !edited[2].success;
    assert_eq!(verify_ledger(&edited).break_at, Some(2));
    // Truncating the tail is NOT detected by the chain alone (documented gap; `lictor curve` checks the pool).
    assert!(verify_ledger(&entries[..4]).chain_ok);
}

#[test]
fn observe_receipt_is_intact_but_not_fuse_ok() {
    let sr = receipt();
    let seed = load_seed(&fixture_dir().join("key.hex")).unwrap();
    let mut body = sr.body.clone();
    body.budget.mode = FuseMode::Observe;
    let (ok, notes) = evaluate_fuse(&body.budget, &body.counts, body.calibration_digest.as_deref(), false);
    body.fuse_ok = ok;
    body.fuse_notes = notes;
    let obs = sign(body, &seed).unwrap();
    let r = verify(&obs, Some(&expected_pubkey()));
    assert!(r.intact(), "{r:?}");
    assert!(!r.fuse_ok);
    assert!(r
        .notes
        .iter()
        .any(|n| n == "the fuse observed but did not enforce -- this receipt does not attest protection"));
    // A receipt whose flag lies (fuse_ok: true under Observe) is still intact but reported.
    let mut lying = obs.body.clone();
    lying.fuse_ok = true;
    let lying = sign(lying, &seed).unwrap();
    let r = verify(&lying, None);
    assert!(r.intact() && !r.fuse_ok);
    assert!(r.notes.iter().any(|n| n.contains("disagrees with the recomputed evaluation")));
}

#[test]
fn ephemeral_key_receipt_is_not_fuse_ok() {
    let sr = receipt();
    let mut body = sr.body.clone();
    body.fuse_notes.push("ephemeral signing key".into());
    body.fuse_ok = false;
    let eph = sign(body, &[3u8; 32]).unwrap();
    let r = verify(&eph, None);
    assert!(r.intact(), "{r:?}");
    assert!(!r.fuse_ok);
    assert!(r.notes.iter().any(|n| n == "ephemeral signing key"));
}

#[test]
fn schema_and_counts_and_envelope_digest_are_checked() {
    let sr = receipt();
    let seed = load_seed(&fixture_dir().join("key.hex")).unwrap();
    let mut b = sr.body.clone();
    b.schema = "lictor-receipt/v0".into();
    let r = verify(&sign(b, &seed).unwrap(), None);
    assert!(!r.schema_ok && r.sig_ok && !r.intact());
    let mut b = sr.body.clone();
    b.outcome.steps += 1;
    let r = verify(&sign(b, &seed).unwrap(), None);
    assert!(!r.counts_ok && r.sig_ok && !r.intact());
    let mut b = sr.body.clone();
    b.envelope["limits"]["speed"] = serde_json::json!({"f64": "4081000000000000"});
    let r = verify(&sign(b, &seed).unwrap(), None);
    assert!(!r.envelope_digest_ok && r.sig_ok && !r.intact());
    let mut b = sr.body.clone();
    b.ticks.truncate(31);
    let r = verify(&sign(b, &seed).unwrap(), None);
    assert!(!r.ticks_ok && !r.intact());
    let mut b = sr.body.clone();
    b.ticks[5].t += 1;
    let r = verify(&sign(b, &seed).unwrap(), None);
    assert!(!r.ticks_ok && r.break_at == Some(273), "{r:?}");
    let mut b = sr.body.clone();
    b.ticks_policy = "none".into();
    b.ticks.clear();
    let r = verify(&sign(b, &seed).unwrap(), None);
    assert!(r.intact(), "a `none` policy embeds nothing and is still intact: {r:?}");
    assert_eq!(ZERO_HASH.len(), 64);
}
