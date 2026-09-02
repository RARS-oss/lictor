// SPDX-License-Identifier: MIT
//! Handoff/ack: the committed handoff.json + ack.json verify under operator.hex; every `AckError` variant is
//! reachable; the nonce file round-trips.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lictor_core::AckDecision;
use lictor_receipt::{
    ack_signing_bytes, load_nonces, load_seed, pubkey_hex, save_nonces, sign_ack, verify_ack, AckError,
    AckToken, HandoffRecord, HANDOFF_SCHEMA,
};

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("receipt")
}

fn load() -> (HandoffRecord, AckToken, [u8; 32]) {
    let h =
        serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("handoff.json")).unwrap()).unwrap();
    let a = serde_json::from_str(&std::fs::read_to_string(fixture_dir().join("ack.json")).unwrap()).unwrap();
    let seed = load_seed(&fixture_dir().join("operator.hex")).unwrap();
    (h, a, seed)
}

#[test]
fn committed_ack_is_accepted() {
    let (h, a, seed) = load();
    let op = pubkey_hex(&seed);
    assert_eq!(a.operator, op);
    assert_eq!(a.schema, HANDOFF_SCHEMA);
    assert_eq!(h.schema, HANDOFF_SCHEMA);
    assert_eq!(h.compute_digest().unwrap(), h.digest);
    assert_eq!(a.handoff_digest, h.digest);
    let operators = vec!["deadbeef".to_string(), op.clone()];
    let v = verify_ack(&a, &operators, Some(&h.digest), h.seq, &[0, 0]).unwrap();
    assert_eq!(v.decision, AckDecision::Resume);
    assert_eq!(v.operator_slot, 1);
    assert_eq!(v.nonce, 7);
    assert_eq!(v.handoff_seq, h.seq);
    // A shorter last_nonce table than the operator list reads as 0 for the missing slots.
    assert!(verify_ack(&a, &operators, Some(&h.digest), h.seq, &[]).is_ok());
    // The signing bytes are the canonical six-field object.
    let bytes = String::from_utf8(ack_signing_bytes(&a).unwrap()).unwrap();
    assert!(bytes.starts_with("{\"decision\":\"resume\",\"handoff_digest\":\""));
    assert!(bytes.ends_with(&format!(
        "\"nonce\":7,\"note\":\"oracle resume\",\"operator\":\"{op}\",\"schema\":\"{HANDOFF_SCHEMA}\"}}"
    )));
    assert!(!bytes.contains("sig"));
}

#[test]
fn every_error_variant() {
    let (h, a, seed) = load();
    let op = pubkey_hex(&seed);
    let operators = vec![op.clone()];
    let pending = Some(h.digest.as_str());

    assert_eq!(verify_ack(&a, &["other".to_string()], pending, 0, &[0]), Err(AckError::UnknownOperator));

    let mut bad_sig = a.clone();
    bad_sig.sig = format!("00{}", &a.sig[2..]);
    assert_eq!(verify_ack(&bad_sig, &operators, pending, 0, &[0]), Err(AckError::BadSignature));
    let mut edited_note = a.clone();
    edited_note.note = "oracle resume!".into();
    assert_eq!(verify_ack(&edited_note, &operators, pending, 0, &[0]), Err(AckError::BadSignature));
    let mut bad_schema = a.clone();
    bad_schema.schema = "lictor-handoff/v0".into();
    assert_eq!(verify_ack(&bad_schema, &operators, pending, 0, &[0]), Err(AckError::BadSignature));

    let other = sign_ack(&"ab".repeat(32), AckDecision::Abort, 9, "", &seed).unwrap();
    assert_eq!(verify_ack(&other, &operators, pending, 0, &[0]), Err(AckError::WrongHandoff));

    assert_eq!(verify_ack(&a, &operators, None, 0, &[0]), Err(AckError::NoPendingHandoff));

    assert_eq!(verify_ack(&a, &operators, pending, 0, &[7]), Err(AckError::NonceReplay));
    assert_eq!(verify_ack(&a, &operators, pending, 0, &[8]), Err(AckError::NonceReplay));
    assert!(verify_ack(&a, &operators, pending, 0, &[6]).is_ok());

    let long = "n".repeat(201);
    assert!(sign_ack(&h.digest, AckDecision::Retune, 8, &long, &seed).is_err());
    let mut too_long = a.clone();
    too_long.note = long;
    assert_eq!(verify_ack(&too_long, &operators, pending, 0, &[0]), Err(AckError::TooLongNote));
    let exactly = "n".repeat(200);
    let ok = sign_ack(&h.digest, AckDecision::Retune, 8, &exactly, &seed).unwrap();
    assert_eq!(verify_ack(&ok, &operators, pending, 0, &[7]).unwrap().decision, AckDecision::Retune);
}

#[test]
fn handoff_digest_excludes_resolution_fields_and_binds_the_run() {
    let (h, a, _) = load();
    let mut resolved = h.clone();
    resolved.ack = Some(a);
    resolved.resolved_tick = Some(98);
    resolved.outcome = "resumed".into();
    assert_eq!(resolved.compute_digest().unwrap(), h.digest, "attaching the ack must not move the digest");
    for (label, mutant) in [
        ("run_id", {
            let mut m = h.clone();
            m.run_id.push('x');
            m
        }),
        ("arm_id", {
            let mut m = h.clone();
            m.arm_id.push('x');
            m
        }),
        ("episode_index", {
            let mut m = h.clone();
            m.episode_index += 1;
            m
        }),
        ("chain_at", {
            let mut m = h.clone();
            m.chain_at = "00".repeat(32);
            m
        }),
        ("tick", {
            let mut m = h.clone();
            m.tick += 1;
            m
        }),
    ] {
        assert_ne!(mutant.compute_digest().unwrap(), h.digest, "{label} must be digested");
    }
}

#[test]
fn nonce_file_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".lictor").join("verifier_nonce.json");
    assert!(load_nonces(&path).unwrap().is_empty());
    let mut m = BTreeMap::new();
    m.insert("ab".repeat(32), 7u64);
    m.insert("cd".repeat(32), 12u64);
    save_nonces(&path, &m).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("{\n  \"abab"), "pretty JSON with sorted keys: {text}");
    assert_eq!(load_nonces(&path).unwrap(), m);
    std::fs::write(&path, "not json").unwrap();
    assert!(load_nonces(&path).is_err());
}
