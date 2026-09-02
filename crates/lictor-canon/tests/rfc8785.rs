// SPDX-License-Identifier: MIT
//! RFC 8785 vectors (the section 3.2.3 example, the appendix B UTF-16 sorting vector, string escaping) plus the
//! float-free profile's rejections, driven by tests/fixtures/jcs/vectors.json. Each vector carries its input as
//! JSON TEXT (so the fixture file's own keys stay in the profile's key class) and either the expected canonical
//! text or the expected error message.

use lictor_canon::{canon, sha256_jcs, CanonError, CANONICAL_ID};

#[derive(serde::Deserialize)]
struct Vector {
    name: String,
    input: String,
    #[serde(default)]
    expected: Option<String>,
    #[serde(default)]
    error: Option<String>,
}

fn vectors() -> Vec<Vector> {
    let text = include_str!("fixtures/jcs/vectors.json");
    serde_json::from_str(text).expect("vectors.json parses")
}

#[test]
fn every_vector_canonicalises_as_expected() {
    let vs = vectors();
    assert!(vs.len() >= 12);
    for v in &vs {
        let value: serde_json::Value =
            serde_json::from_str(&v.input).unwrap_or_else(|e| panic!("{}: {e}", v.name));
        match (canon(&value), &v.expected, &v.error) {
            (Ok(bytes), Some(exp), None) => {
                let got = String::from_utf8(bytes).expect("canonical bytes are UTF-8");
                assert_eq!(&got, exp, "{}", v.name);
                // Idempotent: canonical text re-parses and re-canonicalises to itself.
                let again: serde_json::Value = serde_json::from_str(&got).unwrap();
                assert_eq!(canon(&again).unwrap(), got.as_bytes(), "{} (idempotence)", v.name);
            }
            (Err(e), None, Some(msg)) => assert_eq!(&e.to_string(), msg, "{}", v.name),
            (r, _, _) => panic!("{}: unexpected outcome {r:?}", v.name),
        }
    }
}

#[test]
fn appendix_b_order_is_utf16_not_code_point() {
    // U+1F600 (surrogate pair d83d de00) sorts before U+FB33 under UTF-16 code units and after it under code points.
    let v: serde_json::Value = serde_json::from_str("{\"\\ufb33\": 1, \"\\ud83d\\ude00\": 2}").unwrap();
    let got = String::from_utf8(canon(&v).unwrap()).unwrap();
    assert_eq!(got, "{\"\u{1f600}\":2,\"\u{fb33}\":1}");
    let mut by_code_point: Vec<&str> = vec!["\u{fb33}", "\u{1f600}"];
    by_code_point.sort();
    assert_eq!(by_code_point, vec!["\u{fb33}", "\u{1f600}"], "code-point order differs, which is the point");
}

#[test]
fn integer_range_boundaries() {
    let ok: serde_json::Value = serde_json::from_str("[9007199254740991, -9007199254740991]").unwrap();
    assert_eq!(canon(&ok).unwrap(), b"[9007199254740991,-9007199254740991]");
    let too_big: serde_json::Value = serde_json::from_str("[9007199254740992]").unwrap();
    assert_eq!(canon(&too_big), Err(CanonError::IntegerOutOfRange("$[0]".into())));
    let u64_max: serde_json::Value = serde_json::from_str("[18446744073709551615]").unwrap();
    assert_eq!(canon(&u64_max), Err(CanonError::IntegerOutOfRange("$[0]".into())));
    let i64_min: serde_json::Value = serde_json::from_str("[-9223372036854775808]").unwrap();
    assert_eq!(canon(&i64_min), Err(CanonError::IntegerOutOfRange("$[0]".into())));
}

#[test]
fn floats_are_rejected_with_paths() {
    let v = serde_json::json!({"outer": {"inner": [0, 1, {"x": 2.5}]}});
    assert_eq!(canon(&v), Err(CanonError::FloatInBody("$.outer.inner[2].x".into())));
    assert!(matches!(sha256_jcs(&v), Err(CanonError::FloatInBody(_))));
    assert_eq!(CANONICAL_ID, "jcs-floatfree/v1");
}

#[test]
fn sha256_jcs_over_the_rfc_example() {
    let v = vectors().into_iter().next().unwrap();
    let value: serde_json::Value = serde_json::from_str(&v.input).unwrap();
    let d = sha256_jcs(&value).unwrap();
    assert_eq!(d.len(), 64);
    assert_eq!(d, lictor_canon::sha256_hex(v.expected.as_ref().unwrap().as_bytes()));
}
