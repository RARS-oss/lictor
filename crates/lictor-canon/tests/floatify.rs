// SPDX-License-Identifier: MIT
//! `floatify` turns a TOML-sourced config tree (plain JSON reals) into the float-free profile; the result
//! canonicalises, round-trips through `F64Hex`, and leaves integers, strings and booleans untouched.

use lictor_canon::{
    canon, canon_of, digest_of, f64_from_hex, f64_to_hex, floatify, CanonError, F64Array, F64Hex,
};

#[test]
fn toml_like_tree_becomes_float_free() {
    let tree = serde_json::json!({
        "schema": "lictor-envelope/v1",
        "limits": {"speed": 540.0, "accel": 9000.5, "jerk": 1.8e5, "n": 3, "enabled": true},
        "workspace": {"min": [0.0, 0.0], "max": [512.0, 512.0]},
        "nested": [{"a": -0.0}, {"b": [1, 2.5]}],
        "name": "pusht"
    });
    assert!(matches!(canon(&tree), Err(CanonError::FloatInBody(_))));
    let ff = floatify(tree);
    let bytes = canon(&ff).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    assert!(!text.contains("540.0"));
    assert!(text.contains("{\"f64\":\"4080e00000000000\"}"), "{text}");
    assert_eq!(ff["limits"]["n"], 3);
    assert_eq!(ff["limits"]["enabled"], true);
    assert_eq!(ff["name"], "pusht");
    assert_eq!(ff["nested"][0]["a"]["f64"], "8000000000000000");
    assert_eq!(ff["nested"][1]["b"][0], 1);
    assert_eq!(ff["nested"][1]["b"][1]["f64"], f64_to_hex(2.5));
    // Every {"f64":..} decodes back to the original real.
    let speed: F64Hex = serde_json::from_value(ff["limits"]["speed"].clone()).unwrap();
    assert_eq!(speed, F64Hex(540.0));
    assert_eq!(f64_from_hex(ff["limits"]["jerk"]["f64"].as_str().unwrap()).unwrap(), 1.8e5);
}

#[test]
fn floatify_is_idempotent_and_digest_stable() {
    let tree = serde_json::json!({"x": 0.1, "y": [0.2, {"z": 0.3}]});
    let once = floatify(tree.clone());
    let twice = floatify(once.clone());
    assert_eq!(once, twice);
    assert_eq!(digest_of(&once).unwrap(), digest_of(&twice).unwrap());
    assert_ne!(digest_of(&floatify(serde_json::json!({"x": 0.1}))).unwrap(), digest_of(&once).unwrap());
}

#[test]
fn typed_values_and_floatified_values_agree() {
    #[derive(serde::Serialize)]
    struct Typed {
        s: F64Hex,
        a: F64Array,
        n: u32,
    }
    let typed = Typed { s: F64Hex(1.1), a: F64Array::from_slice(&[13.375, 300.4], &[2]), n: 7 };
    let bytes = canon_of(&typed).unwrap();
    assert_eq!(
        String::from_utf8(bytes).unwrap(),
        "{\"a\":{\"f64a\":\"AAAAAADAKkBmZmZmZsZyQA==\",\"shape\":[2]},\"n\":7,\"s\":{\"f64\":\"3ff199999999999a\"}}"
    );
    let ff = floatify(serde_json::json!({"s": 1.1, "n": 7}));
    assert_eq!(ff["s"], serde_json::to_value(F64Hex(1.1)).unwrap());
}

#[test]
fn special_values_survive() {
    for x in [f64::INFINITY, f64::NEG_INFINITY, f64::MAX, f64::MIN_POSITIVE, 5e-324, -0.0] {
        let h = f64_to_hex(x);
        assert_eq!(f64_from_hex(&h).unwrap().to_bits(), x.to_bits());
    }
    // A NaN payload round-trips bit-exactly (the wire encodes it as null; the receipt keeps the bits).
    let nan = f64::from_bits(0x7ff8_0000_dead_beef);
    assert_eq!(f64_from_hex(&f64_to_hex(nan)).unwrap().to_bits(), nan.to_bits());
}
