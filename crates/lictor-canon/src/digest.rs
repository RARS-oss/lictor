// SPDX-License-Identifier: MIT
//! sha256 over canonical bytes.

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::jcs::{canon, check_keys, CanonError};

pub fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// sha256 over `canon(v)` (no key-charset check: the RFC 8785 vectors carry Unicode keys).
pub fn sha256_jcs(v: &Value) -> Result<String, CanonError> {
    Ok(sha256_hex(&canon(v)?))
}

/// to_value + check_keys + reject bare floats -> canonical bytes.
pub fn canon_of<T: serde::Serialize>(t: &T) -> Result<Vec<u8>, CanonError> {
    let v = serde_json::to_value(t).map_err(|e| CanonError::Serialize(e.to_string()))?;
    check_keys(&v)?;
    canon(&v)
}

pub fn digest_of<T: serde::Serialize>(t: &T) -> Result<String, CanonError> {
    Ok(sha256_hex(&canon_of(t)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::f64enc::F64Hex;
    use serde_json::json;

    #[test]
    fn sha256_known_answer() {
        assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn canon_of_checks_keys_and_floats() {
        #[derive(serde::Serialize)]
        struct S {
            b: u32,
            a: F64Hex,
        }
        let bytes = canon_of(&S { b: 2, a: F64Hex(1.0) }).unwrap();
        assert_eq!(bytes, br#"{"a":{"f64":"3ff0000000000000"},"b":2}"#.to_vec());
        assert_eq!(digest_of(&S { b: 2, a: F64Hex(1.0) }).unwrap(), sha256_hex(&bytes));
        assert_eq!(canon_of(&json!({"x": 0.5})), Err(CanonError::FloatInBody("$.x".into())));
        assert!(matches!(canon_of(&json!({"a b": 1})), Err(CanonError::BadKey(_))));
        assert_eq!(sha256_jcs(&json!({"b": 1, "a": [true]})).unwrap(), sha256_hex(br#"{"a":[true],"b":1}"#));
    }
}
