// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! sha256 over canonical bytes.

use serde_json::Value;

use crate::jcs::CanonError;

pub fn sha256_hex(_bytes: &[u8]) -> String {
    todo!("WP-4")
}

pub fn sha256_jcs(_v: &Value) -> Result<String, CanonError> {
    todo!("WP-4")
}

/// to_value + check_keys + reject bare floats -> canonical bytes.
pub fn canon_of<T: serde::Serialize>(_t: &T) -> Result<Vec<u8>, CanonError> {
    todo!("WP-4")
}

pub fn digest_of<T: serde::Serialize>(_t: &T) -> Result<String, CanonError> {
    todo!("WP-4")
}
