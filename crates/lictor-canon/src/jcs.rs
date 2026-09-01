// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! RFC 8785 JSON Canonicalization Scheme restricted to the float-free profile.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CanonError {
    #[error("float in canonical body at {0}")]
    FloatInBody(String),
    #[error("integer out of +-2^53-1 at {0}")]
    IntegerOutOfRange(String),
    #[error("key not [a-z0-9_]+: {0}")]
    BadKey(String),
    #[error("bad f64 encoding: {0}")]
    BadF64(String),
    #[error("serialize: {0}")]
    Serialize(String),
}

pub const CANONICAL_ID: &str = "jcs-floatfree/v1";

/// RFC 8785: keys sorted by UTF-16 code units, no whitespace, minimal escaping; integers only (|v| <= 2^53-1).
pub fn canon(_v: &Value) -> Result<Vec<u8>, CanonError> {
    todo!("WP-4")
}

pub fn check_keys(_v: &Value) -> Result<(), CanonError> {
    todo!("WP-4")
}
