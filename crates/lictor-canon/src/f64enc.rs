// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! Float-free encodings: `{"f64":"<16hex>"}` scalars and `{"f64a":"<base64 LE bytes>","shape":[..]}` arrays.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::jcs::CanonError;

/// Replace every non-integer JSON number x by {"f64": f64_to_hex(x)} recursively (for TOML-sourced config trees).
pub fn floatify(_v: Value) -> Value {
    todo!("WP-4")
}

/// 16 lowercase hex of x.to_bits() (big-endian nibbles)
pub fn f64_to_hex(_x: f64) -> String {
    todo!("WP-4")
}

pub fn f64_from_hex(_s: &str) -> Result<f64, CanonError> {
    todo!("WP-4")
}

/// Serializes as {"f64":"<16hex>"}; deserializes from the same.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F64Hex(pub f64);

impl Serialize for F64Hex {
    fn serialize<S: Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
        todo!("WP-4")
    }
}

impl<'de> Deserialize<'de> for F64Hex {
    fn deserialize<D: Deserializer<'de>>(_d: D) -> Result<Self, D::Error> {
        todo!("WP-4")
    }
}

/// Serializes as {"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[..]}.
#[derive(Clone, Debug, PartialEq)]
pub struct F64Array {
    pub shape: Vec<u32>,
    pub data: Vec<f64>,
}

impl F64Array {
    pub fn from_slice(_d: &[f64], _shape: &[u32]) -> Self {
        todo!("WP-4")
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Serialize for F64Array {
    fn serialize<S: Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
        todo!("WP-4")
    }
}

impl<'de> Deserialize<'de> for F64Array {
    fn deserialize<D: Deserializer<'de>>(_d: D) -> Result<Self, D::Error> {
        todo!("WP-4")
    }
}
