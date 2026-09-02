// SPDX-License-Identifier: MIT
//! Float-free encodings: `{"f64":"<16hex>"}` scalars and `{"f64a":"<base64 LE bytes>","shape":[..]}` arrays.
//!
//! A real number never appears as a JSON number in a canonical artefact: its IEEE-754 bits travel as text, so
//! `+inf`, `-inf`, `-0.0` and every NaN payload round-trip exactly and no JSON parser gets to round anything.

use base64::Engine;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use crate::jcs::CanonError;

/// Replace every non-integer JSON number x by {"f64": f64_to_hex(x)} recursively (for TOML-sourced config trees).
pub fn floatify(v: Value) -> Value {
    match v {
        Value::Number(n) => {
            if n.is_i64() || n.is_u64() {
                Value::Number(n)
            } else {
                let x = n.as_f64().unwrap_or(f64::NAN);
                let mut m = serde_json::Map::new();
                m.insert("f64".to_string(), Value::String(f64_to_hex(x)));
                Value::Object(m)
            }
        }
        Value::Array(a) => Value::Array(a.into_iter().map(floatify).collect()),
        Value::Object(m) => Value::Object(m.into_iter().map(|(k, c)| (k, floatify(c))).collect()),
        other => other,
    }
}

/// 16 lowercase hex of x.to_bits() (big-endian nibbles)
pub fn f64_to_hex(x: f64) -> String {
    format!("{:016x}", x.to_bits())
}

pub fn f64_from_hex(s: &str) -> Result<f64, CanonError> {
    if s.len() != 16 || !s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) {
        return Err(CanonError::BadF64(format!("expected 16 lowercase hex digits, got {s:?}")));
    }
    let bits = u64::from_str_radix(s, 16).map_err(|e| CanonError::BadF64(e.to_string()))?;
    Ok(f64::from_bits(bits))
}

/// Serializes as {"f64":"<16hex>"}; deserializes from the same.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct F64Hex(pub f64);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct F64HexWire {
    f64: String,
}

impl Serialize for F64Hex {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        F64HexWire { f64: f64_to_hex(self.0) }.serialize(s)
    }
}

impl<'de> Deserialize<'de> for F64Hex {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = F64HexWire::deserialize(d)?;
        f64_from_hex(&w.f64).map(F64Hex).map_err(D::Error::custom)
    }
}

/// Serializes as {"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[..]}.
#[derive(Clone, Debug, PartialEq)]
pub struct F64Array {
    pub shape: Vec<u32>,
    pub data: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct F64ArrayWire {
    f64a: String,
    shape: Vec<u32>,
}

fn shape_product(shape: &[u32]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
}

impl F64Array {
    /// Panics (debug) only on a shape that does not describe `d`; production callers pass matching pairs.
    pub fn from_slice(d: &[f64], shape: &[u32]) -> Self {
        debug_assert_eq!(shape_product(shape), Some(d.len()), "F64Array shape must match the data length");
        Self { shape: shape.to_vec(), data: d.to_vec() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The base64 payload alone (little-endian f64 bytes, standard alphabet, padded).
    pub fn payload_base64(&self) -> String {
        let mut bytes = Vec::with_capacity(self.data.len() * 8);
        for x in &self.data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    /// Decode the wire form; `shape` product must equal the number of decoded values.
    pub fn from_wire(f64a: &str, shape: &[u32]) -> Result<Self, CanonError> {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(f64a)
            .map_err(|e| CanonError::BadF64(format!("f64a base64: {e}")))?;
        if bytes.len() % 8 != 0 {
            return Err(CanonError::BadF64(format!(
                "f64a payload is {} bytes, not a multiple of 8",
                bytes.len()
            )));
        }
        let data: Vec<f64> = bytes.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect();
        match shape_product(shape) {
            Some(n) if n == data.len() => Ok(Self { shape: shape.to_vec(), data }),
            _ => {
                Err(CanonError::BadF64(format!("shape {:?} does not describe {} values", shape, data.len())))
            }
        }
    }
}

impl Serialize for F64Array {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        F64ArrayWire { f64a: self.payload_base64(), shape: self.shape.clone() }.serialize(s)
    }
}

impl<'de> Deserialize<'de> for F64Array {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let w = F64ArrayWire::deserialize(d)?;
        F64Array::from_wire(&w.f64a, &w.shape).map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn hex_round_trip_exact_bits() {
        for x in [0.0, -0.0, 1.0, -1.5, f64::INFINITY, f64::NEG_INFINITY, f64::MIN_POSITIVE, 1e300, 0.1] {
            let h = f64_to_hex(x);
            assert_eq!(h.len(), 16);
            assert_eq!(f64_from_hex(&h).unwrap().to_bits(), x.to_bits());
        }
        assert_eq!(f64_to_hex(1.0), "3ff0000000000000");
        assert_eq!(f64_to_hex(f64::INFINITY), "7ff0000000000000");
        let nan = f64_from_hex("7ff8000000000001").unwrap();
        assert!(nan.is_nan());
        assert_eq!(f64_to_hex(nan), "7ff8000000000001");
    }

    #[test]
    fn hex_rejects_bad_text() {
        assert!(f64_from_hex("3FF0000000000000").is_err());
        assert!(f64_from_hex("3ff000000000000").is_err());
        assert!(f64_from_hex("3ff00000000000000").is_err());
        assert!(f64_from_hex("zff0000000000000").is_err());
    }

    #[test]
    fn f64hex_serde() {
        let v = serde_json::to_value(F64Hex(0.5)).unwrap();
        assert_eq!(v, json!({"f64": "3fe0000000000000"}));
        let back: F64Hex = serde_json::from_value(v).unwrap();
        assert_eq!(back, F64Hex(0.5));
        assert!(serde_json::from_value::<F64Hex>(json!({"f64": "3fe0000000000000", "x": 1})).is_err());
        assert!(serde_json::from_value::<F64Hex>(json!(0.5)).is_err());
    }

    #[test]
    fn f64array_serde_and_shape() {
        // The A.3 ticks example: [13.375, 300.4] px.
        let a = F64Array::from_slice(&[13.375, 300.4], &[2]);
        let v = serde_json::to_value(&a).unwrap();
        assert_eq!(v, json!({"f64a": "AAAAAADAKkBmZmZmZsZyQA==", "shape": [2]}));
        let back: F64Array = serde_json::from_value(v).unwrap();
        assert_eq!(back, a);
        let bad = json!({"f64a": "AAAAAADAKkBmZmZmZsZyQA==", "shape": [3]});
        assert!(serde_json::from_value::<F64Array>(bad).is_err());
        let two_d = F64Array::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let v = serde_json::to_value(&two_d).unwrap();
        assert_eq!(serde_json::from_value::<F64Array>(v).unwrap(), two_d);
        let empty = F64Array::from_slice(&[], &[0]);
        assert!(empty.is_empty());
        let v = serde_json::to_value(&empty).unwrap();
        assert_eq!(v, json!({"f64a": "", "shape": [0]}));
        assert_eq!(serde_json::from_value::<F64Array>(v).unwrap(), empty);
    }

    #[test]
    fn floatify_replaces_only_non_integers() {
        let v = json!({"a": 1, "b": 15.0, "c": [2.5, 3, {"d": -0.0}], "e": "1.5", "f": null});
        let got = floatify(v);
        assert_eq!(
            got,
            json!({"a": 1, "b": {"f64": "402e000000000000"}, "c": [{"f64": "4004000000000000"}, 3,
                   {"d": {"f64": "8000000000000000"}}], "e": "1.5", "f": null})
        );
        assert!(crate::jcs::canon(&got).is_ok());
    }
}
