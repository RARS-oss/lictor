// SPDX-License-Identifier: MIT
//! RFC 8785 JSON Canonicalization Scheme restricted to the float-free profile `jcs-floatfree/v1`.
//!
//! The profile (docs/receipt-schema.md): object keys sorted by their UTF-16 code-unit sequence, no whitespace,
//! minimal string escaping (`\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`, other U+0000..U+001F as `\u00xx` with
//! lowercase hex, everything else verbatim UTF-8), integers rendered as plain digits with |v| <= 2^53-1, and NO
//! non-integer numbers at all: a real is carried as `{"f64":"<16hex>"}` (see `f64enc`). Because every key of a
//! lictor artefact is printable ASCII (checked by `check_keys`; the field names themselves are `[a-z0-9_]+`, the
//! content-addressed maps carry repo-relative paths, `lictor:bin` and environment-variable names), Python's
//! `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` -- which sorts by code point --
//! produces the same bytes: for ASCII, code point order IS UTF-16 code-unit order.

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CanonError {
    #[error("float in canonical body at {0}")]
    FloatInBody(String),
    #[error("integer out of +-2^53-1 at {0}")]
    IntegerOutOfRange(String),
    /// The frozen text says `[a-z0-9_]+`; the frozen receipt binds keys such as `lictor:bin`, `envelopes/pusht.toml`
    /// and `OMP_NUM_THREADS`, so the enforced class is the one the rule exists for: printable ASCII (`[!-~]+`).
    #[error("key not printable ascii [!-~]+: {0}")]
    BadKey(String),
    #[error("bad f64 encoding: {0}")]
    BadF64(String),
    #[error("serialize: {0}")]
    Serialize(String),
}

pub const CANONICAL_ID: &str = "jcs-floatfree/v1";

/// The largest integer magnitude the profile accepts (2^53 - 1: exactly representable as an IEEE-754 double,
/// so a JavaScript or Python reader that goes through a double cannot silently round it).
pub const MAX_SAFE_INTEGER: u64 = (1u64 << 53) - 1;

/// RFC 8785: keys sorted by UTF-16 code units, no whitespace, minimal escaping; integers only (|v| <= 2^53-1).
pub fn canon(v: &Value) -> Result<Vec<u8>, CanonError> {
    let mut out = Vec::with_capacity(256);
    let mut path = String::from("$");
    write_value(v, &mut out, &mut path)?;
    Ok(out)
}

/// Every object key (recursively) is non-empty printable ASCII (`^[!-~]+$`: no space, no control character, nothing
/// above U+007E); the first offender is reported with its path. This is exactly the class for which Python's
/// code-point key sort equals JCS UTF-16 code-unit sort, so `adapters/verify_receipt.py` re-checks it rather than
/// assuming it.
pub fn check_keys(v: &Value) -> Result<(), CanonError> {
    let mut path = String::from("$");
    walk_keys(v, &mut path)
}

fn key_ok(k: &str) -> bool {
    !k.is_empty() && k.bytes().all(|b| (0x21..=0x7e).contains(&b))
}

fn walk_keys(v: &Value, path: &mut String) -> Result<(), CanonError> {
    match v {
        Value::Object(m) => {
            for (k, child) in m {
                let n = path.len();
                path.push('.');
                path.push_str(k);
                if !key_ok(k) {
                    return Err(CanonError::BadKey(path.clone()));
                }
                walk_keys(child, path)?;
                path.truncate(n);
            }
            Ok(())
        }
        Value::Array(a) => {
            for (i, child) in a.iter().enumerate() {
                let n = path.len();
                path.push('[');
                path.push_str(&i.to_string());
                path.push(']');
                walk_keys(child, path)?;
                path.truncate(n);
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn write_value(v: &Value, out: &mut Vec<u8>, path: &mut String) -> Result<(), CanonError> {
    match v {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(true) => out.extend_from_slice(b"true"),
        Value::Bool(false) => out.extend_from_slice(b"false"),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i.unsigned_abs() > MAX_SAFE_INTEGER {
                    return Err(CanonError::IntegerOutOfRange(path.clone()));
                }
                out.extend_from_slice(i.to_string().as_bytes());
            } else if let Some(u) = n.as_u64() {
                if u > MAX_SAFE_INTEGER {
                    return Err(CanonError::IntegerOutOfRange(path.clone()));
                }
                out.extend_from_slice(u.to_string().as_bytes());
            } else {
                // Any f64-typed number (even an integral one such as 1.0) is a float in the JSON text.
                return Err(CanonError::FloatInBody(path.clone()));
            }
        }
        Value::String(s) => write_string(s, out),
        Value::Array(a) => {
            out.push(b'[');
            for (i, child) in a.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                let n = path.len();
                path.push('[');
                path.push_str(&i.to_string());
                path.push(']');
                write_value(child, out, path)?;
                path.truncate(n);
            }
            out.push(b']');
        }
        Value::Object(m) => write_object(m, out, path)?,
    }
    Ok(())
}

fn write_object(m: &Map<String, Value>, out: &mut Vec<u8>, path: &mut String) -> Result<(), CanonError> {
    // RFC 8785 sec 3.2.3: sort by the UTF-16 code-unit sequence of the (unescaped) key.
    let mut keys: Vec<(Vec<u16>, &String)> = m.keys().map(|k| (k.encode_utf16().collect(), k)).collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0));
    out.push(b'{');
    for (i, (_, k)) in keys.iter().enumerate() {
        if i > 0 {
            out.push(b',');
        }
        write_string(k, out);
        out.push(b':');
        let n = path.len();
        path.push('.');
        path.push_str(k);
        // The key is present by construction (it came from this map).
        if let Some(child) = m.get(k.as_str()) {
            write_value(child, out, path)?;
        }
        path.truncate(n);
    }
    out.push(b'}');
    Ok(())
}

/// RFC 8785 sec 3.2.2.2: `"` and `\` escaped, U+0000..U+001F as the short forms or `\u00xx` (lowercase),
/// everything else emitted as UTF-8 (including U+007F and `/`).
fn write_string(s: &str, out: &mut Vec<u8>) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    out.push(b'"');
    for &b in s.as_bytes() {
        match b {
            b'"' => out.extend_from_slice(b"\\\""),
            b'\\' => out.extend_from_slice(b"\\\\"),
            0x08 => out.extend_from_slice(b"\\b"),
            0x0c => out.extend_from_slice(b"\\f"),
            b'\n' => out.extend_from_slice(b"\\n"),
            b'\r' => out.extend_from_slice(b"\\r"),
            b'\t' => out.extend_from_slice(b"\\t"),
            0x00..=0x1f => {
                out.extend_from_slice(b"\\u00");
                out.push(HEX[(b >> 4) as usize]);
                out.push(HEX[(b & 0x0f) as usize]);
            }
            _ => out.push(b),
        }
    }
    out.push(b'"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalars_and_nesting() {
        let v = json!({"b": [1, -2, true, null, "x"], "a": {"z": 0, "y": ""}});
        assert_eq!(canon(&v).unwrap(), br#"{"a":{"y":"","z":0},"b":[1,-2,true,null,"x"]}"#.to_vec());
    }

    #[test]
    fn escaping_is_minimal_and_lowercase() {
        let v = json!({"k": "q\"b\\s\u{8}\u{c}\n\r\t\u{1f}\u{7f}/\u{e9}"});
        let got = String::from_utf8(canon(&v).unwrap()).unwrap();
        assert_eq!(got, "{\"k\":\"q\\\"b\\\\s\\b\\f\\n\\r\\t\\u001f\u{7f}/\u{e9}\"}");
    }

    #[test]
    fn utf16_order_puts_astral_before_bmp_high() {
        // U+1F600 (surrogates d83d de00) sorts BEFORE U+FB33 under UTF-16, after it under code points.
        let v = json!({"\u{fb33}": 1, "\u{1f600}": 2, "\u{20ac}": 3});
        let got = String::from_utf8(canon(&v).unwrap()).unwrap();
        assert_eq!(got, "{\"\u{20ac}\":3,\"\u{1f600}\":2,\"\u{fb33}\":1}");
    }

    #[test]
    fn floats_and_big_integers_are_rejected_with_paths() {
        let v = json!({"a": [1, {"b": 1.5}]});
        assert_eq!(canon(&v), Err(CanonError::FloatInBody("$.a[1].b".into())));
        let v = json!({"a": 1.0});
        assert_eq!(canon(&v), Err(CanonError::FloatInBody("$.a".into())));
        let v = json!({"n": 9007199254740993u64});
        assert_eq!(canon(&v), Err(CanonError::IntegerOutOfRange("$.n".into())));
        let v = json!({"n": -9007199254740992i64});
        assert_eq!(canon(&v), Err(CanonError::IntegerOutOfRange("$.n".into())));
        let v = json!({"n": 9007199254740991u64, "m": -9007199254740991i64});
        assert_eq!(canon(&v).unwrap(), br#"{"m":-9007199254740991,"n":9007199254740991}"#.to_vec());
    }

    #[test]
    fn check_keys_reports_the_first_bad_key() {
        assert!(check_keys(&json!({"ok_1": {"a2": [ {"z_": 0} ]}})).is_ok());
        assert!(
            check_keys(&json!({"lictor:bin": 0, "envelopes/pusht.toml": 1, "OMP_NUM_THREADS": "1"})).is_ok()
        );
        assert_eq!(check_keys(&json!({"ok": {"a b": 0}})), Err(CanonError::BadKey("$.ok.a b".into())));
        assert_eq!(check_keys(&json!({"": 0})), Err(CanonError::BadKey("$.".into())));
        assert_eq!(check_keys(&json!([{"a\tb": 0}])), Err(CanonError::BadKey("$[0].a\tb".into())));
        assert_eq!(check_keys(&json!({"caf\u{e9}": 0})), Err(CanonError::BadKey("$.caf\u{e9}".into())));
        assert_eq!(check_keys(&json!({"x\u{7f}": 0})), Err(CanonError::BadKey("$.x\u{7f}".into())));
    }
}
