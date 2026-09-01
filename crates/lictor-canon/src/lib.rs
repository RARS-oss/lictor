// SPDX-License-Identifier: MIT
#![forbid(unsafe_code)]
//! lictor-canon: RFC 8785 JCS over the float-free profile `jcs-floatfree/v1`, the `F64Hex` / `F64Array`
//! encodings, `floatify` for TOML-sourced config trees, and sha256 digests. See docs/ARCHITECTURE.md sec 8.

pub mod digest;
pub mod f64enc;
pub mod jcs;

pub use digest::{canon_of, digest_of, sha256_hex, sha256_jcs};
pub use f64enc::{f64_from_hex, f64_to_hex, floatify, F64Array, F64Hex};
pub use jcs::{canon, check_keys, CanonError, CANONICAL_ID};
