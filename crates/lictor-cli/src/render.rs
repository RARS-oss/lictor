// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0 (deliberately empty: an unused `fn` in a binary crate is a dead-code error).
//! WP-7 puts here the deterministic, length-capped human rendering (sbx render doctrine): `kv(label, value)` aligned to
//! 15 columns, `note(s)`, `glossary(reason: ReasonCode) -> &str` returning `reason_text`, and the control-character /
//! ESC-sequence stripping applied to every host- or operator-supplied string before printing. `--json` short-circuits
//! to serde output everywhere.
