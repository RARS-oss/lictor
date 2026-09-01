// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; replace `run`.
//! `lictor selftest`: one line per check (`  [ok] ...` / `  [FAIL] ...` / `  [skip] ...`), final
//! `SELFTEST PASS` | `SELFTEST FAIL`, exit 0/1. The brake and tier-1 fixtures below are `[]` placeholders until WP-1 /
//! WP-2 fill them; selftest reports `[skip] <name> fixture empty` for an empty array.

/// crates/lictor-detect/tests/fixtures/brake/cases.json (WP-1)
pub const BRAKE_CASES: &str = include_str!("../../../lictor-detect/tests/fixtures/brake/cases.json");
/// crates/lictor-detect/tests/fixtures/tier1/cases.json (WP-2)
pub const TIER1_CASES: &str = include_str!("../../../lictor-detect/tests/fixtures/tier1/cases.json");

#[derive(clap::Args)]
pub struct Args {}

pub fn run(_a: Args, _json: bool) -> anyhow::Result<i32> {
    let _fixtures = (BRAKE_CASES, TIER1_CASES);
    todo!("WP-7")
}
