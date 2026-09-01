// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; replace `run`.
//! `lictor version` -> "lictor 0.1.0 (<git>) sha256=<binary sha>". The git sha and build time come from build.rs;
//! the binary's sha256 is computed at run time from `std::env::current_exe()`.

pub const LICTOR_GIT: &str = env!("LICTOR_GIT");
pub const LICTOR_BUILD_UTC: &str = env!("LICTOR_BUILD_UTC");

#[derive(clap::Args)]
pub struct Args {}

pub fn run(_a: Args, _json: bool) -> anyhow::Result<i32> {
    let _build = (lictor_core::LICTOR_VERSION, LICTOR_GIT, LICTOR_BUILD_UTC);
    todo!("WP-7")
}
