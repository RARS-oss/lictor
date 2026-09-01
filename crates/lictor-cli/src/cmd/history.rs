// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor history`: summarise .lictor/history.jsonl (tail streaks, identical runs).

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// History directory (default: .lictor)
    #[arg(long, value_name = "DIR")]
    pub dir: Option<PathBuf>,
    /// Number of recent episodes to summarise
    #[arg(long, value_name = "N", default_value_t = 20)]
    pub n: usize,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args { dir: _dir, n: _n } = a;
    todo!("WP-7")
}
