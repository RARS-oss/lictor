// SPDX-License-Identifier: MIT
//! OWNER: WP-10. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor bench`: replays a trace through `Staging` + `decide()` n times with `Instant` OUTSIDE the call; prints the
//! frozen lines (the numbers are measured; CI greps the shape) and `allocations   0` from the gated counting
//! allocator. The KiB figure is size_of::<FuseRt>() + size_of::<FuseConfig>() printed at run time.

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Number of decide() calls
    #[arg(long, value_name = "N", default_value_t = 200_000)]
    pub n: u64,
    /// Request trace to replay (default: the embedded bench/fixtures/traces/pusht_000007.ndjson)
    #[arg(long, value_name = "F.ndjson")]
    pub trace: Option<PathBuf>,
    /// t0 | t0t1
    #[arg(long, value_parser = ["t0", "t0t1"], default_value = "t0t1")]
    pub tier: String,
    /// Histogram bucket dump
    #[arg(long, value_name = "F")]
    pub csv: Option<PathBuf>,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args { n: _n, trace: _trace, tier: _tier, csv: _csv } = a;
    // The frozen shape of the measurement: the allocation gate brackets the measured loop (WP-10 fills the loop).
    crate::alloc_count::start();
    let (_allocs, _deallocs) = crate::alloc_count::stop();
    let _since_reset = crate::alloc_count::allocations();
    crate::alloc_count::reset();
    todo!("WP-10")
}
