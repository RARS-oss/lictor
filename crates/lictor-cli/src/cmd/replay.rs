// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor replay`: "replays N/N byte-identical  verdict_chain=<hex>"; exit 1 on any divergence.

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Request trace (#meta line + verbatim request lines)
    #[arg(value_name = "TRACE.ndjson")]
    pub trace: PathBuf,
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// calibration.json
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// observe | enforce (default: the trace's #meta mode)
    #[arg(long, value_parser = ["observe", "enforce"])]
    pub mode: Option<String>,
    /// Number of replays through fresh sessions
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub repeat: u32,
    /// Expected verdict chain head (hex64)
    #[arg(long, value_name = "hex64")]
    pub expect: Option<String>,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        trace: _trace,
        envelope: _envelope,
        calibration: _calibration,
        mode: _mode,
        repeat: _repeat,
        expect: _expect,
    } = a;
    todo!("WP-7")
}
