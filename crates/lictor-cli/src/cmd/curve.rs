// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor curve`: recomputes every metric FROM RECEIPTS; refuses on a broken ledger, mixed/ephemeral keys, a
//! tier1/alpha disagreement with run.json, an index set != the baseline's or the declared pool (unless --partial),
//! and n < 100 (unless --allow-small). Writes <arm>.json (signed) + summary.csv.

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Run directory (results/<run>)
    #[arg(long, value_name = "DIR")]
    pub run: PathBuf,
    /// Baseline arm id (obs-d0)
    #[arg(long, value_name = "ARM")]
    pub baseline: String,
    /// Comma-separated arm ids (default: every arm in the run except the baseline)
    #[arg(long, value_name = "a,b,c")]
    pub arms: Option<String>,
    /// Latency-control arm (default: obs-d<d> derived from the arm's delay_steps when present)
    #[arg(long, value_name = "ARM")]
    pub latency_control: Option<String>,
    /// Point-of-no-return progress epsilon
    #[arg(long, value_name = "EPS", default_value_t = 0.02)]
    pub eps_prog: f64,
    /// Calibration seed range override, lo-hi inclusive
    #[arg(long, value_name = "LO-HI")]
    pub calib_seeds: Option<String>,
    /// Accept n < 100 (marks the point small_n; a pilot, never a headline)
    #[arg(long)]
    pub allow_small: bool,
    /// Accept an index set that differs from the baseline's or the declared pool (missing count as failures)
    #[arg(long)]
    pub partial: bool,
    /// Another run to cross-check verdict_chain_head per (arm, seed)
    #[arg(long, value_name = "DIR")]
    pub compare_run: Option<PathBuf>,
    /// Ed25519 seed file to sign the curve receipt
    #[arg(long, value_name = "F.hex")]
    pub key: Option<PathBuf>,
    /// Output directory
    #[arg(short, long, value_name = "DIR")]
    pub out: PathBuf,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        run: _run,
        baseline: _baseline,
        arms: _arms,
        latency_control: _latency_control,
        eps_prog: _eps_prog,
        calib_seeds: _calib_seeds,
        allow_small: _allow_small,
        partial: _partial,
        compare_run: _compare_run,
        key: _key,
        out: _out,
    } = a;
    todo!("WP-5")
}
