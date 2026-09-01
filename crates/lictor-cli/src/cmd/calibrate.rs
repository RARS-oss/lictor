// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor calibrate`: writes <DIR>/calibration.<alpha>.json per alpha (float-free, self-digested). Refuses eval-pool
//! seeds, and a horizon_ticks that differs from envelope.embodiment.horizon_ticks or from run.env.max_episode_steps.

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Run directory (results/<run>)
    #[arg(long, value_name = "DIR")]
    pub run: PathBuf,
    /// Calibration arm id
    #[arg(long, value_name = "ARM")]
    pub arm: String,
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// Comma-separated alphas as num/den (e.g. 5/100,10/100)
    #[arg(long, value_name = "num/den[,...]")]
    pub alpha: String,
    /// static | binned
    #[arg(long, value_parser = ["static", "binned"], default_value = "binned")]
    pub method: String,
    /// Comma-separated gate terms (default: the envelope's gate)
    #[arg(long, value_name = "csv terms")]
    pub gate: Option<String>,
    /// K-of-N as k,n
    #[arg(long, value_name = "K,N", default_value = "3,5")]
    pub kn: String,
    /// Holdout fraction as num/den
    #[arg(long, value_name = "num/den", default_value = "3/10")]
    pub holdout: String,
    /// 2: center/scale and tau on the fit subset (approximate); 3: disjoint tau subset (exact at K = 1)
    #[arg(long, value_parser = clap::value_parser!(u8).range(2..=3), default_value_t = 2)]
    pub split: u8,
    /// Cap on the number of per-episode scores tau is taken over
    #[arg(long, value_name = "N")]
    pub n_calib: Option<u32>,
    /// Watching band below tau (z units)
    #[arg(long, value_name = "Z", default_value_t = 0.5)]
    pub warn_margin: f64,
    /// Output directory
    #[arg(short, long, value_name = "DIR")]
    pub out: PathBuf,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        run: _run,
        arm: _arm,
        envelope: _envelope,
        alpha: _alpha,
        method: _method,
        gate: _gate,
        kn: _kn,
        holdout: _holdout,
        split: _split,
        n_calib: _n_calib,
        warn_margin: _warn_margin,
        out: _out,
    } = a;
    todo!("WP-5")
}
