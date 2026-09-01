// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor sweep`: Layer-A points, one line per (alpha, detector); with --calibration-dir, points whose
//! (alpha, gate, kn, method) match a calibration.<alpha>.json reuse its tau verbatim.

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Run directory (results/<run>)
    #[arg(long, value_name = "DIR")]
    pub run: PathBuf,
    /// Calibration arm id
    #[arg(long, value_name = "ARM")]
    pub calib_arm: String,
    /// Evaluation arm id
    #[arg(long, value_name = "ARM")]
    pub eval_arm: String,
    /// Comma-separated alphas as num/den
    #[arg(long, value_name = "csv num/den")]
    pub alphas: String,
    /// Directory of calibration.<alpha>.json artefacts to reuse verbatim
    #[arg(long, value_name = "DIR")]
    pub calibration_dir: Option<PathBuf>,
    /// Comma-separated detector names
    #[arg(long, value_name = "csv", default_value = "t0,t1_tce,t1_stall,t1_full,t01,t01_and")]
    pub detectors: String,
    /// static | binned
    #[arg(long, value_parser = ["static", "binned"], default_value = "binned")]
    pub method: String,
    /// Point-of-no-return progress epsilon
    #[arg(long, value_name = "EPS", default_value_t = 0.02)]
    pub eps_prog: f64,
    /// Output sweep.jsonl
    #[arg(short, long, value_name = "F.jsonl")]
    pub out: PathBuf,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        run: _run,
        calib_arm: _calib_arm,
        eval_arm: _eval_arm,
        alphas: _alphas,
        calibration_dir: _calibration_dir,
        detectors: _detectors,
        method: _method,
        eps_prog: _eps_prog,
        out: _out,
    } = a;
    todo!("WP-5")
}
