// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor crash-receipt`: host-side accounting for a serve child that died before episode_end: writes a signed
//! receipt with ended_by="fuse_crash", success=false, terminal_state=fault, fuse_ok=false and appends the ledger. Exit 0.

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// calibration.json
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// observe | enforce
    #[arg(long, value_parser = ["observe", "enforce"])]
    pub mode: String,
    #[arg(long, value_name = "S")]
    pub run_id: String,
    #[arg(long, value_name = "S")]
    pub arm_id: String,
    #[arg(long, value_name = "N")]
    pub episode_index: u32,
    #[arg(long, value_name = "N")]
    pub seed: u64,
    #[arg(long, value_name = "S")]
    pub seed_pool: String,
    #[arg(long, value_name = "hex")]
    pub init_state_digest: String,
    /// BudgetBinding as JSON
    #[arg(long, value_name = "json")]
    pub budget: String,
    /// Ed25519 seed file (64 hex)
    #[arg(long, value_name = "F.hex")]
    pub key: Option<PathBuf>,
    /// Output directory (the same --out the serve child used)
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Extra note recorded in fuse_notes
    #[arg(long, value_name = "S")]
    pub note: Option<String>,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        envelope: _envelope,
        calibration: _calibration,
        mode: _mode,
        run_id: _run_id,
        arm_id: _arm_id,
        episode_index: _episode_index,
        seed: _seed,
        seed_pool: _seed_pool,
        init_state_digest: _init_state_digest,
        budget: _budget,
        key: _key,
        out: _out,
        note: _note,
    } = a;
    todo!("WP-6")
}
