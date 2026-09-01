// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor verify`: prints the aligned block of the CLI SURFACE; exit 0 intact AND fuse held, 1 otherwise.
//! Scripts use `--json` and test the `intact` field (an honest Observe receipt exits 1).

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Signed receipt JSON
    #[arg(value_name = "RECEIPT.json")]
    pub receipt: PathBuf,
    /// Full ticks file: recompute the verdict chain and compare with the signed head
    #[arg(long, value_name = "F.jsonl")]
    pub ticks: Option<PathBuf>,
    /// Timing file: chain integrity only
    #[arg(long, value_name = "F.jsonl")]
    pub timing: Option<PathBuf>,
    /// Ledger: chain integrity and that this receipt's digest appears
    #[arg(long, value_name = "F.jsonl")]
    pub ledger: Option<PathBuf>,
    /// Curve receipt: signature and ledger_head match
    #[arg(long, value_name = "C.json")]
    pub curve: Option<PathBuf>,
    /// calibration.json: budget alpha/gate/kn must equal the file's
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// Expected signing pubkey (hex64); a mismatch makes the receipt not intact
    #[arg(long, value_name = "hex")]
    pub pubkey: Option<String>,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        receipt: _receipt,
        ticks: _ticks,
        timing: _timing,
        ledger: _ledger,
        curve: _curve,
        calibration: _calibration,
        pubkey: _pubkey,
    } = a;
    todo!("WP-7")
}
