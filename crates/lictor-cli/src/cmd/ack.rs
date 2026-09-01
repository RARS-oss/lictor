// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor ack`: sign an operator acknowledgement for a pending handoff (digest or record.json).

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Handoff digest (hex64) or a HandoffRecord JSON file
    #[arg(long, value_name = "digest|record.json")]
    pub handoff: String,
    /// resume | abort | retune
    #[arg(long, value_parser = ["resume", "abort", "retune"])]
    pub decision: String,
    /// Operator seed file (64 hex)
    #[arg(long, value_name = "operator.hex")]
    pub key: PathBuf,
    /// Free-text note bound into the token
    #[arg(long, value_name = "S")]
    pub note: Option<String>,
    /// Nonce (default: 1 + the largest nonce recorded for this operator)
    #[arg(long, value_name = "N")]
    pub nonce: Option<u64>,
    /// Write the token here (default: stdout)
    #[arg(short, long, value_name = "F.json")]
    pub out: Option<PathBuf>,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args { handoff: _handoff, decision: _decision, key: _key, note: _note, nonce: _nonce, out: _out } = a;
    todo!("WP-7")
}
