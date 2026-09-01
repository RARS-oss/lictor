// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; the `Args`/`Sub` flags are interface (CLI SURFACE), replace `run`.
//! `lictor ledger verify <F.jsonl> | append --ledger <F.jsonl> --receipt <R.json>`.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Sub,
}

#[derive(Subcommand)]
pub enum Sub {
    /// Verify the hash chain and print the summary line
    Verify {
        #[arg(value_name = "F.jsonl")]
        file: PathBuf,
    },
    /// Append a signed receipt to a ledger (single writer)
    Append {
        #[arg(long, value_name = "F.jsonl")]
        ledger: PathBuf,
        #[arg(long, value_name = "R.json")]
        receipt: PathBuf,
    },
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    match a.cmd {
        Sub::Verify { file: _file } => todo!("WP-7"),
        Sub::Append { ledger: _ledger, receipt: _receipt } => todo!("WP-7"),
    }
}
