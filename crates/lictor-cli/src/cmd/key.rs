// SPDX-License-Identifier: MIT
//! OWNER: WP-7. Stub written by WP-0; the `Args`/`Sub` flags are interface (CLI SURFACE), replace `run`.
//! `lictor key init|pub`: default path $LICTOR_KEYS/key.hex (or $HOME/.lictor/key.hex); refuses to overwrite without
//! --force; refuses a path under /mnt/[a-z]/ without --i-know; prints "pubkey <hex>", "stored <path> (mode 0600)" and
//! one custody line.

use std::path::PathBuf;

use clap::Subcommand;

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Sub,
}

#[derive(Subcommand)]
pub enum Sub {
    /// Generate a new Ed25519 seed file
    Init {
        /// Output path (default: <key dir>/key.hex, or operator.hex for --role operator)
        #[arg(short, long, value_name = "F.hex")]
        out: Option<PathBuf>,
        /// signer | operator
        #[arg(long, value_parser = ["signer", "operator"], default_value = "signer")]
        role: String,
        /// Overwrite an existing file
        #[arg(long)]
        force: bool,
        /// Allow a path under /mnt/[a-z]/ (DrvFs does not enforce 0600)
        #[arg(long)]
        i_know: bool,
    },
    /// Print the pubkey of a seed file
    Pub {
        /// Seed file (default: <key dir>/key.hex)
        #[arg(long, value_name = "F.hex")]
        key: Option<PathBuf>,
    },
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    match a.cmd {
        Sub::Init { out: _out, role: _role, force: _force, i_know: _i_know } => todo!("WP-7"),
        Sub::Pub { key: _key } => todo!("WP-7"),
    }
}
