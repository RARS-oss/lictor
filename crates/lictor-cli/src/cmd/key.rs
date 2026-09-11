// SPDX-License-Identifier: MIT
//! `lictor key init|pub`: default path $LICTOR_KEYS/key.hex (or $HOME/.lictor/key.hex); refuses to overwrite without
//! --force; refuses a path under /mnt/[a-z]/ without --i-know; prints "pubkey <hex>", "stored <path> (mode 0600)" and
//! one custody line.
//!
//! The seed never leaves the key file: `init` prints only the pubkey and the path; `pub` derives the pubkey from
//! the seed and prints it. A refusal (existing file, DrvFs path, unreadable seed) is a usage error (exit 2) with
//! the reason on stderr; nothing is written in that case.

use std::path::PathBuf;

use clap::Subcommand;
use lictor_receipt::{default_key_dir, keygen, load_seed, pubkey_hex, save_seed};

use crate::render::{json_out, print_lines, sanitize};

/// The custody line printed by `key init` (one line, never reworded by callers).
pub const CUSTODY: &str =
    "custody: this file signs every receipt you produce; back it up off the repo and never commit it";

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

/// `<key dir>/key.hex` for a signer, `<key dir>/operator.hex` for an operator.
pub fn default_path(role: &str) -> PathBuf {
    let name = if role == "operator" { "operator.hex" } else { "key.hex" };
    default_key_dir().join(name)
}

fn display(p: &std::path::Path) -> String {
    sanitize(&p.to_string_lossy().replace('\\', "/"))
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct KeyOut {
    pub pubkey: String,
    pub path: String,
    pub role: String,
    pub mode: String,
}

#[cfg(unix)]
const MODE_TEXT: &str = "mode 0600";
#[cfg(not(unix))]
const MODE_TEXT: &str = "mode 0600 not enforced on this platform";

pub fn run(a: Args, json: bool) -> anyhow::Result<i32> {
    match a.cmd {
        Sub::Init { out, role, force, i_know } => {
            let path = out.unwrap_or_else(|| default_path(&role));
            let seed = keygen();
            if let Err(e) = save_seed(&path, &seed, force, i_know) {
                eprintln!("lictor key init: refused: {e}");
                return Ok(2);
            }
            let out =
                KeyOut { pubkey: pubkey_hex(&seed), path: display(&path), role, mode: MODE_TEXT.to_string() };
            if json {
                json_out(&out)?;
            } else {
                print_lines(&[
                    format!("pubkey {}", out.pubkey),
                    format!("stored {} ({})", out.path, out.mode),
                    CUSTODY.to_string(),
                ]);
            }
            Ok(0)
        }
        Sub::Pub { key } => {
            let path = key.unwrap_or_else(|| default_path("signer"));
            let seed = match load_seed(&path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("lictor key pub: {e}");
                    return Ok(2);
                }
            };
            let out = KeyOut {
                pubkey: pubkey_hex(&seed),
                path: display(&path),
                role: String::new(),
                mode: String::new(),
            };
            if json {
                json_out(&out)?;
            } else {
                print_lines(&[format!("pubkey {}", out.pubkey)]);
            }
            Ok(0)
        }
    }
}
