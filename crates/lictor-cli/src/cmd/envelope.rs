// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; the `Args`/`Sub` flags are interface (CLI SURFACE), replace `run`.
//! `lictor envelope init|check|digest|show|fit`. `init --profile pusht` emits the base TOML embedded below.

use std::path::PathBuf;

use clap::Subcommand;

/// The frozen base envelope, embedded so `envelope init` works without a checkout.
pub const PUSHT_BASE_TOML: &str = include_str!("../../../../envelopes/pusht.base.toml");

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Sub,
}

#[derive(Subcommand)]
pub enum Sub {
    /// Write the base envelope for a profile
    Init {
        /// Embodiment profile
        #[arg(long, value_parser = ["pusht"], default_value = "pusht")]
        profile: String,
        /// Output TOML
        #[arg(short, long, value_name = "F.toml")]
        out: PathBuf,
        /// Operator pubkey (hex64) allowed to sign an AckToken; repeatable
        #[arg(long, value_name = "hex")]
        operator: Vec<String>,
    },
    /// Validate + digest + embodiment digest
    Check {
        #[arg(value_name = "F.toml")]
        file: PathBuf,
    },
    /// Print the envelope digest (embodiment digest on stderr)
    Digest {
        #[arg(value_name = "F.toml")]
        file: PathBuf,
    },
    /// Print the envelope
    Show {
        #[arg(value_name = "F.toml")]
        file: PathBuf,
    },
    /// Fit Tier-0 limits from a recorded calibration arm
    Fit {
        /// Run directory (results/<run>)
        #[arg(long, value_name = "DIR")]
        run: PathBuf,
        /// Calibration arm id
        #[arg(long, value_name = "ARM")]
        arm: String,
        /// Base envelope TOML
        #[arg(long, value_name = "F.toml")]
        base: PathBuf,
        /// Empirical quantile of |v|, |a|, |j|, reach over calibration successes
        #[arg(long, value_name = "Q", default_value_t = 0.999)]
        quantile: f64,
        /// Multiplicative slack on the quantile
        #[arg(long, value_name = "S", default_value_t = 1.25)]
        slack: f64,
        /// Operator pubkey (hex64); writes the oracle envelope (identical embodiment); repeatable
        #[arg(long, value_name = "hex")]
        operator: Vec<String>,
        /// Output TOML
        #[arg(short, long, value_name = "F.toml")]
        out: PathBuf,
        /// Markdown fit report
        #[arg(long, value_name = "F.md")]
        report: Option<PathBuf>,
    },
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    match a.cmd {
        Sub::Init { profile: _profile, out: _out, operator: _operator } => {
            let _base = PUSHT_BASE_TOML;
            todo!("WP-5")
        }
        Sub::Check { file: _file } => todo!("WP-5"),
        Sub::Digest { file: _file } => todo!("WP-5"),
        Sub::Show { file: _file } => todo!("WP-5"),
        Sub::Fit {
            run: _run,
            arm: _arm,
            base: _base,
            quantile: _quantile,
            slack: _slack,
            operator: _operator,
            out: _out,
            report: _report,
        } => todo!("WP-5"),
    }
}
