// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; the `Args` flags are interface (CLI SURFACE), replace `run`.
//! `lictor serve`: NDJSON server on stdin/stdout (lictor-wire/v1). Exit 0 on bye, 3 on internal error; exit 2 at
//! startup when the calibration's embodiment_digest != the envelope's. Without --key an ephemeral key is generated and
//! every receipt carries fuse_ok=false + note "ephemeral signing key" (lictor curve refuses such arms).

use std::path::PathBuf;

#[derive(clap::Args)]
pub struct Args {
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// calibration.json (Tier 1); refused when its embodiment_digest differs from the envelope's
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// observe | enforce
    #[arg(long, value_parser = ["observe", "enforce"], default_value = "observe")]
    pub mode: String,
    /// Ed25519 seed file (64 hex); default $LICTOR_KEYS/key.hex if it exists, else an ephemeral key
    #[arg(long, value_name = "F.hex")]
    pub key: Option<PathBuf>,
    /// Output directory for receipts, ticks, timing and the ledger
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// Write every request line verbatim to this trace file
    #[arg(long, value_name = "F.ndjson")]
    pub trace: Option<PathBuf>,
    /// Which tick events to embed in the receipt
    #[arg(long, value_parser = ["tail32", "all", "none"], default_value = "tail32")]
    pub ticks: String,
    /// Comma-separated Tier-0 trip names to arm (overrides the envelope's tier0_enabled)
    #[arg(long, value_name = "csv")]
    pub tier0: Option<String>,
    /// Disarm Tier 1 even when a calibration is given
    #[arg(long)]
    pub no_tier1: bool,
    /// hold (latch Fault) | abort (exit 3 on the first fatal error; debugging aid only)
    #[arg(long, value_parser = ["hold", "abort"], default_value = "hold")]
    pub on_fault: String,
    /// Latency label recorded in receipts (default: default_latency_label())
    #[arg(long, value_name = "S")]
    pub latency_label: Option<String>,
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        envelope: _envelope,
        calibration: _calibration,
        mode: _mode,
        key: _key,
        out: _out,
        trace: _trace,
        ticks: _ticks,
        tier0: _tier0,
        no_tier1: _no_tier1,
        on_fault: _on_fault,
        latency_label: _latency_label,
    } = a;
    todo!("WP-6")
}
