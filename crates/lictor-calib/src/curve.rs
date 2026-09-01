// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! Layer-B curve: closed-loop metrics from receipts ONLY.

use lictor_receipt::CurveReceiptBody;

pub struct CurveOpts {
    pub eps_prog: f64,
    pub latency_control: Option<String>,
    pub calib_seed_override: Option<(u64, u64)>,
    pub allow_small: bool,
    pub partial: bool,
    pub compare_run: Option<std::path::PathBuf>,
    /// 10_000
    pub n_boot: u32,
    /// 20260830
    pub boot_seed: u64,
}

/// Refuses (Err, exit 1 in the CLI) when: the ledger is broken; the arm's receipts carry >1 pubkey or an ephemeral-key note; the arm's
/// budget.tier1_armed / alpha disagree with run.json; the arm's episode-index set != the baseline's or != the declared pool (unless
/// `partial`); n < 100 (unless `allow_small`). Missing indices count as success = false, stopped = true (terminate_fail accounting).
pub fn curve(
    _run_dir: &std::path::Path,
    _arm: &str,
    _baseline: &str,
    _opts: &CurveOpts,
) -> anyhow::Result<CurveReceiptBody> {
    todo!("WP-5")
}
