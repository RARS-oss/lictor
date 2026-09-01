// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! The signed curve receipt (`lictor-curve/v1`): per-arm closed-loop metrics recomputed from receipts only.

use lictor_canon::F64Hex;

use crate::sign::{ReceiptError, VerifyReport};

/// A paired difference with its bootstrap CI (10 000 resamples, splitmix64 seed 20260830) and exact McNemar on the same pairs.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeltaCi {
    pub diff: F64Hex,
    pub lo: F64Hex,
    pub hi: F64Hex,
    pub mcnemar_p: F64Hex,
    pub mcnemar_b: u32,
    pub mcnemar_c: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveMetrics {
    pub n: u32,
    pub success_rate: F64Hex,
    pub success_ci: [F64Hex; 2],
    pub averted_rate: F64Hex,
    pub flagged_rate: F64Hex,
    pub flagged_ci: [F64Hex; 2],
    pub false_trip_rate: F64Hex,
    pub false_trip_ci: [F64Hex; 2],
    pub intervention_rate: F64Hex,
    pub intervention_tick_frac: F64Hex,
    pub escalation_rate: F64Hex,
    pub lead_mean: F64Hex,
    pub lead_p50: F64Hex,
    pub lead_p10: F64Hex,
    pub aucpdt: F64Hex,
    pub roc_auc: F64Hex,
    pub bacc: F64Hex,
    pub violations_reached_env: u32,
    /// success(arm) - success(obs-d0), paired by episode index
    pub delta_vs_baseline: DeltaCi,
    /// success(arm) - success(obs-d{d}); None when no latency-control arm exists in the run
    pub delta_vs_latency_control: Option<DeltaCi>,
    /// fraction of ticks with Feat::Tce valid (sync d >= 7 has NO chunk overlap -> 0.0)
    pub tce_valid_frac: F64Hex,
    pub latency_p50_ns: u64,
    pub latency_p99_ns: u64,
    pub latency_max_ns: u64,
    pub latency_label: String,
    pub n_fail_baseline: u32,
    pub n_succ_baseline: u32,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveReceiptBody {
    pub schema: String,
    pub canonical: String,
    pub created_epoch: u64,
    pub run_id: String,
    pub arm_id: String,
    pub baseline_arm_id: String,
    pub latency_control_arm_id: Option<String>,
    pub ledger_head: String,
    pub n_episodes: u32,
    pub ledger_chain_ok: bool,
    /// sha256(canon({"envelope_digest": String, "calibration_digest": Option<String>, "budget": BudgetBinding})) of the arm's
    /// receipts (all identical, else refused)
    pub arm_config_digest: String,
    /// the ONE pubkey every receipt of the arm carries (mixed keys or an ephemeral note -> refused)
    pub receipt_pubkey: String,
    /// sha256 of results/<run>/run.json (the unsigned harness manifest, bound here)
    pub run_json_sha256: String,
    /// declared pool (run.json + calibration seed_pool) vs ledger
    pub n_declared: u32,
    pub n_present: u32,
    pub n_missing: u32,
    pub missing_indices: Vec<u32>,
    /// true only under --partial (index sets differ from the baseline's or from the declared pool)
    pub partial: bool,
    /// true only under --allow-small (n < 100): a pilot point, never a headline
    pub small_n: bool,
    /// episode indices whose verdict_chain_head differs from the same (arm, seed) in --compare-run
    pub cross_run_mismatches: Vec<u32>,
    pub pair_mismatches: Vec<u32>,
    pub seed_overlap_with_calibration: Vec<u64>,
    pub metrics: CurveMetrics,
    pub eps_prog: F64Hex,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedCurve {
    pub body: CurveReceiptBody,
    pub body_digest: String,
    pub pubkey: String,
    pub sig: String,
}

pub fn sign_curve(_body: CurveReceiptBody, _seed: &[u8; 32]) -> Result<SignedCurve, ReceiptError> {
    todo!("WP-4")
}

/// ticks_ok/counts_ok/envelope_digest_ok = true (n/a)
pub fn verify_curve(_sc: &SignedCurve, _expect_pubkey: Option<&str>) -> VerifyReport {
    todo!("WP-4")
}
