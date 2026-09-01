// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! One Trace per episode, built from the ticks file + receipt (labels) + the request trace (coverage), pool-filtered.
//! Trace lines are parsed with lictor_runtime::trace::TraceReader (the ONE wire parser); never a second serde_json::Value parser.

use lictor_core::NFEAT;

pub struct Trace {
    pub arm_id: String,
    pub seed: u64,
    pub episode_index: u32,
    pub success: bool,
    pub steps: u32,
    pub coverage: Vec<f64>,
    pub f: Vec<[f64; NFEAT]>,
    pub valid: Vec<u32>,
    pub first_stop_tick: Option<u32>,
    pub init_state_digest: String,
}

/// verifies each receipt + ticks chain first; refuses broken ones
pub fn load_traces(_run_dir: &std::path::Path, _arm_id: &str) -> anyhow::Result<Vec<Trace>> {
    todo!("WP-5")
}
