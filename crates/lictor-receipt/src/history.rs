// SPDX-License-Identifier: MIT
//! OWNER: WP-4. Stub written by WP-0; replace the bodies, keep the signatures.
//! sbx run-memory lineage -> .lictor/history.jsonl, capped at 200.

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeRecord {
    pub ts: String,
    pub run_id: String,
    pub arm_id: String,
    pub seed: u64,
    pub success: bool,
    pub first_trip_reason: Option<String>,
    pub trips: Vec<String>,
    pub stopped: bool,
    pub escalated: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LoopSignals {
    pub tail_streak: Option<(String, u32)>,
    pub identical_run: u32,
    pub last_n: u32,
}

pub fn record_episode(_dir: &std::path::Path, _r: &EpisodeRecord) -> std::io::Result<()> {
    todo!("WP-4")
}

pub fn summarize(_dir: &std::path::Path, _n: usize) -> std::io::Result<LoopSignals> {
    todo!("WP-4")
}
