// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! calibration.json (schema lictor-calibration/v1), float-free; see FILE FORMATS.

use lictor_canon::{F64Array, F64Hex};
use lictor_core::{CalMethod, CalibrationC};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SeedPool {
    pub name: String,
    /// inclusive range
    pub lo: u64,
    pub hi: u64,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tier0Percentiles {
    pub v_max: F64Hex,
    pub a_max: F64Hex,
    pub j_max: F64Hex,
    pub reach_max: F64Hex,
    pub q: F64Hex,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibrationFile {
    pub schema: String,
    pub canonical: String,
    pub method: CalMethod,
    pub alpha_num: u32,
    pub alpha_den: u32,
    /// successful calibration episodes available
    pub n_total: u32,
    /// episodes used to fit center/scale
    pub n_scale: u32,
    /// per-episode scores tau was taken over (2-way: == n_scale; 3-way: a disjoint set) -- k = ceil((n_calib+1)(1-alpha))
    pub n_calib: u32,
    pub n_holdout: u32,
    /// "2way" | "3way"
    pub split: String,
    pub seed_pool: SeedPool,
    pub source_run: String,
    pub source_arm: String,
    pub envelope_digest: String,
    pub embodiment_digest: String,
    pub policy_digest: String,
    pub lictor_git: String,
    pub horizon_ticks: u32,
    pub t_grid: u16,
    pub feature_ids: Vec<String>,
    pub mask: u32,
    pub gate: Vec<String>,
    pub center: F64Array,
    pub scale: F64Array,
    pub tau: F64Hex,
    pub kn: [u8; 2],
    pub warn_margin: F64Hex,
    pub holdout_fpr: F64Hex,
    pub holdout_fpr_k1: F64Hex,
    pub tier0_percentiles: Option<Tier0Percentiles>,
    pub notes: Vec<String>,
    pub digest: String,
}

impl CalibrationFile {
    pub fn load(_p: &std::path::Path) -> anyhow::Result<Self> {
        todo!("WP-5")
    }

    pub fn save(&self, _p: &std::path::Path) -> anyhow::Result<()> {
        todo!("WP-5")
    }

    pub fn compile(&self) -> anyhow::Result<CalibrationC> {
        todo!("WP-5")
    }

    pub fn digest_hex(&self) -> anyhow::Result<String> {
        todo!("WP-5")
    }

    /// -> lictor_runtime::session::CalibrationLoaded (file_sha256 over the bytes on disk)
    pub fn loaded(
        &self,
        _path: &std::path::Path,
    ) -> anyhow::Result<lictor_runtime::session::CalibrationLoaded> {
        todo!("WP-5")
    }
}
