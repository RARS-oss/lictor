// SPDX-License-Identifier: MIT
//! OWNER: WP-5. Stub written by WP-0; replace the bodies, keep the signatures.
//! Statistics: Clopper-Pearson, exact McNemar, point of no return, ROC-AUC, AUCPDT, paired bootstrap (splitmix64).

pub fn clopper_pearson(_k: u32, _n: u32, _conf: f64) -> (f64, f64) {
    todo!("WP-5")
}

pub fn mcnemar_exact(_b: u32, _c: u32) -> f64 {
    todo!("WP-5")
}

/// t_fail = min{t : c*_T - c*_t < eps}
pub fn ponr_tick(_coverage: &[f64], _eps_prog: f64) -> u32 {
    todo!("WP-5")
}

pub fn roc_auc(_pos: &[f64], _neg: &[f64]) -> f64 {
    todo!("WP-5")
}

pub fn aucpdt(_leads: &[i64], _horizon: u32) -> f64 {
    todo!("WP-5")
}

/// (diff, lo, hi); RNG = splitmix64 (in-crate)
pub fn paired_bootstrap_diff(_a: &[bool], _b: &[bool], _n_resamples: u32, _seed: u64) -> (f64, f64, f64) {
    todo!("WP-5")
}
