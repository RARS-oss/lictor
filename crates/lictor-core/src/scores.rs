// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; replace the bodies, keep the signatures.
//! Features and the runtime calibration table.

use serde::{Deserialize, Serialize};

use crate::envelope::GateSpec;

pub const NFEAT: usize = 12;
pub const T_GRID: usize = 100;

/// Frozen feature ids. EVERY channel is oriented LARGER = MORE ANOMALOUS (acm is stored negated).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
pub enum Feat {
    Tce = 0,
    Acc = 1,
    AcmNeg = 2,
    Njr = 3,
    Reach = 4,
    PathIneff = 5,
    Stall = 6,
    SpeedPeak = 7,
    Ext0 = 8,
    Ext1 = 9,
    Ext2 = 10,
    Ext3 = 11,
}

impl Feat {
    pub const NAMES: [&'static str; NFEAT] = [
        "tce",
        "acc",
        "acm_neg",
        "njr",
        "reach",
        "path_ineff",
        "stall",
        "speed_peak",
        "ext0",
        "ext1",
        "ext2",
        "ext3",
    ];
    /// tce acc acm_neg njr reach speed_peak: updated at a chunk boundary, held between
    pub const CHUNK_BOUNDARY_MASK: u32 = 0b0000_1001_1111;
    /// path_ineff stall ext0..3: updated every tick
    pub const PER_TICK_MASK: u32 = 0b1111_0110_0000;

    pub fn from_name(_s: &str) -> Option<Feat> {
        todo!("WP-1")
    }

    #[inline]
    pub fn bit(self) -> u32 {
        1u32 << (self as u32)
    }

    pub fn name(self) -> &'static str {
        todo!("WP-1")
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scores {
    /// raw, dimensionless (0.0 where !valid)
    pub f: [f64; NFEAT],
    /// standardised (0.0 where !valid or unmasked)
    pub z: [f64; NFEAT],
    /// aggregate nonconformity = max over gate terms of min over the term; NEG_INFINITY when nothing valid
    pub s: f64,
    /// bit j set <=> feature j computable this tick
    pub valid: u32,
    /// bit j set <=> z_j > tau (informational; the decision is on s)
    pub fired: u32,
}

/// Static == Binned with t_grid = 1
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalMethod {
    Static,
    Binned,
}

/// Hot-path calibration artefact: fixed arrays (~19 KB), no allocation, no I/O. Built once from calibration.json.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationC {
    pub method: CalMethod,
    pub alpha_num: u32,
    pub alpha_den: u32,
    /// the number of per-episode scores tau was taken over (== calibration.json n_calib; 2-way split: n_fit)
    pub n_calib: u32,
    /// the time base used for binning (PushT: 300; == manifest.horizon_ticks)
    pub horizon_ticks: u32,
    /// 1 (static) or T_GRID (binned)
    pub t_grid: u16,
    /// robust median per (bin, feature)
    pub center: [[f64; NFEAT]; T_GRID],
    /// 1.4826*MAD per (bin, feature), floored at 1e-9
    pub scale: [[f64; NFEAT]; T_GRID],
    /// features entering s (== gate.mask())
    pub mask: u32,
    pub gate: GateSpec,
    /// split-CP quantile on per-episode max s; +INFINITY = never fires
    pub tau: f64,
    pub digest: [u8; 32],
}

impl CalibrationC {
    /// mask 0, gate DISARMED, tau +INFINITY, digest zero (scale 1.0 so a stray division stays finite).
    pub const DISARMED: Self = Self {
        method: CalMethod::Static,
        alpha_num: 0,
        alpha_den: 1,
        n_calib: 0,
        horizon_ticks: 1,
        t_grid: 1,
        center: [[0.0; NFEAT]; T_GRID],
        scale: [[1.0; NFEAT]; T_GRID],
        mask: 0,
        gate: GateSpec::DISARMED,
        tau: f64::INFINITY,
        digest: [0; 32],
    };

    #[inline]
    pub fn bin(&self, t: u32) -> usize {
        bin_of(t, self.horizon_ticks, self.t_grid)
    }

    pub fn armed(&self) -> bool {
        self.mask != 0
    }
}

/// INTEGER arithmetic only; shared by runtime and offline calibrator. A mismatch is a determinism trap.
#[inline]
pub fn bin_of(t: u32, horizon_ticks: u32, t_grid: u16) -> usize {
    let g = t_grid as u64;
    let h = if horizon_ticks == 0 { 1 } else { horizon_ticks as u64 };
    let b = (t as u64) * g / h;
    if b >= g {
        (g - 1) as usize
    } else {
        b as usize
    }
}
