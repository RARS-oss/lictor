// SPDX-License-Identifier: MIT
//! calibration.json (schema lictor-calibration/v1), float-free; see FILE FORMATS.
//!
//! The file is self-digesting: `digest = sha256(canon(self with digest = ""))` under `jcs-floatfree/v1`. `load`
//! recomputes the digest from the parsed content and refuses a mismatch; `save` writes the file pretty-printed
//! with sorted keys after recomputing the digest, so bytes on disk are never trusted. `compile` expands the
//! `[t_grid, NFEAT]` centre/scale tables into the fixed `CalibrationC` arrays the decision path indexes with
//! `lictor_core::bin_of`, and `loaded` packages it for the runtime together with the sha256 of the bytes on disk
//! (the `lictor:calibration` input hash of every receipt).

use std::path::Path;

use lictor_canon::{digest_of, sha256_hex, F64Array, F64Hex, CANONICAL_ID};
use lictor_core::{CalMethod, CalibrationC, Feat, GateSpec, NFEAT, T_GRID};

pub const CALIBRATION_SCHEMA: &str = "lictor-calibration/v1";

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

/// Decode a 64-hex-character digest into 32 bytes.
pub fn hex32(s: &str) -> anyhow::Result<[u8; 32]> {
    let b = s.as_bytes();
    if b.len() != 64 {
        anyhow::bail!("digest `{s}` is not 64 hex characters");
    }
    let nib = |c: u8| -> anyhow::Result<u8> {
        match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => anyhow::bail!("digest `{s}` carries a non-hex character"),
        }
    };
    let mut out = [0u8; 32];
    for (i, o) in out.iter_mut().enumerate() {
        *o = (nib(b[2 * i])? << 4) | nib(b[2 * i + 1])?;
    }
    Ok(out)
}

/// Pretty JSON (2-space indent) with sorted keys at every level plus a trailing newline.
pub fn pretty_sorted<T: serde::Serialize>(t: &T) -> anyhow::Result<String> {
    let v = serde_json::to_value(t)?;
    let mut s = serde_json::to_string_pretty(&v)?;
    s.push('\n');
    Ok(s)
}

impl CalibrationFile {
    /// Parse and verify: schema, canonical id, and the self-digest recomputed from the parsed content.
    pub fn load(p: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        let f: Self = serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        if f.schema != CALIBRATION_SCHEMA {
            anyhow::bail!("{}: schema `{}` (this build reads `{CALIBRATION_SCHEMA}`)", p.display(), f.schema);
        }
        if f.canonical != CANONICAL_ID {
            anyhow::bail!("{}: canonical `{}` (this build reads `{CANONICAL_ID}`)", p.display(), f.canonical);
        }
        let recomputed = f.digest_hex()?;
        if recomputed != f.digest {
            anyhow::bail!(
                "{}: digest mismatch: file says {} but the content digests to {}",
                p.display(),
                f.digest,
                recomputed
            );
        }
        Ok(f)
    }

    /// Recompute the digest, then write pretty-printed with sorted keys (parent directories created).
    pub fn save(&self, p: &Path) -> anyhow::Result<()> {
        let mut f = self.clone();
        f.schema = CALIBRATION_SCHEMA.to_string();
        f.canonical = CANONICAL_ID.to_string();
        f.digest = f.digest_hex()?;
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(p, pretty_sorted(&f)?).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
        Ok(())
    }

    /// Expand into the hot-path table. Refuses a `t_grid` other than 1 or `T_GRID`, a shape other than
    /// `[t_grid, NFEAT]`, a gate that does not parse, a `mask` that is not the gate's mask, a method that does not
    /// match the grid, a zero `horizon_ticks`, or a malformed digest.
    pub fn compile(&self) -> anyhow::Result<CalibrationC> {
        let tg = self.t_grid as usize;
        if tg != 1 && tg != T_GRID {
            anyhow::bail!("t_grid must be 1 or {T_GRID}, got {}", self.t_grid);
        }
        match (self.method, tg) {
            (CalMethod::Static, 1) | (CalMethod::Binned, T_GRID) => {}
            (m, _) => anyhow::bail!("method {m:?} does not match t_grid {}", self.t_grid),
        }
        if self.horizon_ticks == 0 {
            anyhow::bail!("horizon_ticks must be > 0");
        }
        let want = [self.t_grid as u32, NFEAT as u32];
        for (name, arr) in [("center", &self.center), ("scale", &self.scale)] {
            if arr.shape != want || arr.data.len() != tg * NFEAT {
                anyhow::bail!("{name}: shape {:?} != [{}, {NFEAT}]", arr.shape, self.t_grid);
            }
            if arr.data.iter().any(|x| !x.is_finite()) {
                anyhow::bail!("{name}: non-finite entry");
            }
        }
        if self.scale.data.iter().any(|x| *x <= 0.0) {
            anyhow::bail!("scale: every entry must be > 0");
        }
        if self.feature_ids.len() != NFEAT
            || self.feature_ids.iter().zip(Feat::NAMES.iter()).any(|(a, b)| a != b)
        {
            anyhow::bail!("feature_ids must be exactly {:?}", Feat::NAMES);
        }
        let gate = GateSpec::parse(&self.gate).map_err(|e| anyhow::anyhow!("gate: {e}"))?;
        if gate.mask() != self.mask {
            anyhow::bail!("mask {} != the gate's mask {}", self.mask, gate.mask());
        }
        if self.kn[0] == 0 || self.kn[0] > self.kn[1] || self.kn[1] > 63 {
            anyhow::bail!("kn: need 1 <= k <= n <= 63, got {:?}", self.kn);
        }
        if self.alpha_den == 0 || self.alpha_num >= self.alpha_den {
            anyhow::bail!("alpha {}/{} is not in (0, 1)", self.alpha_num, self.alpha_den);
        }
        let tau = self.tau.0;
        if tau.is_nan() {
            anyhow::bail!("tau is NaN");
        }
        let digest = hex32(&self.digest)?;
        let mut c = CalibrationC::DISARMED;
        c.method = self.method;
        c.alpha_num = self.alpha_num;
        c.alpha_den = self.alpha_den;
        c.n_calib = self.n_calib;
        c.horizon_ticks = self.horizon_ticks;
        c.t_grid = self.t_grid;
        for b in 0..tg {
            for j in 0..NFEAT {
                c.center[b][j] = self.center.data[b * NFEAT + j];
                c.scale[b][j] = self.scale.data[b * NFEAT + j];
            }
        }
        c.mask = self.mask;
        c.gate = gate;
        c.tau = tau;
        c.digest = digest;
        Ok(c)
    }

    /// sha256(canon(self with digest = "")).
    pub fn digest_hex(&self) -> anyhow::Result<String> {
        let mut f = self.clone();
        f.digest = String::new();
        Ok(digest_of(&f)?)
    }

    /// -> lictor_runtime::session::CalibrationLoaded (file_sha256 over the bytes on disk)
    pub fn loaded(&self, path: &Path) -> anyhow::Result<lictor_runtime::session::CalibrationLoaded> {
        let bytes = std::fs::read(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?;
        Ok(lictor_runtime::session::CalibrationLoaded {
            c: self.compile()?,
            digest: self.digest_hex()?,
            embodiment_digest: self.embodiment_digest.clone(),
            file_sha256: sha256_hex(&bytes),
            path: path.display().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn sample() -> CalibrationFile {
        let n = T_GRID * NFEAT;
        CalibrationFile {
            schema: CALIBRATION_SCHEMA.to_string(),
            canonical: CANONICAL_ID.to_string(),
            method: CalMethod::Binned,
            alpha_num: 5,
            alpha_den: 100,
            n_total: 196,
            n_scale: 137,
            n_calib: 137,
            n_holdout: 59,
            split: "2way".to_string(),
            seed_pool: SeedPool { name: "calib".into(), lo: 900_000, hi: 900_299 },
            source_run: "run".into(),
            source_arm: "calib-obs".into(),
            envelope_digest: "7c".repeat(32),
            embodiment_digest: "91".repeat(32),
            policy_digest: "ab".repeat(32),
            lictor_git: "nogit".into(),
            horizon_ticks: 300,
            t_grid: T_GRID as u16,
            feature_ids: Feat::NAMES.iter().map(|s| s.to_string()).collect(),
            mask: 255,
            gate: Feat::NAMES[..8].iter().map(|s| s.to_string()).collect(),
            center: F64Array::from_slice(&vec![0.0; n], &[T_GRID as u32, NFEAT as u32]),
            scale: F64Array::from_slice(&vec![1.0; n], &[T_GRID as u32, NFEAT as u32]),
            tau: F64Hex(3.7),
            kn: [3, 5],
            warn_margin: F64Hex(0.5),
            holdout_fpr: F64Hex(0.04),
            holdout_fpr_k1: F64Hex(0.06),
            tier0_percentiles: None,
            notes: vec![],
            digest: String::new(),
        }
    }

    #[test]
    fn save_load_round_trip_and_tamper() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("calibration.a05.json");
        let f = sample();
        f.save(&p).unwrap();
        let g = CalibrationFile::load(&p).unwrap();
        assert_eq!(g.digest, g.digest_hex().unwrap());
        assert_eq!(g.tau, F64Hex(3.7));
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.contains("\"alpha_den\": 100"));
        // sorted keys: alpha_den precedes alpha_num precedes canonical
        let a = text.find("\"alpha_den\"").unwrap();
        let b = text.find("\"alpha_num\"").unwrap();
        let c = text.find("\"canonical\"").unwrap();
        assert!(a < b && b < c);
        // an edit that keeps the JSON valid is refused by the digest check
        let tampered = text.replace("\"n_calib\": 137", "\"n_calib\": 196");
        std::fs::write(&p, tampered).unwrap();
        let e = CalibrationFile::load(&p).unwrap_err().to_string();
        assert!(e.contains("digest mismatch"), "{e}");
    }

    #[test]
    fn compile_expands_the_tables() {
        let mut f = sample();
        f.center.data[3 * NFEAT + 6] = 1.5;
        f.scale.data[3 * NFEAT + 6] = 0.25;
        f.digest = f.digest_hex().unwrap();
        let c = f.compile().unwrap();
        assert_eq!(c.center[3][6], 1.5);
        assert_eq!(c.scale[3][6], 0.25);
        assert_eq!(c.mask, 255);
        assert_eq!(c.gate.n_terms, 8);
        assert_eq!(c.tau, 3.7);
        assert_eq!(c.t_grid, 100);
        assert!(c.armed());
        assert_eq!(hex32(&f.digest).unwrap(), c.digest);
        let mut bad = f.clone();
        bad.mask = 3;
        assert!(bad.compile().is_err());
        let mut bad = f.clone();
        bad.t_grid = 50;
        assert!(bad.compile().is_err());
        let mut bad = f;
        bad.method = CalMethod::Static;
        assert!(bad.compile().is_err());
    }
}
