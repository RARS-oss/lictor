// SPDX-License-Identifier: MIT
//! Tier-1 action-space failure signals (ARCHITECTURE 5.3): black-box, dimensionless, every channel oriented
//! LARGER = MORE ANOMALOUS.
//!
//! INVARIANT: every feature is a function of the `EmbodimentManifest` (`norm_center`, `norm_scale`, `dt`, `horizon`,
//! `exec_steps`) and of the data ONLY. Nothing here reads the four fitted Tier-0 limits or the operator list, which
//! `lictor envelope fit` changes AFTER a calibration was taken; a test greps this file's code for those identifiers.
//!
//! Cadence. The six chunk-boundary features (`tce`, `acc`, `acm_neg`, `njr`, `reach`, `speed_peak`) are recomputed only
//! on a tick that delivers a chunk and are then HELD in `Tier1Rt::held` / `held_valid` and copied out unchanged on
//! every other tick. The per-tick features (`path_ineff`, `stall`, `ext0..3`) are recomputed every call.
//!
//! Contracts with `decide()` (lictor-fuse):
//! - `decide()` pushes the current position into `rt.trail` BEFORE calling `features`; the trail is never pushed here.
//! - `features` is the SOLE writer of `rt.prev` / `rt.have_prev`. After `tce` / `acc` are computed against the previous
//!   RAW chunk, the incoming RAW chunk (the one `decide()` received, never the projected copy) is stored, so TCE is
//!   raw-vs-raw in Observe and Enforce alike: calibration features and enforcement features are the same function.
//! - The overlap between the previous and the incoming chunk is derived from the emit times, not from `exec_steps`:
//!   `s = new.t_emit - prev.t_emit`, `L = min(prev.horizon - s, new.horizon)`; the pair is absent when
//!   `s >= prev.horizon` (or `new.t_emit < prev.t_emit`) and `tce` / `acc` are invalid unless `L >= 2`. In sync mode with
//!   `d` steps of injected delay `L = 7 - d`; async delivery keeps `L = 7`.
//!
//! Float discipline: only `+ - * /`, comparisons and `lictor_core::fmath` (`sqrt`, `min`, `max`, `dist`, `norm`), all
//! accumulations in a fixed ascending order (row, then coordinate). No allocation, no clock, no panic path on data:
//! every dimension is re-derived from the slices actually supplied, never trusted from `cfg` alone.

use lictor_core::{
    fmath, ChunkBuf, ChunkView, Feat, FuseConfig, ObsView, Scores, MAX_D, MAX_EXT, MAX_H, MAX_POS, NFEAT,
};

use crate::window::Trail;

#[derive(Clone)]
pub struct Tier1Rt {
    pub prev: ChunkBuf,
    pub have_prev: bool,
    pub trail: Trail,
    pub held: [f64; NFEAT],
    pub held_valid: u32,
}

impl Tier1Rt {
    pub fn new() -> Self {
        Self {
            prev: ChunkBuf::new(),
            have_prev: false,
            trail: Trail::new(),
            held: [0.0; NFEAT],
            held_valid: 0,
        }
    }

    /// Per-episode reset: forgets the previous chunk, the trail and the held boundary features.
    pub fn reset(&mut self) {
        self.prev.clear();
        self.have_prev = false;
        self.trail.clear();
        self.held = [0.0; NFEAT];
        self.held_valid = 0;
    }
}

impl Default for Tier1Rt {
    fn default() -> Self {
        Self::new()
    }
}

/// ticks (2.0 s at 10 Hz)
pub const PE_WINDOW: usize = 20;

/// stall reference speed v_ref = STALL_VREF_FRAC * norm_scale_iso / dt (PushT: 0.05 * 256 / 0.1 = 128 px/s). Depends on the
/// manifest only -- NEVER on v_max, which `envelope fit` changes after calibration.
pub const STALL_VREF_FRAC: f64 = 0.05;

/// The `eps` of ARCHITECTURE 5 (denominator guards of `acc`, `njr`, `path_ineff`).
const EPS: f64 = 1e-9;

/// `PE_WINDOW` as a float (an integer-to-float conversion, exact).
const PE_WINDOW_F: f64 = PE_WINDOW as f64;

/// Updates `rt` and writes raw features into `sc.f` / `sc.valid`. Chunk-boundary features recompute only when `chunk.is_some()`.
/// CONTRACT: `decide()` pushes `rt.trail` BEFORE calling this. `features` is the SOLE writer of `rt.prev`/`rt.have_prev`: after
/// computing tce/acc it copies the RAW incoming chunk into `rt.prev`, so TCE is always raw-vs-raw in Observe and Enforce alike
/// (calibration features == enforcement features). Every feature depends on the EmbodimentManifest only.
pub fn features(
    cfg: &FuseConfig,
    rt: &mut Tier1Rt,
    obs: ObsView<'_>,
    chunk: Option<ChunkView<'_>>,
    idx: u16,
    sc: &mut Scores,
) {
    // `idx` (the row executed this tick) enters no feature: the boundary features are defined on the whole chunk
    // against the position at delivery, the per-tick features on the trail and `obs.ext`. It is part of the frozen
    // signature so the call site reads like `decide()`'s other stages.
    let _ = idx;

    if let Some(new) = chunk {
        boundary(cfg, rt, obs, new);
    }

    // Held chunk-boundary features (0.0 where invalid or not yet computed).
    let mut valid = rt.held_valid & Feat::CHUNK_BOUNDARY_MASK;
    let mut j = 0;
    while j < NFEAT {
        let bit = 1u32 << j;
        sc.f[j] = if valid & bit != 0 { rt.held[j] } else { 0.0 };
        j += 1;
    }

    // Per-tick trail statistics over the last PE_WINDOW positions.
    let pd = min_usize(cfg.pos_dim, MAX_POS);
    if rt.trail.len() >= PE_WINDOW {
        let n = rt.trail.net(pd, PE_WINDOW);
        let p = rt.trail.path(pd, PE_WINDOW);
        sc.f[Feat::PathIneff as usize] = 1.0 - n / fmath::max(p, EPS);
        let v_ref = STALL_VREF_FRAC * cfg.norm_scale_iso / cfg.dt;
        let denom = v_ref * PE_WINDOW_F * cfg.dt;
        sc.f[Feat::Stall as usize] = 1.0 - fmath::min(1.0, n / denom);
        valid |= Feat::PathIneff.bit() | Feat::Stall.bit();
    }

    // Tier-2 external scalars, verbatim; valid iff supplied.
    let n_ext = min_usize(obs.ext.len(), MAX_EXT);
    let mut k = 0;
    while k < n_ext {
        let j = Feat::Ext0 as usize + k;
        sc.f[j] = obs.ext[k];
        valid |= 1u32 << j;
        k += 1;
    }

    sc.valid = valid;
}

/// Recomputes the six chunk-boundary features against the previous RAW chunk, stores them in `rt.held`, then makes
/// the incoming RAW chunk the new `rt.prev`.
fn boundary(cfg: &FuseConfig, rt: &mut Tier1Rt, obs: ObsView<'_>, new: ChunkView<'_>) {
    let mut held = [0.0; NFEAT];
    let mut held_valid = 0u32;

    let d = min_usize(new.dim as usize, MAX_D);
    let h = new.data.len().checked_div(d).map_or(0, |n| min_usize(min_usize(new.horizon as usize, MAX_H), n));
    let nb = min_usize(min_usize(d, cfg.pos_dim), min_usize(obs.pos.len(), MAX_POS));
    let iso = cfg.norm_scale_iso;
    let abar_new =
        |i: usize, c: usize| -> f64 { (new.data[i * d + c] - cfg.norm_center[c]) * cfg.inv_norm_scale[c] };

    // tce / acc: overlap with the previous raw chunk, derived from the emit times.
    if h >= 1 && rt.have_prev {
        if let Some(prev) = rt.prev.view() {
            let pd = min_usize(prev.dim as usize, MAX_D);
            let ph = prev
                .data
                .len()
                .checked_div(pd)
                .map_or(0, |n| min_usize(min_usize(prev.horizon as usize, MAX_H), n));
            if pd == d && new.t_emit >= prev.t_emit {
                let s = (new.t_emit - prev.t_emit) as usize;
                if s < ph {
                    let l = min_usize(ph - s, h);
                    if l >= 2 {
                        let abar_prev = |i: usize, c: usize| -> f64 {
                            (prev.data[i * d + c] - cfg.norm_center[c]) * cfg.inv_norm_scale[c]
                        };
                        let mut num = 0.0;
                        let mut i = 0;
                        while i < l {
                            let mut c = 0;
                            while c < d {
                                let e = abar_prev(s + i, c) - abar_new(i, c);
                                num += e * e;
                                c += 1;
                            }
                            i += 1;
                        }
                        let mut den = 0.0;
                        let mut i = 1;
                        while i < l {
                            let mut c = 0;
                            while c < d {
                                let e = abar_new(i, c) - abar_new(i - 1, c);
                                den += e * e;
                                c += 1;
                            }
                            i += 1;
                        }
                        held[Feat::Tce as usize] = fmath::sqrt(num / ((l * d) as f64));
                        held[Feat::Acc as usize] = fmath::sqrt(num / (den + EPS));
                        held_valid |= Feat::Tce.bit() | Feat::Acc.bit();
                    }
                }
            }
        }
    }

    // acm_neg: negated mean normalised step between consecutive rows.
    let mut diff = [0.0; MAX_D];
    if h >= 2 {
        let mut acc = 0.0;
        let mut i = 1;
        while i < h {
            let mut c = 0;
            while c < d {
                diff[c] = abar_new(i, c) - abar_new(i - 1, c);
                c += 1;
            }
            acc += fmath::norm(&diff, d);
            i += 1;
        }
        held[Feat::AcmNeg as usize] = -(acc / ((h - 1) as f64));
        held_valid |= Feat::AcmNeg.bit();
    }

    // njr: mean squared third difference over eps + mean squared first difference.
    if h >= 4 {
        let mut num = 0.0;
        let mut i = 0;
        while i + 3 < h {
            let mut c = 0;
            while c < d {
                let v = ((abar_new(i + 3, c) - 3.0 * abar_new(i + 2, c)) + 3.0 * abar_new(i + 1, c))
                    - abar_new(i, c);
                num += v * v;
                c += 1;
            }
            i += 1;
        }
        let mut den = 0.0;
        let mut i = 0;
        while i + 1 < h {
            let mut c = 0;
            while c < d {
                let v = abar_new(i + 1, c) - abar_new(i, c);
                den += v * v;
                c += 1;
            }
            i += 1;
        }
        held[Feat::Njr as usize] = (num / ((h - 3) as f64)) / (EPS + den / ((h - 1) as f64));
        held_valid |= Feat::Njr.bit();
    }

    // reach: the soft twin of the hard reach check, in units of norm_scale_iso.
    if h >= 1 && nb > 0 {
        held[Feat::Reach as usize] = fmath::dist(&new.data[..d], obs.pos, nb) / iso;
        held_valid |= Feat::Reach.bit();
    }

    // speed_peak: the largest commanded step, q_{-1} = p_t where the action and position frames coincide.
    if h >= 1 {
        let mut q0 = [0.0; MAX_D];
        q0[..d].copy_from_slice(&new.data[..d]);
        q0[..nb].copy_from_slice(&obs.pos[..nb]);
        let mut peak = 0.0;
        let mut i = 0;
        while i < h {
            let a = &new.data[i * d..i * d + d];
            let n = if i == 0 {
                fmath::dist(a, &q0, d)
            } else {
                fmath::dist(a, &new.data[(i - 1) * d..i * d], d)
            };
            peak = fmath::max(peak, n);
            i += 1;
        }
        held[Feat::SpeedPeak as usize] = peak / iso;
        held_valid |= Feat::SpeedPeak.bit();
    }

    rt.held = held;
    rt.held_valid = held_valid;

    // The sole write of the previous chunk: the RAW incoming chunk, after tce/acc used the old one.
    rt.prev.copy_from(new);
    rt.have_prev = true;
}

#[inline]
fn min_usize(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_and_reset_are_empty() {
        let mut rt = Tier1Rt::default();
        assert!(!rt.have_prev);
        assert!(rt.trail.is_empty());
        assert_eq!(rt.held_valid, 0);
        assert!(!rt.prev.is_filled());
        rt.trail.push(&[1.0, 2.0], 2);
        rt.held_valid = Feat::CHUNK_BOUNDARY_MASK;
        rt.held[0] = 3.0;
        rt.have_prev = true;
        rt.reset();
        assert!(!rt.have_prev);
        assert!(rt.trail.is_empty());
        assert_eq!(rt.held_valid, 0);
        assert_eq!(rt.held, [0.0; NFEAT]);
    }

    #[test]
    fn window_constants() {
        assert_eq!(PE_WINDOW, 20);
        assert_eq!(PE_WINDOW_F, 20.0);
        let v_ref = STALL_VREF_FRAC * 256.0 / 0.1;
        assert!((v_ref - 128.0).abs() < 1e-12, "{v_ref}");
    }
}
