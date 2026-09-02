// SPDX-License-Identifier: MIT
//! Tier-0 geometric limits and the direction-preserving projection (ARCHITECTURE 5.1).
//!
//! Every check is `trip iff quantity > limit` (strict), evaluated on the RAW chunk with `q_{-1} = p_t`. A check
//! that is not armed in `cfg.tier0_enabled` neither trips nor clamps. Speed, contact and the leash compare the
//! commanded step length `||a_i - q_{i-1}||` against `step_max = v_max * dt` (resp. `v_contact * dt`), which is
//! the `v_i > v_max` row of the table expressed in the same quantity the projection uses, so a trip and a leash
//! never disagree about the same row. Accel and jerk use `inv_dt2` / `inv_dt3` from the compiled config.
//!
//! Projection (`ClampMode::Project`): the sequential leash of 5.1 -- componentwise box clamp, then a shortening of
//! the step towards the PREVIOUS PROJECTED row, then (row 0 only) a pull towards `p_t` on the reach ray. Both
//! leashes scale the offending vector by `limit * LEASH_SHRINK / n` instead of `limit / n`: the factor
//! `LEASH_SHRINK = 1 - 2^-36` (1.5e-11 relative, ~1e-9 px on a 100 px step) is a rounding guard that makes the
//! projected row re-pass the strict check bit-for-bit when `decide()` re-checks it intra-chunk against
//! `last_cmd`, and makes the projection idempotent. Without it a leashed step lands 0-2 ulps either side of
//! `step_max` and would re-trip SPEED on roughly half of the committed rows. Only `+ - * /` and `fmath::sqrt`.
//!
//! Dimensions: box and reach apply to the first `min(dim, pos_dim)` coordinates (the action and the proprioceptive
//! position share a frame for the position kinds this tier is defined on); for a wider action the extra
//! coordinates take `q_{-1}[c] = a_0[c]` (no step on the first row). Nothing here allocates or panics on data:
//! slice lengths are re-derived from the inputs, never trusted from `cfg` alone.

use lictor_core::{
    fmath, ChunkBuf, ChunkView, ClampMode, FuseConfig, ObsView, TripMask, MAX_D, MAX_H, MAX_POS,
};

#[derive(Clone, Copy, Debug, Default)]
pub struct Tier0Out {
    pub trips: u32,
    pub clamped_dims: u32,
    pub peak_speed: f64,
}

/// `1 - 2^-36`: the leash lands the step this far INSIDE the limit (see the module doc).
const LEASH_SHRINK: f64 = 1.0 - 1.0 / 68719476736.0;

/// Which checks are armed for this call and the leash lengths they impose.
#[derive(Clone, Copy)]
struct Arm {
    workspace: bool,
    speed: bool,
    accel: bool,
    jerk: bool,
    reach: bool,
    /// armed AND the agent is inside the contact radius this tick
    contact: bool,
    step_contact: f64,
}

impl Arm {
    fn resolve(cfg: &FuseConfig, obs: ObsView<'_>) -> Self {
        let en = cfg.tier0_enabled;
        let mut contact = false;
        let mut step_contact = 0.0;
        if en & TripMask::CONTACT != 0 {
            if let Some(cl) = cfg.contact {
                let i0 = cl.aux_center[0] as usize;
                let i1 = cl.aux_center[1] as usize;
                if obs.pos.len() >= 2 && obs.aux.len() > i0 && obs.aux.len() > i1 {
                    let centre = [obs.aux[i0], obs.aux[i1]];
                    let dist = fmath::dist(obs.pos, &centre, 2);
                    if dist <= cl.radius {
                        contact = true;
                        step_contact = cl.v_max * cfg.dt;
                    }
                }
            }
        }
        Self {
            workspace: en & TripMask::WORKSPACE != 0,
            speed: en & TripMask::SPEED != 0,
            accel: en & TripMask::ACCEL != 0,
            jerk: en & TripMask::JERK != 0,
            reach: en & TripMask::REACH != 0,
            contact,
            step_contact,
        }
    }

    /// The shortest armed leash, if any leash is armed at all.
    #[inline]
    fn leash(&self, step_max: f64) -> Option<f64> {
        match (self.speed, self.contact) {
            (true, true) => Some(fmath::min(step_max, self.step_contact)),
            (true, false) => Some(step_max),
            (false, true) => Some(self.step_contact),
            (false, false) => None,
        }
    }
}

/// Everything one projected row needs besides the row itself.
struct Proj<'a> {
    cfg: &'a FuseConfig,
    pos: &'a [f64],
    arm: Arm,
    leash: Option<f64>,
    /// box / reach coordinates
    nb: usize,
    /// action coordinates
    d: usize,
}

impl Proj<'_> {
    /// One row of the sequential leash, in place: box clamp, step leash towards `q`, reach pull towards `pos`
    /// (first row only). `a[..d]` holds the raw row on entry and the projected row on exit.
    #[inline]
    fn row(&self, first: bool, q: &[f64; MAX_D], a: &mut [f64; MAX_D]) {
        let cfg = self.cfg;
        if self.arm.workspace {
            let mut c = 0;
            while c < self.nb {
                a[c] = fmath::clamp(a[c], cfg.box_lo[c], cfg.box_hi[c]);
                c += 1;
            }
        }
        if let Some(step) = self.leash {
            let n = fmath::dist(a, q, self.d);
            if n > step {
                let s = step * LEASH_SHRINK / n;
                let mut c = 0;
                while c < self.d {
                    a[c] = q[c] + (a[c] - q[c]) * s;
                    c += 1;
                }
            }
        }
        if first && self.arm.reach && self.nb > 0 {
            let r = fmath::dist(a, self.pos, self.nb);
            if r > cfg.reach_max {
                let s = cfg.reach_max * LEASH_SHRINK / r;
                let mut c = 0;
                while c < self.nb {
                    a[c] = self.pos[c] + (a[c] - self.pos[c]) * s;
                    c += 1;
                }
            }
        }
    }
}

#[inline]
fn min_usize(a: usize, b: usize) -> usize {
    if a < b {
        a
    } else {
        b
    }
}

/// Number of box/reach coordinates: the action and position dims share a frame up to the shorter of the two.
#[inline]
fn box_dims(cfg: &FuseConfig, obs: ObsView<'_>, d: usize) -> usize {
    min_usize(min_usize(d, cfg.pos_dim), min_usize(obs.pos.len(), MAX_POS))
}

/// Bits of the coordinates where `a` differs from `raw` (bitwise inequality).
#[inline]
fn moved_dims(raw: &[f64], a: &[f64], d: usize) -> u32 {
    let mut m = 0u32;
    let mut c = 0;
    while c < d {
        if a[c] != raw[c] {
            m |= 1u32 << c;
        }
        c += 1;
    }
    m
}

/// Whole-chunk check at a chunk boundary. Writes the projected chunk into `out` (== ch when nothing clamped).
/// Exactly `horizon` iterations. Zero allocation.
pub fn check_chunk(cfg: &FuseConfig, obs: ObsView<'_>, ch: ChunkView<'_>, out: &mut ChunkBuf) -> Tier0Out {
    out.copy_from(ch);
    let mut res = Tier0Out::default();
    let d = min_usize(ch.dim as usize, MAX_D);
    if d == 0 {
        return res;
    }
    let h = min_usize(min_usize(ch.horizon as usize, MAX_H), ch.data.len() / d);
    if h == 0 {
        return res;
    }
    let nb = box_dims(cfg, obs, d);
    let arm = Arm::resolve(cfg, obs);
    let row = |i: usize| -> &[f64] { &ch.data[i * d..i * d + d] };

    // q_{-1}: the current position where the frames coincide, a_0 itself beyond that.
    let mut q0 = [0.0; MAX_D];
    q0[..d].copy_from_slice(row(0));
    q0[..nb].copy_from_slice(&obs.pos[..nb]);

    // ---- trips on the RAW chunk ----
    let mut trips = 0u32;
    let mut peak_step = 0.0;
    let mut i = 0;
    while i < h {
        let a = row(i);
        let n = if i == 0 { fmath::dist(a, &q0, d) } else { fmath::dist(a, row(i - 1), d) };
        peak_step = fmath::max(peak_step, n);
        if arm.speed && n > cfg.step_max {
            trips |= TripMask::SPEED;
        }
        if arm.contact && n > arm.step_contact {
            trips |= TripMask::CONTACT;
        }
        if arm.workspace && outside_box(cfg, a, nb) {
            trips |= TripMask::WORKSPACE;
        }
        i += 1;
    }
    let mut diff = [0.0; MAX_D];
    if arm.accel && h >= 2 {
        // i in [0, H-1): a_{i+1} - 2 a_i + a_{i-1}
        let mut i = 0;
        while i + 1 < h {
            let am1: &[f64] = if i == 0 { &q0[..d] } else { row(i - 1) };
            let a0 = row(i);
            let a1 = row(i + 1);
            let mut c = 0;
            while c < d {
                diff[c] = (a1[c] - 2.0 * a0[c]) + am1[c];
                c += 1;
            }
            if fmath::norm(&diff, d) * cfg.inv_dt2 > cfg.a_max {
                trips |= TripMask::ACCEL;
            }
            i += 1;
        }
    }
    if arm.jerk && h >= 3 {
        // i in [0, H-2): a_{i+2} - 3 a_{i+1} + 3 a_i - a_{i-1}
        let mut i = 0;
        while i + 2 < h {
            let am1: &[f64] = if i == 0 { &q0[..d] } else { row(i - 1) };
            let a0 = row(i);
            let a1 = row(i + 1);
            let a2 = row(i + 2);
            let mut c = 0;
            while c < d {
                diff[c] = ((a2[c] - 3.0 * a1[c]) + 3.0 * a0[c]) - am1[c];
                c += 1;
            }
            if fmath::norm(&diff, d) * cfg.inv_dt3 > cfg.j_max {
                trips |= TripMask::JERK;
            }
            i += 1;
        }
    }
    if arm.reach && nb > 0 && fmath::dist(row(0), obs.pos, nb) > cfg.reach_max {
        trips |= TripMask::REACH;
    }
    res.trips = trips;
    res.peak_speed = peak_step * cfg.inv_dt;

    // ---- projection ----
    if cfg.clamp != ClampMode::Project {
        return res;
    }
    let proj = Proj { cfg, pos: obs.pos, arm, leash: arm.leash(cfg.step_max), nb, d };
    let mut clamped = 0u32;
    let mut q = q0;
    let mut a = [0.0; MAX_D];
    let mut i = 0;
    while i < h {
        let raw = row(i);
        a[..d].copy_from_slice(raw);
        proj.row(i == 0, &q, &mut a);
        clamped |= moved_dims(raw, &a, d);
        out.row_mut(i)[..d].copy_from_slice(&a[..d]);
        q = a;
        i += 1;
    }
    res.clamped_dims = clamped;
    res
}

/// Intra-chunk per-tick check of the single committed action `a` against predecessor `prev` (box, speed, reach, contact).
/// Writes the (possibly leashed) action into `out[..dim]`.
pub fn check_action(
    cfg: &FuseConfig,
    obs: ObsView<'_>,
    prev: &[f64],
    a: &[f64],
    out: &mut [f64],
) -> Tier0Out {
    let mut res = Tier0Out::default();
    let d = min_usize(min_usize(cfg.dim, MAX_D), min_usize(a.len(), min_usize(prev.len(), out.len())));
    if d == 0 {
        return res;
    }
    let nb = box_dims(cfg, obs, d);
    let arm = Arm::resolve(cfg, obs);

    let mut trips = 0u32;
    let n = fmath::dist(a, prev, d);
    if arm.speed && n > cfg.step_max {
        trips |= TripMask::SPEED;
    }
    if arm.contact && n > arm.step_contact {
        trips |= TripMask::CONTACT;
    }
    if arm.workspace && outside_box(cfg, a, nb) {
        trips |= TripMask::WORKSPACE;
    }
    if arm.reach && nb > 0 && fmath::dist(a, obs.pos, nb) > cfg.reach_max {
        trips |= TripMask::REACH;
    }
    res.trips = trips;
    res.peak_speed = n * cfg.inv_dt;

    let mut q = [0.0; MAX_D];
    q[..d].copy_from_slice(&prev[..d]);
    let mut p = [0.0; MAX_D];
    p[..d].copy_from_slice(&a[..d]);
    if cfg.clamp == ClampMode::Project {
        let proj = Proj { cfg, pos: obs.pos, arm, leash: arm.leash(cfg.step_max), nb, d };
        proj.row(true, &q, &mut p);
        res.clamped_dims = moved_dims(a, &p, d);
    }
    out[..d].copy_from_slice(&p[..d]);
    res
}

#[inline]
fn outside_box(cfg: &FuseConfig, a: &[f64], nb: usize) -> bool {
    let mut c = 0;
    while c < nb {
        if a[c] < cfg.box_lo[c] || a[c] > cfg.box_hi[c] {
            return true;
        }
        c += 1;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shrink_is_just_inside_one() {
        let s = LEASH_SHRINK;
        assert!(s < 1.0);
        assert!(s > 1.0 - 2e-11);
    }

    #[test]
    fn leash_resolution() {
        let arm = Arm {
            workspace: true,
            speed: true,
            accel: true,
            jerk: true,
            reach: true,
            contact: true,
            step_contact: 40.0,
        };
        assert_eq!(arm.leash(100.0), Some(40.0));
        let no_speed = Arm { speed: false, ..arm };
        assert_eq!(no_speed.leash(100.0), Some(40.0));
        let no_contact = Arm { contact: false, ..arm };
        assert_eq!(no_contact.leash(100.0), Some(100.0));
        let none = Arm { speed: false, contact: false, ..arm };
        assert_eq!(none.leash(100.0), None);
    }

    #[test]
    fn moved_dims_is_bitwise() {
        assert_eq!(moved_dims(&[1.0, 2.0, 3.0], &[1.0, 2.5, 3.0], 3), 0b010);
        assert_eq!(moved_dims(&[1.0, 2.0], &[1.0, 2.0], 2), 0);
        assert_eq!(moved_dims(&[0.0, 0.0], &[-0.0, 1.0], 2), 0b10);
    }
}
