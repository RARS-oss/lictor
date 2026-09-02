// SPDX-License-Identifier: MIT
//! The K-of-N hit window and the fixed-size position trail.
//!
//! `Window` is a u64 shift register: `push` shifts the newest hit in at bit 0 and masks to N bits, `hits` is the
//! popcount. Integer arithmetic only; it never allocates.
//!
//! `Trail` is a fixed ring of the last `TRAIL_W` proprioceptive positions (`head` is the index of the NEXT write
//! slot, the newest sample sits at `(head + TRAIL_W - 1) % TRAIL_W`, `len` saturates at `TRAIL_W`). `net` and
//! `path` are the two per-tick path statistics behind the `path_ineff` and `stall` features; both use
//! `lictor_core::fmath::dist` (fixed left-to-right accumulation) and both return `0.0` while fewer than `w`
//! samples have been pushed, so a caller can never read a half-filled window.

use lictor_core::{fmath, MAX_POS};

#[derive(Clone, Copy, Debug, Default)]
pub struct Window {
    pub bits: u64,
}

impl Window {
    /// `bits = ((bits << 1) | hit) & mask`. The mask is `(1 << n) - 1` (n <= 63, precomputed by `compile`).
    #[inline]
    pub fn push(&mut self, hit: bool, mask: u64) {
        let h = if hit { 1u64 } else { 0u64 };
        self.bits = ((self.bits << 1) | h) & mask;
    }

    /// popcount of the ring (0..=63).
    #[inline]
    pub fn hits(&self) -> u8 {
        // count_ones() of a u64 is at most 64, so the narrowing is lossless.
        self.bits.count_ones() as u8
    }

    pub fn clear(&mut self) {
        self.bits = 0;
    }
}

pub const TRAIL_W: usize = 32;

#[derive(Clone, Copy)]
pub struct Trail {
    buf: [[f64; MAX_POS]; TRAIL_W],
    head: u8,
    len: u8,
}

impl Trail {
    pub const fn new() -> Self {
        Self { buf: [[0.0; MAX_POS]; TRAIL_W], head: 0, len: 0 }
    }

    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Records `p[..d]` as the newest sample. `d` is clamped to `MAX_POS` and to `p.len()` (never panics on data).
    pub fn push(&mut self, p: &[f64], d: usize) {
        let d = dim_of(p, d);
        let slot = self.head as usize;
        self.buf[slot][..d].copy_from_slice(&p[..d]);
        // The narrowing casts are lossless: both values are < TRAIL_W = 32.
        self.head = ((slot + 1) % TRAIL_W) as u8;
        if (self.len as usize) < TRAIL_W {
            self.len += 1;
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Ring slot of the sample `k` steps older than the newest (`k == 0` is the newest).
    #[inline]
    fn slot_back(&self, k: usize) -> usize {
        (self.head as usize + TRAIL_W - 1 - k) % TRAIL_W
    }

    /// `||p_t - p_{t-w+1}||` over the first `d` coordinates: the straight-line displacement across the last `w`
    /// samples. `0.0` when `w == 0` or fewer than `w` samples have been pushed.
    pub fn net(&self, d: usize, w: usize) -> f64 {
        if w == 0 || w > TRAIL_W || self.len() < w {
            return 0.0;
        }
        let d = if d > MAX_POS { MAX_POS } else { d };
        let newest = self.slot_back(0);
        let oldest = self.slot_back(w - 1);
        fmath::dist(&self.buf[newest], &self.buf[oldest], d)
    }

    /// Sum of the `w - 1` consecutive distances over the last `w` samples, accumulated oldest -> newest.
    /// `0.0` when `w < 2` or fewer than `w` samples have been pushed.
    pub fn path(&self, d: usize, w: usize) -> f64 {
        if !(2..=TRAIL_W).contains(&w) || self.len() < w {
            return 0.0;
        }
        let d = if d > MAX_POS { MAX_POS } else { d };
        let mut acc = 0.0;
        // k counts from the oldest sample of the window (w-1 steps back) towards the newest.
        let mut k = w - 1;
        while k >= 1 {
            let older = self.slot_back(k);
            let newer = self.slot_back(k - 1);
            acc += fmath::dist(&self.buf[older], &self.buf[newer], d);
            k -= 1;
        }
        acc
    }
}

impl Default for Trail {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn dim_of(p: &[f64], d: usize) -> usize {
    let d = if d > MAX_POS { MAX_POS } else { d };
    if d > p.len() {
        p.len()
    } else {
        d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_shift_and_mask() {
        let mask = (1u64 << 5) - 1;
        let mut w = Window::default();
        for _ in 0..7 {
            w.push(true, mask);
        }
        assert_eq!(w.bits, 0b11111);
        assert_eq!(w.hits(), 5);
        w.push(false, mask);
        assert_eq!(w.bits, 0b11110);
        assert_eq!(w.hits(), 4);
        w.clear();
        assert_eq!(w.bits, 0);
        assert_eq!(w.hits(), 0);
    }

    #[test]
    fn trail_ring_wraps() {
        let mut t = Trail::new();
        assert!(t.is_empty());
        for i in 0..(TRAIL_W + 5) {
            let p = [i as f64, 0.0];
            t.push(&p, 2);
        }
        assert_eq!(t.len(), TRAIL_W);
        // newest = 36, 31 steps back = 5
        assert_eq!(t.net(2, TRAIL_W), 31.0);
        assert_eq!(t.path(2, TRAIL_W), 31.0);
        assert_eq!(t.net(2, TRAIL_W + 1), 0.0);
        assert_eq!(t.net(2, 0), 0.0);
        assert_eq!(t.path(2, 1), 0.0);
        assert_eq!(t.net(2, 1), 0.0);
    }

    #[test]
    fn trail_net_vs_path() {
        let mut t = Trail::new();
        t.push(&[0.0, 0.0], 2);
        t.push(&[3.0, 4.0], 2);
        t.push(&[0.0, 0.0], 2);
        assert_eq!(t.net(2, 3), 0.0);
        assert_eq!(t.path(2, 3), 10.0);
        assert_eq!(t.net(2, 2), 5.0);
        assert_eq!(t.path(2, 2), 5.0);
        assert_eq!(t.net(2, 4), 0.0);
        assert_eq!(t.path(2, 4), 0.0);
    }
}
