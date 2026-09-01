// SPDX-License-Identifier: MIT
//! OWNER: WP-2. Stub written by WP-0; replace the bodies, keep the signatures.
//! The K-of-N hit window and the fixed-size position trail.
//!
//! `Trail::new`/`clear`/`push`/`len`/`is_empty` are written for real (ring-buffer bookkeeping over the private
//! fields; `head` is the index of the NEXT write slot, the newest sample sits at `(head + TRAIL_W - 1) % TRAIL_W`,
//! `len` saturates at `TRAIL_W`). `net`/`path` and `Window`'s methods are WP-2's.

use lictor_core::MAX_POS;

#[derive(Clone, Copy, Debug, Default)]
pub struct Window {
    pub bits: u64,
}

impl Window {
    #[inline]
    pub fn push(&mut self, _hit: bool, _mask: u64) {
        todo!("WP-2")
    }

    #[inline]
    pub fn hits(&self) -> u8 {
        todo!("WP-2")
    }

    pub fn clear(&mut self) {
        todo!("WP-2")
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

    pub fn push(&mut self, p: &[f64], d: usize) {
        let slot = self.head as usize;
        self.buf[slot][..d].copy_from_slice(&p[..d]);
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

    /// ||p_t - p_{t-w+1}||
    pub fn net(&self, _d: usize, _w: usize) -> f64 {
        todo!("WP-2")
    }

    /// sum of consecutive distances over the last w
    pub fn path(&self, _d: usize, _w: usize) -> f64 {
        todo!("WP-2")
    }
}

impl Default for Trail {
    fn default() -> Self {
        Self::new()
    }
}
