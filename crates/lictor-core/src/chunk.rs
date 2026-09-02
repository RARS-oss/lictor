// SPDX-License-Identifier: MIT
//! The action-chunk IR: pre-allocated owned storage (`ChunkBuf`), the borrowed `Copy` view every detector
//! receives (`ChunkView`), and the per-tick observation view (`ObsView`). Nothing here allocates; every accessor
//! clamps instead of panicking on an out-of-range index (`debug_assert!` in debug builds only).

use serde::{Deserialize, Serialize};

use crate::fmath;

pub const MAX_H: usize = 64; // pi0 chunks are 50 -> headroom
pub const MAX_D: usize = 32; // pi0 padded action_dim is 32 -> headroom
pub const MAX_POS: usize = 32; // proprioceptive position dim
pub const MAX_AUX: usize = 16; // embodiment-declared auxiliary scalars (privileged sim state)
pub const MAX_EXT: usize = 4; // Tier-2 external scalars

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    EePosition,
    EeDelta,
    JointPosition,
    JointVelocity,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkError {
    TooLong,
    TooWide,
    LenMismatch,
    ExecExceedsHorizon,
    Empty,
}

/// Pre-allocated, owned chunk storage (16.5 KB). Lives inside `FuseRt`; never allocated on the hot path.
#[derive(Clone)]
pub struct ChunkBuf {
    data: [f64; MAX_H * MAX_D],
    seq: u32,
    t_emit: u32,
    horizon: u16,
    dim: u16,
    exec_steps: u16,
    filled: bool,
}

impl ChunkBuf {
    pub const fn new() -> Self {
        Self {
            data: [0.0; MAX_H * MAX_D],
            seq: 0,
            t_emit: 0,
            horizon: 0,
            dim: 0,
            exec_steps: 0,
            filled: false,
        }
    }

    /// Row-major copy of `src` (len == h*d). Non-finite values are accepted here and caught by `decide()`.
    ///
    /// Checks, in order: `h > 0 && d > 0` (`Empty`), `h <= MAX_H` (`TooLong`), `d <= MAX_D` (`TooWide`),
    /// `exec <= h` (`ExecExceedsHorizon`), `src.len() == h*d` (`LenMismatch`). On error the buffer is untouched.
    pub fn fill(
        &mut self,
        seq: u32,
        t_emit: u32,
        h: u16,
        d: u16,
        exec: u16,
        src: &[f64],
    ) -> Result<(), ChunkError> {
        if h == 0 || d == 0 {
            return Err(ChunkError::Empty);
        }
        if h as usize > MAX_H {
            return Err(ChunkError::TooLong);
        }
        if d as usize > MAX_D {
            return Err(ChunkError::TooWide);
        }
        if exec > h {
            return Err(ChunkError::ExecExceedsHorizon);
        }
        let n = h as usize * d as usize;
        if src.len() != n {
            return Err(ChunkError::LenMismatch);
        }
        self.data[..n].copy_from_slice(src);
        self.seq = seq;
        self.t_emit = t_emit;
        self.horizon = h;
        self.dim = d;
        self.exec_steps = exec;
        self.filled = true;
        Ok(())
    }

    /// Copies header and rows of `v` in. A view whose `data` is shorter than `horizon*dim` (only possible for a
    /// hand-built view) is copied as far as it goes; a longer one is truncated to the buffer.
    pub fn copy_from(&mut self, v: ChunkView<'_>) {
        let n = if v.data.len() < MAX_H * MAX_D { v.data.len() } else { MAX_H * MAX_D };
        self.data[..n].copy_from_slice(&v.data[..n]);
        self.seq = v.seq;
        self.t_emit = v.t_emit;
        self.horizon = v.horizon;
        self.dim = v.dim;
        self.exec_steps = v.exec_steps;
        self.filled = true;
    }

    /// None when !filled.
    pub fn view(&self) -> Option<ChunkView<'_>> {
        if !self.filled {
            return None;
        }
        let n = self.horizon as usize * self.dim as usize;
        Some(ChunkView {
            seq: self.seq,
            t_emit: self.t_emit,
            horizon: self.horizon,
            dim: self.dim,
            exec_steps: self.exec_steps,
            data: &self.data[..n],
        })
    }

    /// len == dim; i clamped to horizon-1 (an empty slice when the buffer has no rows).
    pub fn row_mut(&mut self, i: usize) -> &mut [f64] {
        let h = self.horizon as usize;
        let d = self.dim as usize;
        debug_assert!(h == 0 || i < h, "row_mut index {i} >= horizon {h}");
        if h == 0 || d == 0 {
            return &mut self.data[..0];
        }
        let i = if i < h { i } else { h - 1 };
        let start = i * d;
        &mut self.data[start..start + d]
    }

    pub fn clear(&mut self) {
        self.filled = false;
    }

    pub fn is_filled(&self) -> bool {
        self.filled
    }
}

impl Default for ChunkBuf {
    fn default() -> Self {
        Self::new()
    }
}

/// Borrowed, Copy view. THE chunk IR passed to detectors.
#[derive(Clone, Copy, Debug)]
pub struct ChunkView<'a> {
    pub seq: u32,        // monotone chunk index within the episode, from 0
    pub t_emit: u32,     // absolute env step at which index 0 applies
    pub horizon: u16,    // H (PushT: 15)
    pub dim: u16,        // d (PushT: 2)
    pub exec_steps: u16, // S -- the irrevocably committed prefix (PushT: 8)
    pub data: &'a [f64], // len == horizon*dim, row-major: data[i*dim + c]
}

impl<'a> ChunkView<'a> {
    /// Row `i`, clamped to `horizon-1` (never panics in release; debug_assert in debug).
    /// A view with no rows, or whose `data` is too short for the clamped row, yields an empty slice.
    #[inline]
    pub fn action(&self, i: usize) -> &'a [f64] {
        let h = self.horizon as usize;
        let d = self.dim as usize;
        debug_assert!(h == 0 || i < h, "action index {i} >= horizon {h}");
        if h == 0 || d == 0 {
            return &self.data[..0];
        }
        let i = if i < h { i } else { h - 1 };
        let start = i * d;
        match self.data.get(start..start + d) {
            Some(row) => row,
            None => &self.data[..0],
        }
    }

    #[inline]
    pub fn overlap_len(&self) -> usize {
        (self.horizon - self.exec_steps) as usize
    }

    pub fn all_finite(&self) -> bool {
        fmath::all_finite(self.data)
    }
}

/// Everything the fuse is told about the world this tick.
#[derive(Clone, Copy, Debug)]
pub struct ObsView<'a> {
    /// absolute env step
    pub t: u32,
    /// proprioceptive position, len == manifest.pos_dim (mandatory)
    pub pos: &'a [f64],
    /// proprioceptive velocity when manifest.provides_vel (PushT: Some -- gym_pusht info["vel_agent"] every step);
    /// None only for embodiments that provide none
    pub vel: Option<&'a [f64]>,
    /// layout declared by manifest.aux_layout (PushT: block_x, block_y, block_theta, coverage)
    pub aux: &'a [f64],
    /// Tier-2 scalars ext0..ext3 from the policy process; len <= MAX_EXT; may be empty
    pub ext: &'a [f64],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_then_view_round_trips_header_and_rows() {
        let mut b = ChunkBuf::new();
        assert!(b.view().is_none());
        let src = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        b.fill(3, 40, 3, 2, 2, &src).unwrap();
        let v = b.view().unwrap();
        assert_eq!((v.seq, v.t_emit, v.horizon, v.dim, v.exec_steps), (3, 40, 3, 2, 2));
        assert_eq!(v.data, &src);
        assert_eq!(v.action(1), &[3.0, 4.0]);
        assert_eq!(v.overlap_len(), 1);
        b.clear();
        assert!(b.view().is_none());
    }

    #[test]
    fn copy_from_reproduces_the_view() {
        let mut a = ChunkBuf::new();
        a.fill(7, 8, 2, 3, 1, &[0.5, 1.5, 2.5, 3.5, 4.5, 5.5]).unwrap();
        let mut b = ChunkBuf::new();
        b.copy_from(a.view().unwrap());
        let (va, vb) = (a.view().unwrap(), b.view().unwrap());
        assert_eq!(va.data, vb.data);
        assert_eq!(
            (va.seq, va.t_emit, va.horizon, va.dim, va.exec_steps),
            (vb.seq, vb.t_emit, vb.horizon, vb.dim, vb.exec_steps)
        );
    }
}
