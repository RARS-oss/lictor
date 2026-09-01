// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; replace the bodies, keep the signatures.
//! The action-chunk IR. `new`/`view`/`is_filled`/`clear` are written for real (they are trivially-derived
//! accessors over the private fields); `fill`/`copy_from`/`row_mut`/`action`/`all_finite` are WP-1's.

use serde::{Deserialize, Serialize};

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
    pub fn fill(
        &mut self,
        _seq: u32,
        _t_emit: u32,
        _h: u16,
        _d: u16,
        _exec: u16,
        _src: &[f64],
    ) -> Result<(), ChunkError> {
        todo!("WP-1")
    }

    pub fn copy_from(&mut self, _v: ChunkView<'_>) {
        todo!("WP-1")
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

    /// len == dim; i clamped to horizon-1.
    pub fn row_mut(&mut self, _i: usize) -> &mut [f64] {
        todo!("WP-1")
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
    #[inline]
    pub fn action(&self, _i: usize) -> &'a [f64] {
        todo!("WP-1")
    }

    #[inline]
    pub fn overlap_len(&self) -> usize {
        (self.horizon - self.exec_steps) as usize
    }

    pub fn all_finite(&self) -> bool {
        todo!("WP-1")
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
