// SPDX-License-Identifier: MIT
//! OWNER: WP-2. Stub written by WP-0; replace the bodies, keep the signatures.
//! Tier-0 geometric limits and the direction-preserving projection.

use lictor_core::{ChunkBuf, ChunkView, FuseConfig, ObsView};

#[derive(Clone, Copy, Debug, Default)]
pub struct Tier0Out {
    pub trips: u32,
    pub clamped_dims: u32,
    pub peak_speed: f64,
}

/// Whole-chunk check at a chunk boundary. Writes the projected chunk into `out` (== ch when nothing clamped).
/// Exactly `horizon` iterations. Zero allocation.
pub fn check_chunk(
    _cfg: &FuseConfig,
    _obs: ObsView<'_>,
    _ch: ChunkView<'_>,
    _out: &mut ChunkBuf,
) -> Tier0Out {
    todo!("WP-2")
}

/// Intra-chunk per-tick check of the single committed action `a` against predecessor `prev` (box, speed, reach, contact).
/// Writes the (possibly leashed) action into `out[..dim]`.
pub fn check_action(
    _cfg: &FuseConfig,
    _obs: ObsView<'_>,
    _prev: &[f64],
    _a: &[f64],
    _out: &mut [f64],
) -> Tier0Out {
    todo!("WP-2")
}
