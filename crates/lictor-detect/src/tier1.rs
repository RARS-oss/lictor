// SPDX-License-Identifier: MIT
//! OWNER: WP-2. Stub written by WP-0; replace the bodies, keep the signatures.
//! Tier-1 features (every feature depends on the EmbodimentManifest only).

use lictor_core::{ChunkBuf, ChunkView, FuseConfig, ObsView, Scores, NFEAT};

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
        todo!("WP-2")
    }

    pub fn reset(&mut self) {
        todo!("WP-2")
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

/// Updates `rt` and writes raw features into `sc.f` / `sc.valid`. Chunk-boundary features recompute only when `chunk.is_some()`.
/// CONTRACT: `decide()` pushes `rt.trail` BEFORE calling this. `features` is the SOLE writer of `rt.prev`/`rt.have_prev`: after
/// computing tce/acc it copies the RAW incoming chunk into `rt.prev`, so TCE is always raw-vs-raw in Observe and Enforce alike
/// (calibration features == enforcement features). Every feature depends on the EmbodimentManifest only.
pub fn features(
    _cfg: &FuseConfig,
    _rt: &mut Tier1Rt,
    _obs: ObsView<'_>,
    _chunk: Option<ChunkView<'_>>,
    _idx: u16,
    _sc: &mut Scores,
) {
    todo!("WP-2")
}
