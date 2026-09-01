// SPDX-License-Identifier: MIT
//! OWNER: WP-3. Stub written by WP-0; replace the bodies, keep the signatures.
//! `decide()` (the contract), `FuseRt` (pre-allocated runtime state) and the `Fuse` wrapper.
//! The `Fuse` wrapper methods are written for real (they only forward to `FuseRt` / `decide`).

use lictor_core::{
    ChunkBuf, ChunkView, EpisodeInit, FuseConfig, FuseState, ObsView, SafetyVerdict, VerifiedAck, MAX_D,
    MAX_OPERATORS, MAX_POS,
};
use lictor_detect::{tier1::Tier1Rt, window::Window};

use crate::tally::Tally;

pub struct TickInput<'a> {
    pub obs: ObsView<'a>,
    /// Some ONLY on the tick a new chunk arrives
    pub chunk: Option<ChunkView<'a>>,
    /// index into the CURRENT chunk for this tick
    pub idx: u16,
    /// the ONLY time-like input; an integer, supplied by the host
    pub missed_ticks: u8,
    pub ack: Option<VerifiedAck>,
    /// set by the runtime on a wire/schema violation -> TripMask::SCHEMA in GUARD
    pub schema_fault: bool,
}

/// Pre-allocated runtime state (~58 KB: three 16.5 KB ChunkBufs + an 8 KB Trail; `lictor bench` prints the exact size_of).
/// Constructed once per process; `reset()` per episode. Never allocates.
pub struct FuseRt {
    /// chunk currently executing (post-projection in Enforce; RAW in Observe, see WP-3)
    pub cur: ChunkBuf,
    /// projection target
    pub scratch: ChunkBuf,
    pub t1: Tier1Rt,
    pub window: Window,
    pub state: FuseState,
    pub seq: u32,
    /// obs.t of the previous tick (time-continuity guard: t == last_t + 1 + missed_ticks; first tick t == 0)
    pub last_t: u32,
    /// the seq the next delivered chunk MUST carry (0 after reset; chunk.seq continuity guard)
    pub next_chunk_seq: u32,
    pub pos: [f64; MAX_POS],
    pub prev_pos: [f64; MAX_POS],
    pub vel: [f64; MAX_POS],
    pub have_prev_pos: bool,
    /// last action actually emitted
    pub last_cmd: [f64; MAX_D],
    /// latched hold setpoint (Held/Escalated/Fault/Terminated)
    pub hold: [f64; MAX_POS],
    pub clean_run: u16,
    pub clamp_streak: u8,
    pub clamps: u16,
    pub brake_ticks: u16,
    pub stopped_ticks: u8,
    pub held_ticks: u16,
    pub held_clean: u16,
    pub escalated_ticks: u32,
    pub rearms: u8,
    pub handoff_seq: u32,
    pub handoff_pending: bool,
    pub chunk_ok_this_tick: bool,
    pub last_nonce: [u64; MAX_OPERATORS],
    pub tally: Tally,
}

impl FuseRt {
    pub fn new() -> Self {
        todo!("WP-3")
    }

    /// clears everything; state = Armed; tally = Tally::default() (max_s/max_z = NEG_INFINITY)
    pub fn reset(&mut self, _init: EpisodeInit) {
        todo!("WP-3")
    }
}

impl Default for FuseRt {
    fn default() -> Self {
        Self::new()
    }
}

/// THE CONTRACT. Pure: no allocation, no clock read, no lock, no syscall, no panic path, fixed iteration bounds.
/// Same (cfg, rt-before, inp) => bit-identical verdict and rt-after on any IEEE-754 platform.
#[inline(never)]
pub fn decide(_cfg: &FuseConfig, _rt: &mut FuseRt, _inp: &TickInput<'_>) -> SafetyVerdict {
    todo!("WP-3")
}

pub struct Fuse {
    cfg: FuseConfig,
    rt: FuseRt,
}

impl Fuse {
    pub fn new(cfg: FuseConfig) -> Self {
        Self { cfg, rt: FuseRt::new() }
    }

    pub fn cfg(&self) -> &FuseConfig {
        &self.cfg
    }

    pub fn rt(&self) -> &FuseRt {
        &self.rt
    }

    pub fn reset(&mut self, init: EpisodeInit) {
        self.rt.reset(init)
    }

    #[inline]
    pub fn step(&mut self, inp: &TickInput<'_>) -> SafetyVerdict {
        decide(&self.cfg, &mut self.rt, inp)
    }

    pub fn finish(&self) -> Tally {
        self.rt.tally
    }
}
