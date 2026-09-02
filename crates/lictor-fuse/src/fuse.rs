// SPDX-License-Identifier: MIT
//! `decide()` (the contract), `FuseRt` (pre-allocated runtime state) and the `Fuse` wrapper.
//!
//! `decide` runs the nine ordered stages of docs/ARCHITECTURE.md sec 5.4 (GUARD, INGEST, TIER 0, TIER 1,
//! CONFORMAL, WINDOW, FSM, ACTION, TALLY) over fixed-size arrays. Nothing here allocates, reads a clock, takes
//! a lock, or panics on data: every slice length is re-derived from the inputs, every counter saturates, and
//! every `Option` is matched rather than unwrapped.
//!
//! Chunk buffers. `scratch` always holds the Tier-0 projection of the most recently delivered chunk (== the raw
//! chunk under `ClampMode::Off`); on intra-chunk ticks its row `idx` is overwritten with the leashed committed
//! row. In Enforce mode `cur` receives the projection of every delivered chunk, accepted (no `TIER0_HARD` bit)
//! or rejected: `idx` indexes the most recently delivered chunk on the following intra-chunk ticks, so that is
//! the chunk the per-tick checks must look at, and a rejected chunk never reaches the action (the FSM is in
//! Braking or a later stop state, and every later Armed tick re-checks the row it executes, brake included).
//! 5.4 words the rejected branch as "chunks_rejected++" only; keeping a stale chunk instead would leave the very
//! first tick after a rejected first delivery with nothing to index (a spurious SCHEMA fault). In Observe mode
//! `cur` ALWAYS receives the RAW chunk, accepted or not: `cur.action(idx)` is the pass-through action the verdict
//! must carry on every tick, and the projection in `scratch` only feeds the would-be decision. `Tier1Rt::prev`
//! (the raw chunk Tier 1 compares against) is written by `lictor_detect::tier1::features` alone; `decide` never
//! touches it.
//!
//! GUARD ticks (stage 0). A non-finite, schema-breaking, discontinuous or watchdog-missing tick latches Fault
//! through `fsm::next` row 1 and returns the hold action in BOTH modes: there is no trustworthy policy row to
//! pass through (the chunk itself may be the broken input), so fail-closed applies to Observe as well. Such a
//! verdict carries `action_src = Hold`, `substituted = true` (the action really was substituted),
//! `violation_reached_env = false` (nothing reached the environment unchecked), `scores` at their disarmed
//! values and `brake_margin = NaN` (no brake check ran). The hold is `clamp(last finite position)`: the current
//! `obs.pos` when it is finite, else the position ingested on the previous tick, else the zero vector (whose
//! clamp is the box corner -- on PushT `(17, 17)` px, harmless, and only reachable when the very first tick of
//! an episode is already non-finite). `last_t` and `seq` still advance on a GUARD tick so that the
//! continuity guard of the next tick is evaluated against the tick that actually arrived; `next_chunk_seq` does
//! not (the chunk was never ingested).
//!
//! `substituted` in Enforce mode is `action_src != Policy`: a Policy row is bit-for-bit the raw row (the
//! projection moves a row only when a soft check trips, which puts the FSM in Clamped), and the other three
//! sources are substitutions by construction even on a tick where the projected row happens to coincide with
//! the raw one. `stopped` is `true` in Observe mode (the brake is assumed to have stopped the robot) and
//! `||v_hat|| <= v_stop_eps` in Enforce mode. `v_hat` is `obs.vel` when supplied, else the finite difference
//! `(pos - prev_pos) / dt` (zero on the first tick of an episode).

use lictor_core::{
    fmath, ActionSource, ChunkBuf, ChunkView, EpisodeInit, FuseConfig, FuseMode, FuseState, ObsView,
    ReasonCode, SafetyVerdict, Scores, Status, TripMask, VerifiedAck, MAX_AUX, MAX_D, MAX_EXT, MAX_OPERATORS,
    MAX_POS,
};
use lictor_detect::{
    brake::{brake_action, brake_feasible, hold_action, BrakeOut},
    conformal::{aggregate, standardise, trip},
    tier0::{check_action, check_chunk},
    tier1::{features, Tier1Rt},
    window::Window,
};

use crate::{
    fsm::{self, FsmInput},
    tally::{Tally, TickEvents},
};

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
    /// chunk currently executing (post-projection in Enforce; RAW in Observe, see the module doc)
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
    /// Everything zero, no chunk, `state = Idle` (no episode); `reset()` arms it.
    pub fn new() -> Self {
        Self {
            cur: ChunkBuf::new(),
            scratch: ChunkBuf::new(),
            t1: Tier1Rt::new(),
            window: Window::default(),
            state: FuseState::Idle,
            seq: 0,
            last_t: 0,
            next_chunk_seq: 0,
            pos: [0.0; MAX_POS],
            prev_pos: [0.0; MAX_POS],
            vel: [0.0; MAX_POS],
            have_prev_pos: false,
            last_cmd: [0.0; MAX_D],
            hold: [0.0; MAX_POS],
            clean_run: 0,
            clamp_streak: 0,
            clamps: 0,
            brake_ticks: 0,
            stopped_ticks: 0,
            held_ticks: 0,
            held_clean: 0,
            escalated_ticks: 0,
            rearms: 0,
            handoff_seq: 0,
            handoff_pending: false,
            chunk_ok_this_tick: false,
            last_nonce: [0; MAX_OPERATORS],
            tally: Tally::default(),
        }
    }

    /// clears everything; state = Armed; tally = Tally::default() (max_s/max_z = NEG_INFINITY)
    pub fn reset(&mut self, init: EpisodeInit) {
        // The episode header (index, seed, delay, exec mode) is bound into the receipt by the runtime; nothing
        // in the decision rule depends on it, so it is accepted and not stored.
        let _ = init;
        self.cur.clear();
        self.scratch.clear();
        self.t1.reset();
        self.window.clear();
        self.state = FuseState::Armed;
        self.seq = 0;
        self.last_t = 0;
        self.next_chunk_seq = 0;
        self.pos = [0.0; MAX_POS];
        self.prev_pos = [0.0; MAX_POS];
        self.vel = [0.0; MAX_POS];
        self.have_prev_pos = false;
        self.last_cmd = [0.0; MAX_D];
        self.hold = [0.0; MAX_POS];
        self.clean_run = 0;
        self.clamp_streak = 0;
        self.clamps = 0;
        self.brake_ticks = 0;
        self.stopped_ticks = 0;
        self.held_ticks = 0;
        self.held_clean = 0;
        self.escalated_ticks = 0;
        self.rearms = 0;
        self.handoff_seq = 0;
        self.handoff_pending = false;
        self.chunk_ok_this_tick = false;
        self.last_nonce = [0; MAX_OPERATORS];
        self.tally = Tally::default();
    }
}

impl Default for FuseRt {
    fn default() -> Self {
        Self::new()
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

/// Copies `min(dst.len(), src.len())` values; never panics on a short row.
#[inline]
fn copy_row(dst: &mut [f64], src: &[f64]) {
    let n = min_usize(dst.len(), src.len());
    dst[..n].copy_from_slice(&src[..n]);
}

/// Scores at their disarmed values: nothing valid, `s = NEG_INFINITY`.
#[inline]
fn disarmed_scores() -> Scores {
    Scores { s: f64::NEG_INFINITY, ..Scores::default() }
}

fn status_of(state: FuseState) -> Status {
    match state {
        FuseState::Idle | FuseState::Armed => Status::Nominal,
        FuseState::Watching => Status::Watching,
        FuseState::Clamped => Status::Clamped,
        FuseState::Braking => Status::Braking,
        FuseState::Held => Status::Held,
        FuseState::Escalated => Status::Escalated,
        FuseState::Fault => Status::Fault,
        FuseState::Terminated => Status::Terminated,
    }
}

/// Stage 7's source per post-state (stage 7 of 5.4). `Idle` never survives `reset()`; it holds, fail-closed.
fn source_of(state: FuseState) -> ActionSource {
    match state {
        FuseState::Armed | FuseState::Watching => ActionSource::Policy,
        FuseState::Clamped => ActionSource::Clamped,
        FuseState::Braking => ActionSource::Brake,
        FuseState::Held
        | FuseState::Escalated
        | FuseState::Fault
        | FuseState::Terminated
        | FuseState::Idle => ActionSource::Hold,
    }
}

/// Stage 0: the fail-closed bits of this tick (0 when the tick is admissible).
fn guard(cfg: &FuseConfig, rt: &FuseRt, inp: &TickInput<'_>) -> u32 {
    let obs = inp.obs;
    let t = obs.t;
    let idx = inp.idx as usize;
    let mut bits = 0u32;

    // Non-finite anywhere in the observation or the delivered chunk.
    let vel_bad = match obs.vel {
        Some(v) => !fmath::all_finite(v),
        None => false,
    };
    let chunk_bad = match inp.chunk {
        Some(c) => !c.all_finite(),
        None => false,
    };
    if !fmath::all_finite(obs.pos)
        || vel_bad
        || !fmath::all_finite(obs.aux)
        || !fmath::all_finite(obs.ext)
        || chunk_bad
    {
        bits |= TripMask::NONFINITE;
    }

    // Schema: dims, horizon, idx, the host's own flag, and the two continuity checks.
    let mut schema = inp.schema_fault
        || idx >= cfg.horizon
        || obs.pos.len() != cfg.pos_dim
        || obs.aux.len() > MAX_AUX
        || obs.ext.len() > MAX_EXT;
    if let Some(v) = obs.vel {
        schema |= v.len() != cfg.pos_dim;
    }
    match inp.chunk {
        Some(c) => {
            let h = c.horizon as usize;
            let d = c.dim as usize;
            schema |= d != cfg.dim
                || h != cfg.horizon
                || c.exec_steps as usize != cfg.exec
                || c.data.len() != h * d
                || c.seq != rt.next_chunk_seq
                || c.t_emit > t
                || u32::from(inp.idx) != t - c.t_emit;
        }
        // A non-delivery tick needs a chunk to index into.
        None => schema |= !rt.cur.is_filled(),
    }
    if rt.seq == 0 {
        schema |= t != 0;
    } else {
        let expected = rt.last_t.checked_add(1).and_then(|x| x.checked_add(u32::from(inp.missed_ticks)));
        schema |= expected != Some(t);
    }
    if schema {
        bits |= TripMask::SCHEMA;
    }

    if inp.missed_ticks > cfg.hyst.watchdog_ticks {
        bits |= TripMask::WATCHDOG;
    }
    bits
}

/// The GUARD tick: latch Fault via row 1, hold, return (see the module doc).
fn fault_tick(
    cfg: &FuseConfig,
    rt: &mut FuseRt,
    inp: &TickInput<'_>,
    bits: u32,
    prev_state: FuseState,
) -> SafetyVerdict {
    let obs = inp.obs;
    let dim = min_usize(cfg.dim, MAX_D);
    let pos_dim = min_usize(cfg.pos_dim, MAX_POS);

    // The last finite position is what the hold latches onto.
    if obs.pos.len() == cfg.pos_dim && fmath::all_finite(obs.pos) {
        copy_row(&mut rt.pos[..pos_dim], obs.pos);
    }
    rt.last_t = obs.t;

    let (state, extra, reason) = fsm::next(
        cfg,
        rt,
        FsmInput {
            trips: bits,
            predictive: false,
            warn: false,
            soft_clampable: cfg.clamp == lictor_core::ClampMode::Project,
            stopped: false,
            chunk_boundary: inp.chunk.is_some(),
            chunk_ok: false,
            ack: None,
        },
    );

    let mut action = [0.0; MAX_D];
    hold_action(cfg, &rt.hold[..pos_dim], &mut action[..dim]);

    let v = SafetyVerdict {
        seq: rt.seq,
        t: obs.t,
        status: status_of(state),
        state,
        prev_state,
        trips: bits | extra,
        action,
        action_dim: dim as u16,
        action_src: ActionSource::Hold,
        substituted: true,
        clamped_dims: 0,
        scores: disarmed_scores(),
        tau: cfg.calib.tau,
        window: rt.window.bits,
        window_hits: rt.window.hits(),
        brake_margin: f64::NAN,
        reason,
        handoff_seq: None,
        ack_consumed: false,
        violation_reached_env: false,
    };
    rt.tally.record(&v, TickEvents::default());
    rt.last_cmd = action;
    rt.seq = rt.seq.saturating_add(1);
    v
}

/// THE CONTRACT. Pure: no allocation, no clock read, no lock, no syscall, no panic path, fixed iteration bounds.
/// Same (cfg, rt-before, inp) => bit-identical verdict and rt-after on any IEEE-754 platform.
#[inline(never)]
pub fn decide(cfg: &FuseConfig, rt: &mut FuseRt, inp: &TickInput<'_>) -> SafetyVerdict {
    let obs = inp.obs;
    let t = obs.t;
    let idx = inp.idx as usize;
    let prev_state = rt.state;
    let dim = min_usize(cfg.dim, MAX_D);
    let pos_dim = min_usize(cfg.pos_dim, MAX_POS);
    let observe = cfg.mode == FuseMode::Observe;

    // ---- 0. GUARD (before anything else, including both continuity checks).
    let bits = guard(cfg, rt, inp);
    if bits != 0 {
        return fault_tick(cfg, rt, inp, bits, prev_state);
    }

    // ---- 1. INGEST
    rt.prev_pos = rt.pos;
    copy_row(&mut rt.pos[..pos_dim], obs.pos);
    match obs.vel {
        Some(v) => copy_row(&mut rt.vel[..pos_dim], v),
        None => {
            let mut c = 0;
            while c < pos_dim {
                rt.vel[c] = if rt.have_prev_pos { (rt.pos[c] - rt.prev_pos[c]) * cfg.inv_dt } else { 0.0 };
                c += 1;
            }
        }
    }
    rt.have_prev_pos = true;
    rt.t1.trail.push(&rt.pos, pos_dim);
    rt.last_t = t;
    let chunk_boundary = inp.chunk.is_some();
    if chunk_boundary {
        rt.next_chunk_seq = rt.next_chunk_seq.wrapping_add(1);
    }

    // ---- 2. TIER 0
    let mut trips0;
    let clamped_dims;
    let brake: BrakeOut;
    match inp.chunk {
        Some(ch) => {
            let t0 = check_chunk(cfg, obs, ch, &mut rt.scratch);
            trips0 = t0.trips;
            clamped_dims = t0.clamped_dims;
            // The brake check runs on the projected chunk from the row executed THIS tick.
            brake = match rt.scratch.view() {
                Some(v) => brake_feasible(cfg, &rt.pos[..pos_dim], &rt.vel[..pos_dim], v, idx),
                None => BrakeOut { feasible: false, margin: f64::NEG_INFINITY, stop_dist: 0.0 },
            };
            if !brake.feasible {
                trips0 |= TripMask::BRAKE;
            }
            trips0 &= cfg.tier0_enabled | TripMask::TIER0_HARD;
            let rejected = trips0 & TripMask::TIER0_HARD != 0;
            if observe {
                // Pass-through source: the RAW chunk, accepted or not.
                rt.cur.copy_from(ch);
            } else if let Some(v) = rt.scratch.view() {
                // The projection, accepted or rejected: `idx` indexes THIS chunk on the following ticks (see
                // the module doc); a rejected chunk never reaches the action while the FSM is stopping.
                rt.cur.copy_from(v);
            }
            rt.tally.record_chunk(rejected);
        }
        None => {
            // Defensive: `cur` is filled (GUARD), and `scratch` was filled by the same delivery; keep them
            // shape-consistent even if a caller rebuilt `rt` by hand.
            if !rt.scratch.is_filled() {
                if let Some(v) = rt.cur.view() {
                    rt.scratch.copy_from(v);
                }
            }
            let (tr, cd, br) = match rt.cur.view() {
                Some(cv) => {
                    let t0 =
                        check_action(cfg, obs, &rt.last_cmd[..dim], cv.action(idx), rt.scratch.row_mut(idx));
                    let br = brake_feasible(cfg, &rt.pos[..pos_dim], &rt.vel[..pos_dim], cv, idx);
                    (t0.trips, t0.clamped_dims, br)
                }
                None => (0, 0, BrakeOut { feasible: false, margin: f64::NEG_INFINITY, stop_dist: 0.0 }),
            };
            trips0 = tr;
            clamped_dims = cd;
            brake = br;
            if !brake.feasible {
                trips0 |= TripMask::BRAKE;
            }
            trips0 &= cfg.tier0_enabled | TripMask::TIER0_HARD;
        }
    }

    // ---- 3. TIER 1 (features() is the sole writer of t1.prev: raw-vs-raw TCE in both modes)
    let mut sc = disarmed_scores();
    if cfg.calib.armed() {
        features(cfg, &mut rt.t1, obs, inp.chunk, inp.idx, &mut sc);
    }

    // ---- 4. CONFORMAL
    standardise(&cfg.calib, t, &mut sc);
    let s = aggregate(&cfg.calib, &mut sc);
    let hit = trip(&cfg.calib, &sc);
    let warn = !hit && s > cfg.calib.tau - cfg.hyst.warn_margin;

    // ---- 5. WINDOW
    rt.window.push(hit, cfg.window_mask);
    let window_hits = rt.window.hits();
    let predictive = window_hits >= cfg.hyst.k;

    // ---- 6. FSM
    let stopped = observe || fmath::norm(&rt.vel, pos_dim) <= cfg.hyst.v_stop_eps;
    let chunk_ok = chunk_boundary && trips0 == 0;
    rt.chunk_ok_this_tick = chunk_ok;
    // Candidate rows, copied out before the FSM takes `rt` as a whole.
    let mut policy_row = [0.0; MAX_D];
    let mut clamped_row = [0.0; MAX_D];
    if let Some(v) = rt.cur.view() {
        copy_row(&mut policy_row[..dim], v.action(idx));
    }
    if let Some(v) = rt.scratch.view() {
        copy_row(&mut clamped_row[..dim], v.action(idx));
    }
    let clamps_before = rt.clamps;
    let rearms_before = rt.rearms;
    let (state, extra, reason) = fsm::next(
        cfg,
        rt,
        FsmInput {
            trips: trips0,
            predictive,
            warn,
            soft_clampable: cfg.clamp == lictor_core::ClampMode::Project,
            stopped,
            chunk_boundary,
            chunk_ok,
            ack: inp.ack,
        },
    );
    let trips = trips0 | extra;

    // ---- 7. ACTION
    let would_src = source_of(state);
    let mut action = [0.0; MAX_D];
    let (action_src, substituted, violation_reached_env) = if observe {
        action = policy_row;
        (ActionSource::Policy, false, would_src != ActionSource::Policy)
    } else {
        match would_src {
            ActionSource::Policy => action = policy_row,
            ActionSource::Clamped => action = clamped_row,
            ActionSource::Brake => {
                brake_action(cfg, &rt.pos[..pos_dim], &rt.vel[..pos_dim], &mut action[..dim])
            }
            ActionSource::Hold => hold_action(cfg, &rt.hold[..pos_dim], &mut action[..dim]),
        }
        (would_src, would_src != ActionSource::Policy, false)
    };

    let entered_hold = state == FuseState::Held && prev_state != FuseState::Held;
    let entered_escalated = state == FuseState::Escalated && prev_state != FuseState::Escalated;
    let ack_consumed = inp.ack.is_some()
        && matches!(
            reason,
            ReasonCode::RearmedAck | ReasonCode::TerminatedAbort | ReasonCode::TerminatedRetune
        );

    let v = SafetyVerdict {
        seq: rt.seq,
        t,
        status: status_of(state),
        state,
        prev_state,
        trips,
        action,
        action_dim: dim as u16,
        action_src,
        substituted,
        clamped_dims,
        scores: sc,
        tau: cfg.calib.tau,
        window: rt.window.bits,
        window_hits,
        brake_margin: brake.margin,
        reason,
        handoff_seq: if entered_escalated { Some(rt.handoff_seq) } else { None },
        ack_consumed,
        violation_reached_env,
    };

    // ---- 8. TALLY
    rt.tally.record(
        &v,
        TickEvents {
            entered_hold,
            entered_escalated,
            rearmed: rt.rearms != rearms_before,
            clamped: rt.clamps != clamps_before,
            live_z: cfg.calib.mask & sc.valid,
        },
    );
    rt.last_cmd = action;
    rt.seq = rt.seq.saturating_add(1);
    v
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

    /// The per-episode tally with `terminal_state` = the state the fuse is in now.
    pub fn finish(&self) -> Tally {
        let mut t = self.rt.tally;
        t.terminal_state = self.rt.state;
        t
    }
}
