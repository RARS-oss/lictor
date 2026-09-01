// SPDX-License-Identifier: MIT
//! OWNER: WP-1. Stub written by WP-0; replace the bodies, keep the signatures.
//! The safety configuration: embodiment manifest, limits, brake model, hysteresis, gate, and the compiled `FuseConfig`.

use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

use crate::{
    chunk::{ActionKind, MAX_D, MAX_POS},
    scores::CalibrationC,
    state::FuseMode,
};

pub const MAX_TERMS: usize = 8;
pub const MAX_OPERATORS: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentManifest {
    /// "gym_pusht/PushT-v0"
    pub id: String,
    /// "px" | "rad" | "m" -- documentation, bound into the digest
    pub units: String,
    /// ee_position
    pub action_kind: ActionKind,
    /// 2
    pub action_dim: u16,
    /// 2
    pub pos_dim: u16,
    /// 15   (the chunk length the adapter DELIVERS)
    pub horizon: u16,
    /// 8    (the committed prefix)
    pub exec_steps: u16,
    /// 10   control period dt = den/num seconds, exact rational
    pub control_hz_num: u32,
    /// 1
    pub control_hz_den: u32,
    /// true (PushT): the harness sends the simulator velocity as obs.vel; false -> finite-difference v_hat
    /// (recorded as a fuse_note)
    pub provides_vel: bool,
    /// 300  episode time base for calibration binning; MUST equal the env's max_episode_steps
    /// (checked by `lictor calibrate`)
    pub horizon_ticks: u32,
    /// len == action_dim; Tier-1 uses abar = (a - center)/scale
    pub norm_center: Vec<f64>,
    /// len == action_dim; all > 0
    pub norm_scale: Vec<f64>,
    /// ["block_x","block_y","block_theta","coverage"]
    pub aux_layout: Vec<String>,
    /// <= 4 names for ext0..ext3; [] when unused
    pub ext_names: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrakeKind {
    /// Exact forward rollout of the env's 2nd-order PD agent (gym-pusht: k_p=100, k_v=20, dt=0.01, 10 substeps).
    PdSecondOrder,
    /// Closed form d_stop = ||v|| / k_v (setpoint-held first-order decay).
    FirstOrderDecay,
    /// Closed form d_stop = ||v||^2 / (2 a_max).
    BoundedAccel,
    /// Closed form d_stop = ||v||^2/(2 a_max) + ||v|| a_max/(2 j_max) (Ruckig two-phase upper bound).
    JerkLimited,
    /// Velocity-command embodiments: brake = ramp to zero; feasibility uses BoundedAccel.
    ZeroVelocityHold,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrakeModel {
    pub kind: BrakeKind,
    /// 100.0 (PushT)
    pub k_p: f64,
    /// 20.0
    pub k_v: f64,
    /// 10 -- must equal the env's inner loop
    pub substeps: u16,
    /// 0.01 -- physics substep, NOT the control period
    pub dt: f64,
    /// 8 -- irrevocable prefix simulated forward
    pub commit_steps: u16,
    /// 8 -- extra hold steps simulated after the prefix (PdSecondOrder only)
    pub brake_steps: u16,
    /// 1 -- dead time before a brake bites (closed-form kinds only)
    pub react_ticks: u16,
}

/// `aux_center`: aux indices of the object centre.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactLimit {
    pub radius: f64,
    pub v_max: f64,
    pub aux_center: [u8; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampMode {
    Off,
    Project,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RearmPolicy {
    Auto,
    AckOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hysteresis {
    /// 3   K-of-N on the predictive channel
    pub k: u8,
    /// 5   1 <= K <= N <= 63 (window_mask = (1 << n) - 1 must fit in u64)
    pub n: u8,
    /// 0.5 (z units): s > tau - warn_margin -> Watching
    pub warn_margin: f64,
    /// 5   clean ticks to leave Watching/Clamped
    pub clear_ticks: u8,
    /// 3   consecutive clamped CHUNKS -> Braking
    pub clamp_streak_to_brake: u8,
    /// 60
    pub max_clamps_per_episode: u16,
    /// 2   Braking -> Held once ||v|| <= v_stop_eps this many ticks
    pub stop_confirm_ticks: u8,
    /// 5.0 units/s
    pub v_stop_eps: f64,
    /// 30  Braking that never stops -> Fault
    pub brake_timeout_ticks: u16,
    /// auto (PushT default) | ack_only
    pub rearm: RearmPolicy,
    /// 10  clean ticks in Held before auto re-arm
    pub rearm_hold: u16,
    /// 2   auto re-arms per episode
    pub max_rearms: u8,
    /// 30  Held this long -> Escalated (handoff)
    pub escalate_after_hold_ticks: u16,
    /// 200 Escalated with no ack -> Terminated
    pub handoff_timeout_ticks: u32,
    /// 2   missed_ticks above this -> Fault
    pub watchdog_ticks: u8,
}

/// Fixed-order DNF over Tier-1 features: s = max_i min_{j in terms[i]} z_j. A singleton term is a plain channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateSpec {
    pub n_terms: u8,
    pub terms: [u32; MAX_TERMS],
}

impl GateSpec {
    pub const DISARMED: Self = Self { n_terms: 0, terms: [0; MAX_TERMS] };

    /// Parse ["tce","acc|path_ineff"] -> terms (each '|' = AND within a term). Unknown name -> Err.
    pub fn parse(_terms: &[String]) -> Result<Self, EnvelopeError> {
        todo!("WP-1")
    }

    /// OR of all terms.
    pub fn mask(&self) -> u32 {
        todo!("WP-1")
    }

    pub fn names(&self) -> Vec<String> {
        todo!("WP-1")
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitRecord {
    pub source_run: String,
    pub quantile: f64,
    pub slack: f64,
    pub n_episodes: u32,
    pub fitted_utc: String,
    pub note: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    /// "lictor-envelope/v1"
    pub schema: String,
    /// "pusht-base-v1"
    pub envelope_id: String,
    pub embodiment: EmbodimentManifest,
    /// len == pos_dim, env units
    pub box_lo: Vec<f64>,
    pub box_hi: Vec<f64>,
    /// subtracted from the box before EVERY check
    pub margin: f64,
    /// units/s
    pub v_max: f64,
    /// units/s^2
    pub a_max: f64,
    /// units/s^3
    pub j_max: f64,
    /// units: ||a_0 - p_t||
    pub reach_max: f64,
    pub contact: Option<ContactLimit>,
    pub brake: BrakeModel,
    pub clamp_mode: ClampMode,
    pub hysteresis: Hysteresis,
    /// subset of ["workspace","speed","accel","jerk","reach","contact","brake"]; nonfinite/schema/watchdog always on
    pub tier0_enabled: Vec<String>,
    /// default gate terms, overridable by calibration.json
    pub gate: Vec<String>,
    /// Ed25519 pubkeys (hex64) allowed to sign an AckToken; slot = index
    pub operators: Vec<String>,
    /// present when produced by `lictor envelope fit`
    pub fit: Option<FitRecord>,
    /// MUST be true in v1; recorded so a future `false` is auditable
    pub fail_closed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError {
    Parse(String),
    Invalid(String),
    Unsupported(String),
}

impl SafetyEnvelope {
    #[cfg(feature = "std")]
    pub fn from_toml(_s: &str) -> Result<Self, EnvelopeError> {
        todo!("WP-1")
    }

    #[cfg(feature = "std")]
    pub fn to_toml(&self) -> Result<String, EnvelopeError> {
        todo!("WP-1")
    }

    /// sha256 over lictor_canon::canon(floatify(to_value(self))). std only.
    #[cfg(feature = "std")]
    pub fn digest_hex(&self) -> String {
        todo!("WP-1")
    }

    /// sha256 over canon(floatify(to_value(&self.embodiment))) -- the sub-digest a calibration binds to. Tier-1 features
    /// depend on the EmbodimentManifest ONLY (never on v_max/a_max/j_max/reach_max/operators), so `envelope fit` and an
    /// operator-list change do not invalidate a calibration; a manifest change does. std only.
    #[cfg(feature = "std")]
    pub fn embodiment_digest(&self) -> String {
        todo!("WP-1")
    }

    /// dims, positivity, 1 <= k <= n <= 63, <= MAX_OPERATORS, horizon_ticks > 0, fail_closed == true
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        todo!("WP-1")
    }

    pub fn dt(&self) -> f64 {
        self.embodiment.control_hz_den as f64 / self.embodiment.control_hz_num as f64
    }

    /// One-shot, off-loop: validate + flatten Vec -> fixed arrays + precompute derived constants.
    pub fn compile(
        &self,
        _mode: FuseMode,
        _calib: Option<CalibrationC>,
    ) -> Result<FuseConfig, EnvelopeError> {
        todo!("WP-1")
    }

    pub fn tier0_mask(&self) -> Result<u32, EnvelopeError> {
        todo!("WP-1")
    }
}

/// The immutable, hot-path-ready configuration. No Vec, no String; every derived constant baked.
/// `envelope_digest` / `embodiment_digest` are filled by `compile` under `std` (via `digest_hex` / `embodiment_digest`);
/// under no_std they are `[0; 32]` (the MCU path carries digests out-of-band -- roadmap 9).
#[derive(Clone, Debug)]
pub struct FuseConfig {
    pub envelope_digest: [u8; 32],
    pub embodiment_digest: [u8; 32],
    pub calib_digest: Option<[u8; 32]>,
    pub mode: FuseMode,
    pub kind: ActionKind,
    pub dim: usize,
    pub pos_dim: usize,
    pub horizon: usize,
    pub exec: usize,
    pub overlap: usize,
    pub dt: f64,
    pub inv_dt: f64,
    pub inv_dt2: f64,
    pub inv_dt3: f64,
    pub norm_center: [f64; MAX_D],
    pub inv_norm_scale: [f64; MAX_D],
    pub norm_scale_iso: f64,
    /// ALREADY margin-adjusted
    pub box_lo: [f64; MAX_POS],
    pub box_hi: [f64; MAX_POS],
    pub v_max: f64,
    pub a_max: f64,
    pub j_max: f64,
    pub reach_max: f64,
    /// v_max * dt -- the per-step leash
    pub step_max: f64,
    pub contact: Option<ContactLimit>,
    pub brake: BrakeModel,
    pub clamp: ClampMode,
    pub hyst: Hysteresis,
    /// (1 << n) - 1, n <= 63
    pub window_mask: u64,
    /// TripMask bits armed
    pub tier0_enabled: u32,
    pub n_operators: u8,
    /// CalibrationC::DISARMED when Tier 1 is off
    pub calib: CalibrationC,
}
