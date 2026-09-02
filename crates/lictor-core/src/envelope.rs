// SPDX-License-Identifier: MIT
//! The safety configuration: embodiment manifest, limits, brake model, hysteresis, gate, and the compiled
//! `FuseConfig`. TOML in/out and the canonical digests are `std`-only (`toml`, `serde_json`, `lictor_canon`);
//! `validate`, `tier0_mask`, `GateSpec` and `compile` are `no_std` and allocation-free apart from error strings.

use alloc::{format, string::String, vec::Vec};
use serde::{Deserialize, Serialize};

use crate::{
    chunk::{ActionKind, MAX_AUX, MAX_D, MAX_EXT, MAX_H, MAX_POS},
    fmath,
    scores::{CalibrationC, Feat, T_GRID},
    state::FuseMode,
    verdict::TripMask,
};

pub const MAX_TERMS: usize = 8;
pub const MAX_OPERATORS: usize = 8;

/// The only envelope schema this build reads or writes.
const SCHEMA: &str = "lictor-envelope/v1";
/// Relative tolerance for `substeps * dt == control period` (`PdSecondOrder` only).
const PERIOD_RTOL: f64 = 1e-9;
/// The Tier-0 bits an envelope may arm by name (`nonfinite`/`schema`/`watchdog` are always on).
const ARMABLE: u32 = TripMask::TIER0_SOFT | TripMask::BRAKE;

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
    /// Names are exact (`Feat::NAMES`, no trimming); an empty list parses to `DISARMED`.
    pub fn parse(terms: &[String]) -> Result<Self, EnvelopeError> {
        if terms.len() > MAX_TERMS {
            return Err(inv(format!("gate: {} terms exceed MAX_TERMS = {MAX_TERMS}", terms.len())));
        }
        let mut g = Self::DISARMED;
        for (i, term) in terms.iter().enumerate() {
            let mut bits = 0u32;
            for name in term.split('|') {
                match Feat::from_name(name) {
                    Some(f) => bits |= f.bit(),
                    None => return Err(inv(format!("gate term {i}: unknown feature `{name}`"))),
                }
            }
            g.terms[i] = bits;
        }
        g.n_terms = terms.len() as u8;
        Ok(g)
    }

    /// OR of all terms.
    pub fn mask(&self) -> u32 {
        self.terms[..self.active_terms()].iter().fold(0, |m, t| m | t)
    }

    /// One `a|b|c` string per term, channels in `Feat` id order.
    pub fn names(&self) -> Vec<String> {
        self.terms[..self.active_terms()]
            .iter()
            .map(|bits| {
                let parts: Vec<&str> =
                    Feat::ALL.iter().filter(|f| bits & f.bit() != 0).map(|f| f.name()).collect();
                parts.join("|")
            })
            .collect()
    }

    fn active_terms(&self) -> usize {
        let n = self.n_terms as usize;
        if n < MAX_TERMS {
            n
        } else {
            MAX_TERMS
        }
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

impl core::fmt::Display for EnvelopeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            EnvelopeError::Parse(m) => write!(f, "envelope parse error: {m}"),
            EnvelopeError::Invalid(m) => write!(f, "envelope invalid: {m}"),
            EnvelopeError::Unsupported(m) => write!(f, "envelope unsupported: {m}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EnvelopeError {}

fn inv(msg: String) -> EnvelopeError {
    EnvelopeError::Invalid(msg)
}

fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

fn nonneg(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

fn is_hex64_lower(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

impl EmbodimentManifest {
    fn validate(&self) -> Result<(), EnvelopeError> {
        if self.id.is_empty() {
            return Err(inv("embodiment.id is empty".into()));
        }
        let d = self.action_dim as usize;
        let pd = self.pos_dim as usize;
        if d == 0 || d > MAX_D {
            return Err(inv(format!("embodiment.action_dim {d} is not in 1..={MAX_D}")));
        }
        if pd == 0 || pd > MAX_POS {
            return Err(inv(format!("embodiment.pos_dim {pd} is not in 1..={MAX_POS}")));
        }
        if d != pd {
            return Err(inv(format!(
                "embodiment.action_dim {d} != pos_dim {pd} (the brake rollout maps action rows onto positions)"
            )));
        }
        if self.horizon == 0 || self.horizon as usize > MAX_H {
            return Err(inv(format!("embodiment.horizon {} is not in 1..={MAX_H}", self.horizon)));
        }
        if self.exec_steps == 0 || self.exec_steps > self.horizon {
            return Err(inv(format!(
                "embodiment.exec_steps {} must satisfy 1 <= exec_steps <= horizon {}",
                self.exec_steps, self.horizon
            )));
        }
        if self.control_hz_num == 0 || self.control_hz_den == 0 {
            return Err(inv("embodiment.control_hz_num and control_hz_den must be > 0".into()));
        }
        if self.horizon_ticks == 0 {
            return Err(inv("embodiment.horizon_ticks must be > 0".into()));
        }
        if self.norm_center.len() != d || self.norm_scale.len() != d {
            return Err(inv(format!("embodiment.norm_center/norm_scale must have action_dim = {d} entries")));
        }
        if let Some(c) = self.norm_center.iter().position(|x| !x.is_finite()) {
            return Err(inv(format!("embodiment.norm_center[{c}] is not finite")));
        }
        if let Some(c) = self.norm_scale.iter().position(|x| !positive(*x)) {
            return Err(inv(format!("embodiment.norm_scale[{c}] must be finite and > 0")));
        }
        if self.aux_layout.len() > MAX_AUX {
            return Err(inv(format!("embodiment.aux_layout has more than MAX_AUX = {MAX_AUX} entries")));
        }
        if self.ext_names.len() > MAX_EXT {
            return Err(inv(format!("embodiment.ext_names has more than MAX_EXT = {MAX_EXT} entries")));
        }
        Ok(())
    }
}

impl SafetyEnvelope {
    /// Parse the TOML text. Unknown keys (top level or inside `[embodiment] [brake] [contact] [hysteresis] [fit]`)
    /// are a `Parse` error. Does NOT validate: call `validate` (or `compile`, which validates) before use.
    #[cfg(feature = "std")]
    pub fn from_toml(s: &str) -> Result<Self, EnvelopeError> {
        let table: toml::Table =
            s.parse().map_err(|e: toml::de::Error| EnvelopeError::Parse(format!("{e}")))?;
        check_unknown_keys(&table)?;
        table.try_into().map_err(|e: toml::de::Error| EnvelopeError::Parse(format!("{e}")))
    }

    /// Render as TOML (tables `[embodiment] [brake] [hysteresis]` and, when present, `[contact] [fit]`).
    /// `from_toml(to_toml(e)) == e` for every validated envelope.
    #[cfg(feature = "std")]
    pub fn to_toml(&self) -> Result<String, EnvelopeError> {
        toml::to_string(self).map_err(|e| inv(format!("to_toml: {e}")))
    }

    /// sha256 over lictor_canon::canon(floatify(to_value(self))). std only.
    #[cfg(feature = "std")]
    pub fn digest_hex(&self) -> String {
        canonical_digest(self)
    }

    /// sha256 over canon(floatify(to_value(&self.embodiment))) -- the sub-digest a calibration binds to. Tier-1 features
    /// depend on the EmbodimentManifest ONLY (never on v_max/a_max/j_max/reach_max/operators), so `envelope fit` and an
    /// operator-list change do not invalidate a calibration; a manifest change does. std only.
    #[cfg(feature = "std")]
    pub fn embodiment_digest(&self) -> String {
        canonical_digest(&self.embodiment)
    }

    /// dims, positivity, 1 <= k <= n <= 63, <= MAX_OPERATORS, horizon_ticks > 0, fail_closed == true
    pub fn validate(&self) -> Result<(), EnvelopeError> {
        if self.schema != SCHEMA {
            return Err(EnvelopeError::Unsupported(format!(
                "schema `{}` (this build reads `{SCHEMA}`)",
                self.schema
            )));
        }
        if self.envelope_id.is_empty() {
            return Err(inv("envelope_id is empty".into()));
        }
        if !self.fail_closed {
            return Err(inv("fail_closed must be true in lictor-envelope/v1".into()));
        }
        self.embodiment.validate()?;
        let pd = self.embodiment.pos_dim as usize;
        if self.box_lo.len() != pd || self.box_hi.len() != pd {
            return Err(inv(format!("box_lo/box_hi must have pos_dim = {pd} entries")));
        }
        if !nonneg(self.margin) {
            return Err(inv("margin must be finite and >= 0".into()));
        }
        for (c, (lo, hi)) in self.box_lo.iter().zip(&self.box_hi).enumerate() {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(inv(format!("box dim {c}: box_lo/box_hi must be finite")));
            }
            if lo + self.margin >= hi - self.margin {
                return Err(inv(format!(
                    "box dim {c}: box_lo + margin ({}) must be < box_hi - margin ({})",
                    lo + self.margin,
                    hi - self.margin
                )));
            }
        }
        for (name, x) in [
            ("v_max", self.v_max),
            ("a_max", self.a_max),
            ("j_max", self.j_max),
            ("reach_max", self.reach_max),
        ] {
            if !positive(x) {
                return Err(inv(format!("{name} must be finite and > 0")));
            }
        }
        if let Some(cl) = &self.contact {
            if !positive(cl.radius) || !positive(cl.v_max) {
                return Err(inv("contact.radius and contact.v_max must be finite and > 0".into()));
            }
            let n_aux = self.embodiment.aux_layout.len();
            if cl.aux_center.iter().any(|i| *i as usize >= n_aux) {
                return Err(inv(format!(
                    "contact.aux_center {:?} indexes outside aux_layout (len {n_aux})",
                    cl.aux_center
                )));
            }
        }
        self.validate_brake()?;
        let h = &self.hysteresis;
        if h.k == 0 || h.k > h.n || h.n > 63 {
            return Err(inv(format!("hysteresis: need 1 <= k <= n <= 63, got k = {}, n = {}", h.k, h.n)));
        }
        if !nonneg(h.warn_margin) {
            return Err(inv("hysteresis.warn_margin must be finite and >= 0".into()));
        }
        if !nonneg(h.v_stop_eps) {
            return Err(inv("hysteresis.v_stop_eps must be finite and >= 0".into()));
        }
        let t0 = self.tier0_mask()?;
        if t0 & TripMask::CONTACT != 0 && self.contact.is_none() {
            return Err(inv("tier0_enabled lists `contact` but there is no [contact] table".into()));
        }
        GateSpec::parse(&self.gate)?;
        if self.operators.len() > MAX_OPERATORS {
            return Err(inv(format!(
                "{} operators exceed MAX_OPERATORS = {MAX_OPERATORS}",
                self.operators.len()
            )));
        }
        for (i, op) in self.operators.iter().enumerate() {
            if !is_hex64_lower(op) {
                return Err(inv(format!("operators[{i}]: expected 64 lowercase hex chars")));
            }
            if self.operators[..i].contains(op) {
                return Err(inv(format!("operators[{i}] duplicates an earlier key")));
            }
        }
        if let Some(fit) = &self.fit {
            if !fit.quantile.is_finite() || fit.quantile <= 0.0 || fit.quantile > 1.0 {
                return Err(inv("fit.quantile must be in (0, 1]".into()));
            }
            if !positive(fit.slack) {
                return Err(inv("fit.slack must be finite and > 0".into()));
            }
        }
        Ok(())
    }

    fn validate_brake(&self) -> Result<(), EnvelopeError> {
        let b = &self.brake;
        if !positive(b.k_p) || !positive(b.k_v) {
            return Err(inv("brake.k_p and brake.k_v must be finite and > 0".into()));
        }
        if !positive(b.dt) {
            return Err(inv("brake.dt must be finite and > 0".into()));
        }
        if b.substeps == 0 {
            return Err(inv("brake.substeps must be >= 1".into()));
        }
        if b.commit_steps == 0 || b.commit_steps > self.embodiment.horizon {
            return Err(inv(format!(
                "brake.commit_steps {} must satisfy 1 <= commit_steps <= horizon {}",
                b.commit_steps, self.embodiment.horizon
            )));
        }
        let kind = self.embodiment.action_kind;
        let position_kind = matches!(kind, ActionKind::EePosition | ActionKind::JointPosition);
        let velocity_kind = matches!(kind, ActionKind::JointVelocity | ActionKind::EeDelta);
        match b.kind {
            BrakeKind::PdSecondOrder => {
                if !position_kind {
                    return Err(inv(
                        "brake.kind = pd_second_order needs action_kind ee_position or joint_position".into(),
                    ));
                }
                let period = self.dt();
                let inner = (b.substeps as f64) * b.dt;
                if fmath::abs(inner - period) > PERIOD_RTOL * period {
                    return Err(inv(format!(
                        "brake: substeps * dt = {inner} must equal the control period {period}"
                    )));
                }
            }
            BrakeKind::ZeroVelocityHold => {
                if !velocity_kind {
                    return Err(inv(
                        "brake.kind = zero_velocity_hold needs action_kind joint_velocity or ee_delta".into(),
                    ));
                }
            }
            BrakeKind::FirstOrderDecay | BrakeKind::BoundedAccel | BrakeKind::JerkLimited => {}
        }
        Ok(())
    }

    pub fn dt(&self) -> f64 {
        self.embodiment.control_hz_den as f64 / self.embodiment.control_hz_num as f64
    }

    /// One-shot, off-loop: validate + flatten Vec -> fixed arrays + precompute derived constants.
    ///
    /// Tier 1: when `calib` is `Some` it is embedded as is (its `gate`/`mask`/`tau` win; it must agree with the
    /// manifest's `horizon_ticks`). When `None`, the envelope's own `gate` is embedded on top of
    /// `CalibrationC::DISARMED` (center 0, scale 1, `tau = +inf`): features are computed and recorded (the
    /// observe-mode calibration traces need them) but the predictive channel can never fire. An empty `gate`
    /// yields exactly `CalibrationC::DISARMED` (`armed() == false`).
    pub fn compile(&self, mode: FuseMode, calib: Option<CalibrationC>) -> Result<FuseConfig, EnvelopeError> {
        self.validate()?;
        let m = &self.embodiment;
        let dim = m.action_dim as usize;
        let pos_dim = m.pos_dim as usize;
        let horizon = m.horizon as usize;
        let exec = m.exec_steps as usize;
        let dt = self.dt();
        let inv_dt = 1.0 / dt;
        let inv_dt2 = inv_dt * inv_dt;
        let inv_dt3 = inv_dt2 * inv_dt;
        let mut norm_center = [0.0; MAX_D];
        let mut inv_norm_scale = [0.0; MAX_D];
        let mut norm_scale_iso = f64::INFINITY;
        for (dst, src) in norm_center.iter_mut().zip(&m.norm_center) {
            *dst = *src;
        }
        for (dst, src) in inv_norm_scale.iter_mut().zip(&m.norm_scale) {
            *dst = 1.0 / *src;
        }
        for s in &m.norm_scale {
            norm_scale_iso = fmath::min(norm_scale_iso, *s);
        }
        let mut box_lo = [0.0; MAX_POS];
        let mut box_hi = [0.0; MAX_POS];
        for (dst, src) in box_lo.iter_mut().zip(&self.box_lo) {
            *dst = *src + self.margin;
        }
        for (dst, src) in box_hi.iter_mut().zip(&self.box_hi) {
            *dst = *src - self.margin;
        }
        let gate = GateSpec::parse(&self.gate)?;
        let calib_digest = calib.as_ref().map(|c| c.digest);
        let calib = match calib {
            Some(c) => {
                check_calibration(&c, m)?;
                c
            }
            None => {
                let mut c = CalibrationC::DISARMED;
                c.gate = gate;
                c.mask = gate.mask();
                c.horizon_ticks = m.horizon_ticks;
                c
            }
        };
        let (envelope_digest, embodiment_digest) = self.digest_bytes();
        Ok(FuseConfig {
            envelope_digest,
            embodiment_digest,
            calib_digest,
            mode,
            kind: m.action_kind,
            dim,
            pos_dim,
            horizon,
            exec,
            overlap: horizon - exec,
            dt,
            inv_dt,
            inv_dt2,
            inv_dt3,
            norm_center,
            inv_norm_scale,
            norm_scale_iso,
            box_lo,
            box_hi,
            v_max: self.v_max,
            a_max: self.a_max,
            j_max: self.j_max,
            reach_max: self.reach_max,
            step_max: self.v_max * dt,
            contact: self.contact,
            brake: self.brake,
            clamp: self.clamp_mode,
            hyst: self.hysteresis,
            window_mask: (1u64 << self.hysteresis.n) - 1,
            tier0_enabled: self.tier0_mask()?,
            n_operators: self.operators.len() as u8,
            calib,
        })
    }

    /// `TripMask` bits named in `tier0_enabled`. Only the seven armable names are accepted; `nonfinite`, `schema`
    /// and `watchdog` are always on and may not be listed.
    pub fn tier0_mask(&self) -> Result<u32, EnvelopeError> {
        let mut m = 0u32;
        for name in &self.tier0_enabled {
            match TripMask::from_name(name) {
                Some(bit) if bit & ARMABLE != 0 => m |= bit,
                _ => {
                    let valid: Vec<&str> = TripMask::names(ARMABLE).collect();
                    return Err(inv(format!("tier0_enabled: `{name}` is not one of {valid:?}")));
                }
            }
        }
        Ok(m)
    }

    #[cfg(feature = "std")]
    fn digest_bytes(&self) -> ([u8; 32], [u8; 32]) {
        (hex32(&self.digest_hex()), hex32(&self.embodiment_digest()))
    }

    #[cfg(not(feature = "std"))]
    fn digest_bytes(&self) -> ([u8; 32], [u8; 32]) {
        ([0; 32], [0; 32])
    }
}

/// A calibration is embedded only if it can be binned and aggregated consistently with this manifest.
fn check_calibration(c: &CalibrationC, m: &EmbodimentManifest) -> Result<(), EnvelopeError> {
    if c.t_grid == 0 || c.t_grid as usize > T_GRID {
        return Err(inv(format!("calibration: t_grid {} is not in 1..={T_GRID}", c.t_grid)));
    }
    if c.alpha_den == 0 {
        return Err(inv("calibration: alpha_den must be > 0".into()));
    }
    if c.mask != c.gate.mask() {
        return Err(inv(format!("calibration: mask {:#x} != gate.mask() {:#x}", c.mask, c.gate.mask())));
    }
    if c.tau.is_nan() {
        return Err(inv("calibration: tau is NaN".into()));
    }
    if c.armed() && c.horizon_ticks != m.horizon_ticks {
        return Err(inv(format!(
            "calibration horizon_ticks {} != embodiment.horizon_ticks {} (binning would diverge)",
            c.horizon_ticks, m.horizon_ticks
        )));
    }
    Ok(())
}

/// 64 lowercase hex chars -> 32 bytes; anything else -> zeros (unreachable for a `sha256_hex` output).
#[cfg(feature = "std")]
fn hex32(s: &str) -> [u8; 32] {
    fn nib(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            b'A'..=b'F' => c - b'A' + 10,
            _ => 0,
        }
    }
    let mut out = [0u8; 32];
    let b = s.as_bytes();
    if b.len() != 64 {
        return out;
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = (nib(b[2 * i]) << 4) | nib(b[2 * i + 1]);
    }
    out
}

/// sha256 hex over `lictor_canon::canon(floatify(serde_json::to_value(t)))`.
#[cfg(feature = "std")]
fn canonical_digest<T: Serialize>(t: &T) -> String {
    let v = serde_json::to_value(t).expect("envelope types serialize to JSON");
    let bytes =
        lictor_canon::canon(&lictor_canon::floatify(v)).expect("floatified envelope JSON is canonical");
    lictor_canon::sha256_hex(&bytes)
}

/// Every key the TOML surface accepts, per table. Kept beside the structs; the round-trip test catches drift
/// (a field missing here makes `from_toml(to_toml(e))` fail).
#[cfg(feature = "std")]
fn check_unknown_keys(t: &toml::Table) -> Result<(), EnvelopeError> {
    const TOP: &[&str] = &[
        "schema",
        "envelope_id",
        "embodiment",
        "box_lo",
        "box_hi",
        "margin",
        "v_max",
        "a_max",
        "j_max",
        "reach_max",
        "contact",
        "brake",
        "clamp_mode",
        "hysteresis",
        "tier0_enabled",
        "gate",
        "operators",
        "fit",
        "fail_closed",
    ];
    const TABLES: &[(&str, &[&str])] = &[
        (
            "embodiment",
            &[
                "id",
                "units",
                "action_kind",
                "action_dim",
                "pos_dim",
                "horizon",
                "exec_steps",
                "control_hz_num",
                "control_hz_den",
                "provides_vel",
                "horizon_ticks",
                "norm_center",
                "norm_scale",
                "aux_layout",
                "ext_names",
            ],
        ),
        ("brake", &["kind", "k_p", "k_v", "substeps", "dt", "commit_steps", "brake_steps", "react_ticks"]),
        ("contact", &["radius", "v_max", "aux_center"]),
        (
            "hysteresis",
            &[
                "k",
                "n",
                "warn_margin",
                "clear_ticks",
                "clamp_streak_to_brake",
                "max_clamps_per_episode",
                "stop_confirm_ticks",
                "v_stop_eps",
                "brake_timeout_ticks",
                "rearm",
                "rearm_hold",
                "max_rearms",
                "escalate_after_hold_ticks",
                "handoff_timeout_ticks",
                "watchdog_ticks",
            ],
        ),
        ("fit", &["source_run", "quantile", "slack", "n_episodes", "fitted_utc", "note"]),
    ];
    for k in t.keys() {
        if !TOP.contains(&k.as_str()) {
            return Err(EnvelopeError::Parse(format!("unknown key `{k}`")));
        }
    }
    for (name, allowed) in TABLES {
        if let Some(v) = t.get(*name) {
            let sub =
                v.as_table().ok_or_else(|| EnvelopeError::Parse(format!("`{name}` must be a table")))?;
            for k in sub.keys() {
                if !allowed.contains(&k.as_str()) {
                    return Err(EnvelopeError::Parse(format!("unknown key `{name}.{k}`")));
                }
            }
        }
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::ToString, vec};

    fn manifest() -> EmbodimentManifest {
        EmbodimentManifest {
            id: "gym_pusht/PushT-v0".into(),
            units: "px".into(),
            action_kind: ActionKind::EePosition,
            action_dim: 2,
            pos_dim: 2,
            horizon: 15,
            exec_steps: 8,
            control_hz_num: 10,
            control_hz_den: 1,
            provides_vel: true,
            horizon_ticks: 300,
            norm_center: vec![256.0, 256.0],
            norm_scale: vec![256.0, 256.0],
            aux_layout: vec!["block_x".into(), "block_y".into(), "block_theta".into(), "coverage".into()],
            ext_names: vec![],
        }
    }

    fn envelope() -> SafetyEnvelope {
        SafetyEnvelope {
            schema: SCHEMA.into(),
            envelope_id: "pusht-base-v1".into(),
            embodiment: manifest(),
            box_lo: vec![15.0, 15.0],
            box_hi: vec![497.0, 497.0],
            margin: 2.0,
            v_max: 1000.0,
            a_max: 20000.0,
            j_max: 400000.0,
            reach_max: 150.0,
            contact: None,
            brake: BrakeModel {
                kind: BrakeKind::PdSecondOrder,
                k_p: 100.0,
                k_v: 20.0,
                substeps: 10,
                dt: 0.01,
                commit_steps: 8,
                brake_steps: 8,
                react_ticks: 1,
            },
            clamp_mode: ClampMode::Project,
            hysteresis: Hysteresis {
                k: 3,
                n: 5,
                warn_margin: 0.5,
                clear_ticks: 5,
                clamp_streak_to_brake: 3,
                max_clamps_per_episode: 60,
                stop_confirm_ticks: 2,
                v_stop_eps: 5.0,
                brake_timeout_ticks: 30,
                rearm: RearmPolicy::Auto,
                rearm_hold: 10,
                max_rearms: 2,
                escalate_after_hold_ticks: 30,
                handoff_timeout_ticks: 200,
                watchdog_ticks: 2,
            },
            tier0_enabled: ["workspace", "speed", "accel", "jerk", "reach", "brake"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            gate: Feat::NAMES[..8].iter().map(|s| s.to_string()).collect(),
            operators: vec![],
            fit: None,
            fail_closed: true,
        }
    }

    #[test]
    fn gate_parse_mask_names() {
        let g = GateSpec::parse(&["tce".to_string(), "acc|path_ineff".to_string()]).unwrap();
        assert_eq!(g.n_terms, 2);
        assert_eq!(g.terms[0], Feat::Tce.bit());
        assert_eq!(g.terms[1], Feat::Acc.bit() | Feat::PathIneff.bit());
        assert_eq!(g.mask(), Feat::Tce.bit() | Feat::Acc.bit() | Feat::PathIneff.bit());
        assert_eq!(g.names(), vec!["tce".to_string(), "acc|path_ineff".to_string()]);
        assert!(matches!(GateSpec::parse(&["nope".to_string()]), Err(EnvelopeError::Invalid(_))));
        assert!(matches!(GateSpec::parse(&["tce|".to_string()]), Err(EnvelopeError::Invalid(_))));
        assert!(matches!(GateSpec::parse(&[" tce".to_string()]), Err(EnvelopeError::Invalid(_))));
        assert_eq!(GateSpec::parse(&[]).unwrap(), GateSpec::DISARMED);
        let nine: Vec<String> = (0..9).map(|_| "tce".to_string()).collect();
        assert!(matches!(GateSpec::parse(&nine), Err(EnvelopeError::Invalid(_))));
        let eight =
            GateSpec::parse(&Feat::NAMES[..8].iter().map(|s| s.to_string()).collect::<Vec<_>>()).unwrap();
        assert_eq!(eight.n_terms, 8);
        assert_eq!(eight.mask(), 0xff);
        assert_eq!(eight.names(), Feat::NAMES[..8].iter().map(|s| s.to_string()).collect::<Vec<_>>());
        let twelve: Vec<String> = Feat::NAMES.iter().map(|s| s.to_string()).collect();
        assert!(matches!(GateSpec::parse(&twelve), Err(EnvelopeError::Invalid(_))), "12 terms > MAX_TERMS");
        let one_and = GateSpec::parse(&[Feat::NAMES.join("|")]).unwrap();
        assert_eq!(one_and.n_terms, 1);
        assert_eq!(one_and.mask(), 0xfff);
        assert_eq!(one_and.names(), vec![Feat::NAMES.join("|")]);
    }

    #[test]
    fn tier0_names_map_to_tripmask_bits() {
        let e = envelope();
        let m = e.tier0_mask().unwrap();
        assert_eq!(
            m,
            TripMask::WORKSPACE
                | TripMask::SPEED
                | TripMask::ACCEL
                | TripMask::JERK
                | TripMask::REACH
                | TripMask::BRAKE
        );
        let mut bad = envelope();
        bad.tier0_enabled.push("nonfinite".into());
        assert!(matches!(bad.tier0_mask(), Err(EnvelopeError::Invalid(_))));
        bad.tier0_enabled.pop();
        bad.tier0_enabled.push("contact".into());
        assert_eq!(bad.tier0_mask().unwrap(), m | TripMask::CONTACT);
        assert!(matches!(bad.validate(), Err(EnvelopeError::Invalid(_))));
    }

    #[test]
    fn programmatic_envelope_validates_and_compiles() {
        let e = envelope();
        e.validate().unwrap();
        let cfg = e.compile(FuseMode::Enforce, None).unwrap();
        assert_eq!(cfg.step_max, 100.0);
        assert_eq!(&cfg.box_lo[..2], &[17.0, 17.0]);
        assert_eq!(&cfg.box_hi[..2], &[495.0, 495.0]);
        assert_eq!(cfg.overlap, 7);
        assert_eq!(cfg.window_mask, 31);
        assert_eq!(cfg.norm_scale_iso, 256.0);
        assert_eq!(cfg.inv_dt, 10.0);
        assert_eq!(cfg.calib.mask, 0xff);
        assert_eq!(cfg.calib.tau, f64::INFINITY);
        assert_eq!(cfg.calib.horizon_ticks, 300);
        assert!(cfg.calib_digest.is_none());
        let mut off = envelope();
        off.gate.clear();
        let cfg = off.compile(FuseMode::Observe, None).unwrap();
        assert!(!cfg.calib.armed());
    }

    #[test]
    fn calibration_embedding_is_checked() {
        let e = envelope();
        let mut c = CalibrationC::DISARMED;
        c.gate = GateSpec::parse(&e.gate).unwrap();
        c.mask = c.gate.mask();
        c.horizon_ticks = 299;
        c.digest = [7; 32];
        assert!(matches!(e.compile(FuseMode::Enforce, Some(c)), Err(EnvelopeError::Invalid(_))));
        c.horizon_ticks = 300;
        let cfg = e.compile(FuseMode::Enforce, Some(c)).unwrap();
        assert_eq!(cfg.calib_digest, Some([7; 32]));
        c.mask = 1;
        assert!(matches!(e.compile(FuseMode::Enforce, Some(c)), Err(EnvelopeError::Invalid(_))));
    }

    #[test]
    fn validate_rejects_each_documented_violation() {
        fn bad(f: impl FnOnce(&mut SafetyEnvelope)) -> EnvelopeError {
            let mut e = envelope();
            f(&mut e);
            e.validate().expect_err("expected rejection")
        }
        assert!(matches!(bad(|e| e.schema = "lictor-envelope/v0".into()), EnvelopeError::Unsupported(_)));
        assert!(matches!(bad(|e| e.fail_closed = false), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.hysteresis.n = 64), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.hysteresis.k = 0), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.hysteresis.k = 6), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.embodiment.horizon_ticks = 0), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.embodiment.exec_steps = 16), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.embodiment.norm_scale[1] = 0.0), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.embodiment.pos_dim = 3), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.box_hi = vec![18.0, 497.0]), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.margin = f64::NAN), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.v_max = 0.0), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.j_max = f64::INFINITY), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.brake.substeps = 5), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.brake.commit_steps = 16), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.brake.kind = BrakeKind::ZeroVelocityHold), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.gate.push("bogus".into())), EnvelopeError::Invalid(_)));
        assert!(matches!(bad(|e| e.operators.push("abc".into())), EnvelopeError::Invalid(_)));
        let key = "0".repeat(64);
        assert!(matches!(bad(|e| e.operators = vec![key.clone(), key.clone()]), EnvelopeError::Invalid(_)));
        assert!(matches!(
            bad(|e| e.operators = (0..9).map(|i| format!("{i:064x}")).collect()),
            EnvelopeError::Invalid(_)
        ));
        let mut ok = envelope();
        ok.operators = (0..8).map(|i| format!("{i:064x}")).collect();
        ok.validate().unwrap();
        assert!(matches!(
            bad(|e| e.contact = Some(ContactLimit { radius: 80.0, v_max: 400.0, aux_center: [0, 9] })),
            EnvelopeError::Invalid(_)
        ));
        let mut with_contact = envelope();
        with_contact.contact = Some(ContactLimit { radius: 80.0, v_max: 400.0, aux_center: [0, 1] });
        with_contact.tier0_enabled.push("contact".into());
        with_contact.validate().unwrap();
        assert_eq!(
            with_contact.compile(FuseMode::Enforce, None).unwrap().tier0_enabled & TripMask::CONTACT,
            TripMask::CONTACT
        );
    }

    #[test]
    fn error_display_names_the_variant() {
        assert_eq!(inv("x".into()).to_string(), "envelope invalid: x");
        assert_eq!(EnvelopeError::Parse("p".into()).to_string(), "envelope parse error: p");
        assert_eq!(EnvelopeError::Unsupported("u".into()).to_string(), "envelope unsupported: u");
    }
}
