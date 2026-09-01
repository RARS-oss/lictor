# lictor -- IMPLEMENTATION PLAN (milestone 1, parallel execution by disjoint-file work packages)

Companion to `docs/ARCHITECTURE.md` (which carries the same interface freeze plus the rationale, math and experiment protocol). This document is what an implementation agent reads: PART A is the INTERFACE FREEZE (verbatim; nobody invents an interface), PART B is WP-0 (the skeleton that makes the workspace compile from minute one), PART C is the 12 further work packages plus the final INTEGRATION package, PART D is the ownership table proving disjointness.

## 0. Environment and rules for EVERY agent

Host: Windows 11 + WSL2 Ubuntu (kernel 6.6). All Rust work runs in WSL:

```bash
export PATH="$HOME/.cargo/bin:$PATH"                 # cargo 1.98 stable
export CARGO_TARGET_DIR=/mnt/d/lictor/target         # C: is FULL (~1 GB free). NEVER build into the repo or C:.
cd /mnt/c/Users/Daniil/Desktop/Robots/lictor         # == C:\Users\Daniil\Desktop\Robots\lictor
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace
```
Python: `/mnt/d/lictor/venv/bin/python` (3.12; torch 2.7.1+cu126 installed; `lerobot[pusht]==0.6.1` installing). GTX 1060 3GB Pascal: fp32 only; CPU fallback must work. Heavy artefacts (models, results, target) go to `/mnt/d/lictor/`. `harness/env.sh` exports everything.

Rules:
1. **Exactly one agent writes a given path.** Your `files` list is exhaustive. Never edit a file outside it. If a frozen signature is wrong, do not patch around it: report to WP-0 (single owner of every `Cargo.toml`, `lib.rs`, `main.rs`, `cmd/mod.rs`), who amends the freeze.
2. **Code against the freeze immediately**; do not wait for other packages. WP-0 lands in ~60 minutes and every module file you own already exists as a `todo!()` stub with the frozen signature; replace its body, keep its signature.
3. **House style** (bulla / sbx): Rust 2021, `#![forbid(unsafe_code)]` in every library crate (the only `unsafe` is the counting allocator: `lictor-cli/src/alloc_count.rs` behind `#![deny(unsafe_code)]` + `#[allow(unsafe_code)] mod alloc_count;` in `main.rs`, and `lictor-fuse/tests/alloc.rs`), `clippy -D warnings` clean (every frozen `new()` has a frozen `Default`, every frozen `len()` a frozen `is_empty()`; no `#[allow(clippy::...)]` on frozen items), `cargo fmt`, MIT header comment on every file (`// SPDX-License-Identifier: MIT`), fixture tests (JSON/TOML fixtures under `tests/fixtures/`), deterministic capped human output, `--json` everywhere. No emojis. ASCII only in code and docs.
4. **Banned vocabulary as claims about lictor**: "safety-rated", "hard real-time" / "hard-real-time", "PL d", "SIL 2", "certified", "engineered to <standard>", "<standard> principles", "certified limits", and any "-like"/"-style" reference to a standard's clause without the disclaimer "terms borrowed for readability; no conformance is claimed or tested". Anywhere: code comments, docs, commit messages, figure captions. Also banned as claims: "proof" for the 40-replay evidence, "bit-identical on every platform" (say "on the tested target"), "a receipt can never hide ..." (say "the receipt binds the host's declaration of ...").
5. **Decision-path deny list** (`lictor-core`, `lictor-detect`, `lictor-fuse` ONLY): f32, `mul_add`, `powi`, `powf`, `exp`, `ln`, `sin`, `cos`, `atan2`, `hypot`, `f64::min`/`f64::max`/`f64::clamp` (NaN semantics), `as` casts from f64, `HashMap` iteration, `rayon`, SIMD reductions, `Instant`/`SystemTime`, RNG, allocation, panics on data (`debug_assert!` only). Use `lictor_core::fmath` for `sqrt/min/max/abs/clamp/norm/dist`. `lictor-calib`/`lictor-receipt` are off-path and MAY use `ln`/`exp`/`lgamma` (Clopper-Pearson, McNemar, bootstrap).
6. **Determinism of files**: every JSON artefact you write is pretty-printed with sorted keys; canonical bytes are recomputed, never trusted from disk.
7. **Tests live in files you own.** Unit tests inline (`#[cfg(test)]`) in your module files, integration tests in `tests/*.rs` files listed for you.
8. **Acceptance = the commands listed for your package pass**, run from a clean `git status` of your files only. Report measured numbers, not "works".
9. Commit only when the orchestrator asks; branch names `wp-<n>-<slug>`.
10. **Python** is always run as `python -m pytest ...` / `python harness/run.py` from the repo root; `pyproject.toml` (WP-0) sets `[tool.pytest.ini_options] pythonpath = ["."]`, `testpaths = ["harness/tests", "adapters/tests"]`. Keys never live in the repo: `lictor key init` defaults to `$LICTOR_KEYS` / `$HOME/.lictor` and refuses `/mnt/[a-z]/` without `--i-know`.

Integration order: WP-0 alone (T+0..60 min) -> everything else in parallel -> WP-13 INTEGRATION last. Test-level dependencies are listed per package; compile-level dependencies are satisfied by the WP-0 stubs.

---

# PART A -- INTERFACE FREEZE

Sections A.1-A.4 are verbatim copies of the normative sections of `docs/ARCHITECTURE.md`. If the two ever differ, ARCHITECTURE wins and the difference is a WP-0 bug.

## A.0 WP-0 freeze amendments (applied in the code and folded into the text below)

1. `lictor_runtime::wire::Response::Verdict` carries `Box<VerdictMsg>` (clippy `large_enum_variant`: 672 vs 376 bytes; the JSON is unchanged).
2. `lictor_calib::sweep::sweep` takes `opts: &SweepOpts { method, horizon_ticks, eps_prog }` instead of three scalars (clippy `too_many_arguments`, limit 7).
3. The `[workspace.dependencies]` entries for `lictor-core`, `lictor-detect`, `lictor-fuse` carry `default-features = false`; std consumers add `features = ["std"]` (cargo ignores a per-crate `default-features = false` on a `workspace = true` dependency, which would have made `make nostd-check` a no-op).

## A.1 Frozen Rust types

## FROZEN RUST TYPES (normative; verbatim in the code)

Rules that apply to every definition below:

- Edition 2021, `#![forbid(unsafe_code)]` in every library crate. The ONLY `unsafe` in the workspace is the counting allocator: `crates/lictor-cli/src/main.rs` carries `#![deny(unsafe_code)]` plus `#[allow(unsafe_code)] mod alloc_count;` (a `GlobalAlloc` wrapper over `System` with an `AtomicBool` gate, always installed, counting only while the gate is set), and `crates/lictor-fuse/tests/alloc.rs` (an integration-test crate root, outside the library's `forbid`) carries `#![allow(unsafe_code)]` with a header comment. Nothing else.
- Clippy policy: `cargo clippy --workspace --all-targets -- -D warnings` must be green on the WP-0 `todo!()` stubs. Therefore every frozen `pub fn new() -> Self` has a frozen `impl Default` beside it (`ChunkBuf`, `Trail`, `Tier1Rt`, `FuseRt`, `Hist`, `Staging`) and every frozen `len()` has a frozen `is_empty()` beside it (`Trail`, `F64Array`). No `#[allow(clippy::...)]` on frozen items and no `[workspace.lints]` table.
- `lictor-core`, `lictor-detect`, `lictor-fuse` are `#![cfg_attr(not(feature = "std"), no_std)]`. They contain no I/O, no crypto, no clock, no allocation on the decision path.
- Float discipline in `lictor-core` / `lictor-detect` / `lictor-fuse` (the decision path): `f64` only; the ONLY operations are `+ - * /`, `sqrt`, `min`, `max`, `abs`, comparisons. All are correctly rounded by IEEE-754, so verdicts are bit-identical on the tested target (x86_64 SSE2, release and debug) and are expected to be on any target where Rust `f64` is strict binary64 with no extended-precision intermediates (i586/x87 excluded) and no FMA contraction; untested elsewhere. FORBIDDEN: `f32` in the decision path, `mul_add`, `powi`, `powf`, `exp`, `ln`, `sin`, `cos`, `atan2`, `hypot`, `HashMap` iteration, `rayon`, SIMD reductions, `Instant`/`SystemTime`, RNG, `f64::min`/`f64::max`/`f64::clamp` (NaN semantics differ from `fmath`), and `as` casts FROM `f64`. Norms use `sqrt(sum x*x)` with fixed left-to-right accumulation (index ascending). `min`/`max`/`abs`/`sqrt` go through `lictor_core::fmath` so std and no_std are identical. SCOPE: this deny list applies to `lictor-core`, `lictor-detect`, `lictor-fuse` only; `lictor-calib` and `lictor-receipt` are off the decision path and MAY use `ln`/`exp`/`lgamma`-style functions (Clopper-Pearson, McNemar, bootstrap) and `f64::total_cmp` sorting.
- Any NaN/Inf reaching `decide()` -> `TripMask::NONFINITE` -> `FuseState::Fault` -> hold action (fail-closed). The fuse never panics on data; `debug_assert!` only.
- Every serde enum on a wire/file surface is `#[serde(rename_all = "snake_case")]`.
- Signatures below are frozen. Bodies are the owning WP's. Private fields may be added; public surface may not change without a WP-0 amendment.

### crates/lictor-core/src/lib.rs (WP-0 owns; declarations + re-exports only)

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! lictor-core: frozen types for the deterministic safety fuse. No logic beyond
//! constructors, validation and trivially-derived accessors. See docs/ARCHITECTURE.md.
//! DECISION-PATH DENY LIST (also enforced by review): f32, mul_add, powi, powf, exp, ln,
//! sin, cos, atan2, hypot, HashMap iteration, rayon, SIMD reductions, Instant, SystemTime, RNG.
#[cfg(feature = "std")] extern crate std;
extern crate alloc;

pub mod fmath;
pub mod chunk;
pub mod envelope;
pub mod scores;
pub mod verdict;
pub mod state;
pub mod reason;
pub mod ack;

pub use ack::{AckDecision, VerifiedAck};
pub use chunk::{ActionKind, ChunkBuf, ChunkError, ChunkView, ObsView, MAX_AUX, MAX_D, MAX_EXT, MAX_H, MAX_POS};
pub use envelope::{BrakeKind, BrakeModel, ClampMode, ContactLimit, EmbodimentManifest, EnvelopeError,
                   FitRecord, FuseConfig, GateSpec, Hysteresis, RearmPolicy, SafetyEnvelope, MAX_OPERATORS, MAX_TERMS};
pub use reason::{reason_text, ReasonCode};
pub use scores::{bin_of, CalMethod, CalibrationC, Feat, Scores, NFEAT, T_GRID};
pub use state::{EpisodeInit, ExecMode, FuseMode, FuseState};
pub use verdict::{ActionSource, SafetyVerdict, Status, TripMask};

pub const LICTOR_VERSION: &str = env!("CARGO_PKG_VERSION");
```

### crates/lictor-core/src/fmath.rs

```rust
//! The only float helpers allowed on the decision path. Identical results under std and no_std.
//! `libm` is an UNCONDITIONAL dependency of lictor-core (there is no `libm` feature): under `std` the hardware sqrt is
//! used, under no_std `libm::sqrt`; both are correctly rounded, hence bit-identical.
#[inline] pub fn sqrt(x: f64) -> f64 { #[cfg(feature = "std")] { x.sqrt() } #[cfg(not(feature = "std"))] { libm::sqrt(x) } }
#[inline] pub fn abs(x: f64) -> f64 { f64::from_bits(x.to_bits() & !(1u64 << 63)) }
/// `if a < b { a } else { b }` -- deliberately NOT f64::min (NaN never reaches here; fail-closed earlier).
#[inline] pub fn min(a: f64, b: f64) -> f64 { if a < b { a } else { b } }
#[inline] pub fn max(a: f64, b: f64) -> f64 { if a > b { a } else { b } }
#[inline] pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 { max(lo, min(x, hi)) }
/// Euclidean norm with fixed left-to-right accumulation over `v[..n]`.
#[inline] pub fn norm(v: &[f64], n: usize) -> f64 { let mut s = 0.0; let mut i = 0; while i < n { s = s + v[i] * v[i]; i += 1; } sqrt(s) }
/// ||a[..n] - b[..n]|| with fixed order.
#[inline] pub fn dist(a: &[f64], b: &[f64], n: usize) -> f64 { let mut s = 0.0; let mut i = 0; while i < n { let d = a[i] - b[i]; s = s + d * d; i += 1; } sqrt(s) }
#[inline] pub fn all_finite(v: &[f64]) -> bool { v.iter().all(|x| x.is_finite()) }
```

### crates/lictor-core/src/chunk.rs -- the action-chunk IR

```rust
use serde::{Deserialize, Serialize};

pub const MAX_H: usize = 64;    // pi0 chunks are 50 -> headroom
pub const MAX_D: usize = 32;    // pi0 padded action_dim is 32 -> headroom
pub const MAX_POS: usize = 32;  // proprioceptive position dim
pub const MAX_AUX: usize = 16;  // embodiment-declared auxiliary scalars (privileged sim state)
pub const MAX_EXT: usize = 4;   // Tier-2 external scalars

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind { EePosition, EeDelta, JointPosition, JointVelocity, Other }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkError { TooLong, TooWide, LenMismatch, ExecExceedsHorizon, Empty }

/// Pre-allocated, owned chunk storage (16.5 KB). Lives inside `FuseRt`; never allocated on the hot path.
#[derive(Clone)]
pub struct ChunkBuf {
    data: [f64; MAX_H * MAX_D],
    seq: u32, t_emit: u32, horizon: u16, dim: u16, exec_steps: u16, filled: bool,
}
impl ChunkBuf {
    pub const fn new() -> Self;
    /// Row-major copy of `src` (len == h*d). Non-finite values are accepted here and caught by `decide()`.
    pub fn fill(&mut self, seq: u32, t_emit: u32, h: u16, d: u16, exec: u16, src: &[f64]) -> Result<(), ChunkError>;
    pub fn copy_from(&mut self, v: ChunkView<'_>);
    pub fn view(&self) -> Option<ChunkView<'_>>;          // None when !filled
    pub fn row_mut(&mut self, i: usize) -> &mut [f64];     // len == dim; i clamped to horizon-1
    pub fn clear(&mut self);
    pub fn is_filled(&self) -> bool;
}
impl Default for ChunkBuf { fn default() -> Self { Self::new() } }

/// Borrowed, Copy view. THE chunk IR passed to detectors.
#[derive(Clone, Copy, Debug)]
pub struct ChunkView<'a> {
    pub seq: u32,         // monotone chunk index within the episode, from 0
    pub t_emit: u32,      // absolute env step at which index 0 applies
    pub horizon: u16,     // H (PushT: 15)
    pub dim: u16,         // d (PushT: 2)
    pub exec_steps: u16,  // S -- the irrevocably committed prefix (PushT: 8)
    pub data: &'a [f64],  // len == horizon*dim, row-major: data[i*dim + c]
}
impl<'a> ChunkView<'a> {
    /// Row `i`, clamped to `horizon-1` (never panics in release; debug_assert in debug).
    #[inline] pub fn action(&self, i: usize) -> &'a [f64];
    #[inline] pub fn overlap_len(&self) -> usize { (self.horizon - self.exec_steps) as usize }
    pub fn all_finite(&self) -> bool;
}

/// Everything the fuse is told about the world this tick.
#[derive(Clone, Copy, Debug)]
pub struct ObsView<'a> {
    pub t: u32,                 // absolute env step
    pub pos: &'a [f64],         // proprioceptive position, len == manifest.pos_dim (mandatory)
    pub vel: Option<&'a [f64]>, // proprioceptive velocity when manifest.provides_vel (PushT: Some -- gym_pusht info["vel_agent"] every step); None only for embodiments that provide none
    pub aux: &'a [f64],         // layout declared by manifest.aux_layout (PushT: block_x, block_y, block_theta, coverage)
    pub ext: &'a [f64],         // Tier-2 scalars ext0..ext3 from the policy process; len <= MAX_EXT; may be empty
}
```

### crates/lictor-core/src/envelope.rs -- the safety configuration

```rust
use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};
use crate::{chunk::{ActionKind, MAX_D, MAX_POS}, scores::CalibrationC, state::FuseMode};

pub const MAX_TERMS: usize = 8;
pub const MAX_OPERATORS: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentManifest {
    pub id: String,                 // "gym_pusht/PushT-v0"
    pub units: String,              // "px" | "rad" | "m" -- documentation, bound into the digest
    pub action_kind: ActionKind,    // ee_position
    pub action_dim: u16,            // 2
    pub pos_dim: u16,               // 2
    pub horizon: u16,               // 15   (the chunk length the adapter DELIVERS)
    pub exec_steps: u16,            // 8    (the committed prefix)
    pub control_hz_num: u32,        // 10   control period dt = den/num seconds, exact rational
    pub control_hz_den: u32,        // 1
    pub provides_vel: bool,         // true (PushT): the harness sends the simulator velocity as obs.vel; false -> finite-difference v_hat (recorded as a fuse_note)
    pub horizon_ticks: u32,         // 300  episode time base for calibration binning; MUST equal the env's max_episode_steps (checked by `lictor calibrate`)
    pub norm_center: Vec<f64>,      // len == action_dim; Tier-1 uses abar = (a - center)/scale
    pub norm_scale: Vec<f64>,       // len == action_dim; all > 0
    pub aux_layout: Vec<String>,    // ["block_x","block_y","block_theta","coverage"]
    pub ext_names: Vec<String>,     // <= 4 names for ext0..ext3; [] when unused
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
    pub k_p: f64,          // 100.0 (PushT)
    pub k_v: f64,          // 20.0
    pub substeps: u16,     // 10 -- must equal the env's inner loop
    pub dt: f64,           // 0.01 -- physics substep, NOT the control period
    pub commit_steps: u16, // 8 -- irrevocable prefix simulated forward
    pub brake_steps: u16,  // 8 -- extra hold steps simulated after the prefix (PdSecondOrder only)
    pub react_ticks: u16,  // 1 -- dead time before a brake bites (closed-form kinds only)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactLimit { pub radius: f64, pub v_max: f64, pub aux_center: [u8; 2] } // aux indices of the object centre

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampMode { Off, Project }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RearmPolicy { Auto, AckOnly }

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hysteresis {
    pub k: u8,                          // 3   K-of-N on the predictive channel
    pub n: u8,                          // 5   1 <= K <= N <= 63 (window_mask = (1 << n) - 1 must fit in u64)
    pub warn_margin: f64,               // 0.5 (z units): s > tau - warn_margin -> Watching
    pub clear_ticks: u8,                // 5   clean ticks to leave Watching/Clamped
    pub clamp_streak_to_brake: u8,      // 3   consecutive clamped CHUNKS -> Braking
    pub max_clamps_per_episode: u16,    // 60
    pub stop_confirm_ticks: u8,         // 2   Braking -> Held once ||v|| <= v_stop_eps this many ticks
    pub v_stop_eps: f64,                // 5.0 units/s
    pub brake_timeout_ticks: u16,       // 30  Braking that never stops -> Fault
    pub rearm: RearmPolicy,             // auto (PushT default) | ack_only
    pub rearm_hold: u16,                // 10  clean ticks in Held before auto re-arm
    pub max_rearms: u8,                 // 2   auto re-arms per episode
    pub escalate_after_hold_ticks: u16, // 30  Held this long -> Escalated (handoff)
    pub handoff_timeout_ticks: u32,     // 200 Escalated with no ack -> Terminated
    pub watchdog_ticks: u8,             // 2   missed_ticks above this -> Fault
}

/// Fixed-order DNF over Tier-1 features: s = max_i min_{j in terms[i]} z_j. A singleton term is a plain channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateSpec { pub n_terms: u8, pub terms: [u32; MAX_TERMS] }
impl GateSpec {
    pub const DISARMED: Self = Self { n_terms: 0, terms: [0; MAX_TERMS] };
    /// Parse ["tce","acc|path_ineff"] -> terms (each '|' = AND within a term). Unknown name -> Err.
    pub fn parse(terms: &[String]) -> Result<Self, EnvelopeError>;
    pub fn mask(&self) -> u32;   // OR of all terms
    pub fn names(&self) -> Vec<String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitRecord { pub source_run: String, pub quantile: f64, pub slack: f64, pub n_episodes: u32, pub fitted_utc: String, pub note: String }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    pub schema: String,             // "lictor-envelope/v1"
    pub envelope_id: String,        // "pusht-base-v1"
    pub embodiment: EmbodimentManifest,
    pub box_lo: Vec<f64>,           // len == pos_dim, env units
    pub box_hi: Vec<f64>,
    pub margin: f64,                // subtracted from the box before EVERY check
    pub v_max: f64,                 // units/s
    pub a_max: f64,                 // units/s^2
    pub j_max: f64,                 // units/s^3
    pub reach_max: f64,             // units: ||a_0 - p_t||
    pub contact: Option<ContactLimit>,
    pub brake: BrakeModel,
    pub clamp_mode: ClampMode,
    pub hysteresis: Hysteresis,
    pub tier0_enabled: Vec<String>, // subset of ["workspace","speed","accel","jerk","reach","contact","brake"]; nonfinite/schema/watchdog always on
    pub gate: Vec<String>,          // default gate terms, overridable by calibration.json
    pub operators: Vec<String>,     // Ed25519 pubkeys (hex64) allowed to sign an AckToken; slot = index
    pub fit: Option<FitRecord>,     // present when produced by `lictor envelope fit`
    pub fail_closed: bool,          // MUST be true in v1; recorded so a future `false` is auditable
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError { Parse(String), Invalid(String), Unsupported(String) }
impl SafetyEnvelope {
    #[cfg(feature = "std")] pub fn from_toml(s: &str) -> Result<Self, EnvelopeError>;
    #[cfg(feature = "std")] pub fn to_toml(&self) -> Result<String, EnvelopeError>;
    /// sha256 over lictor_canon::canon(floatify(to_value(self))). std only.
    #[cfg(feature = "std")] pub fn digest_hex(&self) -> String;
    /// sha256 over canon(floatify(to_value(&self.embodiment))) -- the sub-digest a calibration binds to. Tier-1 features
    /// depend on the EmbodimentManifest ONLY (never on v_max/a_max/j_max/reach_max/operators), so `envelope fit` and an
    /// operator-list change do not invalidate a calibration; a manifest change does. std only.
    #[cfg(feature = "std")] pub fn embodiment_digest(&self) -> String;
    pub fn validate(&self) -> Result<(), EnvelopeError>;   // dims, positivity, 1 <= k <= n <= 63, <= MAX_OPERATORS, horizon_ticks > 0, fail_closed == true
    pub fn dt(&self) -> f64 { self.embodiment.control_hz_den as f64 / self.embodiment.control_hz_num as f64 }
    /// One-shot, off-loop: validate + flatten Vec -> fixed arrays + precompute derived constants.
    pub fn compile(&self, mode: FuseMode, calib: Option<CalibrationC>) -> Result<FuseConfig, EnvelopeError>;
    pub fn tier0_mask(&self) -> Result<u32, EnvelopeError>;
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
    pub dim: usize, pub pos_dim: usize, pub horizon: usize, pub exec: usize, pub overlap: usize,
    pub dt: f64, pub inv_dt: f64, pub inv_dt2: f64, pub inv_dt3: f64,
    pub norm_center: [f64; MAX_D], pub inv_norm_scale: [f64; MAX_D], pub norm_scale_iso: f64,
    pub box_lo: [f64; MAX_POS], pub box_hi: [f64; MAX_POS],   // ALREADY margin-adjusted
    pub v_max: f64, pub a_max: f64, pub j_max: f64, pub reach_max: f64,
    pub step_max: f64,          // v_max * dt -- the per-step leash
    pub contact: Option<ContactLimit>,
    pub brake: BrakeModel,
    pub clamp: ClampMode,
    pub hyst: Hysteresis,
    pub window_mask: u64,       // (1 << n) - 1, n <= 63
    pub tier0_enabled: u32,     // TripMask bits armed
    pub n_operators: u8,
    pub calib: CalibrationC,    // CalibrationC::DISARMED when Tier 1 is off
}
```

### crates/lictor-core/src/scores.rs -- features and the runtime calibration table

```rust
use serde::{Deserialize, Serialize};
use crate::envelope::GateSpec;

pub const NFEAT: usize = 12;
pub const T_GRID: usize = 100;

/// Frozen feature ids. EVERY channel is oriented LARGER = MORE ANOMALOUS (acm is stored negated).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
pub enum Feat { Tce = 0, Acc = 1, AcmNeg = 2, Njr = 3, Reach = 4, PathIneff = 5, Stall = 6, SpeedPeak = 7, Ext0 = 8, Ext1 = 9, Ext2 = 10, Ext3 = 11 }
impl Feat {
    pub const NAMES: [&'static str; NFEAT] = ["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak","ext0","ext1","ext2","ext3"];
    pub const CHUNK_BOUNDARY_MASK: u32 = 0b0000_1001_1111; // tce acc acm_neg njr reach speed_peak: updated at a chunk boundary, held between
    pub const PER_TICK_MASK: u32      = 0b1111_0110_0000; // path_ineff stall ext0..3: updated every tick
    pub fn from_name(s: &str) -> Option<Feat>;
    #[inline] pub fn bit(self) -> u32 { 1u32 << (self as u32) }
    pub fn name(self) -> &'static str;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scores {
    pub f: [f64; NFEAT],   // raw, dimensionless (0.0 where !valid)
    pub z: [f64; NFEAT],   // standardised (0.0 where !valid or unmasked)
    pub s: f64,            // aggregate nonconformity = max over gate terms of min over the term; NEG_INFINITY when nothing valid
    pub valid: u32,        // bit j set <=> feature j computable this tick
    pub fired: u32,        // bit j set <=> z_j > tau (informational; the decision is on s)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalMethod { Static, Binned }   // Static == Binned with t_grid = 1

/// Hot-path calibration artefact: fixed arrays (~19 KB), no allocation, no I/O. Built once from calibration.json.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationC {
    pub method: CalMethod,
    pub alpha_num: u32, pub alpha_den: u32,
    pub n_calib: u32,            // the number of per-episode scores tau was taken over (== calibration.json n_calib; 2-way split: n_fit)
    pub horizon_ticks: u32,      // the time base used for binning (PushT: 300; == manifest.horizon_ticks)
    pub t_grid: u16,             // 1 (static) or T_GRID (binned)
    pub center: [[f64; NFEAT]; T_GRID],   // robust median per (bin, feature)
    pub scale:  [[f64; NFEAT]; T_GRID],   // 1.4826*MAD per (bin, feature), floored at 1e-9
    pub mask: u32,               // features entering s (== gate.mask())
    pub gate: GateSpec,
    pub tau: f64,                // split-CP quantile on per-episode max s; +INFINITY = never fires
    pub digest: [u8; 32],
}
impl CalibrationC {
    pub const DISARMED: Self;    // mask 0, gate DISARMED, tau +INFINITY, digest zero
    #[inline] pub fn bin(&self, t: u32) -> usize { bin_of(t, self.horizon_ticks, self.t_grid) }
    pub fn armed(&self) -> bool { self.mask != 0 }
}
/// INTEGER arithmetic only; shared by runtime and offline calibrator. A mismatch is a determinism trap.
#[inline] pub fn bin_of(t: u32, horizon_ticks: u32, t_grid: u16) -> usize {
    let g = t_grid as u64; let h = if horizon_ticks == 0 { 1 } else { horizon_ticks as u64 };
    let b = (t as u64) * g / h; if b >= g { (g - 1) as usize } else { b as usize }
}
```

### crates/lictor-core/src/verdict.rs

```rust
use serde::{Deserialize, Serialize};
use crate::{chunk::MAX_D, reason::ReasonCode, scores::Scores, state::FuseState};

/// Severity lattice (sbx feedback lineage). Ord derives from declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status { Nominal, Watching, Clamped, Braking, Held, Escalated, Fault, Terminated }

pub struct TripMask;
impl TripMask {
    pub const NONE: u32            = 0;
    pub const WORKSPACE: u32       = 1 << 0;
    pub const SPEED: u32           = 1 << 1;
    pub const ACCEL: u32           = 1 << 2;
    pub const JERK: u32            = 1 << 3;
    pub const REACH: u32           = 1 << 4;
    pub const CONTACT: u32         = 1 << 5;
    pub const BRAKE: u32           = 1 << 6;   // braking infeasible
    pub const NONFINITE: u32       = 1 << 7;
    pub const SCHEMA: u32          = 1 << 8;
    pub const WATCHDOG: u32        = 1 << 9;
    pub const TIER1_CP: u32        = 1 << 10;  // K-of-N predictive fire
    pub const CLAMP_BUDGET: u32    = 1 << 11;
    pub const HANDOFF_TIMEOUT: u32 = 1 << 12;
    pub const OPERATOR_ABORT: u32  = 1 << 13;
    pub const BRAKE_TIMEOUT: u32   = 1 << 14;
    pub const REARM_BUDGET: u32    = 1 << 15;
    pub const TIER0_SOFT: u32 = Self::WORKSPACE | Self::SPEED | Self::ACCEL | Self::JERK | Self::REACH | Self::CONTACT;
    pub const TIER0_HARD: u32 = Self::BRAKE | Self::NONFINITE | Self::SCHEMA | Self::WATCHDOG;
    pub const NAMES: [&'static str; 16] = ["workspace","speed","accel","jerk","reach","contact","brake","nonfinite",
        "schema","watchdog","tier1_cp","clamp_budget","handoff_timeout","operator_abort","brake_timeout","rearm_budget"];
    pub fn names(m: u32) -> impl Iterator<Item = &'static str>;
    pub fn from_name(s: &str) -> Option<u32>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSource { Policy, Clamped, Brake, Hold }

/// Copy, no String, no allocation. ~600 bytes.
#[derive(Clone, Copy, Debug)]
pub struct SafetyVerdict {
    pub seq: u32,                  // tick sequence within the episode, from 0
    pub t: u32,                    // absolute env step
    pub status: Status,
    pub state: FuseState,          // state AFTER this tick
    pub prev_state: FuseState,
    pub trips: u32,                // TripMask bits raised THIS tick
    pub action: [f64; MAX_D],      // the action the executor MUST apply (only [..action_dim] meaningful)
    pub action_dim: u16,
    pub action_src: ActionSource,
    pub substituted: bool,         // action != raw policy action (always false in Observe mode)
    pub clamped_dims: u32,         // dims moved by the Tier-0 projection
    pub scores: Scores,
    pub tau: f64,
    pub window: u64,               // K-of-N ring AFTER this tick
    pub window_hits: u8,           // popcount(window)
    pub brake_margin: f64,         // units of slack in the braking check; < 0 == infeasible
    pub reason: ReasonCode,
    pub handoff_seq: Option<u32>,  // Some on the tick Escalated is entered
    pub ack_consumed: bool,        // a VerifiedAck was applied this tick
    pub violation_reached_env: bool, // Observe mode: the would-be action_src was != Policy
}
```

### crates/lictor-core/src/state.rs

```rust
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuseState { #[default] Idle, Armed, Watching, Clamped, Braking, Held, Escalated, Fault, Terminated }
impl FuseState { pub fn is_stop(self) -> bool { matches!(self, Self::Braking | Self::Held | Self::Escalated | Self::Fault | Self::Terminated) }
                 pub fn is_terminal(self) -> bool { matches!(self, Self::Fault | Self::Terminated) } }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuseMode { Observe, Enforce }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecMode { Sync, Async }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeInit { pub episode_index: u32, pub seed: u64, pub delay_steps: u16, pub exec: ExecMode }
```

### crates/lictor-core/src/reason.rs -- the human-escalation glossary (sbx `hint_for` lineage)

```rust
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    Ok, ObserveOnly, WatchBand,
    ClampWorkspace, ClampSpeed, ClampAccel, ClampJerk, ClampReach, ClampContact,
    BrakeInfeasible, BrakeTier1Cp, BrakeClampBudget, Stopping, HeldStandstill,
    EscalateHoldTimeout, EscalateRearmBudget, EscalateTier1Persistent,
    FaultNonFinite, FaultSchema, FaultWatchdog, FaultBrakeTimeout, FaultInternal,
    RearmedAuto, RearmedAck, TerminatedAbort, TerminatedTimeout, TerminatedRetune, EpisodeEnd,
}
/// One sentence per code, <= 120 chars, plain English, explaining WHY the fuse acted.
pub fn reason_text(r: ReasonCode) -> &'static str;
```

### crates/lictor-core/src/ack.rs -- crypto-free ack types (verification lives in lictor-receipt)

```rust
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AckDecision { Resume, Abort, Retune }
/// An AckToken whose signature, operator membership and handoff digest were ALREADY verified by the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedAck { pub decision: AckDecision, pub operator_slot: u8, pub nonce: u64, pub handoff_seq: u32 }
```

### crates/lictor-detect (lib.rs: `pub mod tier0; pub mod brake; pub mod window; pub mod tier1; pub mod conformal;`)

```rust
// tier0.rs
use lictor_core::*;
#[derive(Clone, Copy, Debug, Default)]
pub struct Tier0Out { pub trips: u32, pub clamped_dims: u32, pub peak_speed: f64 }
/// Whole-chunk check at a chunk boundary. Writes the projected chunk into `out` (== ch when nothing clamped).
/// Exactly `horizon` iterations. Zero allocation.
pub fn check_chunk(cfg: &FuseConfig, obs: ObsView<'_>, ch: ChunkView<'_>, out: &mut ChunkBuf) -> Tier0Out;
/// Intra-chunk per-tick check of the single committed action `a` against predecessor `prev` (box, speed, reach, contact).
/// Writes the (possibly leashed) action into `out[..dim]`.
pub fn check_action(cfg: &FuseConfig, obs: ObsView<'_>, prev: &[f64], a: &[f64], out: &mut [f64]) -> Tier0Out;

// brake.rs
#[derive(Clone, Copy, Debug, Default)]
pub struct BrakeOut { pub feasible: bool, pub margin: f64, pub stop_dist: f64 }
/// Braking feasibility of the committed prefix of `ch` starting at row `from`, from plant state (p0, v0).
/// `from` = the row executed THIS tick (`idx`): at a chunk boundary `from == idx` (0 for sync/freeze delivery, `d` for async drop).
/// `v0` = obs.vel when the manifest provides it (PushT), else the finite-difference v_hat.
/// PdSecondOrder: EXACTLY (commit_steps + brake_steps) * substeps iterations of the PD map.
/// Closed-form kinds: EXACTLY commit_steps iterations of Euler + stopping ball. No allocation.
pub fn brake_feasible(cfg: &FuseConfig, p0: &[f64], v0: &[f64], ch: ChunkView<'_>, from: usize) -> BrakeOut;
/// The Braking-phase action: setpoint := clamp_box(p) (position kinds) or ramp-to-zero (velocity kinds).
pub fn brake_action(cfg: &FuseConfig, p: &[f64], v: &[f64], out: &mut [f64]);
/// The Held action: the latched hold setpoint (already clamped) copied into out.
pub fn hold_action(cfg: &FuseConfig, p_latch: &[f64], out: &mut [f64]);

// window.rs
#[derive(Clone, Copy, Debug, Default)]
pub struct Window { pub bits: u64 }
impl Window { #[inline] pub fn push(&mut self, hit: bool, mask: u64); #[inline] pub fn hits(&self) -> u8; pub fn clear(&mut self); }
pub const TRAIL_W: usize = 32;
#[derive(Clone, Copy)]
pub struct Trail { buf: [[f64; MAX_POS]; TRAIL_W], head: u8, len: u8 }
impl Trail { pub const fn new() -> Self; pub fn clear(&mut self); pub fn push(&mut self, p: &[f64], d: usize);
             pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool; pub fn net(&self, d: usize, w: usize) -> f64;  /* ||p_t - p_{t-w+1}|| */
             pub fn path(&self, d: usize, w: usize) -> f64; /* sum of consecutive distances over the last w */ }
impl Default for Trail { fn default() -> Self { Self::new() } }

// tier1.rs
#[derive(Clone)]
pub struct Tier1Rt { pub prev: ChunkBuf, pub have_prev: bool, pub trail: Trail, pub held: [f64; NFEAT], pub held_valid: u32 }
impl Tier1Rt { pub fn new() -> Self; pub fn reset(&mut self); }
impl Default for Tier1Rt { fn default() -> Self { Self::new() } }
pub const PE_WINDOW: usize = 20;   // ticks (2.0 s at 10 Hz)
/// stall reference speed v_ref = STALL_VREF_FRAC * norm_scale_iso / dt (PushT: 0.05 * 256 / 0.1 = 128 px/s). Depends on the
/// manifest only -- NEVER on v_max, which `envelope fit` changes after calibration.
pub const STALL_VREF_FRAC: f64 = 0.05;
/// Updates `rt` and writes raw features into `sc.f` / `sc.valid`. Chunk-boundary features recompute only when `chunk.is_some()`.
/// CONTRACT: `decide()` pushes `rt.trail` BEFORE calling this. `features` is the SOLE writer of `rt.prev`/`rt.have_prev`: after
/// computing tce/acc it copies the RAW incoming chunk into `rt.prev`, so TCE is always raw-vs-raw in Observe and Enforce alike
/// (calibration features == enforcement features). Every feature depends on the EmbodimentManifest only.
pub fn features(cfg: &FuseConfig, rt: &mut Tier1Rt, obs: ObsView<'_>, chunk: Option<ChunkView<'_>>, idx: u16, sc: &mut Scores);

// conformal.rs
/// z_j = (f_j - center[b][j]) / scale[b][j] for masked & valid j; b = cal.bin(t). Others 0.0.
pub fn standardise(cal: &CalibrationC, t: u32, sc: &mut Scores);
/// s = max over terms of min over the term's channels (all masked & valid), NEG_INFINITY if no term is fully valid.
/// Also sets sc.fired (z_j > tau per channel, informational). Returns s.
pub fn aggregate(cal: &CalibrationC, sc: &mut Scores) -> f64;
/// STRICT: s > tau.
#[inline] pub fn trip(cal: &CalibrationC, sc: &Scores) -> bool;
```

### crates/lictor-fuse (lib.rs: `pub mod fuse; pub mod fsm; pub mod tally; pub use fuse::{decide, Fuse, FuseRt, TickInput}; pub use fsm::{next, FsmInput}; pub use tally::Tally;`)

```rust
// fuse.rs
use lictor_core::*; use lictor_detect::{tier1::Tier1Rt, window::Window};
pub struct TickInput<'a> {
    pub obs: ObsView<'a>,
    pub chunk: Option<ChunkView<'a>>,   // Some ONLY on the tick a new chunk arrives
    pub idx: u16,                        // index into the CURRENT chunk for this tick
    pub missed_ticks: u8,                // the ONLY time-like input; an integer, supplied by the host
    pub ack: Option<VerifiedAck>,
    pub schema_fault: bool,             // set by the runtime on a wire/schema violation -> TripMask::SCHEMA in GUARD
}
/// Pre-allocated runtime state (~58 KB: three 16.5 KB ChunkBufs + an 8 KB Trail; `lictor bench` prints the exact size_of).
/// Constructed once per process; `reset()` per episode. Never allocates.
pub struct FuseRt {
    pub cur: ChunkBuf,                   // chunk currently executing (post-projection in Enforce; RAW in Observe, see WP-3)
    pub scratch: ChunkBuf,               // projection target
    pub t1: Tier1Rt,
    pub window: Window,
    pub state: FuseState,
    pub seq: u32,
    pub last_t: u32,                     // obs.t of the previous tick (time-continuity guard: t == last_t + 1 + missed_ticks; first tick t == 0)
    pub next_chunk_seq: u32,             // the seq the next delivered chunk MUST carry (0 after reset; chunk.seq continuity guard)
    pub pos: [f64; MAX_POS], pub prev_pos: [f64; MAX_POS], pub vel: [f64; MAX_POS], pub have_prev_pos: bool,
    pub last_cmd: [f64; MAX_D],          // last action actually emitted
    pub hold: [f64; MAX_POS],            // latched hold setpoint (Held/Escalated/Fault/Terminated)
    pub clean_run: u16, pub clamp_streak: u8, pub clamps: u16,
    pub brake_ticks: u16, pub stopped_ticks: u8, pub held_ticks: u16, pub held_clean: u16,
    pub escalated_ticks: u32, pub rearms: u8, pub handoff_seq: u32, pub handoff_pending: bool,
    pub chunk_ok_this_tick: bool,
    pub last_nonce: [u64; MAX_OPERATORS],
    pub tally: Tally,
}
impl FuseRt { pub fn new() -> Self; /// clears everything; state = Armed; tally = Tally::default() (max_s/max_z = NEG_INFINITY)
              pub fn reset(&mut self, init: EpisodeInit); }
impl Default for FuseRt { fn default() -> Self { Self::new() } }

/// THE CONTRACT. Pure: no allocation, no clock read, no lock, no syscall, no panic path, fixed iteration bounds.
/// Same (cfg, rt-before, inp) => bit-identical verdict and rt-after on any IEEE-754 platform.
#[inline(never)]
pub fn decide(cfg: &FuseConfig, rt: &mut FuseRt, inp: &TickInput<'_>) -> SafetyVerdict;

pub struct Fuse { cfg: FuseConfig, rt: FuseRt }
impl Fuse {
    pub fn new(cfg: FuseConfig) -> Self;
    pub fn cfg(&self) -> &FuseConfig;
    pub fn rt(&self) -> &FuseRt;
    pub fn reset(&mut self, init: EpisodeInit);
    #[inline] pub fn step(&mut self, inp: &TickInput<'_>) -> SafetyVerdict { decide(&self.cfg, &mut self.rt, inp) }
    pub fn finish(&self) -> Tally;
}

// fsm.rs
#[derive(Clone, Copy, Debug)]
pub struct FsmInput { pub trips: u32, pub predictive: bool, pub warn: bool, pub soft_clampable: bool,
                      pub stopped: bool, pub chunk_boundary: bool, pub chunk_ok: bool, pub ack: Option<VerifiedAck> }
/// Pure transition per the ARCHITECTURE table (rows evaluated top-down, first match wins). Updates counters in `rt`,
/// returns (new_state, extra_trips, reason).
pub fn next(cfg: &FuseConfig, rt: &mut FuseRt, inp: FsmInput) -> (FuseState, u32, ReasonCode);

// tally.rs -- Copy, fixed arrays; converted to receipt VerdictCounts by lictor-receipt
#[derive(Clone, Copy, Debug)]
pub struct Tally {
    pub ticks: u32, pub nominal: u32, pub watching: u32, pub clamped: u32, pub braking: u32, pub held: u32,
    pub escalated: u32, pub fault: u32, pub terminated: u32, pub substituted: u32,
    pub chunks_seen: u32, pub chunks_rejected: u32, pub clamps: u32, pub holds: u32, pub rearms: u32, pub escalations: u32,
    pub trips_by_bit: [u32; 16], pub fired_by_feat: [u32; NFEAT],
    pub first_trip_tick: Option<u32>, pub first_trip_reason: Option<ReasonCode>,
    pub first_stop_tick: Option<u32>, pub handoff_tick: Option<u32>,
    pub violations_reached_env: u32, pub max_s: f64, pub max_z: [f64; NFEAT], pub terminal_state: FuseState,
}
/// Hand-written (NOT derived): zeros/None everywhere, `terminal_state = Idle`, `max_s = f64::NEG_INFINITY`,
/// `max_z = [f64::NEG_INFINITY; NFEAT]` -- `s` is NEG_INFINITY while nothing is valid or Tier 1 is disarmed, and 0.0 would
/// bias the per-episode `max_s` that feeds ROC-AUC. F64Hex encodes +-inf exactly; the wire encodes it as `null`.
impl Default for Tally { fn default() -> Self; }
```

### crates/lictor-canon (lib.rs re-exports everything below)

```rust
use serde_json::Value;
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CanonError {
    #[error("float in canonical body at {0}")] FloatInBody(String),
    #[error("integer out of +-2^53-1 at {0}")] IntegerOutOfRange(String),
    #[error("key not [a-z0-9_]+: {0}")] BadKey(String),
    #[error("bad f64 encoding: {0}")] BadF64(String),
    #[error("serialize: {0}")] Serialize(String),
}
pub const CANONICAL_ID: &str = "jcs-floatfree/v1";
/// RFC 8785: keys sorted by UTF-16 code units, no whitespace, minimal escaping; integers only (|v| <= 2^53-1).
pub fn canon(v: &Value) -> Result<Vec<u8>, CanonError>;
pub fn check_keys(v: &Value) -> Result<(), CanonError>;
pub fn sha256_hex(bytes: &[u8]) -> String;
pub fn sha256_jcs(v: &Value) -> Result<String, CanonError>;
/// Replace every non-integer JSON number x by {"f64": f64_to_hex(x)} recursively (for TOML-sourced config trees).
pub fn floatify(v: Value) -> Value;
pub fn f64_to_hex(x: f64) -> String;            // 16 lowercase hex of x.to_bits() (big-endian nibbles)
pub fn f64_from_hex(s: &str) -> Result<f64, CanonError>;
/// Serializes as {"f64":"<16hex>"}; deserializes from the same.
#[derive(Clone, Copy, Debug, PartialEq)] pub struct F64Hex(pub f64);
/// Serializes as {"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[..]}.
#[derive(Clone, Debug, PartialEq)] pub struct F64Array { pub shape: Vec<u32>, pub data: Vec<f64> }
impl F64Array { pub fn from_slice(d: &[f64], shape: &[u32]) -> Self; pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool; }
/// to_value + check_keys + reject bare floats -> canonical bytes.
pub fn canon_of<T: serde::Serialize>(t: &T) -> Result<Vec<u8>, CanonError>;
pub fn digest_of<T: serde::Serialize>(t: &T) -> Result<String, CanonError>;
```

### crates/lictor-receipt (lib.rs: `pub mod tick; pub mod body; pub mod sign; pub mod ledger; pub mod curve; pub mod handoff; pub mod history; pub mod keys;` + re-exports)

```rust
use std::collections::BTreeMap; use lictor_canon::{F64Array, F64Hex}; use lictor_core::*;
pub const RECEIPT_SCHEMA: &str = "lictor-receipt/v1";
pub const TICKS_SCHEMA: &str = "lictor-ticks/v1";
pub const LEDGER_SCHEMA: &str = "lictor-ledger/v1";
pub const CURVE_SCHEMA: &str = "lictor-curve/v1";
pub const HANDOFF_SCHEMA: &str = "lictor-handoff/v1";
pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

// tick.rs -- the VERDICT chain (replayable) and the TIMING chain (honestly not)
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TickEvent {
    pub seq: u32, pub t: u32,
    pub state: FuseState, pub prev_state: FuseState, pub status: Status,
    pub trips: u32, pub action_src: ActionSource, pub substituted: bool, pub clamped_dims: u32,
    pub action: F64Array,      // [action_dim]
    pub f: F64Array,           // [NFEAT] raw features (what `lictor sweep` re-scores)
    pub z: F64Array,           // [NFEAT]
    pub valid: u32, pub fired: u32,
    pub s: F64Hex, pub tau: F64Hex, pub window_hits: u8, pub brake_margin: F64Hex,
    pub reason: ReasonCode, pub handoff_seq: Option<u32>, pub violation_reached_env: bool,
    pub prev: String, pub hash: String,   // hash = sha256(canon(self with hash = "")) -- see docs/receipt-schema.md
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimingEvent { pub seq: u32, pub decide_ns: u64, pub io_ns: u64, pub prev: String, pub hash: String }
pub fn tick_event(prev: &str, v: &SafetyVerdict) -> TickEvent;
pub fn timing_event(prev: &str, seq: u32, decide_ns: u64, io_ns: u64) -> TimingEvent;
#[derive(Clone, Debug, PartialEq)] pub struct ChainReport { pub ok: bool, pub break_at: Option<u32>, pub head: String, pub n: u32 }
pub fn verify_tick_chain(events: &[TickEvent], genesis_prev: &str) -> ChainReport;
pub fn verify_timing_chain(events: &[TimingEvent]) -> ChainReport;

// body.rs
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunBinding { pub run_id: String, pub arm_id: String, pub episode_index: u32, pub seed: u64,
    pub seed_pool: String, pub init_state_digest: String,
    pub env: BTreeMap<String, String>, pub policy: BTreeMap<String, String>, pub host: BTreeMap<String, String> }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BudgetBinding { pub mode: FuseMode, pub delay_steps: u16, pub tick_ms: u32, pub exec_mode: ExecMode, pub stitch: String,
    pub on_escalate: String, pub tier0_armed: Vec<String>, pub tier1_armed: bool, pub gate: Vec<String>,
    pub alpha_num: u32, pub alpha_den: u32, pub kn: [u8; 2] }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FaultBinding { pub kind: String, pub params: BTreeMap<String, String>, pub stream_seed: u64 }
#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct VerdictCounts { pub ticks: u32, pub nominal: u32, pub watching: u32, pub clamped: u32, pub braking: u32, pub held: u32,
    pub escalated: u32, pub fault: u32, pub terminated: u32, pub substituted: u32, pub chunks_seen: u32, pub chunks_rejected: u32,
    pub clamps: u32, pub holds: u32, pub rearms: u32, pub escalations: u32, pub trips_by_bit: Vec<u32>, pub fired_by_feat: Vec<u32>,
    pub first_trip_tick: Option<u32>, pub first_trip_reason: Option<String>, pub first_stop_tick: Option<u32>,
    pub handoff_tick: Option<u32>, pub violations_reached_env: u32, pub terminal_state: FuseState }
impl From<&lictor_fuse::Tally> for VerdictCounts { fn from(t: &lictor_fuse::Tally) -> Self; }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeOutcome { pub steps: u32, pub success: bool, pub terminated: bool, pub truncated: bool,
    pub max_coverage: F64Hex, pub final_coverage: F64Hex, pub reward_sum: F64Hex,
    pub ended_by: String,          // "success"|"truncated"|"escalation_terminate"|"fault"|"fuse_crash"|"abort"|"retune"
    pub max_s: F64Hex, pub max_z: F64Array }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LatencySummary { pub n: u32, pub p50_ns: u64, pub p90_ns: u64, pub p99_ns: u64, pub p999_ns: u64, pub max_ns: u64, pub label: String }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReceiptBody {
    pub schema: String, pub canonical: String, pub created_epoch: u64,
    pub lictor_version: String, pub lictor_git: String, pub lictor_sha256: String,
    pub client: String,                    // the `hello.client` string (host-declared)
    pub run: RunBinding, pub budget: BudgetBinding, pub fault_injection: Option<FaultBinding>,
    pub envelope: serde_json::Value,       // floatify(envelope) -- the full config, float-free
    pub envelope_digest: String, pub calibration_digest: Option<String>,
    pub inputs: BTreeMap<String, String>,  // content-addressed: key -> sha256. Host-declared entries (from episode_begin.inputs) use repo-relative paths;
                                           // fuse-computed entries use the keys "lictor:bin", "lictor:envelope", "lictor:calibration" (absent when no calibration)
    pub counts: VerdictCounts, pub outcome: EpisodeOutcome, pub handoffs: Vec<HandoffRecord>,
    pub verdict_events: u32, pub verdict_chain_head: String, pub timing_events: u32, pub timing_chain_head: String,
    pub latency: LatencySummary,
    pub ticks_policy: String,              // "tail32"|"all"|"none"
    #[serde(default)] pub ticks: Vec<TickEvent>,   // embedded per ticks_policy; the full stream is in the ticks file
    pub fuse_ok: bool, pub fuse_notes: Vec<String>,
    pub ledger_prev: Option<String>,
}
impl ReceiptBody { pub fn canonical(&self) -> Result<Vec<u8>, lictor_canon::CanonError>; pub fn digest_hex(&self) -> Result<String, lictor_canon::CanonError>; }
/// The honest verdict on the RECORD (not on the run). intact != fuse_ok. `ephemeral_key`: the receipt was signed by a key generated
/// at `lictor serve` startup (no `--key`); `verify` recomputes it as `body.fuse_notes.iter().any(|n| n == "ephemeral signing key")`.
pub fn evaluate_fuse(budget: &BudgetBinding, counts: &VerdictCounts, calibration_digest: Option<&str>, ephemeral_key: bool) -> (bool, Vec<String>);

// sign.rs
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedReceipt { pub body: ReceiptBody, pub body_digest: String, pub pubkey: String, pub sig: String }
pub fn sign(body: ReceiptBody, seed: &[u8; 32]) -> Result<SignedReceipt, ReceiptError>;
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct VerifyReport { pub schema_ok: bool, pub sig_ok: bool, pub digest_ok: bool, pub ticks_ok: bool, pub pubkey_ok: bool,
                          pub envelope_digest_ok: bool, pub counts_ok: bool, pub fuse_ok: bool, pub break_at: Option<u32>, pub notes: Vec<String> }
/// pubkey_ok = (expect_pubkey.is_none() || sr.pubkey == expected). envelope_digest_ok = recomputed sha256(canon(body.envelope)) == body.envelope_digest.
/// counts_ok = counts.ticks == verdict_events == timing_events == outcome.steps.
impl VerifyReport { pub fn intact(&self) -> bool { self.schema_ok && self.sig_ok && self.digest_ok && self.ticks_ok && self.pubkey_ok && self.envelope_digest_ok && self.counts_ok } }
pub fn verify(sr: &SignedReceipt, expect_pubkey: Option<&str>) -> VerifyReport;
/// Recompute the verdict chain from a full ticks file. `ChainReport.ok` is CHAIN INTEGRITY ONLY (a re-chained file passes it);
/// the caller MUST compare `report.head` with `sr.body.verdict_chain_head` and report `HEAD MISMATCH` when they differ, and MUST
/// cross-check the ticks header (run_id, arm_id, episode_index, genesis) against the body and the embedded tail's first `prev`.
pub fn verify_ticks_file(sr: &SignedReceipt, ticks: &[TickEvent]) -> ChainReport;
/// The two committed TEST keys (their pubkeys), so `lictor verify` can warn "signed with the committed test key".
pub const TEST_PUBKEYS: [&str; 2];   // filled in by WP-4 from tests/fixtures/receipt/key.hex and bench/fixtures/key.hex
#[derive(Debug, thiserror::Error)] pub enum ReceiptError { #[error("canon: {0}")] Canon(#[from] lictor_canon::CanonError), #[error("key: {0}")] Key(String), #[error("io: {0}")] Io(String) }

// ledger.rs -- a hash chain per arm: dropping, reordering or editing an entry WITHOUT the signing key breaks the chain at a reported seq.
// It does not defend against the key-holder (who can rebuild and re-sign); tail truncation of a DECLARED pool is detected by `lictor curve`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry { pub seq: u32, pub run_id: String, pub arm_id: String, pub episode_index: u32, pub seed: u64,
    pub init_state_digest: String, pub receipt_digest: String, pub verdict_chain_head: String,
    pub success: bool, pub fuse_ok: bool, pub tripped: bool, pub stopped: bool, pub escalated: bool,
    pub prev: String, pub hash: String }
pub fn ledger_entry(prev: &str, seq: u32, sr: &SignedReceipt) -> LedgerEntry;
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LedgerReport { pub chain_ok: bool, pub break_at: Option<u32>, pub episodes: u32, pub successes: u32, pub trips: u32, pub stops: u32, pub escalations: u32, pub head: String }
pub fn verify_ledger(entries: &[LedgerEntry]) -> LedgerReport;
pub fn read_ledger(path: &std::path::Path) -> Result<Vec<LedgerEntry>, ReceiptError>;
pub fn append_ledger(path: &std::path::Path, sr: &SignedReceipt) -> Result<LedgerEntry, ReceiptError>;   // reads head, appends, single writer

// handoff.rs -- the human-escalation surface
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AckToken { pub schema: String, pub handoff_digest: String, pub decision: AckDecision, pub operator: String,
                      pub nonce: u64, pub note: String, pub sig: String }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HandoffRecord { pub schema: String, pub run_id: String, pub arm_id: String, pub episode_index: u32,   // replay-binding: a captured ack
    pub seq: u32, pub tick: u32, pub reason: ReasonCode, pub reason_text: String,                                   // from another run/arm/episode cannot verify
    pub reasons: Vec<String>, pub trips: u32, pub fired: u32, pub window_hits: u8, pub top_z: Vec<(String, F64Hex)>,
    pub chain_at: String,    // == the `hash` of the TickEvent of the tick on which Escalated was entered (the verdict-chain head after that tick)
    pub envelope_digest: String, pub calibration_digest: Option<String>,
    pub digest: String,  // sha256(canon(self with digest="", ack=None, resolved_tick=None, outcome="")) -- WHAT THE OPERATOR SIGNS
    pub ack: Option<AckToken>, pub resolved_tick: Option<u32>, pub outcome: String }  // "resumed"|"aborted"|"retune"|"timeout"|"unacked"
pub fn ack_signing_bytes(t: &AckToken) -> Result<Vec<u8>, ReceiptError>;   // canon({handoff_digest,decision,operator,nonce,note,schema})
pub fn sign_ack(handoff_digest: &str, decision: AckDecision, nonce: u64, note: &str, seed: &[u8; 32]) -> Result<AckToken, ReceiptError>;
#[derive(Debug, Clone, PartialEq, Eq)] pub enum AckError { UnknownOperator, BadSignature, WrongHandoff, NonceReplay, NoPendingHandoff, TooLongNote }
/// Pure verification: operator listed in `operators` (slot = index), Ed25519 verify_strict, digest == pending, nonce > last_nonce[slot].
/// `last_nonce` lives in the Session ACROSS episodes (never reset per episode) and is persisted per pubkey in
/// `<out>/.lictor/verifier_nonce.json` when `--out` is set; without that file acks are replayable across processes (SECURITY.md).
pub fn verify_ack(t: &AckToken, operators: &[String], pending_digest: Option<&str>, pending_seq: u32, last_nonce: &[u64]) -> Result<VerifiedAck, AckError>;
pub fn load_nonces(path: &std::path::Path) -> Result<BTreeMap<String, u64>, ReceiptError>;   // pubkey -> last nonce; missing file == empty
pub fn save_nonces(path: &std::path::Path, m: &BTreeMap<String, u64>) -> Result<(), ReceiptError>;

// curve.rs
/// A paired difference with its bootstrap CI (10 000 resamples, splitmix64 seed 20260830) and exact McNemar on the same pairs.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeltaCi { pub diff: F64Hex, pub lo: F64Hex, pub hi: F64Hex, pub mcnemar_p: F64Hex, pub mcnemar_b: u32, pub mcnemar_c: u32 }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveMetrics { pub n: u32, pub success_rate: F64Hex, pub success_ci: [F64Hex; 2], pub averted_rate: F64Hex, pub flagged_rate: F64Hex,
    pub flagged_ci: [F64Hex; 2], pub false_trip_rate: F64Hex, pub false_trip_ci: [F64Hex; 2], pub intervention_rate: F64Hex,
    pub intervention_tick_frac: F64Hex, pub escalation_rate: F64Hex, pub lead_mean: F64Hex, pub lead_p50: F64Hex, pub lead_p10: F64Hex,
    pub aucpdt: F64Hex, pub roc_auc: F64Hex, pub bacc: F64Hex, pub violations_reached_env: u32,
    pub delta_vs_baseline: DeltaCi,                    // success(arm) - success(obs-d0), paired by episode index
    pub delta_vs_latency_control: Option<DeltaCi>,     // success(arm) - success(obs-d{d}); None when no latency-control arm exists in the run
    pub tce_valid_frac: F64Hex,                        // fraction of ticks with Feat::Tce valid (sync d >= 7 has NO chunk overlap -> 0.0)
    pub latency_p50_ns: u64, pub latency_p99_ns: u64, pub latency_max_ns: u64, pub latency_label: String,
    pub n_fail_baseline: u32, pub n_succ_baseline: u32 }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveReceiptBody { pub schema: String, pub canonical: String, pub created_epoch: u64, pub run_id: String, pub arm_id: String,
    pub baseline_arm_id: String, pub latency_control_arm_id: Option<String>, pub ledger_head: String, pub n_episodes: u32, pub ledger_chain_ok: bool,
    /// sha256(canon({"envelope_digest": String, "calibration_digest": Option<String>, "budget": BudgetBinding})) of the arm's receipts (all identical, else refused)
    pub arm_config_digest: String,
    pub receipt_pubkey: String,                        // the ONE pubkey every receipt of the arm carries (mixed keys or an ephemeral note -> refused)
    pub run_json_sha256: String,                       // sha256 of results/<run>/run.json (the unsigned harness manifest, bound here)
    pub n_declared: u32, pub n_present: u32, pub n_missing: u32, pub missing_indices: Vec<u32>,   // declared pool (run.json + calibration seed_pool) vs ledger
    pub partial: bool,                                 // true only under --partial (index sets differ from the baseline's or from the declared pool)
    pub small_n: bool,                                 // true only under --allow-small (n < 100): a pilot point, never a headline
    pub cross_run_mismatches: Vec<u32>,                // episode indices whose verdict_chain_head differs from the same (arm, seed) in --compare-run
    pub pair_mismatches: Vec<u32>, pub seed_overlap_with_calibration: Vec<u64>, pub metrics: CurveMetrics, pub eps_prog: F64Hex }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedCurve { pub body: CurveReceiptBody, pub body_digest: String, pub pubkey: String, pub sig: String }
pub fn sign_curve(body: CurveReceiptBody, seed: &[u8; 32]) -> Result<SignedCurve, ReceiptError>;
pub fn verify_curve(sc: &SignedCurve, expect_pubkey: Option<&str>) -> VerifyReport;   // ticks_ok/counts_ok/envelope_digest_ok = true (n/a)

// history.rs -- sbx run-memory lineage -> .lictor/history.jsonl, capped at 200
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeRecord { pub ts: String, pub run_id: String, pub arm_id: String, pub seed: u64, pub success: bool,
                           pub first_trip_reason: Option<String>, pub trips: Vec<String>, pub stopped: bool, pub escalated: bool }
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LoopSignals { pub tail_streak: Option<(String, u32)>, pub identical_run: u32, pub last_n: u32 }
pub fn record_episode(dir: &std::path::Path, r: &EpisodeRecord) -> std::io::Result<()>;
pub fn summarize(dir: &std::path::Path, n: usize) -> std::io::Result<LoopSignals>;

// keys.rs
pub fn keygen() -> [u8; 32];                                    // getrandom
pub fn pubkey_hex(seed: &[u8; 32]) -> String;
pub fn load_seed(path: &std::path::Path) -> Result<[u8; 32], ReceiptError>;   // 64 hex chars
/// mode 0600 on unix (behind `#[cfg(unix)]` via std::os::unix::fs::PermissionsExt; a no-op with a stderr warning elsewhere).
/// Refuses (ReceiptError::Key) a path under `/mnt/[a-z]/` (WSL DrvFs does not enforce 0600) unless `allow_drvfs`; refuses to
/// overwrite an existing file unless `force`.
pub fn save_seed(path: &std::path::Path, seed: &[u8; 32], force: bool, allow_drvfs: bool) -> Result<(), ReceiptError>;
/// `$LICTOR_KEYS` if set, else `$HOME/.lictor` (ext4 on this machine) -- NEVER the repo directory.
pub fn default_key_dir() -> std::path::PathBuf;
```

### crates/lictor-runtime (lib.rs: `pub mod wire; pub mod codec; pub mod session; pub mod episode; pub mod latency; pub mod trace;`)

```rust
// wire.rs -- serde types for lictor-wire/v1 (see the WIRE PROTOCOL section).
// `deny_unknown_fields` on the internally-tagged `Request` enum does NOT reject unknown keys inside the variant payloads, so it
// is on EVERY payload struct below as well (fail-closed: an unknown key anywhere in a request is a schema fault).
// Every real-valued REQUEST field is Option<f64> (JSON null == non-finite -> NaN -> GUARD NONFINITE). Real-valued RESPONSE
// fields are f64; serde_json emits `null` for a non-finite f64, and exactly two response fields can be non-finite BY DESIGN:
// `scores.s` (NEG_INFINITY while no gate term is fully valid or Tier 1 is disarmed) and `tau` (+INFINITY when disarmed or k > n),
// same for `hello_ok.calibration.tau`. The Python client maps null -> -inf / +inf for those keys ONLY.
pub const PROTO: &str = "lictor-wire/v1";
#[derive(Debug, Clone, serde::Deserialize)] #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request { Hello(HelloReq), EpisodeBegin(EpisodeBeginReq), Tick(TickReq), EpisodeEnd(EpisodeEndReq), Bye(ByeReq) }
#[derive(Debug, Clone, serde::Serialize)] #[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response { HelloOk(HelloOk), EpisodeOk(EpisodeOk), Verdict(Box<VerdictMsg>), EpisodeReceipt(EpisodeReceiptMsg), ByeOk { id: u64 }, Error(ErrorMsg) }
// ---- request payloads (all `#[derive(Debug, Clone, serde::Deserialize)] #[serde(deny_unknown_fields)]`; keys == the WIRE PROTOCOL JSON)
pub struct HelloReq { pub id: u64, pub proto: String, pub client: String, pub mode: FuseMode, pub embodiment_id: String, pub action_dim: u16,
    pub pos_dim: u16, pub horizon: u16, pub exec_steps: u16, pub envelope_digest: String, pub calibration_digest: Option<String> }
pub struct RunMsg { pub run_id: String, pub arm_id: String, pub episode_index: u32, pub seed: u64, pub seed_pool: String, pub init_state_digest: String }
pub struct BudgetMsg { pub delay_steps: u16, pub tick_ms: u32, pub exec_mode: ExecMode, pub stitch: String, pub on_escalate: String }
pub struct BindingMsg { pub env: BTreeMap<String, String>, pub policy: BTreeMap<String, String>, pub host: BTreeMap<String, String> }
pub struct FaultInjectionMsg { pub kind: String, pub params: BTreeMap<String, String>, pub stream_seed: u64 }
pub struct EpisodeBeginReq { pub id: u64, pub run: RunMsg, pub budget: BudgetMsg, pub binding: BindingMsg, pub fault_injection: Option<FaultInjectionMsg>,
    #[serde(default)] pub inputs: BTreeMap<String, String> }   // host-declared content hashes: repo-relative path -> sha256 (absent == empty)
pub struct ObsMsg { pub pos: Vec<Option<f64>>, pub vel: Option<Vec<Option<f64>>>, pub aux: Vec<Option<f64>>, pub ext: Vec<Option<f64>> }
pub struct ChunkMsg { pub seq: u32, pub t_emit: u32, pub h: u16, pub d: u16, pub exec: u16, pub a: Vec<Vec<Option<f64>>> }
pub struct TickReq { pub id: u64, pub t: u32, pub idx: u16, pub missed_ticks: u8, pub obs: ObsMsg, pub chunk: Option<ChunkMsg>, pub ack: Option<AckToken> }
pub struct OutcomeMsg { pub steps: u32, pub success: bool, pub terminated: bool, pub truncated: bool, pub max_coverage: Option<f64>,
    pub final_coverage: Option<f64>, pub reward_sum: Option<f64>, pub ended_by: String }
pub struct EpisodeEndReq { pub id: u64, pub t: u32, pub outcome: OutcomeMsg }
pub struct ByeReq { pub id: u64 }
// ---- response payloads (all `#[derive(Debug, Clone, serde::Serialize)]`)
pub struct CalibrationInfo { pub method: CalMethod, pub alpha_num: u32, pub alpha_den: u32, pub n_calib: u32, pub tau: f64, pub kn: [u8; 2], pub gate: Vec<String>, pub digest: String }
pub struct HelloOk { pub id: u64, pub proto: String, pub lictor: String, pub git: String, pub lictor_sha256: String, pub envelope_digest: String,
    pub embodiment_digest: String, pub calibration_digest: Option<String>, pub pubkey: String, pub ephemeral_key: bool, pub mode: FuseMode,
    pub tier0_armed: Vec<String>, pub tier1_armed: bool, pub calibration: Option<CalibrationInfo>, pub features: Vec<String>, pub trip_names: Vec<String>,
    pub latency_label: String }
pub struct EpisodeOk { pub id: u64, pub state: FuseState, pub seq: u32 }
pub struct ScoresMsg { pub f: Vec<f64>, pub z: Vec<f64>, pub s: f64, pub valid: u32, pub fired: u32 }
pub struct VerdictMsg { pub id: u64, pub t: u32, pub seq: u32, pub status: Status, pub state: FuseState, pub prev_state: FuseState, pub trips: Vec<String>,
    pub trip_mask: u32, pub action: Vec<f64>, pub action_src: ActionSource, pub substituted: bool, pub clamped_dims: u32, pub scores: ScoresMsg, pub tau: f64,
    pub window_hits: u8, pub brake_margin: f64, pub reason: ReasonCode, pub reason_text: String, pub handoff: Option<HandoffRecord>,
    pub ack_result: Option<String>, pub violation_reached_env: bool, pub verdict_ns: u64, pub chain: String }
pub struct EpisodeReceiptMsg { pub id: u64, pub receipt_path: String, pub ticks_path: String, pub body_digest: String, pub verdict_chain_head: String,
    pub timing_chain_head: String, pub fuse_ok: bool, pub fuse_notes: Vec<String>, pub counts: VerdictCounts, pub ledger_seq: u32, pub ledger_head: String }
pub struct ErrorMsg { pub id: u64, pub code: String, pub message: String, pub fatal: bool }

// codec.rs
pub const MAX_LINE: usize = 1 << 20;
pub fn read_request(r: &mut impl std::io::BufRead, buf: &mut String) -> std::io::Result<Option<Request>>;  // None at EOF; Err(InvalidData) on malformed
pub fn write_response(w: &mut impl std::io::Write, resp: &Response) -> std::io::Result<()>;              // one line + flush

// session.rs -- one fuse, one episode at a time; shared by serve, replay, bench, selftest, and the fuse crate's alloc/determinism tests
/// A calibration ALREADY loaded and compiled by the caller (lictor-cli via lictor-calib). lictor-runtime does NOT depend on
/// lictor-calib (lictor-calib depends on lictor-runtime for the wire/trace types).
pub struct CalibrationLoaded { pub c: CalibrationC, pub digest: String, pub embodiment_digest: String, pub file_sha256: String, pub path: String }
pub struct SessionConfig { pub envelope: SafetyEnvelope, pub envelope_toml_sha: String, pub calibration: Option<CalibrationLoaded>,
    pub mode: FuseMode, pub tier0_override: Option<u32>, pub tier1: bool, pub ticks_policy: String, pub key: Option<[u8; 32]>,
    pub out_dir: Option<std::path::PathBuf>, pub trace: Option<std::path::PathBuf>, pub lictor_git: String, pub lictor_sha256: String,
    pub latency_label: String }   // default: `default_latency_label()`
/// Off-path staging: converts a TickReq into a TickInput with buffers allocated ONCE (null -> NaN; chunk rows -> ChunkBuf).
/// Used by Session and by bench/alloc/determinism tests, so there is exactly ONE wire -> TickInput conversion in the workspace.
pub struct Staging { /* pos/vel/aux/ext: [f64; MAX_*] + lengths, have_vel, chunk: ChunkBuf, have_chunk */ }
impl Staging {
    pub fn new() -> Self;
    /// Copies the request into the buffers. Err(message) on a dimension / horizon / idx / seq violation (the caller raises SCHEMA).
    pub fn stage(&mut self, cfg: &FuseConfig, req: &TickReq) -> Result<(), String>;
    /// Borrow the staged buffers as the pure input. `schema_fault` forces TripMask::SCHEMA in GUARD.
    pub fn input(&self, req: &TickReq, ack: Option<VerifiedAck>, schema_fault: bool) -> TickInput<'_>;
}
impl Default for Staging { fn default() -> Self { Self::new() } }
pub struct Session { /* Fuse + Staging + chains + timing + episode assembly + pending handoff + last_nonce (kept ACROSS episodes) */ }
impl Session {
    /// Refuses (Err) a calibration whose `embodiment_digest` != `cfg.envelope.embodiment_digest()` -- fatal at startup, never a note.
    pub fn new(cfg: SessionConfig) -> anyhow::Result<Self>;
    /// Handle one request; measures decide_ns around `decide()` and folds the chains AFTER it returns (never inside).
    /// `episode_end` is ACCEPTED while faulted (writes a receipt with terminal_state = fault, fuse_ok = false, ended_by as sent).
    pub fn handle(&mut self, req: Request) -> Response;
    /// Called by the serve loop after the response is flushed: wall-clock from request-line read to response flush, recorded into
    /// the timing chain entry of tick `seq` (the entry is emitted on the NEXT request or at episode_end; never inside decide()).
    pub fn note_io_ns(&mut self, seq: u32, ns: u64);
    pub fn fault_latched(&self) -> bool;
    pub fn verdict_chain_head(&self) -> &str;
    pub fn latency(&self) -> &latency::Hist;
}
/// WSL2_LABEL when /proc/version contains "microsoft" or "WSL", else "measured on <uname -sr>, non-RT kernel -- not a real-time environment".
pub fn default_latency_label() -> String;

// episode.rs -- assemble ReceiptBody from bindings + Tally + chains; sign; write receipt/ticks/timing; append ledger LAST
pub struct EpisodePaths { pub receipt: std::path::PathBuf, pub ticks: std::path::PathBuf, pub timing: std::path::PathBuf, pub ledger: std::path::PathBuf }
pub fn episode_paths(out_dir: &std::path::Path, run_id: &str, arm_id: &str, episode_index: u32) -> EpisodePaths;
pub fn write_episode(paths: &EpisodePaths, sr: &SignedReceipt, ticks: &[TickEvent], timing: &[TimingEvent]) -> anyhow::Result<LedgerEntry>;
/// Host-side crash accounting (`lictor crash-receipt`): when the serve child died before `episode_end`, write a minimal SIGNED receipt
/// with zero counts, `outcome = {steps: 0, success: false, ended_by: "fuse_crash"}`, `terminal_state: fault`, `fuse_ok: false`,
/// `fuse_notes: ["fuse process died before episode_end; host-written crash receipt"]`, empty ticks/timing files, and append the ledger.
pub fn write_crash_episode(cfg: &SessionConfig, run: RunBinding, budget: BudgetBinding, client: &str, note: &str) -> anyhow::Result<LedgerEntry>;

// latency.rs
pub struct Hist { /* hdrhistogram::Histogram<u64>, 1ns..10s, 3 sig figs */ }
impl Hist { pub fn new() -> Self; pub fn record(&mut self, ns: u64); pub fn summary(&self, label: &str) -> LatencySummary; pub fn to_csv(&self) -> String; }
impl Default for Hist { fn default() -> Self { Self::new() } }
pub const WSL2_LABEL: &str = "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment";

// trace.rs -- request trace files: "#meta {json}" first line, then every request line VERBATIM
pub struct TraceWriter; impl TraceWriter { pub fn open(p: &std::path::Path, meta: &serde_json::Value) -> std::io::Result<Self>; pub fn line(&mut self, raw: &str) -> std::io::Result<()>; }
pub struct TraceReader; impl TraceReader { pub fn open(p: &std::path::Path) -> std::io::Result<(Self, serde_json::Value)>; }
impl Iterator for TraceReader { type Item = std::io::Result<(u64 /*line_no*/, String /*raw*/, Request)>; }
```

### crates/lictor-calib (lib.rs: `pub mod file; pub mod traces; pub mod calibrate; pub mod envfit; pub mod metrics; pub mod sweep; pub mod curve;`)

```rust
// file.rs -- calibration.json (schema lictor-calibration/v1), float-free; see FILE FORMATS
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SeedPool { pub name: String, pub lo: u64, pub hi: u64 }   // inclusive range
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tier0Percentiles { pub v_max: F64Hex, pub a_max: F64Hex, pub j_max: F64Hex, pub reach_max: F64Hex, pub q: F64Hex }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibrationFile { pub schema: String, pub canonical: String, pub method: CalMethod, pub alpha_num: u32, pub alpha_den: u32,
    pub n_total: u32,      // successful calibration episodes available
    pub n_scale: u32,      // episodes used to fit center/scale
    pub n_calib: u32,      // per-episode scores tau was taken over (2-way: == n_scale; 3-way: a disjoint set) -- k = ceil((n_calib+1)(1-alpha))
    pub n_holdout: u32, pub split: String,   // "2way" | "3way"
    pub seed_pool: SeedPool, pub source_run: String, pub source_arm: String,
    pub envelope_digest: String, pub embodiment_digest: String, pub policy_digest: String, pub lictor_git: String,
    pub horizon_ticks: u32, pub t_grid: u16, pub feature_ids: Vec<String>, pub mask: u32, pub gate: Vec<String>,
    pub center: F64Array, pub scale: F64Array, pub tau: F64Hex, pub kn: [u8; 2], pub warn_margin: F64Hex,
    pub holdout_fpr: F64Hex, pub holdout_fpr_k1: F64Hex, pub tier0_percentiles: Option<Tier0Percentiles>, pub notes: Vec<String>, pub digest: String }
impl CalibrationFile { pub fn load(p: &std::path::Path) -> anyhow::Result<Self>; pub fn save(&self, p: &std::path::Path) -> anyhow::Result<()>;
                       pub fn compile(&self) -> anyhow::Result<CalibrationC>; pub fn digest_hex(&self) -> anyhow::Result<String>;
                       /// -> lictor_runtime::session::CalibrationLoaded (file_sha256 over the bytes on disk)
                       pub fn loaded(&self, path: &std::path::Path) -> anyhow::Result<lictor_runtime::session::CalibrationLoaded>; }
// traces.rs -- one Trace per episode, built from the ticks file + receipt (labels) + the request trace (coverage), pool-filtered.
// Trace lines are parsed with lictor_runtime::trace::TraceReader (the ONE wire parser); never a second serde_json::Value parser.
pub struct Trace { pub arm_id: String, pub seed: u64, pub episode_index: u32, pub success: bool, pub steps: u32,
                   pub coverage: Vec<f64>, pub f: Vec<[f64; NFEAT]>, pub valid: Vec<u32>, pub first_stop_tick: Option<u32>, pub init_state_digest: String }
pub fn load_traces(run_dir: &std::path::Path, arm_id: &str) -> anyhow::Result<Vec<Trace>>;   // verifies each receipt + ticks chain first; refuses broken ones
// calibrate.rs
pub struct CalibrateOpts { pub alpha_num: u32, pub alpha_den: u32, pub method: CalMethod, pub gate: Vec<String>, pub holdout_num: u32, pub holdout_den: u32,
                           pub kn: [u8; 2], pub warn_margin: f64, pub n_calib_cap: Option<u32>, pub horizon_ticks: u32,
                           pub split: u8 }   // 2: center/scale AND tau on the fit subset (bound approximate); 3: center/scale on A (40 %), tau on B (30 %), holdout C (30 %) (bound exact at K = 1)
pub fn robust_center_scale(fit: &[Trace], method: CalMethod, horizon_ticks: u32) -> ([[f64; NFEAT]; T_GRID], [[f64; NFEAT]; T_GRID], u16);
pub fn episode_max_s(tr: &Trace, cal: &CalibrationC) -> f64;
pub fn split_quantile(sorted_max_s: &[f64], alpha_num: u32, alpha_den: u32) -> (f64, Option<String>);  // (tau, note); +INF when k > n
pub fn calibrate(traces: &[Trace], opts: &CalibrateOpts, envelope: &SafetyEnvelope, policy_digest: &str, source_run: &str) -> anyhow::Result<CalibrationFile>;   // binds envelope.digest_hex() AND envelope.embodiment_digest()
// envfit.rs
pub struct FitOpts { pub quantile: f64, pub slack: f64 }
pub fn fit_envelope(base: &SafetyEnvelope, traces: &[Trace], opts: &FitOpts, operators: &[String]) -> anyhow::Result<(SafetyEnvelope, String /*report md*/)>;   // `operators` replaces base.operators (the oracle envelope); embodiment untouched
// metrics.rs
pub fn clopper_pearson(k: u32, n: u32, conf: f64) -> (f64, f64);
pub fn mcnemar_exact(b: u32, c: u32) -> f64;
pub fn ponr_tick(coverage: &[f64], eps_prog: f64) -> u32;      // t_fail = min{t : c*_T - c*_t < eps}
pub fn roc_auc(pos: &[f64], neg: &[f64]) -> f64;
pub fn aucpdt(leads: &[i64], horizon: u32) -> f64;
pub fn paired_bootstrap_diff(a: &[bool], b: &[bool], n_resamples: u32, seed: u64) -> (f64, f64, f64);   // (diff, lo, hi); RNG = splitmix64 (in-crate)
// sweep.rs
pub struct DetectorSpec { pub name: String, pub gate: Vec<String>, pub kn: [u8; 2], pub tier0: bool }
pub struct SweepPoint { /* exactly the sweep.jsonl keys in FILE FORMATS */ }
/// `artefacts`: calibration.<alpha>.json files already produced by `lictor calibrate`; when an (alpha, gate, kn, method) matches one,
/// its center/scale/tau are used VERBATIM (`tau_source: "artefact"`) so the Layer-A point is comparable to the Layer-B arm;
/// otherwise a full fit with no holdout is done (`tau_source: "fit"`).
pub struct SweepOpts { pub method: CalMethod, pub horizon_ticks: u32, pub eps_prog: f64 }   // WP-0 amendment: bundled so `sweep` stays within clippy::too_many_arguments
pub fn sweep(calib: &[Trace], eval: &[Trace], alphas: &[(u32, u32)], dets: &[DetectorSpec], opts: &SweepOpts,
             artefacts: &[CalibrationFile]) -> Vec<SweepPoint>;
// curve.rs -- closed-loop metrics from receipts ONLY
pub struct CurveOpts { pub eps_prog: f64, pub latency_control: Option<String>, pub calib_seed_override: Option<(u64, u64)>, pub allow_small: bool,
                       pub partial: bool, pub compare_run: Option<std::path::PathBuf>, pub n_boot: u32, pub boot_seed: u64 }   // 10_000, 20260830
/// Refuses (Err, exit 1 in the CLI) when: the ledger is broken; the arm's receipts carry >1 pubkey or an ephemeral-key note; the arm's
/// budget.tier1_armed / alpha disagree with run.json; the arm's episode-index set != the baseline's or != the declared pool (unless
/// `partial`); n < 100 (unless `allow_small`). Missing indices count as success = false, stopped = true (terminate_fail accounting).
pub fn curve(run_dir: &std::path::Path, arm: &str, baseline: &str, opts: &CurveOpts) -> anyhow::Result<CurveReceiptBody>;
```

## A.2 Wire protocol

## WIRE PROTOCOL -- `lictor-wire/v1` (normative)

Transport: one `lictor serve` child process per harness worker, spawned by Python; requests on the child's stdin, responses on its stdout, diagnostics on stderr. Rust writes NOTHING to stdout except response lines.

Framing: UTF-8, exactly one JSON object per line, `\n`-terminated, no embedded newlines, no `\r`, max line 1 MiB. Requests carry `id: u64`, strictly increasing from 1; every response echoes `id`. Strictly alternating request -> response.

Numbers: plain JSON numbers, IEEE-754 f64. Both serde_json (ryu) and CPython `repr` emit shortest-round-trip decimals and both parsers are correctly rounded, so f64 survives the boundary EXACTLY. Python MUST send `np.asarray(x, dtype=np.float64).tolist()` (never a float32 repr). A non-finite value MUST be sent as JSON `null`; the fuse treats `null` in any numeric slot as NaN -> `nonfinite` -> Fault. Python converts non-finite values to `None` explicitly (`np.isfinite`) and serialises with `json.dumps(..., allow_nan=False)`, so a NaN can never leak as the non-JSON literal `NaN` (serde_json rejects it -- still fail-closed, but with a misleading schema message). In RESPONSES exactly two real fields may be non-finite BY DESIGN and are emitted as `null`: `scores.s` (`-inf` while no gate term is fully valid, i.e. every tick before the first chunk with `L >= 2`, and always while Tier 1 is disarmed) and `tau` (`+inf` when disarmed or `k > n`); `hello_ok.calibration.tau` likewise. The client maps `null -> -math.inf` / `+math.inf` for those keys ONLY; every other response number is finite by construction (`session_episode.rs` asserts the first tick's `s` round-trips). (Design decision: the wire is never hashed, so readability wins; the receipt/tick/calibration files are hashed, so THEY are float-free.)

Error discipline (fail-closed): any malformed line, unknown field at ANY nesting level (every payload struct is `deny_unknown_fields`), dimension mismatch, non-monotonic `id`, `tick` before `episode_begin`, `idx >= horizon`, an `idx`/`t_emit` inconsistency (below), a time discontinuity (`t != prev_t + 1 + missed_ticks`; the first tick of an episode has `t == 0`) or a chunk-sequence discontinuity (`chunk.seq != next_chunk_seq`; the first chunk has `seq == 0`) -> an `error` response, `TripMask::SCHEMA`, and the fuse latches `FuseState::Fault` for the rest of the episode; every subsequent `tick` returns `status:"fault"`, `action_src:"hold"` and the hold action. It never returns the raw policy action after a fault. The two continuity checks are integer compares inside `decide()` GUARD (against `FuseRt.last_t` / `FuseRt.next_chunk_seq`), so a host that lies about time cannot silently corrupt `v_hat` or the trail. The host treats `fatal:true` as episode-abort: it stops stepping the env and MUST STILL send `episode_end` with `ended_by:"fault"`; the Session ACCEPTS `episode_end` while faulted and writes a receipt with `terminal_state:"fault"`, `fuse_ok:false`, so the episode enters the ledger as a failure instead of vanishing from `lictor curve`.

Host contract (fail-closed at the host boundary): the Python side (a) sends exactly one `tick` per `env.step`, (b) executes `verdict.action` and nothing else, (c) treats ANY of {read timeout, `error.fatal`, child exit, `id` mismatch} as a brake: for `ee_position`/`joint_position` kinds the fallback is ALWAYS `clamp_box(current measured position)` (never a stale `brake`/`hold` setpoint latched ticks ago -- on a real arm that is a lunge), for velocity kinds it is the zero vector; the episode is marked `ended_by:"fault"`. "Stale last policy action" and "raw policy action" are never the timeout path. (d) On a `LictorFault` raised for {timeout, child exit, `id` mismatch} the client KILLS the child and marks itself dead; the harness respawns a fresh `lictor serve` and re-`hello`s before the next episode (a late verdict for id N must never be read as the reply to N+1). If the child is dead the harness runs `lictor crash-receipt` (CLI SURFACE) so the episode is recorded as `ended_by:"fuse_crash"`, `success:false`; retry cap: 1 respawn per episode index, then the arm aborts with a message. (e) After `error.fatal` with a LIVE child the host sends `episode_end` (`ended_by:"fault"`) as above. `--on-fault abort` (exit 3 on the first fatal error) is a debugging aid only and is never used in an experiment arm.

**Trust model (what the fuse verifies vs what the host declares).** The fuse verifies: `mode`, dims, `horizon`/`exec_steps`, the envelope digest, the calibration digest and its `embodiment_digest`, its own binary hash, and every `tick` (schema, finiteness, continuity). EVERYTHING else is the host's declaration, signed by proxy: `binding` (package versions, policy revision, weights hash, `n_action_steps`), `run` (seed, pool, `init_state_digest`), `budget.delay_steps`/`exec_mode`/`stitch`, `fault_injection`, `inputs`, and `outcome.success`. A receipt therefore defends against post-hoc edits by anyone WITHOUT the signing key and against accidental corruption; it does not defend against the experimenter, who holds the key. External anchoring (roadmap 10) is what would change that. Every sentence in these documents of the form "the receipt binds X" means "the receipt binds the host's declaration of X" unless X is in the verified list above.

### hello (once per process)

```json
{"id":1,"kind":"hello","proto":"lictor-wire/v1","client":"lictor_client/0.1.0",
 "mode":"observe",
 "embodiment_id":"gym_pusht/PushT-v0","action_dim":2,"pos_dim":2,"horizon":15,"exec_steps":8,
 "envelope_digest":"7c4a...64hex","calibration_digest":null}
```
`mode` in {"observe","enforce"} must equal the server's `--mode`; `embodiment_id`/dims/horizon/exec_steps must equal the loaded envelope; `envelope_digest` and `calibration_digest` (or null) must equal the loaded artefacts -- any mismatch is a fatal `error` with `code:"envelope"`.

```json
{"id":1,"kind":"hello_ok","proto":"lictor-wire/v1","lictor":"0.1.0","git":"a3f1c9e","lictor_sha256":"...64hex",
 "envelope_digest":"7c4a...","embodiment_digest":"91d0...","calibration_digest":null,"pubkey":"64e8...64hex","ephemeral_key":false,
 "mode":"observe","tier0_armed":["workspace","speed","accel","jerk","reach","brake"],"tier1_armed":false,
 "calibration":null,
 "latency_label":"measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment",
 "features":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak","ext0","ext1","ext2","ext3"],
 "trip_names":["workspace","speed","accel","jerk","reach","contact","brake","nonfinite","schema","watchdog","tier1_cp","clamp_budget","handoff_timeout","operator_abort","brake_timeout","rearm_budget"]}
```
When a calibration is loaded: `"calibration":{"method":"binned","alpha_num":5,"alpha_den":100,"n_calib":137,"tau":3.7142857142857144,"kn":[3,5],"gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],"digest":"e1b8..."}` (`n_calib` = the number of scores `tau` was taken over; `tau` is `null` on the wire when `+inf`). The client asserts `hello_ok.tier1_armed == arm.tier1` and `(alpha_num, alpha_den) == arm.alpha` and raises otherwise, so a `t01-*` arm launched without `--calibration` cannot silently run as a `t0` arm.

### episode_begin (once per episode)

```json
{"id":2,"kind":"episode_begin",
 "run":{"run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","episode_index":7,"seed":7,"seed_pool":"eval",
        "init_state_digest":"3f1c...64hex"},
 "budget":{"delay_steps":0,"tick_ms":100,"exec_mode":"sync","stitch":"drop","on_escalate":"terminate_fail"},
 "binding":{"env":{"env_id":"gym_pusht/PushT-v0","gym_pusht":"<importlib.metadata.version>","gymnasium":"<importlib.metadata.version>",
                   "pymunk":"<importlib.metadata.version>","numpy":"<importlib.metadata.version>",
                   "obs_type":"pixels_agent_pos","control_hz":"10","max_episode_steps":"300","vel_source":"info.vel_agent","coverage_t0":"env.unwrapped._get_coverage()"},
            "policy":{"repo_id":"lerobot/diffusion_pusht","revision":"<HF commit sha>","weights_sha256":"<sha256 of the safetensors actually loaded>","horizon":"16",
                      "n_action_steps":"15","n_obs_steps":"2","num_inference_steps":"100","device":"cuda","dtype":"float32",
                      "normalization_migrated":"true","migration_script":"lerobot/processor/migrate_policy_normalization.py"},
            "host":{"os":"<platform.platform()>","cpu":"<platform.processor()>","gpu":"<torch.cuda.get_device_name(0) or cpu>","torch":"<torch.__version__>","lerobot":"<importlib.metadata.version>",
                    "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONHASHSEED":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}},
 "fault_injection":null,
 "inputs":{"envelopes/pusht.toml":"<sha256>","harness/pusht_rollout.py":"<sha256>","harness/executor.py":"<sha256>","harness/compat.py":"<sha256>"}}
```
Every `binding` value is read at run time (`importlib.metadata.version`, `torch.__version__`, `platform`); the placeholders above are NOT to be pasted (on the dev machine today: gym-pusht 0.1.6, gymnasium 1.3.0, pymunk 6.11.1, numpy 2.2.6, torch 2.7.1+cu126, torchvision 0.22.1+cu126, lerobot 0.6.1 -- newer than the pins the 65.4 % number was published with, which is why the reproduction gate is hard). `fault_injection` is either `null` or `{"kind":"action_spike","params":{"p":"0.02","mag":"180"},"stream_seed":1234567}` -- the receipt binds the host's DECLARATION of injection (a host that injects and sends `null` hides it; see the trust model). `inputs` is the host's content-addressed manifest of the harness files it ran (repo-relative path -> sha256); the Session adds `lictor:bin`, `lictor:envelope` and (when loaded) `lictor:calibration`, computed by the fuse itself. `init_state_digest` = sha256 of the float64 bytes of the env's post-reset observation state (agent_pos, block_pos, block_angle) -- the paired-seed evidence (arms whose digests disagree for an episode index are refused at aggregation).

```json
{"id":2,"kind":"episode_ok","state":"armed","seq":0}
```

### tick (once per env step -- the ONLY hot message)

```json
{"id":31,"kind":"tick","t":24,"idx":0,"missed_ticks":0,
 "obs":{"pos":[213.5,301.0],"vel":[41.2,-12.7],"aux":[256.0,256.0,0.7853981633974483,0.61],"ext":[]},
 "chunk":{"seq":3,"t_emit":24,"h":15,"d":2,"exec":8,
          "a":[[214.0,300.2],[216.1,299.0],[218.0,297.7],[219.6,296.3],[221.0,294.9],[222.2,293.6],[223.3,292.4],[224.3,291.3],
               [225.2,290.3],[226.0,289.4],[226.7,288.6],[227.3,287.9],[227.8,287.3],[228.2,286.8],[228.5,286.4]]},
 "ack":null}
```
`chunk` is non-null ONLY on the tick a new chunk is delivered. On that tick: `t_emit <= t`, `idx == t - t_emit`, `idx < h`, `h == envelope.embodiment.horizon` (the fuse ALWAYS receives the FULL 15-row chunk; the harness never trims rows), `d == action_dim`, `exec == exec_steps`, `seq == next_chunk_seq`. Delivery labelling per exec mode: sync (hold-last during the delay, then execute from row 0) -> `t_emit = t`, `idx = 0`; async `stitch=freeze` (RTC freeze: the d already-planned rows are executed anyway) -> `t_emit = t`, `idx = 0`; async `stitch=drop` (time-aligned: row `d` applies now) -> `t_emit = t - d` (the generation step), `idx = d`. `brake_feasible` is evaluated from row `idx` at delivery. On non-delivery ticks `chunk` is `null` and `idx` indexes the chunk currently executing. Consequence that is DISCLOSED, not hidden: Tier-1 overlap is computed from `t_emit` differences (`s = new.t_emit - prev.t_emit`, `L = min(prev.h - s, new.h)`), so in sync mode `L = 7 - d`: `tce`/`acc` are structurally INVALID for `d >= 7` and degraded for `d in 3..6`, while async arms keep `L = 7`. `lictor curve` binds `tce_valid_frac` per arm and F1 plots it. `obs.vel` is the simulator velocity (`info["vel_agent"]`; the manifest says `provides_vel = true`); at `t = 0` the harness sends `aux[3] = env.unwrapped._get_coverage()` because `reset()` info carries no `coverage`. `ack` is `null` or an `AckToken` object (see FILE FORMATS); it is verified by the runtime before `decide()` runs.

```json
{"id":31,"kind":"verdict","t":24,"seq":24,
 "status":"nominal","state":"armed","prev_state":"armed",
 "trips":[],"trip_mask":0,
 "action":[214.0,300.2],"action_src":"policy","substituted":false,"clamped_dims":0,
 "scores":{"f":[0.012,0.31,-0.021,1.9,0.11,0.08,0.0,0.34,0.0,0.0,0.0,0.0],
           "z":[0.4,0.9,-0.2,0.1,0.7,0.3,0.0,1.1,0.0,0.0,0.0,0.0],
           "s":1.1,"valid":255,"fired":0},
 "tau":3.7142857142857144,"window_hits":0,"brake_margin":41.2,
 "reason":"ok","reason_text":"All checks passed; the policy action is applied unchanged.",
 "handoff":null,"ack_result":null,"violation_reached_env":false,
 "verdict_ns":1180,"chain":"5b7e...64hex"}
```
`scores.s` and `tau` are `null` when non-finite (see Numbers). `handoff` is non-null exactly on the tick `Escalated` is entered and carries a `HandoffRecord` (FILE FORMATS). `ack_result` is `null`, `"accepted"`, or `"rejected:<unknown_operator|bad_signature|wrong_handoff|nonce_replay|no_pending_handoff>"`. `verdict_ns` is measured around `decide()` with the clock read OUTSIDE the pure function. `chain` is the verdict-chain head after this tick.

### episode_end

```json
{"id":331,"kind":"episode_end","t":300,
 "outcome":{"steps":300,"success":false,"terminated":false,"truncated":true,
            "max_coverage":0.71,"final_coverage":0.68,"reward_sum":178.2,"ended_by":"truncated"}}
```
`ended_by` in {"success","truncated","escalation_terminate","fault","abort","retune"} (`"fuse_crash"` is written only by `lictor crash-receipt`, never sent on the wire). `episode_end` is accepted in EVERY session state after `episode_begin`, including a latched Fault; `max_coverage`/`final_coverage`/`reward_sum` may be `null` (non-finite) and are then recorded as NaN hex.

```json
{"id":331,"kind":"episode_receipt",
 "receipt_path":"results/2026-09-01T09-14Z-pilot/t01-a05-d0/receipts/000007.json",
 "ticks_path":"results/2026-09-01T09-14Z-pilot/t01-a05-d0/ticks/000007.jsonl",
 "body_digest":"be21...","verdict_chain_head":"5b7e...","timing_chain_head":"22a0...",
 "fuse_ok":true,"fuse_notes":[],
 "counts":{"ticks":300,"nominal":281,"watching":12,"clamped":0,"braking":4,"held":3,"escalated":0,"fault":0,"terminated":0,
           "substituted":7,"chunks_seen":38,"chunks_rejected":0,"clamps":0,"holds":1,"rearms":1,"escalations":0,
           "first_trip_tick":181,"first_trip_reason":"brake_tier1_cp","first_stop_tick":181,"handoff_tick":null,
           "violations_reached_env":0,"terminal_state":"armed"},
 "ledger_seq":7,"ledger_head":"aa31..."}
```

### bye / error

```json
{"id":332,"kind":"bye"}
{"id":332,"kind":"bye_ok"}
{"id":31,"kind":"error","code":"schema","message":"tick.chunk.a has 14 rows, expected h=15","fatal":true}
```
`code` in {"schema","envelope","state","protocol","internal"}. `fatal:true` for everything except an `ack` rejection (which is reported inside the verdict, not as an error).

### Tracing and replay

`lictor serve --trace FILE` appends a first line `#meta {"proto":"lictor-wire/v1","envelope_digest":"...","calibration_digest":null,"mode":"observe","lictor":"0.1.0","git":"..."}` and then EVERY request line verbatim (responses are recomputable -- that IS the determinism claim). `lictor replay FILE` re-feeds the request lines through a fresh Session and compares the verdict-chain head.

### Python client (`adapters/lictor_client.py`)

```python
CLIENT_VERSION = "lictor_client/0.1.0"          # sent as hello.client; bound into every receipt as body.client

class LictorFault(Exception):
    """Raised on timeout / fatal error / child exit / id mismatch. The child has been KILLED and the client is dead."""
    last_safe_action: list[float] | None   # clamp_box(current measured position) computed by the caller-supplied `safe_action` callback, or None
    reason: str                            # "timeout" | "fatal" | "exit" | "id_mismatch" | "protocol"

class LictorClient:
    def __init__(self, binary: str | Path, envelope: Path, calibration: Path | None = None, mode: str = "observe",
                 out_dir: Path | None = None, trace: Path | None = None, key: Path | None = None,
                 tier0: str | None = None, no_tier1: bool = False, ticks: str = "tail32", timeout_s: float = 5.0,
                 safe_action=None, latency_label: str | None = None) -> None: ...
    def hello(self, embodiment_id: str, action_dim: int, pos_dim: int, horizon: int, exec_steps: int,
              envelope_digest: str, calibration_digest: str | None,
              expect_tier1: bool, expect_alpha: tuple[int, int] | None) -> dict: ...   # asserts tier1_armed / alpha, raises otherwise
    def episode_begin(self, run: dict, budget: dict, binding: dict, fault_injection: dict | None = None,
                      inputs: dict[str, str] | None = None) -> dict: ...
    def tick(self, t: int, idx: int, pos, aux, chunk: dict | None = None, vel=None, ext=(), missed_ticks: int = 0,
             ack: dict | None = None) -> dict: ...   # maps null -> -inf for scores.s and +inf for tau; nothing else
    def episode_end(self, outcome: dict) -> dict: ...
    def close(self) -> None: ...
    @property
    def alive(self) -> bool: ...

def chunk_msg(seq: int, t_emit: int, a: np.ndarray, exec_steps: int) -> dict: ...   # a: (h, d) float64 -> {"seq","t_emit","h","d","exec","a"} with NaN -> None
def finite_or_none(x) -> list: ...                                                    # np.isfinite mask -> None; used by every numeric field
# every request is serialised with json.dumps(obj, allow_nan=False, separators=(",", ":")) after finite_or_none
```
Includes `win_to_wsl(path)` / `wsl_to_win(path)` bridging (bulla_mcp.py style) and auto-detection of a `lictor.exe` under `/mnt/c/...` when `lictor` is absent from `$PATH` and `$LICTOR_BIN` is unset (insurance only; a Windows-native build compiles because `keys.rs` guards `PermissionsExt` with `#[cfg(unix)]`).

## A.3 File formats

## FILE FORMATS (normative)

### Canonical encoding profile `jcs-floatfree/v1` (applies to every file below EXCEPT the envelope TOML, the wire trace and the harness index)

1. RFC 8785 JCS: object keys sorted by UTF-16 code-unit sequence, no whitespace, minimal escaping, integers rendered as plain digits.
2. Float-free: the JSON text contains NO non-integer numbers. A real number is `{"f64":"<16 lowercase hex of IEEE-754 bits>"}`; an array of reals is `{"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[...]}`. Integers must satisfy |v| <= 2^53-1.
3. Every key matches `[a-z0-9_]+`.

Consequence: Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()` over `json.loads(file)` is BYTE-IDENTICAL to lictor's canonical bytes. That is what makes `adapters/verify_receipt.py` a stdlib-only verifier. Files on disk are written pretty-printed (2-space indent, keys sorted) for humans; the canonical bytes are always recomputed, never trusted from disk.

Digests: `sha256(canon(value))`, hex64. `body_digest` of a signed object = sha256 over canon(body). Ed25519 signature = over the same canonical bytes (`verify_strict`). Self-digesting objects (calibration.json, HandoffRecord) compute their digest over canon(self with the digest field set to "").

### Envelope TOML -- `envelopes/pusht.base.toml` (schema `lictor-envelope/v1`)

```toml
schema = "lictor-envelope/v1"
envelope_id = "pusht-base-v1"
# Base values are PLACEHOLDERS: loose enough not to touch the baseline. `lictor envelope fit` writes envelopes/pusht.toml.
box_lo = [15.0, 15.0]          # agent radius 15 px; env box is [0,512]^2
box_hi = [497.0, 497.0]
margin = 2.0
v_max = 1000.0                 # px/s   (100 px per 0.1 s control step)
a_max = 20000.0                # px/s^2
j_max = 400000.0               # px/s^3
reach_max = 150.0              # px, ||a_0 - p_t||
clamp_mode = "project"
tier0_enabled = ["workspace", "speed", "accel", "jerk", "reach", "brake"]   # "contact" is perception-assisted; off by default
gate = ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak"]   # plain max over 8 channels (calibration.json may override)
operators = []                 # hex64 Ed25519 pubkeys allowed to sign an AckToken. NEVER edited at run time by Python: the oracle arm
                               # uses envelopes/pusht.oracle.toml, produced by `lictor envelope fit ... --operator <hex>` (same embodiment,
                               # hence the same embodiment_digest and the same calibration; a different envelope_digest, printed by verify)
fail_closed = true

[embodiment]
id = "gym_pusht/PushT-v0"
units = "px"
action_kind = "ee_position"
action_dim = 2
pos_dim = 2
horizon = 15                   # n_action_steps = horizon(16) - n_obs_steps(2) + 1 = 15 (see ARCHITECTURE sec 0)
exec_steps = 8                 # stock n_action_steps -- the executed prefix is identical to published LeRobot
control_hz_num = 10
control_hz_den = 1
provides_vel = true            # gym_pusht info["vel_agent"] every step -> obs.vel; the PD brake rollout starts from the true v0
horizon_ticks = 300            # == max_episode_steps; the calibration time base (`lictor calibrate` refuses a mismatch)
norm_center = [256.0, 256.0]
norm_scale = [256.0, 256.0]
aux_layout = ["block_x", "block_y", "block_theta", "coverage"]
ext_names = []

[brake]
kind = "pd_second_order"       # gym-pusht agent: acc = k_p (a - p) - k_v v ; v += acc dt ; p += v dt ; 10 substeps of dt = 0.01
k_p = 100.0
k_v = 20.0
substeps = 10
dt = 0.01
commit_steps = 8
brake_steps = 8
react_ticks = 1

[hysteresis]
k = 3
n = 5
warn_margin = 0.5
clear_ticks = 5
clamp_streak_to_brake = 3
max_clamps_per_episode = 60
stop_confirm_ticks = 2
v_stop_eps = 5.0
brake_timeout_ticks = 30
rearm = "auto"
rearm_hold = 10
max_rearms = 2
escalate_after_hold_ticks = 30
handoff_timeout_ticks = 200
watchdog_ticks = 2

# [contact]  (optional, disarmed unless listed in tier0_enabled)
# radius = 80.0
# v_max = 400.0
# aux_center = [0, 1]

# [fit]  (written by `lictor envelope fit`)
# source_run = "2026-09-01T09-14Z-pilot"
# quantile = 0.999
# slack = 1.25
# n_episodes = 196
# fitted_utc = "2026-09-01T11:02:00Z"
# note = "p99.9 of |v|,|a|,|j|,reach over calibration successes x 1.25"
```
Envelope digest = sha256(canon(floatify(json(envelope)))) where `json(envelope)` is serde's JSON of `SafetyEnvelope` (struct field names as keys, TOML tables become nested objects, `contact`/`fit` absent -> `null`). Embodiment digest = the same over `json(envelope.embodiment)` alone; a calibration binds THAT, so `envelope fit` (which changes `v_max`..`reach_max`) and the oracle operator list do not invalidate it.

### calibration.json (schema `lictor-calibration/v1`, float-free)

```json
{"schema":"lictor-calibration/v1","canonical":"jcs-floatfree/v1",
 "method":"binned","alpha_num":5,"alpha_den":100,
 "n_total":196,"n_scale":137,"n_calib":137,"n_holdout":59,"split":"2way",
 "seed_pool":{"name":"calib","lo":900000,"hi":900299},
 "source_run":"2026-09-01T09-14Z-pilot","source_arm":"calib-obs",
 "envelope_digest":"7c4a...","embodiment_digest":"91d0...","policy_digest":"<weights sha256>","lictor_git":"a3f1c9e",
 "horizon_ticks":300,"t_grid":100,
 "feature_ids":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak","ext0","ext1","ext2","ext3"],
 "mask":255,
 "gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],
 "center":{"f64a":"...","shape":[100,12]},
 "scale":{"f64a":"...","shape":[100,12]},
 "tau":{"f64":"400db6db6db6db6e"},
 "kn":[3,5],
 "warn_margin":{"f64":"3fe0000000000000"},
 "holdout_fpr":{"f64":"3fa47ae147ae147b"},
 "holdout_fpr_k1":{"f64":"3fb0000000000000"},
 "tier0_percentiles":null,
 "notes":[],
 "digest":"<sha256 over canon(self with digest = \"\")>"}
```
`n_calib` is ALWAYS the number of per-episode scores `tau` was taken over (`k = ceil((n_calib + 1)(1 - alpha))`; a 2-way split of 196 successes gives 137, not 196); `n_scale` is the number of episodes center/scale were fitted on; `n_total` the successes available. `split` is `"2way"` (default: `n_scale == n_calib`, the bound is approximate because the standardisation was fitted on the same episodes) or `"3way"` (`--split 3`: standardise on A, `tau` on B, holdout on C; exact bound at K = 1, fewer scores per part). `tau` is `{"f64":"7ff0000000000000"}` (+inf) when `k > n_calib` -- the file records the degenerate point honestly. `holdout_fpr` is the empirical episode-level firing rate on the held-out subset with the configured K-of-N; `holdout_fpr_k1` with K=1 (the rate alpha actually bounds). `tier0_percentiles` is non-null only when produced together with `envelope fit` (a `Tier0Percentiles` object: `{"v_max":{"f64":..},"a_max":..,"j_max":..,"reach_max":..,"q":..}`). `embodiment_digest` MUST equal the served envelope's (`lictor serve`/`replay` refuse otherwise) and `policy_digest` MUST equal `binding.policy.weights_sha256` at `episode_begin` (fatal `error{code:"envelope"}` otherwise).

### Ticks file -- `results/<run>/<arm>/ticks/<episode:06>.jsonl` (schema `lictor-ticks/v1`)

First line: `{"schema":"lictor-ticks/v1","run_id":"...","arm_id":"...","episode_index":7,"genesis":"0000...0000"}`. Then one `TickEvent` per line, canonical (compact, sorted keys), e.g.:

```json
{"action":{"f64a":"AAAAAADAKkBmZmZmZsZyQA==","shape":[2]},"action_src":"policy","brake_margin":{"f64":"404499999999999a"},"clamped_dims":0,"f":{"f64a":"...","shape":[12]},"fired":0,"handoff_seq":null,"hash":"5b7e...","prev":"0000...","prev_state":"armed","reason":"ok","s":{"f64":"3ff199999999999a"},"seq":24,"state":"armed","status":"nominal","substituted":false,"t":24,"tau":{"f64":"400db6db6db6db6e"},"trips":0,"valid":255,"violation_reached_env":false,"window_hits":0,"z":{"f64a":"...","shape":[12]}}
```
`hash = sha256(canon(event with "hash":""))`; `prev` of seq 0 is the genesis (64 zeros). Companion `timing/<episode:06>.jsonl` holds `TimingEvent`s with the same rule.

### Receipt -- `results/<run>/<arm>/receipts/<episode:06>.json` (schema `lictor-receipt/v1`)

```json
{"body":{
   "schema":"lictor-receipt/v1","canonical":"jcs-floatfree/v1","created_epoch":1788327242,
   "lictor_version":"0.1.0","lictor_git":"a3f1c9e","lictor_sha256":"...","client":"lictor_client/0.1.0",
   "run":{"run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","episode_index":7,"seed":7,"seed_pool":"eval","init_state_digest":"3f1c...",
          "env":{"env_id":"gym_pusht/PushT-v0","gym_pusht":"<from importlib.metadata>","gymnasium":"<from importlib.metadata>","pymunk":"<from importlib.metadata>","numpy":"<from importlib.metadata>","obs_type":"pixels_agent_pos","control_hz":"10","max_episode_steps":"300","vel_source":"info.vel_agent","coverage_t0":"env.unwrapped._get_coverage()"},
          "policy":{"repo_id":"lerobot/diffusion_pusht","revision":"...","weights_sha256":"...","horizon":"16","n_action_steps":"15","n_obs_steps":"2","num_inference_steps":"100","device":"cuda","dtype":"float32","normalization_migrated":"true","migration_script":"lerobot/processor/migrate_policy_normalization.py"},
          "host":{"os":"<platform.platform()>","cpu":"...","gpu":"NVIDIA GeForce GTX 1060 3GB","torch":"2.7.1+cu126","lerobot":"0.6.1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONHASHSEED":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}},
   "budget":{"mode":"enforce","delay_steps":0,"tick_ms":100,"exec_mode":"sync","stitch":"drop","on_escalate":"terminate_fail",
             "tier0_armed":["workspace","speed","accel","jerk","reach","brake"],"tier1_armed":true,
             "gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],"alpha_num":5,"alpha_den":100,"kn":[3,5]},
   "fault_injection":null,
   "envelope":{ "...floatified SafetyEnvelope..." },
   "envelope_digest":"7c4a...","calibration_digest":"e1b8...",
   "inputs":{"envelopes/pusht.toml":"...","harness/pusht_rollout.py":"...","harness/executor.py":"...","harness/compat.py":"...",
             "lictor:bin":"...","lictor:envelope":"...","lictor:calibration":"..."},
   "counts":{"ticks":300,"nominal":281,"watching":12,"clamped":0,"braking":4,"held":3,"escalated":0,"fault":0,"terminated":0,"substituted":7,
             "chunks_seen":38,"chunks_rejected":0,"clamps":0,"holds":1,"rearms":1,"escalations":0,
             "trips_by_bit":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],"fired_by_feat":[3,1,0,0,0,9,11,0,0,0,0,0],
             "first_trip_tick":181,"first_trip_reason":"brake_tier1_cp","first_stop_tick":181,"handoff_tick":null,
             "violations_reached_env":0,"terminal_state":"armed"},
   "outcome":{"steps":300,"success":false,"terminated":false,"truncated":true,"max_coverage":{"f64":"3fe6b851eb851eb8"},
              "final_coverage":{"f64":"3fe5c28f5c28f5c3"},"reward_sum":{"f64":"4066466666666666"},"ended_by":"truncated",
              "max_s":{"f64":"4010a3d70a3d70a4"},"max_z":{"f64a":"...","shape":[12]}},
   "handoffs":[],
   "verdict_events":300,"verdict_chain_head":"5b7e...","timing_events":300,"timing_chain_head":"22a0...",
   "latency":{"n":300,"p50_ns":1180,"p90_ns":1400,"p99_ns":2600,"p999_ns":4400,"max_ns":61200,
              "label":"measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment"},
   "ticks_policy":"tail32","ticks":[ "...last 32 TickEvents..." ],
   "fuse_ok":true,"fuse_notes":[],
   "ledger_prev":"9c02..."},
 "body_digest":"be21...","pubkey":"64e8...","sig":"c4a1...128hex"}
```
`fuse_ok` rules (`evaluate_fuse`): false with a note when `mode == observe` ("the fuse observed but did not enforce -- this receipt does not attest protection"); when `violations_reached_env > 0`; when `tier1_armed` with no `calibration_digest`; when `terminal_state == fault` (including host-written `fuse_crash` receipts); when `delay_steps > 0` and `exec_mode` missing; when the signing key was ephemeral ("ephemeral signing key"). A valid, signed, chain-intact receipt can honestly say the fuse did nothing (intact != fuse_ok).

### Ledger -- `results/<run>/<arm>/ledger.jsonl` (schema `lictor-ledger/v1`)

First line `{"schema":"lictor-ledger/v1","run_id":"...","arm_id":"...","genesis":"0000...0000"}`, then one entry per episode:

```json
{"arm_id":"t01-a05-d0","episode_index":7,"escalated":false,"fuse_ok":true,"hash":"aa31...","init_state_digest":"3f1c...","prev":"9c02...","receipt_digest":"be21...","run_id":"2026-09-01T09-14Z-pilot","seed":7,"seq":7,"stopped":true,"success":false,"tripped":true,"verdict_chain_head":"5b7e..."}
```
Known gap, partially closed: a backward chain cannot by itself detect tail truncation. The eval pool is DECLARED (run.json `seeds`, calibration.json `seed_pool`), so `lictor curve` reads the declared pool from `run.json` + the arm's calibration (a `--calib-seeds` flag is only an override and is recorded), asserts the ledger's index set == the declared pool, and otherwise refuses -- or, under `--partial`, binds `partial: true, n_declared, n_present, n_missing, missing_indices` into the signed curve receipt. Truncation of a declared pool is therefore detected; truncation by the key-holder who also rewrites run.json is not (trust model). External anchoring (Rekor / RFC 3161) is roadmap.

### HandoffRecord and AckToken (schema `lictor-handoff/v1`)

```json
{"schema":"lictor-handoff/v1","run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0-oracle","episode_index":7,
 "seq":0,"tick":97,"reason":"escalate_hold_timeout","reason_text":"Held 30 ticks without a clean re-arm window; a human must decide.",
 "reasons":["tier1_cp","rearm_budget"],"trips":33792,"fired":96,"window_hits":4,
 "top_z":[["stall",{"f64":"4010a3d70a3d70a4"}],["path_ineff",{"f64":"400a666666666666"}],["tce",{"f64":"3ff8000000000000"}]],
 "chain_at":"5b7e...","envelope_digest":"7c4a...","calibration_digest":"e1b8...",
 "digest":"9a41...","ack":null,"resolved_tick":null,"outcome":"unacked"}
```
```json
{"schema":"lictor-handoff/v1","handoff_digest":"9a41...","decision":"resume","operator":"ab12...64hex","nonce":7,"note":"oracle resume","sig":"c4...128hex"}
```
Signed bytes = canon({"schema","handoff_digest","decision","operator","nonce","note"}). `run_id`/`arm_id`/`episode_index` are digested, so an ack captured on one run/arm/episode does not verify on a paired re-run of the same seed; per-operator `last_nonce` is kept across episodes in the Session and persisted in `<out>/.lictor/verifier_nonce.json`. `note` and `reason_text` are stripped of control characters before any terminal rendering.

### Curve receipt -- `results/<run>/curve/<arm>.json` (schema `lictor-curve/v1`) + `results/<run>/curve/summary.csv`

Body per `CurveReceiptBody`; the CSV has one row per arm with columns: `pilot,arm_id,n,n_missing,partial,delay_steps,exec_mode,alpha,detector,success_rate,success_lo,success_hi,delta_vs_baseline,delta_vs_baseline_lo,delta_vs_baseline_hi,mcnemar_p,mcnemar_b,mcnemar_c,latency_control_arm,delta_vs_latency_control,delta_vs_latency_control_lo,delta_vs_latency_control_hi,mcnemar_p_lc,mcnemar_b_lc,mcnemar_c_lc,averted_rate,flagged_rate,flagged_lo,flagged_hi,false_trip_rate,false_trip_lo,false_trip_hi,intervention_rate,intervention_tick_frac,escalation_rate,lead_mean,lead_p50,lead_p10,aucpdt,roc_auc,bacc,violations_reached_env,tce_valid_frac,latency_p50_ns,latency_p99_ns,latency_max_ns,latency_label,ledger_chain_ok,pair_mismatches,cross_run_mismatches,receipt_pubkey`. `pilot` is `1` when `small_n` (n < 100) and `0` otherwise; `figures.py` hollows pilot points and never draws a line through them.

### Sweep -- `results/<run>/sweep.jsonl` (Layer A, one line per (alpha, detector))

```json
{"alpha_num":5,"alpha_den":100,"detector":"t01","gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],"k":3,"n":5,"tier0":true,"method":"binned","tau":3.7142857142857144,"tau_source":"artefact","n_calib":137,"n_fail":35,"n_succ":65,"tpr":0.6,"tpr_ci":[0.42,0.76],"fpr":0.06,"fpr_ci":[0.02,0.15],"lead_mean":41.2,"lead_p50":37.0,"lead_p10":4.0,"aucpdt":0.51,"roc_auc":0.71,"bacc":0.77,"fire_frac_mean":0.08,"holdout_fpr":0.05,"holdout_fpr_k1":0.07,"eps_prog":0.02}
```
(Plain JSON floats: sweep.jsonl is an analysis product, not a signed artefact. `tau` is `null` when `+inf` (degenerate alpha). `tau_source` is `"artefact"` when the point reused a `calibration.<alpha>.json` produced by `lictor calibrate` -- the Layer-A-vs-Layer-B validity comparison is made ONLY on such points -- and `"fit"` when `sweep` fitted a full-sample tau itself.)

### Harness index -- `results/<run>/<arm>/index.jsonl` (convenience only; rebuilt from the ledger at startup)

```json
{"run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","episode_index":7,"seed":7,"seed_pool":"eval","init_state_digest":"3f1c...","success":false,"steps":300,"ended_by":"truncated","max_coverage":0.71,"final_coverage":0.68,"receipt":"receipts/000007.json","receipt_digest":"be21...","first_stop_tick":181,"tripped":true,"stopped":true,"escalated":false,"policy_action_equal_ticks":293,"wall_s":18.4,"ts":"2026-09-01T09:41:02Z"}
```

### Trace -- `results/<run>/<arm>/traces/<episode:06>.ndjson`

`#meta {...}` then every request line verbatim (wire format, plain floats). Replay fixture and calibration source.

### Results tree

```
results/<run_id>/
  run.json                      # arms, seeds, pins, commands, lictor build (harness-written)
  sweep.jsonl                   # Layer A
  curve/<arm>.json  summary.csv # Layer B, from receipts
  calibration.<alpha>.json      # per alpha
  <arm_id>/
    ledger.jsonl  index.jsonl
    receipts/NNNNNN.json  ticks/NNNNNN.jsonl  timing/NNNNNN.jsonl  traces/NNNNNN.ndjson
  .lictor/verifier_nonce.json   # per-pubkey last ack nonce (under --out)
$LICTOR_KEYS/ (default $HOME/.lictor/):  key.hex  operator.hex  history.jsonl   # keys NEVER live in the repo or under /mnt/[a-z]/ (DrvFs ignores 0600)
```
Heavy results live on D: (`LICTOR_RESULTS=/mnt/d/lictor/results`, symlinked or passed via `--out`); the repo commits only `bench/fixtures/`, `docs/figures/`, the fitted `envelopes/pusht.toml` + `envelopes/pusht.oracle.toml`, `docs/envelope_fit_report.md` and the `research/*.md` memos (all generated-in-repo files are owned by WP-13).

## A.4 CLI surface

## CLI SURFACE (normative; clap derive; every command accepts `--json`)

```
lictor serve      --envelope <F.toml> [--calibration <F.json>] [--mode observe|enforce]
                  [--key <F.hex>] [--out <DIR>] [--trace <F.ndjson>] [--ticks tail32|all|none]
                  [--tier0 <csv of trip names>] [--no-tier1] [--on-fault hold|abort] [--latency-label <S>]
                    NDJSON server on stdin/stdout (lictor-wire/v1). Exit 0 on bye, 3 on internal error.
                    Refuses at startup (exit 2) a calibration whose embodiment_digest != the envelope's. Without --key an ephemeral
                    key is generated and every receipt carries fuse_ok=false + note "ephemeral signing key" (lictor curve refuses such arms).
                    --on-fault abort is a debugging aid only. --latency-label defaults to default_latency_label() (WSL2 sentence here).
lictor verify     <RECEIPT.json> [--ticks <F.jsonl>] [--timing <F.jsonl>] [--ledger <F.jsonl>] [--curve <C.json>]
                  [--calibration <F.json>] [--pubkey <hex>] [--json]
                    prints the aligned block below; exit 0 intact AND fuse held, 1 otherwise. Also recomputes envelope_digest from the
                    embedded body.envelope, checks counts.ticks == verdict_events == timing_events == outcome.steps, and with --ticks
                    cross-checks the ticks header (run_id, arm_id, episode_index, genesis) and the embedded tail's first `prev`
                    against the file, and compares the recomputed head with the signed head (HEAD MISMATCH). With --calibration it
                    checks budget.alpha/gate/kn against the file. With --pubkey a mismatch is `pubkey        MISMATCH` and intact NO.
                    Prints `WARNING: signed with the committed test key` when the pubkey is one of TEST_PUBKEYS.
lictor replay     <TRACE.ndjson> --envelope <F> [--calibration <F>] [--mode observe|enforce] [--repeat N] [--expect <hex64>]
                    "replays N/N byte-identical  verdict_chain=<hex>" ; "timing chain head varies (by design -- wall-clock is not replayed)"
lictor calibrate  --run <DIR> --arm <ARM> --envelope <F> --alpha <num/den>[,...] [--method static|binned]
                  [--gate <csv terms>] [--kn 3,5] [--holdout 3/10] [--split 2|3] [--n-calib N] [--warn-margin 0.5] -o <DIR>
                    writes <DIR>/calibration.<alpha>.json per alpha (float-free, self-digested). Refuses eval-pool seeds, and a
                    horizon_ticks that differs from envelope.embodiment.horizon_ticks or from run.env.max_episode_steps.
lictor sweep      --run <DIR> --calib-arm <ARM> --eval-arm <ARM> --alphas <csv num/den> [--calibration-dir <DIR>]
                  --detectors t0,t1_tce,t1_stall,t1_full,t01,t01_and [--method binned] [--eps-prog 0.02] -o <F.jsonl>
                    with --calibration-dir, points whose (alpha, gate, kn, method) match a calibration.<alpha>.json reuse its tau verbatim.
lictor curve      --run <DIR> --baseline <ARM> [--arms a,b,c] [--latency-control <ARM>] [--eps-prog 0.02] [--calib-seeds 900000-900299]
                  [--allow-small] [--partial] [--compare-run <DIR>] [--key <F.hex>] -o <DIR>
                    recomputes every metric FROM RECEIPTS; refuses on a broken ledger, on mixed/ephemeral receipt keys, on a
                    tier1/alpha disagreement with run.json, on an index set != the baseline's or != the declared pool (unless
                    --partial), and on n < 100 (unless --allow-small; then small_n=true and the CSV row starts with pilot=1).
                    --latency-control defaults to obs-d<d> derived from the arm's delay_steps when that arm exists in the run.
                    Missing indices count as success=false, stopped=true. Writes <arm>.json (signed) + summary.csv.
lictor envelope   init  --profile pusht -o <F.toml> [--operator <hex>]...
                  check <F.toml>                       -> validate + digest + embodiment digest
                  digest <F.toml>                      -> "<envelope digest>" (and "embodiment=<hex>" on stderr)
                  show <F.toml>
                  fit   --run <DIR> --arm <ARM> --base <F.toml> [--quantile 0.999] [--slack 1.25] [--operator <hex>]... -o <F.toml> [--report <F.md>]
                    --operator writes the oracle envelope (identical embodiment, hence identical embodiment_digest and calibration).
lictor bench      [--n 200000] [--trace <F.ndjson>] [--tier t0|t0t1] [--csv <F>] [--json]
                    "verdict  p50 X us  p99 X us  p99.9 X us  p99.99 X us  max X us  (n=N)" + per-tier lines + "allocations   0"
                    the allocation count comes from the always-installed gated counting allocator (alloc_count.rs); the "arena"
                    figure is size_of::<FuseRt>() + size_of::<FuseConfig>() printed at run time, never a constant.
lictor selftest   [--json]    -> one line per check, final "SELFTEST PASS" | "SELFTEST FAIL"
lictor ledger     verify <F.jsonl> | append --ledger <F.jsonl> --receipt <R.json>
lictor crash-receipt --envelope <F.toml> [--calibration <F.json>] --mode observe|enforce --run-id <S> --arm-id <S> --episode-index <N>
                  --seed <N> --seed-pool <S> --init-state-digest <hex> --budget <json> [--key <F.hex>] --out <DIR> [--note <S>]
                    host-side accounting for a serve child that died before episode_end: writes a signed receipt with
                    ended_by="fuse_crash", success=false, terminal_state=fault, fuse_ok=false and appends the ledger. Exit 0.
lictor key        init [-o <F.hex>] [--role signer|operator] [--force] [--i-know] | pub [--key <F.hex>]
                    default path $LICTOR_KEYS/key.hex (or $HOME/.lictor/key.hex); refuses to overwrite without --force; refuses a
                    path under /mnt/[a-z]/ without --i-know; prints "pubkey <hex>", "stored <path> (mode 0600)" and one custody line.
lictor ack        --handoff <digest|record.json> --decision resume|abort|retune --key <operator.hex> [--note S] [--nonce N] [-o <F.json>]
lictor history    [--dir <DIR>] [--n 20]
lictor version    -> "lictor 0.1.0 (<git>) sha256=<binary sha>"
```

Exit codes: `0` ok; `1` verification failed / fuse not held / determinism mismatch; `2` usage; `3` internal. NOTE for scripts: `lictor verify` exits 1 on an honest Observe receipt (intact but not enforced), so CI and bench scripts MUST use `--json` and test the `intact` field rather than the exit code when they mean "is the record intact".

Reference output (frozen strings -- CI greps them; do not reword):

```
$ lictor verify results/r/t01-a05-d0/receipts/000007.json --ticks results/r/t01-a05-d0/ticks/000007.jsonl --pubkey 64e8...
  schema         ok    lictor-receipt/v1
  signature      ok    (key 64e8281d05cf...)
  pubkey         ok    (matches --pubkey)
  body digest    ok    be21...
  envelope       ok    digest recomputed 7c4a...  embodiment 91d0...
  counts         ok    ticks=300 verdict_events=300 timing_events=300 steps=300
  verdict chain  ok    head=5b7e...  (300 ticks)
  timing chain   ok    head=22a0...  (not replayable -- by design)
  intact         YES
  FUSE HELD      (7 substitutions, 0 violations reached the environment)
  bindings       envelope=7c4a... calibration=e1b8...(alpha=5/100,n_calib=137) policy=lerobot/diffusion_pusht@a3f1...
                 seeds pool=eval seed=7 init_state=3f1c...  client=lictor_client/0.1.0

$ lictor verify r.json --ticks rechained.jsonl          # attacker recomputed hashes 17..299: the chain links, the head does not
  verdict chain  HEAD MISMATCH recomputed=9d02... signed=5b7e...
  intact         NO

$ lictor verify r.json --pubkey 0000...
  pubkey         MISMATCH  (receipt 64e8281d05cf..., expected 0000...)
  intact         NO

$ lictor verify crates/lictor-receipt/tests/fixtures/receipt/receipt_000007.json
  WARNING: signed with the committed test key

$ lictor verify observe.json
  ...
  intact         YES
  FUSE NOT ENFORCED
  notes:  - the fuse observed but did not enforce -- this receipt does not attest protection

$ lictor verify tampered.json
  schema         ok    lictor-receipt/v1
  signature      FAIL  (key 64e8281d05cf...)
  body digest    FAIL  be21...
  intact         NO
  notes:  - Ed25519 signature does not verify against the body

$ lictor verify r.json --ticks edited.jsonl
  verdict chain  BROKEN at seq=17

$ lictor ledger verify results/r/t01-a05-d0/ledger.jsonl
  episodes 500   chain ok    success 341/500 (68.2%)   stops 61   escalated 22   fuse_ok 500/500
$ ... after deleting 43 entries ...
  ledger BROKEN at seq=3 -- entries are missing or edited; this curve point cannot be trusted

$ lictor replay --repeat 40 bench/fixtures/traces/pusht_000007.ndjson --envelope envelopes/pusht.toml --calibration bench/fixtures/calibration.a05.json --mode enforce
  ticks=300  repeats=40
  replays 40/40 byte-identical  verdict_chain=5b7e...
  timing chain head varies (by design -- wall-clock is not replayed)
  states nominal=281 watching=12 clamped=0 braking=4 held=3 escalated=0 fault=0
  RESULT  DETERMINISTIC

$ lictor bench --n 200000                              # X = measured value; CI greps the SHAPE of these lines, never the numbers
  tier=t0t1  horizon=15  exec=8  dim=2  n=200000
  verdict  p50 X us  p99 X us  p99.9 X us  p99.99 X us  max X us  (n=200000)
  tier0    p50 X us  p99 X us
  brake    p50 X us  p99 X us
  tier1    p50 X us  p99 X us
  allocations   0  (state preallocated X KiB = size_of FuseRt + FuseConfig)
  environment   WSL2 (Hyper-V utility VM, non-RT host) -- NOT a real-time measurement

$ lictor curve --run results/r --baseline obs-d0 --arms t01-a05-d0 -o results/r/curve
  t01-a05-d0  n=500  success 0.xxx [0.xxx,0.xxx]  delta_vs_obs-d0 +0.xxx [-0.xxx,+0.xxx] p=0.xx  delta_vs_obs-d0(lc) ...  tce_valid 1.000  pubkey 64e8...
$ lictor curve --run results/r --baseline obs-d0 --arms t01-a05-d3 -o results/r/curve
  t01-a05-d3  ... REFUSED: episode index set differs from baseline (missing 3 indices: 17,211,340); re-run them or pass --partial
```

---

# PART B -- WP-0 "SKELETON" (runs alone first; ~60 minutes; blocking)

**Owner: one agent. Permanently owns every file marked PERM; creates every file marked STUB and never touches it again (ownership transfers to the package named).**

## B.1 Files

PERM (WP-0 keeps these for the life of the project):
```
Cargo.toml                       Cargo.lock                       rust-toolchain.toml              rustfmt.toml
.cargo/config.toml               Makefile                         .gitignore                       LICENSE
pyproject.toml                   .github/workflows/ci.yml
crates/lictor-core/Cargo.toml    crates/lictor-core/src/lib.rs
crates/lictor-canon/Cargo.toml   crates/lictor-canon/src/lib.rs
crates/lictor-detect/Cargo.toml  crates/lictor-detect/src/lib.rs
crates/lictor-fuse/Cargo.toml    crates/lictor-fuse/src/lib.rs
crates/lictor-receipt/Cargo.toml crates/lictor-receipt/src/lib.rs
crates/lictor-runtime/Cargo.toml crates/lictor-runtime/src/lib.rs
crates/lictor-calib/Cargo.toml   crates/lictor-calib/src/lib.rs
crates/lictor-cli/Cargo.toml     crates/lictor-cli/src/main.rs    crates/lictor-cli/src/cmd/mod.rs
crates/lictor-cli/build.rs       (embeds git sha via `git rev-parse --short HEAD`, "nogit" fallback, into env!("LICTOR_GIT"))
harness/__init__.py  harness/tests/__init__.py  adapters/__init__.py  adapters/tests/__init__.py  (empty)
docs/ARCHITECTURE.md  docs/IMPLEMENTATION_PLAN.md  docs/ANALYSIS.md   (copied verbatim from the scratchpad; frozen)
```
STUB (created by WP-0 with the frozen signatures and `todo!()` bodies, then owned by the named package):
```
crates/lictor-core/src/{fmath,chunk,envelope,scores,verdict,state,reason,ack}.rs        -> WP-1
crates/lictor-detect/src/brake.rs                                                       -> WP-1
envelopes/pusht.base.toml   (written BYTE-FOR-BYTE from FILE FORMATS -- it is frozen text; cmd/envelope.rs include_str!s it)   -> WP-1
crates/lictor-detect/tests/fixtures/brake/cases.json   (placeholder `[]`; cmd/selftest.rs include_str!s it)                   -> WP-1
crates/lictor-detect/tests/fixtures/tier1/cases.json   (placeholder `[]`; cmd/selftest.rs include_str!s it)                   -> WP-2
crates/lictor-detect/src/{tier0,window,tier1,conformal}.rs                              -> WP-2
crates/lictor-fuse/src/{fuse,fsm,tally}.rs                                              -> WP-3
crates/lictor-canon/src/{jcs,f64enc,digest}.rs                                          -> WP-4
crates/lictor-receipt/src/{tick,body,sign,ledger,curve,handoff,history,keys}.rs         -> WP-4
crates/lictor-calib/src/{file,traces,calibrate,envfit,metrics,sweep,curve}.rs           -> WP-5
crates/lictor-cli/src/cmd/{calibrate,sweep,curve,envelope}.rs                           -> WP-5
crates/lictor-runtime/src/{wire,codec,session,episode,latency,trace}.rs                 -> WP-6
crates/lictor-cli/src/cmd/serve.rs   crates/lictor-cli/src/cmd/crash_receipt.rs         -> WP-6
crates/lictor-cli/src/cmd/{verify,replay,selftest,ledger,key,ack,history,version}.rs    -> WP-7
crates/lictor-cli/src/render.rs                                                         -> WP-7
crates/lictor-cli/src/cmd/bench.rs   crates/lictor-cli/src/alloc_count.rs               -> WP-10
    (alloc_count.rs stub: `#![allow(unsafe_code)]`-free file body is WP-10's; WP-0 writes the module with `pub static COUNTING: AtomicBool`,
     `pub fn allocations() -> u64`, `pub fn reset()` as todo!()-free counters and a `#[global_allocator] static A: Counting = Counting;`
     wrapper over `std::alloc::System` gated on COUNTING -- ~25 lines, the only unsafe in the binary, so the workspace builds and the
     bench binary's allocator is in place from minute one)
bench/run_all.sh  scripts/ci_python.sh   (stubs: `#!/usr/bin/env bash\nset -euo pipefail\necho "stub: not yet implemented"; exit 0`)  -> WP-10
```

## B.2 Content spec

1. **Workspace `Cargo.toml`** exactly as ARCHITECTURE sec 3 (members, `[workspace.package]`, `[workspace.dependencies]`, release profile with `panic = "abort"`, the `bench-debug` profile).
2. **Per-crate `Cargo.toml`**: `[package] name/version.workspace/edition.workspace/license.workspace/description`; deps via `{ workspace = true }`. Features: `lictor-core` `default = ["std"]`, `std = ["serde/std", "dep:toml", "dep:serde_json", "dep:lictor-canon"]`, and `libm = { workspace = true }` UNCONDITIONAL (no `libm` feature exists); `lictor-detect`: `lictor-core = { workspace = true, default-features = false }`, `default = ["std"]`, `std = ["lictor-core/std"]` (the `[workspace.dependencies]` entries for core/detect/fuse carry `default-features = false` -- cargo ignores a `default-features = false` stated only on a `workspace = true` dependency -- and receipt/runtime/calib/cli request `features = ["std"]` explicitly); `lictor-fuse`: core and detect both `default-features = false`, `std = ["lictor-core/std", "lictor-detect/std"]` -- so `--no-default-features` REALLY builds them without std. `lictor-canon`: serde, serde_json, sha2, hex, base64, thiserror. `lictor-receipt`: core(std), canon, fuse, serde, serde_json, sha2, hex, ed25519-dalek, getrandom, thiserror. `lictor-runtime`: core, detect, fuse, receipt, canon, serde, serde_json, anyhow, thiserror, hdrhistogram (NOT calib). `lictor-calib`: core, detect, fuse, receipt, canon, runtime, serde, serde_json, toml, anyhow, thiserror. `lictor-cli`: everything + clap, anyhow, hdrhistogram; `[[bin]] name = "lictor"`. PRE-DECLARED `[dev-dependencies]` (only WP-0 edits any Cargo.toml, so they are all written now): `lictor-core`: serde_json; `lictor-detect`: serde_json; `lictor-fuse`: serde_json, lictor-runtime (dev-dep cycle, legal), lictor-canon; `lictor-canon`: none extra; `lictor-receipt`: tempfile; `lictor-runtime`: tempfile, lictor-calib; `lictor-calib`: tempfile; `lictor-cli`: tempfile, serde_json.
3. **`rust-toolchain.toml`**: `[toolchain] channel = "stable" components = ["rustfmt", "clippy"]`. **`rustfmt.toml`**: `max_width = 110`, `use_small_heuristics = "Max"`. **`.cargo/config.toml`**: `[build] rustflags = ["-D", "warnings"]` ONLY -- no `target-dir` (CARGO_TARGET_DIR env is mandatory and documented in the Makefile header).
4. **`Makefile`** targets: `check` (fmt --check, clippy, test), `build` (release build of lictor-cli), `bench`, `selftest`, `ci` (= check + build + `bash bench/run_all.sh` + `bash scripts/ci_python.sh`), `demo` (`bash scripts/demo.sh`), `nostd-check` (`cargo check -p lictor-core -p lictor-detect -p lictor-fuse --no-default-features`). Every target guards `CARGO_TARGET_DIR` is set (`$(if $(CARGO_TARGET_DIR),,$(error set CARGO_TARGET_DIR)))`).
5. **`.github/workflows/ci.yml`** (bulla shape): ubuntu-latest, `dtolnay/rust-toolchain@stable` with rustfmt+clippy, `env: RUSTFLAGS: -D warnings`, steps: fmt --check; clippy --workspace --all-targets -- -D warnings; `cargo test --workspace`; `make nostd-check`; `cargo build --release -p lictor-cli`; `bash bench/run_all.sh "$CARGO_TARGET_DIR/release/lictor"`; `bash scripts/ci_python.sh`; `python3 adapters/verify_receipt.py crates/lictor-receipt/tests/fixtures/receipt/receipt_000007.json | grep -q '^OK'`. Set `CARGO_TARGET_DIR: ${{ github.workspace }}/target` in the workflow env. No GPU, no torch in CI.
6. **`.gitignore`**: `/target`, `/results`, `*.pyc`, `__pycache__/`, `.lictor/`, `/venv`, `*.svg.tmp`, `*.hex` (keys; the two committed TEST keys are force-added with a `# test key` comment in the file). **`LICENSE`**: MIT, RARS-oss. **`pyproject.toml`**: `[tool.pytest.ini_options] pythonpath = ["."] testpaths = ["harness/tests", "adapters/tests"]` and nothing else.
7. **Every `lib.rs`**: `// SPDX-License-Identifier: MIT`, the crate doc comment, `#![forbid(unsafe_code)]`, no_std attribute where frozen, `pub mod` declarations and `pub use` re-exports EXACTLY as the freeze -- no logic.
8. **Every STUB module**: the frozen `pub` items with `todo!("WP-n")` bodies (constants with placeholder values where a body is required, e.g. `CalibrationC::DISARMED` may be a `todo!()`-free const with zeros/INFINITY so downstream compiles), the frozen doc comments, and a header `//! OWNER: WP-n. Stub written by WP-0; replace the bodies, keep the signatures.` Structs with private fields get their private fields as in the freeze so `new()` can be written later without a signature change. Derive lists exactly as frozen -- including `Default` on `FuseState` (`#[default] Idle`), the hand-written `impl Default for Tally`, the `impl Default` beside every `new()` (`Trail`, `Tier1Rt`, `FuseRt`, `Hist`, `Staging`) and `is_empty` beside every `len()`, so `clippy -D warnings` is green on the stubs. `wire.rs` is stubbed with EVERY payload struct of A.1 (fields and `deny_unknown_fields`), since `Staging`, `bench.rs` and the fuse tests construct `TickReq`. `envelopes/pusht.base.toml` and the two placeholder `cases.json` files exist from minute one so every `include_str!` resolves.
9. **`crates/lictor-cli/src/main.rs`**: `#![deny(unsafe_code)]` (NOT forbid) then `#[allow(unsafe_code)] mod alloc_count;` then `mod cmd; mod render;`; clap `#[derive(Parser)] struct Cli { #[command(subcommand)] cmd: cmd::Cmd, #[arg(long, global = true)] json: bool }`; `fn main() { std::process::exit(cmd::dispatch(Cli::parse())) }`. **`cmd/mod.rs`**: `pub mod serve; pub mod verify; ...` for every command, `#[derive(Subcommand)] pub enum Cmd { Serve(serve::Args), Verify(verify::Args), Replay(replay::Args), Calibrate(calibrate::Args), Sweep(sweep::Args), Curve(curve::Args), Envelope(envelope::Args), Bench(bench::Args), Selftest(selftest::Args), Ledger(ledger::Args), CrashReceipt(crash_receipt::Args), Key(key::Args), Ack(ack::Args), History(history::Args), Version(version::Args) }` and `pub fn dispatch(cli: Cli) -> i32` matching each to `<mod>::run(args, json)`, mapping `Err` to exit 3 with the message on stderr. Each STUB cmd file contains the FULL clap `Args` struct for its flags per the CLI SURFACE (flags are interface) and `pub fn run(a: Args, json: bool) -> anyhow::Result<i32> { todo!("WP-n") }`.
10. **`build.rs`** for lictor-cli: emits `cargo:rustc-env=LICTOR_GIT=<short sha|nogit>`, `cargo:rustc-env=LICTOR_BUILD_UTC=...` and `cargo:rerun-if-changed=.git/HEAD` (so `env!("LICTOR_GIT")` never fails on a fresh checkout without git); the binary's own sha256 is computed at runtime from `std::env::current_exe()` (by WP-7's `version` and WP-6's session), not at build time.
11. Copy `docs/ARCHITECTURE.md` and `docs/IMPLEMENTATION_PLAN.md` from the scratchpad verbatim.

## B.3 Acceptance (WP-0 is done when ALL pass)

```bash
cargo build --workspace                                   # green with todo!() bodies
cargo clippy --workspace --all-targets -- -D warnings     # green (allow(unused) is NOT permitted; unused stub params are named `_x`)
cargo fmt --all --check
cargo test --workspace --no-run                           # compiles test targets (none exist yet except doc)
make nostd-check                                          # core/detect/fuse compile without std (--no-default-features; no libm feature)
grep -rn "allow(clippy" crates | wc -l                    # 0 -- the freeze carries its own Default/is_empty, no lint suppression
$CARGO_TARGET_DIR/debug/lictor --help | grep -q serve     # every subcommand listed
grep -rn "todo!(\"WP-" crates | wc -l                     # > 60 stubs, each naming its owner
```
Report: the stub count per package and the `cargo build` wall time.

---

# PART C -- WORK PACKAGES WP-1 .. WP-13

Each package: id, title, EXACT owned files (globally disjoint; see PART D), dependencies (compile: WP-0 only, always; test: as listed), content spec, tests, acceptance commands. "Fixture" = a JSON/TOML file under a `tests/fixtures/` directory you own, with hand-computed expected values and a comment (or `_note` key) explaining the computation.

---

## WP-1 -- Core types, envelope, TOML, compile(), brake feasibility

Files:
```
crates/lictor-core/src/fmath.rs   crates/lictor-core/src/chunk.rs    crates/lictor-core/src/envelope.rs
crates/lictor-core/src/scores.rs  crates/lictor-core/src/verdict.rs  crates/lictor-core/src/state.rs
crates/lictor-core/src/reason.rs  crates/lictor-core/src/ack.rs
crates/lictor-core/tests/envelope_fixture.rs   crates/lictor-core/tests/chunk.rs   crates/lictor-core/tests/reason_glossary.rs
crates/lictor-core/tests/fixtures/envelope/pusht_base.toml   crates/lictor-core/tests/fixtures/envelope/invalid_*.toml (>= 4)
crates/lictor-core/tests/fixtures/envelope/digest.json
crates/lictor-detect/src/brake.rs
crates/lictor-detect/tests/brake.rs
crates/lictor-detect/tests/fixtures/brake/cases.json   crates/lictor-detect/tests/fixtures/brake/gen.py
envelopes/pusht.base.toml
```
Dependencies: WP-0 (compile). Test-level: `digest_hex` needs WP-4's `lictor_canon::canon`; until WP-4 lands, mark that single test `#[ignore = "needs WP-4"]` and un-ignore in WP-13.

Spec:
- `fmath.rs`, `chunk.rs`, `scores.rs`, `verdict.rs`, `state.rs`, `ack.rs`: implement every frozen item. `ChunkBuf::fill` validates `h <= MAX_H`, `d <= MAX_D`, `exec <= h`, `src.len() == h*d`, `h > 0`. `ChunkView::action(i)` clamps `i` to `horizon-1` in release (debug_assert in debug). `TripMask::names/from_name`, `Feat::from_name/name`, `bin_of` exactly as frozen. `CalibrationC::DISARMED`: `mask 0`, `gate DISARMED`, `tau = f64::INFINITY`, `t_grid = 1`, zeros elsewhere.
- `reason.rs`: `reason_text` -- one plain-English sentence <= 120 chars per code, no jargon (e.g. `BrakeInfeasible` -> "The committed motion could not be stopped inside the workspace; the fuse is braking."). This IS the human-escalation glossary.
- `envelope.rs`: `from_toml`/`to_toml` (toml 0.8; TOML tables `[embodiment] [brake] [hysteresis] [contact] [fit]` map to the struct fields; unknown keys are an error), `validate` (dims consistent, `box_lo < box_hi` after margin, positivity of limits/scales/hz, `1 <= k <= n <= 63`, `horizon_ticks > 0`, `operators.len() <= MAX_OPERATORS` and each 64 hex, `gate` parses, `tier0_enabled` names valid, `fail_closed == true`, `exec_steps <= horizon`), `tier0_mask`, `GateSpec::parse/mask/names`, `dt`, `compile` (margin-adjust the box, precompute `inv_dt*`, `step_max = v_max*dt`, `norm_scale_iso = min_c norm_scale[c]`, `window_mask`, `n_operators`, `embodiment_digest` bytes, embed `CalibrationC` or `DISARMED`; gate from calibration when present else from the envelope), `digest_hex` = `lictor_canon::sha256_hex(canon(floatify(serde_json::to_value(self))))` and `embodiment_digest` = the same over `to_value(&self.embodiment)`, both behind `std`. `state.rs`: `FuseState` derives `Default` with `#[default] Idle` exactly as frozen.
- `envelopes/pusht.base.toml`: WP-0 already wrote it byte-for-byte from FILE FORMATS (with `provides_vel = true`, `horizon_ticks = 300`, `operators = []`); WP-1 owns it from then on and must keep `crates/lictor-core/tests/fixtures/envelope/pusht_base.toml` identical to it.
- `brake.rs`: `brake_feasible` per ARCHITECTURE 5.2: `PdSecondOrder` = exact rollout with EXACTLY `(commit_steps - from + brake_steps) * substeps` iterations tracking the running `margin` (min over substeps and dims of distance inside the margin-adjusted box), `stop_dist = ||p_end - p_t||`; `v0` is whatever the caller passes (the simulator velocity on PushT; decide() supplies the finite difference only when the manifest says `provides_vel = false`); `from` = the row executed this tick (0 at a sync/freeze delivery, `d` at an async-drop delivery, `idx` intra-chunk); closed-form kinds = `commit_steps - from` Euler steps + stopping ball with `react_ticks`; `ZeroVelocityHold` uses the BoundedAccel formula. `brake_action` (position kinds: `clamp(p, lo, hi)`; velocity kinds: ramp) and `hold_action` (copy latched, clamp defensively). Only `+ - * / sqrt min max` via `fmath`; fixed accumulation order; zero allocation; `from >= commit_steps` -> only the brake tail is simulated.
- `tests/fixtures/brake/gen.py`: a 30-line Python reimplementation of the gym-pusht PD loop (k_p=100, k_v=20, dt=0.01, 10 substeps) that emits `cases.json` (>= 8 cases: straight chunk inside the box; chunk driving into the wall at 3 speeds; `from` = 0, 3, 7; a `v0` != 0 case; one closed-form case per kind). Rust asserts `margin` and `stop_dist` to 1e-12 (`abs(a-b) <= 1e-12 * max(1, abs(b))`).

Tests: `envelope_fixture.rs` (round-trip TOML -> struct -> TOML -> struct equality; every `invalid_*.toml` rejected with the expected `EnvelopeError` variant, including `n = 64` and `horizon_ticks = 0`; `compile` constants: `step_max == 100.0`, `box_lo == [17,17]`, `overlap == 7`, `window_mask == 31`; digest stable across two loads and equal to `digest.json`, and `embodiment_digest` UNCHANGED when `v_max`/`operators` are edited but changed when `norm_scale` is -- the ignored-until-WP-4 cases). `chunk.rs` (fill/view/row_mut, every `ChunkError`, `action(i)` clamping, `all_finite`). `reason_glossary.rs` (every variant has text, <= 120 chars, ends with a period, contains no banned vocabulary). `brake.rs` (fixture parity; iteration-count property via a counting closure is not possible without hooks -- instead assert `brake_feasible` on a 64-row chunk with `from=0` completes and margin is monotone non-increasing in `v0` magnitude).

Acceptance:
```bash
cargo test -p lictor-core -p lictor-detect --test brake --test envelope_fixture --test chunk --test reason_glossary
cargo clippy -p lictor-core -p lictor-detect --all-targets -- -D warnings && cargo fmt --all --check
cargo check -p lictor-core -p lictor-detect --no-default-features
python3 crates/lictor-detect/tests/fixtures/brake/gen.py --check   # regenerates to a temp file and diffs against cases.json: identical
```

---

## WP-2 -- Detectors: Tier-0 limits + projection, windowing, Tier-1 features, conformal runtime

Files:
```
crates/lictor-detect/src/tier0.rs   crates/lictor-detect/src/window.rs   crates/lictor-detect/src/tier1.rs   crates/lictor-detect/src/conformal.rs
crates/lictor-detect/tests/tier0.rs  crates/lictor-detect/tests/window.rs  crates/lictor-detect/tests/tier1.rs  crates/lictor-detect/tests/conformal.rs
crates/lictor-detect/tests/fixtures/tier0/cases.json   crates/lictor-detect/tests/fixtures/tier1/cases.json   crates/lictor-detect/tests/fixtures/conformal/cases.json
crates/lictor-detect/tests/fixtures/tier1/gen.py
```
Dependencies: WP-0 (compile). Test-level: WP-1 (`FuseConfig` via `compile`; build configs in tests through `SafetyEnvelope::from_toml(include_str!("../../../../envelopes/pusht.base.toml"))`).

Spec:
- `tier0.rs`: `check_chunk` implements the ARCHITECTURE 5.1 table over all rows (workspace, speed, accel, jerk, reach, contact) with strict `>` comparisons, respecting `cfg.tier0_enabled` (disabled checks neither trip nor clamp), and the sequential direction-preserving leash projection into `out` when `cfg.clamp == Project` (copy-through when `Off`). `clamped_dims` = dims where `out != ch` anywhere. `peak_speed` = max commanded speed. `check_action`: box + speed (against `prev`) + reach (against `obs.pos`) + contact on a single action, leash into `out`. Fixed iteration count = horizon; no allocation.
- `window.rs`: `Window::push(hit, mask)` = `((bits << 1) | hit) & mask`; `hits` = `count_ones`; `Trail` fixed ring of positions with `net(d, w)` = distance between newest and the (w-1)-th older sample, `path(d, w)` = sum of consecutive distances over the last w samples (fixed order oldest->newest); both return 0.0 when `len < w`.
- `tier1.rs`: `features` per ARCHITECTURE 5.3 with the exact formulas on `abar`; `speed_peak = max_i ||a_i - q_{i-1}|| / norm_scale_iso` and `stall` with `v_ref = STALL_VREF_FRAC * norm_scale_iso / dt` -- NO feature may read `cfg.v_max`/`a_max`/`j_max`/`reach_max` (a test greps `tier1.rs` for those identifiers and fails on a hit); chunk-boundary features recomputed only when `chunk.is_some()` (then stored in `rt.held`/`held_valid` and copied on other ticks); `tce`/`acc` need `rt.have_prev` and `L >= 2` (L = overlap between `rt.prev` and the new chunk computed from `t_emit` difference: `s = new.t_emit - prev.t_emit`, `L = min(prev.horizon - s, new.horizon)`; if `s >= prev.horizon` the overlap is absent); `path_ineff`/`stall` need `trail.len() >= PE_WINDOW`; `ext_j` valid iff `obs.ext.len() > j`. The trail is pushed by `decide()` (WP-3) before calling `features` -- document this contract in the doc comment. `features` is the SOLE writer of `rt.prev`/`rt.have_prev`: AFTER computing tce/acc it copies the new RAW chunk in (`decide()` never touches `t1.prev`), so TCE is raw-vs-raw in Observe and Enforce alike. `Trail::is_empty`, `impl Default for Trail`, `impl Default for Tier1Rt` exactly as frozen.
- `conformal.rs`: `standardise` (bin via `cal.bin(t)`, only masked & valid), `aggregate` (fixed-order DNF: for each term in `0..n_terms`, if all channels of the term are `mask & valid`, term value = min z over the term in ascending channel order; s = max over term values; `NEG_INFINITY` if no term qualifies; also `fired` bits = `z_j > tau` for masked & valid j), `trip` = `s > tau` strict.

Tests: `tier0.rs` fixtures (>= 10 cases: inside box -> no trips; one row outside -> WORKSPACE + clamped_dims; speed 1.5x -> SPEED + leash preserves direction (unit vectors equal to 1e-12) and idempotence (projecting the projection changes nothing); accel and jerk cases; reach case; disabled check produces no trip; `Off` mode passes through; contact armed vs disarmed). `window.rs` (K-of-N sequence table; mask wrap; Trail net/path hand values). `tier1.rs` fixtures generated by `gen.py` (numpy reimplementation of the 8 formulas -- with the manifest-only `speed_peak`/`stall` definitions -- on 3 synthetic chunk pairs incl. a frozen-policy chunk where `acm_neg` dominates and a dithering trail where `path_ineff` dominates; the `L < 2` absent case for a sync `d = 7` pair (`t_emit` difference 15); hold-between-boundaries behaviour; and a case proving that `features` on a PROJECTED previous chunk would give a different `tce` than on the raw one, i.e. the raw-vs-raw contract matters) asserted to 1e-12. `conformal.rs` (empty mask -> NEG_INFINITY and no trip; all-invalid; singleton gate == plain max; AND-pair gate == min; boundary `s == tau` must NOT trip; `bin_of` at t = 0, 299, 300, 10^6).

Acceptance:
```bash
cargo test -p lictor-detect
cargo clippy -p lictor-detect --all-targets -- -D warnings && cargo fmt --all --check
cargo check -p lictor-detect --no-default-features
python3 crates/lictor-detect/tests/fixtures/tier1/gen.py --check
```

---

## WP-3 -- The fuse: decide(), state machine, FuseRt, Tally

Files:
```
crates/lictor-fuse/src/fuse.rs   crates/lictor-fuse/src/fsm.rs   crates/lictor-fuse/src/tally.rs
crates/lictor-fuse/tests/state_machine.rs   crates/lictor-fuse/tests/observe_passthrough.rs   crates/lictor-fuse/tests/decide_order.rs
crates/lictor-fuse/tests/fixtures/fsm/transitions.json   crates/lictor-fuse/tests/fixtures/episode/synthetic_300.json
crates/lictor-fuse/tests/common/mod.rs
```
Dependencies: WP-0 (compile). Test-level: WP-1, WP-2. Do NOT write `tests/alloc.rs` or `tests/determinism.rs` (WP-10).

Spec:
- `fuse.rs`: `FuseRt::new/reset` + `impl Default` (reset clears everything, sets `state = Armed`, `seq = 0`, `last_t = 0`, `next_chunk_seq = 0`, `have_prev_pos = false`, `hold = 0`, `tally = Tally::default()` whose `max_s`/`max_z` are `NEG_INFINITY`). `decide` implements the nine ordered stages of ARCHITECTURE 5.4 verbatim, calling WP-2/WP-1 functions through their frozen signatures. GUARD before anything else, INCLUDING the two continuity checks (`t == 0` on the first tick else `t == last_t + 1 + missed_ticks`; `chunk.seq == next_chunk_seq`, `chunk.t_emit <= t`, `idx == t - chunk.t_emit`) -> SCHEMA. Stage 1 NEVER writes `t1.prev` (WP-2's `features` is the sole writer); `v_hat = obs.vel` when `Some`, else the finite difference. Stage 2 calls `brake_feasible(..., from = idx)` at a boundary. In `Fault`, the hold is `clamp(last finite pos)`; if `pos` itself is non-finite use `rt.pos` from the previous tick (or `hold` if never set; on the very first tick that is `clamp(zeros)` = the box corner -- harmless on PushT and documented). `stopped` = `true` in Observe mode. The verdict's `action` is copied into `rt.last_cmd`; `prev_state` is the state before the tick. `Fuse::finish` returns the tally with `terminal_state` set.
- `fsm.rs`: `next` implements the 24-row table of ARCHITECTURE sec 6 (rows 1-23 plus 17b, the explicit `clamp_mode == Off` soft-trip row) top-down, first match wins, with every counter side effect listed; reasons per the table; `extra_trips` = the bits the FSM itself raises (TIER1_CP, CLAMP_BUDGET, HANDOFF_TIMEOUT, OPERATOR_ABORT, BRAKE_TIMEOUT, REARM_BUDGET). Ack handling: `ack.handoff_seq` must equal `rt.handoff_seq` and `ack.nonce > rt.last_nonce[slot]`, else the ack is ignored (the runtime already verified signature/operator; the fsm only re-checks the two integers). `Retune` -> Terminated with reason `TerminatedRetune`.
- `tally.rs`: `Tally` with the HAND-WRITTEN `impl Default` (`max_s = NEG_INFINITY`, `max_z = [NEG_INFINITY; NFEAT]`, `terminal_state = Idle`, zeros/None elsewhere) + `fn record(&mut self, v: &SafetyVerdict, entered_hold: bool, ...)` helpers used by `decide` (keep them `pub(crate)`).
- Observe mode: everything computed identically; `action` = raw policy row `cur.action(idx)` (from the RAW chunk -- keep a raw copy: in Observe mode `rt.cur` holds the raw chunk and `rt.scratch` the projection; document); `action_src = Policy`; `substituted = false`; `violation_reached_env = would-be-src != Policy`.

Tests: `state_machine.rs` drives `fsm::next` through `fixtures/fsm/transitions.json` (every one of the 24 rows incl. 17b at least once, with the expected `(to, extra_trips, reason)` and counter values; unreachable-combination guards, e.g. an ack in `Armed` is ignored; `AckOnly` vs `Auto` re-arm; rearm budget exhaustion -> Escalated; handoff timeout -> Terminated). `observe_passthrough.rs`: run `fixtures/episode/synthetic_300.json` (a 300-tick synthetic episode with chunks every 8 ticks, containing a box breach at chunk 12 and a stall from tick 200) through `Fuse` in Observe mode and assert `action == policy action` and `substituted == false` on EVERY tick while `counts.violations_reached_env > 0` and the state sequence still visits `Clamped`/`Braking`; then in Enforce mode assert `violations_reached_env == 0` and that the first `Braking` tick's action equals `clamp(pos)`. `decide_order.rs`: NaN in obs -> Fault on that tick with hold action and Fault persists even when later inputs are finite; NaN in `aux` faults even though nothing consumes aux (deliberate fail-closed, asserted so nobody "fixes" it); `idx >= horizon` -> SCHEMA; `t` jumping 24 -> 30 with `missed_ticks = 0` -> SCHEMA, and 24 -> 30 with `missed_ticks = 5` -> accepted; a chunk with `seq` 3 after 1 -> SCHEMA; `missed_ticks > watchdog` -> WATCHDOG; a chunk with a `TIER0_HARD` trip is rejected and `chunks_rejected` increments; hold is latched (does not change while the (synthetic) position drifts); an episode with Tier 1 disarmed ends with `tally.max_s == NEG_INFINITY` (not 0.0); an async-drop delivery (`t_emit = t - 3`, `idx = 3`) is accepted and `brake_feasible` sees `from = 3` (asserted via a chunk whose rows 0..2 alone would be infeasible).

Acceptance:
```bash
cargo test -p lictor-fuse --test state_machine --test observe_passthrough --test decide_order
cargo clippy -p lictor-fuse --all-targets -- -D warnings && cargo fmt --all --check
cargo check -p lictor-fuse --no-default-features
grep -nE "Instant|SystemTime|HashMap|rayon|mul_add|powf|powi|\.exp\(|\.ln\(|f32|f64::min|f64::max|f64::clamp" crates/lictor-fuse/src crates/lictor-detect/src crates/lictor-core/src | grep -v "^.*//" ; test $? -eq 1   # no hits (deny list, incl. NaN-semantics min/max)
```

---

## WP-4 -- Canonical encoding (JCS float-free), receipts, chains, ledger, handoff/ack, signing, history, keys, and the stdlib Python verifier

Files:
```
crates/lictor-canon/src/jcs.rs   crates/lictor-canon/src/f64enc.rs   crates/lictor-canon/src/digest.rs
crates/lictor-canon/tests/rfc8785.rs   crates/lictor-canon/tests/key_charset.rs   crates/lictor-canon/tests/floatify.rs
crates/lictor-canon/tests/fixtures/jcs/vectors.json
crates/lictor-receipt/src/{tick,body,sign,ledger,curve,handoff,history,keys}.rs
crates/lictor-receipt/tests/{chain,tamper,ledger,ack,history,python_parity}.rs
crates/lictor-receipt/tests/fixtures/receipt/receipt_000007.json   crates/lictor-receipt/tests/fixtures/receipt/ticks_000007.jsonl
crates/lictor-receipt/tests/fixtures/receipt/timing_000007.jsonl   crates/lictor-receipt/tests/fixtures/receipt/ledger.jsonl
crates/lictor-receipt/tests/fixtures/receipt/key.hex   crates/lictor-receipt/tests/fixtures/receipt/pubkey.txt   crates/lictor-receipt/tests/fixtures/receipt/operator.hex
crates/lictor-receipt/tests/fixtures/receipt/handoff.json   crates/lictor-receipt/tests/fixtures/receipt/ack.json
adapters/verify_receipt.py   adapters/tests/test_verify.py
docs/receipt-schema.md
```
Dependencies: WP-0 (compile); WP-3's `Tally` is only used in `From<&Tally>` (compiles against the stub). Vendoring: fetch `RARS-oss/bulla` `crates/bulla-core/src/lib.rs` (`gh api repos/RARS-oss/bulla/contents/crates/bulla-core/src/lib.rs -H 'Accept: application/vnd.github.raw+json'`) and keep a provenance header in `sign.rs`/`ledger.rs` (`//! Vendored from RARS-oss/bulla @ <sha>, crates/bulla-core/src/lib.rs; renamed per docs/ANALYSIS.md sec 5; canonical bytes replaced by lictor-canon JCS.`). If the repo is unreachable, implement from the freeze and say so in the header.

Spec:
- `jcs.rs`: `canon` per RFC 8785 (sort keys by UTF-16 code units; serialise strings with JCS escaping: `\"`, `\\`, `\b \f \n \r \t`, other control chars as `\u00xx` lowercase hex; integers as digits; `true/false/null`; arrays/objects without whitespace); any `Number` that is not an integer with |v| <= 2^53-1 -> `FloatInBody(path)`; `check_keys` walks every object key against `^[a-z0-9_]+$` -> `BadKey(path)`. `f64enc.rs`: `f64_to_hex` (`format!("{:016x}", x.to_bits())`), `f64_from_hex`, `F64Hex` and `F64Array` custom `Serialize`/`Deserialize` (`{"f64":..}` / `{"f64a":..,"shape":[..]}`; base64 standard padded; `shape` product must equal `data.len()`), `floatify`. `digest.rs`: `sha256_hex`, `sha256_jcs`, `canon_of`, `digest_of`.
- `tick.rs`: `tick_event` builds the event from a verdict with `hash = sha256_hex(canon_of(&event_with_empty_hash))`; `verify_tick_chain` recomputes each hash and checks `prev` linkage from `genesis_prev` (first `break_at` = the seq of the first bad event); `timing_event`/`verify_timing_chain` likewise.
- `body.rs`: the frozen structs; `From<&Tally>`; `canonical`/`digest_hex`; `evaluate_fuse` with EXACTLY these notes (strings are grep targets): "the fuse observed but did not enforce -- this receipt does not attest protection"; "N tier-0 violations reached the environment"; "tier 1 armed without a calibration digest"; "fuse latched Fault"; "delay injected without an exec mode".
- `sign.rs`: Ed25519 (`ed25519-dalek` 2, `verify_strict`), `sign` (body_digest + sig over canonical bytes), `verify` (schema == RECEIPT_SCHEMA, canonical == CANONICAL_ID, `pubkey_ok` when an expected pubkey is given, sig, digest, `envelope_digest_ok` = sha256(canon(body.envelope)) recomputed, `counts_ok` = `counts.ticks == verdict_events == timing_events == outcome.steps`, embedded `ticks` chain-consistent with `verdict_chain_head` when `ticks_policy == all` or the tail links to the head, `fuse_ok` recomputed via `evaluate_fuse(.., ephemeral_key = notes contain "ephemeral signing key")` must equal the stored flag), `verify_ticks_file` (chain integrity only; the head comparison and header cross-check are the CLI's, per the frozen doc comment), `TEST_PUBKEYS`.
- `ledger.rs`: entry hash = sha256(canon(entry with hash="")); `verify_ledger` (linkage + seq continuity + recomputed hashes; `break_at`); `read_ledger` (skips the header line); `append_ledger` (reads the current head, writes one line, fsync; header line written when the file is created).
- `handoff.rs`: `HandoffRecord` self-digest (with `run_id`/`arm_id`/`episode_index` digested; no `state_digest` -- `chain_at` is the escalation tick's TickEvent hash); `ack_signing_bytes`; `sign_ack`; `verify_ack` (operator index = slot; `verify_strict`; digest equality; `nonce > last_nonce[slot]`; note <= 200 chars); `load_nonces`/`save_nonces` (`.lictor/verifier_nonce.json`: `{pubkey: nonce}` pretty JSON).
- `curve.rs`: the frozen structs (`DeltaCi`, the extended `CurveMetrics`/`CurveReceiptBody`) + `sign_curve`/`verify_curve` (same mechanism as receipts, schema `lictor-curve/v1`).
- `history.rs`: sbx `feedback/history.rs` lineage (std + serde only): append JSONL capped at 200 lines (rewrite when exceeded), `summarize` -> `tail_streak` (the most recent `first_trip_reason` and how many of the last n episodes share it, if >= 2), `identical_run` (consecutive identical (arm, trips) tails).
- `keys.rs`: `keygen` via `getrandom`, `pubkey_hex`, `load_seed`/`save_seed` (hex64 text; 0600 on unix via `std::os::unix::fs::PermissionsExt` behind an EXPLICIT `#[cfg(unix)]` so a Windows-native build compiles; refuses `/mnt/[a-z]/` paths unless `allow_drvfs`; refuses to overwrite unless `force`), `default_key_dir` (`$LICTOR_KEYS` else `$HOME/.lictor`).
- Fixtures: a Rust test (`tests/chain.rs`, gated by `LICTOR_WRITE_FIXTURES=1`) generates a deterministic 300-tick synthetic receipt + ticks + timing + a 5-entry ledger from a FIXED seed key (`key.hex` committed; it is a test key, say so in the file) so the committed fixtures are reproducible; `handoff.json`/`ack.json` from `operator.hex`.
- `adapters/verify_receipt.py`: STDLIB ONLY (`json, hashlib, base64, sys, pathlib, re`) plus a vendored pure-Python Ed25519 verify (~60 lines, the classic reference implementation; slow is fine). Usage: `verify_receipt.py RECEIPT.json [--ticks TICKS.jsonl] [--ledger LEDGER.jsonl] [--pubkey HEX] [--parity]`. Recomputes canonical bytes with `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()` AFTER asserting every key matches `^[a-z0-9_]+$` (one-line walk; this is what makes Python code-point ordering equal to JCS UTF-16 ordering, so it is re-checked here, never assumed), checks `body_digest`, the signature, `--pubkey` equality (`FAIL: pubkey mismatch`), the embedded tick tail, the full ticks chain AND its head against `verdict_chain_head` when given (`FAIL: verdict chain head mismatch`), the ledger chain when given; prints exactly `OK` or `FAIL: <reason>`; `--parity` additionally prints `canonical bytes: IDENTICAL` when the recomputed digest equals `body_digest` and `signature: ok`. Exit 0/1. `--pubkey` is REQUIRED by `scripts/demo.sh`, `bench/tamper` and CI. Keep it readable in one screen plus the Ed25519 block.
- `docs/receipt-schema.md`: the canonical profile, every field of every schema with its meaning and why it is bound (ARCHITECTURE sec 8 table), the chain rules, the ledger tail-truncation gap, the divergences from bulla-core (so a future shared crate is mechanical).

Tests: `rfc8785.rs` (the RFC 8785 appendix test vectors in `vectors.json`, incl. Unicode key sorting and escaping; float rejection; integer range). `key_charset.rs` (every committed fixture in this crate and in `crates/lictor-receipt/tests/fixtures` passes `check_keys`). `floatify.rs`. `chain.rs` (build, verify, head equality across two builds). `tamper.rs` (EVERY top-level and nested scalar field of the fixture body mutated one at a time -> `sig_ok == false`; one tick edited at seq 17 -> `verify_ticks_file.break_at == Some(17)`; a RE-CHAINED ticks file (tick 17 edited and hashes 17..299 recomputed) -> `ok == true` but `head != body.verdict_chain_head`; a whitespace-only edit of the pretty-printed receipt -> still intact (documented: canonical bytes are recomputed); a receipt re-signed with a different key -> `sig_ok == true` but `pubkey_ok == false` when the original pubkey is expected; ledger entry 3 deleted -> `break_at == Some(3)`; an Observe receipt -> `intact() && !fuse_ok` with the exact note; an ephemeral-key receipt -> `!fuse_ok`). `ledger.rs` (append/read round trip in a tempdir; header handling). `ack.rs` (accept; each `AckError` variant). `history.rs`. `python_parity.rs` (spawns `python3 adapters/verify_receipt.py --parity <fixture>` when `python3` is available and asserts the two exact lines; skips with a message otherwise).

Acceptance:
```bash
cargo test -p lictor-canon -p lictor-receipt
cargo clippy -p lictor-canon -p lictor-receipt --all-targets -- -D warnings && cargo fmt --all --check
python3 adapters/verify_receipt.py crates/lictor-receipt/tests/fixtures/receipt/receipt_000007.json --ticks crates/lictor-receipt/tests/fixtures/receipt/ticks_000007.jsonl --pubkey $(cat crates/lictor-receipt/tests/fixtures/receipt/pubkey.txt) --parity | tee /tmp/p.txt
grep -q '^OK' /tmp/p.txt && grep -q 'canonical bytes: IDENTICAL' /tmp/p.txt
/mnt/d/lictor/venv/bin/python -m pytest adapters/tests/test_verify.py -q
```

---

## WP-5 -- Offline calibration, envelope fit, metrics, Layer-A sweep, Layer-B curve (+ their CLI commands)

Files:
```
crates/lictor-calib/src/{file,traces,calibrate,envfit,metrics,sweep,curve}.rs
crates/lictor-calib/tests/{quantile,coverage,metrics,binning,sweep_synthetic,curve_refusals}.rs
crates/lictor-calib/tests/fixtures/calib/synthetic_traces.json   crates/lictor-calib/tests/fixtures/calib/expected.json
crates/lictor-cli/src/cmd/calibrate.rs   crates/lictor-cli/src/cmd/sweep.rs   crates/lictor-cli/src/cmd/curve.rs   crates/lictor-cli/src/cmd/envelope.rs
docs/calibration.md
```
Dependencies: WP-0 (compile). Test-level: WP-1 (bin_of, envelope), WP-2 (conformal aggregate -- reuse `lictor_detect::conformal::{standardise, aggregate}` for `episode_max_s`; never reimplement), WP-4 (receipt/ticks reading, digests, curve signing), WP-6 (`TraceReader` for `load_traces`).

Spec:
- `file.rs`: `CalibrationFile`, `SeedPool`, `Tier0Percentiles` EXACTLY as frozen (keys in that order); `load` (verifies `digest`), `save` (pretty, sorted keys, recomputed digest), `compile` -> `CalibrationC` (expand `center`/`scale` F64Array [t_grid, NFEAT] into the fixed arrays; `t_grid` 1 or 100 only; `n_calib` copied verbatim), `digest_hex`, `loaded` -> `lictor_runtime::session::CalibrationLoaded`.
- `traces.rs`: `load_traces(run_dir, arm)` reads `ledger.jsonl` (verify chain; refuse if broken), every receipt (verify signature/digest; refuse if not intact), every ticks file (verify chain against the receipt head; refuse on break), and every trace via `lictor_runtime::trace::TraceReader` (for `coverage` per tick: the harness sends `aux[3] = coverage` on every tick, `_get_coverage()` at t = 0 -- reconstruct `coverage[t]` from the parsed `TickReq`s; NEVER a second hand-rolled NDJSON parser); builds `Trace`s in episode-index order; `fuse_crash` receipts yield a Trace with `steps = 0`, `success = false` and no ticks.
- `calibrate.rs`: ARCHITECTURE sec 7 steps 2-9 exactly: successes only; refuses eval-pool seeds; refuses `horizon_ticks != envelope.embodiment.horizon_ticks` or `!= run.env.max_episode_steps`; deterministic split by sorted episode index -- `split = 2`: 70/30 fit/holdout (`n_scale = n_calib = |fit|`), `split = 3`: 40/30/30 (`n_scale = |A|`, `n_calib = |B|`, holdout = C); `robust_center_scale` (median, 1.4826*MAD floored 1e-9, per bin per feature, `total_cmp` sort, fixed order, deterministic fills, dropped features noted); `episode_max_s`; `split_quantile` over the `n_calib` scores (`k = ceil((n_calib+1)*(den-num)/den)` in integer arithmetic; `+INF` + note when `k > n_calib`); holdout FPR at K-of-N and at K=1; binds `envelope_digest`, `embodiment_digest`, `policy_digest`; write notes.
- `envfit.rs`: from the calibration arm's SUCCESSFUL traces, recompute the Tier-0 quantities (commanded speed, accel, jerk, reach) from the request-line chunks with the same formulas as `lictor-detect` (call `lictor_detect::tier0` helpers where exposed; otherwise document the duplication and test parity), take the empirical `quantile` (nearest-rank on `total_cmp`-sorted values), multiply by `slack`, write a new envelope with `[fit]` filled, `operators` replaced by the `--operator` list when given (the oracle envelope), the `embodiment` table byte-identical to the base (a test asserts `embodiment_digest` unchanged), and a markdown report (per quantity: n, p50, p99, p99.9, chosen limit, base limit).
- `metrics.rs`: Clopper-Pearson (exact beta quantiles via an in-crate regularised incomplete beta with bisection -- no external crate), McNemar exact (two-sided binomial on b, c), `ponr_tick`, `roc_auc` (rank-based, ties averaged), `aucpdt` (VLA-FAIL: area under the detection-rate-vs-lead-time curve normalised by horizon), `paired_bootstrap_diff` with splitmix64.
- `sweep.rs`: for each detector spec and alpha: when an `artefacts` entry matches (alpha, gate, kn, method) use its center/scale/tau verbatim (`tau_source: "artefact"`), else calibrate on the calib traces (successes; full fit, no holdout; `tau_source: "fit"`); then evaluate on the eval traces against the `obs-d0` labels (`success` from the receipt): TPR/FPR (episode fired = K-of-N fire at any tick OR, when `tier0`, any TIER0 stop-worthy trip recomputed from `trips` bits in the ticks), lead vs `ponr_tick`, AUCPDT, ROC-AUC on `max_s`, bacc, `fire_frac_mean`; emit `SweepPoint` rows with EXACTLY the sweep.jsonl keys (`tau` serialised as `null` when infinite).
- `curve.rs`: `curve(run_dir, arm, baseline, opts)`: load both arms (verified) plus the latency-control arm when present (`opts.latency_control`, default `obs-d<delay_steps>` if it exists in the run); read `run.json` (sha256 bound; declared seeds; per-arm `tier1`/`alpha` -- refuse on disagreement with the receipts' `budget`); refuse if the arm's receipts carry more than one `pubkey` or any `"ephemeral signing key"` note; compute the declared pool (run.json seeds intersected with the arm's pool) and the index sets: differences from the baseline or from the declared pool -> refuse unless `opts.partial`, in which case `partial = true` and `n_declared/n_present/n_missing/missing_indices` are bound; missing indices in the ARM count as `success = false, stopped = true`; missing in the BASELINE are excluded from pairing and listed; `n < 100` -> refuse unless `opts.allow_small` (`small_n = true`); check `init_state_digest` per episode index (collect `pair_mismatches`; exclude them); check calibration seed overlap; compute every `CurveMetrics` field per ARCHITECTURE 10.7 from receipts only, including `delta_vs_baseline` and `delta_vs_latency_control` (`paired_bootstrap_diff` with `opts.n_boot`/`boot_seed` + `mcnemar_exact`), `tce_valid_frac` (from the ticks' `valid` bit 0), `latency_label` (from the receipts; refuse if the arm's receipts disagree), `ledger_chain_ok`, `arm_config_digest` = sha256(canon({"envelope_digest", "calibration_digest", "budget"})) (refuse if it differs across the arm's receipts), `receipt_pubkey`, `cross_run_mismatches` (when `opts.compare_run` is given: same (arm, seed) with a different `verdict_chain_head`); return the body (signing/writing is the CLI's job via WP-4's `sign_curve`).
- CLI: `calibrate` (`--split 2|3`), `sweep` (`--calibration-dir`), `curve` (`--latency-control`, `--allow-small`, `--partial`, `--compare-run`; refuses with exit 1 and the frozen line `ledger BROKEN at seq=N -- entries are missing or edited; this curve point cannot be trusted` when `ledger_chain_ok` is false, and with `REFUSED: <reason>` lines of the CLI SURFACE shape for the other refusals; writes `<arm>.json` signed when `--key` given else unsigned with `"pubkey":""`, and `summary.csv` with the frozen column list, `pilot` first), `envelope init|check|digest|show|fit` per the CLI SURFACE (`--operator` on `init` and `fit`). `envelope init --profile pusht` emits the base TOML from `include_str!("../../../../envelopes/pusht.base.toml")` (the file exists from WP-0).
- `docs/calibration.md`: the procedure, the honest guarantee statement (approximate under the 2-way split because center/scale are fitted on the same episodes; exact at K = 1 under `--split 3`), the K-of-N caveat, the seed-pool rule, the binning rule, the embodiment-digest binding rule, the file schema, and worked numbers with `n_calib = 137` (2-way at 196 successes): alpha = 1/100 -> k = 137 (tau = the max, barely non-degenerate); 2/100 -> 136; 5/100 -> 132; 10/100 -> 125; 20/100 -> 111; 5/1000 -> 138 > 137 -> +INF; and with `n_calib = 59` (3-way): 1/100 -> 60 > 59 -> +INF; 2/100 -> 59; 5/100 -> 57; 10/100 -> 54; 20/100 -> 48; plus the pilot's `n_calib ~ 45` (100 calibration episodes): 1/100 and 2/100 -> +INF, 5/100 -> 44.

Tests: `quantile.rs` (k computation incl. `k > n`; the n = 137 and n = 59 tables above; known small arrays), `curve_refusals.rs` (a synthetic two-arm run in a tempdir built from WP-4 fixtures: missing index -> refused / `--partial` counts it as a failure; mixed pubkeys -> refused; n = 10 -> refused / `--allow-small` sets `small_n`; `tier1` disagreement with run.json -> refused), `coverage.rs` (synthetic exchangeable data: 2 000 calibration draws, 20 000 fresh nominal draws; empirical firing rate within Monte-Carlo tolerance of alpha for K=1; strictly below for K=3-of-5), `metrics.rs` (Clopper-Pearson against known table values, e.g. k=0,n=100 -> (0, 0.0362); McNemar b=5,c=15 -> 0.0414; ROC-AUC on a hand case; ponr on a hand coverage trace), `binning.rs` (calibrator uses `lictor_core::bin_of` -- a test that reimplementation would fail: compare against the core function on 1 000 random t), `sweep_synthetic.rs` (a synthetic trace set where a feature is informative by construction gives TPR > 0.9 at FPR <= 0.1).

Acceptance:
```bash
cargo test -p lictor-calib
cargo clippy -p lictor-calib -p lictor-cli --all-targets -- -D warnings && cargo fmt --all --check
$CARGO_TARGET_DIR/debug/lictor envelope init --profile pusht -o /tmp/e.toml && $CARGO_TARGET_DIR/debug/lictor envelope check /tmp/e.toml
$CARGO_TARGET_DIR/debug/lictor calibrate --help && $CARGO_TARGET_DIR/debug/lictor sweep --help && $CARGO_TARGET_DIR/debug/lictor curve --help
```

---

## WP-6 -- Runtime: wire schema + codec, Session, episode writer, latency histogram, trace files, `lictor serve`, Python client

Files:
```
crates/lictor-runtime/src/{wire,codec,session,episode,latency,trace}.rs
crates/lictor-runtime/tests/{roundtrip,session_fault,session_episode}.rs
crates/lictor-runtime/tests/fixtures/wire/golden_requests.ndjson   crates/lictor-runtime/tests/fixtures/wire/golden_responses.ndjson
crates/lictor-runtime/tests/fixtures/wire/bad_lines.ndjson
crates/lictor-cli/src/cmd/serve.rs   crates/lictor-cli/src/cmd/crash_receipt.rs
adapters/lictor_client.py   adapters/tests/test_client.py
docs/wire-protocol.md
```
Dependencies: WP-0 (compile). Test-level: WP-1, WP-2, WP-3 (a real `decide`), WP-4 (chains/signing). The calibration-loaded path is tested with a hand-built `CalibrationLoaded` (no test-level dependency on WP-5, so the WP-5 -> WP-6 edge -- `load_traces` needs `TraceReader` -- keeps the graph acyclic).

Spec:
- `wire.rs`: EXACTLY the frozen payload structs of A.1 (WP-0 stubbed them; WP-6 fills derives/attributes: `#[serde(deny_unknown_fields)]` on EVERY request payload struct, not only the enum), real-valued request fields as `Option<f64>` (`null` == non-finite; `Staging` maps `None` -> `f64::NAN` so the GUARD raises NONFINITE), `chunk.a` as `Vec<Vec<Option<f64>>>`. Response structs mirror the JSON examples exactly (`trips` as names + `trip_mask` integer; `scores` object; `handoff` as `Option<HandoffRecord>`; `ack_result`); `scores.s` and `tau` are plain `f64` and serde_json emits `null` for +-inf -- intended and documented in the module comment.
- `codec.rs`: line reader (rejects > MAX_LINE, non-UTF-8, embedded `\r`), `read_request` returns `Ok(None)` at EOF, `Err(InvalidData(msg))` with the serde message on malformed input; `write_response` writes compact JSON + `\n` and flushes.
- `session.rs`: `Staging` (frozen; the ONE wire -> `TickInput` conversion; `stage` checks dims/h/d/exec/idx-vs-t_emit and returns `Err` for the session to raise SCHEMA; the continuity checks live in `decide()`), `CalibrationLoaded`, `default_latency_label`. `Session::new` compiles the envelope (+ the already-loaded calibration; REFUSES with an error when `calibration.embodiment_digest != envelope.embodiment_digest()`), computes `envelope_toml_sha` and the binary sha256 (`current_exe`), prepares `Fuse`, `Staging`, chain heads, `Hist`, and loads `<out>/.lictor/verifier_nonce.json` into the per-operator `last_nonce` table (kept across episodes; saved after every accepted ack). `handle`: hello (cross-checks mode/dims/digests -> `error{code:"envelope",fatal:true}` on mismatch; replies hello_ok with `embodiment_digest`, `ephemeral_key`, `latency_label`), episode_begin (stores bindings incl. `inputs` and `client`; refuses with `error{code:"envelope",fatal:true}` when a calibration is loaded and `binding.policy.weights_sha256 != calibration.policy_digest`; `fuse.reset`; genesis chains; opens the ticks/timing writers under `out_dir` if set), tick (state checks; verify an `ack` via `verify_ack` against the pending handoff -> `VerifiedAck` or `ack_result:"rejected:..."`; `staging.stage`; `let t0 = Instant::now(); let v = fuse.step(&staging.input(..)); let decide_ns = t0.elapsed()`; THEN fold `tick_event`/`timing_event`, write ticks/timing lines, build the response incl. `HandoffRecord` (with run_id/arm_id/episode_index, `chain_at` = this tick's event hash) when `v.handoff_seq.is_some()` (store as pending); `io_ns` arrives later via `note_io_ns(seq, ns)` and is folded into that seq's `TimingEvent` on the next request), episode_end (ACCEPTED in every post-episode_begin state including latched Fault: build `ReceiptBody` per the freeze from bindings + `fuse.finish()` + outcome + heads + `latency.summary(cfg.latency_label)` + tail ticks + `inputs` (host entries + `lictor:bin`, `lictor:envelope`, `lictor:calibration`) + `client`; `evaluate_fuse(.., ephemeral_key)`; sign with the key (or an ephemeral key generated at startup when `--key` absent, with `fuse_notes += "ephemeral signing key"`), `write_episode` (receipt, ticks header+lines already streamed, timing, then ledger LAST), reply), bye. Fault latch: any schema/protocol error -> reply `error{fatal:true}` AND mark the session faulted for the rest of the episode; every subsequent tick of that episode is still fed through `decide()` with `TickInput.schema_fault = true`, so the GUARD stage raises `TripMask::SCHEMA`, latches `Fault` and returns the hold action through the normal frozen path (the session never post-processes a verdict).
- `episode.rs`: `episode_paths`, `write_episode` (mkdir -p, pretty JSON with sorted keys for the receipt, JSONL for ticks/timing with the schema header line, `append_ledger` last), `write_crash_episode` (the frozen host-side crash receipt).
- `cmd/crash_receipt.rs`: the `lictor crash-receipt` command of the CLI SURFACE -> `write_crash_episode`; prints the ledger seq and receipt path; exit 0.
- `latency.rs`: `hdrhistogram` wrapper + `impl Default`; `summary` fills `LatencySummary` with the label passed in (the session passes `cfg.latency_label`); `to_csv` bucket dump for F5.
- `trace.rs`: `#meta` + verbatim request lines; reader yields `(line_no, raw, Request)`.
- `cmd/serve.rs`: parse args; load `--calibration` through `lictor_calib::CalibrationFile::load(..).loaded(..)` (lictor-cli depends on both crates; lictor-runtime does not depend on lictor-calib) -> `SessionConfig` (key from `--key` or `$LICTOR_KEYS/key.hex` if it exists, else ephemeral with a stderr warning; `latency_label` from `--latency-label` or `default_latency_label()`) -> `Session::new` (exit 2 with the message on an embodiment-digest mismatch) -> loop: read stdin line (`io_t0`), `handle`, write stdout, `note_io_ns`; stderr for diagnostics; `--on-fault abort` exits 3 on the first fatal error instead of latching (debugging only). Exit 0 on `bye`/EOF.
- `adapters/lictor_client.py`: the frozen surface (`CLIENT_VERSION`, `LictorFault`, `LictorClient`, `chunk_msg`, `finite_or_none`); spawns the binary with `-u`-free pipes (`bufsize=1`, text mode, `encoding="utf-8"`), `id` monotone, `select`-based read timeout; every request goes through `finite_or_none` then `json.dumps(..., allow_nan=False, separators=(",", ":"))`; `tick()` maps `null -> -math.inf` for `scores.s` and `+math.inf` for `tau` ONLY; `hello()` asserts `tier1_armed == expect_tier1` and the calibration alpha == `expect_alpha` (raises `LictorFault(reason="protocol")`); on timeout / `fatal` with a dead child / exit / id mismatch it KILLS the child (`terminate()` then `kill()`), sets `alive = False` and raises `LictorFault(last_safe_action = safe_action() if given else None, reason)`; on `fatal` with a live child it returns the error dict and lets the harness send `episode_end`; `win_to_wsl`/`wsl_to_win`; binary discovery order: explicit arg, `$LICTOR_BIN`, `$CARGO_TARGET_DIR/release/lictor`, `/mnt/d/lictor/target/release/lictor`, `lictor` on PATH, `/mnt/c/.../lictor.exe`.
- `docs/wire-protocol.md`: the WIRE PROTOCOL section verbatim plus the fail-closed host contract and a sequence diagram (ASCII).

Tests: `roundtrip.rs` (every golden request line parses and re-serialises semantically; every golden response line matches the struct; every `bad_lines.ndjson` line is rejected with the expected reason -- the file MUST include an unknown key inside `obs`, inside `chunk`, inside `run` and inside `ack`, a `chunk` with 14 rows, `t_emit > t`, `idx != t - t_emit`, a second chunk with `seq` 0, and the literal `NaN`). `session_fault.rs` (tick before episode_begin -> error fatal + subsequent ticks are `fault`/`hold`; unknown nested field -> error; `h` mismatch -> error; non-monotonic id -> error; `t` jump with `missed_ticks = 0` -> `schema`; `null` in `obs.pos` -> verdict `status:"fault"` with `trips` containing `nonfinite`; `episode_end` AFTER a fatal error still yields an `episode_receipt` with `counts.terminal_state == "fault"`, `fuse_ok == false` and a ledger entry). `session_episode.rs` (a full synthetic 300-tick episode through `Session` in a tempdir: receipt verifies with `lictor_receipt::verify` (incl. `counts_ok`, `envelope_digest_ok`), ticks file verifies against the head, ledger has 1 entry, the first tick's response has `"s":null` and `"tau":null` when Tier 1 is disarmed and they round-trip through the Rust structs, `hello` digest mismatch is fatal, an embodiment-digest mismatch is refused by `Session::new`, a `weights_sha256` != `policy_digest` is refused at `episode_begin`, `write_crash_episode` produces a receipt that verifies with `fuse_ok == false`, two runs of the same episode give identical `verdict_chain_head` and different `timing_chain_head`, and an accepted ack's nonce persists to `verifier_nonce.json` and is rejected as `nonce_replay` by a fresh Session in the same `out_dir`).

Acceptance:
```bash
cargo test -p lictor-runtime
cargo clippy -p lictor-runtime -p lictor-cli --all-targets -- -D warnings && cargo fmt --all --check
cargo build --release -p lictor-cli
printf '%s\n' '{"id":1,"kind":"hello","proto":"lictor-wire/v1","client":"t","mode":"observe","embodiment_id":"gym_pusht/PushT-v0","action_dim":2,"pos_dim":2,"horizon":15,"exec_steps":8,"envelope_digest":"'$($CARGO_TARGET_DIR/release/lictor envelope digest envelopes/pusht.base.toml)'","calibration_digest":null}' '{"id":2,"kind":"bye"}' | $CARGO_TARGET_DIR/release/lictor serve --envelope envelopes/pusht.base.toml --mode observe | grep -q hello_ok
/mnt/d/lictor/venv/bin/python -m pytest adapters/tests/test_client.py -q      # drives the real binary through a 20-tick synthetic episode; asserts null->inf mapping, allow_nan=False, kill-on-timeout, hello tier1/alpha asserts
```

---

## WP-7 -- CLI: verify, replay, selftest, ledger, key, ack, history, version, render

Files:
```
crates/lictor-cli/src/cmd/{verify,replay,selftest,ledger,key,ack,history,version}.rs
crates/lictor-cli/src/render.rs
crates/lictor-cli/tests/cli.rs
crates/lictor-cli/tests/fixtures/cli/README.md   (documents which fixtures from other crates the CLI tests reuse, by relative path)
```
Dependencies: WP-0 (compile). Test-level: WP-4 (fixtures + verify), WP-6 (`Session`, `TraceReader`), WP-3.

Spec (frozen output strings are in the CLI SURFACE; do not reword):
- `render.rs`: deterministic, length-capped human rendering (sbx render doctrine): `kv(label, value)` aligned to 15 columns, `note(s)`, a `glossary(reason: ReasonCode) -> &str` that returns `reason_text` -- the human-escalation channel; `--json` short-circuits to serde output everywhere.
- `verify`: load `SignedReceipt`; `lictor_receipt::verify(sr, --pubkey)`; optional `--ticks` (`verify_ticks_file`, THEN compare `report.head` with `body.verdict_chain_head` -> the frozen `verdict chain  HEAD MISMATCH recomputed=<hex> signed=<hex>` line, and cross-check the ticks header run_id/arm_id/episode_index/genesis and the embedded tail's first `prev` against the file), `--timing` (chain only), `--ledger` (`verify_ledger` + the receipt's digest must appear), `--curve` (`verify_curve` and `ledger_head` match), `--calibration` (budget alpha/gate/kn equal the file's); print the aligned block incl. the `pubkey`, `envelope` and `counts` lines; `WARNING: signed with the committed test key` when `pubkey in TEST_PUBKEYS`; `FUSE HELD (...)` or `FUSE NOT ENFORCED` + notes; exit 0 only when intact AND fuse_ok; else 1 (scripts use `--json` and the `intact` field).
- `replay`: `TraceReader` -> for `--repeat N`: fresh `Session` (no out_dir, no key, `mode` from the `#meta` unless overridden) -> feed every request -> collect the verdict chain head; print the frozen lines; `--expect` compares; exit 1 on any divergence (name the first diverging repeat and tick).
- `selftest`: no arguments; runs in a tempdir: envelope fixture round trip + digest stability; brake fixture parity (`crates/lictor-detect/tests/fixtures/brake/cases.json` via `include_str!` -- the file exists from WP-0 as `[]`; selftest reports `[skip] brake fixture empty` until WP-1 fills it, and likewise for tier-1); a synthetic 300-tick episode through `Session` -> receipt -> `verify` ok -> ticks chain ok -> `replay --repeat 8` byte-identical -> four tampers (flip a count in the body -> signature FAIL; edit tick 17 -> chain BROKEN at seq=17; edit tick 17 and re-chain 17..299 -> HEAD MISMATCH; delete ledger entry 3 -> ledger BROKEN at seq=3) -> `verify_receipt.py --pubkey` parity when python3 exists. One line per check (`  [ok] ...` / `  [FAIL] ...` / `  [skip] ...`), final `SELFTEST PASS`/`SELFTEST FAIL`, exit 0/1.
- `ledger verify|append`; `key init|pub` (default path `default_key_dir()/key.hex`, `--role operator` -> `operator.hex`; `--force` to overwrite, `--i-know` for `/mnt/[a-z]/`; prints `pubkey <hex>`, `stored <path> (mode 0600)` and `custody: this file signs every receipt you produce; back it up off the repo and never commit it`); `ack` (loads the handoff record or takes a digest; nonce default = 1 + the largest nonce found in `default_key_dir()/ack_nonce.json` for that operator; writes the token JSON to `-o` or stdout); `history` (`summarize`, prints tail-streak lines like `tail-streak: brake_tier1_cp in 4 of the last 6 episodes on t01-a05-d0`); `version` (version, git, binary sha256). `render.rs` strips ASCII control characters (and ESC sequences) from every host- or operator-supplied string before printing (`AckToken.note`, `reason_text`, `fuse_notes`, arm/run ids) so a signed note cannot inject terminal escapes.

Tests: `cli.rs` uses `assert_cmd`-free plain `std::process::Command` on the debug binary: `verify` on the WP-4 fixture prints `signature      ok`, the test-key WARNING, and `FUSE HELD` (or `FUSE NOT ENFORCED` for the observe fixture) and exits accordingly; tampered copy prints `signature      FAIL`, exit 1; re-chained ticks print `HEAD MISMATCH`; `--pubkey 00..` prints `pubkey         MISMATCH`; `key init` into a `/mnt/c/...`-shaped tempdir path is refused without `--i-know` (test via a `LICTOR_KEYS` pointing at a path under a fake `/mnt/c` inside the tempdir); `replay --repeat 3` on the WP-6 golden trace prints `replays 3/3 byte-identical`; `selftest` prints `SELFTEST PASS`; `key init` + `key pub` round trip; `ack` produces a token that `verify_ack` accepts; `version` prints three fields.

Acceptance:
```bash
cargo test -p lictor-cli
cargo clippy -p lictor-cli --all-targets -- -D warnings && cargo fmt --all --check
cargo build --release -p lictor-cli && $CARGO_TARGET_DIR/release/lictor selftest | tee /tmp/s.txt && grep -q "SELFTEST PASS" /tmp/s.txt
```

---

## WP-8 -- PushT harness: lerobot compat + migration, policy with the n_action_steps=15 override, executor with delay injection, arms, resumable paired-seed runner, fault injection, replay policy, microbench

Files:
```
harness/compat.py  harness/seeds.py  harness/pusht_rollout.py  harness/executor.py  harness/inject.py  harness/policy_replay.py
harness/arms.py  harness/run.py  harness/microbench.py  harness/env.sh  harness/requirements.txt  harness/README.md
harness/tests/test_h15.py  harness/tests/test_executor.py  harness/tests/test_harness.py  harness/tests/test_seeds.py
```
Dependencies: WP-0 (docs) -- start immediately against the freeze; live runs need WP-6's `adapters/lictor_client.py` and a built binary. Everything in `harness/tests` except `test_h15.py` must run WITHOUT torch/lerobot (use `policy_replay`).

Spec:
- `env.sh`: exports `PATH`, `CARGO_TARGET_DIR=/mnt/d/lictor/target`, `LICTOR_RESULTS=/mnt/d/lictor/results`, `LICTOR_MODELS=/mnt/d/lictor/models`, `HF_HOME=/mnt/d/lictor/hf`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 SDL_VIDEODRIVER=dummy`, `LICTOR_BIN=$CARGO_TARGET_DIR/release/lictor`, and activates `/mnt/d/lictor/venv`.
- `compat.py`: isolates EVERY lerobot import (first `import lerobot.policies` -- the cv2 circular import is the known failure mode; only `opencv-python-headless` may be installed); asserts `lerobot.__version__ == "0.6.1"` (env override `LICTOR_ALLOW_LEROBOT_VERSION=1` to proceed with a warning); `load_policy(device) -> (policy, pre, post, PolicyBinding dict)`: loads from `$LICTOR_MODEL` (default `$LICTOR_MODELS/diffusion_pusht_migrated`, already present); if that directory is missing, runs `python -m lerobot.processor.migrate_policy_normalization --pretrained-path lerobot/diffusion_pusht --output-dir <that dir>` (the script is `lerobot/processor/migrate_policy_normalization.py`, named explicitly, never searched for) and then loads `DiffusionPolicy.from_pretrained(dir)` + `make_pre_post_processors(policy.config, pretrained_path=dir)`; binds `repo_id`, `revision` (HF commit sha), `weights_sha256` (sha256 of the safetensors actually loaded), `horizon/n_action_steps/n_obs_steps/num_inference_steps`, `device`, `dtype`, `normalization_migrated`, `migration_script`. `make_env(seed)` -> `gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array")`; `EnvBinding` from `importlib.metadata.version` for gym-pusht, gymnasium, pymunk, numpy (NEVER from the doc examples) plus `max_episode_steps`, `vel_source = "info.vel_agent"`, `coverage_t0 = "env.unwrapped._get_coverage()"`. `host_binding()` from `platform`/`torch`. `inputs_manifest()` -> sha256 of `harness/{pusht_rollout,executor,compat,inject}.py` and the envelope file, repo-relative keys.
- `pusht_rollout.py`: ONE episode. `policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1` AFTER loading, BEFORE the first call, and the six runtime asserts of ARCHITECTURE sec 0: (1) `policy.diffusion.config is policy.config` and `n_action_steps == 15`; (2) `all(len(q) == 0 for q in policy._queues.values())` before EVERY chunk call -- the harness NEVER calls `select_action`; (3) chunk shape `(1, 15, 2)`; (4) the hand-built history batch (observation DUPLICATED at t = 0, `[obs_{t-1}, obs_t]` afterwards, through the 0.6 preprocessor) `torch.equal` to what `populate_queues` builds (checked on the first two chunks of every episode in `--strict` mode, on every chunk in `test_h15`); (5) `torch.equal(post(a8), post(a15)[:, :8])` in `test_h15`; (6) `policy.diffusion.num_inference_steps == 100`, fp32 parameters, `torch.are_deterministic_algorithms_enabled()`. Before every `predict_action_chunk`: `torch.manual_seed(hash64(episode_seed, chunk_idx))`; chunk -> numpy float64 (15, 2) via the policy postprocessor (unnormalised); `init_state_digest` = sha256 of `np.asarray([agent_x, agent_y, block_x, block_y, block_angle], float64).tobytes()` after `reset(seed)`; per tick: `client.tick(t, idx, pos=agent_pos, vel=info["vel_agent"], aux=[block_x, block_y, block_angle, coverage], chunk=..., ...)` with `coverage = env.unwrapped._get_coverage()` at t = 0 and `info["coverage"]` afterwards, chunk labelling per the WIRE rule (sync/freeze: `t_emit = t, idx = 0`; async drop: `t_emit = t - d, idx = d`; always all 15 rows), execute `verdict["action"]` VERBATIM via `env.step`; in observe mode assert `verdict["action"] == policy_action` (count `policy_action_equal_ticks`); handle `--on-escalate` (`terminate_fail`: stop, `ended_by="escalation_terminate"`; `oracle_resume`: call the binary `subprocess.run([lictor, "ack", "--handoff", digest, "--decision", "resume", "--key", operator_hex, "--nonce", str(n)])` and pass the token in the next tick's `ack`; the oracle arm serves `envelopes/pusht.oracle.toml`; `continue`); on a `fatal` error with a live child: stop stepping, send `episode_end(ended_by="fault")`, return; on `LictorFault` (child dead): apply `clamp_box(current agent_pos)` for the step, return `ended_by="fuse_crash"` so `run.py` writes the crash receipt and respawns. Returns the index record (FILE FORMATS) incl. `receipt_digest` from `episode_receipt`.
- `executor.py`: `DelayedExecutor(policy_fn, d, exec_mode, stitch, exec_steps=8)`: sync = request at t, available at t+d, hold-last meanwhile; async = keep draining the previous chunk's spare actions (7), drop/freeze stitch, hold-last on underrun; yields `(t, idx, chunk_or_None, policy_action)` per tick; deterministic; pure Python (no torch) so it is unit-testable with a fake policy.
- `inject.py`: the five injectors on `numpy.random.Philox(key=episode_seed ^ 0xFA17)`; each returns the `fault_injection` binding dict; applied to the chunk (`action_spike`, `chunk_truncate`), the observation (`obs_dropout`, `obs_noise`), or the executor (`latency_spike`).
- `policy_replay.py`: `--policy replay:FILE` -- a policy that returns recorded chunks from `bench/fixtures/chunks/pusht_chunks.ndjson` (one `{"seed":..,"chunk_idx":..,"a":[[..]]}` per line; missing entries -> hold the last chunk) and a fake env (`FakePushT`) with the exact PD agent dynamics and a scripted coverage curve, so `run.py` works with NO torch (CI).
- `arms.py`: the ARCHITECTURE 10.5 table as data (`Arm(arm_id, mode, tier0, tier1: bool, alpha: tuple[int, int] | None, d, exec, stitch, on_escalate, calibration, envelope, injection, rearm)`; the oracle arm's `envelope = "envelopes/pusht.oracle.toml"`); `arm_ids()`; `--arms` accepts globs. `LictorClient.hello` is always called with `expect_tier1 = arm.tier1, expect_alpha = arm.alpha`.
- `run.py`: `--run-id`, `--arms`, `--seeds 0-59|900000-900099`, `--pool eval|calib`, `--envelope`, `--calibration-dir`, `--out $LICTOR_RESULTS`, `--workers N` (multiprocessing; one `lictor serve` child per worker; arm-major seed-ascending assignment; ONE env per worker, never batched), `--plan [--budget-min M]` (prints the projected wall-clock per step from `microbench.json` and REFUSES to start a step whose projection exceeds the budget; picks 1 or 2 workers from the measured probe), `--summary` (table from ledgers; `pair mismatch` lines; `cross-run` lines comparing `verdict_chain_head` for (arm, seed) shared with `--compare-run`), `--check-pools` (prints `pools disjoint: ok`), `--resume` default / `--fresh`, `--on-escalate`, `--policy replay:FILE`, `--device cuda|cpu`. Writes `run.json` (arms with `tier1`/`alpha`, seeds, pins, commands, lictor build, worker count) and records its sha256 in `--summary`; rebuilds `index.jsonl` from `ledger.jsonl` at startup (verifying the chain via `lictor ledger verify`); per episode: `pusht_rollout.run_episode` -> the fuse writes receipt/ticks/ledger -> harness appends `index.jsonl`; on `ended_by == "fuse_crash"`: run `lictor crash-receipt ...` (so the ledger gets the failure), respawn the worker's child, re-`hello`, retry that index ONCE, then abort the arm with a message; `--resume` never re-runs an index that has a ledger entry (crash receipts included); refuses to aggregate arms whose `init_state_digest` disagrees.
- `microbench.py`: the day-0 gate table of ARCHITECTURE 10.1 (each probe prints `PASS`/`FAIL`/`INFO` + the number, first row `import lerobot.policies`); writes `results/<run>/microbench.json` (`environment` = the WSL2 label sentence, s/episode GPU B=1 and CPU, 1-vs-2-worker episodes/min, chunk latency, determinism check on seeds 0..2, h15 assert, bench p99).
- `seeds.py`: `EVAL = range(0, 500)`, `CALIB = range(900000, 900300)`, `PILOT_EVAL = range(0, 60)`, `PILOT_CALIB = range(900000, 900100)`, `splitmix64(x)` and `hash64(seed, chunk) = splitmix64(((seed << 32) ^ chunk) & MASK64)` with every step reduced `& MASK64` (Python ints are unbounded) and the frozen test vectors `hash64(0,0) = 0xe220a8397b1dcdaf`, `hash64(7,3) = 0xfd323448a4497c68`, `hash64(900000,37) = 0xaf19a08d78d230a5`, `hash64(499,0) = 0xb267f5be46d03c23`; `parse_range`.
- `requirements.txt`: `lerobot[pusht,diffusion]==0.6.1`, `torch==2.7.1+cu126` (index URL comment), `opencv-python-headless` (PINNED to the venv's version; `opencv-python` MUST NOT be installed alongside it -- the cv2 circular import), `numpy`, `matplotlib`, `pytest` (exact versions filled from the venv at implementation time with `pip freeze`).

Tests: `test_seeds.py` (disjointness; the four frozen hash64 vectors; `hash64(2**40, 0)` stays below 2**64). `test_executor.py` (sync d=2 delays the chunk by 2 ticks and repeats the last action twice; async d<=7 never underruns and drops d actions; freeze stitch executes them; d=9 async underruns to hold-last; determinism across two runs). `test_harness.py` (`run.py --policy replay:... --arms obs-d0,t01-a05-d0 --seeds 0-3 --out tmp` with a mocked `LictorClient` that returns the policy action -> 8 index records; `--check-pools` prints the line; resume: delete the last index line, rerun, exactly one episode re-executed). `test_h15.py` (torch + lerobot; skipped when unavailable): the six asserts of sec 0 on seeds 0..2 -- with identical RNG state the first 8 of the 15 actions equal a stock-config `select_action`'s 8 actions bit-exactly (`torch.equal` through the same postprocessor), the manual batch equals `populate_queues`' batch at t = 0 (duplicated) and t = 8, and `predict_action_chunk` twice with the same seed gives identical chunks. `test_harness.py` additionally covers: a mocked child death mid-episode -> `lictor crash-receipt` invoked (mocked binary) -> ledger line -> respawn -> the next index runs; a mocked `fatal` error -> `episode_end(ended_by="fault")` sent.

Acceptance:
```bash
source harness/env.sh
python -m pytest harness/tests -q -k "not h15"                      # no torch needed
python -m pytest harness/tests/test_h15.py -q                        # needs the venv + checkpoint (report skip reason if it cannot run)
python harness/run.py --check-pools | grep -q "pools disjoint: ok"
python harness/microbench.py --out /tmp/mb && cat /tmp/mb/microbench.json   # report s/episode (GPU B=1 and CPU), 1-vs-2 workers, h15 PASS, determinism PASS on seeds 0..2
python harness/run.py --plan --budget-min 60 --arms calib-obs,obs-d0,t01-a05-d0 --seeds pilot   # prints the 220-episode projection; refuses if over budget
python harness/run.py --run-id smoke --arms obs-d0 --seeds 0-9 --out /tmp/smoke && python harness/run.py --run-id smoke --out /tmp/smoke --summary
```
FIRST DELIVERABLE (report it the moment you have it): measured seconds/episode on GPU and CPU, and whether `test_h15` passes.

---

## WP-9 -- Analysis, statistics, figures, experiment write-up, reproduction guide

Files:
```
harness/analyze.py   harness/stats.py   harness/figures.py   harness/tests/test_stats.py   harness/tests/test_figures.py
docs/EXPERIMENT.md   docs/REPRODUCE.md   docs/figures/.gitkeep
```
Dependencies: WP-0 (docs). Start immediately; consume ONLY `sweep.jsonl`, `curve/summary.csv`, `curve/<arm>.json`, `microbench.json` and `lictor bench --csv` output (never the harness index).

Spec:
- `stats.py`: Clopper-Pearson (scipy-free: bisection on the regularised incomplete beta via `math.lgamma` continued fraction), McNemar exact, paired bootstrap (10 000 resamples, `np.random.default_rng(20260830)`), PoNR proxy (mirrors `lictor-calib`; a test asserts equality on a shared fixture), AUCPDT, F1-timeliness hypervolume (ActProbe). These are cross-checks of the Rust numbers, not their source.
- `analyze.py`: reads a run directory, joins `summary.csv` with arm metadata, emits tidy CSVs: per-arm Y1/Y2/Y3 with CIs and BOTH deltas exactly as `lictor curve` wrote them (`delta_vs_baseline*`, `delta_vs_latency_control*` columns -- analyze.py never computes a delta itself and never reads receipts), the alpha sweep table (`tau_source == "artefact"` points only for the Layer-A-vs-B comparison), lead-time quantiles, `tce_valid_frac`, the latency table with its `latency_label`; cross-checks Rust CI/McNemar/bootstrap values against `stats.py` on the CSV numbers and prints `stats cross-check: ok` or the diffs.
- `figures.py`: F1-F9 of ARCHITECTURE 10.9 with matplotlib only (no seaborn), SVG, every point labelled with n, CI bars, `pilot == 1` points hollow and never connected by a line, the `tce_valid_frac` bar and the sync-`d >= 7` caption on F1, twin top axes on F1 with the re-scaling caption, the `latency_label` sentence on F5 and on every latency table verbatim, the PoNR proxy definition plus the `t_fail = 0` artefact and the eps sensitivity inline on F3, `--from-fixture` mode that renders every figure from committed sample CSVs (in `docs/figures/sample/`? NO -- that directory is not owned; embed the sample data as Python literals in `figures.py`) so CI can render without a run.
- `docs/EXPERIMENT.md`: paper-shaped protocol: conditions, seeds, pools, delay realisation, metric definitions, statistics, figures, threats to validity (PoNR proxy, exchangeability, WSL2, single policy, simulator fidelity, chunk-spec dependence), reproducibility appendix.
- `docs/REPRODUCE.md`: exact command sequence for ARCHITECTURE 10.8 (the 220-episode pilot steps 0-5, then the overnight priority list) incl. `source harness/env.sh`, venv path, pins, seed pools, `--plan --budget-min`, expected artefacts and where they land on D:, how to verify any receipt with the stdlib verifier (`--pubkey` always), and the sentence "every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement" above every timing table; `docs/EXPERIMENT.md` carries the same sentence in its threats-to-validity section and the trust-model paragraph of ARCHITECTURE sec 8.

Tests: `test_stats.py` (known values as in WP-5; PoNR on a hand trace; bootstrap determinism). `test_figures.py` (`figures.py --from-fixture --out tmp` produces 9 SVG files each containing the required caption strings).

Acceptance:
```bash
/mnt/d/lictor/venv/bin/python -m pytest harness/tests/test_stats.py harness/tests/test_figures.py -q
/mnt/d/lictor/venv/bin/python harness/figures.py --from-fixture --out /tmp/figs && ls /tmp/figs/*.svg | wc -l   # 9
grep -L "not a real-time environment" /tmp/figs/latency-hist.svg ; test $? -eq 1
```

---

## WP-10 -- Bench vectors, alloc/determinism tests, `lictor bench`, demo, CI scripts, fixtures

Files:
```
bench/README.md   bench/run_all.sh   bench/fixtures/gen_synthetic.py
bench/latency/run_experiment.sh   bench/determinism/run_experiment.sh   bench/tamper/run_experiment.sh
bench/workspace-breach/run_experiment.sh   bench/brake/run_experiment.sh   bench/failure-prediction/run_experiment.sh
bench/fixtures/traces/pusht_000007.ndjson   bench/fixtures/traces/breach.ndjson   bench/fixtures/traces/brake.ndjson
bench/fixtures/traces/failpred/  (40 synthetic episodes: 20 success, 20 fail, labelled in bench/fixtures/traces/failpred/labels.json)
bench/fixtures/chunks/pusht_chunks.ndjson   bench/fixtures/calibration.a05.json   bench/fixtures/key.hex   bench/fixtures/envelope.toml
crates/lictor-cli/src/cmd/bench.rs   crates/lictor-cli/src/alloc_count.rs
crates/lictor-fuse/tests/alloc.rs   crates/lictor-fuse/tests/determinism.rs
scripts/demo.sh   scripts/ci_python.sh
```
Dependencies: WP-0 (compile). Test-level: WP-3, WP-6, WP-7, WP-8 (`run.py --policy replay:`).

Spec:
- `gen_synthetic.py` (numpy only): generates every fixture deterministically (seed 20260830) as VALID `lictor-wire/v1` request traces with the FakePushT PD dynamics: `pusht_000007.ndjson` (300 ticks, smooth trajectory, one Tier-1 stall from tick 200), `breach.ndjson` (setpoints leaving the box in 137 ticks), `brake.ndjson` (a committed prefix that cannot stop before the wall), `failpred/*.ndjson` with labels, `pusht_chunks.ndjson` for the replay policy; `calibration.a05.json` is produced by running `lictor calibrate` over the failpred successes (the script shells out to the binary when given `--bin`). These synthetic fixtures are replaced by real recorded ones in WP-13; the file names do not change.
- `alloc_count.rs` (lictor-cli; the ONLY unsafe in the binary, `#[allow(unsafe_code)]` applied at the `mod` declaration in `main.rs`): `struct Counting; unsafe impl GlobalAlloc for Counting` wrapping `System`, counting `alloc`/`dealloc` into `AtomicU64`s ONLY while `COUNTING: AtomicBool` is set; `#[global_allocator] static A: Counting = Counting;` always installed (a global allocator cannot be installed at run time); `pub fn start()`, `pub fn stop() -> (u64, u64)`. Cost while not counting: one relaxed load per allocation, off the measured path anyway.
- `bench.rs`: replay `--trace` (default: the embedded `pusht_000007.ndjson` via `include_str!`) through `lictor_runtime::session::Staging` + `decide()` `n` times in a loop with `Instant` OUTSIDE the call; separate histograms for total, and per-tier by calling `check_chunk`, `brake_feasible`, `features` directly on the same inputs; the `allocations` line = `alloc_count::start()` before the loop, `stop()` after (must be 0/0); the KiB figure = `size_of::<FuseRt>() + size_of::<FuseConfig>()` printed at run time; print the frozen lines (the numbers are `X` placeholders in the freeze; CI greps the shape); `--csv` bucket dump.
- `alloc.rs` (lictor-fuse integration test; `#![allow(unsafe_code)]` with a header comment -- outside the library's `forbid`): its own `#[global_allocator]` counting allocator (a test crate root may have one), builds `TickInput`s through `lictor_runtime::session::Staging` from the embedded trace (dev-dependency on lictor-runtime, pre-declared by WP-0); warm up 100 ticks, then 10 000 ticks of `decide()`; assert 0 allocs / 0 deallocs; print `decide: 10000 ticks, 0 allocations, 0 deallocations`.
- `determinism.rs`: 40 replays of the trace through fresh `Fuse`s (same `Staging` path) -> all verdict-stream digests (sha256 over the canonical tick events via lictor-canon) equal; plus `cargo test --profile bench-debug` equivalence documented via a `#[test]` that prints the digest for the script to compare between `--release` and debug runs.
- Bench vectors (each `run_experiment.sh BIN` prints ONE fixed result line; assertions read out of the SIGNED RECEIPTS via `python3` heredocs, never stdout): `latency` (`lictor bench --n 200000`; asserts `allocations   0` and p99 <= 50 us); `determinism` (`replay --repeat 40` -> `replays 40/40 byte-identical`, `RESULT  DETERMINISTIC`; plus the release-vs-debug digest equality: `cross-build: verdict chain identical`); `tamper` (serve the fixture trace with `bench/fixtures/key.hex` to produce a receipt in a tempdir, then: flip `counts.ticks` (one byte of the SIGNED BODY) -> `signature      FAIL` and `body digest    FAIL`; edit tick 17 and run `verify --ticks` -> `verdict chain  BROKEN at seq=17`; edit tick 17 AND re-chain 17..299 -> `verdict chain  HEAD MISMATCH`; re-sign the receipt with a fresh key -> `pubkey         MISMATCH` under `--pubkey`; delete ledger entry 3 -> `ledger BROKEN at seq=3`; an observe receipt -> `intact         YES` + `FUSE NOT ENFORCED`; stdlib verifier with `--pubkey` -> `OK` then `FAIL: ...` for each of the above; the script's comment states what is NOT detected: a whitespace edit of the pretty-printed file, and a ticks-file edit outside the embedded tail when `--ticks` is not passed); `workspace-breach` (`breach.ndjson` in observe vs enforce; from the receipts: `fuse off  violations_reached_env=137  FUSE NOT ENFORCED` / `fuse on   violations_reached_env=0  FUSE HELD`); `brake` (`brake.ndjson` enforce: `holds>=1`, and a Python PD re-simulation of the executed actions never leaves the box: `excursions=0`); `failure-prediction` (calibrate on failpred successes, sweep, assert `flagged_rate` within [0.6, 1.0] and `false_trip_rate <= 0.2` -- bands committed in the script; the n = 20 curve call uses `--allow-small` and the script asserts `small_n: true` in the signed curve receipt).
- `run_all.sh BIN`: runs every vector, tees to `/tmp/lictor-bench/*.txt`, greps every frozen line, exits non-zero on the first miss, prints a summary table.
- `scripts/ci_python.sh`: `python3 -m pytest harness/tests -q -k "not h15"`, `python3 harness/run.py --check-pools`, `python3 harness/run.py --policy replay:bench/fixtures/chunks/pusht_chunks.ndjson --arms obs-d0,t01-a05-d0 --seeds 0-3 --out /tmp/ci --envelope bench/fixtures/envelope.toml --calibration-dir bench/fixtures` then `grep -q '"success":' /tmp/ci/obs-d0/index.jsonl`, `python3 harness/figures.py --from-fixture --out /tmp/ci/figs`.
- `scripts/demo.sh`: the README "See it in 30 seconds": build, `selftest`, the tampers with their output ("edit one byte of the SIGNED BODY" -> signature FAIL; "edit one tick and run verify --ticks" -> chain BROKEN; re-chain -> HEAD MISMATCH), `verify_receipt.py --pubkey <bench key pubkey>` OK/FAIL. Must finish in < 60 s on the dev box. Uses the committed bench test key and says so in its output.

Tests: the two Rust test files above; the shell vectors are their own tests.

Acceptance:
```bash
cargo test -p lictor-fuse --release --test alloc -- --nocapture | grep -q "decide: 10000 ticks, 0 allocations, 0 deallocations"
cargo test -p lictor-fuse --test determinism
cargo build --release -p lictor-cli && bash bench/run_all.sh $CARGO_TARGET_DIR/release/lictor     # every frozen grep passes
bash scripts/ci_python.sh && bash scripts/demo.sh | grep -q "SELFTEST PASS"
```

---

## WP-11 -- Adapter stubs: LeRobot ProcessorStep and openpi websocket proxy (roadmap-level, importable, clearly labelled UNVERIFIED)

Files:
```
adapters/lerobot/__init__.py   adapters/lerobot/lictor_lerobot.py
adapters/openpi/__init__.py    adapters/openpi/lictor_proxy.py
adapters/README.md
adapters/tests/test_lerobot_stub.py   adapters/tests/test_openpi_stub.py
```
Dependencies: WP-0 (docs). Start immediately.

Spec:
- Every file starts with a docstring: `STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12.`
- `lictor_lerobot.py`: `Fuse` convenience wrapper over `adapters.lictor_client.LictorClient` (`start(envelope, calibration, out, mode)`, `begin(episode, seed, policy, env)` computing bindings via `harness.compat` when importable else from explicit kwargs, `tick(state, action, chunk=None, ext=None)`, `end(success, steps, progress)`), and `LictorStep`, a lerobot v0.6 `ProcessorStep`-shaped class (`__call__(transition)`, `feature_contract`, `reset`, `state_dict`/`load_state_dict`) that imports `lerobot.processor` lazily and falls back to a duck-typed base when lerobot is absent; it maps the env_postprocessor transition's action through the fuse and replaces it with `verdict.action`. The 10-line adoption story from the README lives in its docstring.
- `lictor_proxy.py`: an asyncio websocket proxy skeleton for openpi's `WebsocketPolicyServer` wire format (msgpack-numpy frames; observation dict in, `{"actions": (H, D)}` out): `--listen 0.0.0.0:8001 --upstream ws://host:8000 --envelope ... --calibration ...`; per client: forward observations upstream, lower the returned chunk to the lictor chunk IR (`horizon=H`, `exec_steps` from `--exec-steps`), run ONE `tick` per action through `LictorClient` as the client's execution deque advances (proxy-side executor gate, documented as the simplification), substitute the fuse's action chunk, never drop the connection on a fault (return a hold chunk + an error frame), and log a receipt per episode boundary (`--episode-boundary reset-key`). `websockets` and `msgpack` imported lazily; a `--dry-run` mode that round-trips a synthetic observation through the codec without a network.
- `adapters/README.md`: what is verified (`lictor_client.py`, `verify_receipt.py`) vs UNVERIFIED (both stubs), how to try each, the exact upstream extension points (LeRobot `env_postprocessor`, openpi `serve_policy.py` port), and the CVE-2026-25874 note (pickle over gRPC in LeRobot async; the Rust proxy path is roadmap).

Tests: `test_lerobot_stub.py` (imports without lerobot; `LictorStep` duck-typed with a fake transition and a mocked `LictorClient` replaces the action). `test_openpi_stub.py` (imports without websockets; codec round trip in `--dry-run` with a fake msgpack encoder when `msgpack` is absent).

Acceptance:
```bash
/mnt/d/lictor/venv/bin/python -m pytest adapters/tests/test_lerobot_stub.py adapters/tests/test_openpi_stub.py -q
python3 -c "import adapters.lerobot.lictor_lerobot, adapters.openpi.lictor_proxy"     # system python, no lerobot/openpi
grep -l "UNVERIFIED" adapters/lerobot/lictor_lerobot.py adapters/openpi/lictor_proxy.py adapters/README.md | wc -l   # 3
```

---

## WP-12 -- README (house style), verdict schema, envelope schema, SECURITY, banner, research index

Files:
```
README.md   SECURITY.md   docs/verdict-schema.md   docs/envelope-schema.md   docs/banner.svg   research/README.md   CONTRIBUTING.md
```
Dependencies: WP-0 (docs). Start immediately; fill measured numbers (badge, demo output, curve figure) in WP-13.

Spec:
- `README.md` in the exact bulla/sbx house style: centred `docs/banner.svg`; shields.io badge row (Rust 2021, CI, clippy+fmt clean, `verdict p99 <X> us (WSL2)` reserved and filled from `lictor bench` in WP-13, zero-alloc, MIT); one-line thesis; links row (ARCHITECTURE / EXPERIMENT / REPRODUCE / research); **The hole** anchored on the THIRD-PARTY numbers (pi0.5 43.7 % RoboChallenge; Penn pi0-FAST-DROID 42.3 % over 300+ trials; RoboArena) with GEN-1.5 quoted only in the ANALYSIS sec 2.1 recommended sentence with its self-reported/unverified caveat; LeRobot's official safety story being a Python `torch.clamp`; **What it is** (two planes; the three pillars; receipts; the trust-model sentence: "receipts bind the host's declarations and defend against edits by anyone without the key"); **See it in 30 seconds** (`scripts/demo.sh` with REAL pasted output; the headline wording is exactly "edit one byte of the signed body and the signature fails; edit one tick in the 300-line ticks file and `verify --ticks` reports the break; re-chain it and the head no longer matches"); **Results** embedding `docs/figures/curve-safety-latency.svg` and `latency-hist.svg` with their captions (WSL2 label, `tce_valid_frac` note, hollow pilot points); **The honest boundary** reproducing ARCHITECTURE sec 11 IN FULL; **Determinism** (the `replays 40/40 byte-identical` block and the "timing chain head varies (by design)" line, with sbx's "determinism is the precondition, not decoration", worded as evidence on the tested target, never "proof"); **Roadmap** = ARCHITECTURE sec 12; **Authorship** crediting Claude (Claude Fable 5.1 via Claude Code) for design and implementation with the author's direction, and bulla/sbx lineage; MIT.
- `SECURITY.md`: the trust model of ARCHITECTURE sec 8 verbatim (what the fuse verifies vs what the host declares; receipts defend against edits by anyone without the key and against corruption, not against the experimenter; anchoring is roadmap); key custody (`$LICTOR_KEYS`/`$HOME/.lictor`, never the repo, never `/mnt/[a-z]/`; ephemeral keys mark `fuse_ok = false`); the two COMMITTED TEST KEYS listed by pubkey (`crates/lictor-receipt/tests/fixtures/receipt/key.hex`, `bench/fixtures/key.hex`) with the statement that `lictor verify` warns on them; operator keys and what a forged ack would need; ack replay: bound to run/arm/episode and to the per-operator nonce, replayable across processes only without `verifier_nonce.json`; the declared-pool truncation check and its limit; no network surface in milestone 1; the openpi/LeRobot proxies are unverified; reporting contact; key rotation (`lictor key init --force`); which claims are NOT made (the full banned list of rule 4 as not-claims).
- `docs/verdict-schema.md` (sbx `verdict-schema.md` twin): `Status` lattice and ordering, `FuseState`, every `TripMask` bit, `ActionSource`, every `ReasonCode` with its `reason_text`, the verdict JSON on the wire and in the ticks file, and the diff/streak semantics of `history`.
- `docs/envelope-schema.md`: every envelope field, units, defaults, validation rules, the digest rule, and the fit procedure summary (pointing at `docs/calibration.md` for the CP side).
- `docs/banner.svg`: hand-written SVG (no external fonts; text as paths not required) in the bulla banner style; ASCII-safe.
- `research/README.md`: index of the memos behind every number cited in the README with their URLs from `docs/ANALYSIS.md` sec 9, plus the UNVERIFIED list.
- `CONTRIBUTING.md`: the house rules (fmt/clippy/fixtures/banned vocabulary/CARGO_TARGET_DIR).

Tests: none in code; acceptance is textual.

Acceptance:
```bash
! grep -rniE "safety-rated|hard[ -]real[ -]time|PL d|SIL 2|\bcertified\b|engineered to|principles of|[0-9]{4,5} principles|certified limits|determinism proof|every conforming platform|can never hide" README.md SECURITY.md CONTRIBUTING.md docs/*.md research/*.md harness/*.py adapters/*.py crates | grep -viE "never|not claim|banned|does not|no claim|not-claim|borrowed for readability"
grep -c "honest boundary" README.md   # >= 1
python3 -c "import xml.dom.minidom,sys; xml.dom.minidom.parse('docs/banner.svg')"
```

---

## WP-13 -- INTEGRATION (last; may touch any file to fix cross-package breakage; records every fix)

Files: the GENERATED-IN-REPO outputs -- `envelopes/pusht.toml`, `envelopes/pusht.oracle.toml`, `docs/envelope_fit_report.md`, `docs/figures/*.svg`, `research/*.md` (memos behind cited numbers, one per number) -- plus fixes anywhere, with a `git log` message per fix naming the package it corrects.
Dependencies: WP-1 .. WP-12.

Checklist (all must pass, in order):
```bash
source harness/env.sh
cargo fmt --all --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace && make nostd-check
cargo build --release -p lictor-cli && $CARGO_TARGET_DIR/release/lictor selftest
bash bench/run_all.sh $CARGO_TARGET_DIR/release/lictor && bash scripts/ci_python.sh && bash scripts/demo.sh
# un-ignore WP-1's digest test; regenerate any fixture whose generator changed; re-run
python harness/microbench.py --out $LICTOR_RESULTS/mb                                   # day-0 gate PASS lines
python harness/run.py --run-id smoke20 --arms obs-d0,t01-a05-d0 --seeds 0-9 --out $LICTOR_RESULTS --envelope envelopes/pusht.base.toml --calibration-dir bench/fixtures   # 20 real PushT episodes through the real fuse
$CARGO_TARGET_DIR/release/lictor ledger verify $LICTOR_RESULTS/smoke20/obs-d0/ledger.jsonl
$CARGO_TARGET_DIR/release/lictor curve --run $LICTOR_RESULTS/smoke20 --baseline obs-d0 --arms t01-a05-d0 --allow-small -o $LICTOR_RESULTS/smoke20/curve    # n=10 -> small_n=true, pilot=1
python3 adapters/verify_receipt.py $LICTOR_RESULTS/smoke20/obs-d0/receipts/000000.json --ticks $LICTOR_RESULTS/smoke20/obs-d0/ticks/000000.jsonl --pubkey $($CARGO_TARGET_DIR/release/lictor key pub)
$CARGO_TARGET_DIR/release/lictor replay --repeat 40 $LICTOR_RESULTS/smoke20/obs-d0/traces/000000.ndjson --envelope envelopes/pusht.base.toml
```
Then: copy the real `traces/000007.ndjson` from the smoke run over `bench/fixtures/traces/pusht_000007.ndjson` (and regenerate `pusht_chunks.ndjson` from it), re-run `bench/run_all.sh`; fill the README badge/demo output/`lictor bench` numbers (WSL2-labelled); after the pilot's `envelope fit`, commit `envelopes/pusht.toml`, `envelopes/pusht.oracle.toml` (`--operator $(lictor key pub --key $LICTOR_KEYS/operator.hex)`) and `docs/envelope_fit_report.md`; report the measured s/episode (B=1), the 1-vs-2-worker result, `decide_ns` p50/p99, and whether the observe-identity and h15 asserts passed. Hand the orchestrator the exact commands for the 220-episode pilot (ARCHITECTURE 10.8 steps 0-5) with `--plan --budget-min 60` output attached.

---

# PART D -- OWNERSHIP TABLE (every path exactly once)

| Path (repo-relative) | Owner |
|---|---|
| `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `rustfmt.toml`, `.cargo/config.toml`, `Makefile`, `pyproject.toml`, `.gitignore`, `LICENSE`, `.github/workflows/ci.yml` | WP-0 |
| `crates/*/Cargo.toml` (8), `crates/*/src/lib.rs` (7), `crates/lictor-cli/src/main.rs`, `crates/lictor-cli/src/cmd/mod.rs`, `crates/lictor-cli/build.rs` | WP-0 |
| `harness/__init__.py`, `harness/tests/__init__.py`, `adapters/__init__.py`, `adapters/tests/__init__.py` | WP-0 |
| `docs/ARCHITECTURE.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/ANALYSIS.md` | WP-0 |
| `crates/lictor-core/src/{fmath,chunk,envelope,scores,verdict,state,reason,ack}.rs`, `crates/lictor-core/tests/**`, `crates/lictor-detect/src/brake.rs`, `crates/lictor-detect/tests/brake.rs`, `crates/lictor-detect/tests/fixtures/brake/**`, `envelopes/pusht.base.toml` | WP-1 |
| `crates/lictor-detect/src/{tier0,window,tier1,conformal}.rs`, `crates/lictor-detect/tests/{tier0,window,tier1,conformal}.rs`, `crates/lictor-detect/tests/fixtures/{tier0,tier1,conformal}/**` | WP-2 |
| `crates/lictor-fuse/src/{fuse,fsm,tally}.rs`, `crates/lictor-fuse/tests/{state_machine,observe_passthrough,decide_order}.rs`, `crates/lictor-fuse/tests/common/mod.rs`, `crates/lictor-fuse/tests/fixtures/**` | WP-3 |
| `crates/lictor-canon/src/{jcs,f64enc,digest}.rs`, `crates/lictor-canon/tests/**`, `crates/lictor-receipt/src/**` (except `lib.rs`), `crates/lictor-receipt/tests/**`, `adapters/verify_receipt.py`, `adapters/tests/test_verify.py`, `docs/receipt-schema.md` | WP-4 |
| `crates/lictor-calib/src/**` (except `lib.rs`), `crates/lictor-calib/tests/**`, `crates/lictor-cli/src/cmd/{calibrate,sweep,curve,envelope}.rs`, `docs/calibration.md` | WP-5 |
| `crates/lictor-runtime/src/**` (except `lib.rs`), `crates/lictor-runtime/tests/**`, `crates/lictor-cli/src/cmd/serve.rs`, `crates/lictor-cli/src/cmd/crash_receipt.rs`, `adapters/lictor_client.py`, `adapters/tests/test_client.py`, `docs/wire-protocol.md` | WP-6 |
| `crates/lictor-cli/src/cmd/{verify,replay,selftest,ledger,key,ack,history,version}.rs`, `crates/lictor-cli/src/render.rs`, `crates/lictor-cli/tests/**` | WP-7 |
| `harness/{compat,seeds,pusht_rollout,executor,inject,policy_replay,arms,run,microbench}.py`, `harness/env.sh`, `harness/requirements.txt`, `harness/README.md`, `harness/tests/{test_h15,test_executor,test_harness,test_seeds}.py` | WP-8 |
| `harness/{analyze,stats,figures}.py`, `harness/tests/{test_stats,test_figures}.py`, `docs/EXPERIMENT.md`, `docs/REPRODUCE.md`, `docs/figures/.gitkeep` | WP-9 |
| `bench/**` (all), `crates/lictor-cli/src/cmd/bench.rs`, `crates/lictor-cli/src/alloc_count.rs`, `crates/lictor-fuse/tests/{alloc,determinism}.rs`, `scripts/demo.sh`, `scripts/ci_python.sh` | WP-10 |
| `adapters/lerobot/**`, `adapters/openpi/**`, `adapters/README.md`, `adapters/tests/{test_lerobot_stub,test_openpi_stub}.py` | WP-11 |
| `README.md`, `SECURITY.md`, `CONTRIBUTING.md`, `docs/verdict-schema.md`, `docs/envelope-schema.md`, `docs/banner.svg`, `research/README.md` | WP-12 |
| `envelopes/pusht.toml`, `envelopes/pusht.oracle.toml`, `docs/envelope_fit_report.md`, `docs/figures/*.svg` (not `.gitkeep`), `research/*.md` except `research/README.md` (generated outputs); plus fixes anywhere, logged | WP-13 |

Disjointness check (run by WP-0 after writing this table into the repo and by WP-13 at the end): every path listed under exactly one owner. Explicit exceptions to the `src/**` globs: every `crates/*/src/lib.rs` is WP-0's (the WP-4/5/6 globs read "except lib.rs"); `crates/lictor-cli/src/` is split file-by-file: `main.rs`, `cmd/mod.rs`, `build.rs` (WP-0), `alloc_count.rs`, `cmd/bench.rs` (WP-10), `render.rs` + the eight WP-7 `cmd/*.rs`, the four WP-5 `cmd/*.rs`, `cmd/serve.rs` + `cmd/crash_receipt.rs` (WP-6). STUB files (B.1) are written once by WP-0 and owned thereafter by the named package -- `envelopes/pusht.base.toml` and `crates/lictor-detect/tests/fixtures/brake/cases.json` by WP-1, `crates/lictor-detect/tests/fixtures/tier1/cases.json` by WP-2, `alloc_count.rs` by WP-10. `crates/lictor-fuse/tests/` is split by file name between WP-3 (`state_machine`, `observe_passthrough`, `decide_order`, `common/`, `fixtures/`) and WP-10 (`alloc`, `determinism`); `crates/lictor-detect/tests/` between WP-1 (`brake.rs`, `fixtures/brake/`) and WP-2 (the rest); `harness/` between WP-0 (`__init__.py`s), WP-8 and WP-9 as listed; `adapters/` between WP-0 (`__init__.py`s), WP-4, WP-6 and WP-11; `docs/` between WP-0 (`ARCHITECTURE`, `IMPLEMENTATION_PLAN`, `ANALYSIS`), WP-4, WP-5, WP-6, WP-9, WP-12 and WP-13 (`envelope_fit_report.md`, `figures/*.svg`); `envelopes/` between WP-1 (`pusht.base.toml`) and WP-13 (the two fitted files); `research/` between WP-12 (`README.md`) and WP-13 (memos).

# PART E -- DEPENDENCY GRAPH AND WAVES

```
WP-0 (alone, ~60 min)
  |-- WP-1 core+brake ----------+
  |-- WP-2 detect --------------+--> WP-3 fuse (tests) --+
  |-- WP-4 canon+receipt+pyver -+--> WP-6 runtime+serve --+--> WP-7 cli-rest --+
  |-- WP-5 calib (tests: 1,2,4,6)                          |                    +--> WP-10 bench/CI (tests: 3,6,7,8) --+
  |-- WP-8 harness (live: 6) ---+--------------------------+                                                          +--> WP-13 INTEGRATION
  |-- WP-9 analysis/figures ----------------------------------------------------------------------------------------+
  |-- WP-11 adapter stubs --------------------------------------------------------------------------------------------+
  +-- WP-12 docs/README ----------------------------------------------------------------------------------------------+
```
All twelve packages start coding at T+60 against the freeze; the arrows are TEST-level dependencies (when a package's acceptance can first be fully green). Expected critical path: WP-0 -> WP-1/2 -> WP-3 -> WP-6 -> WP-7 -> WP-10 -> WP-13.

# PART F -- THE SIX RISKIEST ITEMS (watch these first)

1. **`n_action_steps = 15` on lerobot 0.6.1 + the pre-migration checkpoint** (WP-8 `test_h15`, day-0 gate). If the queue/slicing behaviour differs from the main-branch source, TCE/ACC become unavailable; fallback is `conditional_sample()` directly. Never touch `num_inference_steps`.
2. **Seconds per episode on the GTX 1060** (unverified; ~3 700 launch-bound UNet forwards per episode at B = 1; the honest expectation is 12-20 s/ep, NOT the batched 1.46 s reference; CPU is unusable for arms). The pilot is therefore 220 episodes and `run.py --plan --budget-min` refuses over-budget steps; the whole schedule scales off WP-8's first measurement; Layer A survives regardless because it needs one observe pass. Batching across envs is off the table (per-episode seeding).
3. **Envelope fit vs parity gate** (WP-5 + harness): too tight destroys the baseline, too loose makes Tier 0 decorative; the injection arms carry the enforcement demonstration either way.
4. **Cross-package determinism** (WP-2/3/4/6): an accidental `mul_add`, a reordered sum, a `HashMap`, or a serde float on a hashed path silently breaks byte-identical replay and the Python parity. `float_repro`-style fixtures, the deny-list grep, `replay --repeat 40` and the stdlib verifier in CI are the tripwires.
5. **Torch determinism on Pascal / paired seeds**: if `obs-d0` twice on seeds 0..2 diverges, McNemar is invalid; CPU inference for headline arms is the documented fallback (and then the pilot shrinks further), and `init_state_digest` + `pair_mismatches` + `cross_run_mismatches` make any residual drift visible rather than silent.
6. **The venv and the checkpoint**: `import lerobot.policies` is the first day-0 row (the cv2 circular import from a stray `opencv-python` killed it once); the migrated checkpoint lives at `/mnt/d/lictor/models/diffusion_pusht_migrated` and `compat.py` names the migration script explicitly. Both are on WP-8's critical path before any s/episode number exists.

---

# PART G -- REVIEW RESPONSES (what changed after the two reviews, and the two items rebutted)

Every BLOCKING item was fixed in the freeze unless listed under "rebutted"; IMPORTANT items were fixed where cheap; the nits we agreed with are folded in. Item numbers refer to the reviewers' lists.

**Critic 1, blocking -- all fixed.** (1) `FuseState` derives `Default` (`#[default] Idle`); `Tally` has a hand-written `Default` (also fixing the `max_s = 0.0` bias); `VerdictCounts` keeps `derive(Default)`. (2) Frozen `impl Default` beside every frozen `new()` and `is_empty()` beside every `len()`; no lint tables. (3) `main.rs` is `deny(unsafe_code)` + `#[allow(unsafe_code)] mod alloc_count;`, the allocator is always installed and gated by an `AtomicBool`; `alloc_count.rs` is WP-0-stubbed and WP-10-owned. (4) `decide()` never writes `t1.prev`; `features` is the sole writer (raw-vs-raw), stated in 5.3, 5.4, WP-2 and WP-3. (5) `sign_curve`/`verify_curve`, `Session::note_io_ns`, `Staging`, and every wire payload struct are frozen verbatim; lictor-fuse gets a pre-declared dev-dependency on lictor-runtime. (6) WP-0 writes `envelopes/pusht.base.toml` byte-for-byte and `[]` placeholder fixtures, so every `include_str!` resolves at minute one; selftest reports `[skip]` on empty fixtures. (7) `libm` is an unconditional dependency; detect/fuse depend on core with `default-features = false`; `make nostd-check` is `--no-default-features`. (8) `null == +-inf` for `scores.s`, `tau`, `hello_ok.calibration.tau` is frozen on both sides with a round-trip test. (9) `provides_vel = true`; the harness sends `info["vel_agent"]`; the finite difference is a recorded fallback; the "exact" claim is conditioned on the simulator `v0`. (10) The pilot is 220 episodes (calib-obs x100, obs-d0 x60, t01-a05-d0 x60); `--plan --budget-min` refuses over-budget steps; batching is declared incompatible with per-episode seeding.

**Critic 1, important -- fixed:** the six h15 runtime asserts and the offline `predict_action_chunk` path (sec 0, WP-8); `import lerobot.policies` day-0 row, `opencv-python-headless` pin, the migration script named; bindings from `importlib.metadata` with `<from importlib.metadata>` placeholders; `coverage` at t = 0 from `_get_coverage()`; the calib -> runtime dependency inverted (`CalibrationLoaded`; calib depends on runtime for the ONE wire parser); `Cargo.lock`, `pyproject.toml`, `docs/ANALYSIS.md` owned by WP-0, generated outputs by WP-13, `lib.rs` exceptions written into the disjointness note, all dev-deps pre-declared; `deny_unknown_fields` on every payload struct with the nested cases in `bad_lines.ndjson`; `max_s`/`max_z` start at `NEG_INFINITY`; FSM row 17b; `HandoffRecord.state_digest` dropped (`chain_at` defined), `arm_config_digest` defined as a named canonical object, `SeedPool`/`Tier0Percentiles` named; `LictorFault`, `chunk_msg`, `hash64` (mod 2^64, four test vectors) frozen; `FuseRt` ~58 KB and `bench` prints `size_of`; deny-list scope (`f64::min/max`, `as` from f64; calib/receipt may use ln/exp); `ed25519-dalek` without `rand_core`, `build.rs` rerun-if-changed; the 1-vs-2-worker probe. **Nits taken:** tree synced (CONTRIBUTING, .gitignore, pyproject), `use` merged, `null` documented for the two tau sites, `n <= 63`, `horizon_ticks` added to the manifest (and checked against `max_episode_steps`), `latency_label` as a `SessionConfig` field with WSL2 detection, key-regex assert in the Python verifier, bench numbers are `X`, `cfg(unix)` explicit.

**Critic 2, blocking -- all fixed.** (1) The IEC/ISO phrase is deleted everywhere; sec 11.5 and rule 4 carry the borrowed-ideas sentence and the extended banned list; WP-12's grep covers `hard[ -]real[ -]time`, `engineered to`, `principles`, `certified limits`, `determinism proof`, `every conforming platform`, `can never hide`. (2) A trust-model paragraph is in the WIRE section, sec 8 and SECURITY.md; the three overclaims are reworded to "binds the host's declaration of". (3) Wire amendment: `t_emit <= t`, `idx == t - t_emit`, full 15 rows always, drop/freeze/sync labelling defined, `brake_feasible(from = idx)`, and the sync `L = 7 - d` confound disclosed with `tce_valid_frac` in the curve receipt and on F1. (4) `speed_peak` and `stall` are redefined on `norm_scale_iso`/`dt` (manifest only), `embodiment_digest()` is frozen and bound into calibration.json, `serve`/`replay` refuse a mismatch at startup, `episode_begin` refuses `weights_sha256 != policy_digest`; the oracle arm gets `envelopes/pusht.oracle.toml` from `envelope fit --operator` (no run-time TOML edits by Python). (5) `episode_end` is accepted while faulted; the host must send it after any fatal error; a dead child yields a signed `lictor crash-receipt` (`fuse_crash`, success=false) and one respawn; `lictor curve` refuses index-set mismatches unless `--partial`, binds `n_missing`/`missing_indices`, and counts missing as failures. (6) Tamper demo restated precisely (body byte -> signature AND digest FAIL; tick edit needs `--ticks`; re-chain -> frozen `HEAD MISMATCH` line; whitespace -> nothing, stated); `--pubkey` in `verify_receipt.py` (required in demo/bench/CI), `pubkey_ok` in `VerifyReport`, `TEST_PUBKEYS` warning. (7) `n_calib` is the number of scores tau was taken over (137 under 2-way), `n_total`/`n_scale`/`split` added, worked numbers redone for 137, 59 and the pilot's ~45; the guarantee is stated as approximate under 2-way and exact under `--split 3`; `sweep --calibration-dir` reuses the same artefacts (`tau_source`). (8) `episode_begin.inputs` (host) + `lictor:*` keys (fuse); `note_io_ns` frozen; `DeltaCi` with both deltas, bootstrap CIs and McNemar in `CurveMetrics`, matching CSV columns, `--latency-control`.

**Critic 2, important -- fixed:** host fallback is always `clamp_box(current position)`; the client kills the child and the harness respawns and re-hellos; time and chunk-seq continuity are GUARD checks in `decide()`; arms declare `tier1`/`alpha`, `hello` asserts them, `curve` refuses disagreement with run.json; `HandoffRecord` binds run/arm/episode, `last_nonce` persists across episodes and in `verifier_nonce.json`; keys default to `$LICTOR_KEYS`/`$HOME/.lictor`, DrvFs paths refused without `--i-know`, `--force` to overwrite, ephemeral keys -> `fuse_ok = false` and refused by `curve`, `receipt_pubkey` bound; declared-pool truncation detection; determinism probe on seeds 0..2 and `cross_run_mismatches`; every listed overclaim reworded (tested target, "expected sub-microsecond", "determinism evidence", "checked stopping condition", vocabulary disclaimers, "rounds to zero additional control steps", targets not results); `latency_label` on curve metrics, CSV, microbench.json, REPRODUCE/EXPERIMENT; `verify` recomputes the envelope digest, checks counts, ticks header, `--calibration`; `--allow-small`/`small_n`/`pilot` for n < 100; the oracle envelope file. **Nits taken:** `horizon_ticks` checked against `max_episode_steps`; `allow_nan=False`; first-tick NaN hold documented; `Fault` described as the fail-closed latch; clock-skew row; aux-NaN fail-closed stated in GUARD; control characters stripped in render; `t_fail = 0` artefact in the F3 caption; 65.4 % gate called a smoke check with the n=500 run as the reproduction; `run.json` sha256 and `client` bound; the `--json`/`intact` note beside the exit codes; roadmap 9 and 11 reworded; `key init` prints custody; test keys listed in SECURITY.md.

**Rebutted (with reasons):**

- *Critic 2, "3-way split OR reword" -- we did both rather than making 3-way the default.* A 3-way split at 196 successes leaves 59 scores for tau, which makes alpha = 1/100 degenerate and widens every threshold; the pilot's 100 calibration episodes would leave ~20. The default therefore stays 2-way with the bound stated as approximate and `holdout_fpr_k1` as the empirical check; `--split 3` is available and its degenerate alphas are documented, and the overnight run recalibrates both ways so F7 shows the difference.
- *Critic 2, "run.py writes a host-side ledger entry signed with the harness's own key".* Rejected in that form: it would put a second signing implementation (pure-Python Ed25519 signing, a second key, a second pubkey in run.json) on the trust path. The equivalent accounting is done by the lictor binary itself (`lictor crash-receipt` -> `write_crash_episode`), signed with the same key and marked in `fuse_notes` as host-written; `lictor curve` sees exactly one pubkey per arm, which is also what lets it refuse mixed keys.
- *Critic 1, "make `[workspace.lints.clippy]` allow the two lints" (offered as an alternative).* Not taken; the freeze carries `Default`/`is_empty` instead so the public surface is idiomatic and nothing is suppressed.
- *Critic 1, "or drop the `allocations 0` line from `lictor bench`" (offered as an alternative).* Not taken; the gated always-installed allocator keeps the line and costs one relaxed atomic load per allocation off the measured path.
