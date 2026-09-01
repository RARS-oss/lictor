// SPDX-License-Identifier: MIT
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! lictor-core: frozen types for the deterministic safety fuse. No logic beyond
//! constructors, validation and trivially-derived accessors. See docs/ARCHITECTURE.md.
//! DECISION-PATH DENY LIST (also enforced by review): f32, mul_add, powi, powf, exp, ln,
//! sin, cos, atan2, hypot, HashMap iteration, rayon, SIMD reductions, Instant, SystemTime, RNG.
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod ack;
pub mod chunk;
pub mod envelope;
pub mod fmath;
pub mod reason;
pub mod scores;
pub mod state;
pub mod verdict;

pub use ack::{AckDecision, VerifiedAck};
pub use chunk::{
    ActionKind, ChunkBuf, ChunkError, ChunkView, ObsView, MAX_AUX, MAX_D, MAX_EXT, MAX_H, MAX_POS,
};
pub use envelope::{
    BrakeKind, BrakeModel, ClampMode, ContactLimit, EmbodimentManifest, EnvelopeError, FitRecord, FuseConfig,
    GateSpec, Hysteresis, RearmPolicy, SafetyEnvelope, MAX_OPERATORS, MAX_TERMS,
};
pub use reason::{reason_text, ReasonCode};
pub use scores::{bin_of, CalMethod, CalibrationC, Feat, Scores, NFEAT, T_GRID};
pub use state::{EpisodeInit, ExecMode, FuseMode, FuseState};
pub use verdict::{ActionSource, SafetyVerdict, Status, TripMask};

pub const LICTOR_VERSION: &str = env!("CARGO_PKG_VERSION");
