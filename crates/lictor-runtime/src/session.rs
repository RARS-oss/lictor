// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; replace the bodies (and add the private fields), keep the signatures.
//! One fuse, one episode at a time; shared by serve, replay, bench, selftest, and the fuse crate's alloc/determinism tests.

use lictor_core::{CalibrationC, FuseConfig, FuseMode, SafetyEnvelope, VerifiedAck};
use lictor_fuse::TickInput;

use crate::latency;
use crate::wire::{Request, Response, TickReq};

/// A calibration ALREADY loaded and compiled by the caller (lictor-cli via lictor-calib). lictor-runtime does NOT depend on
/// lictor-calib (lictor-calib depends on lictor-runtime for the wire/trace types).
pub struct CalibrationLoaded {
    pub c: CalibrationC,
    pub digest: String,
    pub embodiment_digest: String,
    pub file_sha256: String,
    pub path: String,
}

pub struct SessionConfig {
    pub envelope: SafetyEnvelope,
    pub envelope_toml_sha: String,
    pub calibration: Option<CalibrationLoaded>,
    pub mode: FuseMode,
    pub tier0_override: Option<u32>,
    pub tier1: bool,
    pub ticks_policy: String,
    pub key: Option<[u8; 32]>,
    pub out_dir: Option<std::path::PathBuf>,
    pub trace: Option<std::path::PathBuf>,
    pub lictor_git: String,
    pub lictor_sha256: String,
    /// default: `default_latency_label()`
    pub latency_label: String,
}

/// Off-path staging: converts a TickReq into a TickInput with buffers allocated ONCE (null -> NaN; chunk rows -> ChunkBuf).
/// Used by Session and by bench/alloc/determinism tests, so there is exactly ONE wire -> TickInput conversion in the workspace.
/// Private fields (pos/vel/aux/ext: [f64; MAX_*] + lengths, have_vel, chunk: ChunkBuf, have_chunk) are WP-6's to add.
pub struct Staging {}

impl Staging {
    pub fn new() -> Self {
        todo!("WP-6")
    }

    /// Copies the request into the buffers. Err(message) on a dimension / horizon / idx / seq violation (the caller raises SCHEMA).
    pub fn stage(&mut self, _cfg: &FuseConfig, _req: &TickReq) -> Result<(), String> {
        todo!("WP-6")
    }

    /// Borrow the staged buffers as the pure input. `schema_fault` forces TripMask::SCHEMA in GUARD.
    pub fn input(&self, _req: &TickReq, _ack: Option<VerifiedAck>, _schema_fault: bool) -> TickInput<'_> {
        todo!("WP-6")
    }
}

impl Default for Staging {
    fn default() -> Self {
        Self::new()
    }
}

/// Fuse + Staging + chains + timing + episode assembly + pending handoff + last_nonce (kept ACROSS episodes).
/// Private fields are WP-6's to add.
pub struct Session {}

impl Session {
    /// Refuses (Err) a calibration whose `embodiment_digest` != `cfg.envelope.embodiment_digest()` -- fatal at startup, never a note.
    pub fn new(_cfg: SessionConfig) -> anyhow::Result<Self> {
        todo!("WP-6")
    }

    /// Handle one request; measures decide_ns around `decide()` and folds the chains AFTER it returns (never inside).
    /// `episode_end` is ACCEPTED while faulted (writes a receipt with terminal_state = fault, fuse_ok = false, ended_by as sent).
    pub fn handle(&mut self, _req: Request) -> Response {
        todo!("WP-6")
    }

    /// Called by the serve loop after the response is flushed: wall-clock from request-line read to response flush, recorded into
    /// the timing chain entry of tick `seq` (the entry is emitted on the NEXT request or at episode_end; never inside decide()).
    pub fn note_io_ns(&mut self, _seq: u32, _ns: u64) {
        todo!("WP-6")
    }

    pub fn fault_latched(&self) -> bool {
        todo!("WP-6")
    }

    pub fn verdict_chain_head(&self) -> &str {
        todo!("WP-6")
    }

    pub fn latency(&self) -> &latency::Hist {
        todo!("WP-6")
    }
}

/// WSL2_LABEL when /proc/version contains "microsoft" or "WSL", else "measured on <uname -sr>, non-RT kernel -- not a real-time environment".
pub fn default_latency_label() -> String {
    todo!("WP-6")
}
