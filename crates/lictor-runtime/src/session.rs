// SPDX-License-Identifier: MIT
//! One fuse, one episode at a time; shared by serve, replay, bench, selftest, and the fuse crate's alloc/determinism tests.
//!
//! The Session is the ONLY place where wire messages become fuse inputs (`Staging`) and fuse verdicts become
//! chained events, handoff records and receipts. Order per tick: id check -> ack verification -> `stage` ->
//! `Instant::now()` -> `decide()` -> `elapsed()` -> chain fold -> response. Nothing is hashed or timed inside
//! `decide()`, and the session never post-processes a verdict: a schema fault is fed INTO `decide()` as
//! `schema_fault = true` so the hold action comes out of the frozen path.
//!
//! Fault latch: any schema/protocol error replies `error{fatal:true}` and marks the episode faulted; every later
//! tick of that episode still runs `decide()` with `schema_fault = true` (GUARD raises `TripMask::SCHEMA`, the FSM
//! latches `Fault`, the hold action is returned as a normal verdict). `episode_end` is accepted while faulted.
//!
//! Error codes: `schema` (malformed line, unknown key, dimension/horizon/idx/t_emit/seq violation), `envelope`
//! (hello/episode_begin cross-check mismatch), `state` (message out of order: tick before episode_begin, a second
//! episode_begin, hello mid-episode, episode_end without an episode), `protocol` (wrong proto id, non-monotonic
//! id), `internal` (an I/O failure while writing artefacts).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use lictor_canon::{floatify, F64Array, F64Hex};
use lictor_core::{
    reason_text, AckDecision, CalibrationC, ChunkBuf, EpisodeInit, Feat, FuseConfig, FuseMode, FuseState,
    ObsView, SafetyEnvelope, SafetyVerdict, TripMask, VerifiedAck, MAX_AUX, MAX_D, MAX_EXT, MAX_H, MAX_POS,
    NFEAT,
};
use lictor_fuse::{Fuse, TickInput};
use lictor_receipt::{
    evaluate_fuse, load_nonces, save_nonces, sign, tick_event, timing_event, verify_ack, AckError,
    BudgetBinding, EpisodeOutcome, FaultBinding, HandoffRecord, ReceiptBody, RunBinding, TickEvent,
    TimingEvent, VerdictCounts, HANDOFF_SCHEMA, RECEIPT_SCHEMA, ZERO_HASH,
};

use crate::episode::{
    episode_paths, fuse_inputs, ledger_head, now_epoch, paths_display, write_episode, EpisodePaths,
};
use crate::latency::{self, Hist, WSL2_LABEL};
use crate::wire::{
    CalibrationInfo, EpisodeBeginReq, EpisodeEndReq, EpisodeOk, EpisodeReceiptMsg, ErrorMsg, HelloOk,
    HelloReq, Request, Response, ScoresMsg, TickReq, VerdictMsg, PROTO,
};

/// Where the per-operator ack nonces persist under `--out`.
pub const NONCE_FILE: &str = ".lictor/verifier_nonce.json";

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

impl SessionConfig {
    /// A configuration with every optional feature off: no calibration, no key (ephemeral), no output, no trace.
    /// `envelope_toml_sha` is the sha256 of `toml_text` (pass the file bytes the envelope was parsed from).
    pub fn minimal(envelope: SafetyEnvelope, toml_text: &str, mode: FuseMode) -> Self {
        Self {
            envelope,
            envelope_toml_sha: lictor_canon::sha256_hex(toml_text.as_bytes()),
            calibration: None,
            mode,
            tier0_override: None,
            tier1: true,
            ticks_policy: "tail32".to_string(),
            key: None,
            out_dir: None,
            trace: None,
            lictor_git: "nogit".to_string(),
            lictor_sha256: ZERO_HASH.to_string(),
            latency_label: default_latency_label(),
        }
    }
}

/// Off-path staging: converts a TickReq into a TickInput with buffers allocated ONCE (null -> NaN; chunk rows -> ChunkBuf).
/// Used by Session and by bench/alloc/determinism tests, so there is exactly ONE wire -> TickInput conversion in the workspace.
pub struct Staging {
    pos: [f64; MAX_POS],
    vel: [f64; MAX_POS],
    aux: [f64; MAX_AUX],
    ext: [f64; MAX_EXT],
    n_pos: usize,
    n_vel: usize,
    n_aux: usize,
    n_ext: usize,
    have_vel: bool,
    flat: [f64; MAX_H * MAX_D],
    chunk: ChunkBuf,
    have_chunk: bool,
    /// The seq the next delivered chunk must carry (reset when a tick with `t == 0` starts an episode). The same
    /// check lives in `decide()` GUARD against `FuseRt.next_chunk_seq`; this one turns it into a schema message.
    next_seq: u32,
}

fn nan_or(x: Option<f64>) -> f64 {
    x.unwrap_or(f64::NAN)
}

fn copy_opt(dst: &mut [f64], src: &[Option<f64>]) -> usize {
    let n = src.len().min(dst.len());
    for (d, s) in dst.iter_mut().zip(src.iter()).take(n) {
        *d = nan_or(*s);
    }
    n
}

impl Staging {
    pub fn new() -> Self {
        Self {
            pos: [0.0; MAX_POS],
            vel: [0.0; MAX_POS],
            aux: [0.0; MAX_AUX],
            ext: [0.0; MAX_EXT],
            n_pos: 0,
            n_vel: 0,
            n_aux: 0,
            n_ext: 0,
            have_vel: false,
            flat: [0.0; MAX_H * MAX_D],
            chunk: ChunkBuf::new(),
            have_chunk: false,
            next_seq: 0,
        }
    }

    /// Copies the request into the buffers. Err(message) on a dimension / horizon / idx / seq violation (the caller raises SCHEMA).
    /// The observation is copied (best effort, truncated to the buffer) BEFORE the checks so that a faulted tick
    /// still hands `decide()` the freshest position for the hold latch; `have_chunk` is false on every error.
    pub fn stage(&mut self, cfg: &FuseConfig, req: &TickReq) -> Result<(), String> {
        let o = &req.obs;
        self.n_pos = copy_opt(&mut self.pos, &o.pos);
        self.have_vel = o.vel.is_some();
        self.n_vel = match &o.vel {
            Some(v) => copy_opt(&mut self.vel, v),
            None => 0,
        };
        self.n_aux = copy_opt(&mut self.aux, &o.aux);
        self.n_ext = copy_opt(&mut self.ext, &o.ext);
        self.have_chunk = false;
        if req.t == 0 {
            self.next_seq = 0;
        }
        if o.pos.len() != cfg.pos_dim {
            return Err(format!(
                "tick.obs.pos has {} entries, expected pos_dim={}",
                o.pos.len(),
                cfg.pos_dim
            ));
        }
        if let Some(v) = &o.vel {
            if v.len() != cfg.pos_dim {
                return Err(format!(
                    "tick.obs.vel has {} entries, expected pos_dim={}",
                    v.len(),
                    cfg.pos_dim
                ));
            }
        }
        if o.aux.len() > MAX_AUX {
            return Err(format!("tick.obs.aux has {} entries, at most {MAX_AUX} allowed", o.aux.len()));
        }
        if o.ext.len() > MAX_EXT {
            return Err(format!("tick.obs.ext has {} entries, at most {MAX_EXT} allowed", o.ext.len()));
        }
        if req.idx as usize >= cfg.horizon {
            return Err(format!("tick.idx {} is not below horizon={}", req.idx, cfg.horizon));
        }
        let Some(c) = &req.chunk else {
            return Ok(());
        };
        if c.h as usize != cfg.horizon {
            return Err(format!("tick.chunk.h is {}, expected h={}", c.h, cfg.horizon));
        }
        if c.d as usize != cfg.dim {
            return Err(format!("tick.chunk.d is {}, expected action_dim={}", c.d, cfg.dim));
        }
        if c.exec as usize != cfg.exec {
            return Err(format!("tick.chunk.exec is {}, expected exec_steps={}", c.exec, cfg.exec));
        }
        if c.a.len() != cfg.horizon {
            return Err(format!("tick.chunk.a has {} rows, expected h={}", c.a.len(), cfg.horizon));
        }
        for (i, row) in c.a.iter().enumerate() {
            if row.len() != cfg.dim {
                return Err(format!("tick.chunk.a[{i}] has {} entries, expected d={}", row.len(), cfg.dim));
            }
        }
        if c.t_emit > req.t {
            return Err(format!("tick.chunk.t_emit {} is after t={}", c.t_emit, req.t));
        }
        if u32::from(req.idx) != req.t - c.t_emit {
            return Err(format!(
                "tick.idx {} is not t - t_emit = {} - {} = {}",
                req.idx,
                req.t,
                c.t_emit,
                req.t - c.t_emit
            ));
        }
        if c.seq != self.next_seq {
            return Err(format!("tick.chunk.seq is {}, expected next_chunk_seq={}", c.seq, self.next_seq));
        }
        let n = cfg.horizon * cfg.dim;
        for (i, row) in c.a.iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                self.flat[i * cfg.dim + j] = nan_or(*x);
            }
        }
        self.chunk
            .fill(c.seq, c.t_emit, c.h, c.d, c.exec, &self.flat[..n])
            .map_err(|e| format!("tick.chunk rejected by ChunkBuf: {e:?}"))?;
        self.have_chunk = true;
        self.next_seq = c.seq.wrapping_add(1);
        Ok(())
    }

    /// Borrow the staged buffers as the pure input. `schema_fault` forces TripMask::SCHEMA in GUARD.
    pub fn input(&self, req: &TickReq, ack: Option<VerifiedAck>, schema_fault: bool) -> TickInput<'_> {
        TickInput {
            obs: ObsView {
                t: req.t,
                pos: &self.pos[..self.n_pos],
                vel: if self.have_vel { Some(&self.vel[..self.n_vel]) } else { None },
                aux: &self.aux[..self.n_aux],
                ext: &self.ext[..self.n_ext],
            },
            chunk: if self.have_chunk { self.chunk.view() } else { None },
            idx: req.idx,
            missed_ticks: req.missed_ticks,
            ack,
            schema_fault,
        }
    }
}

impl Default for Staging {
    fn default() -> Self {
        Self::new()
    }
}

/// The timing entry of the last tick, waiting for its `io_ns` (folded on the next request or at episode_end).
struct PendingTiming {
    seq: u32,
    decide_ns: u64,
    io_ns: u64,
}

/// Everything that lives from `episode_begin` to `episode_end`.
struct Episode {
    run: RunBinding,
    budget: BudgetBinding,
    fault_injection: Option<FaultBinding>,
    inputs: BTreeMap<String, String>,
    paths: Option<EpisodePaths>,
    verdict_head: String,
    timing_head: String,
    ticks: Vec<TickEvent>,
    timing: Vec<TimingEvent>,
    pending_timing: Option<PendingTiming>,
    handoffs: Vec<HandoffRecord>,
    /// index into `handoffs` of the unresolved record, if any
    pending_handoff: Option<usize>,
}

/// Fuse + Staging + chains + timing + episode assembly + pending handoff + last_nonce (kept ACROSS episodes).
pub struct Session {
    cfg: SessionConfig,
    fuse: Fuse,
    staging: Staging,
    key: [u8; 32],
    ephemeral: bool,
    pubkey: String,
    envelope_digest: String,
    embodiment_digest: String,
    calibration_digest: Option<String>,
    policy_digest: Option<String>,
    envelope_json: serde_json::Value,
    last_id: u64,
    hello_ok: bool,
    client: String,
    ep: Option<Episode>,
    faulted: bool,
    /// ticks arrived with no episode: the fuse was reset once and every tick is fed as a schema fault
    phantom: bool,
    last_nonce: BTreeMap<String, u64>,
    hist: Hist,
}

fn err(id: u64, code: &str, message: String) -> Response {
    Response::Error(ErrorMsg { id, code: code.to_string(), message, fatal: true })
}

fn ack_error_name(e: &AckError) -> &'static str {
    match e {
        AckError::UnknownOperator => "unknown_operator",
        AckError::BadSignature => "bad_signature",
        AckError::WrongHandoff => "wrong_handoff",
        AckError::NonceReplay => "nonce_replay",
        AckError::NoPendingHandoff => "no_pending_handoff",
        AckError::TooLongNote => "too_long_note",
    }
}

fn names(mask: u32) -> Vec<String> {
    TripMask::names(mask).map(str::to_string).collect()
}

impl Session {
    /// Refuses (Err) a calibration whose `embodiment_digest` != `cfg.envelope.embodiment_digest()` -- fatal at startup, never a note.
    pub fn new(cfg: SessionConfig) -> anyhow::Result<Self> {
        cfg.envelope.validate().map_err(|e| anyhow::anyhow!("envelope: {e}"))?;
        let embodiment_digest = cfg.envelope.embodiment_digest();
        if let Some(c) = &cfg.calibration {
            if c.embodiment_digest != embodiment_digest {
                anyhow::bail!(
                    "calibration {} binds embodiment digest {} but the envelope's embodiment digest is {}",
                    c.path,
                    c.embodiment_digest,
                    embodiment_digest
                );
            }
        }
        let calib = if cfg.tier1 { cfg.calibration.as_ref().map(|c| c.c) } else { None };
        let mut fcfg = cfg.envelope.compile(cfg.mode, calib).map_err(|e| anyhow::anyhow!("compile: {e}"))?;
        if let Some(m) = cfg.tier0_override {
            fcfg.tier0_enabled = m;
        }
        let calibration_digest =
            if cfg.tier1 { cfg.calibration.as_ref().map(|c| c.digest.clone()) } else { None };
        let (key, ephemeral) = match cfg.key {
            Some(k) => (k, false),
            None => (lictor_receipt::keygen(), true),
        };
        let last_nonce = match &cfg.out_dir {
            Some(d) => load_nonces(&d.join(NONCE_FILE))?,
            None => BTreeMap::new(),
        };
        let envelope_json = floatify(serde_json::to_value(&cfg.envelope)?);
        let envelope_digest = cfg.envelope.digest_hex();
        let policy_digest = match (&cfg.calibration, cfg.tier1) {
            (Some(c), true) => {
                let d = calibration_policy_digest(c);
                if d.is_none() {
                    eprintln!(
                        "lictor: warning: calibration {} carries no readable policy_digest; the episode_begin weights check is skipped",
                        c.path
                    );
                }
                d
            }
            _ => None,
        };
        Ok(Self {
            fuse: Fuse::new(fcfg),
            staging: Staging::new(),
            key,
            ephemeral,
            pubkey: lictor_receipt::pubkey_hex(&key),
            envelope_digest,
            embodiment_digest,
            calibration_digest,
            policy_digest,
            envelope_json,
            last_id: 0,
            hello_ok: false,
            client: String::new(),
            ep: None,
            faulted: false,
            phantom: false,
            last_nonce,
            hist: Hist::new(),
            cfg,
        })
    }

    /// Handle one request; measures decide_ns around `decide()` and folds the chains AFTER it returns (never inside).
    /// `episode_end` is ACCEPTED while faulted (writes a receipt with terminal_state = fault, fuse_ok = false, ended_by as sent).
    pub fn handle(&mut self, req: Request) -> Response {
        self.fold_pending_timing();
        let id = match &req {
            Request::Hello(h) => h.id,
            Request::EpisodeBegin(b) => b.id,
            Request::Tick(t) => t.id,
            Request::EpisodeEnd(e) => e.id,
            Request::Bye(b) => b.id,
        };
        if id <= self.last_id {
            let msg = format!("request id {id} is not greater than the previous id {}", self.last_id);
            return self.fail(id, "protocol", msg);
        }
        self.last_id = id;
        match req {
            Request::Hello(h) => self.hello(h),
            Request::EpisodeBegin(b) => self.episode_begin(b),
            Request::Tick(t) => self.tick(t),
            Request::EpisodeEnd(e) => self.episode_end(e),
            Request::Bye(b) => Response::ByeOk { id: b.id },
        }
    }

    /// A line the codec rejected (malformed JSON, unknown key, framing): reply `error{code:"schema",fatal:true}` and
    /// latch the fault for the rest of the episode (the next tick carries `schema_fault = true` into `decide()`).
    pub fn handle_bad_line(&mut self, id: Option<u64>, message: &str) -> Response {
        self.fold_pending_timing();
        let id = id.unwrap_or(self.last_id);
        self.fail(id, "schema", message.to_string())
    }

    /// Called by the serve loop after the response is flushed: wall-clock from request-line read to response flush, recorded into
    /// the timing chain entry of tick `seq` (the entry is emitted on the NEXT request or at episode_end; never inside decide()).
    pub fn note_io_ns(&mut self, seq: u32, ns: u64) {
        if let Some(ep) = &mut self.ep {
            if let Some(p) = &mut ep.pending_timing {
                if p.seq == seq {
                    p.io_ns = ns;
                }
            }
        }
    }

    pub fn fault_latched(&self) -> bool {
        self.faulted
    }

    pub fn verdict_chain_head(&self) -> &str {
        match &self.ep {
            Some(ep) => &ep.verdict_head,
            None => ZERO_HASH,
        }
    }

    pub fn latency(&self) -> &latency::Hist {
        &self.hist
    }

    /// The compiled fuse configuration (for bench/selftest).
    pub fn fuse_cfg(&self) -> &FuseConfig {
        self.fuse.cfg()
    }

    /// The signing pubkey (hex64) of this session.
    pub fn pubkey(&self) -> &str {
        &self.pubkey
    }

    /// True while an episode is open (between `episode_begin` and `episode_end`).
    pub fn in_episode(&self) -> bool {
        self.ep.is_some()
    }

    /// `#meta` object for a trace file of this session.
    pub fn trace_meta(&self) -> serde_json::Value {
        serde_json::json!({
            "proto": PROTO,
            "envelope_digest": self.envelope_digest,
            "calibration_digest": self.calibration_digest,
            "mode": self.cfg.mode,
            "lictor": lictor_core::LICTOR_VERSION,
            "git": self.cfg.lictor_git,
        })
    }

    // ---- internals ----------------------------------------------------------------------------------------

    fn fail(&mut self, id: u64, code: &str, message: String) -> Response {
        if self.ep.is_some() {
            self.faulted = true;
        }
        err(id, code, message)
    }

    fn fold_pending_timing(&mut self) {
        if let Some(ep) = &mut self.ep {
            if let Some(p) = ep.pending_timing.take() {
                let e = timing_event(&ep.timing_head, p.seq, p.decide_ns, p.io_ns);
                ep.timing_head = e.hash.clone();
                ep.timing.push(e);
            }
        }
    }

    fn tier1_armed(&self) -> bool {
        self.calibration_digest.is_some()
    }

    fn budget_binding(&self, b: &crate::wire::BudgetMsg) -> BudgetBinding {
        let fc = self.fuse.cfg();
        let (alpha_num, alpha_den) =
            if self.tier1_armed() { (fc.calib.alpha_num, fc.calib.alpha_den) } else { (0, 1) };
        BudgetBinding {
            mode: self.cfg.mode,
            delay_steps: b.delay_steps,
            tick_ms: b.tick_ms,
            exec_mode: b.exec_mode,
            stitch: b.stitch.clone(),
            on_escalate: b.on_escalate.clone(),
            tier0_armed: names(fc.tier0_enabled),
            tier1_armed: self.tier1_armed(),
            gate: fc.calib.gate.names(),
            alpha_num,
            alpha_den,
            kn: [fc.hyst.k, fc.hyst.n],
        }
    }

    fn hello(&mut self, h: HelloReq) -> Response {
        if self.ep.is_some() {
            return self.fail(h.id, "state", "hello during an episode".to_string());
        }
        if h.proto != PROTO {
            return err(h.id, "protocol", format!("proto {:?} is not {PROTO:?}", h.proto));
        }
        let m = &self.cfg.envelope.embodiment;
        let mut mismatch = Vec::new();
        if h.mode != self.cfg.mode {
            mismatch.push(format!("mode {:?} != served {:?}", h.mode, self.cfg.mode));
        }
        if h.embodiment_id != m.id {
            mismatch.push(format!("embodiment_id {:?} != envelope {:?}", h.embodiment_id, m.id));
        }
        if h.action_dim != m.action_dim {
            mismatch.push(format!("action_dim {} != envelope {}", h.action_dim, m.action_dim));
        }
        if h.pos_dim != m.pos_dim {
            mismatch.push(format!("pos_dim {} != envelope {}", h.pos_dim, m.pos_dim));
        }
        if h.horizon != m.horizon {
            mismatch.push(format!("horizon {} != envelope {}", h.horizon, m.horizon));
        }
        if h.exec_steps != m.exec_steps {
            mismatch.push(format!("exec_steps {} != envelope {}", h.exec_steps, m.exec_steps));
        }
        if h.envelope_digest != self.envelope_digest {
            mismatch
                .push(format!("envelope_digest {} != loaded {}", h.envelope_digest, self.envelope_digest));
        }
        if h.calibration_digest != self.calibration_digest {
            mismatch.push(format!(
                "calibration_digest {:?} != loaded {:?}",
                h.calibration_digest, self.calibration_digest
            ));
        }
        if !mismatch.is_empty() {
            self.hello_ok = false;
            return err(h.id, "envelope", mismatch.join("; "));
        }
        self.hello_ok = true;
        self.client = h.client;
        let fc = self.fuse.cfg();
        let calibration = if self.tier1_armed() {
            Some(CalibrationInfo {
                method: fc.calib.method,
                alpha_num: fc.calib.alpha_num,
                alpha_den: fc.calib.alpha_den,
                n_calib: fc.calib.n_calib,
                tau: fc.calib.tau,
                kn: [fc.hyst.k, fc.hyst.n],
                gate: fc.calib.gate.names(),
                digest: self.calibration_digest.clone().unwrap_or_default(),
            })
        } else {
            None
        };
        Response::HelloOk(HelloOk {
            id: h.id,
            proto: PROTO.to_string(),
            lictor: lictor_core::LICTOR_VERSION.to_string(),
            git: self.cfg.lictor_git.clone(),
            lictor_sha256: self.cfg.lictor_sha256.clone(),
            envelope_digest: self.envelope_digest.clone(),
            embodiment_digest: self.embodiment_digest.clone(),
            calibration_digest: self.calibration_digest.clone(),
            pubkey: self.pubkey.clone(),
            ephemeral_key: self.ephemeral,
            mode: self.cfg.mode,
            tier0_armed: names(fc.tier0_enabled),
            tier1_armed: self.tier1_armed(),
            calibration,
            features: Feat::NAMES.iter().map(|s| s.to_string()).collect(),
            trip_names: TripMask::NAMES.iter().map(|s| s.to_string()).collect(),
            latency_label: self.cfg.latency_label.clone(),
        })
    }

    fn episode_begin(&mut self, b: EpisodeBeginReq) -> Response {
        if self.ep.is_some() {
            return self.fail(b.id, "state", "episode_begin while an episode is open".to_string());
        }
        if let Some(pd) = &self.policy_digest {
            let w = b.binding.policy.get("weights_sha256").cloned().unwrap_or_default();
            if &w != pd {
                return err(
                    b.id,
                    "envelope",
                    format!("binding.policy.weights_sha256 {w:?} != calibration policy_digest {pd:?}"),
                );
            }
        }
        let run = RunBinding {
            run_id: b.run.run_id,
            arm_id: b.run.arm_id,
            episode_index: b.run.episode_index,
            seed: b.run.seed,
            seed_pool: b.run.seed_pool,
            init_state_digest: b.run.init_state_digest,
            env: b.binding.env,
            policy: b.binding.policy,
            host: b.binding.host,
        };
        let budget = self.budget_binding(&b.budget);
        let fault_injection = b.fault_injection.map(|f| FaultBinding {
            kind: f.kind,
            params: f.params,
            stream_seed: f.stream_seed,
        });
        let mut inputs = b.inputs;
        inputs.extend(fuse_inputs(&self.cfg));
        let paths =
            self.cfg.out_dir.as_ref().map(|d| episode_paths(d, &run.run_id, &run.arm_id, run.episode_index));
        self.fuse.reset(EpisodeInit {
            episode_index: run.episode_index,
            seed: run.seed,
            delay_steps: budget.delay_steps,
            exec: budget.exec_mode,
        });
        self.staging = Staging::new();
        self.hist = Hist::new();
        self.faulted = false;
        self.phantom = false;
        self.ep = Some(Episode {
            run,
            budget,
            fault_injection,
            inputs,
            paths,
            verdict_head: ZERO_HASH.to_string(),
            timing_head: ZERO_HASH.to_string(),
            ticks: Vec::new(),
            timing: Vec::new(),
            pending_timing: None,
            handoffs: Vec::new(),
            pending_handoff: None,
        });
        Response::EpisodeOk(EpisodeOk { id: b.id, state: self.fuse.rt().state, seq: self.fuse.rt().seq })
    }

    /// Verify an ack against the pending handoff; never touches the fuse.
    fn check_ack(&self, t: &TickReq) -> (Option<VerifiedAck>, Option<String>) {
        let Some(tok) = &t.ack else {
            return (None, None);
        };
        let ops = &self.cfg.envelope.operators;
        let slots: Vec<u64> = ops.iter().map(|o| self.last_nonce.get(o).copied().unwrap_or(0)).collect();
        let (pending_digest, pending_seq) = match &self.ep {
            Some(ep) => match ep.pending_handoff.and_then(|i| ep.handoffs.get(i)) {
                Some(h) => (Some(h.digest.as_str()), h.seq),
                None => (None, 0),
            },
            None => (None, 0),
        };
        match verify_ack(tok, ops, pending_digest, pending_seq, &slots) {
            Ok(v) => (Some(v), Some("accepted".to_string())),
            Err(e) => (None, Some(format!("rejected:{}", ack_error_name(&e)))),
        }
    }

    fn tick(&mut self, t: TickReq) -> Response {
        if self.ep.is_none() {
            // No episode: the first offending tick is an error; the fuse is then reset once and every further
            // tick is fed through decide() as a schema fault so the host only ever receives the hold action.
            if !self.phantom {
                self.phantom = true;
                self.fuse.reset(EpisodeInit {
                    episode_index: 0,
                    seed: 0,
                    delay_steps: 0,
                    exec: lictor_core::ExecMode::Sync,
                });
                self.faulted = true;
                return err(t.id, "state", "tick before episode_begin".to_string());
            }
            let _ = self.staging.stage(self.fuse.cfg(), &t);
            let v = self.fuse.step(&self.staging.input(&t, None, true));
            return Response::Verdict(Box::new(self.verdict_msg(
                &t,
                &v,
                None,
                None,
                0,
                ZERO_HASH.to_string(),
            )));
        }
        let (ack, ack_result) = self.check_ack(&t);
        let stage_err = self.staging.stage(self.fuse.cfg(), &t).err();
        if stage_err.is_some() {
            self.faulted = true;
        }
        let schema_fault = self.faulted;
        let t0 = Instant::now();
        let v = self.fuse.step(&self.staging.input(&t, ack, schema_fault));
        let decide_ns = u64::try_from(t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.hist.record(decide_ns);

        // ---- everything below runs AFTER decide() returned: hashing, records, bookkeeping ----
        if let (Some(tok), Some(va)) = (&t.ack, ack) {
            if ack_result.as_deref() == Some("accepted") {
                self.last_nonce.insert(tok.operator.clone(), va.nonce);
                if let Some(d) = &self.cfg.out_dir {
                    if let Err(e) = save_nonces(&d.join(NONCE_FILE), &self.last_nonce) {
                        eprintln!("lictor: warning: could not persist ack nonces: {e}");
                    }
                }
            }
        }
        let ep = self.ep.as_mut().expect("episode is open");
        let event = tick_event(&ep.verdict_head, &v);
        ep.verdict_head = event.hash.clone();
        let chain = event.hash.clone();
        ep.ticks.push(event);
        ep.pending_timing = Some(PendingTiming { seq: v.seq, decide_ns, io_ns: 0 });

        // resolve the pending handoff (ack applied, or handoff timeout)
        if let Some(i) = ep.pending_handoff {
            let resolved = if v.ack_consumed {
                match t.ack.as_ref().map(|a| a.decision) {
                    Some(AckDecision::Resume) => Some("resumed"),
                    Some(AckDecision::Abort) => Some("aborted"),
                    Some(AckDecision::Retune) => Some("retune"),
                    None => None,
                }
            } else if v.prev_state == FuseState::Escalated && v.state == FuseState::Terminated {
                Some("timeout")
            } else {
                None
            };
            if let Some(outcome) = resolved {
                if let Some(h) = ep.handoffs.get_mut(i) {
                    h.ack = if v.ack_consumed { t.ack.clone() } else { None };
                    h.resolved_tick = Some(v.t);
                    h.outcome = outcome.to_string();
                }
                ep.pending_handoff = None;
            }
        }
        let handoff = if let Some(seq) = v.handoff_seq {
            let rec = self.handoff_record(&v, seq, &chain);
            let ep = self.ep.as_mut().expect("episode is open");
            ep.handoffs.push(rec.clone());
            ep.pending_handoff = Some(ep.handoffs.len() - 1);
            Some(rec)
        } else {
            None
        };
        if let Some(msg) = stage_err {
            // The fault is already in the chain (this tick ran with schema_fault = true); the wire reply is the
            // error so the host aborts the episode and sends episode_end(ended_by = "fault").
            return err(t.id, "schema", msg);
        }
        let msg = self.verdict_msg(&t, &v, handoff, ack_result, decide_ns, chain);
        Response::Verdict(Box::new(msg))
    }

    fn handoff_record(&self, v: &SafetyVerdict, seq: u32, chain_at: &str) -> HandoffRecord {
        let ep = self.ep.as_ref().expect("episode is open");
        let fc = self.fuse.cfg();
        let mut zs: Vec<(String, f64)> = (0..NFEAT)
            .filter(|j| v.scores.valid & fc.calib.mask & (1u32 << j) != 0)
            .map(|j| (Feat::NAMES[j].to_string(), v.scores.z[j]))
            .collect();
        zs.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        zs.truncate(3);
        let rec = HandoffRecord {
            schema: HANDOFF_SCHEMA.to_string(),
            run_id: ep.run.run_id.clone(),
            arm_id: ep.run.arm_id.clone(),
            episode_index: ep.run.episode_index,
            seq,
            tick: v.t,
            reason: v.reason,
            reason_text: reason_text(v.reason).to_string(),
            reasons: names(v.trips),
            trips: v.trips,
            fired: v.scores.fired,
            window_hits: v.window_hits,
            top_z: zs.into_iter().map(|(n, z)| (n, F64Hex(z))).collect(),
            chain_at: chain_at.to_string(),
            envelope_digest: self.envelope_digest.clone(),
            calibration_digest: self.calibration_digest.clone(),
            digest: String::new(),
            ack: None,
            resolved_tick: None,
            outcome: "unacked".to_string(),
        };
        rec.with_digest().expect("a handoff record is float-free and key-clean by construction")
    }

    fn verdict_msg(
        &self,
        t: &TickReq,
        v: &SafetyVerdict,
        handoff: Option<HandoffRecord>,
        ack_result: Option<String>,
        verdict_ns: u64,
        chain: String,
    ) -> VerdictMsg {
        let dim = (v.action_dim as usize).min(MAX_D);
        VerdictMsg {
            id: t.id,
            t: v.t,
            seq: v.seq,
            status: v.status,
            state: v.state,
            prev_state: v.prev_state,
            trips: names(v.trips),
            trip_mask: v.trips,
            action: v.action[..dim].to_vec(),
            action_src: v.action_src,
            substituted: v.substituted,
            clamped_dims: v.clamped_dims,
            scores: ScoresMsg {
                f: v.scores.f.to_vec(),
                z: v.scores.z.to_vec(),
                s: v.scores.s,
                valid: v.scores.valid,
                fired: v.scores.fired,
            },
            tau: v.tau,
            window_hits: v.window_hits,
            brake_margin: v.brake_margin,
            reason: v.reason,
            reason_text: reason_text(v.reason).to_string(),
            handoff,
            ack_result,
            violation_reached_env: v.violation_reached_env,
            verdict_ns,
            chain,
        }
    }

    fn episode_end(&mut self, e: EpisodeEndReq) -> Response {
        if self.ep.is_none() {
            return self.fail(e.id, "state", "episode_end without an open episode".to_string());
        }
        self.fold_pending_timing();
        let ep = self.ep.take().expect("episode is open");
        self.faulted = false;
        let tally = self.fuse.finish();
        let counts = VerdictCounts::from(&tally);
        let (fuse_ok, fuse_notes) =
            evaluate_fuse(&ep.budget, &counts, self.calibration_digest.as_deref(), self.ephemeral);
        let ticks = match self.cfg.ticks_policy.as_str() {
            "all" => ep.ticks.clone(),
            "none" => Vec::new(),
            _ => ep.ticks[ep.ticks.len().saturating_sub(32)..].to_vec(),
        };
        let ledger_prev = match &ep.paths {
            Some(p) => match ledger_head(&p.ledger) {
                Ok(h) => h,
                Err(er) => return err(e.id, "internal", format!("ledger: {er}")),
            },
            None => None,
        };
        let o = &e.outcome;
        let body = ReceiptBody {
            schema: RECEIPT_SCHEMA.to_string(),
            canonical: lictor_canon::CANONICAL_ID.to_string(),
            created_epoch: now_epoch(),
            lictor_version: lictor_core::LICTOR_VERSION.to_string(),
            lictor_git: self.cfg.lictor_git.clone(),
            lictor_sha256: self.cfg.lictor_sha256.clone(),
            client: self.client.clone(),
            run: ep.run.clone(),
            budget: ep.budget.clone(),
            fault_injection: ep.fault_injection.clone(),
            envelope: self.envelope_json.clone(),
            envelope_digest: self.envelope_digest.clone(),
            calibration_digest: self.calibration_digest.clone(),
            inputs: ep.inputs.clone(),
            counts: counts.clone(),
            outcome: EpisodeOutcome {
                steps: o.steps,
                success: o.success,
                terminated: o.terminated,
                truncated: o.truncated,
                max_coverage: F64Hex(nan_or(o.max_coverage)),
                final_coverage: F64Hex(nan_or(o.final_coverage)),
                reward_sum: F64Hex(nan_or(o.reward_sum)),
                ended_by: o.ended_by.clone(),
                max_s: F64Hex(tally.max_s),
                max_z: F64Array::from_slice(&tally.max_z, &[NFEAT as u32]),
            },
            handoffs: ep.handoffs.clone(),
            verdict_events: ep.ticks.len() as u32,
            verdict_chain_head: ep.verdict_head.clone(),
            timing_events: ep.timing.len() as u32,
            timing_chain_head: ep.timing_head.clone(),
            latency: self.hist.summary(&self.cfg.latency_label),
            ticks_policy: self.cfg.ticks_policy.clone(),
            ticks,
            fuse_ok,
            fuse_notes: fuse_notes.clone(),
            ledger_prev,
        };
        let sr = match sign(body, &self.key) {
            Ok(sr) => sr,
            Err(er) => return err(e.id, "internal", format!("sign: {er}")),
        };
        let (receipt_path, ticks_path, ledger_seq, ledger_head_hex) = match &ep.paths {
            Some(p) => match write_episode(p, &sr, &ep.ticks, &ep.timing) {
                Ok(entry) => {
                    let (r, t) = paths_display(p);
                    (r, t, entry.seq, entry.hash)
                }
                Err(er) => return err(e.id, "internal", format!("write episode: {er}")),
            },
            None => (String::new(), String::new(), 0, ZERO_HASH.to_string()),
        };
        Response::EpisodeReceipt(EpisodeReceiptMsg {
            id: e.id,
            receipt_path,
            ticks_path,
            body_digest: sr.body_digest,
            verdict_chain_head: sr.body.verdict_chain_head,
            timing_chain_head: sr.body.timing_chain_head,
            fuse_ok,
            fuse_notes,
            counts,
            ledger_seq,
            ledger_head: ledger_head_hex,
        })
    }
}

/// The `policy_digest` bound by calibration.json. The frozen `CalibrationLoaded` carries no such field and the
/// runtime does not depend on lictor-calib, so it is read from the JSON file at `CalibrationLoaded.path` (a plain
/// `serde_json::Value` lookup). `None` when the path is empty / unreadable / has no `policy_digest` key -- the
/// `episode_begin` weights check is then skipped with a startup warning (reported as a freeze gap).
pub fn calibration_policy_digest(c: &CalibrationLoaded) -> Option<String> {
    if c.path.is_empty() {
        return None;
    }
    let text = std::fs::read_to_string(&c.path).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("policy_digest")?.as_str().map(str::to_string)
}

/// WSL2_LABEL when /proc/version contains "microsoft" or "WSL", else "measured on <uname -sr>, non-RT kernel -- not a real-time environment".
pub fn default_latency_label() -> String {
    if let Ok(v) = std::fs::read_to_string("/proc/version") {
        let lower = v.to_ascii_lowercase();
        if lower.contains("microsoft") || v.contains("WSL") {
            return WSL2_LABEL.to_string();
        }
    }
    let uname = std::process::Command::new("uname")
        .arg("-sr")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| format!("{} {}", std::env::consts::OS, std::env::consts::ARCH));
    format!("measured on {uname}, non-RT kernel -- not a real-time environment")
}

/// The output-directory-relative path helper used by serve for per-episode trace files.
pub fn trace_path(out_dir: &std::path::Path, run_id: &str, arm_id: &str, episode_index: u32) -> PathBuf {
    out_dir.join(run_id).join(arm_id).join("traces").join(format!("{episode_index:06}.ndjson"))
}
