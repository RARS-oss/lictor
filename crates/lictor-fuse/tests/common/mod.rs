// SPDX-License-Identifier: MIT
//! Shared helpers for the lictor-fuse integration tests: the compiled base envelope, owned observation /
//! chunk builders that hand out the borrowed views `decide()` takes, name <-> enum conversions through the
//! frozen serde names, and the loader of `fixtures/episode/synthetic_300.json`.
#![allow(dead_code)]

use lictor_core::{
    AckDecision, ChunkView, EpisodeInit, ExecMode, FuseConfig, FuseMode, FuseState, ObsView, ReasonCode,
    SafetyEnvelope, SafetyVerdict, TripMask, VerifiedAck,
};
use lictor_fuse::{Fuse, TickInput};
use serde_json::Value;

/// The base envelope; `crates/lictor-fuse/tests/common/` is four levels below the repo root.
pub const BASE_TOML: &str = include_str!("../../../../envelopes/pusht.base.toml");
pub const EPISODE_JSON: &str = include_str!("../fixtures/episode/synthetic_300.json");

pub fn envelope() -> SafetyEnvelope {
    SafetyEnvelope::from_toml(BASE_TOML).expect("base envelope parses")
}

/// The base envelope compiled without a calibration (Tier 1 carries the envelope gate with tau = +inf).
pub fn cfg(mode: FuseMode) -> FuseConfig {
    cfg_with(mode, |_| {})
}

/// The base envelope, edited by `f`, compiled without a calibration.
pub fn cfg_with(mode: FuseMode, f: impl FnOnce(&mut SafetyEnvelope)) -> FuseConfig {
    let mut e = envelope();
    f(&mut e);
    e.compile(mode, None).expect("edited envelope compiles")
}

pub fn init() -> EpisodeInit {
    EpisodeInit { episode_index: 0, seed: 0, delay_steps: 0, exec: ExecMode::Sync }
}

/// An owned observation; `view()` borrows it as the `ObsView` the fuse takes.
#[derive(Clone, Debug)]
pub struct Obs {
    pub t: u32,
    pub pos: Vec<f64>,
    pub vel: Option<Vec<f64>>,
    pub aux: Vec<f64>,
    pub ext: Vec<f64>,
}

impl Obs {
    pub fn at(t: u32, pos: [f64; 2], vel: [f64; 2]) -> Self {
        Self { t, pos: pos.to_vec(), vel: Some(vel.to_vec()), aux: vec![256.0, 256.0, 0.0, 0.1], ext: vec![] }
    }

    pub fn view(&self) -> ObsView<'_> {
        ObsView { t: self.t, pos: &self.pos, vel: self.vel.as_deref(), aux: &self.aux, ext: &self.ext }
    }
}

/// An owned chunk; `view()` borrows it as the `ChunkView` the fuse takes.
#[derive(Clone, Debug)]
pub struct Chunk {
    pub seq: u32,
    pub t_emit: u32,
    pub h: u16,
    pub d: u16,
    pub exec: u16,
    pub a: Vec<f64>,
}

impl Chunk {
    /// A 15 x 2 chunk with `exec = 8` from explicit rows.
    pub fn rows(seq: u32, t_emit: u32, rows: &[[f64; 2]]) -> Self {
        let a = rows.iter().flat_map(|r| r.iter().copied()).collect();
        Self { seq, t_emit, h: rows.len() as u16, d: 2, exec: 8, a }
    }

    /// A straight 15-row chunk: row i = `from + (i + 1) * step`.
    pub fn straight(seq: u32, t_emit: u32, from: [f64; 2], step: [f64; 2]) -> Self {
        let rows: Vec<[f64; 2]> = (0..15)
            .map(|i| {
                let k = (i + 1) as f64;
                [from[0] + k * step[0], from[1] + k * step[1]]
            })
            .collect();
        Self::rows(seq, t_emit, &rows)
    }

    pub fn view(&self) -> ChunkView<'_> {
        ChunkView {
            seq: self.seq,
            t_emit: self.t_emit,
            horizon: self.h,
            dim: self.d,
            exec_steps: self.exec,
            data: &self.a,
        }
    }

    pub fn row(&self, i: usize) -> [f64; 2] {
        [self.a[i * 2], self.a[i * 2 + 1]]
    }
}

pub fn ack(decision: AckDecision, slot: u8, nonce: u64, handoff_seq: u32) -> VerifiedAck {
    VerifiedAck { decision, operator_slot: slot, nonce, handoff_seq }
}

/// One `Fuse::step` from owned inputs.
pub fn tick(fuse: &mut Fuse, obs: &Obs, chunk: Option<&Chunk>, idx: u16, missed: u8) -> SafetyVerdict {
    tick_full(fuse, obs, chunk, idx, missed, None, false)
}

pub fn tick_full(
    fuse: &mut Fuse,
    obs: &Obs,
    chunk: Option<&Chunk>,
    idx: u16,
    missed: u8,
    ack: Option<VerifiedAck>,
    schema_fault: bool,
) -> SafetyVerdict {
    let inp = TickInput {
        obs: obs.view(),
        chunk: chunk.map(|c| c.view()),
        idx,
        missed_ticks: missed,
        ack,
        schema_fault,
    };
    fuse.step(&inp)
}

/// The first `n` entries of `v.action`.
pub fn action2(v: &SafetyVerdict) -> [f64; 2] {
    [v.action[0], v.action[1]]
}

pub fn clamp2(cfg: &FuseConfig, p: [f64; 2]) -> [f64; 2] {
    let c = |x: f64, lo: f64, hi: f64| {
        if x < lo {
            lo
        } else if x > hi {
            hi
        } else {
            x
        }
    };
    [c(p[0], cfg.box_lo[0], cfg.box_hi[0]), c(p[1], cfg.box_lo[1], cfg.box_hi[1])]
}

// ---- names <-> enums through the frozen serde names -------------------------------------------------------

pub fn state_of(name: &str) -> FuseState {
    serde_json::from_value(Value::String(name.to_string())).unwrap_or_else(|_| panic!("unknown state {name}"))
}

pub fn state_name(s: FuseState) -> String {
    match serde_json::to_value(s) {
        Ok(Value::String(n)) => n,
        other => panic!("state serialises to {other:?}"),
    }
}

pub fn reason_of(name: &str) -> ReasonCode {
    serde_json::from_value(Value::String(name.to_string()))
        .unwrap_or_else(|_| panic!("unknown reason {name}"))
}

pub fn reason_name(r: ReasonCode) -> String {
    match serde_json::to_value(r) {
        Ok(Value::String(n)) => n,
        other => panic!("reason serialises to {other:?}"),
    }
}

pub fn decision_of(name: &str) -> AckDecision {
    serde_json::from_value(Value::String(name.to_string()))
        .unwrap_or_else(|_| panic!("unknown decision {name}"))
}

/// A trip mask from a JSON array of bit names.
pub fn trips_of(v: &Value) -> u32 {
    v.as_array()
        .unwrap_or_else(|| panic!("trip list expected, got {v}"))
        .iter()
        .map(|n| n.as_str().expect("trip name"))
        .fold(0, |m, n| m | TripMask::from_name(n).unwrap_or_else(|| panic!("unknown trip {n}")))
}

pub fn trip_names(m: u32) -> Vec<&'static str> {
    TripMask::names(m).collect()
}

// ---- the synthetic episode fixture ------------------------------------------------------------------------

/// `fixtures/episode/synthetic_300.json`: 300 observations and one 15 x 2 chunk every 8 ticks.
pub struct Episode {
    pub obs: Vec<Obs>,
    pub chunks: Vec<Chunk>,
}

impl Episode {
    pub fn load() -> Self {
        let v: Value = serde_json::from_str(EPISODE_JSON).expect("episode fixture parses");
        let f64s =
            |x: &Value| -> Vec<f64> { x.as_array().unwrap().iter().map(|y| y.as_f64().unwrap()).collect() };
        let obs = v["obs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| Obs {
                t: o["t"].as_u64().unwrap() as u32,
                pos: f64s(&o["pos"]),
                vel: Some(f64s(&o["vel"])),
                aux: f64s(&o["aux"]),
                ext: f64s(&o["ext"]),
            })
            .collect();
        let chunks = v["chunks"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| {
                let rows: Vec<[f64; 2]> = c["a"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| {
                        let r = f64s(r);
                        [r[0], r[1]]
                    })
                    .collect();
                let mut ch = Chunk::rows(
                    c["seq"].as_u64().unwrap() as u32,
                    c["t_emit"].as_u64().unwrap() as u32,
                    &rows,
                );
                ch.h = c["h"].as_u64().unwrap() as u16;
                ch.d = c["d"].as_u64().unwrap() as u16;
                ch.exec = c["exec"].as_u64().unwrap() as u16;
                ch
            })
            .collect();
        Self { obs, chunks }
    }

    /// The chunk delivered at tick `t`, if any, and the row index this tick executes.
    pub fn at(&self, t: u32) -> (Option<&Chunk>, u16) {
        let cur = self.chunks.iter().rev().find(|c| c.t_emit <= t).expect("a chunk covers every tick");
        let delivered = if cur.t_emit == t { Some(cur) } else { None };
        (delivered, (t - cur.t_emit) as u16)
    }

    /// The raw policy row executed at tick `t`.
    pub fn policy_row(&self, t: u32) -> [f64; 2] {
        let (_, idx) = self.at(t);
        let cur = self.chunks.iter().rev().find(|c| c.t_emit <= t).unwrap();
        cur.row(idx as usize)
    }
}

/// Runs the whole synthetic episode through `fuse` (already reset) and returns every verdict in order.
pub fn run_episode(fuse: &mut Fuse, ep: &Episode) -> Vec<SafetyVerdict> {
    ep.obs
        .iter()
        .map(|o| {
            let (chunk, idx) = ep.at(o.t);
            tick(fuse, o, chunk, idx, 0)
        })
        .collect()
}

/// A compact `t:state` timeline of the ticks where the state changes (for failure messages).
pub fn timeline(vs: &[SafetyVerdict]) -> String {
    let mut out = String::new();
    let mut last: Option<FuseState> = None;
    for v in vs {
        if last != Some(v.state) {
            out.push_str(&format!("{}:{} ", v.t, state_name(v.state)));
            last = Some(v.state);
        }
    }
    out
}
