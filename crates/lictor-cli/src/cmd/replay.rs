// SPDX-License-Identifier: MIT
//! `lictor replay`: "replays N/N byte-identical  verdict_chain=<hex>"; exit 1 on any divergence.
//!
//! The trace (`#meta` line + every request line verbatim) is fed `--repeat` times through a FRESH `Session`
//! (no output directory, no key -- receipts are not written; `mode` from the `#meta` unless overridden). Every
//! repeat records the per-tick verdict-chain value and the per-episode chain head; a repeat is byte-identical when
//! both sequences equal the first repeat's. The timing chain is wall-clock and is never compared. `--expect`
//! compares the replayed head with a signed one (`body.verdict_chain_head` of the receipt the trace produced).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use lictor_core::{FuseMode, SafetyEnvelope, Status};
use lictor_runtime::session::{CalibrationLoaded, Session, SessionConfig};
use lictor_runtime::trace::{read_trace, TraceLine};
use lictor_runtime::wire::{Request, Response};

use crate::cmd::serve::{binary_sha256, load_calibration, load_envelope, parse_mode};
use crate::cmd::version::LICTOR_GIT;
use crate::render::{json_out, print_lines, sanitize};

#[derive(clap::Args)]
pub struct Args {
    /// Request trace (#meta line + verbatim request lines)
    #[arg(value_name = "TRACE.ndjson")]
    pub trace: PathBuf,
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// calibration.json
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// observe | enforce (default: the trace's #meta mode)
    #[arg(long, value_parser = ["observe", "enforce"])]
    pub mode: Option<String>,
    /// Number of replays through fresh sessions
    #[arg(long, value_name = "N", default_value_t = 1)]
    pub repeat: u32,
    /// Expected verdict chain head (hex64)
    #[arg(long, value_name = "hex64")]
    pub expect: Option<String>,
}

/// What one replay needs; the envelope and calibration are loaded once and cloned per session.
pub struct ReplayInput<'a> {
    pub lines: &'a [TraceLine],
    pub envelope: &'a SafetyEnvelope,
    pub toml_text: &'a str,
    pub calibration: Option<&'a CalibrationLoaded>,
    pub mode: FuseMode,
    pub git: String,
    pub sha256: String,
}

/// The first repeat and tick at which a replay stopped agreeing with the first one.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Divergence {
    pub repeat: u32,
    pub tick: u32,
    pub seq: u32,
    pub first: String,
    pub other: String,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct ReplayOutcome {
    pub ticks: u32,
    pub repeats: u32,
    pub identical: u32,
    /// The head of the last episode of the trace (or the last tick's chain value when the trace ends mid-episode).
    pub verdict_chain: String,
    /// Every episode's head, in trace order.
    pub heads: Vec<String>,
    pub states: BTreeMap<String, u32>,
    /// Fatal error responses seen in the first repeat.
    pub errors: u32,
    pub first_error: Option<String>,
    pub divergence: Option<Divergence>,
    pub deterministic: bool,
    pub expect: Option<String>,
    pub expect_ok: Option<bool>,
}

/// One pass through a fresh session.
struct Pass {
    chains: Vec<(u32, u32, String)>,
    heads: Vec<String>,
    states: BTreeMap<String, u32>,
    errors: u32,
    first_error: Option<String>,
}

const STATE_ORDER: [(&str, Status); 8] = [
    ("nominal", Status::Nominal),
    ("watching", Status::Watching),
    ("clamped", Status::Clamped),
    ("braking", Status::Braking),
    ("held", Status::Held),
    ("escalated", Status::Escalated),
    ("fault", Status::Fault),
    ("terminated", Status::Terminated),
];

fn state_name(s: Status) -> &'static str {
    STATE_ORDER.iter().find(|(_, st)| *st == s).map(|(n, _)| *n).unwrap_or("other")
}

fn clone_loaded(c: &CalibrationLoaded) -> CalibrationLoaded {
    CalibrationLoaded {
        c: c.c,
        digest: c.digest.clone(),
        embodiment_digest: c.embodiment_digest.clone(),
        file_sha256: c.file_sha256.clone(),
        path: c.path.clone(),
    }
}

fn one_pass(inp: &ReplayInput<'_>) -> anyhow::Result<Pass> {
    let mut cfg = SessionConfig::minimal(inp.envelope.clone(), inp.toml_text, inp.mode);
    cfg.calibration = inp.calibration.map(clone_loaded);
    cfg.lictor_git = inp.git.clone();
    cfg.lictor_sha256 = inp.sha256.clone();
    let mut s = Session::new(cfg)?;
    let mut pass = Pass {
        chains: Vec::new(),
        heads: Vec::new(),
        states: STATE_ORDER.iter().map(|(n, _)| (n.to_string(), 0u32)).collect(),
        errors: 0,
        first_error: None,
    };
    for (_, _, req) in inp.lines {
        let req: Request = req.clone();
        match s.handle(req) {
            Response::Verdict(v) => {
                *pass.states.entry(state_name(v.status).to_string()).or_insert(0) += 1;
                pass.chains.push((v.t, v.seq, v.chain.clone()));
            }
            Response::EpisodeReceipt(m) => pass.heads.push(m.verdict_chain_head.clone()),
            Response::Error(e) => {
                pass.errors += 1;
                if pass.first_error.is_none() {
                    pass.first_error = Some(format!(
                        "id={} code={} fatal={}: {}",
                        e.id,
                        e.code,
                        e.fatal,
                        sanitize(&e.message)
                    ));
                }
            }
            Response::HelloOk(_) | Response::EpisodeOk(_) | Response::ByeOk { .. } => {}
        }
    }
    Ok(pass)
}

/// Replay `repeat` times; the first pass is the reference.
pub fn replay(inp: &ReplayInput<'_>, repeat: u32) -> anyhow::Result<ReplayOutcome> {
    let repeat = repeat.max(1);
    let first = one_pass(inp)?;
    let verdict_chain = match first.heads.last() {
        Some(h) => h.clone(),
        None => first.chains.last().map(|(_, _, c)| c.clone()).unwrap_or_default(),
    };
    let mut out = ReplayOutcome {
        ticks: first.chains.len() as u32,
        repeats: repeat,
        identical: 1,
        verdict_chain,
        heads: first.heads.clone(),
        states: first.states.clone(),
        errors: first.errors,
        first_error: first.first_error.clone(),
        divergence: None,
        deterministic: false,
        expect: None,
        expect_ok: None,
    };
    for k in 1..repeat {
        let p = one_pass(inp)?;
        let same_chains = p.chains == first.chains;
        let same_heads = p.heads == first.heads;
        if same_chains && same_heads {
            out.identical += 1;
            continue;
        }
        if out.divergence.is_none() {
            let d = first
                .chains
                .iter()
                .zip(p.chains.iter())
                .find(|(a, b)| a != b)
                .map(|(a, b)| Divergence {
                    repeat: k,
                    tick: a.0,
                    seq: a.1,
                    first: a.2.clone(),
                    other: b.2.clone(),
                })
                .or_else(|| {
                    first.chains.iter().zip(p.chains.iter()).last().map(|(a, _)| Divergence {
                        repeat: k,
                        tick: a.0,
                        seq: a.1,
                        first: first.heads.last().cloned().unwrap_or_default(),
                        other: p.heads.last().cloned().unwrap_or_default(),
                    })
                })
                .unwrap_or(Divergence {
                    repeat: k,
                    tick: 0,
                    seq: 0,
                    first: first.heads.last().cloned().unwrap_or_default(),
                    other: p.heads.last().cloned().unwrap_or_default(),
                });
            out.divergence = Some(d);
        }
    }
    out.deterministic = out.identical == repeat && out.ticks > 0;
    Ok(out)
}

/// The human lines of an outcome (frozen strings).
pub fn render(out: &ReplayOutcome) -> Vec<String> {
    let mut lines = vec![
        format!("  ticks={}  repeats={}", out.ticks, out.repeats),
        format!(
            "  replays {}/{} byte-identical  verdict_chain={}",
            out.identical, out.repeats, out.verdict_chain
        ),
        "  timing chain head varies (by design -- wall-clock is not replayed)".to_string(),
    ];
    let mut states = String::from("  states");
    for (name, _) in STATE_ORDER.iter() {
        let n = out.states.get(*name).copied().unwrap_or(0);
        if *name == "terminated" && n == 0 {
            continue;
        }
        states.push_str(&format!(" {name}={n}"));
    }
    lines.push(states);
    if out.heads.len() > 1 {
        lines.push(format!("  episodes={}  (verdict_chain is the last episode's head)", out.heads.len()));
    }
    if let Some(e) = &out.first_error {
        lines.push(format!("  errors={}  first: {e}", out.errors));
    }
    if let Some(d) = &out.divergence {
        lines.push(format!(
            "  DIVERGENCE repeat {} tick {} (seq {}): {} != {}",
            d.repeat, d.tick, d.seq, d.first, d.other
        ));
    }
    if let (Some(e), Some(ok)) = (&out.expect, out.expect_ok) {
        if ok {
            lines.push(format!("  expect  ok    {e}"));
        } else {
            lines.push(format!("  expect  MISMATCH  expected={e} replayed={}", out.verdict_chain));
        }
    }
    if out.ticks == 0 {
        lines.push("  RESULT  NO VERDICTS (the trace produced no ticks; see errors)".to_string());
    } else if out.deterministic && out.expect_ok.unwrap_or(true) {
        lines.push("  RESULT  DETERMINISTIC".to_string());
    } else if out.deterministic {
        lines.push("  RESULT  DETERMINISTIC BUT NOT THE EXPECTED HEAD".to_string());
    } else {
        lines.push("  RESULT  NONDETERMINISTIC".to_string());
    }
    lines
}

/// Exit code: 0 only when every replay agreed, at least one verdict was produced and `--expect` (if any) matched.
pub fn exit_code(out: &ReplayOutcome) -> i32 {
    if out.deterministic && out.expect_ok.unwrap_or(true) {
        0
    } else {
        1
    }
}

/// Load the trace and the artefacts, decide the mode, replay.
pub fn replay_paths(
    trace: &Path,
    envelope: &Path,
    calibration: Option<&Path>,
    mode: Option<FuseMode>,
    repeat: u32,
) -> anyhow::Result<ReplayOutcome> {
    let (meta, lines) = read_trace(trace).map_err(|e| anyhow::anyhow!("{}: {e}", trace.display()))?;
    let mode = match mode {
        Some(m) => m,
        None => match meta.get("mode").and_then(|m| m.as_str()) {
            Some(m) => parse_mode(m)?,
            None => anyhow::bail!("{}: #meta carries no mode; pass --mode", trace.display()),
        },
    };
    let toml_text =
        std::fs::read_to_string(envelope).map_err(|e| anyhow::anyhow!("{}: {e}", envelope.display()))?;
    let (env, _) = load_envelope(envelope)?;
    if let Some(d) = meta.get("envelope_digest").and_then(|m| m.as_str()) {
        if d != env.digest_hex() {
            eprintln!(
                "lictor replay: warning: trace #meta envelope_digest {} != loaded {} (hello will be refused)",
                sanitize(d),
                env.digest_hex()
            );
        }
    }
    let calibration = calibration.map(load_calibration).transpose()?;
    let inp = ReplayInput {
        lines: &lines,
        envelope: &env,
        toml_text: &toml_text,
        calibration: calibration.as_ref(),
        mode,
        git: LICTOR_GIT.to_string(),
        sha256: binary_sha256().unwrap_or_else(|_| lictor_receipt::ZERO_HASH.to_string()),
    };
    replay(&inp, repeat)
}

pub fn run(a: Args, json: bool) -> anyhow::Result<i32> {
    let Args { trace, envelope, calibration, mode, repeat, expect } = a;
    let mode = mode.as_deref().map(parse_mode).transpose()?;
    let mut out = match replay_paths(&trace, &envelope, calibration.as_deref(), mode, repeat) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("lictor replay: {e:#}");
            return Ok(2);
        }
    };
    if let Some(e) = expect {
        let e = e.trim().to_lowercase();
        out.expect_ok = Some(e == out.verdict_chain);
        out.expect = Some(e);
    }
    if json {
        json_out(&out)?;
    } else {
        print_lines(&render(&out));
    }
    Ok(exit_code(&out))
}
