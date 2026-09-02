// SPDX-License-Identifier: MIT
//! `lictor serve`: NDJSON server on stdin/stdout (lictor-wire/v1). Exit 0 on bye/EOF, 3 on internal error; exit 2 at
//! startup when the calibration's embodiment_digest != the envelope's (or any other startup refusal). Without --key
//! an ephemeral key is generated and every receipt carries fuse_ok=false + note "ephemeral signing key" (lictor
//! curve refuses such arms).
//!
//! Cheap to start (one envelope parse, one compile, one sha256 of the binary): the harness runs one serve child per
//! env slot, 32 at a time. Every artefact path derives from `--out` + the `episode_begin` bindings, so instances
//! never collide. stdout carries response lines ONLY; diagnostics go to stderr.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use lictor_core::{FuseMode, SafetyEnvelope, TripMask};
use lictor_runtime::codec::{peek_id, read_request, write_response};
use lictor_runtime::session::{default_latency_label, trace_path, CalibrationLoaded, Session, SessionConfig};
use lictor_runtime::trace::TraceWriter;
use lictor_runtime::wire::{Request, Response};

use crate::cmd::version::LICTOR_GIT;

#[derive(clap::Args)]
pub struct Args {
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// calibration.json (Tier 1); refused when its embodiment_digest differs from the envelope's
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// observe | enforce
    #[arg(long, value_parser = ["observe", "enforce"], default_value = "observe")]
    pub mode: String,
    /// Ed25519 seed file (64 hex); default $LICTOR_KEYS/key.hex if it exists, else an ephemeral key
    #[arg(long, value_name = "F.hex")]
    pub key: Option<PathBuf>,
    /// Output directory for receipts, ticks, timing and the ledger
    #[arg(long, value_name = "DIR")]
    pub out: Option<PathBuf>,
    /// Write every request line verbatim to this trace file
    #[arg(long, value_name = "F.ndjson")]
    pub trace: Option<PathBuf>,
    /// Which tick events to embed in the receipt
    #[arg(long, value_parser = ["tail32", "all", "none"], default_value = "tail32")]
    pub ticks: String,
    /// Comma-separated Tier-0 trip names to arm (overrides the envelope's tier0_enabled)
    #[arg(long, value_name = "csv")]
    pub tier0: Option<String>,
    /// Disarm Tier 1 even when a calibration is given
    #[arg(long)]
    pub no_tier1: bool,
    /// hold (latch Fault) | abort (exit 3 on the first fatal error; debugging aid only)
    #[arg(long, value_parser = ["hold", "abort"], default_value = "hold")]
    pub on_fault: String,
    /// Latency label recorded in receipts (default: default_latency_label())
    #[arg(long, value_name = "S")]
    pub latency_label: Option<String>,
}

/// sha256 (hex) of the running binary, read from `std::env::current_exe()`.
pub fn binary_sha256() -> anyhow::Result<String> {
    use sha2::Digest;
    let exe = std::env::current_exe()?;
    let bytes = std::fs::read(&exe)?;
    Ok(hex::encode(sha2::Sha256::digest(&bytes)))
}

pub fn parse_mode(s: &str) -> anyhow::Result<FuseMode> {
    match s {
        "observe" => Ok(FuseMode::Observe),
        "enforce" => Ok(FuseMode::Enforce),
        other => anyhow::bail!("mode must be observe or enforce, got {other:?}"),
    }
}

/// Parse `--tier0 workspace,speed,...` into a TripMask; only the armable Tier-0 names are accepted.
pub fn parse_tier0(csv: &str) -> anyhow::Result<u32> {
    let mut m = 0u32;
    for name in csv.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        let bit = TripMask::from_name(name)
            .ok_or_else(|| anyhow::anyhow!("--tier0: unknown trip name {name:?}"))?;
        if bit & (TripMask::TIER0_SOFT | TripMask::BRAKE) == 0 {
            anyhow::bail!("--tier0: {name:?} is not an armable Tier-0 check");
        }
        m |= bit;
    }
    Ok(m)
}

/// Load the envelope TOML and return it with the sha256 of the file bytes (bound as `lictor:envelope`).
pub fn load_envelope(p: &Path) -> anyhow::Result<(SafetyEnvelope, String)> {
    let text = std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    let env = SafetyEnvelope::from_toml(&text).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    Ok((env, lictor_canon::sha256_hex(text.as_bytes())))
}

/// Load calibration.json through lictor-calib into the runtime's `CalibrationLoaded`.
pub fn load_calibration(p: &Path) -> anyhow::Result<CalibrationLoaded> {
    let f = lictor_calib::file::CalibrationFile::load(p)?;
    f.loaded(p)
}

/// `--key`, else `$LICTOR_KEYS/key.hex` (or `$HOME/.lictor/key.hex`) when it exists, else None (ephemeral).
pub fn resolve_key(key: Option<&Path>) -> anyhow::Result<Option<[u8; 32]>> {
    if let Some(p) = key {
        return Ok(Some(lictor_receipt::load_seed(p)?));
    }
    let default = lictor_receipt::default_key_dir().join("key.hex");
    if default.is_file() {
        return Ok(Some(lictor_receipt::load_seed(&default)?));
    }
    Ok(None)
}

fn run_id_of(req: &Request) -> Option<(String, String, u32)> {
    match req {
        Request::EpisodeBegin(b) => Some((b.run.run_id.clone(), b.run.arm_id.clone(), b.run.episode_index)),
        _ => None,
    }
}

pub fn run(a: Args, _json: bool) -> anyhow::Result<i32> {
    let Args {
        envelope,
        calibration,
        mode,
        key,
        out,
        trace,
        ticks,
        tier0,
        no_tier1,
        on_fault,
        latency_label,
    } = a;
    let (envelope, envelope_toml_sha) = match load_envelope(&envelope) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("lictor serve: {e:#}");
            return Ok(2);
        }
    };
    let calibration = match calibration.as_deref().map(load_calibration).transpose() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("lictor serve: calibration: {e:#}");
            return Ok(2);
        }
    };
    let tier0_override = match tier0.as_deref().map(parse_tier0).transpose() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("lictor serve: {e:#}");
            return Ok(2);
        }
    };
    let key = match resolve_key(key.as_deref()) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("lictor serve: key: {e:#}");
            return Ok(2);
        }
    };
    if key.is_none() {
        eprintln!(
            "lictor serve: warning: no --key and no {}: signing with an EPHEMERAL key; every receipt carries \
             fuse_ok=false and the note \"ephemeral signing key\" (lictor curve refuses such arms)",
            lictor_receipt::default_key_dir().join("key.hex").display()
        );
    }
    let cfg = SessionConfig {
        envelope,
        envelope_toml_sha,
        calibration,
        mode: parse_mode(&mode)?,
        tier0_override,
        tier1: !no_tier1,
        ticks_policy: ticks,
        key,
        out_dir: out.clone(),
        trace: trace.clone(),
        lictor_git: LICTOR_GIT.to_string(),
        lictor_sha256: binary_sha256()?,
        latency_label: latency_label.unwrap_or_else(default_latency_label),
    };
    let mut session = match Session::new(cfg) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lictor serve: refused: {e:#}");
            return Ok(2);
        }
    };
    let abort_on_fault = on_fault == "abort";
    let meta = session.trace_meta();
    let mut process_trace = match &trace {
        Some(p) => Some(TraceWriter::open(p, &meta)?),
        None => None,
    };
    let mut episode_trace: Option<TraceWriter> = None;
    let mut hello_raw: Option<String> = None;

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut reader = stdin.lock();
    let mut writer = std::io::BufWriter::new(stdout.lock());
    let mut buf = String::new();
    loop {
        let io_t0 = Instant::now();
        let parsed = match read_request(&mut reader, &mut buf) {
            Ok(Some(req)) => Ok(req),
            Ok(None) => return Ok(0),
            Err(e) if e.kind() == std::io::ErrorKind::InvalidData => Err(e.to_string()),
            Err(e) => return Err(e.into()),
        };
        let resp = match parsed {
            Ok(req) => {
                if let Some(w) = process_trace.as_mut() {
                    w.line(&buf)?;
                }
                if let Request::Hello(_) = &req {
                    hello_raw = Some(buf.clone());
                }
                if let (Some(out_dir), Some((run_id, arm_id, idx))) = (&out, run_id_of(&req)) {
                    let p = trace_path(out_dir, &run_id, &arm_id, idx);
                    let mut w = TraceWriter::open(&p, &meta)?;
                    if let Some(h) = &hello_raw {
                        w.line(h)?;
                    }
                    episode_trace = Some(w);
                }
                if let Some(w) = episode_trace.as_mut() {
                    if !matches!(req, Request::Hello(_)) {
                        w.line(&buf)?;
                    }
                }
                let is_end = matches!(req, Request::EpisodeEnd(_));
                let is_bye = matches!(req, Request::Bye(_));
                let resp = session.handle(req);
                if is_end {
                    episode_trace = None;
                }
                write_response(&mut writer, &resp)?;
                if let Response::Verdict(v) = &resp {
                    let ns = u64::try_from(io_t0.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    session.note_io_ns(v.seq, ns);
                }
                if is_bye {
                    writer.flush()?;
                    return Ok(0);
                }
                resp
            }
            Err(msg) => {
                let resp = session.handle_bad_line(peek_id(&buf), &msg);
                write_response(&mut writer, &resp)?;
                resp
            }
        };
        if let Response::Error(e) = &resp {
            eprintln!("lictor serve: error id={} code={} fatal={}: {}", e.id, e.code, e.fatal, e.message);
            if e.fatal && abort_on_fault {
                writer.flush()?;
                return Ok(3);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier0_csv() {
        assert_eq!(parse_tier0("workspace,speed").unwrap(), TripMask::WORKSPACE | TripMask::SPEED);
        assert_eq!(parse_tier0(" brake , reach ").unwrap(), TripMask::BRAKE | TripMask::REACH);
        assert!(parse_tier0("schema").is_err());
        assert!(parse_tier0("nope").is_err());
        assert_eq!(parse_tier0("").unwrap(), 0);
    }

    #[test]
    fn mode_parse() {
        assert_eq!(parse_mode("enforce").unwrap(), FuseMode::Enforce);
        assert!(parse_mode("x").is_err());
    }
}
