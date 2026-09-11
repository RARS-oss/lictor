// SPDX-License-Identifier: MIT
//! `lictor ack`: sign an operator acknowledgement for a pending handoff (digest or record.json).
//!
//! `--handoff` is either the 64-hex handoff digest or a `HandoffRecord` JSON file, whose digest is RECOMPUTED
//! from the record (sha256 over canon(self with digest/ack/resolved_tick/outcome cleared)) and must equal the
//! digest the file carries -- a record that does not self-verify is refused. The nonce defaults to
//! 1 + the largest nonce recorded for this operator in `default_key_dir()/ack_nonce.json` (a pubkey -> last
//! nonce map, the same shape as the runtime's verifier_nonce.json); the file is updated after signing so the
//! next default is strictly greater. The token is written pretty-printed with sorted keys to `-o` or stdout.

use std::path::{Path, PathBuf};

use lictor_core::AckDecision;
use lictor_receipt::{
    default_key_dir, load_nonces, load_seed, save_nonces, sign_ack, AckToken, HandoffRecord,
};

use crate::render::{glossary, json_out, kv, pretty_sorted, print_lines, sanitize, short};

/// The per-operator nonce memory of `lictor ack`, under the key directory.
pub const NONCE_FILE: &str = "ack_nonce.json";

#[derive(clap::Args)]
pub struct Args {
    /// Handoff digest (hex64) or a HandoffRecord JSON file
    #[arg(long, value_name = "digest|record.json")]
    pub handoff: String,
    /// resume | abort | retune
    #[arg(long, value_parser = ["resume", "abort", "retune"])]
    pub decision: String,
    /// Operator seed file (64 hex)
    #[arg(long, value_name = "operator.hex")]
    pub key: PathBuf,
    /// Free-text note bound into the token
    #[arg(long, value_name = "S")]
    pub note: Option<String>,
    /// Nonce (default: 1 + the largest nonce recorded for this operator)
    #[arg(long, value_name = "N")]
    pub nonce: Option<u64>,
    /// Write the token here (default: stdout)
    #[arg(short, long, value_name = "F.json")]
    pub out: Option<PathBuf>,
}

pub fn parse_decision(s: &str) -> anyhow::Result<AckDecision> {
    match s {
        "resume" => Ok(AckDecision::Resume),
        "abort" => Ok(AckDecision::Abort),
        "retune" => Ok(AckDecision::Retune),
        other => anyhow::bail!("decision must be resume, abort or retune, got {other:?}"),
    }
}

fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The digest to sign and, for a record file, the record itself (for the human summary).
pub fn resolve_handoff(arg: &str) -> anyhow::Result<(String, Option<HandoffRecord>)> {
    if is_hex64(arg) {
        return Ok((arg.to_lowercase(), None));
    }
    let p = Path::new(arg);
    let text = std::fs::read_to_string(p).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    let rec: HandoffRecord = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("{}: not a handoff record: {e}", p.display()))?;
    let recomputed = rec.compute_digest()?;
    if recomputed != rec.digest {
        anyhow::bail!(
            "{}: the record's digest {} does not match the recomputed {}",
            p.display(),
            short(&rec.digest),
            short(&recomputed)
        );
    }
    Ok((recomputed, Some(rec)))
}

/// 1 + the largest nonce stored for `operator` (1 when none).
pub fn next_nonce(nonce_file: &Path, operator: &str) -> anyhow::Result<u64> {
    let m = load_nonces(nonce_file)?;
    Ok(m.get(operator).copied().unwrap_or(0).saturating_add(1))
}

/// Remember `nonce` for `operator` when it is larger than the stored one.
pub fn remember_nonce(nonce_file: &Path, operator: &str, nonce: u64) -> anyhow::Result<()> {
    let mut m = load_nonces(nonce_file)?;
    let cur = m.get(operator).copied().unwrap_or(0);
    if nonce > cur {
        m.insert(operator.to_string(), nonce);
        save_nonces(nonce_file, &m)?;
    }
    Ok(())
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct AckOut {
    pub token: AckToken,
    pub out: Option<String>,
    pub nonce_file: String,
}

pub fn run(a: Args, json: bool) -> anyhow::Result<i32> {
    let Args { handoff, decision, key, note, nonce, out } = a;
    let decision = parse_decision(&decision)?;
    let seed = match load_seed(&key) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lictor ack: {e}");
            return Ok(2);
        }
    };
    let (digest, record) = match resolve_handoff(&handoff) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("lictor ack: {e:#}");
            return Ok(2);
        }
    };
    let operator = lictor_receipt::pubkey_hex(&seed);
    let nonce_file = default_key_dir().join(NONCE_FILE);
    let nonce = match nonce {
        Some(n) => n,
        None => next_nonce(&nonce_file, &operator)?,
    };
    let note = note.unwrap_or_default();
    let token = match sign_ack(&digest, decision, nonce, &note, &seed) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("lictor ack: {e}");
            return Ok(2);
        }
    };
    if let Err(e) = remember_nonce(&nonce_file, &operator, nonce) {
        eprintln!("lictor ack: warning: could not record the nonce in {}: {e}", nonce_file.display());
    }
    let text = pretty_sorted(&token)?;
    if let Some(p) = &out {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        std::fs::write(p, text.as_bytes()).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?;
    }
    let ack_out = AckOut {
        token: token.clone(),
        out: out.as_ref().map(|p| p.to_string_lossy().replace('\\', "/")),
        nonce_file: nonce_file.to_string_lossy().replace('\\', "/"),
    };
    if json {
        json_out(&ack_out)?;
        return Ok(0);
    }
    match &out {
        Some(p) => {
            let mut lines = vec![
                kv("handoff", &short(&digest)),
                kv("decision", &decision_name(decision)),
                kv("operator", &short(&operator)),
                kv("nonce", &nonce.to_string()),
                kv("token", &p.to_string_lossy().replace('\\', "/")),
            ];
            if let Some(r) = &record {
                lines.push(kv(
                    "reason",
                    &format!(
                        "{} tick={} {}/{}#{} -- {}",
                        sanitize(&r.reason_text),
                        r.tick,
                        r.run_id,
                        r.arm_id,
                        r.episode_index,
                        glossary(r.reason)
                    ),
                ));
            }
            if !note.is_empty() {
                lines.push(kv("note", &note));
            }
            print_lines(&lines);
        }
        None => print!("{text}"),
    }
    Ok(0)
}

fn decision_name(d: AckDecision) -> String {
    match d {
        AckDecision::Resume => "resume",
        AckDecision::Abort => "abort",
        AckDecision::Retune => "retune",
    }
    .to_string()
}
