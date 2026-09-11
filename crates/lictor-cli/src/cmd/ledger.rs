// SPDX-License-Identifier: MIT
//! `lictor ledger verify <F.jsonl> | append --ledger <F.jsonl> --receipt <R.json>`.
//!
//! `verify` recomputes the whole chain and prints the frozen summary line (exit 1 on a break). `append`
//! refuses a receipt that does not verify on its own (an unverifiable receipt in a ledger is worthless) and
//! otherwise appends one canonical line under the single-writer contract of `append_ledger`.

use std::path::{Path, PathBuf};

use clap::Subcommand;
use lictor_receipt::{
    append_ledger, read_ledger, verify, verify_ledger, LedgerEntry, SignedReceipt, LEDGER_SCHEMA,
};

use crate::render::{json_out, print_lines, sanitize, short, status};

#[derive(clap::Args)]
pub struct Args {
    #[command(subcommand)]
    pub cmd: Sub,
}

#[derive(Subcommand)]
pub enum Sub {
    /// Verify the hash chain and print the summary line
    Verify {
        #[arg(value_name = "F.jsonl")]
        file: PathBuf,
    },
    /// Append a signed receipt to a ledger (single writer)
    Append {
        #[arg(long, value_name = "F.jsonl")]
        ledger: PathBuf,
        #[arg(long, value_name = "R.json")]
        receipt: PathBuf,
    },
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct VerifyOut {
    pub chain_ok: bool,
    pub break_at: Option<u32>,
    pub episodes: u32,
    pub successes: u32,
    pub stops: u32,
    pub escalations: u32,
    pub trips: u32,
    pub fuse_ok: u32,
    pub head: String,
    pub run_id: String,
    pub arm_id: String,
    pub lines: Vec<String>,
}

/// The header line of a ledger (run/arm), if readable.
fn ledger_header(p: &Path) -> Option<serde_json::Value> {
    let text = std::fs::read_to_string(p).ok()?;
    let first = text.lines().find(|l| !l.trim().is_empty())?;
    let v: serde_json::Value = serde_json::from_str(first).ok()?;
    (v.get("schema").and_then(|s| s.as_str()) == Some(LEDGER_SCHEMA)).then_some(v)
}

/// Read, verify and render one ledger.
pub fn verify_file(p: &Path) -> anyhow::Result<VerifyOut> {
    let entries: Vec<LedgerEntry> = read_ledger(p).map_err(|e| anyhow::anyhow!("{e}"))?;
    let rep = verify_ledger(&entries);
    let header = ledger_header(p);
    let run_id =
        header.as_ref().and_then(|h| h.get("run_id")).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let arm_id =
        header.as_ref().and_then(|h| h.get("arm_id")).and_then(|v| v.as_str()).unwrap_or("").to_string();
    let fuse_ok = entries.iter().filter(|e| e.fuse_ok).count() as u32;
    let mut lines = Vec::new();
    if rep.chain_ok {
        let pct =
            if rep.episodes == 0 { 0.0 } else { 100.0 * f64::from(rep.successes) / f64::from(rep.episodes) };
        lines.push(format!(
            "  episodes {}   chain ok    success {}/{} ({:.1}%)   stops {}   escalated {}   fuse_ok {}/{}",
            rep.episodes, rep.successes, rep.episodes, pct, rep.stops, rep.escalations, fuse_ok, rep.episodes
        ));
        lines.push(format!("  head {}   run {}   arm {}", rep.head, sanitize(&run_id), sanitize(&arm_id)));
        let foreign = entries.iter().filter(|e| e.run_id != run_id || e.arm_id != arm_id).count();
        if !run_id.is_empty() && foreign > 0 {
            lines.push(format!("  warning: {foreign} entries name a different run/arm than the header"));
        }
    } else {
        let at = rep.break_at.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string());
        lines.push(format!(
            "  ledger BROKEN at seq={at} -- entries are missing or edited; this curve point cannot be trusted"
        ));
    }
    Ok(VerifyOut {
        chain_ok: rep.chain_ok,
        break_at: rep.break_at,
        episodes: rep.episodes,
        successes: rep.successes,
        stops: rep.stops,
        escalations: rep.escalations,
        trips: rep.trips,
        fuse_ok,
        head: rep.head,
        run_id,
        arm_id,
        lines,
    })
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct AppendOut {
    pub seq: u32,
    pub hash: String,
    pub prev: String,
    pub receipt_digest: String,
    pub ledger: String,
}

pub fn run(a: Args, json: bool) -> anyhow::Result<i32> {
    match a.cmd {
        Sub::Verify { file } => {
            let out = match verify_file(&file) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("lictor ledger: {e:#}");
                    return Ok(1);
                }
            };
            if json {
                json_out(&out)?;
            } else {
                print_lines(&out.lines);
            }
            Ok(if out.chain_ok { 0 } else { 1 })
        }
        Sub::Append { ledger, receipt } => {
            let text = std::fs::read_to_string(&receipt)
                .map_err(|e| anyhow::anyhow!("{}: {e}", receipt.display()))?;
            let sr: SignedReceipt =
                serde_json::from_str(&text).map_err(|e| anyhow::anyhow!("{}: {e}", receipt.display()))?;
            let rep = verify(&sr, None);
            if !rep.intact() {
                eprintln!(
                    "lictor ledger: refusing to append {}: the receipt does not verify ({})",
                    receipt.display(),
                    rep.notes.join("; ")
                );
                return Ok(1);
            }
            let entry =
                append_ledger(&ledger, &sr).map_err(|e| anyhow::anyhow!("{}: {e}", ledger.display()))?;
            let out = AppendOut {
                seq: entry.seq,
                hash: entry.hash.clone(),
                prev: entry.prev.clone(),
                receipt_digest: entry.receipt_digest.clone(),
                ledger: ledger.to_string_lossy().replace('\\', "/"),
            };
            if json {
                json_out(&out)?;
            } else {
                print_lines(&[
                    status("appended", "ok", &format!("seq={} hash={}", out.seq, out.hash)),
                    status(
                        "receipt",
                        "ok",
                        &format!("digest {} (key {})", short(&out.receipt_digest), short(&sr.pubkey)),
                    ),
                ]);
            }
            Ok(0)
        }
    }
}
