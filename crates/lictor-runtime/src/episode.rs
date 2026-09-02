// SPDX-License-Identifier: MIT
//! Episode artefacts on disk: receipt (pretty JSON, sorted keys), ticks and timing files (one canonical event per
//! line under a schema header), and the ledger append -- always LAST, so a ledger entry never precedes the
//! receipt it names. Also the host-side crash receipt (`lictor crash-receipt` -> `write_crash_episode`).
//!
//! Layout under `out_dir` (FILE FORMATS, results tree):
//! `<out>/<run_id>/<arm_id>/{receipts/NNNNNN.json, ticks/NNNNNN.jsonl, timing/NNNNNN.jsonl, ledger.jsonl}`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use lictor_canon::{canon, canon_of, floatify, F64Array, F64Hex};
use lictor_core::{FuseState, NFEAT};
use lictor_receipt::{
    append_ledger, evaluate_fuse, read_ledger, sign, BudgetBinding, EpisodeOutcome, LatencySummary,
    LedgerEntry, ReceiptBody, RunBinding, SignedReceipt, TickEvent, TimingEvent, VerdictCounts,
    RECEIPT_SCHEMA, TICKS_SCHEMA, ZERO_HASH,
};

use crate::session::SessionConfig;

/// The note every host-written crash receipt carries (frozen text).
pub const NOTE_CRASH: &str = "fuse process died before episode_end; host-written crash receipt";

pub struct EpisodePaths {
    pub receipt: std::path::PathBuf,
    pub ticks: std::path::PathBuf,
    pub timing: std::path::PathBuf,
    pub ledger: std::path::PathBuf,
}

pub fn episode_paths(
    out_dir: &std::path::Path,
    run_id: &str,
    arm_id: &str,
    episode_index: u32,
) -> EpisodePaths {
    let arm = out_dir.join(run_id).join(arm_id);
    EpisodePaths {
        receipt: arm.join("receipts").join(format!("{episode_index:06}.json")),
        ticks: arm.join("ticks").join(format!("{episode_index:06}.jsonl")),
        timing: arm.join("timing").join(format!("{episode_index:06}.jsonl")),
        ledger: arm.join("ledger.jsonl"),
    }
}

fn ensure_parent(p: &Path) -> std::io::Result<()> {
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Write bytes, fsync, and only then let the caller proceed (the ledger append must never precede the files it names).
fn write_synced(p: &Path, bytes: &[u8]) -> std::io::Result<()> {
    ensure_parent(p)?;
    let mut f = std::fs::File::create(p)?;
    f.write_all(bytes)?;
    f.sync_all()
}

/// Pretty JSON (2-space indent) with sorted keys: `serde_json::Value` objects are `BTreeMap`s, so `to_value`
/// followed by `to_string_pretty` sorts every level.
pub fn pretty_sorted<T: serde::Serialize>(t: &T) -> anyhow::Result<String> {
    let v = serde_json::to_value(t)?;
    let mut s = serde_json::to_string_pretty(&v)?;
    s.push('\n');
    Ok(s)
}

#[derive(serde::Serialize)]
struct TicksHeader<'a> {
    schema: &'a str,
    run_id: &'a str,
    arm_id: &'a str,
    episode_index: u32,
    genesis: &'a str,
}

/// The ticks header line; the timing file uses the same header plus `"stream":"timing"` (docs/receipt-schema.md sec 2).
fn header_line(sr: &SignedReceipt, timing: bool) -> anyhow::Result<Vec<u8>> {
    let b = &sr.body;
    let h = TicksHeader {
        schema: TICKS_SCHEMA,
        run_id: &b.run.run_id,
        arm_id: &b.run.arm_id,
        episode_index: b.run.episode_index,
        genesis: ZERO_HASH,
    };
    let mut v = serde_json::to_value(&h)?;
    if timing {
        v["stream"] = serde_json::Value::String("timing".to_string());
    }
    Ok(canon(&v)?)
}

fn jsonl<T: serde::Serialize>(header: Vec<u8>, events: &[T]) -> anyhow::Result<Vec<u8>> {
    let mut out = header;
    out.push(b'\n');
    for e in events {
        out.extend_from_slice(&canon_of(e)?);
        out.push(b'\n');
    }
    Ok(out)
}

/// Receipt, ticks, timing, then the ledger LAST. Every file is fsynced before the ledger append.
pub fn write_episode(
    paths: &EpisodePaths,
    sr: &SignedReceipt,
    ticks: &[TickEvent],
    timing: &[TimingEvent],
) -> anyhow::Result<LedgerEntry> {
    write_synced(&paths.receipt, pretty_sorted(sr)?.as_bytes())?;
    write_synced(&paths.ticks, &jsonl(header_line(sr, false)?, ticks)?)?;
    write_synced(&paths.timing, &jsonl(header_line(sr, true)?, timing)?)?;
    Ok(append_ledger(&paths.ledger, sr)?)
}

/// Hash of the last ledger entry of this arm, or None when the ledger does not exist yet / is empty.
pub fn ledger_head(ledger: &Path) -> anyhow::Result<Option<String>> {
    if !ledger.exists() {
        return Ok(None);
    }
    Ok(read_ledger(ledger)?.last().map(|e| e.hash.clone()))
}

/// The three fuse-computed `inputs` entries (`lictor:bin`, `lictor:envelope`, `lictor:calibration`).
pub fn fuse_inputs(cfg: &SessionConfig) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    m.insert("lictor:bin".to_string(), cfg.lictor_sha256.clone());
    m.insert("lictor:envelope".to_string(), cfg.envelope_toml_sha.clone());
    if let Some(c) = &cfg.calibration {
        m.insert("lictor:calibration".to_string(), c.file_sha256.clone());
    }
    m
}

/// Zero counts with `terminal_state = Fault` and the frozen array lengths.
pub fn zero_counts() -> VerdictCounts {
    VerdictCounts {
        trips_by_bit: vec![0; 16],
        fired_by_feat: vec![0; NFEAT],
        terminal_state: FuseState::Fault,
        ..VerdictCounts::default()
    }
}

/// Seconds since the Unix epoch (0 when the clock is before it).
pub fn now_epoch() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Host-side crash accounting (`lictor crash-receipt`): when the serve child died before `episode_end`, write a minimal SIGNED receipt
/// with zero counts, `outcome = {steps: 0, success: false, ended_by: "fuse_crash"}`, `terminal_state: fault`, `fuse_ok: false`,
/// `fuse_notes: ["fuse process died before episode_end; host-written crash receipt"]`, empty ticks/timing files, and append the ledger.
pub fn write_crash_episode(
    cfg: &SessionConfig,
    run: RunBinding,
    budget: BudgetBinding,
    client: &str,
    note: &str,
) -> anyhow::Result<LedgerEntry> {
    let out_dir = cfg
        .out_dir
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("crash receipt needs an output directory (--out)"))?;
    let paths = episode_paths(out_dir, &run.run_id, &run.arm_id, run.episode_index);
    let (seed, ephemeral) = match cfg.key {
        Some(k) => (k, false),
        None => (lictor_receipt::keygen(), true),
    };
    let calibration_digest = cfg.calibration.as_ref().map(|c| c.digest.clone());
    let counts = zero_counts();
    let (_, mut notes) = evaluate_fuse(&budget, &counts, calibration_digest.as_deref(), ephemeral);
    notes.push(NOTE_CRASH.to_string());
    if !note.trim().is_empty() {
        notes.push(note.trim().to_string());
    }
    let body = ReceiptBody {
        schema: RECEIPT_SCHEMA.to_string(),
        canonical: lictor_canon::CANONICAL_ID.to_string(),
        created_epoch: now_epoch(),
        lictor_version: lictor_core::LICTOR_VERSION.to_string(),
        lictor_git: cfg.lictor_git.clone(),
        lictor_sha256: cfg.lictor_sha256.clone(),
        client: client.to_string(),
        run,
        budget,
        fault_injection: None,
        envelope: floatify(serde_json::to_value(&cfg.envelope)?),
        envelope_digest: cfg.envelope.digest_hex(),
        calibration_digest,
        inputs: fuse_inputs(cfg),
        counts,
        outcome: EpisodeOutcome {
            steps: 0,
            success: false,
            terminated: false,
            truncated: false,
            max_coverage: F64Hex(f64::NAN),
            final_coverage: F64Hex(f64::NAN),
            reward_sum: F64Hex(f64::NAN),
            ended_by: "fuse_crash".to_string(),
            max_s: F64Hex(f64::NEG_INFINITY),
            max_z: F64Array::from_slice(&[f64::NEG_INFINITY; NFEAT], &[NFEAT as u32]),
        },
        handoffs: Vec::new(),
        verdict_events: 0,
        verdict_chain_head: ZERO_HASH.to_string(),
        timing_events: 0,
        timing_chain_head: ZERO_HASH.to_string(),
        latency: LatencySummary {
            n: 0,
            p50_ns: 0,
            p90_ns: 0,
            p99_ns: 0,
            p999_ns: 0,
            max_ns: 0,
            label: cfg.latency_label.clone(),
        },
        ticks_policy: cfg.ticks_policy.clone(),
        ticks: Vec::new(),
        fuse_ok: false,
        fuse_notes: notes,
        ledger_prev: ledger_head(&paths.ledger)?,
    };
    let sr = sign(body, &seed)?;
    write_episode(&paths, &sr, &[], &[])
}

/// Convenience for callers that only have the receipt path: the sibling paths of `episode_paths`.
pub fn paths_display(paths: &EpisodePaths) -> (String, String) {
    (path_str(&paths.receipt), path_str(&paths.ticks))
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_are_six_digit() {
        let p = episode_paths(Path::new("results"), "run", "arm", 7);
        assert_eq!(path_str(&p.receipt), "results/run/arm/receipts/000007.json");
        assert_eq!(path_str(&p.ticks), "results/run/arm/ticks/000007.jsonl");
        assert_eq!(path_str(&p.timing), "results/run/arm/timing/000007.jsonl");
        assert_eq!(path_str(&p.ledger), "results/run/arm/ledger.jsonl");
    }

    #[test]
    fn pretty_is_sorted() {
        #[derive(serde::Serialize)]
        struct S {
            zeta: u32,
            alpha: u32,
        }
        let s = pretty_sorted(&S { zeta: 1, alpha: 2 }).unwrap();
        assert_eq!(s, "{\n  \"alpha\": 2,\n  \"zeta\": 1\n}\n");
    }
}
