// SPDX-License-Identifier: MIT
//! A hash chain per arm: dropping, reordering or editing an entry WITHOUT the signing key breaks the chain at a reported seq.
//! It does not defend against the key-holder (who can rebuild and re-sign); tail truncation of a DECLARED pool is detected
//! by `lictor curve`.
//!
//! Vendored from RARS-oss/bulla @ 173e1cd65fb43353a9752f95077937aa4bb0f8e2, crates/bulla-core/src/lib.rs
//! (`LedgerEntry`, `ledger_entry`, `verify_ledger`); renamed per docs/ANALYSIS.md sec 5; canonical bytes replaced by
//! lictor-canon JCS: `hash = sha256(canon(entry with hash = ""))` instead of sha256 over a serde-ordered core struct.
//! `read_ledger` / `append_ledger` (the header line, fsync) are lictor additions.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use lictor_canon::{canon_of, sha256_hex};

use crate::sign::{ReceiptError, SignedReceipt};
use crate::{LEDGER_SCHEMA, ZERO_HASH};

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry {
    pub seq: u32,
    pub run_id: String,
    pub arm_id: String,
    pub episode_index: u32,
    pub seed: u64,
    pub init_state_digest: String,
    pub receipt_digest: String,
    pub verdict_chain_head: String,
    pub success: bool,
    pub fuse_ok: bool,
    pub tripped: bool,
    pub stopped: bool,
    pub escalated: bool,
    pub prev: String,
    pub hash: String,
}

impl LedgerEntry {
    /// sha256(canon(self with hash = "")). Entries are float-free and key-clean by construction.
    pub fn recompute_hash(&self) -> String {
        let mut e = self.clone();
        e.hash = String::new();
        sha256_hex(&canon_of(&e).expect("ledger entries are float-free and key-clean by construction"))
    }
}

/// The first line of a ledger file.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LedgerHeader {
    pub schema: String,
    pub run_id: String,
    pub arm_id: String,
    pub genesis: String,
}

pub fn ledger_entry(prev: &str, seq: u32, sr: &SignedReceipt) -> LedgerEntry {
    let b = &sr.body;
    let mut e = LedgerEntry {
        seq,
        run_id: b.run.run_id.clone(),
        arm_id: b.run.arm_id.clone(),
        episode_index: b.run.episode_index,
        seed: b.run.seed,
        init_state_digest: b.run.init_state_digest.clone(),
        receipt_digest: sr.body_digest.clone(),
        verdict_chain_head: b.verdict_chain_head.clone(),
        success: b.outcome.success,
        fuse_ok: b.fuse_ok,
        tripped: b.counts.first_trip_tick.is_some(),
        stopped: b.counts.first_stop_tick.is_some(),
        escalated: b.counts.escalations > 0 || b.counts.handoff_tick.is_some(),
        prev: prev.to_string(),
        hash: String::new(),
    };
    e.hash = e.recompute_hash();
    e
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LedgerReport {
    pub chain_ok: bool,
    pub break_at: Option<u32>,
    pub episodes: u32,
    pub successes: u32,
    pub trips: u32,
    pub stops: u32,
    pub escalations: u32,
    pub head: String,
}

/// Linkage from the genesis, seq continuity from 0 and every hash recomputed; `break_at` is the seq of the first
/// bad entry (the expected seq when the entry's own seq is wrong, i.e. the position of a dropped entry).
pub fn verify_ledger(entries: &[LedgerEntry]) -> LedgerReport {
    let mut prev = ZERO_HASH.to_string();
    let mut break_at = None;
    let (mut successes, mut trips, mut stops, mut escalations) = (0u32, 0u32, 0u32, 0u32);
    for (i, e) in entries.iter().enumerate() {
        let expected_seq = i as u32;
        if (e.seq != expected_seq || e.prev != prev || e.hash != e.recompute_hash()) && break_at.is_none() {
            break_at = Some(if e.seq == expected_seq { e.seq } else { expected_seq });
        }
        successes += u32::from(e.success);
        trips += u32::from(e.tripped);
        stops += u32::from(e.stopped);
        escalations += u32::from(e.escalated);
        prev = e.hash.clone();
    }
    LedgerReport {
        chain_ok: break_at.is_none(),
        break_at,
        episodes: entries.len() as u32,
        successes,
        trips,
        stops,
        escalations,
        head: prev,
    }
}

/// Parse a ledger file: the header line (any line carrying a `schema` key; normally the first) is skipped, blank
/// lines are ignored, every other line must be a `LedgerEntry`.
pub fn read_ledger(path: &Path) -> Result<Vec<LedgerEntry>, ReceiptError> {
    let f = File::open(path).map_err(|e| ReceiptError::Io(format!("{}: {e}", path.display())))?;
    let mut out = Vec::new();
    for (i, line) in BufReader::new(f).lines().enumerate() {
        let line = line.map_err(|e| ReceiptError::Io(format!("{}:{}: {e}", path.display(), i + 1)))?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        let v: serde_json::Value = serde_json::from_str(t)
            .map_err(|e| ReceiptError::Io(format!("{}:{}: not JSON: {e}", path.display(), i + 1)))?;
        if v.get("schema").is_some() {
            if v.get("schema").and_then(|s| s.as_str()) != Some(LEDGER_SCHEMA) {
                return Err(ReceiptError::Io(format!(
                    "{}:{}: not a {LEDGER_SCHEMA} header",
                    path.display(),
                    i + 1
                )));
            }
            continue;
        }
        let e: LedgerEntry = serde_json::from_value(v).map_err(|e| {
            ReceiptError::Io(format!("{}:{}: not a ledger entry: {e}", path.display(), i + 1))
        })?;
        out.push(e);
    }
    Ok(out)
}

/// Reads the current head, appends one canonical line (header first when the file is created), fsyncs.
/// Single writer by contract (one `lictor serve` per arm).
pub fn append_ledger(path: &Path, sr: &SignedReceipt) -> Result<LedgerEntry, ReceiptError> {
    let existing = if path.exists() { read_ledger(path)? } else { Vec::new() };
    let head = existing.last().map(|e| e.hash.clone()).unwrap_or_else(|| ZERO_HASH.to_string());
    let entry = ledger_entry(&head, existing.len() as u32, sr);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let fresh = !path.exists();
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    if fresh {
        let header = LedgerHeader {
            schema: LEDGER_SCHEMA.to_string(),
            run_id: sr.body.run.run_id.clone(),
            arm_id: sr.body.run.arm_id.clone(),
            genesis: ZERO_HASH.to_string(),
        };
        f.write_all(&canon_of(&header)?)?;
        f.write_all(b"\n")?;
    }
    f.write_all(&canon_of(&entry)?)?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(entry)
}
