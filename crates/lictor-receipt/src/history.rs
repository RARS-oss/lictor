// SPDX-License-Identifier: MIT
//! sbx run-memory lineage (sbx `feedback/history.rs`) -> `<dir>/history.jsonl`, capped at 200 lines.
//!
//! One line per episode; when the cap is exceeded the file is rewritten with the newest 200. `summarize`
//! reads the last `n` records and reports two loop signals: `tail_streak` (the most recent first-trip reason and
//! how many of the last n share it, when at least 2 do) and `identical_run` (how many consecutive most-recent
//! episodes carry the same (arm, trips) signature). std + serde only.

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

/// The cap on `history.jsonl`; older lines are dropped on the write that exceeds it.
pub const HISTORY_CAP: usize = 200;
pub const HISTORY_FILE: &str = "history.jsonl";

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeRecord {
    pub ts: String,
    pub run_id: String,
    pub arm_id: String,
    pub seed: u64,
    pub success: bool,
    pub first_trip_reason: Option<String>,
    pub trips: Vec<String>,
    pub stopped: bool,
    pub escalated: bool,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LoopSignals {
    pub tail_streak: Option<(String, u32)>,
    pub identical_run: u32,
    pub last_n: u32,
}

fn read_records(path: &Path) -> std::io::Result<Vec<EpisodeRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for line in BufReader::new(File::open(path)?).lines() {
        let line = line?;
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        // A corrupt line (a partial write) is skipped rather than poisoning the history.
        if let Ok(r) = serde_json::from_str::<EpisodeRecord>(t) {
            out.push(r);
        }
    }
    Ok(out)
}

/// True when the file is empty or its last byte is a line feed.
fn ends_with_newline(path: &Path) -> std::io::Result<bool> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = File::open(path)?;
    let len = f.metadata()?.len();
    if len == 0 {
        return Ok(true);
    }
    f.seek(SeekFrom::Start(len - 1))?;
    let mut last = [0u8; 1];
    f.read_exact(&mut last)?;
    Ok(last[0] == b'\n')
}

fn line_of(r: &EpisodeRecord) -> std::io::Result<String> {
    serde_json::to_string(r).map_err(std::io::Error::other)
}

pub fn record_episode(dir: &Path, r: &EpisodeRecord) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = dir.join(HISTORY_FILE);
    let existing = read_records(&path)?;
    if existing.len() + 1 > HISTORY_CAP {
        // Rewrite with the newest HISTORY_CAP records (including this one).
        let keep_from = existing.len() + 1 - HISTORY_CAP;
        let tmp = dir.join(format!("{HISTORY_FILE}.tmp"));
        {
            let mut f = File::create(&tmp)?;
            for old in &existing[keep_from..] {
                f.write_all(line_of(old)?.as_bytes())?;
                f.write_all(b"\n")?;
            }
            f.write_all(line_of(r)?.as_bytes())?;
            f.write_all(b"\n")?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        return Ok(());
    }
    let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
    if !ends_with_newline(&path)? {
        // A torn previous write must not glue this record onto the partial line.
        f.write_all(b"\n")?;
    }
    f.write_all(line_of(r)?.as_bytes())?;
    f.write_all(b"\n")?;
    f.sync_all()?;
    Ok(())
}

/// Loop signals over the last `n` records (a missing file yields empty signals).
pub fn summarize(dir: &Path, n: usize) -> std::io::Result<LoopSignals> {
    let all = read_records(&dir.join(HISTORY_FILE))?;
    let start = all.len().saturating_sub(n);
    let tail = &all[start..];
    let last_n = tail.len() as u32;
    let Some(last) = tail.last() else {
        return Ok(LoopSignals { tail_streak: None, identical_run: 0, last_n });
    };
    let tail_streak = last.first_trip_reason.as_ref().and_then(|reason| {
        let count = tail.iter().filter(|r| r.first_trip_reason.as_ref() == Some(reason)).count() as u32;
        (count >= 2).then(|| (reason.clone(), count))
    });
    let identical_run =
        tail.iter().rev().take_while(|r| r.arm_id == last.arm_id && r.trips == last.trips).count() as u32;
    Ok(LoopSignals { tail_streak, identical_run, last_n })
}
