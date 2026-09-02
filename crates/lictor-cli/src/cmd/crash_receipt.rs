// SPDX-License-Identifier: MIT
//! `lictor crash-receipt`: host-side accounting for a serve child that died before episode_end: writes a signed
//! receipt with ended_by="fuse_crash", success=false, terminal_state=fault, fuse_ok=false and appends the ledger. Exit 0.
//!
//! `--budget` is JSON. It may be the full `BudgetBinding` or only the host part the harness sent in
//! `episode_begin.budget` (`delay_steps`, `tick_ms`, `exec_mode`, `stitch`, `on_escalate`); the fuse-derived fields
//! (`mode`, `tier0_armed`, `tier1_armed`, `gate`, `alpha_num`, `alpha_den`, `kn`) are filled from the envelope and
//! the calibration exactly as `lictor serve` would have filled them, so the crash receipt of an arm carries the same
//! budget as its normal receipts.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lictor_core::TripMask;
use lictor_receipt::{BudgetBinding, RunBinding};
use lictor_runtime::episode::write_crash_episode;
use lictor_runtime::session::{default_latency_label, SessionConfig};

use crate::cmd::serve::{binary_sha256, load_calibration, load_envelope, parse_mode, resolve_key};
use crate::cmd::version::LICTOR_GIT;

/// The `client` string a host-written crash receipt carries (no `hello` was seen by this process).
pub const CRASH_CLIENT: &str = "lictor crash-receipt";

#[derive(clap::Args)]
pub struct Args {
    /// Envelope TOML
    #[arg(long, value_name = "F.toml")]
    pub envelope: PathBuf,
    /// calibration.json
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// observe | enforce
    #[arg(long, value_parser = ["observe", "enforce"])]
    pub mode: String,
    #[arg(long, value_name = "S")]
    pub run_id: String,
    #[arg(long, value_name = "S")]
    pub arm_id: String,
    #[arg(long, value_name = "N")]
    pub episode_index: u32,
    #[arg(long, value_name = "N")]
    pub seed: u64,
    #[arg(long, value_name = "S")]
    pub seed_pool: String,
    #[arg(long, value_name = "hex")]
    pub init_state_digest: String,
    /// BudgetBinding as JSON
    #[arg(long, value_name = "json")]
    pub budget: String,
    /// Ed25519 seed file (64 hex)
    #[arg(long, value_name = "F.hex")]
    pub key: Option<PathBuf>,
    /// Output directory (the same --out the serve child used)
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Extra note recorded in fuse_notes
    #[arg(long, value_name = "S")]
    pub note: Option<String>,
}

/// Merge the host's budget JSON with the fuse-derived fields (see the module doc).
pub fn budget_from_json(cfg: &SessionConfig, budget_json: &str) -> anyhow::Result<BudgetBinding> {
    let mut v: serde_json::Value =
        serde_json::from_str(budget_json).map_err(|e| anyhow::anyhow!("--budget is not JSON: {e}"))?;
    let obj = v.as_object_mut().ok_or_else(|| anyhow::anyhow!("--budget must be a JSON object"))?;
    let calib = if cfg.tier1 { cfg.calibration.as_ref() } else { None };
    let tier0: Vec<String> = match cfg.tier0_override {
        Some(m) => TripMask::names(m).map(str::to_string).collect(),
        None => TripMask::names(cfg.envelope.tier0_mask()?).map(str::to_string).collect(),
    };
    let gate: Vec<String> = match calib {
        Some(c) => c.c.gate.names(),
        None => cfg.envelope.gate.clone(),
    };
    let (alpha_num, alpha_den) = match calib {
        Some(c) => (c.c.alpha_num, c.c.alpha_den),
        None => (0, 1),
    };
    let derived: [(&str, serde_json::Value); 7] = [
        ("mode", serde_json::to_value(cfg.mode)?),
        ("tier0_armed", serde_json::to_value(&tier0)?),
        ("tier1_armed", serde_json::Value::Bool(calib.is_some())),
        ("gate", serde_json::to_value(&gate)?),
        ("alpha_num", serde_json::Value::from(alpha_num)),
        ("alpha_den", serde_json::Value::from(alpha_den)),
        ("kn", serde_json::to_value([cfg.envelope.hysteresis.k, cfg.envelope.hysteresis.n])?),
    ];
    for (k, val) in derived {
        obj.insert(k.to_string(), val);
    }
    let b: BudgetBinding = serde_json::from_value(v).map_err(|e| anyhow::anyhow!("--budget: {e}"))?;
    Ok(b)
}

pub fn run(a: Args, json: bool) -> anyhow::Result<i32> {
    let Args {
        envelope,
        calibration,
        mode,
        run_id,
        arm_id,
        episode_index,
        seed,
        seed_pool,
        init_state_digest,
        budget,
        key,
        out,
        note,
    } = a;
    let (envelope, envelope_toml_sha) = load_envelope(&envelope)?;
    let calibration = calibration.as_deref().map(load_calibration).transpose()?;
    let cfg = SessionConfig {
        envelope,
        envelope_toml_sha,
        calibration,
        mode: parse_mode(&mode)?,
        tier0_override: None,
        tier1: true,
        ticks_policy: "tail32".to_string(),
        key: resolve_key(key.as_deref())?,
        out_dir: Some(out),
        trace: None,
        lictor_git: LICTOR_GIT.to_string(),
        lictor_sha256: binary_sha256()?,
        latency_label: default_latency_label(),
    };
    let budget = budget_from_json(&cfg, &budget)?;
    let run = RunBinding {
        run_id,
        arm_id,
        episode_index,
        seed,
        seed_pool,
        init_state_digest,
        env: BTreeMap::new(),
        policy: BTreeMap::new(),
        host: BTreeMap::new(),
    };
    let paths = lictor_runtime::episode::episode_paths(
        cfg.out_dir.as_deref().expect("set above"),
        &run.run_id,
        &run.arm_id,
        run.episode_index,
    );
    let entry = write_crash_episode(&cfg, run, budget, CRASH_CLIENT, note.as_deref().unwrap_or(""))?;
    let receipt = paths.receipt.to_string_lossy().replace('\\', "/");
    if json {
        let v = serde_json::json!({
            "ledger_seq": entry.seq,
            "ledger_head": entry.hash,
            "receipt": receipt,
            "receipt_digest": entry.receipt_digest,
            "ended_by": "fuse_crash",
            "fuse_ok": false,
        });
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!("ledger seq     {}", entry.seq);
        println!("receipt        {receipt}");
        println!("ended_by       fuse_crash (host-written crash receipt; fuse_ok=false)");
    }
    Ok(0)
}
