// SPDX-License-Identifier: MIT
//! `lictor verify`: prints the aligned block of the CLI SURFACE; exit 0 intact AND fuse held, 1 otherwise.
//! Scripts use `--json` and test the `intact` field (an honest Observe receipt exits 1).
//!
//! Every check recomputes canonical bytes from the parsed document (never trusted from disk): the signature and
//! body digest (`lictor_receipt::verify`), the envelope digest, the counts identity, the embedded tick tail,
//! and -- with the optional companions -- the full verdict chain (`verify_ticks_file` THEN the head comparison
//! and the ticks-header / embedded-tail cross-checks), the timing chain, the ledger chain (the receipt's digest
//! must appear at its own run/arm/episode), the curve receipt (signature and `ledger_head`) and the calibration
//! file (budget alpha/gate/kn and the self-digest). `intact` is the verdict on the RECORD; `fuse_ok` on what the
//! record says the fuse did -- `intact != fuse_ok`.

use std::path::{Path, PathBuf};

use lictor_canon::{canon, sha256_hex, sha256_jcs};
use lictor_receipt::{
    read_ledger, verify, verify_curve, verify_ledger, verify_ticks_file, verify_timing_chain, SignedCurve,
    SignedReceipt, TickEvent, TimingEvent, VerdictCounts, LEDGER_SCHEMA, TEST_PUBKEYS, TICKS_SCHEMA,
    ZERO_HASH,
};

use crate::render::{glossary, json_out, kv, notes, print_lines, sanitize, short, status};

#[derive(clap::Args)]
pub struct Args {
    /// Signed receipt JSON
    #[arg(value_name = "RECEIPT.json")]
    pub receipt: PathBuf,
    /// Full ticks file: recompute the verdict chain and compare with the signed head
    #[arg(long, value_name = "F.jsonl")]
    pub ticks: Option<PathBuf>,
    /// Timing file: chain integrity only
    #[arg(long, value_name = "F.jsonl")]
    pub timing: Option<PathBuf>,
    /// Ledger: chain integrity and that this receipt's digest appears
    #[arg(long, value_name = "F.jsonl")]
    pub ledger: Option<PathBuf>,
    /// Curve receipt: signature and ledger_head match
    #[arg(long, value_name = "C.json")]
    pub curve: Option<PathBuf>,
    /// calibration.json: budget alpha/gate/kn must equal the file's
    #[arg(long, value_name = "F.json")]
    pub calibration: Option<PathBuf>,
    /// Expected signing pubkey (hex64); a mismatch makes the receipt not intact
    #[arg(long, value_name = "hex")]
    pub pubkey: Option<String>,
}

/// The inputs of one verification (the CLI flags, or a caller such as `selftest`).
#[derive(Clone, Debug, Default)]
pub struct Opts {
    pub receipt: PathBuf,
    pub ticks: Option<PathBuf>,
    pub timing: Option<PathBuf>,
    pub ledger: Option<PathBuf>,
    pub curve: Option<PathBuf>,
    pub calibration: Option<PathBuf>,
    pub pubkey: Option<String>,
}

/// One named check with its rendered status word and detail.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Check {
    pub name: String,
    pub ok: bool,
    pub status: String,
    pub detail: String,
}

/// The whole verification: the checks, the two verdicts and the rendered human block.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct Outcome {
    pub intact: bool,
    pub fuse_ok: bool,
    pub test_key: bool,
    pub pubkey: String,
    pub body_digest: String,
    pub envelope_digest: String,
    pub embodiment_digest: Option<String>,
    pub calibration_digest: Option<String>,
    pub verdict_chain_head: String,
    pub timing_chain_head: String,
    pub counts: Option<VerdictCounts>,
    pub checks: Vec<Check>,
    pub notes: Vec<String>,
    pub lines: Vec<String>,
}

impl Outcome {
    /// 0 when intact AND the fuse held, else 1.
    pub fn exit_code(&self) -> i32 {
        if self.intact && self.fuse_ok {
            0
        } else {
            1
        }
    }

    /// The check named `name`, if it ran.
    #[allow(dead_code)] // kept for the receipt-check API; unused by the CLI path (CI builds with -D warnings)
    pub fn check(&self, name: &str) -> Option<&Check> {
        self.checks.iter().find(|c| c.name == name)
    }

    fn push(&mut self, name: &str, ok: bool, st: &str, detail: &str) {
        self.checks.push(Check {
            name: name.to_string(),
            ok,
            status: st.to_string(),
            detail: detail.to_string(),
        });
        self.lines.push(status(name, st, detail));
    }
}

/// `Ok(detail)` when the check passed; `Err((status word, detail))` otherwise.
type CheckResult = Result<String, (String, String)>;

fn read_json<T: serde::de::DeserializeOwned>(p: &Path) -> Result<T, String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", p.display()))
}

/// Read a `.jsonl` companion: the header (the first non-empty line, which must carry `schema`) and the events.
fn read_jsonl<T: serde::de::DeserializeOwned>(p: &Path) -> Result<(serde_json::Value, Vec<T>), String> {
    let text = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let mut header: Option<serde_json::Value> = None;
    let mut events = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.is_empty() {
            continue;
        }
        if header.is_none() {
            let v: serde_json::Value =
                serde_json::from_str(t).map_err(|e| format!("{}:{}: {e}", p.display(), i + 1))?;
            if v.get("schema").is_none() {
                return Err(format!("{}:{}: first line is not a schema header", p.display(), i + 1));
            }
            header = Some(v);
            continue;
        }
        let e: T = serde_json::from_str(t).map_err(|e| format!("{}:{}: {e}", p.display(), i + 1))?;
        events.push(e);
    }
    let header = header.ok_or_else(|| format!("{}: empty file", p.display()))?;
    Ok((header, events))
}

fn header_str<'a>(h: &'a serde_json::Value, k: &str) -> &'a str {
    h.get(k).and_then(|v| v.as_str()).unwrap_or("")
}

/// The ticks-file header must name this receipt's run/arm/episode and the genesis.
fn check_header(h: &serde_json::Value, sr: &SignedReceipt) -> Vec<String> {
    let run = &sr.body.run;
    let mut bad = Vec::new();
    if header_str(h, "schema") != TICKS_SCHEMA {
        bad.push(format!("schema {:?}", header_str(h, "schema")));
    }
    if header_str(h, "run_id") != run.run_id {
        bad.push(format!("run_id {:?}", header_str(h, "run_id")));
    }
    if header_str(h, "arm_id") != run.arm_id {
        bad.push(format!("arm_id {:?}", header_str(h, "arm_id")));
    }
    if h.get("episode_index").and_then(|v| v.as_u64()) != Some(u64::from(run.episode_index)) {
        bad.push(format!("episode_index {}", h.get("episode_index").cloned().unwrap_or_default()));
    }
    if header_str(h, "genesis") != ZERO_HASH {
        bad.push("genesis is not the zero hash".to_string());
    }
    bad
}

/// The full verdict chain: integrity, length, head vs the signed head, header and embedded-tail cross-checks.
fn check_ticks(sr: &SignedReceipt, p: &Path) -> CheckResult {
    let (header, ticks): (serde_json::Value, Vec<TickEvent>) =
        read_jsonl(p).map_err(|e| ("FAIL".to_string(), e))?;
    let bad = check_header(&header, sr);
    if !bad.is_empty() {
        return Err(("FAIL".to_string(), format!("ticks header mismatch: {}", bad.join(", "))));
    }
    let rep = verify_ticks_file(sr, &ticks);
    if !rep.ok {
        let at = rep.break_at.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string());
        return Err((format!("BROKEN at seq={at}"), String::new()));
    }
    let b = &sr.body;
    if rep.n != b.verdict_events {
        return Err((
            "FAIL".to_string(),
            format!("file has {} events, the receipt binds verdict_events={}", rep.n, b.verdict_events),
        ));
    }
    if rep.head != b.verdict_chain_head {
        return Err((
            format!("HEAD MISMATCH recomputed={} signed={}", short(&rep.head), short(&b.verdict_chain_head)),
            String::new(),
        ));
    }
    if let Some(first) = b.ticks.first() {
        match ticks.iter().find(|e| e.seq == first.seq) {
            Some(e) if e.prev == first.prev && e.hash == first.hash => {}
            Some(e) => {
                return Err((
                    "FAIL".to_string(),
                    format!(
                        "embedded tail starts at seq={} with prev={} but the file has prev={}",
                        first.seq,
                        short(&first.prev),
                        short(&e.prev)
                    ),
                ))
            }
            None => {
                return Err((
                    "FAIL".to_string(),
                    format!("embedded tail starts at seq={} which the file does not contain", first.seq),
                ))
            }
        }
    }
    Ok(format!("head={}  ({} ticks)", short(&rep.head), rep.n))
}

/// The timing chain: integrity, length and head vs the signed head (the values are wall-clock; never replayed).
fn check_timing(sr: &SignedReceipt, p: &Path) -> CheckResult {
    let (header, timing): (serde_json::Value, Vec<TimingEvent>) =
        read_jsonl(p).map_err(|e| ("FAIL".to_string(), e))?;
    let bad = check_header(&header, sr);
    if !bad.is_empty() {
        return Err(("FAIL".to_string(), format!("timing header mismatch: {}", bad.join(", "))));
    }
    let rep = verify_timing_chain(&timing);
    if !rep.ok {
        let at = rep.break_at.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string());
        return Err((format!("BROKEN at seq={at}"), String::new()));
    }
    let b = &sr.body;
    if rep.n != b.timing_events {
        return Err((
            "FAIL".to_string(),
            format!("file has {} events, the receipt binds timing_events={}", rep.n, b.timing_events),
        ));
    }
    if rep.head != b.timing_chain_head {
        return Err((
            format!("HEAD MISMATCH recomputed={} signed={}", short(&rep.head), short(&b.timing_chain_head)),
            String::new(),
        ));
    }
    Ok(format!("head={}  (not replayable -- by design)", short(&rep.head)))
}

/// The ledger: header, chain integrity, and this receipt's entry (digest, position, `ledger_prev`).
/// Returns the ledger head on success so `--curve` can compare it.
fn check_ledger(sr: &SignedReceipt, p: &Path) -> Result<(String, String), (String, String)> {
    let (header, _): (serde_json::Value, Vec<serde_json::Value>) =
        read_jsonl(p).map_err(|e| ("FAIL".to_string(), e))?;
    if header_str(&header, "schema") != LEDGER_SCHEMA {
        return Err(("FAIL".to_string(), format!("header schema {:?}", header_str(&header, "schema"))));
    }
    let entries = read_ledger(p).map_err(|e| ("FAIL".to_string(), e.to_string()))?;
    let rep = verify_ledger(&entries);
    if !rep.chain_ok {
        let at = rep.break_at.map(|s| s.to_string()).unwrap_or_else(|| "?".to_string());
        return Err((
            format!("BROKEN at seq={at}"),
            "entries are missing or edited; this curve point cannot be trusted".to_string(),
        ));
    }
    let run = &sr.body.run;
    let mine: Vec<_> = entries
        .iter()
        .filter(|e| e.run_id == run.run_id && e.arm_id == run.arm_id && e.episode_index == run.episode_index)
        .collect();
    let Some(entry) = mine.iter().find(|e| e.receipt_digest == sr.body_digest) else {
        return Err((
            "FAIL".to_string(),
            format!("receipt digest {} is not in the ledger", short(&sr.body_digest)),
        ));
    };
    let prev = sr.body.ledger_prev.clone().unwrap_or_else(|| ZERO_HASH.to_string());
    if entry.prev != prev {
        return Err((
            "FAIL".to_string(),
            format!(
                "ledger entry prev={} but the receipt binds ledger_prev={}",
                short(&entry.prev),
                short(&prev)
            ),
        ));
    }
    Ok((format!("seq={} head={}  ({} entries)", entry.seq, short(&rep.head), rep.episodes), rep.head))
}

/// The curve receipt: signature and digest, the same run/arm, and `ledger_head` against the verified ledger.
fn check_curve(sr: &SignedReceipt, p: &Path, ledger_head: Option<&str>) -> CheckResult {
    let sc: SignedCurve = read_json(p).map_err(|e| ("FAIL".to_string(), e))?;
    let rep = verify_curve(&sc, None);
    if !rep.intact() {
        let why = if rep.notes.is_empty() { "signature or digest".to_string() } else { rep.notes.join("; ") };
        return Err(("FAIL".to_string(), format!("curve {}: {why}", sanitize(&sc.body.arm_id))));
    }
    let run = &sr.body.run;
    if sc.body.run_id != run.run_id || sc.body.arm_id != run.arm_id {
        return Err((
            "FAIL".to_string(),
            format!(
                "curve is for {}/{}, receipt is {}/{}",
                sc.body.run_id, sc.body.arm_id, run.run_id, run.arm_id
            ),
        ));
    }
    match ledger_head {
        Some(h) if h == sc.body.ledger_head => Ok(format!(
            "ledger_head={} n={} (key {})",
            short(&sc.body.ledger_head),
            sc.body.n_episodes,
            short(&sc.pubkey)
        )),
        Some(h) => Err((
            "FAIL".to_string(),
            format!(
                "curve ledger_head={} but the ledger's head is {}",
                short(&sc.body.ledger_head),
                short(h)
            ),
        )),
        None => {
            Ok(format!("signature ok; ledger_head={} unchecked (pass --ledger)", short(&sc.body.ledger_head)))
        }
    }
}

/// calibration.json against the budget: alpha, gate, kn, and the self-digest against `calibration_digest`.
fn check_calibration(sr: &SignedReceipt, p: &Path) -> CheckResult {
    let v: serde_json::Value = read_json(p).map_err(|e| ("FAIL".to_string(), e))?;
    let b = &sr.body.budget;
    let mut bad = Vec::new();
    let num = v.get("alpha_num").and_then(|x| x.as_u64());
    let den = v.get("alpha_den").and_then(|x| x.as_u64());
    if num != Some(u64::from(b.alpha_num)) || den != Some(u64::from(b.alpha_den)) {
        bad.push(format!(
            "alpha {}/{} != file {}/{}",
            b.alpha_num,
            b.alpha_den,
            num.map(|x| x.to_string()).unwrap_or_else(|| "?".into()),
            den.map(|x| x.to_string()).unwrap_or_else(|| "?".into())
        ));
    }
    let gate: Vec<String> = v
        .get("gate")
        .and_then(|g| g.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    if gate != b.gate {
        bad.push(format!("gate {:?} != file {:?}", b.gate, gate));
    }
    let kn: Vec<u64> = v
        .get("kn")
        .and_then(|k| k.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64()).collect())
        .unwrap_or_default();
    if kn != [u64::from(b.kn[0]), u64::from(b.kn[1])] {
        bad.push(format!("kn {:?} != file {:?}", b.kn, kn));
    }
    if !b.tier1_armed {
        bad.push("receipt has tier 1 disarmed; no calibration is bound".to_string());
    }
    let file_digest = v.get("digest").and_then(|d| d.as_str()).unwrap_or("").to_string();
    let mut core = v.clone();
    if let Some(o) = core.as_object_mut() {
        o.insert("digest".to_string(), serde_json::Value::String(String::new()));
    }
    let recomputed = match canon(&core) {
        Ok(bytes) => sha256_hex(&bytes),
        Err(e) => {
            bad.push(format!("file is not canonicalisable: {e}"));
            String::new()
        }
    };
    if !recomputed.is_empty() && recomputed != file_digest {
        bad.push(format!("self-digest {} != recomputed {}", short(&file_digest), short(&recomputed)));
    }
    match sr.body.calibration_digest.as_deref() {
        Some(d) if d == recomputed => {}
        Some(d) => {
            bad.push(format!("receipt calibration_digest {} != file {}", short(d), short(&recomputed)))
        }
        None => bad.push("receipt carries no calibration_digest".to_string()),
    }
    if !bad.is_empty() {
        return Err(("FAIL".to_string(), bad.join("; ")));
    }
    let n_calib = v.get("n_calib").and_then(|x| x.as_u64()).unwrap_or(0);
    Ok(format!(
        "alpha={}/{} gate={} kn={},{} n_calib={n_calib} digest={}",
        b.alpha_num,
        b.alpha_den,
        b.gate.len(),
        b.kn[0],
        b.kn[1],
        short(&recomputed)
    ))
}

fn push_result(out: &mut Outcome, name: &str, r: CheckResult) -> bool {
    match r {
        Ok(detail) => {
            out.push(name, true, "ok", &detail);
            true
        }
        Err((st, detail)) => {
            out.push(name, false, &st, &detail);
            false
        }
    }
}

/// Run every check named by `o`; never fails (an unreadable file is a failed check).
pub fn run_checks(o: &Opts) -> Outcome {
    let mut out = Outcome::default();
    let sr: SignedReceipt = match read_json(&o.receipt) {
        Ok(sr) => sr,
        Err(e) => {
            out.push("schema", false, "FAIL", &format!("not a signed receipt: {e}"));
            out.lines.push(status("intact", "NO", ""));
            out.notes.push(e);
            return out;
        }
    };
    let b = &sr.body;
    out.pubkey = sr.pubkey.clone();
    out.body_digest = sr.body_digest.clone();
    out.envelope_digest = b.envelope_digest.clone();
    out.calibration_digest = b.calibration_digest.clone();
    out.verdict_chain_head = b.verdict_chain_head.clone();
    out.timing_chain_head = b.timing_chain_head.clone();
    out.counts = Some(b.counts.clone());
    out.embodiment_digest = b.envelope.get("embodiment").and_then(|e| sha256_jcs(e).ok());
    out.test_key = TEST_PUBKEYS.contains(&sr.pubkey.as_str());
    if out.test_key {
        out.lines.push("  WARNING: signed with the committed test key".to_string());
    }

    let rep = verify(&sr, o.pubkey.as_deref());
    out.push("schema", rep.schema_ok, if rep.schema_ok { "ok" } else { "FAIL" }, &sanitize(&b.schema));
    let key = format!("(key {})", short(&sr.pubkey));
    out.push("signature", rep.sig_ok, if rep.sig_ok { "ok" } else { "FAIL" }, &key);
    if let Some(expected) = &o.pubkey {
        if rep.pubkey_ok {
            out.push("pubkey", true, "ok", "(matches --pubkey)");
        } else {
            let detail = format!("(receipt {}, expected {})", short(&sr.pubkey), short(expected));
            out.push("pubkey", false, "MISMATCH", &format!(" {detail}"));
        }
    }
    out.push(
        "body digest",
        rep.digest_ok,
        if rep.digest_ok { "ok" } else { "FAIL" },
        &short(&sr.body_digest),
    );
    let emb = out.embodiment_digest.as_deref().map(short).unwrap_or_else(|| "n/a".to_string());
    out.push(
        "envelope",
        rep.envelope_digest_ok,
        if rep.envelope_digest_ok { "ok" } else { "FAIL" },
        &format!("digest recomputed {}  embodiment {emb}", short(&b.envelope_digest)),
    );
    out.push(
        "counts",
        rep.counts_ok,
        if rep.counts_ok { "ok" } else { "FAIL" },
        &format!(
            "ticks={} verdict_events={} timing_events={} steps={}",
            b.counts.ticks, b.verdict_events, b.timing_events, b.outcome.steps
        ),
    );

    let mut intact = rep.intact();
    match &o.ticks {
        Some(p) => intact &= push_result(&mut out, "verdict chain", check_ticks(&sr, p)),
        None => {
            if rep.ticks_ok {
                out.push(
                    "verdict chain",
                    true,
                    "ok",
                    &format!(
                        "head={}  (embedded {}: {} ticks; pass --ticks for the full chain)",
                        short(&b.verdict_chain_head),
                        b.ticks_policy,
                        b.ticks.len()
                    ),
                );
            } else if let Some(at) = rep.break_at {
                out.push("verdict chain", false, &format!("BROKEN at seq={at}"), "(embedded ticks)");
            } else {
                out.push("verdict chain", false, "FAIL", "embedded ticks do not match the body");
            }
        }
    }
    if let Some(p) = &o.timing {
        intact &= push_result(&mut out, "timing chain", check_timing(&sr, p));
    }
    let mut ledger_head: Option<String> = None;
    if let Some(p) = &o.ledger {
        match check_ledger(&sr, p) {
            Ok((detail, head)) => {
                out.push("ledger", true, "ok", &detail);
                ledger_head = Some(head);
            }
            Err((st, detail)) => {
                out.push("ledger", false, &st, &detail);
                intact = false;
            }
        }
    }
    if let Some(p) = &o.curve {
        intact &= push_result(&mut out, "curve", check_curve(&sr, p, ledger_head.as_deref()));
    }
    if let Some(p) = &o.calibration {
        intact &= push_result(&mut out, "calibration", check_calibration(&sr, p));
    }

    out.intact = intact;
    out.fuse_ok = intact && rep.fuse_ok;
    out.lines.push(status("intact", if intact { "YES" } else { "NO" }, ""));
    if intact {
        if out.fuse_ok {
            out.lines.push(kv(
                "FUSE HELD",
                &format!(
                    "({} substitutions, {} violations reached the environment)",
                    b.counts.substituted, b.counts.violations_reached_env
                ),
            ));
        } else {
            out.lines.push(kv("FUSE NOT ENFORCED", ""));
        }
        let calib = match (&b.calibration_digest, b.budget.tier1_armed) {
            (Some(d), true) => format!("{}(alpha={}/{})", short(d), b.budget.alpha_num, b.budget.alpha_den),
            _ => "none".to_string(),
        };
        let policy = match (b.run.policy.get("repo_id"), b.run.policy.get("revision")) {
            (Some(r), Some(rev)) => format!("{r}@{}", short(rev)),
            (Some(r), None) => r.clone(),
            (None, _) => {
                format!(
                    "weights={}",
                    short(b.run.policy.get("weights_sha256").map(String::as_str).unwrap_or(""))
                )
            }
        };
        out.lines.push(kv(
            "bindings",
            &format!("envelope={} calibration={calib} policy={policy}", short(&b.envelope_digest)),
        ));
        out.lines.push(kv(
            "",
            &format!(
                "seeds pool={} seed={} init_state={}  client={}",
                b.run.seed_pool,
                b.run.seed,
                short(&b.run.init_state_digest),
                b.client
            ),
        ));
        for h in &b.handoffs {
            out.lines.push(kv(
                "handoff",
                &format!(
                    "seq={} tick={} {} outcome={} -- {}",
                    h.seq,
                    h.tick,
                    lictor_receipt::body::reason_name(h.reason),
                    h.outcome,
                    glossary(h.reason)
                ),
            ));
        }
    }
    out.notes = rep.notes.clone();
    for c in &out.checks {
        if !c.ok && c.status == "FAIL" && !c.detail.is_empty() {
            out.notes.push(format!("{}: {}", c.name, c.detail));
        }
    }
    let rendered = notes(&out.notes);
    out.lines.extend(rendered);
    out
}

pub fn run(a: Args, json: bool) -> anyhow::Result<i32> {
    let Args { receipt, ticks, timing, ledger, curve, calibration, pubkey } = a;
    let o =
        Opts { receipt, ticks, timing, ledger, curve, calibration, pubkey: pubkey.map(|p| p.to_lowercase()) };
    let out = run_checks(&o);
    if json {
        json_out(&out)?;
    } else {
        print_lines(&out.lines);
    }
    Ok(out.exit_code())
}
