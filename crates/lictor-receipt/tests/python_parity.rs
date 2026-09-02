// SPDX-License-Identifier: MIT
//! Spawn the stdlib Python verifier over the committed fixture and assert the exact parity lines. Skips with a
//! message when no Python interpreter is available.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures").join("receipt")
}

fn find_python() -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("LICTOR_PYTHON") {
        candidates.push(PathBuf::from(p));
    }
    candidates.push(PathBuf::from("/mnt/d/lictor/venv/bin/python"));
    candidates.push(PathBuf::from("python3"));
    candidates.push(PathBuf::from("python"));
    candidates
        .into_iter()
        .find(|c| Command::new(c).arg("--version").output().map(|o| o.status.success()).unwrap_or(false))
}

fn run(py: &Path, args: &[String]) -> (i32, String) {
    let out = Command::new(py)
        .current_dir(repo_root())
        .arg("adapters/verify_receipt.py")
        .args(args)
        .output()
        .expect("spawn python");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    (out.status.code().unwrap_or(-1), stdout)
}

#[test]
fn python_verifier_agrees_byte_for_byte() {
    let Some(py) = find_python() else {
        eprintln!("SKIP python_parity: no python interpreter found (set LICTOR_PYTHON or install python3)");
        return;
    };
    let d = fixture_dir();
    let pubkey = std::fs::read_to_string(d.join("pubkey.txt")).unwrap().trim().to_string();
    let receipt = d.join("receipt_000007.json").to_string_lossy().to_string();
    let ticks = d.join("ticks_000007.jsonl").to_string_lossy().to_string();
    let ledger = d.join("ledger.jsonl").to_string_lossy().to_string();

    let (code, out) = run(
        &py,
        &[
            receipt.clone(),
            "--ticks".into(),
            ticks.clone(),
            "--ledger".into(),
            ledger,
            "--pubkey".into(),
            pubkey.clone(),
            "--parity".into(),
        ],
    );
    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(code, 0, "python verifier failed:\n{out}");
    assert!(lines.contains(&"canonical bytes: IDENTICAL"), "{out}");
    assert!(lines.contains(&"signature: ok"), "{out}");
    assert_eq!(lines.last().copied(), Some("OK"), "{out}");

    // The wrong expected key is a FAIL with the exact reason.
    let (code, out) = run(&py, &[receipt.clone(), "--pubkey".into(), "ab".repeat(32)]);
    assert_eq!(code, 1);
    assert_eq!(out.trim(), "FAIL: pubkey mismatch");

    // A ticks file with one edited tick: the Python chain walk breaks where Rust's does.
    let tmp = tempfile::tempdir().unwrap();
    let text = std::fs::read_to_string(&ticks).unwrap();
    let mut lines: Vec<String> = text.lines().map(str::to_string).collect();
    lines[18] = lines[18].replacen("\"trips\":0", "\"trips\":1", 1);
    let edited = tmp.path().join("ticks.jsonl");
    std::fs::write(&edited, lines.join("\n") + "\n").unwrap();
    let (code, out) = run(
        &py,
        &[receipt, "--ticks".into(), edited.to_string_lossy().to_string(), "--pubkey".into(), pubkey],
    );
    assert_eq!(code, 1);
    assert_eq!(out.trim(), "FAIL: verdict chain broken at seq 17");
}
