// SPDX-License-Identifier: MIT
//! Every committed fixture in this crate and in crates/lictor-receipt/tests/fixtures passes `check_keys`:
//! .json files as a whole, .jsonl files line by line. This is the property that makes Python's code-point key
//! sort equal to the JCS UTF-16 sort in adapters/verify_receipt.py.

use std::path::{Path, PathBuf};

use lictor_canon::{check_keys, CanonError};

fn fixture_roots() -> Vec<PathBuf> {
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        here.join("tests").join("fixtures"),
        here.join("..").join("lictor-receipt").join("tests").join("fixtures"),
    ]
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, out);
        } else if matches!(p.extension().and_then(|e| e.to_str()), Some("json") | Some("jsonl")) {
            out.push(p);
        }
    }
}

#[test]
fn every_committed_fixture_passes_check_keys() {
    let mut files = Vec::new();
    for root in fixture_roots() {
        walk(&root, &mut files);
    }
    assert!(files.len() >= 6, "expected the canon and receipt fixtures, found {files:?}");
    let mut checked = 0usize;
    for f in &files {
        let text = std::fs::read_to_string(f).unwrap();
        let docs: Vec<&str> = if f.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            text.lines().filter(|l| !l.trim().is_empty()).collect()
        } else {
            vec![text.as_str()]
        };
        for (i, d) in docs.iter().enumerate() {
            let v: serde_json::Value =
                serde_json::from_str(d).unwrap_or_else(|e| panic!("{}:{}: {e}", f.display(), i + 1));
            check_keys(&v).unwrap_or_else(|e| panic!("{}:{}: {e}", f.display(), i + 1));
            checked += 1;
        }
    }
    assert!(checked > 300, "checked {checked} documents");
}

#[test]
fn the_key_class_is_printable_ascii() {
    // The receipt binds repo-relative paths, `lictor:bin` and environment-variable names as map keys.
    let ok = serde_json::json!({"lictor:bin": "x", "envelopes/pusht.toml": "y", "OMP_NUM_THREADS": "1", "a-b.c~d": 0});
    assert!(check_keys(&ok).is_ok());
    for bad in ["a b", "", "caf\u{e9}", "x\u{7f}", "tab\tkey", "\u{fb33}"] {
        let v = serde_json::json!({ bad: 1 });
        assert!(matches!(check_keys(&v), Err(CanonError::BadKey(_))), "{bad:?} should be rejected");
    }
}
