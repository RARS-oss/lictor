// SPDX-License-Identifier: MIT
//! Build script for the `lictor` binary: embeds the short git sha ("nogit" on a checkout without git) and the
//! build time (UTC) as `LICTOR_GIT` / `LICTOR_BUILD_UTC`. The binary's own sha256 is computed at RUN time from
//! `std::env::current_exe()` (by `version` and the session), never here.
#![forbid(unsafe_code)]

use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn git_short_sha() -> String {
    let out = Command::new("git").args(["-c", "safe.directory=*", "rev-parse", "--short", "HEAD"]).output();
    match out {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s.is_empty() {
                "nogit".to_string()
            } else {
                s
            }
        }
        _ => "nogit".to_string(),
    }
}

/// Civil date from days since 1970-01-01 (Howard Hinnant's algorithm), UTC, no external crate.
fn utc_stamp(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();
    let git_dir = Path::new(&manifest_dir).join("..").join("..").join(".git");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
    // When HEAD is a symbolic ref, also watch the ref it points at so a new commit re-runs this script.
    if let Ok(head) = std::fs::read_to_string(git_dir.join("HEAD")) {
        if let Some(r) = head.trim().strip_prefix("ref: ") {
            println!("cargo:rerun-if-changed={}", git_dir.join(r).display());
        }
    }
    println!("cargo:rustc-env=LICTOR_GIT={}", git_short_sha());
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    println!("cargo:rustc-env=LICTOR_BUILD_UTC={}", utc_stamp(secs));
}
