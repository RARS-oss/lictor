// SPDX-License-Identifier: MIT
//! Deterministic, length-capped human rendering (sbx render doctrine) shared by every command.
//!
//! * `kv(label, value)` -- one line of the aligned block, the label padded to `LABEL_WIDTH` columns
//!   (`  schema         ok    lictor-receipt/v1`).
//! * `status(label, st, detail)` -- `kv` with a status word (`ok` / `FAIL` / `YES` / `NO`) padded to `STATUS_WIDTH`.
//! * `note(s)` / `notes(list)` -- the `  notes:  - ...` block.
//! * `glossary(reason)` -- the plain-English sentence for a reason code (`reason_text`), the human-escalation channel.
//! * `sanitize(s)` -- strips ASCII/C1 control characters and ESC sequences from every host- or operator-supplied
//!   string (`AckToken.note`, `reason_text`, `fuse_notes`, arm/run ids, client strings) so a signed note cannot inject
//!   terminal escapes. `cap(s, n)` cuts a value at `n` characters with `...`.
//! * `short(hex)` -- the first `SHORT_HEX` characters of a digest plus `...` (the full value is in `--json`).
//! * `json_out(t)` -- `--json` short-circuits to serde output: pretty-printed, keys sorted at every level.
//!
//! Nothing here reads a clock or the environment; the same inputs render the same bytes.

use lictor_core::{reason_text, ReasonCode};

/// Label column width of the aligned block.
pub const LABEL_WIDTH: usize = 15;
/// Status word width inside a value (`ok    `, `FAIL  `).
pub const STATUS_WIDTH: usize = 6;
/// Longest value a human line carries before it is cut with `...`.
pub const MAX_VALUE: usize = 200;
/// Hex prefix shown for digests and keys in human output.
pub const SHORT_HEX: usize = 12;

/// Strip C0/C1 control characters (tab, newline and carriage return become one space) and ESC sequences
/// (CSI `ESC [ ... final`, OSC/DCS/SOS/PM/APC `ESC ] ... BEL|ST`, two-character escapes) from a string.
pub fn sanitize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\u{1b}' {
            match it.peek().copied() {
                Some('[') => {
                    it.next();
                    // parameter (0x30-0x3f) and intermediate (0x20-0x2f) bytes, then one final byte 0x40-0x7e
                    for d in it.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&d) {
                            break;
                        }
                    }
                }
                Some(']') | Some('P') | Some('X') | Some('^') | Some('_') => {
                    it.next();
                    let mut prev_esc = false;
                    for d in it.by_ref() {
                        if d == '\u{07}' || (prev_esc && d == '\\') {
                            break;
                        }
                        prev_esc = d == '\u{1b}';
                    }
                }
                Some(_) => {
                    it.next();
                }
                None => {}
            }
            continue;
        }
        if c == '\n' || c == '\r' || c == '\t' {
            if !out.ends_with(' ') {
                out.push(' ');
            }
            continue;
        }
        if c.is_control() {
            continue;
        }
        out.push(c);
    }
    out
}

/// Cut a value at `max` characters with a trailing `...` (never splits a UTF-8 sequence).
pub fn cap(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(3);
    let mut out: String = s.chars().take(keep).collect();
    out.push_str("...");
    out
}

/// The first `SHORT_HEX` characters of a hex digest plus `...`; a short input is returned as is.
pub fn short(hex: &str) -> String {
    if hex.len() <= SHORT_HEX {
        return hex.to_string();
    }
    format!("{}...", &hex[..SHORT_HEX])
}

/// One aligned line: two spaces, the label padded to `LABEL_WIDTH`, the (sanitized, capped) value.
pub fn kv(label: &str, value: &str) -> String {
    let value = cap(&sanitize(value), MAX_VALUE);
    if value.is_empty() {
        return format!("  {label}");
    }
    format!("  {label:<LABEL_WIDTH$}{value}")
}

/// `kv` with a status word padded to `STATUS_WIDTH` in front of the detail (`ok    detail`).
pub fn status(label: &str, st: &str, detail: &str) -> String {
    if detail.is_empty() {
        kv(label, st)
    } else {
        kv(label, &format!("{st:<STATUS_WIDTH$}{detail}"))
    }
}

/// The first line of the notes block.
pub fn note(s: &str) -> String {
    format!("  notes:  - {}", cap(&sanitize(s), MAX_VALUE))
}

/// The whole notes block: the first line carries the `notes:` label, the following lines align with it.
pub fn notes(list: &[String]) -> Vec<String> {
    list.iter()
        .enumerate()
        .map(|(i, s)| if i == 0 { note(s) } else { format!("          - {}", cap(&sanitize(s), MAX_VALUE)) })
        .collect()
}

/// The plain-English sentence for a reason code (`lictor_core::reason_text`).
pub fn glossary(reason: ReasonCode) -> &'static str {
    reason_text(reason)
}

/// Pretty JSON (2-space indent) with keys sorted at every level (`serde_json::Value` objects are BTreeMaps).
pub fn pretty_sorted<T: serde::Serialize>(t: &T) -> anyhow::Result<String> {
    let v = serde_json::to_value(t)?;
    let mut s = serde_json::to_string_pretty(&v)?;
    s.push('\n');
    Ok(s)
}

/// `--json` output: one pretty-printed, key-sorted document on stdout.
pub fn json_out<T: serde::Serialize>(t: &T) -> anyhow::Result<()> {
    use std::io::Write;
    let s = pretty_sorted(t)?;
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    w.write_all(s.as_bytes())?;
    w.flush()?;
    Ok(())
}

/// Print rendered lines to stdout.
pub fn print_lines(lines: &[String]) {
    for l in lines {
        println!("{l}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_controls_and_escapes() {
        assert_eq!(sanitize("a\u{1b}[31mb\u{7}c"), "abc");
        assert_eq!(sanitize("x\u{1b}]0;title\u{7}y"), "xy");
        assert_eq!(sanitize("x\u{1b}]0;title\u{1b}\\y"), "xy");
        assert_eq!(sanitize("p\u{1b}cq"), "pq");
        assert_eq!(sanitize("line1\nline2\ttab\r\n"), "line1 line2 tab ");
        assert_eq!(sanitize("\u{0}\u{1f}\u{7f}\u{85}\u{9b}z"), "z");
        assert_eq!(sanitize("plain text"), "plain text");
        assert_eq!(sanitize("dangling\u{1b}"), "dangling");
    }

    #[test]
    fn alignment_and_caps() {
        assert_eq!(kv("schema", "ok    lictor-receipt/v1"), "  schema         ok    lictor-receipt/v1");
        assert_eq!(status("signature", "ok", "(key abc)"), "  signature      ok    (key abc)");
        assert_eq!(status("intact", "YES", ""), "  intact         YES");
        assert_eq!(kv("FUSE NOT ENFORCED", ""), "  FUSE NOT ENFORCED");
        assert_eq!(cap("abcdef", 6), "abcdef");
        assert_eq!(cap("abcdefg", 6), "abc...");
        assert_eq!(short("0123456789abcdef"), "0123456789ab...");
        assert_eq!(short("0123"), "0123");
        let n = notes(&["one".to_string(), "two".to_string()]);
        assert_eq!(n[0], "  notes:  - one");
        assert_eq!(n[1], "          - two");
        assert!(glossary(ReasonCode::Ok).ends_with('.'));
    }

    #[test]
    fn json_is_sorted() {
        #[derive(serde::Serialize)]
        struct S {
            zeta: u32,
            alpha: u32,
        }
        assert_eq!(
            pretty_sorted(&S { zeta: 1, alpha: 2 }).unwrap(),
            "{\n  \"alpha\": 2,\n  \"zeta\": 1\n}\n"
        );
    }
}
