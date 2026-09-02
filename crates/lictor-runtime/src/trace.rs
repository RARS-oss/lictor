// SPDX-License-Identifier: MIT
//! Request trace files: a first line `#meta {json}`, then every request line VERBATIM (wire format, plain floats).
//!
//! Responses are never traced -- they are recomputable, which is the determinism claim `lictor replay` checks.
//! The reader yields `(line_no, raw, Request)` for every non-empty line after the meta line; a line that does not
//! parse is an `Err(InvalidData)` item (the file is evidence, so a corrupt line is reported, not skipped).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Write};
use std::path::Path;

use crate::codec::parse_request;
use crate::wire::Request;

pub const META_PREFIX: &str = "#meta ";

/// One replayable line of a trace: (line_no, raw, parsed).
pub type TraceLine = (u64, String, Request);

pub struct TraceWriter {
    w: BufWriter<File>,
}

impl TraceWriter {
    /// Create (truncate) the file, creating parent directories, and write the `#meta` line.
    pub fn open(p: &std::path::Path, meta: &serde_json::Value) -> std::io::Result<Self> {
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let mut w = BufWriter::new(File::create(p)?);
        let text = serde_json::to_string(meta).map_err(|e| Error::other(e.to_string()))?;
        w.write_all(META_PREFIX.as_bytes())?;
        w.write_all(text.as_bytes())?;
        w.write_all(b"\n")?;
        w.flush()?;
        Ok(Self { w })
    }

    /// Append one request line verbatim (a trailing newline is added; `raw` must not contain one).
    pub fn line(&mut self, raw: &str) -> std::io::Result<()> {
        self.w.write_all(raw.as_bytes())?;
        self.w.write_all(b"\n")?;
        self.w.flush()
    }
}

pub struct TraceReader {
    r: BufReader<File>,
    line_no: u64,
    path: String,
}

impl TraceReader {
    /// Open a trace and return the parsed `#meta` object (the first line MUST be a meta line).
    pub fn open(p: &std::path::Path) -> std::io::Result<(Self, serde_json::Value)> {
        let mut r = BufReader::new(File::open(p)?);
        let mut first = String::new();
        r.read_line(&mut first)?;
        let first = first.trim_end_matches(['\n', '\r']);
        let meta_text = first.strip_prefix(META_PREFIX).ok_or_else(|| {
            Error::new(ErrorKind::InvalidData, format!("{}: first line is not a #meta line", p.display()))
        })?;
        let meta: serde_json::Value = serde_json::from_str(meta_text).map_err(|e| {
            Error::new(ErrorKind::InvalidData, format!("{}: #meta is not JSON: {e}", p.display()))
        })?;
        Ok((Self { r, line_no: 1, path: p.display().to_string() }, meta))
    }
}

impl Iterator for TraceReader {
    /// (line_no, raw, parsed)
    type Item = std::io::Result<(u64, String, Request)>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let mut raw = String::new();
            match self.r.read_line(&mut raw) {
                Ok(0) => return None,
                Ok(_) => {}
                Err(e) => return Some(Err(e)),
            }
            self.line_no += 1;
            while raw.ends_with('\n') || raw.ends_with('\r') {
                raw.pop();
            }
            if raw.trim().is_empty() {
                continue;
            }
            let line_no = self.line_no;
            return Some(match parse_request(&raw) {
                Ok(req) => Ok((line_no, raw, req)),
                Err(e) => {
                    Err(Error::new(ErrorKind::InvalidData, format!("{}:{}: {}", self.path, line_no, e)))
                }
            });
        }
    }
}

/// Read a whole trace: the meta object and every request line (the replay input).
pub fn read_trace(p: &Path) -> std::io::Result<(serde_json::Value, Vec<TraceLine>)> {
    let (reader, meta) = TraceReader::open(p)?;
    let lines = reader.collect::<std::io::Result<Vec<_>>>()?;
    Ok((meta, lines))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_read() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("sub").join("t.ndjson");
        let meta = serde_json::json!({"proto": "lictor-wire/v1", "mode": "observe"});
        {
            let mut w = TraceWriter::open(&p, &meta).unwrap();
            w.line("{\"id\":1,\"kind\":\"bye\"}").unwrap();
            w.line("{\"id\":2,\"kind\":\"bye\"}").unwrap();
        }
        let text = std::fs::read_to_string(&p).unwrap();
        assert!(text.starts_with("#meta {"));
        assert_eq!(text.lines().count(), 3);
        let (r, m) = TraceReader::open(&p).unwrap();
        assert_eq!(m, meta);
        let items: Vec<_> = r.map(|x| x.unwrap()).collect();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].0, 2);
        assert_eq!(items[0].1, "{\"id\":1,\"kind\":\"bye\"}");
        assert!(matches!(items[1].2, Request::Bye(ref b) if b.id == 2));
        let (m2, lines) = read_trace(&p).unwrap();
        assert_eq!(m2, meta);
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn bad_meta_and_bad_line() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.ndjson");
        std::fs::write(&p, "{\"id\":1,\"kind\":\"bye\"}\n").unwrap();
        assert!(TraceReader::open(&p).is_err());
        std::fs::write(&p, "#meta {}\n\n{\"id\":1,\"kind\":\"nope\"}\n").unwrap();
        let (mut r, _) = TraceReader::open(&p).unwrap();
        let e = r.next().unwrap().unwrap_err();
        assert!(e.to_string().contains(":3:"), "{e}");
    }
}
