// SPDX-License-Identifier: MIT
//! NDJSON framing for `lictor-wire/v1`: one request per line in, one response per line out.
//!
//! Framing rules (docs/wire-protocol.md): UTF-8, exactly one JSON object per line, `\n`-terminated, no embedded
//! `\r`, at most `MAX_LINE` bytes. `read_request` returns `Ok(None)` at EOF and `Err(InvalidData(msg))` for
//! anything malformed -- an over-long line (the remainder of that line is drained so the next line starts clean),
//! invalid UTF-8, an embedded carriage return, an empty line, or a serde rejection (unknown key at any nesting
//! level, wrong type, the non-JSON literal `NaN`). The caller answers such a line with `error{fatal:true}`.

use std::io::{BufRead, Error, ErrorKind, Read};

use crate::wire::{Request, Response};

pub const MAX_LINE: usize = 1 << 20;

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Consume the rest of the current line (after an over-long read) so the stream is positioned at the next line.
fn drain_line(r: &mut impl BufRead) -> std::io::Result<()> {
    loop {
        let buf = r.fill_buf()?;
        if buf.is_empty() {
            return Ok(());
        }
        match buf.iter().position(|b| *b == b'\n') {
            Some(i) => {
                r.consume(i + 1);
                return Ok(());
            }
            None => {
                let n = buf.len();
                r.consume(n);
            }
        }
    }
}

/// A strict mirror of `lictor_receipt::AckToken`. The frozen `AckToken` carries no `deny_unknown_fields` (it is a
/// file-format struct owned by lictor-receipt), so a tick whose `ack` carries an unknown key would otherwise slip
/// through serde; the wire contract says an unknown key at ANY nesting level is a schema fault, so the `ack`
/// object of a tick is re-checked against this mirror (only when an ack is present -- a rare message).
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct AckTokenStrict {
    #[allow(dead_code)]
    schema: String,
    #[allow(dead_code)]
    handoff_digest: String,
    #[allow(dead_code)]
    decision: lictor_core::AckDecision,
    #[allow(dead_code)]
    operator: String,
    #[allow(dead_code)]
    nonce: u64,
    #[allow(dead_code)]
    note: String,
    #[allow(dead_code)]
    sig: String,
}

fn strict_ack_check(line: &str) -> Result<(), String> {
    let v: serde_json::Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
    match v.get("ack") {
        Some(ack) if !ack.is_null() => serde_json::from_value::<AckTokenStrict>(ack.clone())
            .map(|_| ())
            .map_err(|e| format!("tick.ack: {e}")),
        _ => Ok(()),
    }
}

/// Parse one already-read line (the framing was validated by the caller). Shared by `read_request` and the
/// trace reader, so there is exactly one wire parser in the workspace.
pub fn parse_request(line: &str) -> std::io::Result<Request> {
    let req =
        serde_json::from_str::<Request>(line).map_err(|e| invalid(format!("malformed request: {e}")))?;
    if let Request::Tick(t) = &req {
        if t.ack.is_some() {
            strict_ack_check(line).map_err(|e| invalid(format!("malformed request: {e}")))?;
        }
    }
    Ok(req)
}

/// Best-effort `id` of a line that failed to parse, so the error response can still echo it.
pub fn peek_id(line: &str) -> Option<u64> {
    serde_json::from_str::<serde_json::Value>(line).ok()?.get("id")?.as_u64()
}

/// None at EOF; Err(InvalidData) on malformed. `buf` receives the raw line (newline stripped) so the caller can
/// trace it verbatim; it is cleared first.
pub fn read_request(r: &mut impl std::io::BufRead, buf: &mut String) -> std::io::Result<Option<Request>> {
    buf.clear();
    let n = {
        let mut limited = r.by_ref().take(MAX_LINE as u64 + 1);
        limited.read_line(buf)?
    };
    if n == 0 {
        return Ok(None);
    }
    if n > MAX_LINE {
        if !buf.ends_with('\n') {
            drain_line(r)?;
        }
        buf.clear();
        return Err(invalid(format!("line exceeds MAX_LINE ({MAX_LINE} bytes)")));
    }
    if buf.ends_with('\n') {
        buf.pop();
    }
    if buf.contains('\r') {
        return Err(invalid("line contains a carriage return".to_string()));
    }
    if buf.trim().is_empty() {
        return Err(invalid("empty line".to_string()));
    }
    parse_request(buf).map(Some)
}

/// one line + flush
pub fn write_response(w: &mut impl std::io::Write, resp: &Response) -> std::io::Result<()> {
    let text = serde_json::to_string(resp).map_err(|e| Error::other(e.to_string()))?;
    w.write_all(text.as_bytes())?;
    w.write_all(b"\n")?;
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn eof_and_bye() {
        let mut r = Cursor::new(b"{\"id\":2,\"kind\":\"bye\"}\n".to_vec());
        let mut buf = String::new();
        assert!(matches!(read_request(&mut r, &mut buf), Ok(Some(Request::Bye(b))) if b.id == 2));
        assert_eq!(buf, "{\"id\":2,\"kind\":\"bye\"}");
        assert!(matches!(read_request(&mut r, &mut buf), Ok(None)));
    }

    #[test]
    fn rejects_bad_framing() {
        let mut buf = String::new();
        let mut r = Cursor::new(b"{\"id\":2,\"kind\":\"bye\"}\r\n".to_vec());
        let e = read_request(&mut r, &mut buf).unwrap_err();
        assert_eq!(e.kind(), ErrorKind::InvalidData);
        assert!(e.to_string().contains("carriage return"));

        let mut r = Cursor::new(b"\n{\"id\":3,\"kind\":\"bye\"}\n".to_vec());
        assert!(read_request(&mut r, &mut buf).unwrap_err().to_string().contains("empty line"));
        assert!(matches!(read_request(&mut r, &mut buf), Ok(Some(Request::Bye(b))) if b.id == 3));

        let mut r = Cursor::new(vec![0xff, 0xfe, b'\n']);
        assert_eq!(read_request(&mut r, &mut buf).unwrap_err().kind(), ErrorKind::InvalidData);

        let mut r = Cursor::new(b"{\"id\":1,\"kind\":\"bye\",\"x\":1}\n".to_vec());
        assert!(read_request(&mut r, &mut buf).unwrap_err().to_string().contains("unknown field"));
    }

    #[test]
    fn over_long_line_is_drained() {
        let mut data = vec![b'{'; MAX_LINE + 10];
        data.push(b'\n');
        data.extend_from_slice(b"{\"id\":9,\"kind\":\"bye\"}\n");
        let mut r = Cursor::new(data);
        let mut buf = String::new();
        let e = read_request(&mut r, &mut buf).unwrap_err();
        assert!(e.to_string().contains("MAX_LINE"));
        assert!(matches!(read_request(&mut r, &mut buf), Ok(Some(Request::Bye(b))) if b.id == 9));
    }

    #[test]
    fn write_is_one_line() {
        let mut out = Vec::new();
        write_response(&mut out, &Response::ByeOk { id: 7 }).unwrap();
        assert_eq!(out, b"{\"kind\":\"bye_ok\",\"id\":7}\n");
        assert_eq!(peek_id("{\"id\":12,\"kind\":\"nope\"}"), Some(12));
        assert_eq!(peek_id("not json"), None);
    }
}
