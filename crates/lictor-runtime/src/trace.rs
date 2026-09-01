// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; replace the bodies (and add the private fields), keep the signatures.
//! Request trace files: "#meta {json}" first line, then every request line VERBATIM.

use crate::wire::Request;

pub struct TraceWriter;

impl TraceWriter {
    pub fn open(_p: &std::path::Path, _meta: &serde_json::Value) -> std::io::Result<Self> {
        todo!("WP-6")
    }

    pub fn line(&mut self, _raw: &str) -> std::io::Result<()> {
        todo!("WP-6")
    }
}

pub struct TraceReader;

impl TraceReader {
    pub fn open(_p: &std::path::Path) -> std::io::Result<(Self, serde_json::Value)> {
        todo!("WP-6")
    }
}

impl Iterator for TraceReader {
    /// (line_no, raw, parsed)
    type Item = std::io::Result<(u64, String, Request)>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!("WP-6")
    }
}
