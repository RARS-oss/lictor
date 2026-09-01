// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; replace the bodies, keep the signatures.
//! NDJSON framing: one request per line in, one response per line out.

use crate::wire::{Request, Response};

pub const MAX_LINE: usize = 1 << 20;

/// None at EOF; Err(InvalidData) on malformed
pub fn read_request(_r: &mut impl std::io::BufRead, _buf: &mut String) -> std::io::Result<Option<Request>> {
    todo!("WP-6")
}

/// one line + flush
pub fn write_response(_w: &mut impl std::io::Write, _resp: &Response) -> std::io::Result<()> {
    todo!("WP-6")
}
