// SPDX-License-Identifier: MIT
#![deny(unsafe_code)]
//! The `lictor` binary. Every command accepts `--json`. Exit codes: 0 ok; 1 verification failed / fuse not held /
//! determinism mismatch; 2 usage; 3 internal. The ONLY unsafe in this binary is the gated counting allocator in
//! `alloc_count.rs` (allowed at the module declaration below; the crate is otherwise `deny(unsafe_code)`).

#[allow(unsafe_code)]
mod alloc_count;
mod cmd;
mod render;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "lictor",
    version,
    about = "lictor: a deterministic safety fuse over learned robot policies"
)]
struct Cli {
    #[command(subcommand)]
    cmd: cmd::Cmd,
    /// Machine-readable output (every command)
    #[arg(long, global = true)]
    json: bool,
}

fn main() {
    std::process::exit(cmd::dispatch(Cli::parse()))
}
