// SPDX-License-Identifier: MIT
//! Subcommand table and dispatch. Each command module owns its clap `Args` and `pub fn run(a: Args, json: bool) ->
//! anyhow::Result<i32>`; the returned integer is the process exit code. An `Err` is printed to stderr and exits 3.

pub mod ack;
pub mod bench;
pub mod calibrate;
pub mod crash_receipt;
pub mod curve;
pub mod envelope;
pub mod history;
pub mod key;
pub mod ledger;
pub mod replay;
pub mod selftest;
pub mod serve;
pub mod sweep;
pub mod verify;
pub mod version;

use clap::Subcommand;

use crate::Cli;

#[derive(Subcommand)]
pub enum Cmd {
    /// NDJSON fuse server on stdin/stdout (lictor-wire/v1)
    Serve(serve::Args),
    /// Verify a signed receipt (and optionally its ticks, timing, ledger, curve, calibration)
    Verify(verify::Args),
    /// Replay a request trace N times and check byte-identical verdict chains
    Replay(replay::Args),
    /// Offline split-conformal calibration from a recorded arm
    Calibrate(calibrate::Args),
    /// Layer-A offline sweep over (alpha, detector)
    Sweep(sweep::Args),
    /// Layer-B closed-loop curve from receipts only
    Curve(curve::Args),
    /// Envelope tools: init, check, digest, show, fit
    Envelope(envelope::Args),
    /// Decision-path latency and allocation benchmark
    Bench(bench::Args),
    /// End-to-end self test in a temporary directory
    Selftest(selftest::Args),
    /// Ledger tools: verify, append
    Ledger(ledger::Args),
    /// Host-side accounting for a serve child that died before episode_end
    CrashReceipt(crash_receipt::Args),
    /// Key tools: init, pub
    Key(key::Args),
    /// Sign an operator acknowledgement for a pending handoff
    Ack(ack::Args),
    /// Summarise .lictor/history.jsonl
    History(history::Args),
    /// Print version, git sha and the binary's sha256
    Version(version::Args),
}

pub fn dispatch(cli: Cli) -> i32 {
    let json = cli.json;
    let r = match cli.cmd {
        Cmd::Serve(a) => serve::run(a, json),
        Cmd::Verify(a) => verify::run(a, json),
        Cmd::Replay(a) => replay::run(a, json),
        Cmd::Calibrate(a) => calibrate::run(a, json),
        Cmd::Sweep(a) => sweep::run(a, json),
        Cmd::Curve(a) => curve::run(a, json),
        Cmd::Envelope(a) => envelope::run(a, json),
        Cmd::Bench(a) => bench::run(a, json),
        Cmd::Selftest(a) => selftest::run(a, json),
        Cmd::Ledger(a) => ledger::run(a, json),
        Cmd::CrashReceipt(a) => crash_receipt::run(a, json),
        Cmd::Key(a) => key::run(a, json),
        Cmd::Ack(a) => ack::run(a, json),
        Cmd::History(a) => history::run(a, json),
        Cmd::Version(a) => version::run(a, json),
    };
    match r {
        Ok(code) => code,
        Err(e) => {
            eprintln!("lictor: error: {e:#}");
            3
        }
    }
}
