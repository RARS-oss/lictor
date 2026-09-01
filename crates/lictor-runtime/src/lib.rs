// SPDX-License-Identifier: MIT
#![forbid(unsafe_code)]
//! lictor-runtime: the `lictor-wire/v1` schema and NDJSON codec, the `Session` (shared by serve, replay, bench and
//! selftest), the episode writer, the latency histogram and request trace files. Off the decision path (std).
//! Does NOT depend on lictor-calib: the CLI loads calibration.json and hands the runtime a compiled `CalibrationLoaded`.

pub mod codec;
pub mod episode;
pub mod latency;
pub mod session;
pub mod trace;
pub mod wire;
