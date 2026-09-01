// SPDX-License-Identifier: MIT
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! lictor-fuse: `decide()`, the escalation state machine, the pre-allocated runtime state `FuseRt` and the
//! per-episode `Tally`. The decision path: no allocation, no clock, no lock, no syscall, no panic on data,
//! fixed iteration bounds. Float discipline: `lictor_core::fmath` only. See docs/ARCHITECTURE.md sec 5-6.

pub mod fsm;
pub mod fuse;
pub mod tally;

pub use fsm::{next, FsmInput};
pub use fuse::{decide, Fuse, FuseRt, TickInput};
pub use tally::Tally;
