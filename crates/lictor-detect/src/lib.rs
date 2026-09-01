// SPDX-License-Identifier: MIT
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! lictor-detect: Tier-0 limits and projection, brake feasibility, the K-of-N window and position trail,
//! Tier-1 features and the conformal runtime rule. no_std-capable; nothing here allocates, reads a clock or
//! performs I/O. Float discipline: `lictor_core::fmath` only (see the DECISION-PATH DENY LIST in lictor-core).

pub mod brake;
pub mod conformal;
pub mod tier0;
pub mod tier1;
pub mod window;
