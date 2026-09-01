// SPDX-License-Identifier: MIT
#![forbid(unsafe_code)]
//! lictor-calib: the offline lane -- calibration.json, trace loading, split-conformal calibration, envelope fit,
//! statistics, the Layer-A sweep and the Layer-B curve. Off the decision path (std; may use `ln`/`exp`/`lgamma`-style
//! functions and `f64::total_cmp` sorting). Depends on lictor-runtime for the ONE wire parser (`TraceReader`).

pub mod calibrate;
pub mod curve;
pub mod envfit;
pub mod file;
pub mod metrics;
pub mod sweep;
pub mod traces;
