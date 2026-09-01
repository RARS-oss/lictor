// SPDX-License-Identifier: MIT
//! OWNER: WP-6. Stub written by WP-0; replace the bodies (and add the private field), keep the signatures.
//! Latency histogram (hdrhistogram::Histogram<u64>, 1ns..10s, 3 sig figs).

use lictor_receipt::LatencySummary;

/// Private field (the hdrhistogram) is WP-6's to add.
pub struct Hist {}

impl Hist {
    pub fn new() -> Self {
        todo!("WP-6")
    }

    pub fn record(&mut self, _ns: u64) {
        todo!("WP-6")
    }

    pub fn summary(&self, _label: &str) -> LatencySummary {
        todo!("WP-6")
    }

    pub fn to_csv(&self) -> String {
        todo!("WP-6")
    }
}

impl Default for Hist {
    fn default() -> Self {
        Self::new()
    }
}

pub const WSL2_LABEL: &str =
    "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment";
