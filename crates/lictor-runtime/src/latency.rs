// SPDX-License-Identifier: MIT
//! Latency histogram: an `hdrhistogram::Histogram<u64>` over 1 ns .. 10 s with 3 significant figures.
//!
//! Every summary carries the label the session was configured with (`SessionConfig.latency_label`, by default
//! `default_latency_label()`), so no latency number leaves the process without its environment statement.

use hdrhistogram::Histogram;
use lictor_receipt::LatencySummary;

const MIN_NS: u64 = 1;
const MAX_NS: u64 = 10_000_000_000;
const SIGFIG: u8 = 3;

pub struct Hist {
    h: Histogram<u64>,
}

impl Hist {
    pub fn new() -> Self {
        let h = Histogram::<u64>::new_with_bounds(MIN_NS, MAX_NS, SIGFIG)
            .expect("hdrhistogram bounds 1ns..10s at 3 significant figures are valid");
        Self { h }
    }

    /// Record one sample (saturated into the histogram's range; 0 ns is recorded as 1 ns).
    pub fn record(&mut self, ns: u64) {
        self.h.saturating_record(ns.max(MIN_NS));
    }

    pub fn summary(&self, label: &str) -> LatencySummary {
        if self.h.is_empty() {
            return LatencySummary {
                n: 0,
                p50_ns: 0,
                p90_ns: 0,
                p99_ns: 0,
                p999_ns: 0,
                max_ns: 0,
                label: label.to_string(),
            };
        }
        LatencySummary {
            n: u32::try_from(self.h.len()).unwrap_or(u32::MAX),
            p50_ns: self.h.value_at_quantile(0.50),
            p90_ns: self.h.value_at_quantile(0.90),
            p99_ns: self.h.value_at_quantile(0.99),
            p999_ns: self.h.value_at_quantile(0.999),
            max_ns: self.h.max(),
            label: label.to_string(),
        }
    }

    /// Bucket dump for F5: `value_ns,count,cumulative_count,percentile`, one row per recorded value.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("value_ns,count,cumulative_count,percentile\n");
        let mut cumulative = 0u64;
        for v in self.h.iter_recorded() {
            cumulative += v.count_since_last_iteration();
            s.push_str(&format!(
                "{},{},{},{:.6}\n",
                v.value_iterated_to(),
                v.count_at_value(),
                cumulative,
                v.percentile()
            ));
        }
        s
    }

    /// Number of samples recorded.
    pub fn len(&self) -> u64 {
        self.h.len()
    }

    pub fn is_empty(&self) -> bool {
        self.h.is_empty()
    }
}

impl Default for Hist {
    fn default() -> Self {
        Self::new()
    }
}

pub const WSL2_LABEL: &str =
    "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_summary_is_zero() {
        let h = Hist::new();
        let s = h.summary("lbl");
        assert_eq!((s.n, s.p50_ns, s.max_ns), (0, 0, 0));
        assert_eq!(s.label, "lbl");
        assert!(h.is_empty());
        assert!(h.to_csv().starts_with("value_ns,count"));
    }

    #[test]
    fn quantiles_are_monotone() {
        let mut h = Hist::new();
        for i in 1..=1000u64 {
            h.record(i * 100);
        }
        h.record(0);
        let s = h.summary(WSL2_LABEL);
        assert_eq!(s.n, 1001);
        assert_eq!(h.len(), 1001);
        assert!(
            s.p50_ns <= s.p90_ns && s.p90_ns <= s.p99_ns && s.p99_ns <= s.p999_ns && s.p999_ns <= s.max_ns
        );
        assert!(s.max_ns >= 99_000 && s.max_ns <= 100_100);
        let csv = h.to_csv();
        assert!(csv.lines().count() > 100);
        assert!(csv.lines().last().unwrap().ends_with("100.000000"));
    }
}
