// SPDX-License-Identifier: MIT
//! The VERDICT chain (replayable) and the TIMING chain (honestly not).
//!
//! `h_0 = ZERO_HASH`; `h_i = sha256(canon(event_i with hash = ""))` where `event_i.prev = h_{i-1}`. A TickEvent
//! carries only what a replay reproduces byte-for-byte; a TimingEvent carries measured wall-clock and is chained
//! for tamper-evidence only. Hashing happens on the I/O thread after `decide()` returns (docs/ARCHITECTURE.md sec 8).

use lictor_canon::{digest_of, F64Array, F64Hex};
use lictor_core::{ActionSource, FuseState, ReasonCode, SafetyVerdict, Status, NFEAT};

use crate::ZERO_HASH;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TickEvent {
    pub seq: u32,
    pub t: u32,
    pub state: FuseState,
    pub prev_state: FuseState,
    pub status: Status,
    pub trips: u32,
    pub action_src: ActionSource,
    pub substituted: bool,
    pub clamped_dims: u32,
    /// [action_dim]
    pub action: F64Array,
    /// [NFEAT] raw features (what `lictor sweep` re-scores)
    pub f: F64Array,
    /// [NFEAT]
    pub z: F64Array,
    pub valid: u32,
    pub fired: u32,
    pub s: F64Hex,
    pub tau: F64Hex,
    pub window_hits: u8,
    pub brake_margin: F64Hex,
    pub reason: ReasonCode,
    pub handoff_seq: Option<u32>,
    pub violation_reached_env: bool,
    pub prev: String,
    /// hash = sha256(canon(self with hash = "")) -- see docs/receipt-schema.md
    pub hash: String,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimingEvent {
    pub seq: u32,
    pub decide_ns: u64,
    pub io_ns: u64,
    pub prev: String,
    pub hash: String,
}

/// sha256(canon(event)) for an event whose `hash` field is already "". Every chained event is float-free and
/// key-clean by construction, so canonicalisation cannot fail on data; a failure here is a programming error.
fn chained_hash<T: serde::Serialize>(event_with_empty_hash: &T) -> String {
    digest_of(event_with_empty_hash).expect("chained events are float-free and key-clean by construction")
}

impl TickEvent {
    /// Recompute this event's hash from its content (with `hash` blanked).
    pub fn recompute_hash(&self) -> String {
        let mut e = self.clone();
        e.hash = String::new();
        chained_hash(&e)
    }
}

impl TimingEvent {
    /// Recompute this event's hash from its content (with `hash` blanked).
    pub fn recompute_hash(&self) -> String {
        let mut e = self.clone();
        e.hash = String::new();
        chained_hash(&e)
    }
}

pub fn tick_event(prev: &str, v: &SafetyVerdict) -> TickEvent {
    let dim = (v.action_dim as usize).min(v.action.len());
    let mut e = TickEvent {
        seq: v.seq,
        t: v.t,
        state: v.state,
        prev_state: v.prev_state,
        status: v.status,
        trips: v.trips,
        action_src: v.action_src,
        substituted: v.substituted,
        clamped_dims: v.clamped_dims,
        action: F64Array::from_slice(&v.action[..dim], &[dim as u32]),
        f: F64Array::from_slice(&v.scores.f, &[NFEAT as u32]),
        z: F64Array::from_slice(&v.scores.z, &[NFEAT as u32]),
        valid: v.scores.valid,
        fired: v.scores.fired,
        s: F64Hex(v.scores.s),
        tau: F64Hex(v.tau),
        window_hits: v.window_hits,
        brake_margin: F64Hex(v.brake_margin),
        reason: v.reason,
        handoff_seq: v.handoff_seq,
        violation_reached_env: v.violation_reached_env,
        prev: prev.to_string(),
        hash: String::new(),
    };
    e.hash = chained_hash(&e);
    e
}

pub fn timing_event(prev: &str, seq: u32, decide_ns: u64, io_ns: u64) -> TimingEvent {
    let mut e = TimingEvent { seq, decide_ns, io_ns, prev: prev.to_string(), hash: String::new() };
    e.hash = chained_hash(&e);
    e
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChainReport {
    pub ok: bool,
    pub break_at: Option<u32>,
    pub head: String,
    pub n: u32,
}

/// Walk a chain: every event must link to the previous hash (the first to `genesis_prev`), carry a sequence
/// number that continues the first event's, and recompute to its own `hash`. `break_at` is the seq of the first
/// bad event; `head` is the last hash seen (or the genesis when the slice is empty).
fn walk<T>(
    events: &[T],
    genesis_prev: &str,
    seq_of: impl Fn(&T) -> u32,
    prev_of: impl Fn(&T) -> &str,
    hash_of: impl Fn(&T) -> &str,
    recompute: impl Fn(&T) -> String,
) -> ChainReport {
    let mut prev = genesis_prev.to_string();
    let mut break_at = None;
    let first_seq = events.first().map(&seq_of).unwrap_or(0);
    for (i, e) in events.iter().enumerate() {
        let expected_seq = first_seq.wrapping_add(i as u32);
        let good = seq_of(e) == expected_seq && prev_of(e) == prev && recompute(e) == hash_of(e);
        if !good && break_at.is_none() {
            break_at = Some(if seq_of(e) == expected_seq { seq_of(e) } else { expected_seq });
        }
        prev = hash_of(e).to_string();
    }
    ChainReport { ok: break_at.is_none(), break_at, head: prev, n: events.len() as u32 }
}

pub fn verify_tick_chain(events: &[TickEvent], genesis_prev: &str) -> ChainReport {
    walk(events, genesis_prev, |e| e.seq, |e| e.prev.as_str(), |e| e.hash.as_str(), TickEvent::recompute_hash)
}

pub fn verify_timing_chain(events: &[TimingEvent]) -> ChainReport {
    walk(events, ZERO_HASH, |e| e.seq, |e| e.prev.as_str(), |e| e.hash.as_str(), TimingEvent::recompute_hash)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lictor_core::{Scores, MAX_D};

    fn verdict(seq: u32) -> SafetyVerdict {
        let mut action = [0.0; MAX_D];
        action[0] = 13.375;
        action[1] = 300.4;
        SafetyVerdict {
            seq,
            t: seq,
            status: Status::Nominal,
            state: FuseState::Armed,
            prev_state: FuseState::Armed,
            trips: 0,
            action,
            action_dim: 2,
            action_src: ActionSource::Policy,
            substituted: false,
            clamped_dims: 0,
            scores: Scores { s: f64::NEG_INFINITY, valid: 0xff, ..Scores::default() },
            tau: f64::INFINITY,
            window: 0,
            window_hits: 0,
            brake_margin: 41.2,
            reason: ReasonCode::Ok,
            handoff_seq: None,
            ack_consumed: false,
            violation_reached_env: false,
        }
    }

    #[test]
    fn tick_chain_links_and_breaks_where_edited() {
        let mut prev = ZERO_HASH.to_string();
        let mut events = Vec::new();
        for i in 0..5 {
            let e = tick_event(&prev, &verdict(i));
            assert_eq!(e.action.data, vec![13.375, 300.4]);
            assert_eq!(e.action.shape, vec![2]);
            assert_eq!(e.hash, e.recompute_hash());
            prev = e.hash.clone();
            events.push(e);
        }
        let r = verify_tick_chain(&events, ZERO_HASH);
        assert!(r.ok && r.break_at.is_none() && r.n == 5 && r.head == events[4].hash);
        // tail verification from the tail's own first prev
        let tail = &events[2..];
        let r = verify_tick_chain(tail, &tail[0].prev);
        assert!(r.ok && r.head == events[4].hash);
        // edit one field: the hash no longer recomputes at that seq
        let mut edited = events.clone();
        edited[3].trips = 1;
        let r = verify_tick_chain(&edited, ZERO_HASH);
        assert_eq!((r.ok, r.break_at), (false, Some(3)));
        // drop one: linkage/seq break reported at the position of the missing event
        let mut dropped = events.clone();
        dropped.remove(2);
        let r = verify_tick_chain(&dropped, ZERO_HASH);
        assert_eq!((r.ok, r.break_at), (false, Some(2)));
        // empty chain: head is the genesis
        let r = verify_tick_chain(&[], ZERO_HASH);
        assert!(r.ok && r.n == 0 && r.head == ZERO_HASH);
    }

    #[test]
    fn timing_chain() {
        let a = timing_event(ZERO_HASH, 0, 1180, 41000);
        let b = timing_event(&a.hash, 1, 1200, 39000);
        let r = verify_timing_chain(&[a.clone(), b.clone()]);
        assert!(r.ok && r.head == b.hash);
        let mut c = b.clone();
        c.io_ns += 1;
        let r = verify_timing_chain(&[a, c]);
        assert_eq!(r.break_at, Some(1));
    }
}
