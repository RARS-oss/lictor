// SPDX-License-Identifier: MIT
//! Ed25519 signing and verification of receipts; ticks-file verification.
//!
//! Vendored from RARS-oss/bulla @ 173e1cd65fb43353a9752f95077937aa4bb0f8e2, crates/bulla-core/src/lib.rs
//! (`sign`, `verify`, the hex key/signature plumbing); renamed per docs/ANALYSIS.md sec 5; canonical bytes
//! replaced by lictor-canon JCS (`jcs-floatfree/v1`) instead of serde field order. The divergences are listed
//! in docs/receipt-schema.md.
//!
//! One signature per episode over `canon(body)`; `body_digest = sha256(canon(body))`. Verification is
//! `verify_strict` (rejects non-canonical points and small-order keys). `VerifyReport::intact()` is the
//! verdict on the RECORD; `fuse_ok` is the verdict on what the record says the fuse did -- `intact != fuse_ok`.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use lictor_canon::{sha256_hex, CanonError, CANONICAL_ID};

use crate::body::{evaluate_fuse, ReceiptBody, NOTE_EPHEMERAL_KEY};
use crate::tick::{verify_tick_chain, ChainReport, TickEvent};
use crate::{RECEIPT_SCHEMA, ZERO_HASH};

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedReceipt {
    pub body: ReceiptBody,
    pub body_digest: String,
    pub pubkey: String,
    pub sig: String,
}

/// Sign `bytes` with the Ed25519 seed; returns `(pubkey_hex, sig_hex)`.
pub(crate) fn sign_bytes(seed: &[u8; 32], bytes: &[u8]) -> (String, String) {
    let sk = SigningKey::from_bytes(seed);
    let pubkey = hex::encode(sk.verifying_key().to_bytes());
    let sig = hex::encode(sk.sign(bytes).to_bytes());
    (pubkey, sig)
}

/// `verify_strict` of `sig_hex` over `bytes` under `pubkey_hex`; the error text names the first defect.
pub(crate) fn verify_bytes(pubkey_hex: &str, sig_hex: &str, bytes: &[u8]) -> Result<(), String> {
    let pk: [u8; 32] = hex::decode(pubkey_hex)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| "public key is not 32 hex bytes".to_string())?;
    let vk =
        VerifyingKey::from_bytes(&pk).map_err(|_| "public key is not a valid Ed25519 point".to_string())?;
    let sig: [u8; 64] = hex::decode(sig_hex)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or_else(|| "signature is not 64 hex bytes".to_string())?;
    vk.verify_strict(bytes, &Signature::from_bytes(&sig))
        .map_err(|_| "Ed25519 signature does not verify against the canonical bytes".to_string())
}

pub fn sign(body: ReceiptBody, seed: &[u8; 32]) -> Result<SignedReceipt, ReceiptError> {
    let canonical = body.canonical()?;
    let (pubkey, sig) = sign_bytes(seed, &canonical);
    Ok(SignedReceipt { body_digest: sha256_hex(&canonical), pubkey, sig, body })
}

#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct VerifyReport {
    pub schema_ok: bool,
    pub sig_ok: bool,
    pub digest_ok: bool,
    pub ticks_ok: bool,
    pub pubkey_ok: bool,
    pub envelope_digest_ok: bool,
    pub counts_ok: bool,
    pub fuse_ok: bool,
    pub break_at: Option<u32>,
    pub notes: Vec<String>,
}

/// pubkey_ok = (expect_pubkey.is_none() || sr.pubkey == expected). envelope_digest_ok = recomputed sha256(canon(body.envelope)) == body.envelope_digest.
/// counts_ok = counts.ticks == verdict_events == timing_events == outcome.steps.
impl VerifyReport {
    pub fn intact(&self) -> bool {
        self.schema_ok
            && self.sig_ok
            && self.digest_ok
            && self.ticks_ok
            && self.pubkey_ok
            && self.envelope_digest_ok
            && self.counts_ok
    }

    /// A report for a signed object whose only checks are schema, digest, signature and pubkey (the curve
    /// receipt): the per-episode checks are `true` because they do not apply.
    pub(crate) fn from_signature_checks(
        schema_ok: bool,
        canonical: Result<Vec<u8>, CanonError>,
        body_digest: &str,
        pubkey: &str,
        sig: &str,
        expect_pubkey: Option<&str>,
        notes: &mut Vec<String>,
    ) -> Self {
        let (digest_ok, sig_ok) = match canonical {
            Ok(bytes) => {
                let digest_ok = sha256_hex(&bytes) == body_digest;
                if !digest_ok {
                    notes.push("body_digest does not match the recomputed canonical bytes".into());
                }
                let sig_ok = match verify_bytes(pubkey, sig, &bytes) {
                    Ok(()) => true,
                    Err(e) => {
                        notes.push(e);
                        false
                    }
                };
                (digest_ok, sig_ok)
            }
            Err(e) => {
                notes.push(format!("body is not canonicalisable: {e}"));
                (false, false)
            }
        };
        let pubkey_ok = expect_pubkey.is_none_or(|p| p == pubkey);
        if !pubkey_ok {
            notes.push("pubkey mismatch: the receipt was signed by a different key than expected".into());
        }
        if !schema_ok {
            notes.push("schema or canonical profile id is not the one this verifier understands".into());
        }
        VerifyReport {
            schema_ok,
            sig_ok,
            digest_ok,
            ticks_ok: true,
            pubkey_ok,
            envelope_digest_ok: true,
            counts_ok: true,
            fuse_ok: true,
            break_at: None,
            notes: Vec::new(),
        }
    }
}

/// Chain-check the embedded ticks against the body's policy and head. Returns `(ticks_ok, break_at)`.
fn check_embedded_ticks(body: &ReceiptBody, notes: &mut Vec<String>) -> (bool, Option<u32>) {
    let ticks = &body.ticks;
    let n_events = body.verdict_events;
    let expected_len = match body.ticks_policy.as_str() {
        "all" => n_events as usize,
        "tail32" => (n_events as usize).min(32),
        "none" => 0,
        other => {
            notes.push(format!("unknown ticks_policy {other:?}"));
            return (false, None);
        }
    };
    if ticks.len() != expected_len {
        notes.push(format!(
            "ticks_policy {} implies {} embedded ticks, found {}",
            body.ticks_policy,
            expected_len,
            ticks.len()
        ));
        return (false, None);
    }
    if ticks.is_empty() {
        // Nothing embedded: the head must be the genesis when there were no events at all.
        if n_events == 0 && body.verdict_chain_head != ZERO_HASH {
            notes.push("verdict_events == 0 but verdict_chain_head is not the genesis".into());
            return (false, None);
        }
        return (true, None);
    }
    let genesis = if body.ticks_policy == "all" { ZERO_HASH } else { ticks[0].prev.as_str() };
    let report = verify_tick_chain(ticks, genesis);
    if !report.ok {
        notes.push(format!("embedded tick chain breaks at seq {:?}", report.break_at));
        return (false, report.break_at);
    }
    if body.ticks_policy == "tail32" && ticks[0].seq != n_events.saturating_sub(ticks.len() as u32) {
        notes.push("embedded tail does not end at verdict_events".into());
        return (false, Some(ticks[0].seq));
    }
    if report.head != body.verdict_chain_head {
        notes.push("embedded tick chain head differs from verdict_chain_head".into());
        return (false, Some(ticks[ticks.len() - 1].seq));
    }
    (true, None)
}

pub fn verify(sr: &SignedReceipt, expect_pubkey: Option<&str>) -> VerifyReport {
    let body = &sr.body;
    let mut notes = Vec::new();
    let schema_ok = body.schema == RECEIPT_SCHEMA && body.canonical == CANONICAL_ID;
    let base = VerifyReport::from_signature_checks(
        schema_ok,
        body.canonical(),
        &sr.body_digest,
        &sr.pubkey,
        &sr.sig,
        expect_pubkey,
        &mut notes,
    );

    let envelope_digest_ok = match lictor_canon::canon_of(&body.envelope) {
        Ok(bytes) => sha256_hex(&bytes) == body.envelope_digest,
        Err(_) => false,
    };
    if !envelope_digest_ok {
        notes.push("envelope_digest does not match sha256(canon(envelope))".into());
    }

    let c = &body.counts;
    let counts_ok = c.ticks == body.verdict_events
        && body.verdict_events == body.timing_events
        && body.timing_events == body.outcome.steps;
    if !counts_ok {
        notes.push(format!(
            "counts disagree: counts.ticks {} verdict_events {} timing_events {} outcome.steps {}",
            c.ticks, body.verdict_events, body.timing_events, body.outcome.steps
        ));
    }

    let (ticks_ok, break_at) = check_embedded_ticks(body, &mut notes);

    let ephemeral = body.fuse_notes.iter().any(|n| n == NOTE_EPHEMERAL_KEY);
    let (recomputed_ok, recomputed_notes) =
        evaluate_fuse(&body.budget, c, body.calibration_digest.as_deref(), ephemeral);
    if recomputed_ok != body.fuse_ok {
        notes.push(format!(
            "fuse_ok flag {} disagrees with the recomputed evaluation {}",
            body.fuse_ok, recomputed_ok
        ));
    }
    if recomputed_notes != body.fuse_notes {
        notes.push("fuse_notes differ from the recomputed evaluation".into());
    }
    notes.extend(recomputed_notes);
    let fuse_ok = recomputed_ok && body.fuse_ok;

    VerifyReport {
        schema_ok,
        sig_ok: base.sig_ok,
        digest_ok: base.digest_ok,
        ticks_ok,
        pubkey_ok: base.pubkey_ok,
        envelope_digest_ok,
        counts_ok,
        fuse_ok,
        break_at,
        notes,
    }
}

/// Recompute the verdict chain from a full ticks file. `ChainReport.ok` is CHAIN INTEGRITY ONLY (a re-chained file passes it);
/// the caller MUST compare `report.head` with `sr.body.verdict_chain_head` and report `HEAD MISMATCH` when they differ, and MUST
/// cross-check the ticks header (run_id, arm_id, episode_index, genesis) against the body and the embedded tail's first `prev`.
pub fn verify_ticks_file(sr: &SignedReceipt, ticks: &[TickEvent]) -> ChainReport {
    // The head and header comparisons are the caller's by the frozen contract; the receipt is taken so a future
    // ticks profile can pick its genesis from it. Under `lictor-ticks/v1` the genesis is always ZERO_HASH.
    let _ = sr;
    verify_tick_chain(ticks, ZERO_HASH)
}

/// The two committed TEST keys (their pubkeys), so `lictor verify` can warn "signed with the committed test key".
/// [0] = crates/lictor-receipt/tests/fixtures/receipt/key.hex (seed 000102..1f);
/// [1] = bench/fixtures/key.hex (seed 404142..5f; WP-10 commits that file with exactly this seed).
pub const TEST_PUBKEYS: [&str; 2] = [
    "03a107bff3ce10be1d70dd18e74bc09967e4d6309ba50d5f1ddc8664125531b8",
    "2543b92ff1095511476adc8369db6ddc933665a11978dda1404ee1066ca9559d",
];

#[derive(Debug, thiserror::Error)]
pub enum ReceiptError {
    #[error("canon: {0}")]
    Canon(#[from] lictor_canon::CanonError),
    #[error("key: {0}")]
    Key(String),
    #[error("io: {0}")]
    Io(String),
}

impl From<std::io::Error> for ReceiptError {
    fn from(e: std::io::Error) -> Self {
        ReceiptError::Io(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sign_and_verify_bytes_round_trip() {
        let seed = [7u8; 32];
        let (pk, sig) = sign_bytes(&seed, b"hello");
        assert_eq!(pk.len(), 64);
        assert_eq!(sig.len(), 128);
        assert!(verify_bytes(&pk, &sig, b"hello").is_ok());
        assert!(verify_bytes(&pk, &sig, b"hellp").is_err());
        assert!(verify_bytes("00", &sig, b"hello").is_err());
        assert!(verify_bytes(&pk, "00", b"hello").is_err());
    }

    #[test]
    fn test_pubkeys_match_the_committed_seeds() {
        let fixture_seed: [u8; 32] = core::array::from_fn(|i| i as u8);
        let bench_seed: [u8; 32] = core::array::from_fn(|i| 0x40 + i as u8);
        assert_eq!(TEST_PUBKEYS[0], crate::keys::pubkey_hex(&fixture_seed), "fixture key");
        assert_eq!(TEST_PUBKEYS[1], crate::keys::pubkey_hex(&bench_seed), "bench key");
    }
}
