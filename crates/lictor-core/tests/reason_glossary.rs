// SPDX-License-Identifier: MIT
//! The human-escalation glossary: every reason code has one plain-English sentence, <= 120 chars, ending with a
//! period, ASCII only, and free of the banned vocabulary (IMPLEMENTATION_PLAN sec 0 rule 4).

use lictor_core::{reason_text, ReasonCode};

/// Banned as claims about lictor anywhere, including these sentences. Matched case-insensitively.
const BANNED: &[&str] = &[
    "safety-rated",
    "hard real-time",
    "hard-real-time",
    "hard real time",
    "pl d",
    "sil 2",
    "certified",
    "engineered to",
    "principles",
    "proof",
    "can never hide",
    "bit-identical on every platform",
    "guarantee",
];

#[test]
fn every_code_has_a_short_plain_sentence() {
    assert_eq!(ReasonCode::ALL.len(), 28);
    for r in ReasonCode::ALL {
        let t = reason_text(r);
        assert!(!t.is_empty(), "{r:?} has no text");
        assert!(t.len() <= 120, "{r:?}: {} chars > 120: {t}", t.len());
        assert!(t.ends_with('.'), "{r:?}: must end with a period: {t}");
        assert!(t.is_ascii(), "{r:?}: ASCII only: {t}");
        assert!(!t.contains('\n') && !t.contains('\t'), "{r:?}: single line: {t}");
        assert!(t.chars().next().unwrap().is_ascii_uppercase(), "{r:?}: starts with a capital: {t}");
        let lower = t.to_ascii_lowercase();
        for b in BANNED {
            assert!(!lower.contains(b), "{r:?}: contains banned vocabulary `{b}`: {t}");
        }
    }
}

#[test]
fn sentences_are_distinct_and_codes_are_distinct() {
    let mut texts: Vec<&str> = ReasonCode::ALL.iter().map(|r| reason_text(*r)).collect();
    texts.sort_unstable();
    texts.dedup();
    assert_eq!(texts.len(), ReasonCode::ALL.len(), "two codes share a sentence");
    let mut names: Vec<String> = ReasonCode::ALL
        .iter()
        .map(|r| serde_json::to_value(r).unwrap().as_str().unwrap().to_string())
        .collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), ReasonCode::ALL.len());
}

#[test]
fn wire_names_are_snake_case() {
    assert_eq!(serde_json::to_value(ReasonCode::BrakeTier1Cp).unwrap(), "brake_tier1_cp");
    assert_eq!(serde_json::to_value(ReasonCode::EscalateHoldTimeout).unwrap(), "escalate_hold_timeout");
    assert_eq!(
        serde_json::from_str::<ReasonCode>("\"fault_non_finite\"").unwrap(),
        ReasonCode::FaultNonFinite
    );
    for r in ReasonCode::ALL {
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(serde_json::from_str::<ReasonCode>(&s).unwrap(), r);
        assert!(s
            .trim_matches('"')
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_'));
    }
}

#[test]
fn the_glossary_examples_from_the_plan_hold() {
    assert_eq!(
        reason_text(ReasonCode::BrakeInfeasible),
        "The committed motion could not be stopped inside the workspace; the fuse is braking."
    );
    assert!(reason_text(ReasonCode::FaultNonFinite).contains("non-finite"));
    assert!(reason_text(ReasonCode::EscalateHoldTimeout).contains("a person must decide"));
    assert!(reason_text(ReasonCode::ObserveOnly).contains("did not change the action"));
}
