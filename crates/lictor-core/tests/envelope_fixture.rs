// SPDX-License-Identifier: MIT
//! Fixture tests for the envelope TOML surface: round trip, every `invalid_*.toml`, the compile constants of the
//! PushT base envelope, and the canonical digests (the digest cases need WP-4's `lictor_canon`).

use lictor_core::{
    ActionKind, BrakeKind, ContactLimit, EnvelopeError, FitRecord, FuseMode, GateSpec, SafetyEnvelope,
    TripMask,
};

const BASE: &str = include_str!("fixtures/envelope/pusht_base.toml");
const SHIPPED: &str = include_str!("../../../envelopes/pusht.base.toml");
const DIGESTS: &str = include_str!("fixtures/envelope/digest.json");

/// (fixture name, text, expected `EnvelopeError` variant)
const INVALID: &[(&str, &str, &str)] = &[
    ("invalid_n64", include_str!("fixtures/envelope/invalid_n64.toml"), "invalid"),
    (
        "invalid_horizon_ticks_zero",
        include_str!("fixtures/envelope/invalid_horizon_ticks_zero.toml"),
        "invalid",
    ),
    ("invalid_k_gt_n", include_str!("fixtures/envelope/invalid_k_gt_n.toml"), "invalid"),
    ("invalid_fail_open", include_str!("fixtures/envelope/invalid_fail_open.toml"), "invalid"),
    ("invalid_box", include_str!("fixtures/envelope/invalid_box.toml"), "invalid"),
    ("invalid_gate_name", include_str!("fixtures/envelope/invalid_gate_name.toml"), "invalid"),
    ("invalid_tier0_name", include_str!("fixtures/envelope/invalid_tier0_name.toml"), "invalid"),
    ("invalid_negative_vmax", include_str!("fixtures/envelope/invalid_negative_vmax.toml"), "invalid"),
    ("invalid_operator_hex", include_str!("fixtures/envelope/invalid_operator_hex.toml"), "invalid"),
    (
        "invalid_too_many_operators",
        include_str!("fixtures/envelope/invalid_too_many_operators.toml"),
        "invalid",
    ),
    ("invalid_exec_steps", include_str!("fixtures/envelope/invalid_exec_steps.toml"), "invalid"),
    (
        "invalid_contact_without_table",
        include_str!("fixtures/envelope/invalid_contact_without_table.toml"),
        "invalid",
    ),
    ("invalid_brake_period", include_str!("fixtures/envelope/invalid_brake_period.toml"), "invalid"),
    ("invalid_schema", include_str!("fixtures/envelope/invalid_schema.toml"), "unsupported"),
    ("invalid_unknown_key", include_str!("fixtures/envelope/invalid_unknown_key.toml"), "parse"),
    ("invalid_unknown_top_key", include_str!("fixtures/envelope/invalid_unknown_top_key.toml"), "parse"),
    ("invalid_enum", include_str!("fixtures/envelope/invalid_enum.toml"), "parse"),
    ("invalid_wrong_type", include_str!("fixtures/envelope/invalid_wrong_type.toml"), "parse"),
];

fn load(s: &str) -> Result<SafetyEnvelope, EnvelopeError> {
    let e = SafetyEnvelope::from_toml(s)?;
    e.validate()?;
    Ok(e)
}

fn kind(e: &EnvelopeError) -> &'static str {
    match e {
        EnvelopeError::Parse(_) => "parse",
        EnvelopeError::Invalid(_) => "invalid",
        EnvelopeError::Unsupported(_) => "unsupported",
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn fixture_is_byte_identical_to_the_shipped_envelope() {
    assert_eq!(BASE, SHIPPED, "tests/fixtures/envelope/pusht_base.toml must equal envelopes/pusht.base.toml");
    assert_eq!(BASE.lines().count(), 74);
    assert!(BASE.is_ascii());
}

#[test]
fn base_envelope_parses_and_validates() {
    let e = load(BASE).unwrap();
    assert_eq!(e.schema, "lictor-envelope/v1");
    assert_eq!(e.envelope_id, "pusht-base-v1");
    assert_eq!(e.embodiment.id, "gym_pusht/PushT-v0");
    assert_eq!(e.embodiment.action_kind, ActionKind::EePosition);
    assert_eq!((e.embodiment.horizon, e.embodiment.exec_steps), (15, 8));
    assert!(e.embodiment.provides_vel);
    assert_eq!(e.embodiment.horizon_ticks, 300);
    assert_eq!(e.embodiment.aux_layout, ["block_x", "block_y", "block_theta", "coverage"]);
    assert!(e.embodiment.ext_names.is_empty());
    assert_eq!(e.dt(), 0.1);
    assert_eq!(e.box_lo, [15.0, 15.0]);
    assert_eq!(e.box_hi, [497.0, 497.0]);
    assert_eq!((e.v_max, e.a_max, e.j_max, e.reach_max), (1000.0, 20000.0, 400000.0, 150.0));
    assert!(e.contact.is_none());
    assert!(e.fit.is_none());
    assert!(e.operators.is_empty());
    assert_eq!(e.brake.kind, BrakeKind::PdSecondOrder);
    assert_eq!((e.brake.k_p, e.brake.k_v, e.brake.substeps, e.brake.dt), (100.0, 20.0, 10, 0.01));
    assert_eq!((e.brake.commit_steps, e.brake.brake_steps, e.brake.react_ticks), (8, 8, 1));
    assert_eq!((e.hysteresis.k, e.hysteresis.n), (3, 5));
    assert_eq!(e.gate.len(), 8);
    assert_eq!(e.tier0_enabled, ["workspace", "speed", "accel", "jerk", "reach", "brake"]);
    assert!(e.fail_closed);
}

#[test]
fn toml_round_trip_is_identity() {
    let e = load(BASE).unwrap();
    let text = e.to_toml().unwrap();
    assert!(
        text.contains("[embodiment]") && text.contains("[brake]") && text.contains("[hysteresis]"),
        "{text}"
    );
    assert!(!text.contains("[contact]") && !text.contains("[fit]"), "{text}");
    // compact style: arrays inline, one key per line, tables after the scalars (readable diffs for `envelope fit`)
    assert!(text.contains("box_lo = [15.0, 15.0]"), "{text}");
    assert!(text.contains("kind = \"pd_second_order\""), "{text}");
    assert!(text.contains("operators = []"), "{text}");
    assert!(text.find("[embodiment]").unwrap() > text.find("fail_closed = true").unwrap(), "{text}");
    let e2 = load(&text).unwrap();
    assert_eq!(e, e2);

    let mut f = e.clone();
    f.contact = Some(ContactLimit { radius: 80.0, v_max: 400.0, aux_center: [0, 1] });
    f.tier0_enabled.push("contact".into());
    f.fit = Some(FitRecord {
        source_run: "2026-09-01T09-14Z-pilot".into(),
        quantile: 0.999,
        slack: 1.25,
        n_episodes: 196,
        fitted_utc: "2026-09-01T11:02:00Z".into(),
        note: "p99.9 of |v|,|a|,|j|,reach over calibration successes x 1.25".into(),
    });
    f.operators = vec!["ab".repeat(32)];
    let t2 = f.to_toml().unwrap();
    assert!(t2.contains("[contact]") && t2.contains("[fit]"), "{t2}");
    assert_eq!(load(&t2).unwrap(), f);
}

#[test]
fn compile_constants_match_the_plan() {
    let e = load(BASE).unwrap();
    let cfg = e.compile(FuseMode::Enforce, None).unwrap();
    assert_eq!(cfg.step_max, 100.0);
    assert_eq!(&cfg.box_lo[..2], &[17.0, 17.0]);
    assert_eq!(&cfg.box_hi[..2], &[495.0, 495.0]);
    assert_eq!(cfg.overlap, 7);
    assert_eq!(cfg.window_mask, 31);
    assert_eq!((cfg.dim, cfg.pos_dim, cfg.horizon, cfg.exec), (2, 2, 15, 8));
    assert_eq!(cfg.dt, 0.1);
    assert_eq!(cfg.inv_dt, 10.0);
    assert_eq!(cfg.inv_dt2, 100.0);
    assert_eq!(cfg.inv_dt3, 1000.0);
    assert_eq!(cfg.norm_scale_iso, 256.0);
    assert_eq!(&cfg.norm_center[..2], &[256.0, 256.0]);
    assert_eq!(cfg.inv_norm_scale[0], 1.0 / 256.0);
    assert_eq!(cfg.inv_norm_scale[2], 0.0);
    assert_eq!(
        cfg.tier0_enabled,
        TripMask::WORKSPACE
            | TripMask::SPEED
            | TripMask::ACCEL
            | TripMask::JERK
            | TripMask::REACH
            | TripMask::BRAKE
    );
    assert_eq!(cfg.n_operators, 0);
    assert_eq!(cfg.mode, FuseMode::Enforce);
    assert_eq!(cfg.kind, ActionKind::EePosition);
    assert_eq!(cfg.brake, e.brake);
    assert_eq!(cfg.hyst, e.hysteresis);
    assert!(cfg.contact.is_none());
    assert!(cfg.calib_digest.is_none());
    assert_eq!(cfg.calib.mask, 0xff);
    assert_eq!(cfg.calib.gate, GateSpec::parse(&e.gate).unwrap());
    assert_eq!(cfg.calib.tau, f64::INFINITY);
    assert_eq!(cfg.calib.horizon_ticks, 300);
    assert_eq!(cfg.calib.t_grid, 1);
    assert!(cfg.calib.armed());
}

#[test]
fn base_gate_and_tier0_names_round_trip() {
    let e = load(BASE).unwrap();
    let g = GateSpec::parse(&e.gate).unwrap();
    assert_eq!(g.n_terms, 8);
    assert_eq!(g.mask(), 0xff);
    assert_eq!(g.names(), e.gate);
    assert_eq!(e.tier0_mask().unwrap(), TripMask::TIER0_SOFT & !TripMask::CONTACT | TripMask::BRAKE);
}

#[test]
fn invalid_fixtures_are_rejected_with_the_expected_variant() {
    assert!(INVALID.len() >= 4);
    for (name, text, expected) in INVALID {
        assert_ne!(*text, BASE, "{name} is identical to the base envelope");
        let err = load(text).expect_err(name);
        assert_eq!(kind(&err), *expected, "{name}: {err:?}");
        assert!(!format!("{err}").is_empty());
    }
}

#[test]
#[ignore = "needs WP-4: lictor_canon canonical bytes (un-ignore in WP-13)"]
fn digest_is_stable_and_matches_fixture() {
    let a = load(BASE).unwrap();
    let b = load(BASE).unwrap();
    assert_eq!(a.digest_hex(), b.digest_hex());
    assert_eq!(a.embodiment_digest(), b.embodiment_digest());
    assert_eq!(a.digest_hex().len(), 64);
    let d: serde_json::Value = serde_json::from_str(DIGESTS).unwrap();
    assert_eq!(a.digest_hex(), d["envelope_digest"].as_str().unwrap());
    assert_eq!(a.embodiment_digest(), d["embodiment_digest"].as_str().unwrap());
    let cfg = a.compile(FuseMode::Observe, None).unwrap();
    assert_eq!(hex(&cfg.envelope_digest), a.digest_hex());
    assert_eq!(hex(&cfg.embodiment_digest), a.embodiment_digest());
}

#[test]
#[ignore = "needs WP-4: lictor_canon canonical bytes (un-ignore in WP-13)"]
fn embodiment_digest_ignores_limits_and_operators_but_not_the_manifest() {
    let base = load(BASE).unwrap();
    let mut fitted = base.clone();
    fitted.v_max = 1234.5;
    fitted.a_max = 4321.0;
    fitted.operators = vec!["ab".repeat(32)];
    fitted.validate().unwrap();
    assert_eq!(fitted.embodiment_digest(), base.embodiment_digest());
    assert_ne!(fitted.digest_hex(), base.digest_hex());
    let mut manifest = base.clone();
    manifest.embodiment.norm_scale[0] = 255.0;
    manifest.validate().unwrap();
    assert_ne!(manifest.embodiment_digest(), base.embodiment_digest());
    assert_ne!(manifest.digest_hex(), base.digest_hex());
}
