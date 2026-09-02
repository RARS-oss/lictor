# SPDX-License-Identifier: MIT
"""Known-value tests for harness/stats.py (the cross-checks of lictor-calib's metrics)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from harness import stats

REPO = Path(__file__).resolve().parents[2]
RUST_FIXTURE = REPO / "crates" / "lictor-calib" / "tests" / "fixtures" / "calib" / "expected.json"


# ------------------------------------------------------------- incomplete beta


def test_betainc_known_values():
    assert stats.betainc(2, 3, 0.5) == pytest.approx(0.6875, abs=1e-12)
    assert stats.betainc(1, 1, 0.3) == pytest.approx(0.3, abs=1e-12)
    assert stats.betainc(5, 2, 0.0) == 0.0
    assert stats.betainc(5, 2, 1.0) == 1.0
    # symmetry I_x(a, b) = 1 - I_{1-x}(b, a)
    for a, b, x in ((3.5, 7.25, 0.2), (12, 3, 0.77), (0.5, 0.5, 0.31)):
        assert stats.betainc(a, b, x) == pytest.approx(1.0 - stats.betainc(b, a, 1.0 - x), abs=1e-12)


def test_beta_quantile_roundtrip():
    for a, b, p in ((3, 98, 0.025), (51, 50, 0.975), (1, 100, 0.975), (137, 1, 0.5)):
        x = stats.beta_quantile(p, a, b)
        assert stats.betainc(a, b, x) == pytest.approx(p, abs=1e-12)


# ------------------------------------------------------------ Clopper-Pearson


def test_clopper_pearson_k0_n100():
    lo, hi = stats.clopper_pearson(0, 100)
    assert lo == 0.0
    assert hi == pytest.approx(0.0362, abs=5e-5)          # the WP-5 table value
    assert hi == pytest.approx(1.0 - 0.025 ** (1.0 / 100.0), abs=1e-12)


def test_clopper_pearson_edges_and_symmetry():
    assert stats.clopper_pearson(0, 0) == (0.0, 1.0)
    lo, hi = stats.clopper_pearson(100, 100)
    assert hi == 1.0 and lo == pytest.approx(0.025 ** (1.0 / 100.0), abs=1e-12)
    lo30, hi30 = stats.clopper_pearson(30, 100)
    lo70, hi70 = stats.clopper_pearson(70, 100)
    assert lo30 == pytest.approx(1.0 - hi70, abs=1e-12)
    assert hi30 == pytest.approx(1.0 - lo70, abs=1e-12)
    lo, hi = stats.clopper_pearson(50, 100)
    assert (lo, hi) == pytest.approx((0.3983, 0.6017), abs=1e-4)
    with pytest.raises(ValueError):
        stats.clopper_pearson(5, 3)


def test_clopper_pearson_halfwidth_sentence():
    # ARCHITECTURE 10.7: n=60 -> about +-12 pp; n=100 -> about +-10 pp; n=500 -> about +-4 pp
    assert 0.11 <= stats.clopper_pearson_halfwidth(60) <= 0.14
    assert 0.09 <= stats.clopper_pearson_halfwidth(100) <= 0.11
    assert 0.04 <= stats.clopper_pearson_halfwidth(500) <= 0.05


# --------------------------------------------------------------------- McNemar


def test_mcnemar_known_values():
    assert stats.mcnemar_exact(5, 15) == pytest.approx(0.0414, abs=1e-4)   # the WP-5 table value
    assert stats.mcnemar_exact(5, 15) == pytest.approx(2 * 21700 / 1048576, abs=1e-12)
    assert stats.mcnemar_exact(15, 5) == stats.mcnemar_exact(5, 15)
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert stats.mcnemar_exact(7, 7) == 1.0
    assert stats.mcnemar_exact(0, 10) == pytest.approx(2.0 / 1024.0, abs=1e-15)
    with pytest.raises(ValueError):
        stats.mcnemar_exact(-1, 2)


# ------------------------------------------------------------------------ PoNR


def test_ponr_on_hand_trace():
    cov = [0.0, 0.1, 0.3, 0.5, 0.6, 0.62, 0.63, 0.63, 0.62]
    assert stats.ponr_tick(cov, 0.02) == 5      # first t with 0.63 - c*_t < 0.02
    assert stats.ponr_tick(cov, 0.05) == 4
    assert stats.ponr_tick(cov, 0.5) == 2       # 0.63 - 0.3 = 0.33 < 0.5
    assert stats.ponr_tick([0.2] * 10, 0.02) == 0            # no progress at all -> t_fail = 0 (the F3 artefact)
    assert stats.ponr_tick([0.5, 0.4, 0.3], 0.02) == 0       # running max never moves
    assert stats.ponr_tick([], 0.02) == 0
    assert stats.ponr_tick([0.0, 0.0, 0.9], 0.02) == 2


def test_lead_ticks():
    assert stats.lead_ticks(181, 150) == 31
    assert stats.lead_ticks(0, 40) == -40
    assert stats.lead_ticks(120, None) is None


# ---------------------------------------------------------------------- ROC etc


def test_roc_auc_hand_cases():
    assert stats.roc_auc([0.9, 0.8, 0.7], [0.1, 0.2, 0.3]) == 1.0
    assert stats.roc_auc([0.1, 0.2], [0.8, 0.9]) == 0.0
    assert stats.roc_auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert stats.roc_auc([0.8, 0.4], [0.6, 0.2]) == pytest.approx(0.75)
    assert stats.roc_auc([], [0.1]) == 0.5


def test_bacc():
    assert stats.bacc(0.6, 0.06) == pytest.approx(0.77)   # the sweep.jsonl example row


def test_aucpdt_hand_case():
    assert stats.aucpdt([10, 0, None, 5], 10) == pytest.approx(0.425)
    assert stats.aucpdt([], 300) == 0.0
    assert stats.aucpdt([None, None], 300) == 0.0
    assert stats.aucpdt([1000], 300) == pytest.approx(1.0)
    assert stats.aucpdt([-5], 300) == 0.0


def test_f1_timeliness_hypervolume():
    assert stats.f1_timeliness_hypervolume([(0.5, 0.5), (0.8, 0.2), (0.2, 0.9)]) == pytest.approx(0.39)
    assert stats.f1_timeliness_hypervolume([]) == 0.0
    assert stats.f1_timeliness_hypervolume([(1.0, 1.0)]) == 1.0
    assert stats.f1_score(3, 1, 2) == pytest.approx(2.0 / 3.0)
    assert stats.f1_score(3, 2, 2) == pytest.approx(0.6)
    assert stats.f1_score(0, 0, 0) == 0.0
    assert stats.timeliness([50, None, 500, -10], 300) == pytest.approx((50 / 300 + 1.0 + 0.0) / 3)


def test_quantiles_and_lead_summary():
    assert stats.quantile_nearest_rank([3, 1, 2], 0.5) == 2
    assert stats.quantile_nearest_rank([3, 1, 2], 0.1) == 1
    assert stats.quantile_nearest_rank([3, 1, 2], 1.0) == 3
    assert math.isnan(stats.quantile_nearest_rank([], 0.5))
    s = stats.lead_summary([10, None, 30, 20])
    assert s == {"lead_mean": 20.0, "lead_p50": 20.0, "lead_p10": 10.0}
    assert all(math.isnan(v) for v in stats.lead_summary([None]).values())


# ------------------------------------------------------------------- bootstrap


def test_bootstrap_is_deterministic_and_sane():
    base, arm = stats.pairs_from_contingency(500, 313, 14, 23)
    r1 = stats.paired_bootstrap_diff(base, arm)
    r2 = stats.paired_bootstrap_diff(base, arm)
    assert r1 == r2
    diff, lo, hi = r1
    assert diff == pytest.approx((23 - 14) / 500.0)
    assert lo <= diff <= hi
    assert 0.0 < hi - lo < 0.15
    d0 = stats.paired_bootstrap_diff(base, arm, n_resamples=0)
    assert d0 == (diff, diff, diff)
    assert stats.paired_bootstrap_diff([], []) == (math.nan, math.nan, math.nan) or all(math.isnan(v) for v in stats.paired_bootstrap_diff([], []))
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff([True], [True, False])


def test_bootstrap_seed_changes_resamples():
    base, arm = stats.pairs_from_contingency(997, 500, 97, 131)
    res = [stats.paired_bootstrap_diff(base, arm, seed=s) for s in (1, 2, 3)]
    assert len({r[0] for r in res}) == 1                      # the point estimate never depends on the seed
    assert len({(r[1], r[2]) for r in res}) > 1               # the resamples do


def test_contingency_orientation():
    assert stats.orient_contingency(500, 336, 327, 14, 23) == (313, 14, 23)
    assert stats.orient_contingency(500, 336, 327, 23, 14) == (313, 14, 23)   # swapped columns
    assert stats.orient_contingency(500, 336, 327, 10, 10) is None            # totals disagree
    assert stats.orient_contingency(10, 9, 9, 8, 8) is None                   # 'both' would be negative
    base, arm = stats.pairs_from_contingency(6, 2, 1, 2)
    assert base == [True, True, True, False, False, False]
    assert arm == [True, True, False, True, True, False]
    with pytest.raises(ValueError):
        stats.pairs_from_contingency(3, 2, 1, 1)


# ----------------------------------------------- parity with the Rust fixture


def _find_ponr_cases(obj):
    """Accept a few plausible layouts of WP-5's expected.json without guessing too hard."""
    cases = []
    if isinstance(obj, dict):
        for key in ("ponr", "ponr_tick", "point_of_no_return"):
            if key in obj:
                obj = obj[key]
                break
    if isinstance(obj, dict):
        obj = list(obj.values())
    if isinstance(obj, list):
        for c in obj:
            if isinstance(c, dict) and "coverage" in c and any(k in c for k in ("t_fail", "ponr", "expected")):
                cases.append(c)
    return cases


@pytest.mark.skipif(not RUST_FIXTURE.exists(),
                    reason="pending WP-5: shared PoNR fixture crates/lictor-calib/tests/fixtures/calib/expected.json not present")
def test_ponr_matches_rust_fixture():
    with open(RUST_FIXTURE, encoding="utf-8") as fh:
        obj = json.load(fh)
    cases = _find_ponr_cases(obj)
    if not cases:
        pytest.skip("pending WP-5: expected.json carries no ponr cases with coverage/t_fail keys")
    for c in cases:
        expected = c.get("t_fail", c.get("ponr", c.get("expected")))
        eps = float(c.get("eps_prog", c.get("eps", 0.02)))
        assert stats.ponr_tick([float(x) for x in c["coverage"]], eps) == int(expected)
