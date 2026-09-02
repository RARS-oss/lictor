# SPDX-License-Identifier: MIT
"""figures.py --from-fixture renders 9 SVGs with the required caption strings; analyze.py cross-checks."""

from __future__ import annotations

import csv
import html
import json
import struct
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

from harness import analyze, figures, stats

REPO = Path(__file__).resolve().parents[2]
FIGURES_PY = REPO / "harness" / "figures.py"
ANALYZE_PY = REPO / "harness" / "analyze.py"

pytest.importorskip("matplotlib", reason="figures need matplotlib (the venv has it; CI installs it)")


@lru_cache(maxsize=1)
def fixture():
    return figures.fixture_run()


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("figs")
    proc = subprocess.run([sys.executable, str(FIGURES_PY), "--from-fixture", "--out", str(out)],
                          capture_output=True, text=True, cwd=str(REPO), timeout=900)
    assert proc.returncode == 0, proc.stderr
    return out


def _svg(path: Path) -> str:
    return html.unescape(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- rendering


def test_nine_svg_files(rendered):
    svgs = sorted(p.name for p in rendered.glob("*.svg"))
    assert svgs == sorted(figures.FIG_FILES.values())
    assert len(svgs) == 9
    for p in rendered.glob("*.svg"):
        head = p.read_text(encoding="utf-8")[:400]
        assert "<svg" in head


def test_f1_captions_and_axes(rendered):
    s = _svg(rendered / "curve-safety-latency.svg")
    assert figures.CAP_TCE in s
    assert "re-scaling, not a measurement" in s
    assert "tce_valid_frac" in s
    assert analyze.WSL2_LABEL in s
    assert figures.WSL_TAG in s
    assert "n=60 pilot" in s              # the hollow pilot point is labelled as such
    assert "n=500" in s
    assert "20 ms clock" in s and "100 ms control step" in s


def test_f5_carries_the_wsl2_sentence_verbatim(rendered):
    raw = (rendered / "latency-hist.svg").read_text(encoding="utf-8")
    assert "not a real-time environment" in raw          # the acceptance grep, un-unescaped
    assert analyze.WSL2_LABEL in html.unescape(raw)
    assert figures.WSL_TAG in raw
    assert "p99" in raw and "p50" in raw


def test_f3_ponr_definition_and_artefact(rendered):
    s = _svg(rendered / "lead-time.svg")
    assert "PROXY" in s
    assert "t_fail = min{t : c*_T - c*_t < eps_prog}" in s
    assert "t_fail = 0" in s
    assert "eps_prog in {0.01, 0.02, 0.05}" in s
    assert "sensitivity" in s


def test_every_figure_labels_n(rendered):
    for name in figures.FIG_FILES.values():
        s = _svg(rendered / name)
        assert "n=" in s, name


def test_f2_f7_f8_specifics(rendered):
    f2 = _svg(rendered / "pareto-detection.svg")
    assert "ROC-AUC" in f2 and "tau_source == artefact" in f2 and "(fit)" in f2
    f7 = _svg(rendered / "cp-validity.svg")
    assert "K = 1" in f7 and "diagonal" in f7 and "n=59" in f7
    f8 = _svg(rendered / "calib-economy.svg")
    assert "tau=+inf" in f8 and "n_calib" in f8
    f9 = _svg(rendered / "enforcement.svg")
    assert "violations_reached_env" in f9 and "HEAD MISMATCH" in f9 and "inj-obs-d0" in f9
    f6 = _svg(rendered / "success-delta.svg")
    assert "McNemar" in f6 and "vs obs-d0" in f6 and "vs obs-d2" in f6
    f4 = _svg(rendered / "score-bands.svg")
    assert "tau = " in f4 and "IQR" in f4


def test_svg_output_is_deterministic(tmp_path):
    run = fixture()
    a = tmp_path / "a"
    b = tmp_path / "b"
    figures.render_all(run, a, only={"F7"})
    figures.render_all(run, b, only={"F7"})
    assert (a / "cp-validity.svg").read_bytes() == (b / "cp-validity.svg").read_bytes()
    assert b"dc:date" not in (a / "cp-validity.svg").read_bytes()


def test_no_non_ascii_in_svg_text(rendered):
    for name in figures.FIG_FILES.values():
        data = (rendered / name).read_bytes()
        assert all(b < 128 for b in data), name


# --------------------------------------------------------------- pilot rules


def test_pilot_points_break_lines_and_are_hollow():
    pts = [{"x": 0, "pilot": False}, {"x": 1, "pilot": False}, {"x": 2, "pilot": True},
           {"x": 3, "pilot": False}, {"x": 5, "pilot": True}, {"x": 8, "pilot": False}]
    segs = figures.segments_without_pilot(pts)
    assert [[p["x"] for p in s] for s in segs] == [[0, 1], [3], [8]]
    assert all(not p["pilot"] for s in segs for p in s)
    hollow = figures._marker_kwargs("#2a78d6", True)
    filled = figures._marker_kwargs("#2a78d6", False)
    assert hollow["markerfacecolor"] == figures.SURFACE and hollow["markeredgecolor"] == "#2a78d6"
    assert filled["markerfacecolor"] == "#2a78d6"


def test_fixture_marks_small_n_as_pilot():
    run = fixture()
    by_id = {r["arm_id"]: r for r in run["arms"]}
    assert by_id["t01-a05-d8"]["pilot"] == 1 and by_id["t01-a05-d8"]["n"] == 60
    assert by_id["t01-a05-d0"]["pilot"] == 0 and by_id["t01-a05-d0"]["n"] == 500
    assert by_id["t01-a05-d0-oracle"]["pilot"] == 0   # n = 100 is not small_n
    assert by_id["t01-a05-d8"]["tce_valid_frac"] == 0.0          # sync d >= 7 -> no overlap
    assert by_id["t01-a05-d5-async"]["tce_valid_frac"] > 0.9     # async keeps L = 7


# --------------------------------------------------------------- analyze.py


def _f64hex(v: float) -> dict:
    return {"f64": struct.pack(">d", float(v)).hex()}


def _delta_json(d: dict | None):
    if d is None:
        return None
    return {"diff": _f64hex(d["diff"]), "lo": _f64hex(d["lo"]), "hi": _f64hex(d["hi"]),
            "mcnemar_p": _f64hex(d["mcnemar_p"]), "mcnemar_b": d["mcnemar_b"], "mcnemar_c": d["mcnemar_c"]}


def write_fixture_run(run_dir: Path, perturb: str | None = None) -> None:
    """Materialise the embedded fixture as a results tree (summary.csv, curve/*.json, sweep.jsonl)."""
    run = fixture()
    curve_dir = run_dir / "curve"
    curve_dir.mkdir(parents=True)
    rows = [dict(r) for r in run["arms"]]
    if perturb == "success_hi":
        rows[3]["success_hi"] = rows[3]["success_hi"] + 0.01
    analyze.write_csv(curve_dir / "summary.csv", rows, analyze.SUMMARY_COLUMNS)
    for arm_id, c in run["curve"].items():
        body = {
            "schema": "lictor-curve/v1", "canonical": "jcs-floatfree/v1", "created_epoch": 0,
            "run_id": "fixture", "arm_id": arm_id, "baseline_arm_id": "obs-d0",
            "latency_control_arm_id": c["latency_control_arm_id"], "ledger_head": "0" * 64,
            "n_episodes": c["n"], "ledger_chain_ok": True, "arm_config_digest": "0" * 64,
            "receipt_pubkey": c["receipt_pubkey"], "run_json_sha256": "0" * 64,
            "n_declared": c["n"], "n_present": c["n"], "n_missing": 0, "missing_indices": [],
            "partial": False, "small_n": c["small_n"], "cross_run_mismatches": [], "pair_mismatches": [],
            "seed_overlap_with_calibration": [],
            "metrics": {
                "n": c["n"], "success_rate": _f64hex(c["success_rate"]),
                "success_ci": [_f64hex(x) for x in c["success_ci"]],
                "flagged_ci": [_f64hex(x) for x in c["flagged_ci"]],
                "false_trip_ci": [_f64hex(x) for x in c["false_trip_ci"]],
                "delta_vs_baseline": _delta_json(c["delta_vs_baseline"]),
                "delta_vs_latency_control": _delta_json(c["delta_vs_latency_control"]),
                "tce_valid_frac": _f64hex(c["tce_valid_frac"]), "latency_label": analyze.WSL2_LABEL,
                "n_fail_baseline": c["n_fail_baseline"], "n_succ_baseline": c["n_succ_baseline"],
            },
            "eps_prog": _f64hex(0.02),
        }
        with open(curve_dir / f"{arm_id}.json", "w", encoding="utf-8") as fh:
            json.dump({"body": body, "body_digest": "0" * 64, "pubkey": c["receipt_pubkey"], "sig": ""},
                      fh, indent=2, sort_keys=True)
    with open(run_dir / "sweep.jsonl", "w", encoding="utf-8") as fh:
        for s in run["sweep"]:
            fh.write(json.dumps(s, sort_keys=True) + "\n")
    with open(run_dir / "microbench.json", "w", encoding="utf-8") as fh:
        json.dump(run["microbench"], fh, indent=2, sort_keys=True)


def test_curve_json_float_free_roundtrip():
    assert analyze.f64hex({"f64": "400db6db6db6db6e"}) == pytest.approx(3.7142857142857144)
    assert analyze.f64hex({"f64": "7ff0000000000000"}) == float("inf")
    assert analyze.f64array({"f64a": "AAAAAAAA8D8AAAAAAAAAQA==", "shape": [2]}) == [1.0, 2.0]


def test_analyze_cross_check_ok_on_fixture_run(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_fixture_run(run_dir)
    rc = analyze.main(["--run", str(run_dir), "--out", str(tmp_path / "an")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "stats cross-check: ok" in out
    assert analyze.TIMING_SENTENCE in out
    for name in ("arms", "sweep", "layer_ab", "lead", "tce_valid", "latency"):
        assert (tmp_path / "an" / f"{name}.csv").exists(), name
    report = json.loads((tmp_path / "an" / "cross_check.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    # deltas are copied verbatim from summary.csv, never recomputed
    with open(tmp_path / "an" / "arms.csv", newline="", encoding="utf-8") as fh:
        arms = {r["arm_id"]: r for r in csv.DictReader(fh)}
    with open(run_dir / "curve" / "summary.csv", newline="", encoding="utf-8") as fh:
        summ = {r["arm_id"]: r for r in csv.DictReader(fh)}
    for arm_id, r in summ.items():
        for col in ("delta_vs_baseline", "delta_vs_baseline_lo", "delta_vs_baseline_hi",
                    "delta_vs_latency_control", "mcnemar_p", "mcnemar_p_lc"):
            assert arms[arm_id][col] == r[col], (arm_id, col)
    # latency table: every row carries the label verbatim
    with open(tmp_path / "an" / "latency.csv", newline="", encoding="utf-8") as fh:
        lat = list(csv.DictReader(fh))
    assert lat and all(r["latency_label"] for r in lat)
    assert any(r["latency_label"] == analyze.WSL2_LABEL for r in lat)


def test_analyze_reports_a_wrong_ci(tmp_path, capsys):
    run_dir = tmp_path / "run"
    write_fixture_run(run_dir, perturb="success_hi")
    rc = analyze.main(["--run", str(run_dir), "--out", str(tmp_path / "an")])
    out = capsys.readouterr().out
    assert rc == 1
    assert "stats cross-check: 1 diffs" in out
    assert "success_hi" in out


def test_layer_ab_uses_artefact_points_only():
    rows = analyze.layer_ab_table(fixture())
    assert rows
    assert {r["alpha"] for r in rows} == {0.05, 0.1, 0.2}      # alpha 1 % and 2 % are tau_source == fit
    assert all(r["arm_id"].endswith("-d0") for r in rows)
    t01 = [r for r in rows if r["detector"] == "t01" and r["alpha"] == 0.05]
    assert len(t01) == 1 and t01[0]["arm_id"] == "t01-a05-d0"
    assert abs(t01[0]["tpr_gap"]) < 0.05


def test_parse_arm_id():
    p = analyze.parse_arm_id
    assert p("t01-a05-d2-async")["exec"] == "async" and p("t01-a05-d2-async")["d"] == 2
    assert p("t01-a05-d0-oracle")["role"] == "oracle"
    assert p("obs-d0")["role"] == "baseline" and p("obs-d3")["role"] == "latency_control"
    assert p("inj-t0-d0")["role"] == "injection" and p("calib-obs")["role"] == "calibration"
    assert p("t0-d0")["family"] == "t0" and p("t01-a20-d0")["family"] == "t01-a20"
    assert p("garbage")["role"] == "unknown"


def test_bench_csv_reader_and_percentiles(tmp_path):
    p = tmp_path / "bench.csv"
    p.write_text("series,upper_ns,count\nverdict,1000,5\nverdict,2000,90\nverdict,50000,5\nio,20000,100\n", encoding="utf-8")
    h = analyze.read_bench_csv(p)
    assert set(h) == {"verdict", "io"}
    pc = analyze.hist_percentiles(h["verdict"])
    assert pc["p50"] == 2000 and pc["p99"] == 50000 and pc["max"] == 50000 and pc["n"] == 100
    assert set(pc) == {"p50", "p90", "p99", "p99.9", "p99.99", "max", "n"}
    assert pc["p90"] == 2000 and pc["p99.9"] == 50000
    p2 = tmp_path / "bench_us.csv"
    p2.write_text("value_us,n\n1.5,10\n", encoding="utf-8")
    assert analyze.read_bench_csv(p2)["verdict"] == [(1500.0, 10)]


def test_scores_extraction(tmp_path):
    arm = tmp_path / "run" / "obs-d0"
    (arm / "ticks").mkdir(parents=True)
    with open(arm / "ledger.jsonl", "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"schema": "lictor-ledger/v1", "run_id": "r", "arm_id": "obs-d0", "genesis": "0" * 64}) + "\n")
        for ep, ok in ((0, True), (1, False), (2, True)):
            fh.write(json.dumps({"seq": ep, "run_id": "r", "arm_id": "obs-d0", "episode_index": ep, "seed": ep,
                                 "success": ok, "prev": "0" * 64, "hash": "0" * 64}) + "\n")
    scores = {0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0], 2: [float("-inf"), 2.0, 4.0]}
    for ep, ss in scores.items():
        with open(arm / "ticks" / f"{ep:06d}.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"schema": "lictor-ticks/v1", "run_id": "r", "arm_id": "obs-d0", "episode_index": ep, "genesis": "0" * 64}) + "\n")
            for t, s in enumerate(ss):
                fh.write(json.dumps({"seq": t, "t": t, "s": _f64hex(s), "tau": _f64hex(3.5), "hash": "", "prev": ""}) + "\n")
    rows = analyze.extract_score_bands(tmp_path / "run", "obs-d0")
    by = {(r["t"], r["group"]): r for r in rows}
    assert by[(0, "success")]["n"] == 1 and by[(0, "success")]["q50"] == 1.0      # -inf excluded
    assert by[(1, "success")]["n"] == 2 and by[(1, "success")]["q50"] == 2.0
    assert by[(2, "fail")]["q50"] == 6.0 and by[(2, "fail")]["tau"] == 3.5
    out = tmp_path / "bands.csv"
    rc = analyze.main_scores(["--run", str(tmp_path / "run"), "--arm", "obs-d0", "-o", str(out)])
    assert rc == 0 and out.exists()
    fig_rows = figures.read_score_bands(out)
    assert len(fig_rows) == len(rows)
