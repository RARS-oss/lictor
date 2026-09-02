# SPDX-License-Identifier: MIT
"""Tidy analysis tables for a lictor run directory (WP-9).

Inputs (and NOTHING else -- never the harness `index.jsonl`, never a receipt):

- `results/<run>/curve/summary.csv`   written by `lictor curve` from signed receipts
- `results/<run>/curve/<arm>.json`    the signed curve receipts (only their bodies are read,
                                      for `n_fail_baseline` / `n_succ_baseline` / `small_n` / ...)
- `results/<run>/sweep.jsonl`         Layer A (`lictor sweep`)
- `results/<run>/microbench.json`     the day-0 gate table (`harness/microbench.py`)
- `--bench-csv F`                     the bucket dump of `lictor bench --csv F`
- `results/<run>/calibration.<alpha>.json` (optional, only `n_holdout`/`n_calib`/`tau`/
                                      `holdout_fpr*` are read, for the CI on figure F7)

analyze.py never computes a delta itself: `delta_vs_baseline*` and
`delta_vs_latency_control*` are copied VERBATIM from `summary.csv` (ARCHITECTURE 10.7,
M12). It only re-checks them with `harness/stats.py` and prints
`stats cross-check: ok` or the diffs.

Outputs (`--out DIR`, default `<run>/analysis/`), every CSV with a fixed column list:

- `arms.csv`       per-arm Y1/Y2/Y3 with CIs, both deltas, joined with the arm metadata
- `sweep.csv`      the alpha sweep (Layer A) with `tau_source`
- `layer_ab.csv`   Layer-A-vs-Layer-B comparison on `tau_source == "artefact"` points only
- `lead.csv`       lead-time quantiles per arm and per sweep point
- `tce_valid.csv`  the feature-loss confound per point
- `latency.csv`    the latency table; every row carries its `latency_label` verbatim
- `cross_check.json` the stats cross-check report (pretty, sorted keys)

Usage:
    python harness/analyze.py --run results/<run> [--out DIR] [--bench-csv F] [--json]
    python harness/analyze.py scores --run results/<run> --arm t01-a05-d0 -o score_bands.csv

The `scores` subcommand is the ONE documented exception to the input list above:
figure F4 (median +- IQR of `s_t` for successful vs failing episodes) needs per-tick
scores, which none of the frozen analysis products carry. It reads the arm's
hash-chained `ledger.jsonl` (labels) and `ticks/<ep>.jsonl` (the `s` / `tau` of every
tick) -- fuse-written artefacts, never the harness index, never a receipt -- and
writes the `score_bands.csv` that `figures.py --score-bands` consumes.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import os
import re
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from harness import stats  # noqa: E402

WSL2_LABEL = (
    "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host)"
    " -- not a real-time environment"
)
WSL_TAG = "WSL2, not RT"
TIMING_SENTENCE = (
    "every wall-clock figure in this document was measured under WSL2 virtualisation"
    " (Hyper-V utility VM, non-RT host) and is not a real-time measurement"
)

# The frozen column list of results/<run>/curve/summary.csv (ARCHITECTURE FILE FORMATS).
SUMMARY_COLUMNS = [
    "pilot", "arm_id", "n", "n_missing", "partial", "delay_steps", "exec_mode", "alpha",
    "detector", "success_rate", "success_lo", "success_hi", "delta_vs_baseline",
    "delta_vs_baseline_lo", "delta_vs_baseline_hi", "mcnemar_p", "mcnemar_b", "mcnemar_c",
    "latency_control_arm", "delta_vs_latency_control", "delta_vs_latency_control_lo",
    "delta_vs_latency_control_hi", "mcnemar_p_lc", "mcnemar_b_lc", "mcnemar_c_lc",
    "averted_rate", "flagged_rate", "flagged_lo", "flagged_hi", "false_trip_rate",
    "false_trip_lo", "false_trip_hi", "intervention_rate", "intervention_tick_frac",
    "escalation_rate", "lead_mean", "lead_p50", "lead_p10", "aucpdt", "roc_auc", "bacc",
    "violations_reached_env", "tce_valid_frac", "latency_p50_ns", "latency_p99_ns",
    "latency_max_ns", "latency_label", "ledger_chain_ok", "pair_mismatches",
    "cross_run_mismatches", "receipt_pubkey",
]

SWEEP_KEYS = [
    "alpha_num", "alpha_den", "detector", "gate", "k", "n", "tier0", "method", "tau",
    "tau_source", "n_calib", "n_fail", "n_succ", "tpr", "tpr_ci", "fpr", "fpr_ci",
    "lead_mean", "lead_p50", "lead_p10", "aucpdt", "roc_auc", "bacc", "fire_frac_mean",
    "holdout_fpr", "holdout_fpr_k1", "eps_prog",
]

ARM_RE = re.compile(
    r"^(?P<det>inj-obs|inj-t0|calib-obs|obs|t01|t0)"
    r"(?:-a(?P<alpha>\d+))?(?:-d(?P<d>\d+))?(?P<asyn>-async)?"
    r"(?:-(?P<variant>oracle|ackonly))?$"
)

# ------------------------------------------------------------------ small helpers


def parse_cell(s):
    """CSV cell -> None | bool | int | float | str (deterministic, no locale)."""
    if s is None:
        return None
    t = s.strip()
    if t == "":
        return None
    low = t.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        return t


def f64hex(v):
    """Decode a float-free scalar {"f64": "<16 hex>"} (big-endian bit pattern)."""
    if isinstance(v, dict) and "f64" in v:
        return struct.unpack(">d", bytes.fromhex(v["f64"]))[0]
    if v is None:
        return None
    return float(v)


def f64array(v):
    """Decode {"f64a": "<base64 LE>", "shape": [...]} into a flat list of floats."""
    if isinstance(v, dict) and "f64a" in v:
        raw = base64.b64decode(v["f64a"])
        return [struct.unpack("<d", raw[i : i + 8])[0] for i in range(0, len(raw), 8)]
    return list(v)


def parse_arm_id(arm_id: str) -> dict:
    """arm_id = <detector>-a<alpha%>-d<d>[-async][-<escalation>] -> metadata."""
    m = ARM_RE.match(arm_id or "")
    if not m:
        return {
            "detector_family": None, "alpha_pct": None, "d": None, "exec": None,
            "variant": None, "family": None, "role": "unknown",
        }
    det = m.group("det")
    alpha = int(m.group("alpha")) if m.group("alpha") else None
    d = int(m.group("d")) if m.group("d") is not None else None
    exec_mode = "async" if m.group("asyn") else "sync"
    variant = m.group("variant")
    family = det if alpha is None else f"{det}-a{alpha:02d}"
    if det == "calib-obs":
        role = "calibration"
    elif det == "obs":
        role = "baseline" if d == 0 and exec_mode == "sync" else "latency_control"
    elif det.startswith("inj-"):
        role = "injection"
    elif variant == "oracle":
        role = "oracle"
    elif variant == "ackonly":
        role = "ackonly"
    else:
        role = "enforce"
    return {
        "detector_family": det, "alpha_pct": alpha, "d": d, "exec": exec_mode,
        "variant": variant, "family": family, "role": role,
    }


def alpha_value(num, den) -> float | None:
    if num is None or den in (None, 0):
        return None
    return float(num) / float(den)


def alpha_from_text(s) -> float | None:
    """'5/100' -> 0.05; '0.05' -> 0.05; '' -> None."""
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    t = str(s).strip()
    if "/" in t:
        a, b = t.split("/", 1)
        try:
            return float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            return None
    try:
        return float(t)
    except ValueError:
        return None


# -------------------------------------------------------------------- readers


def read_summary_csv(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        header = rd.fieldnames or []
        rows = [{k: parse_cell(v) for k, v in r.items()} for r in rd]
    missing = [c for c in SUMMARY_COLUMNS if c not in header]
    extra = [c for c in header if c not in SUMMARY_COLUMNS]
    for r in rows:
        r["_missing_columns"] = missing
        r["_extra_columns"] = extra
    return rows


def _delta_ci(d) -> dict | None:
    if not isinstance(d, dict):
        return None
    return {
        "diff": f64hex(d.get("diff")), "lo": f64hex(d.get("lo")), "hi": f64hex(d.get("hi")),
        "mcnemar_p": f64hex(d.get("mcnemar_p")), "mcnemar_b": d.get("mcnemar_b"),
        "mcnemar_c": d.get("mcnemar_c"),
    }


def read_curve_json(path) -> dict:
    """Body of a signed curve receipt (schema lictor-curve/v1), floats decoded."""
    with open(path, encoding="utf-8") as fh:
        sc = json.load(fh)
    body = sc.get("body", sc)
    m = body.get("metrics", {})
    out = {
        "schema": body.get("schema"),
        "arm_id": body.get("arm_id"),
        "run_id": body.get("run_id"),
        "baseline_arm_id": body.get("baseline_arm_id"),
        "latency_control_arm_id": body.get("latency_control_arm_id"),
        "ledger_chain_ok": body.get("ledger_chain_ok"),
        "n_episodes": body.get("n_episodes"),
        "n_declared": body.get("n_declared"),
        "n_present": body.get("n_present"),
        "n_missing": body.get("n_missing"),
        "partial": body.get("partial"),
        "small_n": body.get("small_n"),
        "pair_mismatches": len(body.get("pair_mismatches") or []),
        "cross_run_mismatches": len(body.get("cross_run_mismatches") or []),
        "seed_overlap_with_calibration": len(body.get("seed_overlap_with_calibration") or []),
        "receipt_pubkey": body.get("receipt_pubkey"),
        "run_json_sha256": body.get("run_json_sha256"),
        "arm_config_digest": body.get("arm_config_digest"),
        "eps_prog": f64hex(body.get("eps_prog")) if body.get("eps_prog") is not None else None,
        "curve_pubkey": sc.get("pubkey"),
        "n": m.get("n"),
        "n_fail_baseline": m.get("n_fail_baseline"),
        "n_succ_baseline": m.get("n_succ_baseline"),
        "success_rate": f64hex(m.get("success_rate")) if m else None,
        "success_ci": [f64hex(x) for x in m.get("success_ci", [])] if m else None,
        "flagged_ci": [f64hex(x) for x in m.get("flagged_ci", [])] if m else None,
        "false_trip_ci": [f64hex(x) for x in m.get("false_trip_ci", [])] if m else None,
        "delta_vs_baseline": _delta_ci(m.get("delta_vs_baseline")) if m else None,
        "delta_vs_latency_control": _delta_ci(m.get("delta_vs_latency_control")) if m else None,
        "tce_valid_frac": f64hex(m.get("tce_valid_frac")) if m and m.get("tce_valid_frac") is not None else None,
        "latency_label": m.get("latency_label"),
    }
    return out


def read_sweep(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_microbench(path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_calibration_summary(path) -> dict:
    """The handful of fields F7/F8 need from calibration.<alpha>.json."""
    with open(path, encoding="utf-8") as fh:
        c = json.load(fh)
    return {
        "path": os.path.basename(path),
        "alpha_num": c.get("alpha_num"), "alpha_den": c.get("alpha_den"),
        "n_total": c.get("n_total"), "n_scale": c.get("n_scale"),
        "n_calib": c.get("n_calib"), "n_holdout": c.get("n_holdout"), "split": c.get("split"),
        "tau": f64hex(c.get("tau")) if c.get("tau") is not None else None,
        "holdout_fpr": f64hex(c.get("holdout_fpr")) if c.get("holdout_fpr") is not None else None,
        "holdout_fpr_k1": f64hex(c.get("holdout_fpr_k1")) if c.get("holdout_fpr_k1") is not None else None,
        "kn": c.get("kn"), "method": c.get("method"), "gate": c.get("gate"),
        "notes": c.get("notes") or [],
    }


_BENCH_SERIES_COLS = ("series", "tier", "name", "hist", "histogram")
_BENCH_NS_COLS = ("upper_ns", "value_ns", "bucket_ns", "ns", "le_ns", "hi_ns")
_BENCH_US_COLS = ("upper_us", "value_us", "bucket_us", "us", "le_us", "hi_us")
_BENCH_COUNT_COLS = ("count", "n", "hits", "samples")


def read_bench_csv(path) -> dict[str, list[tuple[float, int]]]:
    """`lictor bench --csv` bucket dump -> {series: [(upper_ns, count), ...]}.

    The dump's exact header is WP-10's; this reader accepts a series column named
    series|tier|name (default series "verdict"), a value column named
    upper_ns|value_ns|bucket_ns|ns (or the _us twins, converted) and a count column
    named count|n|hits. Unknown columns are ignored.
    """
    hist: dict[str, list[tuple[float, int]]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        cols = [c.strip().lower() for c in (rd.fieldnames or [])]
        s_col = next((c for c in cols if c in _BENCH_SERIES_COLS), None)
        ns_col = next((c for c in cols if c in _BENCH_NS_COLS), None)
        us_col = next((c for c in cols if c in _BENCH_US_COLS), None)
        n_col = next((c for c in cols if c in _BENCH_COUNT_COLS), None)
        if (ns_col is None and us_col is None) or n_col is None:
            raise ValueError(
                f"bench csv {path}: need a value column ({'|'.join(_BENCH_NS_COLS)} or _us)"
                f" and a count column ({'|'.join(_BENCH_COUNT_COLS)}); got {cols}"
            )
        for raw in rd:
            r = {k.strip().lower(): v for k, v in raw.items() if k is not None}
            series = (r.get(s_col) or "verdict").strip() if s_col else "verdict"
            if ns_col is not None:
                upper = float(r[ns_col])
            else:
                upper = float(r[us_col]) * 1000.0
            count = int(float(r[n_col]))
            hist.setdefault(series, []).append((upper, count))
    for k in hist:
        hist[k].sort(key=lambda t: t[0])
    return hist


def hist_percentiles(buckets, ps=(0.5, 0.9, 0.99, 0.999, 0.9999)) -> dict[str, float]:
    """Percentiles (upper bucket bound) plus max from a sorted (upper, count) list."""
    total = sum(c for _, c in buckets)
    out: dict[str, float] = {}
    if total <= 0:
        return out
    cum = 0
    targets = list(ps)
    ti = 0
    for upper, count in buckets:
        cum += count
        while ti < len(targets) and cum >= targets[ti] * total:
            out[f"p{targets[ti] * 100:g}"] = upper
            ti += 1
    out["max"] = max(u for u, c in buckets if c > 0)
    out["n"] = total
    return out


# --------------------------------------------------------------------- load_run


def load_run(run_dir, bench_csv=None, calib_dir=None) -> dict:
    """Everything the tables and figures consume, from the allowed inputs only."""
    run_dir = Path(run_dir)
    curve_dir = run_dir / "curve"
    summary = curve_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"{summary} (written by `lictor curve -o {curve_dir}`)")
    arms = read_summary_csv(summary)
    curve: dict[str, dict] = {}
    for p in sorted(curve_dir.glob("*.json")):
        try:
            body = read_curve_json(p)
        except (ValueError, KeyError, OSError) as e:
            body = {"arm_id": p.stem, "error": f"{type(e).__name__}: {e}"}
        curve[body.get("arm_id") or p.stem] = body
    for r in arms:
        meta = parse_arm_id(str(r.get("arm_id")))
        r.update({f"meta_{k}": v for k, v in meta.items()})
        cj = curve.get(str(r.get("arm_id")))
        r["n_fail_baseline"] = cj.get("n_fail_baseline") if cj else None
        r["n_succ_baseline"] = cj.get("n_succ_baseline") if cj else None
        r["small_n"] = cj.get("small_n") if cj else None
        r["eps_prog"] = cj.get("eps_prog") if cj else None
    sweep_path = run_dir / "sweep.jsonl"
    sweep = read_sweep(sweep_path) if sweep_path.exists() else []
    mb_path = run_dir / "microbench.json"
    microbench = read_microbench(mb_path) if mb_path.exists() else None
    bench_hist = read_bench_csv(bench_csv) if bench_csv else None
    cdir = Path(calib_dir) if calib_dir else run_dir
    calibrations = []
    for p in sorted(cdir.glob("calibration.*.json")):
        try:
            calibrations.append(read_calibration_summary(p))
        except (ValueError, KeyError, OSError) as e:
            calibrations.append({"path": p.name, "error": f"{type(e).__name__}: {e}"})
    return {
        "run_dir": str(run_dir),
        "arms": arms,
        "curve": curve,
        "sweep": sweep,
        "microbench": microbench,
        "bench_hist": bench_hist,
        "calibrations": calibrations,
        "score_bands": None,
        "leads": None,
        "tamper_log": None,
    }


# ------------------------------------------------------------------- tidy tables

ARMS_COLUMNS = [
    "pilot", "arm_id", "role", "family", "detector", "alpha", "delay_steps", "exec_mode",
    "variant", "n", "n_missing", "partial", "n_fail_baseline", "n_succ_baseline",
    "success_rate", "success_lo", "success_hi",
    "delta_vs_baseline", "delta_vs_baseline_lo", "delta_vs_baseline_hi",
    "mcnemar_p", "mcnemar_b", "mcnemar_c", "latency_control_arm",
    "delta_vs_latency_control", "delta_vs_latency_control_lo", "delta_vs_latency_control_hi",
    "mcnemar_p_lc", "mcnemar_b_lc", "mcnemar_c_lc",
    "averted_rate", "flagged_rate", "flagged_lo", "flagged_hi",
    "false_trip_rate", "false_trip_lo", "false_trip_hi",
    "intervention_rate", "intervention_tick_frac", "escalation_rate",
    "violations_reached_env", "tce_valid_frac", "ledger_chain_ok", "pair_mismatches",
    "cross_run_mismatches", "receipt_pubkey",
]


def arm_sort_key(r: dict):
    return (
        {"baseline": 0, "latency_control": 1, "enforce": 2, "oracle": 3, "ackonly": 4,
         "injection": 5, "calibration": 6, "unknown": 7}.get(r.get("meta_role"), 9),
        str(r.get("meta_family")),
        r.get("meta_d") if r.get("meta_d") is not None else -1,
        0 if r.get("meta_exec") == "sync" else 1,
        str(r.get("arm_id")),
    )


def arms_table(run: dict) -> list[dict]:
    out = []
    for r in sorted(run["arms"], key=arm_sort_key):
        row = {c: r.get(c) for c in ARMS_COLUMNS}
        row["role"] = r.get("meta_role")
        row["family"] = r.get("meta_family")
        row["variant"] = r.get("meta_variant")
        out.append(row)
    return out


SWEEP_COLUMNS = [
    "alpha", "alpha_num", "alpha_den", "detector", "gate", "k", "n", "tier0", "method",
    "tau", "tau_source", "degenerate", "n_calib", "n_fail", "n_succ", "tpr", "tpr_lo",
    "tpr_hi", "fpr", "fpr_lo", "fpr_hi", "lead_mean", "lead_p50", "lead_p10", "aucpdt",
    "roc_auc", "bacc", "fire_frac_mean", "holdout_fpr", "holdout_fpr_k1", "eps_prog",
]


def sweep_table(run: dict) -> list[dict]:
    out = []
    for s in run["sweep"]:
        tpr_ci = s.get("tpr_ci") or [None, None]
        fpr_ci = s.get("fpr_ci") or [None, None]
        gate = s.get("gate")
        row = {
            "alpha": alpha_value(s.get("alpha_num"), s.get("alpha_den")),
            "alpha_num": s.get("alpha_num"), "alpha_den": s.get("alpha_den"),
            "detector": s.get("detector"),
            "gate": "|".join(gate) if isinstance(gate, list) else gate,
            "k": s.get("k"), "n": s.get("n"), "tier0": s.get("tier0"), "method": s.get("method"),
            "tau": s.get("tau"), "tau_source": s.get("tau_source"),
            "degenerate": s.get("tau") is None,
            "n_calib": s.get("n_calib"), "n_fail": s.get("n_fail"), "n_succ": s.get("n_succ"),
            "tpr": s.get("tpr"), "tpr_lo": tpr_ci[0], "tpr_hi": tpr_ci[1],
            "fpr": s.get("fpr"), "fpr_lo": fpr_ci[0], "fpr_hi": fpr_ci[1],
            "lead_mean": s.get("lead_mean"), "lead_p50": s.get("lead_p50"),
            "lead_p10": s.get("lead_p10"), "aucpdt": s.get("aucpdt"),
            "roc_auc": s.get("roc_auc"), "bacc": s.get("bacc"),
            "fire_frac_mean": s.get("fire_frac_mean"), "holdout_fpr": s.get("holdout_fpr"),
            "holdout_fpr_k1": s.get("holdout_fpr_k1"), "eps_prog": s.get("eps_prog"),
        }
        out.append(row)
    out.sort(key=lambda r: (str(r["detector"]), r["alpha"] or 0.0, r["n_calib"] or 0))
    return out


LAYER_AB_COLUMNS = [
    "alpha", "detector", "arm_id", "pilot", "n_layer_a", "n_layer_b",
    "layer_a_tpr", "layer_a_tpr_lo", "layer_a_tpr_hi",
    "layer_b_flagged_rate", "layer_b_flagged_lo", "layer_b_flagged_hi", "tpr_gap",
    "layer_a_fpr", "layer_a_fpr_lo", "layer_a_fpr_hi",
    "layer_b_false_trip_rate", "layer_b_false_trip_lo", "layer_b_false_trip_hi", "fpr_gap",
]


def _detector_of_arm(r: dict) -> str | None:
    det = r.get("detector") or r.get("meta_detector_family")
    return str(det) if det is not None else None


def layer_ab_table(run: dict) -> list[dict]:
    """Layer A (open-loop, artefact tau) vs Layer B (closed-loop, d = 0, sync) per (alpha, detector)."""
    out = []
    d0_arms = [
        r for r in run["arms"]
        if r.get("meta_role") == "enforce" and r.get("meta_d") == 0 and r.get("meta_exec") == "sync"
    ]
    for s in run["sweep"]:
        if s.get("tau_source") != "artefact":
            continue
        a = alpha_value(s.get("alpha_num"), s.get("alpha_den"))
        det = s.get("detector")
        for r in d0_arms:
            r_alpha = alpha_from_text(r.get("alpha"))
            if r_alpha is None:
                r_alpha = alpha_value(r.get("meta_alpha_pct"), 100)
            if _detector_of_arm(r) != det:
                continue
            if det != "t0" and (a is None or r_alpha is None or abs(a - r_alpha) > 1e-12):
                continue
            tpr_ci = s.get("tpr_ci") or [None, None]
            fpr_ci = s.get("fpr_ci") or [None, None]
            row = {
                "alpha": a, "detector": det, "arm_id": r.get("arm_id"), "pilot": r.get("pilot"),
                "n_layer_a": (s.get("n_fail") or 0) + (s.get("n_succ") or 0), "n_layer_b": r.get("n"),
                "layer_a_tpr": s.get("tpr"), "layer_a_tpr_lo": tpr_ci[0], "layer_a_tpr_hi": tpr_ci[1],
                "layer_b_flagged_rate": r.get("flagged_rate"), "layer_b_flagged_lo": r.get("flagged_lo"),
                "layer_b_flagged_hi": r.get("flagged_hi"),
                "tpr_gap": _sub(r.get("flagged_rate"), s.get("tpr")),
                "layer_a_fpr": s.get("fpr"), "layer_a_fpr_lo": fpr_ci[0], "layer_a_fpr_hi": fpr_ci[1],
                "layer_b_false_trip_rate": r.get("false_trip_rate"),
                "layer_b_false_trip_lo": r.get("false_trip_lo"), "layer_b_false_trip_hi": r.get("false_trip_hi"),
                "fpr_gap": _sub(r.get("false_trip_rate"), s.get("fpr")),
            }
            out.append(row)
    out.sort(key=lambda r: (str(r["detector"]), r["alpha"] or 0.0, str(r["arm_id"])))
    return out


def _sub(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


LEAD_COLUMNS = ["source", "id", "alpha", "detector", "delay_steps", "exec_mode", "pilot", "n",
                "lead_mean", "lead_p50", "lead_p10", "aucpdt", "eps_prog"]


def lead_table(run: dict) -> list[dict]:
    out = []
    for r in sorted(run["arms"], key=arm_sort_key):
        if r.get("lead_mean") is None and r.get("aucpdt") is None:
            continue
        out.append({
            "source": "curve", "id": r.get("arm_id"), "alpha": alpha_from_text(r.get("alpha")),
            "detector": _detector_of_arm(r), "delay_steps": r.get("delay_steps"),
            "exec_mode": r.get("exec_mode"), "pilot": r.get("pilot"), "n": r.get("n_fail_baseline"),
            "lead_mean": r.get("lead_mean"), "lead_p50": r.get("lead_p50"), "lead_p10": r.get("lead_p10"),
            "aucpdt": r.get("aucpdt"), "eps_prog": r.get("eps_prog"),
        })
    for s in sweep_table(run):
        out.append({
            "source": "sweep", "id": f"{s['detector']}@{s['alpha_num']}/{s['alpha_den']}",
            "alpha": s["alpha"], "detector": s["detector"], "delay_steps": 0, "exec_mode": "sync",
            "pilot": None, "n": s["n_fail"], "lead_mean": s["lead_mean"], "lead_p50": s["lead_p50"],
            "lead_p10": s["lead_p10"], "aucpdt": s["aucpdt"], "eps_prog": s["eps_prog"],
        })
    return out


TCE_COLUMNS = ["arm_id", "family", "delay_steps", "exec_mode", "pilot", "n", "tce_valid_frac",
               "tce_expected_invalid"]


def tce_table(run: dict) -> list[dict]:
    out = []
    for r in sorted(run["arms"], key=arm_sort_key):
        if r.get("tce_valid_frac") is None:
            continue
        d = r.get("delay_steps")
        expected_invalid = bool(r.get("exec_mode") == "sync" and d is not None and int(d) >= 7)
        out.append({
            "arm_id": r.get("arm_id"), "family": r.get("meta_family"), "delay_steps": d,
            "exec_mode": r.get("exec_mode"), "pilot": r.get("pilot"), "n": r.get("n"),
            "tce_valid_frac": r.get("tce_valid_frac"), "tce_expected_invalid": expected_invalid,
        })
    return out


LATENCY_COLUMNS = ["source", "id", "n", "p50_ns", "p90_ns", "p99_ns", "p999_ns", "p9999_ns",
                   "max_ns", "latency_label"]


def _scale_ns(v, unit: str):
    if v is None:
        return None
    return float(v) * {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}[unit]


def latency_table(run: dict) -> list[dict]:
    """Every wall-clock number with its latency_label; the label is never dropped."""
    out = []
    for r in sorted(run["arms"], key=arm_sort_key):
        if r.get("latency_p50_ns") is None:
            continue
        out.append({
            "source": "curve", "id": r.get("arm_id"), "n": r.get("n"),
            "p50_ns": r.get("latency_p50_ns"), "p90_ns": None, "p99_ns": r.get("latency_p99_ns"),
            "p999_ns": None, "p9999_ns": None, "max_ns": r.get("latency_max_ns"),
            "latency_label": r.get("latency_label") or WSL2_LABEL,
        })
    if run.get("bench_hist"):
        for series, buckets in sorted(run["bench_hist"].items()):
            pc = hist_percentiles(buckets)
            out.append({
                "source": "bench", "id": series, "n": pc.get("n"),
                "p50_ns": pc.get("p50"), "p90_ns": pc.get("p90"), "p99_ns": pc.get("p99"),
                "p999_ns": pc.get("p99.9"), "p9999_ns": pc.get("p99.99"), "max_ns": pc.get("max"),
                "latency_label": WSL2_LABEL,
            })
    mb = run.get("microbench")
    if isinstance(mb, dict):
        label = mb.get("environment") or mb.get("latency_label") or WSL2_LABEL
        for key in sorted(mb.keys()):
            v = mb[key]
            unit = None
            for suffix, u in (("_ns", "ns"), ("_us", "us"), ("_ms", "ms"), ("_s", "s")):
                if key.endswith(suffix):
                    unit = u
            if unit is None or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            out.append({
                "source": "microbench", "id": key, "n": None, "p50_ns": _scale_ns(v, unit),
                "p90_ns": None, "p99_ns": None, "p999_ns": None, "p9999_ns": None, "max_ns": None,
                "latency_label": label,
            })
    return out


# ------------------------------------------------------------------ cross-check


def _close(a, b, tol) -> bool:
    if a is None or b is None:
        return False
    a = float(a)
    b = float(b)
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return abs(a - b) <= tol


def _count(rate, n) -> int | None:
    if rate is None or n is None:
        return None
    k = int(round(float(rate) * int(n)))
    return k


def _check(report: list, where: str, what: str, expected, got, tol, note: str = ""):
    ok = _close(expected, got, tol)
    entry = {"where": where, "what": what, "rust": got, "python": expected, "tol": tol, "ok": ok}
    if note:
        entry["note"] = note
    report.append(entry)
    return ok


def _skip(report: list, where: str, what: str, why: str):
    report.append({"where": where, "what": what, "skipped": why, "ok": True})


def cross_check(run: dict) -> tuple[bool, list[dict]]:
    """Recompute CP / McNemar / bootstrap / bacc from the CSV numbers; never their source."""
    report: list[dict] = []
    by_id = {str(r.get("arm_id")): r for r in run["arms"]}
    for r in sorted(run["arms"], key=arm_sort_key):
        arm = str(r.get("arm_id"))
        n = r.get("n")
        if not isinstance(n, int) or n <= 0:
            _skip(report, arm, "all", "n missing or zero")
            continue
        k = _count(r.get("success_rate"), n)
        if k is not None and r.get("success_lo") is not None:
            lo, hi = stats.clopper_pearson(k, n)
            _check(report, arm, "success_lo", lo, r.get("success_lo"), stats.EXACT_TOL)
            _check(report, arm, "success_hi", hi, r.get("success_hi"), stats.EXACT_TOL)
        nf = r.get("n_fail_baseline")
        ns = r.get("n_succ_baseline")
        if isinstance(nf, int) and r.get("flagged_rate") is not None and r.get("flagged_lo") is not None:
            kf = _count(r.get("flagged_rate"), nf)
            lo, hi = stats.clopper_pearson(kf, nf) if nf > 0 else (0.0, 1.0)
            _check(report, arm, "flagged_lo", lo, r.get("flagged_lo"), stats.EXACT_TOL)
            _check(report, arm, "flagged_hi", hi, r.get("flagged_hi"), stats.EXACT_TOL)
        else:
            _skip(report, arm, "flagged_ci", "no n_fail_baseline (curve json absent)")
        if isinstance(ns, int) and r.get("false_trip_rate") is not None and r.get("false_trip_lo") is not None:
            ks = _count(r.get("false_trip_rate"), ns)
            lo, hi = stats.clopper_pearson(ks, ns) if ns > 0 else (0.0, 1.0)
            _check(report, arm, "false_trip_lo", lo, r.get("false_trip_lo"), stats.EXACT_TOL)
            _check(report, arm, "false_trip_hi", hi, r.get("false_trip_hi"), stats.EXACT_TOL)
        else:
            _skip(report, arm, "false_trip_ci", "no n_succ_baseline (curve json absent)")
        for suffix, base_succ, label in (
            ("", ns, "delta_vs_baseline"),
            ("_lc", _lc_successes(r, by_id), "delta_vs_latency_control"),
        ):
            b = r.get(f"mcnemar_b{suffix}")
            c = r.get(f"mcnemar_c{suffix}")
            p = r.get(f"mcnemar_p{suffix}")
            if b is None or c is None:
                _skip(report, arm, label, "no discordant counts in the CSV row")
                continue
            _check(report, arm, f"mcnemar_p{suffix}", stats.mcnemar_exact(int(b), int(c)), p, stats.EXACT_TOL)
            diff = r.get(label)
            if diff is None or k is None:
                continue
            if not isinstance(base_succ, int):
                _skip(report, arm, f"{label}_bootstrap", "no baseline success count available")
                continue
            table = stats.orient_contingency(n, k, base_succ, int(b), int(c))
            if table is None:
                report.append({"where": arm, "what": f"{label}_contingency", "ok": False,
                               "note": f"n={n} succ_arm={k} succ_base={base_succ} b={b} c={c} do not form a 2x2 table"})
                continue
            both, base_only, arm_only = table
            _check(report, arm, label, (arm_only - base_only) / float(n), diff, stats.EXACT_TOL)
            base_arr, arm_arr = stats.pairs_from_contingency(n, both, base_only, arm_only)
            d, lo, hi = stats.paired_bootstrap_diff(base_arr, arm_arr)
            _check(report, arm, f"{label}_lo", lo, r.get(f"{label}_lo"), stats.BOOT_TOL,
                   "different RNG stream than lictor-calib (splitmix64): Monte-Carlo band")
            _check(report, arm, f"{label}_hi", hi, r.get(f"{label}_hi"), stats.BOOT_TOL,
                   "different RNG stream than lictor-calib (splitmix64): Monte-Carlo band")
    for s in run["sweep"]:
        where = f"sweep {s.get('detector')}@{s.get('alpha_num')}/{s.get('alpha_den')} n_calib={s.get('n_calib')}"
        nf, ns = s.get("n_fail"), s.get("n_succ")
        tpr_ci = s.get("tpr_ci") or [None, None]
        fpr_ci = s.get("fpr_ci") or [None, None]
        if isinstance(nf, int) and nf > 0 and s.get("tpr") is not None and tpr_ci[0] is not None:
            lo, hi = stats.clopper_pearson(_count(s["tpr"], nf), nf)
            _check(report, where, "tpr_ci_lo", lo, tpr_ci[0], 1e-6)
            _check(report, where, "tpr_ci_hi", hi, tpr_ci[1], 1e-6)
        if isinstance(ns, int) and ns > 0 and s.get("fpr") is not None and fpr_ci[0] is not None:
            lo, hi = stats.clopper_pearson(_count(s["fpr"], ns), ns)
            _check(report, where, "fpr_ci_lo", lo, fpr_ci[0], 1e-6)
            _check(report, where, "fpr_ci_hi", hi, fpr_ci[1], 1e-6)
        if s.get("tpr") is not None and s.get("fpr") is not None and s.get("bacc") is not None:
            _check(report, where, "bacc", stats.bacc(float(s["tpr"]), float(s["fpr"])), s["bacc"], 1e-6)
    ok = all(e.get("ok", False) for e in report)
    return ok, report


def _lc_successes(r: dict, by_id: dict) -> int | None:
    lc = r.get("latency_control_arm")
    if not lc:
        return None
    row = by_id.get(str(lc))
    if row is None or row.get("n") is None or row.get("success_rate") is None:
        return None
    if row.get("n") != r.get("n"):
        return None
    return _count(row.get("success_rate"), row.get("n"))


# --------------------------------------------------------------------- writers


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    return str(v)


def write_csv(path, rows: list[dict], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(columns)
        for r in rows:
            w.writerow([_fmt(r.get(c)) for c in columns])


def write_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


def _json_safe(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return repr(obj)
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def render_table(rows: list[dict], columns: list[str], cap: int = 40) -> str:
    """Deterministic, capped, aligned text table."""
    def cell(v):
        if isinstance(v, float) and not (math.isnan(v) or math.isinf(v)):
            return f"{v:.4g}"
        s = _fmt(v)
        return s if len(s) <= 24 else s[:21] + "..."
    shown = rows[:cap]
    grid = [[cell(r.get(c)) for c in columns] for r in shown]
    widths = [max(len(c), *(len(g[i]) for g in grid)) if grid else len(c) for i, c in enumerate(columns)]
    lines = ["  ".join(c.ljust(widths[i]) for i, c in enumerate(columns))]
    for g in grid:
        lines.append("  ".join(g[i].ljust(widths[i]) for i in range(len(columns))))
    if len(rows) > cap:
        lines.append(f"... ({len(rows) - cap} more rows in the CSV)")
    return "\n".join(lines)


def analyze(run_dir, out_dir=None, bench_csv=None, calib_dir=None) -> dict:
    run = load_run(run_dir, bench_csv=bench_csv, calib_dir=calib_dir)
    out = Path(out_dir) if out_dir else Path(run_dir) / "analysis"
    tables = {
        "arms": (arms_table(run), ARMS_COLUMNS),
        "sweep": (sweep_table(run), SWEEP_COLUMNS),
        "layer_ab": (layer_ab_table(run), LAYER_AB_COLUMNS),
        "lead": (lead_table(run), LEAD_COLUMNS),
        "tce_valid": (tce_table(run), TCE_COLUMNS),
        "latency": (latency_table(run), LATENCY_COLUMNS),
    }
    for name, (rows, cols) in tables.items():
        write_csv(out / f"{name}.csv", rows, cols)
    ok, report = cross_check(run)
    write_json(out / "cross_check.json", _json_safe({"ok": ok, "checks": report}))
    return {"run": run, "out": str(out), "tables": tables, "cross_check_ok": ok, "cross_check": report}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "scores":
        return main_scores(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="results/<run> directory")
    ap.add_argument("--out", default=None, help="output directory (default <run>/analysis)")
    ap.add_argument("--bench-csv", default=None, help="bucket dump of `lictor bench --csv`")
    ap.add_argument("--calib-dir", default=None, help="directory of calibration.<alpha>.json (default <run>)")
    ap.add_argument("--json", action="store_true", help="print the cross-check report as JSON")
    a = ap.parse_args(argv)
    res = analyze(a.run, a.out, a.bench_csv, a.calib_dir)
    if a.json:
        print(json.dumps(_json_safe({
            "out": res["out"], "cross_check_ok": res["cross_check_ok"],
            "checks": res["cross_check"],
            "rows": {k: len(v[0]) for k, v in res["tables"].items()},
        }), indent=2, sort_keys=True))
        return 0 if res["cross_check_ok"] else 1
    print(f"run       {res['run']['run_dir']}")
    print(f"out       {res['out']}")
    arms, cols = res["tables"]["arms"]
    print("\narms (Y1/Y2/Y3; deltas verbatim from lictor curve):")
    print(render_table(arms, ["pilot", "arm_id", "n", "success_rate", "success_lo", "success_hi",
                              "delta_vs_baseline", "mcnemar_p", "delta_vs_latency_control",
                              "flagged_rate", "false_trip_rate", "intervention_rate", "tce_valid_frac"]))
    ab, _ = res["tables"]["layer_ab"]
    print("\nlayer A vs B (tau_source == artefact only):")
    print(render_table(ab, ["alpha", "detector", "arm_id", "layer_a_tpr", "layer_b_flagged_rate",
                            "tpr_gap", "layer_a_fpr", "layer_b_false_trip_rate", "fpr_gap"]))
    lat, _ = res["tables"]["latency"]
    print(f"\nlatency ({TIMING_SENTENCE}):")
    print(render_table(lat, LATENCY_COLUMNS))
    bad = [e for e in res["cross_check"] if not e.get("ok", False)]
    if res["cross_check_ok"]:
        print(f"\nstats cross-check: ok ({len(res['cross_check'])} checks)")
    else:
        print(f"\nstats cross-check: {len(bad)} diffs")
        for e in bad[:40]:
            print(f"  {e.get('where')}  {e.get('what')}  rust={_fmt(e.get('rust'))}"
                  f"  python={_fmt(e.get('python'))}  tol={e.get('tol')}  {e.get('note', '')}")
    return 0 if res["cross_check_ok"] else 1


# ------------------------------------------------------ scores (F4 extraction)

SCORE_BANDS_COLUMNS = ["t", "group", "n", "q25", "q50", "q75", "tau"]


def _quantiles(xs: list[float]) -> tuple[float, float, float]:
    return (
        stats.quantile_nearest_rank(xs, 0.25),
        stats.quantile_nearest_rank(xs, 0.5),
        stats.quantile_nearest_rank(xs, 0.75),
    )


def extract_score_bands(run_dir, arm: str) -> list[dict]:
    """Per-tick median/IQR of `s` for successful vs failing episodes of one arm.

    Reads `<run>/<arm>/ledger.jsonl` (success labels; header line skipped) and
    `<run>/<arm>/ticks/<episode:06>.jsonl` (`s`, `tau` per tick, float-free). Ticks
    whose `s` is -inf (no valid gate term yet) are excluded from the quantiles.
    """
    arm_dir = Path(run_dir) / arm
    ledger = arm_dir / "ledger.jsonl"
    labels: dict[int, bool] = {}
    with open(ledger, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if "schema" in e and "episode_index" not in e:
                continue
            labels[int(e["episode_index"])] = bool(e["success"])
    per_t: dict[tuple[int, str], list[float]] = {}
    tau_seen: float | None = None
    for ep in sorted(labels):
        p = arm_dir / "ticks" / f"{ep:06d}.jsonl"
        if not p.exists():
            continue
        group = "success" if labels[ep] else "fail"
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                ev = json.loads(line)
                if "schema" in ev and "seq" not in ev:
                    continue
                s = f64hex(ev.get("s"))
                tau = f64hex(ev.get("tau"))
                if tau is not None and math.isfinite(tau) and tau_seen is None:
                    tau_seen = tau
                if s is None or not math.isfinite(s):
                    continue
                per_t.setdefault((int(ev["t"]), group), []).append(s)
    rows = []
    for (t, group) in sorted(per_t):
        xs = per_t[(t, group)]
        q25, q50, q75 = _quantiles(xs)
        rows.append({"t": t, "group": group, "n": len(xs), "q25": q25, "q50": q50, "q75": q75,
                     "tau": tau_seen})
    return rows


def main_scores(argv) -> int:
    ap = argparse.ArgumentParser(prog="analyze.py scores",
                                 description="per-tick score bands for figure F4 (reads ledger + ticks)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("-o", "--out", required=True, help="score_bands.csv to write")
    a = ap.parse_args(argv)
    rows = extract_score_bands(a.run, a.arm)
    write_csv(a.out, rows, SCORE_BANDS_COLUMNS)
    print(f"score bands: {len(rows)} rows -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
