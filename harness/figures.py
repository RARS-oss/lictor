# SPDX-License-Identifier: MIT
"""Figures F1-F9 of ARCHITECTURE 10.9 (WP-9). matplotlib only, SVG, no seaborn.

Inputs: the run directory read by `harness/analyze.py::load_run` -- `curve/summary.csv`
and the signed curve receipts `curve/<arm>.json`, `sweep.jsonl`, `microbench.json`,
the optional `lictor bench --csv` bucket dump and the optional `calibration.<alpha>.json`
files. Never the harness index, never an episode receipt. `--from-fixture` renders every
figure from the synthetic sample embedded below (`fixture_run()`), so CI renders
without a run.

Rules enforced here (ARCHITECTURE 10.7 / 10.9):

- every point is labelled with its n and carries a CI bar when the inputs carry one;
- `pilot == 1` (n < 100) points are drawn HOLLOW and are never connected by a line;
- sync arms are solid, async arms dashed (a second, separate curve);
- three metric lines per budget point on F1: TPR/FPR (flagged, averted, false-trip),
  task success with the paired deltas' CIs, intervention/escalation rate;
- the `tce_valid_frac` bar under F1(a) and the sync-`d >= 7` caption;
- twin top axes on F1 (100 ms/step on PushT; 20 ms is a re-scaling, not a measurement);
- every wall-clock number is labelled "WSL2, not RT" and F5 carries the frozen
  `latency_label` sentence verbatim;
- the PoNR proxy definition, the `t_fail = 0` artefact and the eps sensitivity on F3.

Optional per-episode / per-tick inputs (not carried by the frozen analysis products):

- `--score-bands F.csv`  columns t,group,n,q25,q50,q75,tau  (from `analyze.py scores`)
- `--leads F.csv`        columns alpha_num,alpha_den,detector,eps_prog,episode_index,lead
                         (blank lead = never detected)
- `--tamper-log F.txt`   the captured terminal output of scripts/demo.sh for F9

Usage:
    python harness/figures.py --run results/<run> --out docs/figures [--bench-csv F] ...
    python harness/figures.py --from-fixture --out /tmp/figs [--only F1,F5]
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import textwrap
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from harness import analyze, stats  # noqa: E402
from harness.analyze import WSL2_LABEL, WSL_TAG  # noqa: E402

FIG_FILES = {
    "F1": "curve-safety-latency.svg",
    "F2": "pareto-detection.svg",
    "F3": "lead-time.svg",
    "F4": "score-bands.svg",
    "F5": "latency-hist.svg",
    "F6": "success-delta.svg",
    "F7": "cp-validity.svg",
    "F8": "calib-economy.svg",
    "F9": "enforcement.svg",
}

# Validated categorical palette (dataviz reference instance; light surface), fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SLOTS
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e6e6e3"
SURFACE = "#ffffff"
BAR_GRAY = "#b8b7b2"

DETECTOR_ORDER = ["t0", "t1_tce", "t1_stall", "t1_full", "t01", "t01_and"]
HEADLINE_FAMILY = "t01-a05"
HEADLINE_DETECTOR = "t01"
HEADLINE_ALPHA = (5, 100)
MS_PER_STEP_PUSHT = 100.0
MS_PER_STEP_VLA = 20.0

# Caption fragments that tests and CI grep for (keep verbatim).
CAP_TCE = "sync d >= 7 has no chunk overlap; tce/acc are invalid there and the per-d valid fraction is shown"
CAP_RESCALE = "the second top axis re-scales d to a 20 ms clock and is a re-scaling, not a measurement"
CAP_PILOT = "hollow points are pilot points (n < 100, small_n) and never carry a line"
CAP_PONR = (
    "lead = t_fail - first_stop_tick with the point-of-no-return PROXY"
    " t_fail = min{t : c*_T - c*_t < eps_prog}, c*_t = max_{u<=t} coverage_u,"
    " taken from the paired obs-d0 trace"
)
CAP_TFAIL0 = (
    "an episode with no progress at all has t_fail = 0, so every detector shows a negative"
    " lead on it (an artefact of the proxy, not a late detection)"
)
CAP_EPS = "eps sensitivity over eps_prog in {0.01, 0.02, 0.05} is drawn as thin lines for the headline alpha"
CAP_LAYER_A = "Layer A (open-loop, tau_source == artefact); the Layer-B closed-loop point at the same alpha is the validity check"

TAMPER_REFERENCE = """$ lictor verify r.json --ticks rechained.jsonl
  verdict chain  HEAD MISMATCH recomputed=9d02... signed=5b7e...
  intact         NO
$ lictor verify r.json --pubkey 0000...
  pubkey         MISMATCH  (receipt 64e8281d05cf..., expected 0000...)
  intact         NO
$ lictor verify tampered.json
  schema         ok    lictor-receipt/v1
  signature      FAIL  (key 64e8281d05cf...)
  body digest    FAIL  be21...
  intact         NO
$ lictor verify r.json --ticks edited.jsonl
  verdict chain  BROKEN at seq=17
$ lictor ledger verify ledger.jsonl        # after deleting 43 entries
  ledger BROKEN at seq=3 -- entries are missing or edited; this curve point cannot be trusted
(reference output: the frozen strings of the CLI surface; pass --tamper-log to embed a capture)"""


# ------------------------------------------------------------------- styling


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 1.8,
        "lines.markersize": 6.5,
        "legend.frameon": False,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "svg.fonttype": "none",       # keep text as <text> so captions are greppable
        "svg.hashsalt": "lictor-figures",  # deterministic element ids
        "path.simplify": False,
        "axes.unicode_minus": False,  # ASCII only: '-' not U+2212 on tick labels
    })
    return plt


def _fmt_alpha(num, den) -> str:
    if num is None or den in (None, 0):
        return "alpha n/a"
    pct = 100.0 * num / den
    return f"alpha={pct:g}%"


def _wrap(parts, width: int = 118) -> str:
    """Wrap each caption part on its own; a part shorter than `width` stays on ONE line.

    The greppable sentences (CAP_*, WSL2_LABEL) are passed as separate parts so that
    wrapping can never split them across <text> elements.
    """
    if isinstance(parts, str):
        parts = [parts]
    out = []
    for p in parts:
        for para in p.split("\n"):
            out.append(textwrap.fill(para, width, break_on_hyphens=False, break_long_words=False))
    return "\n".join(out)


def _caption(fig, text, fig_h_in: float, width: int = 118) -> None:
    wrapped = _wrap(text, width)
    n_lines = wrapped.count("\n") + 1
    bottom = 0.018 + n_lines * (10.0 / 72.0) / fig_h_in
    fig.subplots_adjust(bottom=min(0.45, bottom + 0.05))
    fig.text(0.02, 0.012, wrapped, ha="left", va="bottom", fontsize=7, color=INK2, linespacing=1.35)


def _save(fig, out_dir: Path, key: str, plt) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / FIG_FILES[key]
    fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "harness/figures.py"})
    plt.close(fig)
    return path


def _n_label(ax, x, y, n, pilot: bool, dy: float = 6.0, color: str = MUTED, ha: str = "center"):
    txt = f"n={n}" + (" pilot" if pilot else "")
    ax.annotate(txt, (x, y), textcoords="offset points", xytext=(0, dy), fontsize=5.5,
                color=color, ha=ha, va="bottom", zorder=6)


def _marker_kwargs(color: str, pilot: bool, marker: str = "o") -> dict:
    if pilot:
        return {"marker": marker, "linestyle": "none", "markerfacecolor": SURFACE,
                "markeredgecolor": color, "markeredgewidth": 1.6, "color": color}
    return {"marker": marker, "linestyle": "none", "markerfacecolor": color,
            "markeredgecolor": SURFACE, "markeredgewidth": 1.0, "color": color}


def plot_series(ax, pts: list[dict], color: str, label: str, dashed: bool = False,
                marker: str = "o", n_labels: bool = True, ci: bool = True, zorder: int = 3,
                label_dy: float = 6.0):
    """Points {x, y, lo, hi, n, pilot}; pilot points hollow and never on a line."""
    from matplotlib.lines import Line2D

    pts = sorted((p for p in pts if p.get("y") is not None), key=lambda p: p["x"])
    ls = "--" if dashed else "-"
    for seg in segments_without_pilot(pts):
        if len(seg) >= 2:
            ax.plot([p["x"] for p in seg], [p["y"] for p in seg], linestyle=ls, color=color,
                    marker="none", zorder=zorder)
    for p in pts:
        if ci and p.get("lo") is not None and p.get("hi") is not None:
            e_lo = max(0.0, p["y"] - p["lo"])
            e_hi = max(0.0, p["hi"] - p["y"])
            ax.errorbar([p["x"]], [p["y"]], yerr=[[e_lo], [e_hi]], fmt="none", ecolor=color,
                        elinewidth=1.0, capsize=2, zorder=zorder)
        ax.plot([p["x"]], [p["y"]], zorder=zorder + 1, **_marker_kwargs(color, bool(p.get("pilot")), marker))
        if n_labels and p.get("n") is not None:
            _n_label(ax, p["x"], p["y"], p["n"], bool(p.get("pilot")), dy=label_dy)
    return Line2D([0], [0], color=color, linestyle=ls, marker=marker, markerfacecolor=color,
                  markeredgecolor=SURFACE, label=label)


def segments_without_pilot(pts: list[dict]) -> list[list[dict]]:
    """Runs of consecutive non-pilot points; a pilot point breaks the run and joins none."""
    segs: list[list[dict]] = []
    cur: list[dict] = []
    for p in pts:
        if p.get("pilot"):
            if cur:
                segs.append(cur)
            cur = []
        else:
            cur.append(p)
    if cur:
        segs.append(cur)
    return segs


def _pt(r: dict, x, y_key: str, lo_key: str | None = None, hi_key: str | None = None,
        n_key: str = "n") -> dict:
    return {"x": x, "y": r.get(y_key), "lo": r.get(lo_key) if lo_key else None,
            "hi": r.get(hi_key) if hi_key else None, "n": r.get(n_key),
            "pilot": bool(r.get("pilot")), "arm_id": r.get("arm_id")}


def _family_rows(arms: list[dict], family: str, exec_mode: str, variant=None) -> list[dict]:
    return [r for r in arms if r.get("meta_family") == family and r.get("meta_exec") == exec_mode
            and r.get("meta_variant") == variant and r.get("meta_d") is not None]


def _bench_percentiles(run: dict, series: str = "verdict") -> dict:
    h = (run.get("bench_hist") or {}).get(series)
    return analyze.hist_percentiles(h) if h else {}


# ------------------------------------------------------------------------ F1


def fig_f1(run: dict, out_dir: Path, plt) -> Path:
    from matplotlib.lines import Line2D

    arms = run["arms"]
    fig_h = 13.0
    fig = plt.figure(figsize=(11.0, fig_h))
    gs = fig.add_gridspec(4, 1, height_ratios=[3.2, 0.75, 3.0, 3.0], hspace=0.42,
                          left=0.07, right=0.70, top=0.86)
    ax_a = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharex=ax_a)
    ax_b = fig.add_subplot(gs[2], sharex=ax_a)
    ax_c = fig.add_subplot(gs[3], sharex=ax_a)

    fam_sync = _family_rows(arms, HEADLINE_FAMILY, "sync")
    fam_async = _family_rows(arms, HEADLINE_FAMILY, "async")
    t0_sync = _family_rows(arms, "t0", "sync")
    obs_sync = _family_rows(arms, "obs", "sync")
    obs_async = _family_rows(arms, "obs", "async")

    # (a) flagged / averted / false-trip: colour = metric, marker = family, dash = async
    handles = []
    metrics = [("flagged_rate", "flagged_lo", "flagged_hi", BLUE, "flagged (TPR, of obs-d0 failures)"),
               ("averted_rate", None, None, AQUA, "averted (of obs-d0 failures; no CI in summary.csv)"),
               ("false_trip_rate", "false_trip_lo", "false_trip_hi", ORANGE, "false-trip (FPR, of obs-d0 successes)")]
    for mi, (key, lo, hi, color, label) in enumerate(metrics):
        pts = [_pt(r, r["meta_d"], key, lo, hi, "n_fail_baseline" if key != "false_trip_rate" else "n_succ_baseline")
               for r in fam_sync]
        handles.append(plot_series(ax_a, pts, color, f"{label}", marker="o", label_dy=6.0 if mi != 2 else -12.0))
        pts = [_pt(r, r["meta_d"], key, lo, hi, "n_fail_baseline" if key != "false_trip_rate" else "n_succ_baseline")
               for r in fam_async]
        if pts:
            plot_series(ax_a, pts, color, label, dashed=True, marker="s", n_labels=False)
        pts = [_pt(r, r["meta_d"], key, lo, hi, "n_fail_baseline" if key != "false_trip_rate" else "n_succ_baseline")
               for r in t0_sync]
        if pts:
            plot_series(ax_a, pts, color, label, marker="^", n_labels=False)
    handles += [Line2D([0], [0], color=INK2, linestyle="-", marker="o", label=f"{HEADLINE_FAMILY} sync"),
                Line2D([0], [0], color=INK2, linestyle="--", marker="s", label=f"{HEADLINE_FAMILY} async"),
                Line2D([0], [0], color=INK2, linestyle="-", marker="^", label="t0 (Tier 0 only)"),
                Line2D([0], [0], color=INK2, linestyle="none", marker="o", markerfacecolor=SURFACE,
                       markeredgecolor=INK2, label="pilot point (n < 100): hollow, no line")]
    ax_a.set_ylim(-0.02, 1.05)
    ax_a.set_ylabel("episode-level rate")
    ax_a.set_title("(a) detection: flagged / averted / false-trip per budget point", loc="left")
    ax_a.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1, fontsize=6.2)
    sec1 = ax_a.secondary_xaxis("top", functions=(lambda d: d * MS_PER_STEP_PUSHT, lambda ms: ms / MS_PER_STEP_PUSHT))
    sec1.set_xlabel("added staleness in ms at PushT's 100 ms control step (simulated time)", fontsize=7)
    sec2 = ax_a.secondary_xaxis(1.28, functions=(lambda d: d * MS_PER_STEP_VLA, lambda ms: ms / MS_PER_STEP_VLA))
    sec2.set_xlabel("ms at a 20 ms clock -- a re-scaling, not a measurement", fontsize=7)
    for s in (sec1, sec2):
        s.tick_params(labelsize=6.5)

    # tce_valid_frac bar under (a)
    xs = [r["meta_d"] for r in fam_sync]
    ys = [r.get("tce_valid_frac") or 0.0 for r in fam_sync]
    if xs:
        ax_t.bar(xs, ys, width=0.42, color=BAR_GRAY, label="tce_valid_frac sync", zorder=3)
        for x, y, r in zip(xs, ys, fam_sync):
            ax_t.text(x, y + 0.05, f"{y:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK2)
    xa = [r["meta_d"] + 0.22 for r in fam_async]
    ya = [r.get("tce_valid_frac") or 0.0 for r in fam_async]
    if xa:
        ax_t.bar(xa, ya, width=0.2, facecolor=SURFACE, edgecolor=BAR_GRAY, linewidth=1.2,
                 label="tce_valid_frac async", zorder=3)
    ax_t.set_ylim(0, 1.25)
    ax_t.set_yticks([0, 1])
    ax_t.set_ylabel("tce valid", fontsize=7)
    ax_t.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.2)
    plt.setp(ax_t.get_xticklabels(), visible=False)

    # (b) success: colour = family, dash = async
    handles = []
    for rows, color, label, dashed, marker, dy, nl in (
        (fam_sync, BLUE, f"{HEADLINE_FAMILY} sync (fuse on)", False, "o", 8.0, True),
        (fam_async, BLUE, f"{HEADLINE_FAMILY} async (fuse on)", True, "s", 8.0, False),
        (t0_sync, AQUA, "t0 (Tier 0 only)", False, "^", 16.0, True),
        (obs_sync, ORANGE, "obs-d* sync (latency-only control, fuse off)", False, "o", -13.0, True),
        (obs_async, ORANGE, "obs-d* async (fuse off)", True, "s", -13.0, False),
    ):
        pts = [_pt(r, r["meta_d"], "success_rate", "success_lo", "success_hi") for r in rows]
        if pts:
            handles.append(plot_series(ax_b, pts, color, label, dashed=dashed, marker=marker, label_dy=dy, n_labels=nl))
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel("task success (95 % Clopper-Pearson)")
    ax_b.set_title("(b) task success: fuse on vs the fuse-off latency control at the same d", loc="left")
    ax_b.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.2)
    # paired deltas vs the latency control, printed once per d (headline sync family) under the x axis area
    lines = []
    for r in sorted(fam_sync, key=lambda r: r["meta_d"]):
        d = r.get("delta_vs_latency_control")
        lo, hi = r.get("delta_vs_latency_control_lo"), r.get("delta_vs_latency_control_hi")
        if d is not None and lo is not None and hi is not None:
            lines.append(f"d={r['meta_d']}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]" + (" pilot" if r.get("pilot") else ""))
    if lines:
        ax_b.text(0.01, 0.03, "delta vs latency control (sync), bootstrap 95 % CI:  " + "   ".join(lines),
                  transform=ax_b.transAxes, fontsize=5.4, color=INK2, va="bottom", ha="left")

    # (c) intervention / escalation
    handles = []
    for key, color, label in (("intervention_rate", BLUE, "intervention rate (episodes with >= 1 substitution)"),
                              ("escalation_rate", ORANGE, "escalation rate (episodes reaching Escalated)"),
                              ("intervention_tick_frac", VIOLET, "substituted ticks / ticks")):
        pts = [_pt(r, r["meta_d"], key) for r in fam_sync]
        handles.append(plot_series(ax_c, pts, color, label, marker="o"))
        pts = [_pt(r, r["meta_d"], key) for r in fam_async]
        if pts:
            plot_series(ax_c, pts, color, label, dashed=True, marker="s", n_labels=False)
        pts = [_pt(r, r["meta_d"], key) for r in t0_sync]
        if pts:
            plot_series(ax_c, pts, color, label, marker="^", n_labels=False)
    ax_c.set_ylim(-0.02, 1.05)
    ax_c.set_ylabel("rate")
    ax_c.set_xlabel("d = injected staleness, control steps (100 ms each on PushT)")
    ax_c.set_title("(c) intervention cost", loc="left")
    ax_c.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.2)
    all_d = sorted({r["meta_d"] for r in fam_sync + fam_async + t0_sync + obs_sync + obs_async})
    if all_d:
        ax_a.set_xlim(min(all_d) - 0.5, max(all_d) + 0.5)
        ax_c.set_xticks(all_d)

    # decide_ns inset (WSL2, not RT)
    ins = ax_c.inset_axes([0.36, 0.42, 0.34, 0.50])
    _latency_inset(ins, run, arms)
    fig.suptitle("F1 -- the safety-vs-latency curve on PushT (x = simulated staleness, virtualisation-independent)",
                 x=0.02, ha="left", fontsize=9.5, y=0.985)
    n_pilot = sum(1 for r in arms if r.get("pilot"))
    cap = [
        f"F1. Three metric lines per budget point; sync solid, async dashed; {CAP_PILOT}"
        f" ({n_pilot} pilot point(s) in this run). Every point carries n and a 95 % Clopper-Pearson CI"
        " where summary.csv carries one; the paired deltas vs the latency control are printed under (b)"
        " with their bootstrap CIs (10 000 resamples).",
        f"{CAP_TCE}.",
        f"100 ms per step on PushT; {CAP_RESCALE}.",
        f"The inset is the decide_ns histogram ({WSL_TAG}):",
        f"{WSL2_LABEL}.",
    ]
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F1", plt)


def _latency_inset(ins, run: dict, arms: list[dict]) -> None:
    ins.set_title(f"decide_ns ({WSL_TAG})", fontsize=6.5, loc="left")
    ins.tick_params(labelsize=5.5)
    ins.set_xscale("log")
    hist = (run.get("bench_hist") or {}).get("verdict")
    if hist:
        _draw_hist(ins, hist, BLUE, annotate=("p50", "p99", "max"), fontsize=5)
        ins.set_xlabel("ns (lictor bench)", fontsize=6)
    else:
        rows = [r for r in arms if r.get("latency_p50_ns")]
        ys = list(range(len(rows)))
        for y, r in zip(ys, rows):
            ins.plot([r["latency_p50_ns"], r["latency_p99_ns"], r["latency_max_ns"]], [y, y, y],
                     marker="|", color=BLUE, linewidth=0.8)
        ins.set_yticks(ys)
        ins.set_yticklabels([str(r["arm_id"]) for r in rows], fontsize=4.5)
        ins.set_xlabel("ns: p50 | p99 | max per arm", fontsize=6)
    ins.grid(True, linewidth=0.4)


def _draw_hist(ax, buckets, color, annotate=("p50", "p99", "p99.9", "p99.99", "max"), fontsize=6):
    lefts, widths, counts = [], [], []
    prev = None
    for upper, count in buckets:
        left = prev if prev is not None else upper * 0.8
        if left <= 0:
            left = upper * 0.8
        lefts.append(left)
        widths.append(max(upper - left, upper * 1e-3))
        counts.append(count)
        prev = upper
    ax.bar(lefts, counts, width=widths, align="edge", color=color, alpha=0.85, linewidth=0, zorder=3)
    pc = analyze.hist_percentiles(buckets)
    ymax = max(counts) if counts else 1
    for i, key in enumerate(annotate):
        v = pc.get(key)
        if v is None:
            continue
        ax.axvline(v, color=INK2, linewidth=0.7, zorder=4)
        ax.text(v, ymax * (0.62 - 0.13 * i), f"{key} {v / 1000.0:.3g} us", fontsize=fontsize,
                color=INK2, rotation=90, ha="right", va="top",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.5, "alpha": 0.8})
    ax.set_ylabel("count", fontsize=6)
    return pc


# ------------------------------------------------------------------------ F2


def _alpha_of(s: dict) -> float:
    return analyze.alpha_value(s.get("alpha_num"), s.get("alpha_den")) or 0.0


def _detector_color(det: str) -> str:
    if det in DETECTOR_ORDER:
        return SLOTS[DETECTOR_ORDER.index(det)]
    return SLOTS[min(7, len(DETECTOR_ORDER))]


def fig_f2(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.6
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.0, fig_h), gridspec_kw={"wspace": 0.3, "left": 0.07, "right": 0.98, "top": 0.9})
    sweep = [s for s in run["sweep"] if s.get("detector")]
    main_ncalib = _main_n_calib(sweep)
    sweep = [s for s in sweep if s.get("n_calib") == main_ncalib]
    dets = sorted({s["detector"] for s in sweep}, key=lambda d: (DETECTOR_ORDER.index(d) if d in DETECTOR_ORDER else 99, d))
    handles = []
    for det in dets:
        rows = sorted((s for s in sweep if s["detector"] == det), key=_alpha_of)
        color = _detector_color(det)
        aucs = [s.get("roc_auc") for s in rows if s.get("roc_auc") is not None]
        auc = sum(aucs) / len(aucs) if aucs else float("nan")
        xs = [s.get("fpr") for s in rows]
        ys = [s.get("tpr") for s in rows]
        ax.plot(xs, ys, color=color, linestyle="-", marker="none", zorder=3)
        has_tau = det != "t0" and any(s.get("gate") for s in rows)
        seen_xy = set()
        for j, s in enumerate(rows):
            n_pt = (s.get("n_fail") or 0) + (s.get("n_succ") or 0)
            degenerate = has_tau and s.get("tau") is None
            tci = s.get("tpr_ci") or [s["tpr"], s["tpr"]]
            fci = s.get("fpr_ci") or [s["fpr"], s["fpr"]]
            ax.errorbar([s["fpr"]], [s["tpr"]], xerr=[[max(0, s["fpr"] - fci[0])], [max(0, fci[1] - s["fpr"])]],
                        yerr=[[max(0, s["tpr"] - tci[0])], [max(0, tci[1] - s["tpr"])]], fmt="none",
                        ecolor=color, elinewidth=0.8, capsize=1.5, zorder=3)
            ax.plot([s["fpr"]], [s["tpr"]], zorder=4, **_marker_kwargs(color, degenerate, "x" if degenerate else "o"))
            tag = _fmt_alpha(s.get("alpha_num"), s.get("alpha_den")).replace("alpha=", "a=")
            src = "" if (s.get("tau_source") == "artefact" or not has_tau) else " (fit)"
            xy = (round(s["fpr"], 4), round(s["tpr"], 4))
            if xy in seen_xy:
                continue      # t0 sits on one point for every alpha: label it once
            seen_xy.add(xy)
            label = f"{tag}{src} n={n_pt}" + (" degenerate tau=+inf" if degenerate else "")
            if not has_tau:
                label = f"{det} (no tau) n={n_pt}"
            ax.annotate(label, (s["fpr"], s["tpr"]), textcoords="offset points",
                        xytext=(4, 2 + 5 * (dets.index(det) % 3)), fontsize=4.6, color=MUTED)
        handles.append(ax.plot([], [], color=color, marker="o", label=f"{det}  (ROC-AUC {auc:.2f})")[0])
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=0.8, zorder=1)
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(-0.01, 1.05)
    ax.set_xlabel("false-positive rate (episode fired | obs-d0 success), 95 % CP CI")
    ax.set_ylabel("true-positive rate (episode fired | obs-d0 failure), 95 % CP CI")
    ax.set_title("(a) Layer A: TPR vs FPR over the alpha grid, one line per detector subset", loc="left")
    ax.legend(handles=handles, loc="lower right")

    # (b) held-out FPR vs nominal alpha
    art = [s for s in sweep if s.get("tau_source") == "artefact" and s.get("holdout_fpr") is not None]
    by_alpha: dict[float, dict] = {}
    for s in art:
        by_alpha.setdefault(_alpha_of(s), s)
    calib_n = {analyze.alpha_value(c.get("alpha_num"), c.get("alpha_den")): c.get("n_holdout")
               for c in run.get("calibrations", []) if c.get("alpha_num") is not None}
    xs = sorted(by_alpha)
    if xs:
        ys_kn = [by_alpha[a]["holdout_fpr"] for a in xs]
        ys_k1 = [by_alpha[a].get("holdout_fpr_k1") for a in xs]
        lim = max(0.25, max(xs) * 1.2, max(y for y in ys_kn + [v for v in ys_k1 if v is not None]) * 1.2)
        ax2.plot([0, lim], [0, lim], color=GRID, linewidth=0.8, zorder=1)
        ax2.plot(xs, ys_kn, color=BLUE, marker="o", label="holdout FPR at the configured K-of-N")
        if all(v is not None for v in ys_k1):
            ax2.plot(xs, ys_k1, color=ORANGE, marker="^", label="holdout FPR at K = 1 (the rate alpha bounds)")
        for a, y in zip(xs, ys_kn):
            n_h = calib_n.get(a)
            if n_h:
                lo, hi = stats.clopper_pearson(int(round(y * n_h)), int(n_h))
                ax2.errorbar([a], [y], yerr=[[max(0, y - lo)], [max(0, hi - y)]], fmt="none", ecolor=BLUE,
                             elinewidth=0.8, capsize=1.5)
            ax2.annotate(f"n={n_h if n_h else 'holdout n/a'}", (a, y), textcoords="offset points",
                         xytext=(0, 6), fontsize=5.5, color=MUTED, ha="center")
        ax2.set_xlim(0, lim)
        ax2.set_ylim(0, lim)
        ax2.legend(loc="lower right")
    else:
        ax2.text(0.5, 0.5, "no tau_source == artefact points in sweep.jsonl", ha="center", va="center", color=INK2)
    ax2.set_xlabel("nominal alpha (calibration artefact)")
    ax2.set_ylabel("empirical held-out episode-level firing rate")
    ax2.set_title("(b) held-out FPR vs alpha (diagonal = the bound holds exactly)", loc="left")
    fig.suptitle("F2 -- Layer-A detection Pareto (open-loop, from sweep.jsonl)", x=0.02, ha="left", fontsize=9.5)
    cap = [
        f"F2. {CAP_LAYER_A}.",
        "Points are labelled with alpha and n = n_fail + n_succ of the eval traces;"
        " 'x' marks a degenerate alpha (k > n_calib, tau = +inf, never fires) which is a point about the"
        " calibration size, not data. (fit) marks a tau the sweep fitted itself (tau_source == fit); those are"
        f" never used for the Layer-A-vs-B comparison. t0 has no tau (Tier 0 fires at K = 1). n_calib = {main_ncalib}.",
    ]
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F2", plt)


def _main_n_calib(sweep: list[dict]):
    """The n_calib of the main sweep (the most frequent value; economy rows are the others)."""
    counts: dict = {}
    for s in sweep:
        counts[s.get("n_calib")] = counts.get(s.get("n_calib"), 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], -(kv[0] or 0)))[0][0]


# ------------------------------------------------------------------------ F3


def _cdf(values: list[float]):
    xs = sorted(values)
    n = len(xs)
    return xs, [(i + 1) / n for i in range(n)]


def fig_f3(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.8
    fig, ax = plt.subplots(figsize=(8.6, fig_h), gridspec_kw={"left": 0.08, "right": 0.98, "top": 0.9})
    leads = run.get("leads") or []
    head = [l for l in leads if l.get("detector") == HEADLINE_DETECTOR]
    eps_main = 0.02
    drawn = False
    if head:
        alphas = sorted({(l["alpha_num"], l["alpha_den"]) for l in head}, key=lambda a: a[0] / a[1])
        for i, (num, den) in enumerate(alphas):
            rows = [l for l in head if (l["alpha_num"], l["alpha_den"]) == (num, den) and abs(float(l.get("eps_prog", eps_main)) - eps_main) < 1e-12]
            det = [float(l["lead"]) for l in rows if l.get("lead") is not None]
            if not rows:
                continue
            color = SLOTS[i % len(SLOTS)]
            n_all = len(rows)
            if det:
                xs, ys = _cdf(det)
                ys = [y * len(det) / n_all for y in ys]  # undetected episodes never reach 1.0
                ax.step(xs, ys, where="post", color=color, label=f"{_fmt_alpha(num, den)}  n={n_all} failures, {len(det)} detected")
                ax.plot([xs[-1]], [ys[-1]], **_marker_kwargs(color, False, "o"))
                drawn = True
            if (num, den) == HEADLINE_ALPHA:
                for eps, ls in ((0.01, ":"), (0.05, "-.")):
                    rows_e = [l for l in head if (l["alpha_num"], l["alpha_den"]) == (num, den) and abs(float(l.get("eps_prog", -1)) - eps) < 1e-12]
                    det_e = [float(l["lead"]) for l in rows_e if l.get("lead") is not None]
                    if det_e:
                        xs, ys = _cdf(det_e)
                        ys = [y * len(det_e) / len(rows_e) for y in ys]
                        ax.step(xs, ys, where="post", color=color, linewidth=0.9, linestyle=ls,
                                label=f"{_fmt_alpha(num, den)} eps_prog={eps} (sensitivity)")
        ax.set_ylabel("fraction of obs-d0 failures detected with lead <= x")
    if not drawn:
        # fallback: quantile summary from sweep.jsonl (no per-episode leads in the frozen inputs)
        sweep = [s for s in run["sweep"] if s.get("detector") == HEADLINE_DETECTOR and s.get("n_calib") == _main_n_calib(run["sweep"])]
        sweep.sort(key=_alpha_of)
        for i, s in enumerate(sweep):
            color = SLOTS[i % len(SLOTS)]
            y = i
            p10, p50, mean = s.get("lead_p10"), s.get("lead_p50"), s.get("lead_mean")
            if p50 is None:
                continue
            ax.plot([p10, p50], [y, y], color=color, linewidth=2.5)
            ax.plot([p10], [y], marker="|", color=color, markersize=10)
            ax.plot([p50], [y], marker="o", color=color)
            ax.plot([mean], [y], marker="D", color=color, markerfacecolor=SURFACE)
            ax.annotate(f"{_fmt_alpha(s.get('alpha_num'), s.get('alpha_den'))}  n={s.get('n_fail')} failures  p10={p10:g} p50={p50:g} mean={mean:g}",
                        (p50, y), textcoords="offset points", xytext=(0, 7), fontsize=6, color=INK2, ha="center")
        ax.set_yticks([])
        ax.set_ylabel("per alpha: p10 -- p50 (o), mean (diamond)")
        ax.set_title("quantile summary from sweep.jsonl (per-episode leads not in the frozen inputs; pass --leads for the CDF)",
                     loc="left", fontsize=7.5, color=INK2)
    ax.axvline(0, color=INK2, linewidth=0.8)
    ax.text(0, ax.get_ylim()[1], " PoNR proxy (lead = 0)", fontsize=6, color=INK2, va="top")
    ax.set_xlabel("lead time in ticks before the point-of-no-return proxy (negative = detected after it)")
    if drawn:
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", fontsize=6.2)
    fig.suptitle("F3 -- lead-time distribution before the PoNR proxy, per alpha (Layer A, detector t01)", x=0.02, ha="left", fontsize=9.5)
    cap = [f"F3. {CAP_PONR}; eps_prog = {eps_main}.",
           f"{CAP_TFAIL0}.",
           f"{CAP_EPS}. TCE is a lagging feature (known one chunk late); its lag is inside these leads, not hidden."]
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F3", plt)


# ------------------------------------------------------------------------ F4


def fig_f4(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.4
    fig, ax = plt.subplots(figsize=(8.6, fig_h), gridspec_kw={"left": 0.08, "right": 0.98, "top": 0.9})
    bands = run.get("score_bands") or []
    tau = None
    for row in bands:
        if row.get("tau") is not None and math.isfinite(float(row["tau"])):
            tau = float(row["tau"])
            break
    if tau is None:
        for s in run["sweep"]:
            if s.get("detector") == HEADLINE_DETECTOR and (s.get("alpha_num"), s.get("alpha_den")) == HEADLINE_ALPHA and s.get("tau") is not None:
                tau = float(s["tau"])
                break
    if bands:
        n_by_group = {}
        for group, color in (("success", BLUE), ("fail", ORANGE)):
            rows = sorted((r for r in bands if r.get("group") == group), key=lambda r: r["t"])
            if not rows:
                continue
            ts = [r["t"] for r in rows]
            ax.fill_between(ts, [r["q25"] for r in rows], [r["q75"] for r in rows], color=color, alpha=0.15, linewidth=0)
            ax.plot(ts, [r["q50"] for r in rows], color=color, label=f"{group} episodes: median (line) +- IQR (wash), n={max(r['n'] for r in rows)} at peak")
            n_by_group[group] = max(r["n"] for r in rows)
        if tau is not None:
            ax.axhline(tau, color=INK2, linewidth=1.0, linestyle="-")
            ax.text(ax.get_xlim()[0], tau, f" tau = {tau:.3g} ({_fmt_alpha(*HEADLINE_ALPHA)})", fontsize=6.5, color=INK2, va="bottom")
        ax.set_xlabel("tick t (0 .. horizon_ticks-1)")
        ax.set_ylabel("aggregate score s_t (DNF over standardised features)")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "per-tick scores are not in the frozen analysis products;\n"
                          "run `python harness/analyze.py scores --run <run> --arm obs-d0 -o score_bands.csv`\n"
                          "and pass --score-bands score_bands.csv", ha="center", va="center", color=INK2, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("F4 -- score bands: median +- IQR of s_t for successful vs failing episodes, tau overlaid", x=0.02, ha="left", fontsize=9.5)
    cap = ("F4. s_t is the gated aggregate of ARCHITECTURE 5.4 (max over gate terms of the min z-score); tau is the split-conformal"
           " quantile of the per-episode max_t s_t over n_calib calibration successes. Ticks with no fully-valid gate term"
           " (s = -inf, e.g. before the first chunk overlap) are excluded from the quantiles. Observe-mode scores: the"
           " trajectory is the policy's own, so the bands are not distorted by substitutions.")
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F4", plt)


# ------------------------------------------------------------------------ F5


def fig_f5(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.6
    fig, axes = plt.subplots(1, 3, figsize=(11.5, fig_h), gridspec_kw={"wspace": 0.32, "left": 0.06, "right": 0.985, "top": 0.86})
    ax_d, ax_io, ax_ct = axes
    hist = run.get("bench_hist") or {}
    arms = run["arms"]
    # decide_ns
    ax_d.set_xscale("log")
    if hist.get("verdict"):
        pc = _draw_hist(ax_d, hist["verdict"], BLUE)
        ax_d.set_title(f"(a) decide_ns -- verdict, n={pc.get('n')} ({WSL_TAG})", loc="left")
        for series, color in (("tier0", AQUA), ("brake", ORANGE), ("tier1", VIOLET)):
            if hist.get(series):
                p = analyze.hist_percentiles(hist[series])
                ax_d.plot([], [], color=color, label=f"{series}: p50 {p.get('p50', 0) / 1e3:.3g} us, p99 {p.get('p99', 0) / 1e3:.3g} us")
        ax_d.plot([], [], color=BLUE, label="verdict (decide); verticals = p50/p99/p99.9/p99.99/max")
        ax_d.legend(loc="upper right", fontsize=5.5)
    else:
        rows = [r for r in arms if r.get("latency_p50_ns")]
        for i, r in enumerate(rows):
            ax_d.plot([r["latency_p50_ns"], r["latency_p99_ns"], r["latency_max_ns"]], [i, i, i], marker="|", color=BLUE)
            ax_d.annotate(f"n={r.get('n')}", (r["latency_max_ns"], i), textcoords="offset points", xytext=(4, 0), fontsize=5.5, color=MUTED, va="center")
        ax_d.set_yticks(list(range(len(rows))))
        ax_d.set_yticklabels([str(r["arm_id"]) for r in rows], fontsize=6)
        ax_d.set_title(f"(a) decide_ns per arm: p50 | p99 | max from summary.csv ({WSL_TAG})", loc="left")
    ax_d.set_xlabel("decide_ns (log)")
    # io_ns
    ax_io.set_xscale("log")
    if hist.get("io"):
        pc = _draw_hist(ax_io, hist["io"], ORANGE)
        ax_io.set_title(f"(b) io_ns -- pipe round trip, n={pc.get('n')} ({WSL_TAG})", loc="left")
    else:
        mb = run.get("microbench") or {}
        io_keys = sorted(k for k in mb if k.startswith("io_") and isinstance(mb[k], (int, float)))
        if io_keys:
            for i, k in enumerate(io_keys):
                unit = "ns" if k.endswith("_ns") else "us" if k.endswith("_us") else "ms"
                v = analyze._scale_ns(mb[k], unit)
                ax_io.plot([v], [i], marker="o", color=ORANGE)
                ax_io.annotate(k, (v, i), textcoords="offset points", xytext=(5, 0), fontsize=6, va="center", color=INK2)
            ax_io.set_yticks([])
            ax_io.set_title(f"(b) io_ns from microbench.json ({WSL_TAG})", loc="left")
        else:
            ax_io.text(0.5, 0.5, "io_ns: not in the inputs\n(the timing chains are per-episode files;\n"
                                 "pass --bench-csv with an 'io' series)", ha="center", va="center", color=INK2, fontsize=7, transform=ax_io.transAxes)
            ax_io.set_title(f"(b) io_ns -- pipe round trip ({WSL_TAG})", loc="left")
    ax_io.set_xlabel("io_ns (log)")
    # cyclictest environment baseline
    mb = run.get("microbench") or {}
    ct = mb.get("cyclictest") if isinstance(mb, dict) else None
    if isinstance(ct, dict) and ct:
        keys = [k for k in ("p50_us", "p99_us", "p999_us", "max_us") if ct.get(k) is not None]
        vals = [float(ct[k]) for k in keys]
        ax_ct.bar(range(len(keys)), vals, color=BAR_GRAY, width=0.5)
        ax_ct.set_xticks(range(len(keys)))
        ax_ct.set_xticklabels([k.replace("_us", "") for k in keys])
        for i, v in enumerate(vals):
            ax_ct.text(i, v, f"{v:.3g} us", ha="center", va="bottom", fontsize=6, color=INK2)
        ax_ct.set_ylabel("scheduling latency, us")
        ax_ct.set_title(f"(c) cyclictest baseline ({WSL_TAG})", loc="left")
    else:
        ax_ct.text(0.5, 0.5, "cyclictest environment baseline:\nnot measured on this host\n(WSL2 guest; add a 'cyclictest' object\nto microbench.json to plot it)",
                   ha="center", va="center", color=INK2, fontsize=7, transform=ax_ct.transAxes)
        ax_ct.set_xticks([])
        ax_ct.set_yticks([])
        ax_ct.set_title(f"(c) cyclictest baseline ({WSL_TAG})", loc="left")
    env = (mb.get("environment") if isinstance(mb, dict) else None) or WSL2_LABEL
    fig.suptitle(f"F5 -- what the fuse costs: decide_ns and io_ns ({WSL_TAG})", x=0.02, ha="left", fontsize=9.5)
    cap = [f"F5. {WSL2_LABEL}.",
           f"microbench environment: {env}.",
           "The IPC round trip (io_ns, tens of us) is larger than the verdict itself; both are reported and neither is"
           " hidden in the other. Percentiles are bucket upper bounds of the HdrHistogram dump; the arena figure and the"
           " 'allocations 0' line come from `lictor bench`."]
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F5", plt)


# ------------------------------------------------------------------------ F6


def _contingency_cell(ax, x0, y0, w, h, title: str, table, b, c, p, diff, lo, hi, n, note: str):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=SURFACE, edgecolor=GRID, linewidth=0.8))
    ax.text(x0 + 0.01, y0 + h - 0.015, title, fontsize=6.5, color=INK, va="top", ha="left", weight="bold")
    cw, ch = w * 0.22, h * 0.22
    gx, gy = x0 + w * 0.30, y0 + h * 0.10
    labels_top = ["arm ok", "arm fail"]
    labels_left = ["base ok", "base fail"]
    if table is not None:
        both, base_only, arm_only = table
        neither = n - both - base_only - arm_only
        cells = [[both, base_only], [arm_only, neither]]
    else:
        cells = [[None, b], [c, None]]
    for i in range(2):
        for j in range(2):
            v = cells[i][j]
            discordant = (i != j)
            fc = "#fbe9e2" if discordant else "#eef4fb"
            ax.add_patch(Rectangle((gx + j * (cw + 0.006), gy + (1 - i) * (ch + 0.006)), cw, ch, facecolor=fc, edgecolor="none"))
            ax.text(gx + j * (cw + 0.006) + cw / 2, gy + (1 - i) * (ch + 0.006) + ch / 2, "?" if v is None else str(v),
                    ha="center", va="center", fontsize=7, color=INK)
    for j, lab in enumerate(labels_top):
        ax.text(gx + j * (cw + 0.006) + cw / 2, gy + 2 * ch + 0.02, lab, ha="center", va="bottom", fontsize=5.5, color=INK2)
    for i, lab in enumerate(labels_left):
        ax.text(gx - 0.01, gy + (1 - i) * (ch + 0.006) + ch / 2, lab, ha="right", va="center", fontsize=5.5, color=INK2)
    txt = f"n={n}  b={b} c={c}\nMcNemar exact p={p:.3g}" if p is not None else f"n={n}  b={b} c={c}"
    if diff is not None:
        txt += f"\ndelta={diff:+.3f}"
        if lo is not None and hi is not None:
            txt += f" [{lo:+.3f}, {hi:+.3f}]"
    if note:
        txt += f"\n{note}"
    ax.text(x0 + w * 0.76, y0 + h * 0.55, txt, fontsize=5.6, color=INK2, va="center", ha="left")


def fig_f6(run: dict, out_dir: Path, plt) -> Path:
    arms = [r for r in sorted(run["arms"], key=analyze.arm_sort_key)
            if r.get("mcnemar_b") is not None and r.get("meta_role") not in ("baseline", "calibration")]
    arms = arms[:14]
    by_id = {str(r["arm_id"]): r for r in run["arms"]}
    rows = max(1, len(arms))
    fig_h = 1.15 * rows + 1.8
    # no explicit bottom: a GridSpec with its own bottom would ignore the caption's subplots_adjust
    fig, ax = plt.subplots(figsize=(10.5, fig_h), gridspec_kw={"left": 0.02, "right": 0.98, "top": 0.965})
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    cell_h = 1.0 / rows
    for i, r in enumerate(arms):
        y0 = 1.0 - (i + 1) * cell_h
        n = int(r.get("n") or 0)
        k = analyze._count(r.get("success_rate"), n)
        # vs obs-d0
        base_succ = r.get("n_succ_baseline")
        table = stats.orient_contingency(n, k, base_succ, int(r["mcnemar_b"]), int(r["mcnemar_c"])) if isinstance(base_succ, int) and k is not None else None
        pilot = " (pilot)" if r.get("pilot") else ""
        _contingency_cell(ax, 0.01, y0, 0.48, cell_h * 0.94, f"{r['arm_id']}{pilot} vs obs-d0 (fuse cost)", table,
                          r.get("mcnemar_b"), r.get("mcnemar_c"), r.get("mcnemar_p"), r.get("delta_vs_baseline"),
                          r.get("delta_vs_baseline_lo"), r.get("delta_vs_baseline_hi"), n,
                          "" if table is not None else "full 2x2 needs n_succ_baseline (curve json)")
        lc = r.get("latency_control_arm")
        if lc and r.get("mcnemar_b_lc") is not None:
            lc_row = by_id.get(str(lc))
            lc_succ = analyze._count(lc_row.get("success_rate"), lc_row.get("n")) if lc_row and lc_row.get("n") == n else None
            table_lc = stats.orient_contingency(n, k, lc_succ, int(r["mcnemar_b_lc"]), int(r["mcnemar_c_lc"])) if isinstance(lc_succ, int) and k is not None else None
            _contingency_cell(ax, 0.51, y0, 0.48, cell_h * 0.94, f"{r['arm_id']}{pilot} vs {lc} (latency-only control)", table_lc,
                              r.get("mcnemar_b_lc"), r.get("mcnemar_c_lc"), r.get("mcnemar_p_lc"), r.get("delta_vs_latency_control"),
                              r.get("delta_vs_latency_control_lo"), r.get("delta_vs_latency_control_hi"), n, "")
        elif r.get("meta_role") == "latency_control":
            ax.text(0.75, y0 + cell_h * 0.47, f"this arm IS the latency-only control at d={r.get('meta_d')}: its delta vs obs-d0 is the pure latency cost",
                    ha="center", va="center", fontsize=6.5, color=MUTED)
        else:
            ax.text(0.75, y0 + cell_h * 0.47, "no latency-control arm in this run for this d", ha="center", va="center", fontsize=6.5, color=MUTED)
    fig.suptitle("F6 -- paired success deltas: exact McNemar contingency per arm vs obs-d0 and vs obs-d{d}", x=0.02, ha="left", fontsize=9.5)
    cap = ("F6. Rows = arms; b/c are the discordant pairs exactly as `lictor curve` wrote them (paired by episode index,"
           " same reset state and per-chunk noise); delta = success(arm) - success(control) with its paired-bootstrap 95 % CI"
           " (10 000 resamples). Pilot rows (n < 100) are labelled and never headline. 'both fail' includes missing"
           " indices counted as failures under --partial.")
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F6", plt)


# ------------------------------------------------------------------------ F7


def fig_f7(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.4
    fig, ax = plt.subplots(figsize=(6.4, fig_h), gridspec_kw={"left": 0.11, "right": 0.97, "top": 0.9})
    pts: dict[float, dict] = {}
    for c in run.get("calibrations", []):
        a = analyze.alpha_value(c.get("alpha_num"), c.get("alpha_den"))
        if a is not None and c.get("holdout_fpr") is not None:
            pts[a] = {"kn": c["holdout_fpr"], "k1": c.get("holdout_fpr_k1"), "n": c.get("n_holdout"),
                      "degenerate": c.get("tau") is None or (isinstance(c.get("tau"), float) and math.isinf(c["tau"])),
                      "split": c.get("split"), "kn_spec": c.get("kn")}
    if not pts:
        for s in run["sweep"]:
            if s.get("tau_source") == "artefact" and s.get("holdout_fpr") is not None:
                a = _alpha_of(s)
                pts.setdefault(a, {"kn": s["holdout_fpr"], "k1": s.get("holdout_fpr_k1"), "n": None,
                                   "degenerate": s.get("tau") is None, "split": None, "kn_spec": [s.get("k"), s.get("n")]})
    xs = sorted(pts)
    if xs:
        lim = max(0.25, max(xs) * 1.25, max(max(pts[a]["kn"], pts[a]["k1"] or 0) for a in xs) * 1.25)
        ax.plot([0, lim], [0, lim], color=INK2, linewidth=0.8, label="diagonal: empirical = nominal")
        for a in xs:
            p = pts[a]
            n_h = p["n"]
            for key, color, marker, label in (("kn", BLUE, "o", "K-of-N (conservative upper bound)"),
                                              ("k1", ORANGE, "^", "K = 1 (the rate alpha bounds)")):
                y = p.get(key)
                if y is None:
                    continue
                if n_h:
                    lo, hi = stats.clopper_pearson(int(round(y * n_h)), int(n_h))
                    ax.errorbar([a], [y], yerr=[[max(0, y - lo)], [max(0, hi - y)]], fmt="none", ecolor=color, elinewidth=0.8, capsize=1.5)
                ax.plot([a], [y], **_marker_kwargs(color, bool(p["degenerate"]), marker), label=label if a == xs[0] else None)
            _n_label(ax, a, max(p["kn"], p["k1"] or 0), n_h if n_h else "holdout n/a", False)
            if p["degenerate"]:
                ax.annotate("degenerate (k > n_calib)", (a, p["kn"]), textcoords="offset points", xytext=(0, -10), fontsize=5.5, color=RED, ha="center")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.legend(loc="upper left")
        kn_spec = next((pts[a]["kn_spec"] for a in xs if pts[a]["kn_spec"]), None)
        split = next((pts[a]["split"] for a in xs if pts[a]["split"]), None)
    else:
        ax.text(0.5, 0.5, "no calibration artefacts / artefact sweep points found", ha="center", va="center", color=INK2, transform=ax.transAxes)
        kn_spec, split = None, None
    ax.set_xlabel("nominal alpha")
    ax.set_ylabel("empirical held-out episode-level firing rate (95 % CP CI)")
    fig.suptitle("F7 -- split-conformal validity: nominal alpha vs held-out FPR", x=0.02, ha="left", fontsize=9.5)
    cap = (f"F7. Held-out subset never touched tau. K-of-N = {kn_spec}; split = {split or 'n/a'}: under the default 2-way split"
           " the bound is approximate (center/scale fitted on the episodes tau was taken from), exact at K = 1 under --split 3."
           " A point far above the diagonal means exchangeability with the calibration pool is broken and the guarantee"
           " does not transfer; we say so rather than tune it away. Filled = data, hollow 'x' = degenerate alpha.")
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F7", plt)


# ------------------------------------------------------------------------ F8


def fig_f8(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.2
    fig, ax = plt.subplots(figsize=(7.2, fig_h), gridspec_kw={"left": 0.1, "right": 0.97, "top": 0.9})
    sweep = [s for s in run["sweep"] if s.get("detector") == HEADLINE_DETECTOR]
    alphas = sorted({(s["alpha_num"], s["alpha_den"]) for s in sweep}, key=lambda a: a[0] / a[1])
    finite = [s["tau"] for s in sweep if s.get("tau") is not None]
    top = (max(finite) * 1.15) if finite else 1.0
    multi = False
    for i, (num, den) in enumerate(alphas):
        rows = sorted((s for s in sweep if (s["alpha_num"], s["alpha_den"]) == (num, den)), key=lambda s: s.get("n_calib") or 0)
        color = SLOTS[i % len(SLOTS)]
        if len(rows) > 1:
            multi = True
        xs = [s["n_calib"] for s in rows if s.get("tau") is not None]
        ys = [s["tau"] for s in rows if s.get("tau") is not None]
        ax.plot(xs, ys, color=color, marker="o", label=f"{_fmt_alpha(num, den)}")
        for s in rows:
            if s.get("tau") is None:
                ax.plot([s["n_calib"]], [top], **_marker_kwargs(color, True, "x"))
                ax.annotate("tau=+inf (k > n)", (s["n_calib"], top), textcoords="offset points", xytext=(0, 5), fontsize=5.5, color=RED, ha="center")
            else:
                src = "" if s.get("tau_source") == "artefact" else " fit"
                _n_label(ax, s["n_calib"], s["tau"], f"{s['n_calib']}{src}", False)
    ax.set_xlabel("n_calib (per-episode scores tau was taken over)")
    ax.set_ylabel("tau (split-conformal quantile of max_t s_t)")
    ax.legend(loc="upper right")
    if not multi:
        ax.set_title("single n_calib per alpha: the economy ablation needs sweep rows at n_calib in {25, 50, 100, all}", loc="left", fontsize=7, color=INK2)
    fig.suptitle("F8 -- calibration economy: threshold stability vs n_calib", x=0.02, ha="left", fontsize=9.5)
    cap = ("F8. k = ceil((n_calib + 1)(1 - alpha)) in integer arithmetic; when k > n_calib the quantile does not exist and"
           " tau = +inf (never fires) -- shown as a hollow x at the top, a fact about the calibration size, not a data point."
           " Points labelled 'fit' are full-sample taus the sweep fitted itself; unlabelled points reuse the committed"
           " calibration.<alpha>.json artefact.")
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F8", plt)


# ------------------------------------------------------------------------ F9


def fig_f9(run: dict, out_dir: Path, plt) -> Path:
    fig_h = 5.6
    fig, (ax, ax_t) = plt.subplots(1, 2, figsize=(11.0, fig_h), gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.15, "left": 0.07, "right": 0.98, "top": 0.88})
    inj = [r for r in run["arms"] if str(r.get("arm_id")).startswith("inj-")]
    inj.sort(key=lambda r: (0 if "obs" in str(r["arm_id"]) else 1, str(r["arm_id"])))
    if inj:
        xs = list(range(len(inj)))
        vals = [int(r.get("violations_reached_env") or 0) for r in inj]
        colors = [ORANGE if "obs" in str(r["arm_id"]) else BLUE for r in inj]
        ax.bar(xs, vals, color=colors, width=0.5, zorder=3)
        for x, v, r in zip(xs, vals, inj):
            ax.text(x, v, f"{v}\nn={r.get('n')}" + (" pilot" if r.get("pilot") else ""), ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{r['arm_id']}\n({'fuse off, observe' if 'obs' in str(r['arm_id']) else 'fuse on, enforce'})" for r in inj], fontsize=7)
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    else:
        ax.text(0.5, 0.5, "no inj-* arms in this run", ha="center", va="center", color=INK2, transform=ax.transAxes)
    ax.set_ylabel("violations_reached_env (Tier-0 violations not substituted, summed over the arm)")
    ax.set_title("(a) enforcement under action_spike(p=0.02, mag=180)", loc="left")
    ax_t.axis("off")
    log = run.get("tamper_log") or TAMPER_REFERENCE
    ax_t.text(0.0, 1.0, log, family="monospace", fontsize=6.2, va="top", ha="left", color=INK, transform=ax_t.transAxes)
    ax_t.set_title("(b) tamper demo: the receipt refuses edits without the signing key", loc="left")
    fig.suptitle("F9 -- enforcement pillar: injected spikes reaching the environment, and the tamper demo", x=0.02, ha="left", fontsize=9.5)
    cap = ("F9. The same injected action spikes (Philox(episode_seed ^ 0xFA17), declared in fault_injection) are scored"
           " in observe mode and substituted in enforce mode; the count comes from the signed receipts via `lictor curve`."
           " (b): edit one byte of the signed body and the signature fails; edit one tick in the 300-line ticks file and"
           " `verify --ticks` reports the break; re-chain it and the head no longer matches. A receipt defends against"
           " edits by anyone without the signing key; it does not defend against the key-holder (trust model).")
    _caption(fig, cap, fig_h)
    return _save(fig, out_dir, "F9", plt)


RENDERERS = {"F1": fig_f1, "F2": fig_f2, "F3": fig_f3, "F4": fig_f4, "F5": fig_f5,
             "F6": fig_f6, "F7": fig_f7, "F8": fig_f8, "F9": fig_f9}


def render_all(run: dict, out_dir, only=None) -> list[Path]:
    plt = setup_matplotlib()
    out_dir = Path(out_dir)
    paths = []
    for key in FIG_FILES:
        if only and key not in only:
            continue
        paths.append(RENDERERS[key](run, out_dir, plt))
    return paths


# --------------------------------------------------------------------- fixture

_BASE_N = 500
_BASE_SUCC = 327          # 65.4 % of 500: the published pc_success
_PILOT_N = 60
_PILOT_SUCC = 39

# arm_id: (n, s_base, s_arm, b, c, lc_b, lc_c, flagged, averted, false, interv, tick_frac, esc,
#          lead_mean, lead_p50, lead_p10, aucpdt, roc_auc, bacc, violations, tce_valid_frac, lat_p50, lat_p99, lat_max)
FIXTURE_ARMS = [
    ("obs-d0",           (_BASE_N, _BASE_SUCC, 327, 0, 0, None, None, 0.549, 0.000, 0.052, 0.000, 0.000, 0.023, 39.8, 36.0, 3.0, 0.49, 0.71, 0.75, 2, 0.933, 2130, 8720, 412000)),
    ("obs-d1",           (_BASE_N, _BASE_SUCC, 318, 15, 6, None, None, 0.538, 0.000, 0.055, 0.000, 0.000, 0.021, 38.1, 35.0, 2.0, 0.47, 0.70, 0.74, 3, 0.933, 2140, 8650, 398000)),
    ("obs-d2",           (_BASE_N, _BASE_SUCC, 305, 30, 8, None, None, 0.531, 0.000, 0.061, 0.000, 0.000, 0.025, 36.4, 33.0, 1.0, 0.45, 0.69, 0.73, 3, 0.800, 2120, 8810, 405000)),
    ("obs-d3",           (_BASE_N, _BASE_SUCC, 290, 45, 8, None, None, 0.520, 0.000, 0.064, 0.000, 0.000, 0.027, 34.9, 31.0, 0.0, 0.43, 0.68, 0.73, 4, 0.667, 2110, 8590, 391000)),
    ("obs-d5",           (_BASE_N, _BASE_SUCC, 252, 85, 10, None, None, 0.503, 0.000, 0.073, 0.000, 0.000, 0.031, 31.2, 28.0, -3.0, 0.39, 0.66, 0.71, 5, 0.400, 2150, 8930, 420000)),
    ("obs-d8",           (_PILOT_N, _PILOT_SUCC, 25, 16, 2, None, None, 0.476, 0.000, 0.077, 0.000, 0.000, 0.033, 27.5, 24.0, -6.0, 0.35, 0.63, 0.69, 1, 0.000, 2160, 9010, 388000)),
    ("obs-d2-async",     (_BASE_N, _BASE_SUCC, 321, 12, 6, None, None, 0.543, 0.000, 0.055, 0.000, 0.000, 0.021, 38.8, 35.0, 2.0, 0.48, 0.70, 0.74, 3, 0.933, 2130, 8700, 401000)),
    ("obs-d5-async",     (_BASE_N, _BASE_SUCC, 312, 22, 7, None, None, 0.526, 0.000, 0.058, 0.000, 0.000, 0.023, 36.1, 33.0, 0.0, 0.45, 0.69, 0.73, 4, 0.933, 2140, 8760, 409000)),
    ("t0-d0",            (_BASE_N, _BASE_SUCC, 324, 6, 3, None, None, 0.116, 0.029, 0.021, 0.042, 0.0021, 0.012, 22.0, 18.0, -4.0, 0.21, 0.58, 0.55, 0, 0.933, 2050, 7900, 376000)),
    ("t01-a01-d0",       (_BASE_N, _BASE_SUCC, 330, 12, 15, None, None, 0.451, 0.098, 0.018, 0.128, 0.017, 0.049, 44.7, 41.0, 6.0, 0.41, 0.71, 0.72, 0, 0.933, 2130, 8720, 402000)),
    ("t01-a05-d0",       (_BASE_N, _BASE_SUCC, 336, 14, 23, None, None, 0.612, 0.150, 0.061, 0.192, 0.031, 0.083, 41.2, 37.0, 4.0, 0.51, 0.71, 0.77, 0, 0.933, 2130, 8720, 412000)),
    ("t01-a10-d0",       (_BASE_N, _BASE_SUCC, 333, 21, 27, None, None, 0.705, 0.179, 0.113, 0.264, 0.044, 0.116, 47.9, 44.0, 8.0, 0.56, 0.71, 0.80, 0, 0.933, 2130, 8720, 412000)),
    ("t01-a20-d0",       (_BASE_N, _BASE_SUCC, 322, 38, 33, None, None, 0.792, 0.220, 0.211, 0.386, 0.071, 0.171, 55.3, 52.0, 12.0, 0.61, 0.71, 0.79, 0, 0.933, 2130, 8720, 412000)),
    ("t01-a05-d1",       (_BASE_N, _BASE_SUCC, 327, 17, 17, 10, 19, 0.595, 0.139, 0.064, 0.200, 0.033, 0.087, 39.0, 35.0, 2.0, 0.49, 0.70, 0.76, 0, 0.933, 2140, 8650, 398000)),
    ("t01-a05-d2",       (_BASE_N, _BASE_SUCC, 315, 30, 18, 12, 22, 0.578, 0.121, 0.067, 0.212, 0.036, 0.094, 36.8, 33.0, 0.0, 0.46, 0.69, 0.75, 0, 0.800, 2120, 8810, 405000)),
    ("t01-a05-d3",       (_BASE_N, _BASE_SUCC, 300, 45, 18, 14, 24, 0.549, 0.098, 0.073, 0.226, 0.040, 0.101, 34.1, 30.0, -2.0, 0.43, 0.68, 0.74, 0, 0.667, 2110, 8590, 391000)),
    ("t01-a05-d5",       (_BASE_N, _BASE_SUCC, 262, 80, 15, 15, 25, 0.497, 0.069, 0.086, 0.254, 0.047, 0.118, 29.7, 26.0, -6.0, 0.37, 0.66, 0.71, 0, 0.400, 2150, 8930, 420000)),
    ("t01-a05-d8",       (_PILOT_N, _PILOT_SUCC, 27, 15, 3, 4, 6, 0.429, 0.048, 0.103, 0.283, 0.055, 0.133, 24.9, 21.0, -9.0, 0.31, 0.62, 0.66, 0, 0.000, 2160, 9010, 388000)),
    ("t01-a05-d2-async", (_BASE_N, _BASE_SUCC, 326, 15, 14, 8, 13, 0.601, 0.145, 0.064, 0.198, 0.032, 0.085, 40.3, 36.0, 3.0, 0.50, 0.71, 0.77, 0, 0.933, 2130, 8700, 401000)),
    ("t01-a05-d5-async", (_BASE_N, _BASE_SUCC, 318, 25, 16, 9, 15, 0.584, 0.127, 0.067, 0.216, 0.038, 0.096, 37.5, 34.0, 1.0, 0.47, 0.70, 0.76, 0, 0.933, 2140, 8760, 409000)),
    ("t01-a05-d0-oracle", (100, 65, 71, 4, 10, None, None, 0.629, 0.171, 0.062, 0.190, 0.030, 0.080, 41.0, 37.0, 4.0, 0.51, 0.71, 0.78, 0, 0.933, 2130, 8720, 412000)),
    ("inj-obs-d0",       (_BASE_N, _BASE_SUCC, 301, 40, 14, None, None, 0.618, 0.000, 0.073, 0.000, 0.000, 0.031, 40.1, 36.0, 3.0, 0.50, 0.70, 0.77, 2874, 0.933, 2130, 8720, 415000)),
    ("inj-t0-d0",        (_BASE_N, _BASE_SUCC, 312, 28, 13, None, None, 0.850, 0.052, 0.107, 0.972, 0.021, 0.074, 30.2, 27.0, -1.0, 0.44, 0.66, 0.87, 0, 0.933, 2140, 8790, 421000)),
]

# per detector: (k, n, tier0, roc_auc, {alpha_pct: (tpr, fpr, lead_mean, lead_p50, lead_p10, aucpdt, fire_frac)})
FIXTURE_SWEEP = {
    "t0":       (1, 1, True,  0.58, {1: (0.104, 0.012, 22.0, 18.0, -4.0, 0.21, 0.02), 2: (0.110, 0.015, 22.0, 18.0, -4.0, 0.21, 0.02), 5: (0.116, 0.021, 22.0, 18.0, -4.0, 0.21, 0.02), 10: (0.116, 0.021, 22.0, 18.0, -4.0, 0.21, 0.02), 20: (0.116, 0.021, 22.0, 18.0, -4.0, 0.21, 0.02)}),
    "t1_tce":   (3, 5, False, 0.66, {1: (0.312, 0.009, 31.0, 27.0, -2.0, 0.30, 0.03), 2: (0.399, 0.018, 34.2, 30.0, 0.0, 0.35, 0.04), 5: (0.520, 0.049, 37.0, 33.0, 2.0, 0.42, 0.07), 10: (0.618, 0.104, 41.5, 38.0, 5.0, 0.48, 0.12), 20: (0.717, 0.199, 47.0, 43.0, 9.0, 0.54, 0.22)}),
    "t1_stall": (3, 5, False, 0.69, {1: (0.358, 0.012, 27.0, 24.0, -5.0, 0.28, 0.03), 2: (0.428, 0.021, 29.0, 26.0, -3.0, 0.32, 0.04), 5: (0.549, 0.052, 32.1, 29.0, -1.0, 0.39, 0.08), 10: (0.653, 0.107, 36.0, 33.0, 2.0, 0.45, 0.13), 20: (0.752, 0.208, 41.2, 38.0, 6.0, 0.52, 0.23)}),
    "t1_full":  (3, 5, False, 0.72, {1: (0.399, 0.009, 36.0, 32.0, 1.0, 0.36, 0.03), 2: (0.480, 0.018, 38.5, 35.0, 2.0, 0.41, 0.04), 5: (0.601, 0.055, 41.0, 37.0, 4.0, 0.50, 0.08), 10: (0.699, 0.110, 46.8, 43.0, 7.0, 0.56, 0.13), 20: (0.786, 0.205, 53.9, 50.0, 11.0, 0.61, 0.23)}),
    "t01":      (3, 5, True,  0.71, {1: (0.451, 0.018, 44.7, 41.0, 6.0, 0.41, 0.05), 2: (0.526, 0.031, 40.0, 36.0, 3.0, 0.46, 0.06), 5: (0.601, 0.061, 41.2, 37.0, 4.0, 0.51, 0.08), 10: (0.705, 0.113, 47.9, 44.0, 8.0, 0.56, 0.14), 20: (0.792, 0.211, 55.3, 52.0, 12.0, 0.61, 0.24)}),
    "t01_and":  (3, 5, True,  0.68, {1: (0.376, 0.012, 33.0, 30.0, -1.0, 0.33, 0.04), 2: (0.445, 0.021, 35.7, 32.0, 0.0, 0.38, 0.05), 5: (0.549, 0.046, 38.4, 34.0, 2.0, 0.45, 0.07), 10: (0.647, 0.089, 43.0, 39.0, 5.0, 0.51, 0.11), 20: (0.740, 0.168, 49.7, 46.0, 9.0, 0.57, 0.19)}),
}
FIXTURE_TAU = {1: 4.4123, 2: 4.0512, 5: 3.7142857142857144, 10: 3.2871, 20: 2.7440}
FIXTURE_HOLDOUT = {1: (0.0, 1 / 59), 2: (1 / 59, 2 / 59), 5: (3 / 59, 4 / 59), 10: (5 / 59, 7 / 59), 20: (10 / 59, 13 / 59)}
FIXTURE_ECONOMY = {  # n_calib -> {alpha_pct: tau or None (degenerate)}
    25:  {1: None, 2: None, 5: 4.9021, 10: 3.9104, 20: 3.0217},
    50:  {1: None, 2: 5.2310, 5: 4.1188, 10: 3.5122, 20: 2.8630},
    100: {1: 4.9770, 2: 4.3016, 5: 3.8402, 10: 3.3350, 20: 2.7712},
}
FIXTURE_N_FAIL, FIXTURE_N_SUCC = 173, 327
FIXTURE_GATE = ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak"]
FIXTURE_ALPHAS = [1, 2, 5, 10, 20]
FIXTURE_MICROBENCH = {
    "schema": "lictor-microbench/fixture",
    "environment": WSL2_LABEL,
    "import_lerobot_policies_s": 14.2,
    "policy_load_s": 8.3,
    "s_per_episode_gpu_b1": 153.9,
    "s_per_episode_cpu": 2412.0,
    "ms_per_env_chunk_b32": 123.0,
    "chunk_inference_mean_s": 3.46,
    "episodes_per_min_1_worker": 0.39,
    "episodes_per_min_2_workers": 0.41,
    "h15": "PASS",
    "determinism_seeds_0_2": "PASS",
    "observe_identity_seeds_0_2": "PASS",
    "bench_p99_us": 8.7,
    "bench_allocations": 0,
    "cyclictest": None,
}


def _arm_row(arm_id: str, spec: tuple) -> tuple[dict, dict]:
    (n, s_base, s_arm, b, c, lc_b, lc_c, flagged, averted, false, interv, tick_frac, esc,
     lead_mean, lead_p50, lead_p10, aucpdt, roc_auc, bacc_v, viol, tce, lat50, lat99, latmax) = spec
    meta = analyze.parse_arm_id(arm_id)
    n_fail_b, n_succ_b = n - s_base, s_base
    succ_lo, succ_hi = stats.clopper_pearson(s_arm, n)
    table = stats.orient_contingency(n, s_arm, s_base, b, c)
    assert table is not None, arm_id
    both, base_only, arm_only = table
    base_arr, arm_arr = stats.pairs_from_contingency(n, both, base_only, arm_only)
    diff, lo, hi = stats.paired_bootstrap_diff(base_arr, arm_arr)
    flagged_k = int(round(flagged * n_fail_b))
    false_k = int(round(false * n_succ_b))
    fl_lo, fl_hi = stats.clopper_pearson(flagged_k, n_fail_b)
    ft_lo, ft_hi = stats.clopper_pearson(false_k, n_succ_b)
    d = meta["d"] or 0
    lc_arm = None
    lc = {"delta": None, "lo": None, "hi": None, "p": None, "b": None, "c": None}
    if meta["role"] in ("enforce", "oracle", "ackonly") and d > 0:
        lc_arm = f"obs-d{d}" + ("-async" if meta["exec"] == "async" else "")
    elif meta["role"] in ("enforce", "oracle", "ackonly", "injection"):
        lc_arm = "obs-d0"
    row = {
        "pilot": 1 if n < 100 else 0, "arm_id": arm_id, "n": n, "n_missing": 0, "partial": False,
        "delay_steps": d, "exec_mode": meta["exec"],
        "alpha": f"{meta['alpha_pct']}/100" if meta["alpha_pct"] is not None else None,
        "detector": meta["detector_family"],
        "success_rate": s_arm / n, "success_lo": succ_lo, "success_hi": succ_hi,
        "delta_vs_baseline": diff, "delta_vs_baseline_lo": lo, "delta_vs_baseline_hi": hi,
        "mcnemar_p": stats.mcnemar_exact(b, c), "mcnemar_b": b, "mcnemar_c": c,
        "latency_control_arm": lc_arm,
        "averted_rate": int(round(averted * n_fail_b)) / n_fail_b,
        "flagged_rate": flagged_k / n_fail_b, "flagged_lo": fl_lo, "flagged_hi": fl_hi,
        "false_trip_rate": false_k / n_succ_b, "false_trip_lo": ft_lo, "false_trip_hi": ft_hi,
        "intervention_rate": interv, "intervention_tick_frac": tick_frac, "escalation_rate": esc,
        "lead_mean": lead_mean, "lead_p50": lead_p50, "lead_p10": lead_p10, "aucpdt": aucpdt,
        "roc_auc": roc_auc, "bacc": bacc_v, "violations_reached_env": viol, "tce_valid_frac": tce,
        "latency_p50_ns": lat50, "latency_p99_ns": lat99, "latency_max_ns": latmax,
        "latency_label": WSL2_LABEL, "ledger_chain_ok": True, "pair_mismatches": 0,
        "cross_run_mismatches": 0,
        "receipt_pubkey": "2543b92ff1095511476adc8369db6ddc933665a11978dda1404ee1066ca9559d",
        "_lc_bc": (lc_b, lc_c), "_lc_delta": lc,
    }
    curve = {
        "schema": "lictor-curve/v1", "arm_id": arm_id, "run_id": "fixture", "baseline_arm_id": "obs-d0",
        "latency_control_arm_id": lc_arm, "ledger_chain_ok": True, "n_episodes": n, "n_declared": n,
        "n_present": n, "n_missing": 0, "partial": False, "small_n": n < 100, "pair_mismatches": 0,
        "cross_run_mismatches": 0, "seed_overlap_with_calibration": 0,
        "receipt_pubkey": row["receipt_pubkey"], "run_json_sha256": "0" * 64, "arm_config_digest": "0" * 64,
        "eps_prog": 0.02, "curve_pubkey": row["receipt_pubkey"], "n": n,
        "n_fail_baseline": n_fail_b, "n_succ_baseline": n_succ_b, "success_rate": s_arm / n,
        "success_ci": [succ_lo, succ_hi], "flagged_ci": [fl_lo, fl_hi], "false_trip_ci": [ft_lo, ft_hi],
        "delta_vs_baseline": {"diff": diff, "lo": lo, "hi": hi, "mcnemar_p": row["mcnemar_p"], "mcnemar_b": b, "mcnemar_c": c},
        "delta_vs_latency_control": None, "tce_valid_frac": tce, "latency_label": WSL2_LABEL,
    }
    return row, curve


def fixture_run() -> dict:
    """The embedded synthetic sample: the same dict `analyze.load_run` returns."""
    rows, curve = [], {}
    for arm_id, spec in FIXTURE_ARMS:
        r, c = _arm_row(arm_id, spec)
        rows.append(r)
        curve[arm_id] = c
    by_id = {r["arm_id"]: r for r in rows}
    for r in rows:
        lc_b, lc_c = r.pop("_lc_bc")
        r.pop("_lc_delta")
        lc_arm = r["latency_control_arm"]
        if lc_arm is None:
            for k in ("delta_vs_latency_control", "delta_vs_latency_control_lo", "delta_vs_latency_control_hi",
                      "mcnemar_p_lc", "mcnemar_b_lc", "mcnemar_c_lc"):
                r[k] = None
            continue
        if lc_b is None:  # d = 0: the latency control IS the baseline
            lc_b, lc_c = r["mcnemar_b"], r["mcnemar_c"]
        n = r["n"]
        s_arm = int(round(r["success_rate"] * n))
        s_lc = int(round(by_id[lc_arm]["success_rate"] * n))
        table = stats.orient_contingency(n, s_arm, s_lc, lc_b, lc_c)
        assert table is not None, (r["arm_id"], lc_arm)
        base_arr, arm_arr = stats.pairs_from_contingency(n, *table)
        diff, lo, hi = stats.paired_bootstrap_diff(base_arr, arm_arr)
        r["delta_vs_latency_control"], r["delta_vs_latency_control_lo"], r["delta_vs_latency_control_hi"] = diff, lo, hi
        r["mcnemar_p_lc"], r["mcnemar_b_lc"], r["mcnemar_c_lc"] = stats.mcnemar_exact(lc_b, lc_c), lc_b, lc_c
        curve[r["arm_id"]]["delta_vs_latency_control"] = {"diff": diff, "lo": lo, "hi": hi, "mcnemar_p": r["mcnemar_p_lc"], "mcnemar_b": lc_b, "mcnemar_c": lc_c}
    for r in rows:
        meta = analyze.parse_arm_id(r["arm_id"])
        r.update({f"meta_{k}": v for k, v in meta.items()})
        r["n_fail_baseline"] = curve[r["arm_id"]]["n_fail_baseline"]
        r["n_succ_baseline"] = curve[r["arm_id"]]["n_succ_baseline"]
        r["small_n"] = r["n"] < 100
        r["eps_prog"] = 0.02
        r["_missing_columns"] = []
        r["_extra_columns"] = []
    sweep = []
    for det, (k, nn, tier0, auc, per_alpha) in FIXTURE_SWEEP.items():
        for a in FIXTURE_ALPHAS:
            tpr, fpr, lm, lp50, lp10, aucpdt_v, ff = per_alpha[a]
            tk = int(round(tpr * FIXTURE_N_FAIL))
            fk = int(round(fpr * FIXTURE_N_SUCC))
            tpr_v, fpr_v = tk / FIXTURE_N_FAIL, fk / FIXTURE_N_SUCC
            sweep.append({
                "alpha_num": a, "alpha_den": 100, "detector": det,
                "gate": FIXTURE_GATE if det != "t0" else [], "k": k, "n": nn, "tier0": tier0,
                "method": "binned", "tau": None if det == "t0" else FIXTURE_TAU[a],
                "tau_source": "artefact" if a in (5, 10, 20) else "fit", "n_calib": 137,
                "n_fail": FIXTURE_N_FAIL, "n_succ": FIXTURE_N_SUCC,
                "tpr": tpr_v, "tpr_ci": list(stats.clopper_pearson(tk, FIXTURE_N_FAIL)),
                "fpr": fpr_v, "fpr_ci": list(stats.clopper_pearson(fk, FIXTURE_N_SUCC)),
                "lead_mean": lm, "lead_p50": lp50, "lead_p10": lp10, "aucpdt": aucpdt_v, "roc_auc": auc,
                "bacc": stats.bacc(tpr_v, fpr_v), "fire_frac_mean": ff,
                "holdout_fpr": FIXTURE_HOLDOUT[a][0], "holdout_fpr_k1": FIXTURE_HOLDOUT[a][1], "eps_prog": 0.02,
            })
    for n_calib, taus in FIXTURE_ECONOMY.items():
        base = {s["alpha_num"]: s for s in sweep if s["detector"] == "t01" and s["n_calib"] == 137}
        for a in FIXTURE_ALPHAS:
            s = dict(base[a])
            tau = taus[a]
            s.update({"n_calib": n_calib, "tau": tau, "tau_source": "fit"})
            if tau is None:
                s.update({"tpr": 0.0, "tpr_ci": [0.0, stats.clopper_pearson(0, FIXTURE_N_FAIL)[1]],
                          "fpr": 0.0, "fpr_ci": [0.0, stats.clopper_pearson(0, FIXTURE_N_SUCC)[1]], "bacc": 0.5,
                          "lead_mean": None, "lead_p50": None, "lead_p10": None, "aucpdt": 0.0, "fire_frac_mean": 0.0})
            sweep.append(s)
    calibrations = [{
        "path": f"calibration.a{a:02d}.json", "alpha_num": a, "alpha_den": 100, "n_total": 196, "n_scale": 137,
        "n_calib": 137, "n_holdout": 59, "split": "2way", "tau": FIXTURE_TAU[a],
        "holdout_fpr": FIXTURE_HOLDOUT[a][0], "holdout_fpr_k1": FIXTURE_HOLDOUT[a][1], "kn": [3, 5],
        "method": "binned", "gate": FIXTURE_GATE, "notes": [],
    } for a in FIXTURE_ALPHAS]
    return {
        "run_dir": "<fixture>", "arms": rows, "curve": curve, "sweep": sweep,
        "microbench": dict(FIXTURE_MICROBENCH), "bench_hist": fixture_bench_hist(),
        "calibrations": calibrations, "score_bands": fixture_score_bands(),
        "leads": fixture_leads(), "tamper_log": None,
    }


def _log_buckets(lo_ns: float, hi_ns: float, per_decade: int = 12) -> list[float]:
    out = []
    v = lo_ns
    step = 10 ** (1.0 / per_decade)
    while v <= hi_ns * step:
        out.append(v)
        v *= step
    return out


def fixture_bench_hist() -> dict[str, list[tuple[float, int]]]:
    """Synthetic HdrHistogram-style bucket dumps (log-normal-ish, deterministic)."""
    rng = random.Random(20260830)
    edges = _log_buckets(100.0, 2e7)
    series = {"verdict": (7.6, 0.35, 0.002, 12.5), "tier0": (6.4, 0.30, 0.001, 11.0),
              "brake": (6.9, 0.32, 0.001, 11.5), "tier1": (6.6, 0.30, 0.001, 11.0), "io": (10.2, 0.40, 0.003, 13.5)}
    out = {}
    for name, (mu, sigma, p_tail, tail_mu) in series.items():
        counts = [0] * len(edges)
        for _ in range(20000):
            if rng.random() < p_tail:
                v = math.exp(rng.gauss(tail_mu, 0.5))
            else:
                v = math.exp(rng.gauss(mu, sigma))
            for i, e in enumerate(edges):
                if v <= e:
                    counts[i] += 1
                    break
            else:
                counts[-1] += 1
        out[name] = [(e, c) for e, c in zip(edges, counts) if c > 0]
    return out


def fixture_score_bands() -> list[dict]:
    rng = random.Random(20260831)
    rows = []
    for t in range(300):
        for group in ("success", "fail"):
            if t < 8:
                continue
            base = 0.25 + 0.15 * math.sin(t / 37.0)
            if group == "fail":
                base += max(0.0, (t - 120) / 60.0) ** 1.4
            q50 = base + rng.gauss(0, 0.03)
            iqr = 0.35 + (0.25 if group == "fail" else 0.0) * min(1.0, max(0.0, (t - 120) / 100.0))
            rows.append({"t": t, "group": group, "n": 327 if group == "success" else 173,
                         "q25": q50 - iqr / 2, "q50": q50, "q75": q50 + iqr / 2, "tau": FIXTURE_TAU[5]})
    return rows


def fixture_leads() -> list[dict]:
    rows = []
    for a in FIXTURE_ALPHAS:
        tpr = FIXTURE_SWEEP["t01"][4][a][0]
        for eps in ((0.01, 0.02, 0.05) if a == 5 else (0.02,)):
            rng = random.Random(20260830 + a * 7 + int(eps * 1000))
            shift = {0.01: -6.0, 0.02: 0.0, 0.05: 9.0}[eps]
            for ep in range(FIXTURE_N_FAIL):
                detected = rng.random() < tpr
                lead = None
                if detected:
                    if ep % 23 == 0:   # the t_fail = 0 artefact: no progress at all
                        lead = -int(abs(rng.gauss(40, 15)))
                    else:
                        lead = int(rng.gauss(38 + 4 * math.log(a) + shift, 22))
                rows.append({"alpha_num": a, "alpha_den": 100, "detector": "t01", "eps_prog": eps,
                             "episode_index": ep, "lead": lead})
    return rows


# ------------------------------------------------------------------- CSV inputs


def read_score_bands(path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({"t": int(r["t"]), "group": r["group"], "n": int(r["n"]),
                         "q25": float(r["q25"]), "q50": float(r["q50"]), "q75": float(r["q75"]),
                         "tau": analyze.parse_cell(r.get("tau"))})
    return rows


def read_leads(path) -> list[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            lead = analyze.parse_cell(r.get("lead"))
            rows.append({"alpha_num": int(r["alpha_num"]), "alpha_den": int(r["alpha_den"]),
                         "detector": r["detector"], "eps_prog": float(r.get("eps_prog") or 0.02),
                         "episode_index": int(r["episode_index"]), "lead": lead})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="lictor figures F1-F9 (matplotlib, SVG)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="results/<run> directory")
    src.add_argument("--from-fixture", action="store_true", help="render from the embedded synthetic sample")
    ap.add_argument("--out", required=True, help="output directory for the SVG files")
    ap.add_argument("--bench-csv", default=None, help="bucket dump of `lictor bench --csv`")
    ap.add_argument("--calib-dir", default=None, help="directory of calibration.<alpha>.json (default <run>)")
    ap.add_argument("--score-bands", default=None, help="score_bands.csv from `analyze.py scores` (F4)")
    ap.add_argument("--leads", default=None, help="per-episode leads CSV (F3)")
    ap.add_argument("--tamper-log", default=None, help="captured scripts/demo.sh output (F9)")
    ap.add_argument("--only", default=None, help="comma-separated subset, e.g. F1,F5")
    a = ap.parse_args(argv)
    if a.from_fixture:
        run = fixture_run()
    else:
        run = analyze.load_run(a.run, bench_csv=a.bench_csv, calib_dir=a.calib_dir)
    if a.score_bands:
        run["score_bands"] = read_score_bands(a.score_bands)
    if a.leads:
        run["leads"] = read_leads(a.leads)
    if a.tamper_log:
        with open(a.tamper_log, encoding="utf-8", errors="replace") as fh:
            run["tamper_log"] = "".join(ch if ord(ch) < 127 and (ch >= " " or ch == "\n") else "?" for ch in fh.read())[:4000]
    only = set(x.strip().upper() for x in a.only.split(",")) if a.only else None
    paths = render_all(run, a.out, only=only)
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
