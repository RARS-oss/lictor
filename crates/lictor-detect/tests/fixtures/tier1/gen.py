#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fixture generator for the lictor-detect tests (WP-2).

An INDEPENDENT reimplementation, in plain Python floats (IEEE-754 binary64 -- the
same arithmetic numpy performs on float64 scalars; numpy itself is deliberately not
imported so that the bare `python3` of the acceptance command works), of

  * the eight Tier-1 formulas of ARCHITECTURE 5.3 with the manifest-only
    `speed_peak` / `stall` definitions and the t_emit-derived chunk overlap,
  * the Tier-0 table of ARCHITECTURE 5.1 and the sequential leash projection,
  * the conformal runtime rule (standardise / aggregate / strict trip) and `bin_of`.

Every accumulation is an explicit left-to-right loop in the order the Rust code
uses (`lictor_core::fmath::dist` / `norm`: index ascending, row then coordinate),
so the fixtures are asserted to 1e-12 rather than to a loose tolerance.

Usage:
    python3 gen.py            # (re)write tier1/cases.json, tier0/cases.json, conformal/cases.json
    python3 gen.py --check    # regenerate in memory; exit 1 if any committed fixture differs

Conventions in the JSON: pretty-printed, keys sorted; a `null` aggregate `s`
means NEG_INFINITY and a `null` `tau` means +INFINITY (JSON has no infinities;
the wire encodes them the same way). Box limits in the tier0 configs are the
COMPILED (already margin-adjusted) values, as `FuseConfig` carries them.
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.dirname(HERE)
OUTPUTS = {
    "tier1": os.path.join(HERE, "cases.json"),
    "tier0": os.path.join(FIXTURES, "tier0", "cases.json"),
    "conformal": os.path.join(FIXTURES, "conformal", "cases.json"),
}

EPS = 1e-9
NFEAT = 12
MAX_EXT = 4
T_GRID = 100
PE_WINDOW = 20
TRAIL_W = 32
STALL_VREF_FRAC = 0.05
CHUNK_BOUNDARY_MASK = 0b0000_1001_1111
# 1 - 2^-36: the rounding guard of the Rust leash (see tier0.rs).
SHRINK = 1.0 - 1.0 / 68719476736.0

FEAT = ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak",
        "ext0", "ext1", "ext2", "ext3"]
F = {name: i for i, name in enumerate(FEAT)}
TRIPS = ["workspace", "speed", "accel", "jerk", "reach", "contact"]


# --------------------------------------------------------------------------- fmath

def fmin(a, b):
    return a if a < b else b


def fmax(a, b):
    return a if a > b else b


def clamp(x, lo, hi):
    return fmax(lo, fmin(x, hi))


def dist(a, b, n):
    s = 0.0
    for i in range(n):
        d = a[i] - b[i]
        s += d * d
    return math.sqrt(s)


def norm(v, n):
    s = 0.0
    for i in range(n):
        s += v[i] * v[i]
    return math.sqrt(s)


# --------------------------------------------------------------------------- chunks

def mk_chunk(seq, t_emit, rows, exec_steps=8):
    return {
        "seq": seq,
        "t_emit": t_emit,
        "horizon": len(rows),
        "dim": len(rows[0]),
        "exec_steps": exec_steps,
        "data": [x for r in rows for x in r],
    }


def rows_of(chunk):
    d = chunk["dim"]
    return [chunk["data"][i * d:(i + 1) * d] for i in range(chunk["horizon"])]


def r3(x):
    return round(x, 3)


def arc_rows(p0, step, theta0, dtheta, h):
    """h rows walking `step` px per row along a slowly turning heading (a smooth policy chunk)."""
    rows = []
    x, y = p0
    for i in range(h):
        th = theta0 + i * dtheta
        x += step * math.cos(th)
        y += step * math.sin(th)
        rows.append([r3(x), r3(y)])
    return rows


def line_rows(p0, step_vec, h):
    return [[r3(p0[0] + (i + 1) * step_vec[0]), r3(p0[1] + (i + 1) * step_vec[1])] for i in range(h)]


# --------------------------------------------------------------------------- Tier 1

TIER1_BASE = {
    "dim": 2,
    "pos_dim": 2,
    "horizon": 15,
    "exec": 8,
    "dt": 0.1,
    "norm_center": [256.0, 256.0],
    "norm_scale": [256.0, 256.0],
    "norm_scale_iso": 256.0,
}


class Tier1Model:
    """Mirror of lictor_detect::tier1::features + Tier1Rt (trail pushed by the caller, as decide() does)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.prev = None
        self.trail = []
        self.held = [0.0] * NFEAT
        self.held_valid = 0

    def push_trail(self, pos):
        self.trail.append(list(pos))
        if len(self.trail) > TRAIL_W:
            del self.trail[0]

    def abar(self, chunk, i, c):
        cfg = self.cfg
        return (chunk["data"][i * chunk["dim"] + c] - cfg["norm_center"][c]) * (1.0 / cfg["norm_scale"][c])

    def boundary(self, pos, new):
        cfg = self.cfg
        held = [0.0] * NFEAT
        valid = 0
        d = new["dim"]
        h = new["horizon"]
        nb = min(d, cfg["pos_dim"], len(pos))
        iso = cfg["norm_scale_iso"]
        prev = self.prev
        if h >= 1 and prev is not None and prev["dim"] == d and new["t_emit"] >= prev["t_emit"]:
            s = new["t_emit"] - prev["t_emit"]
            if s < prev["horizon"]:
                L = min(prev["horizon"] - s, h)
                if L >= 2:
                    num = 0.0
                    for i in range(L):
                        for c in range(d):
                            e = self.abar(prev, s + i, c) - self.abar(new, i, c)
                            num += e * e
                    den = 0.0
                    for i in range(1, L):
                        for c in range(d):
                            e = self.abar(new, i, c) - self.abar(new, i - 1, c)
                            den += e * e
                    held[F["tce"]] = math.sqrt(num / float(L * d))
                    held[F["acc"]] = math.sqrt(num / (den + EPS))
                    valid |= (1 << F["tce"]) | (1 << F["acc"])
        if h >= 2:
            acc = 0.0
            for i in range(1, h):
                v = [self.abar(new, i, c) - self.abar(new, i - 1, c) for c in range(d)]
                acc += norm(v, d)
            held[F["acm_neg"]] = -(acc / float(h - 1))
            valid |= 1 << F["acm_neg"]
        if h >= 4:
            num = 0.0
            for i in range(h - 3):
                for c in range(d):
                    v = ((self.abar(new, i + 3, c) - 3.0 * self.abar(new, i + 2, c))
                         + 3.0 * self.abar(new, i + 1, c)) - self.abar(new, i, c)
                    num += v * v
            den = 0.0
            for i in range(h - 1):
                for c in range(d):
                    v = self.abar(new, i + 1, c) - self.abar(new, i, c)
                    den += v * v
            held[F["njr"]] = (num / float(h - 3)) / (EPS + den / float(h - 1))
            valid |= 1 << F["njr"]
        rows = rows_of(new)
        if h >= 1 and nb > 0:
            held[F["reach"]] = dist(rows[0], pos, nb) / iso
            valid |= 1 << F["reach"]
        if h >= 1:
            q0 = list(rows[0])
            q0[:nb] = pos[:nb]
            peak = 0.0
            for i in range(h):
                n = dist(rows[i], q0 if i == 0 else rows[i - 1], d)
                peak = fmax(peak, n)
            held[F["speed_peak"]] = peak / iso
            valid |= 1 << F["speed_peak"]
        self.held = held
        self.held_valid = valid
        self.prev = new

    def features(self, pos, ext, chunk):
        if chunk is not None:
            self.boundary(pos, chunk)
        f = [0.0] * NFEAT
        valid = self.held_valid & CHUNK_BOUNDARY_MASK
        for j in range(NFEAT):
            if (valid >> j) & 1:
                f[j] = self.held[j]
        cfg = self.cfg
        if len(self.trail) >= PE_WINDOW:
            w = self.trail[-PE_WINDOW:]
            pd = cfg["pos_dim"]
            N = dist(w[-1], w[0], pd)
            P = 0.0
            for k in range(1, PE_WINDOW):
                P += dist(w[k - 1], w[k], pd)
            f[F["path_ineff"]] = 1.0 - N / fmax(P, EPS)
            v_ref = STALL_VREF_FRAC * cfg["norm_scale_iso"] / cfg["dt"]
            denom = v_ref * float(PE_WINDOW) * cfg["dt"]
            f[F["stall"]] = 1.0 - fmin(1.0, N / denom)
            valid |= (1 << F["path_ineff"]) | (1 << F["stall"])
        for k in range(min(len(ext), MAX_EXT)):
            f[F["ext0"] + k] = ext[k]
            valid |= 1 << (F["ext0"] + k)
        return f, valid


def tier1_case(name, note, cfg, ticks, extra=None):
    """ticks: list of (t, idx, pos, ext, chunk-or-None). Returns the fixture case with expectations."""
    model = Tier1Model(cfg)
    out_ticks = []
    for (t, idx, pos, ext, chunk) in ticks:
        model.push_trail(pos)
        f, valid = model.features(pos, ext, chunk)
        out_ticks.append({
            "t": t,
            "idx": idx,
            "pos": list(pos),
            "ext": list(ext),
            "chunk": chunk,
            "expect": {"f": f, "valid": valid},
        })
    case = {"name": name, "note": note, "config": cfg, "ticks": out_ticks}
    if extra:
        case.update(extra)
    return case


def tracking_pos(rows, t, t_emit, p0):
    """A plausible measured position: the row executed at t-1 (the plant lags the setpoint by one step)."""
    k = t - t_emit - 1
    if k < 0:
        return list(p0)
    k = min(k, len(rows) - 1)
    return [r3(rows[k][0] - 0.4), r3(rows[k][1] + 0.3)]


def build_tier1():
    cases = []
    base = dict(TIER1_BASE)
    p0 = [256.0, 256.0]

    # 1. sync d=0: three chunks 8 steps apart, then hold; ext appears and grows; per-tick features from t=19.
    A = arc_rows(p0, 12.0, 0.3, 0.05, 15)
    B_rows = [[r3(A[8 + i][0] + 0.9 * i), r3(A[8 + i][1] - 0.6 * i)] for i in range(7)]
    B_rows += arc_rows(B_rows[-1], 12.0, 0.3 + 15 * 0.05, 0.05, 8)
    C_rows = [[r3(B_rows[8 + i][0] - 1.5 * i), r3(B_rows[8 + i][1] + 2.0 * i)] for i in range(7)]
    C_rows += arc_rows(C_rows[-1], 9.0, 1.4, -0.08, 8)
    chunks = {0: mk_chunk(0, 0, A), 8: mk_chunk(1, 8, B_rows), 16: mk_chunk(2, 16, C_rows)}
    rows_by_emit = {0: A, 8: B_rows, 16: C_rows}
    ticks = []
    cur_emit = 0
    for t in range(0, 25):
        if t in chunks:
            cur_emit = t
        pos = tracking_pos(rows_by_emit[cur_emit], t, cur_emit, p0)
        if t < 5:
            ext = []
        elif t < 20:
            ext = [0.25, -1.5]
        else:
            ext = [0.1, 0.2, 0.3, 0.4]
        ticks.append((t, t - cur_emit, pos, ext, chunks.get(t)))
    cases.append(tier1_case(
        "sync_d0_pair_then_hold",
        "Three chunks 8 steps apart (L = 7): tce/acc invalid on the first chunk, valid from the second; boundary "
        "features held between deliveries; path_ineff/stall valid from t = 19 (20 trail samples); ext valid iff "
        "supplied (0, 2, then 4 values).",
        base, ticks))

    # 2. frozen policy: every row equals the current position -> acm_neg = 0 (its maximum), njr = 0, speed_peak = 0.
    frozen = [[256.0, 256.0] for _ in range(15)]
    chunks = {0: mk_chunk(0, 0, frozen), 8: mk_chunk(1, 8, frozen)}
    ticks = [(t, t % 8, [256.0, 256.0], [], chunks.get(t)) for t in range(0, 21)]
    cases.append(tier1_case(
        "frozen_policy",
        "All 15 rows equal the position: acm_neg = 0.0 (a frozen policy has the largest possible score), "
        "njr = 0 (0 / eps), reach = 0, speed_peak = 0; the second chunk gives tce = acc = 0. A motionless trail "
        "gives path_ineff = 1 - 0/eps = 1 and stall = 1 from t = 19.",
        base, ticks))

    # 3. dithering trail: the position zig-zags with little net progress -> path_ineff dominates.
    chunk0 = mk_chunk(0, 0, arc_rows(p0, 6.0, 0.0, 0.1, 15))
    ticks = []
    for t in range(0, 23):
        pos = [r3(256.0 + 5.0 * t), r3(256.0 + (12.0 if t % 2 == 0 else -12.0))]
        ticks.append((t, min(t, 14), pos, [], chunk0 if t == 0 else None))
    cases.append(tier1_case(
        "dithering_trail",
        "A single chunk, then a zig-zag trail (5 px/tick drift, +-12 px dither): over the 20-sample window the net "
        "displacement (~95 px) is a fifth of the path length (~466 px), so path_ineff (~0.80) dominates stall "
        "(~0.63); boundary features stay held.",
        base, ticks))

    # 4-6. sync delay: t_emit differences 15 (absent), 14 (L = 1, invalid) and 13 (L = 2, valid).
    A = arc_rows(p0, 10.0, 1.0, -0.03, 15)
    B = arc_rows(A[-1], 10.0, 0.6, -0.03, 15)
    for d, name, note in [
        (7, "sync_d7_overlap_absent", "t_emit difference 15 == prev.horizon: the overlap is absent; tce/acc invalid."),
        (6, "sync_d6_overlap_l1", "t_emit difference 14: L = 1 < 2; tce/acc invalid."),
        (5, "sync_d5_overlap_l2", "t_emit difference 13: L = 2, the degraded-but-valid minimum; tce/acc valid."),
    ]:
        t2 = 8 + d
        chunks = {0: mk_chunk(0, 0, A), t2: mk_chunk(1, t2, B)}
        ticks = []
        for t in range(0, t2 + 2):
            emit = t2 if t >= t2 else 0
            pos = tracking_pos(A if emit == 0 else B, t, emit, p0)
            ticks.append((t, t - emit, pos, [], chunks.get(t)))
        cases.append(tier1_case(name, note, base, ticks))

    # 7. async drop, d = 3: chunks arrive with t_emit = t - 3 and idx = 3; deliveries 8 apart keep L = 7.
    A = arc_rows(p0, 11.0, -0.5, 0.04, 15)
    B = [[r3(A[8 + i][0] + 0.2 * i), r3(A[8 + i][1] + 0.7 * i)] for i in range(7)] + arc_rows(A[-1], 11.0, -0.2, 0.04, 8)
    chunks = {3: mk_chunk(0, 0, A), 11: mk_chunk(1, 8, B)}
    ticks = []
    for t in range(0, 14):
        if t < 3:
            pos = list(p0)
            idx = 0
        elif t < 11:
            pos = tracking_pos(A, t, 0, p0)
            idx = t
        else:
            pos = tracking_pos(B, t, 8, p0)
            idx = t - 8
        ticks.append((t, idx, pos, [0.5], chunks.get(t)))
    cases.append(tier1_case(
        "async_drop_d3",
        "Async delivery with stitch = drop: the first chunk arrives at t = 3 with t_emit = 0, the second at t = 11 "
        "with t_emit = 8, so L = 7 (async keeps the overlap); a single ext scalar.",
        base, ticks))

    # 8-9. raw-vs-projected previous chunk: a spike in the previous chunk that the Tier-0 leash would shorten.
    A_raw = line_rows(p0, [8.0, 3.0], 15)
    A_raw[10] = [r3(A_raw[10][0] + 250.0), r3(A_raw[10][1] - 40.0)]
    B = line_rows(A_raw[7], [8.0, 3.0], 15)
    t0cfg = dict(TIER0_BASE)
    A_proj_flat = check_chunk(t0cfg, p0, [], mk_chunk(0, 0, A_raw))["out"]
    A_proj = [A_proj_flat[i * 2:(i + 1) * 2] for i in range(15)]
    for name, first_rows, note, extra in [
        ("raw_prev_spike",
         A_raw,
         "The previous chunk carries a 250 px spike in row 10 (inside the overlap rows 8..14). features() stores "
         "the RAW chunk, so tce/acc at the second delivery see the spike.",
         None),
        ("projected_prev_spike",
         A_proj,
         "The same pair, but the first chunk is what the Tier-0 projection would have made of it (spike leashed to "
         "step_max). Its tce at the second delivery differs from raw_prev_spike: feeding features() a projected "
         "chunk would change the calibration/enforcement feature identity, which is why decide() passes the raw chunk.",
         {"tce_differs_from": "raw_prev_spike"}),
    ]:
        chunks = {0: mk_chunk(0, 0, first_rows), 8: mk_chunk(1, 8, B)}
        ticks = []
        for t in range(0, 10):
            emit = 8 if t >= 8 else 0
            pos = tracking_pos(first_rows if emit == 0 else B, t, emit, p0)
            ticks.append((t, t - emit, pos, [], chunks.get(t)))
        cases.append(tier1_case(name, note, base, ticks, extra))

    # 10. short horizon: h = 3 -> njr invalid, acm_neg valid; the second chunk overlaps by L = 1 -> tce/acc invalid.
    cfg3 = dict(base)
    cfg3["horizon"] = 3
    cfg3["exec"] = 2
    A = line_rows(p0, [5.0, -2.0], 3)
    B = line_rows(A[1], [5.0, -2.0], 3)
    chunks = {0: mk_chunk(0, 0, A, 2), 2: mk_chunk(1, 2, B, 2)}
    ticks = []
    for t in range(0, 4):
        emit = 2 if t >= 2 else 0
        pos = tracking_pos(A if emit == 0 else B, t, emit, p0)
        ticks.append((t, t - emit, pos, [], chunks.get(t)))
    cases.append(tier1_case(
        "short_horizon_h3",
        "horizon 3 / exec 2: njr needs H >= 4 (invalid), acm_neg needs H >= 2 (valid); the second chunk 2 steps "
        "later overlaps by L = min(3 - 2, 3) = 1 -> tce/acc invalid.",
        cfg3, ticks))

    # 11. action dim 3 over a 2-D position: box/reach coordinates are the first min(dim, pos_dim) = 2; the third
    #     coordinate takes q_{-1} = a_0 (no first-row step) in speed_peak.
    cfg3d = dict(base)
    cfg3d["dim"] = 3
    cfg3d["norm_center"] = [256.0, 256.0, 0.0]
    cfg3d["norm_scale"] = [256.0, 256.0, 2.0]
    A = [[r3(256.0 + 7.0 * (i + 1)), r3(256.0 - 4.0 * (i + 1)), r3(0.5 * i)] for i in range(15)]
    B = [[r3(A[8 + i][0] + 0.3), r3(A[8 + i][1] - 0.2), r3(A[8 + i][2] + 0.1)] for i in range(7)]
    B += [[r3(B[6][0] + 7.0 * (i + 1)), r3(B[6][1] - 4.0 * (i + 1)), r3(B[6][2] + 0.5 * (i + 1))] for i in range(8)]
    chunks = {0: mk_chunk(0, 0, A), 8: mk_chunk(1, 8, B)}
    ticks = []
    for t in range(0, 10):
        emit = 8 if t >= 8 else 0
        rows = A if emit == 0 else B
        k = t - emit - 1
        pos = list(p0) if k < 0 else [r3(rows[k][0] - 0.4), r3(rows[k][1] + 0.3)]
        ticks.append((t, t - emit, pos, [], chunks.get(t)))
    cases.append(tier1_case(
        "dim3_pos2",
        "Action dim 3 with a 2-D proprioceptive position: reach uses the first two coordinates; speed_peak's "
        "q_{-1} is the position for those and a_0 itself for the third (no step on row 0 there).",
        cfg3d, ticks))
    return cases


# --------------------------------------------------------------------------- Tier 0

TIER0_BASE = {
    "dim": 2,
    "pos_dim": 2,
    "dt": 0.1,
    "box_lo": [17.0, 17.0],
    "box_hi": [495.0, 495.0],
    "v_max": 1000.0,
    "a_max": 20000.0,
    "j_max": 400000.0,
    "reach_max": 150.0,
    "clamp": "project",
    "tier0_enabled": ["workspace", "speed", "accel", "jerk", "reach", "brake"],
    "contact": None,
}


def derived(cfg):
    inv_dt = 1.0 / cfg["dt"]
    inv_dt2 = inv_dt * inv_dt
    inv_dt3 = inv_dt2 * inv_dt
    return inv_dt, inv_dt2, inv_dt3, cfg["v_max"] * cfg["dt"]


def arm_of(cfg, pos, aux):
    en = set(cfg["tier0_enabled"])
    contact = False
    step_contact = 0.0
    if "contact" in en and cfg["contact"] is not None:
        cl = cfg["contact"]
        i0, i1 = cl["aux_center"]
        if len(pos) >= 2 and len(aux) > i0 and len(aux) > i1:
            if dist(pos, [aux[i0], aux[i1]], 2) <= cl["radius"]:
                contact = True
                step_contact = cl["v_max"] * cfg["dt"]
    return {
        "workspace": "workspace" in en,
        "speed": "speed" in en,
        "accel": "accel" in en,
        "jerk": "jerk" in en,
        "reach": "reach" in en,
        "contact": contact,
        "step_contact": step_contact,
    }


def leash_of(arm, step_max):
    if arm["speed"] and arm["contact"]:
        return fmin(step_max, arm["step_contact"])
    if arm["speed"]:
        return step_max
    if arm["contact"]:
        return arm["step_contact"]
    return None


def outside_box(cfg, a, nb):
    for c in range(nb):
        if a[c] < cfg["box_lo"][c] or a[c] > cfg["box_hi"][c]:
            return True
    return False


def project_row(cfg, pos, arm, leash, first, nb, d, q, a):
    a = list(a)
    if arm["workspace"]:
        for c in range(nb):
            a[c] = clamp(a[c], cfg["box_lo"][c], cfg["box_hi"][c])
    if leash is not None:
        n = dist(a, q, d)
        if n > leash:
            s = leash * SHRINK / n
            for c in range(d):
                a[c] = q[c] + (a[c] - q[c]) * s
    if first and arm["reach"] and nb > 0:
        r = dist(a, pos, nb)
        if r > cfg["reach_max"]:
            s = cfg["reach_max"] * SHRINK / r
            for c in range(nb):
                a[c] = pos[c] + (a[c] - pos[c]) * s
    return a


def check_chunk(cfg, pos, aux, chunk):
    inv_dt, inv_dt2, inv_dt3, step_max = derived(cfg)
    rows = rows_of(chunk)
    d = chunk["dim"]
    h = chunk["horizon"]
    nb = min(d, cfg["pos_dim"], len(pos))
    arm = arm_of(cfg, pos, aux)
    q0 = list(rows[0])
    q0[:nb] = pos[:nb]
    trips = set()
    peak = 0.0
    for i in range(h):
        n = dist(rows[i], q0 if i == 0 else rows[i - 1], d)
        peak = fmax(peak, n)
        if arm["speed"] and n > step_max:
            trips.add("speed")
        if arm["contact"] and n > arm["step_contact"]:
            trips.add("contact")
        if arm["workspace"] and outside_box(cfg, rows[i], nb):
            trips.add("workspace")
    if arm["accel"] and h >= 2:
        for i in range(h - 1):
            am1 = q0 if i == 0 else rows[i - 1]
            v = [(rows[i + 1][c] - 2.0 * rows[i][c]) + am1[c] for c in range(d)]
            if norm(v, d) * inv_dt2 > cfg["a_max"]:
                trips.add("accel")
    if arm["jerk"] and h >= 3:
        for i in range(h - 2):
            am1 = q0 if i == 0 else rows[i - 1]
            v = [((rows[i + 2][c] - 3.0 * rows[i + 1][c]) + 3.0 * rows[i][c]) - am1[c] for c in range(d)]
            if norm(v, d) * inv_dt3 > cfg["j_max"]:
                trips.add("jerk")
    if arm["reach"] and nb > 0 and dist(rows[0], pos, nb) > cfg["reach_max"]:
        trips.add("reach")
    out = [list(r) for r in rows]
    clamped = 0
    if cfg["clamp"] == "project":
        leash = leash_of(arm, step_max)
        q = q0
        for i in range(h):
            a = project_row(cfg, pos, arm, leash, i == 0, nb, d, q, rows[i])
            for c in range(d):
                if a[c] != rows[i][c]:
                    clamped |= 1 << c
            out[i] = a
            q = a
    return {
        "trips": sorted(trips, key=TRIPS.index),
        "clamped_dims": clamped,
        "peak_speed": peak * inv_dt,
        "out": [x for r in out for x in r],
    }


def check_action(cfg, pos, aux, prev, a):
    inv_dt, inv_dt2, inv_dt3, step_max = derived(cfg)
    d = min(cfg["dim"], len(a), len(prev))
    nb = min(d, cfg["pos_dim"], len(pos))
    arm = arm_of(cfg, pos, aux)
    trips = set()
    n = dist(a, prev, d)
    if arm["speed"] and n > step_max:
        trips.add("speed")
    if arm["contact"] and n > arm["step_contact"]:
        trips.add("contact")
    if arm["workspace"] and outside_box(cfg, a, nb):
        trips.add("workspace")
    if arm["reach"] and nb > 0 and dist(a, pos, nb) > cfg["reach_max"]:
        trips.add("reach")
    out = list(a[:d])
    clamped = 0
    if cfg["clamp"] == "project":
        out = project_row(cfg, pos, arm, leash_of(arm, step_max), True, nb, d, list(prev[:d]), a[:d])
        for c in range(d):
            if out[c] != a[c]:
                clamped |= 1 << c
    return {
        "trips": sorted(trips, key=TRIPS.index),
        "clamped_dims": clamped,
        "peak_speed": n * inv_dt,
        "out": out,
    }


def t0cfg(**over):
    cfg = dict(TIER0_BASE)
    cfg.update(over)
    return cfg


def build_tier0():
    p0 = [256.0, 256.0]
    chunk_cases = []

    def add(name, note, cfg, pos, aux, rows, exec_steps=8):
        chunk = mk_chunk(0, 0, rows, exec_steps)
        chunk_cases.append({
            "name": name,
            "note": note,
            "config": cfg,
            "pos": pos,
            "aux": aux,
            "chunk": chunk,
            "expect": check_chunk(cfg, pos, aux, chunk),
        })

    add("inside_box_smooth", "10 px steps from the current position: no trip, nothing clamped, out == chunk, "
        "peak_speed = 100 px/s.", t0cfg(), p0, [], line_rows(p0, [10.0, 0.0], 15))
    add("workspace_one_row", "Row 2 leaves the (margin-adjusted) box on x: WORKSPACE, clamped_dims bit 0, that row "
        "is clamped to box_hi; the steps stay within the leash.",
        t0cfg(), [480.0, 256.0], [], [[490.0, 256.0], [500.0, 256.0], [490.0, 256.0]] + line_rows([490.0, 256.0], [-10.0, 0.0], 12))
    steps = [[10.0, 0.0]] * 2 + [[150.0, 0.0]] + [[10.0, 0.0]] * 12
    rows = []
    p = list(p0)
    for s in steps:
        p = [p[0] + s[0], p[1] + s[1]]
        rows.append([r3(p[0]), r3(p[1])])
    add("speed_1p5x_mid_chunk", "One 150 px step (1.5 x step_max) in row 2: SPEED only; the leash shortens that "
        "step to step_max along its direction and later rows follow from the shortened point.", t0cfg(), p0, [], rows)
    add("speed_first_row", "a_0 is 120 px from p_t (q_{-1} = p_t): SPEED on the first row, leashed toward p_t; "
        "reach (120 <= 150) does not trip.", t0cfg(), p0, [], line_rows(p0, [96.0, 72.0], 1) + line_rows([352.0, 328.0], [6.0, 8.0], 14))
    steps = [[-10.0, 0.0]] * 3 + [[-80.0, 0.0]] * 2 + [[-10.0, 0.0]] * 10
    rows = []
    p = [400.0, 256.0]
    for s in steps:
        p = [p[0] + s[0], p[1] + s[1]]
        rows.append([r3(p[0]), r3(p[1])])
    add("accel_only", "a_max lowered to 5000: the 10 -> 80 px step change is 7000 px/s^2: ACCEL trips although "
        "the projection changes nothing (a soft trip whose clamp is a no-op still counts).",
        t0cfg(a_max=5000.0), [400.0, 256.0], [], rows)
    steps = [[-10.0, 0.0]] * 3 + [[-70.0, 0.0]] * 2 + [[-10.0, 0.0]] * 10
    rows = []
    p = [400.0, 256.0]
    for s in steps:
        p = [p[0] + s[0], p[1] + s[1]]
        rows.append([r3(p[0]), r3(p[1])])
    add("jerk_only", "j_max lowered to 50000: the third difference at the step change is 60 px / dt^3 = 60000 "
        "px/s^3: JERK trips; accel 6000 and speed 700 stay under their limits.",
        t0cfg(j_max=50000.0), [400.0, 256.0], [], rows)
    add("reach_only", "v_max raised so a 160 px first step is legal for SPEED, but ||a_0 - p_t|| = 160 > 150: "
        "REACH; a_0 is pulled toward p_t along the ray, later rows are unchanged.",
        t0cfg(v_max=3000.0), p0, [], line_rows(p0, [128.0, 96.0], 1) + line_rows([384.0, 352.0], [4.0, 3.0], 14))
    add("disabled_speed_no_trip", "The same 1.5 x step with speed removed from tier0_enabled: no trip and no "
        "clamp (a disabled check neither trips nor clamps).",
        t0cfg(tier0_enabled=["workspace", "accel", "jerk", "reach", "brake"]), p0, [], line_rows(p0, [150.0, 0.0], 1) + line_rows([406.0, 256.0], [-10.0, 0.0], 14))
    add("clamp_off_passthrough", "clamp_mode = off: SPEED trips but nothing is projected (out == chunk, "
        "clamped_dims = 0).", t0cfg(clamp="off"), p0, [], line_rows(p0, [150.0, 0.0], 1) + line_rows([406.0, 256.0], [-10.0, 0.0], 14))
    contact = {"radius": 80.0, "v_max": 400.0, "aux_center": [0, 1]}
    en_contact = ["workspace", "speed", "accel", "jerk", "reach", "contact", "brake"]
    add("contact_armed_inside_radius", "contact armed and the block centre 44 px away (<= 80): 50 px steps "
        "exceed v_contact * dt = 40 px: CONTACT; the leash is the shorter contact leash; speed (50 < 100) is fine.",
        t0cfg(contact=contact, tier0_enabled=en_contact), p0, [300.0, 256.0, 0.1, 0.2], line_rows(p0, [50.0, 0.0], 4))
    add("contact_disarmed", "The same geometry with contact not listed in tier0_enabled: nothing trips, "
        "nothing is clamped.", t0cfg(contact=contact), p0, [300.0, 256.0, 0.1, 0.2], line_rows(p0, [50.0, 0.0], 4))
    add("contact_armed_outside_radius", "contact armed but the block centre is 150 px away (> 80): the reduced "
        "speed limit does not apply.", t0cfg(contact=contact, tier0_enabled=en_contact), p0, [406.0, 256.0, 0.1, 0.2], line_rows(p0, [50.0, 0.0], 4))
    add("workspace_and_speed", "A 127 px diagonal step that also leaves the box: WORKSPACE and SPEED together (reach 127 <= 150); "
        "both dims clamped.", t0cfg(), [470.0, 470.0], [], [[560.0, 560.0]] + line_rows([560.0, 560.0], [-10.0, -10.0], 14))
    add("wide_action_dim3", "Action dim 3 over a 2-D position: the box applies to the first two coordinates; the "
        "third has q_{-1} = a_0 and takes part in the step norm.",
        t0cfg(dim=3), p0, [], [[r3(256.0 + 10.0 * (i + 1)), 256.0, r3(0.2 * i)] for i in range(15)])

    action_cases = []

    def add_a(name, note, cfg, pos, aux, prev, a):
        action_cases.append({
            "name": name,
            "note": note,
            "config": cfg,
            "pos": pos,
            "aux": aux,
            "prev": prev,
            "a": a,
            "expect": check_action(cfg, pos, aux, prev, a),
        })

    add_a("action_inside", "10 px step inside the box: no trip; out == a.", t0cfg(), p0, [], [256.0, 256.0], [266.0, 256.0])
    add_a("action_speed", "150 px step: SPEED; leashed to step_max along the direction.", t0cfg(), p0, [], [256.0, 256.0], [346.0, 376.0])
    add_a("action_box", "Outside the box on y: WORKSPACE; clamped to box_hi.", t0cfg(), [256.0, 480.0], [], [256.0, 480.0], [256.0, 520.0])
    add_a("action_reach", "160 px from p_t with a large v_max: REACH only; pulled onto the reach ball.", t0cfg(v_max=3000.0), p0, [], [300.0, 256.0], [384.0, 352.0])
    add_a("action_contact", "contact armed, block 44 px away, 60 px step > 40: CONTACT; leashed to 40 px.",
          t0cfg(contact=contact, tier0_enabled=en_contact), p0, [300.0, 256.0, 0.1, 0.2], [256.0, 256.0], [316.0, 256.0])
    add_a("action_off_passthrough", "clamp_mode = off: SPEED trips, out == a.", t0cfg(clamp="off"), p0, [], [256.0, 256.0], [406.0, 256.0])
    add_a("action_disabled_speed", "speed disabled: a 150 px step neither trips nor clamps.",
          t0cfg(tier0_enabled=["workspace", "reach", "brake"]), p0, [], [256.0, 256.0], [406.0, 256.0])
    add_a("action_box_and_speed", "Both outside the box and too fast (127 px, under reach_max): WORKSPACE | SPEED; clamp then leash.",
          t0cfg(), [470.0, 470.0], [], [470.0, 470.0], [560.0, 560.0])
    return {"chunk_cases": chunk_cases, "action_cases": action_cases}


# --------------------------------------------------------------------------- conformal

def bin_of(t, horizon_ticks, t_grid):
    g = t_grid
    h = horizon_ticks if horizon_ticks != 0 else 1
    b = t * g // h
    return g - 1 if b >= g else b


def conformal_expect(case):
    b = bin_of(case["t"], case["horizon_ticks"], case["t_grid"])
    if b >= T_GRID:
        b = T_GRID - 1
    row = case["bins"].get(str(b), {"center": [0.0] * NFEAT, "scale": [1.0] * NFEAT})
    tau = math.inf if case["tau"] is None else case["tau"]
    live = case["mask"] & case["valid"]
    f = case["f"]
    z = [(f[j] - row["center"][j]) / row["scale"][j] if (live >> j) & 1 else 0.0 for j in range(NFEAT)]
    s = -math.inf
    for tm in case["terms"]:
        if tm != 0 and (tm & live) == tm:
            v = math.inf
            for j in range(NFEAT):
                if (tm >> j) & 1:
                    v = fmin(v, z[j])
            s = fmax(s, v)
    fired = 0
    for j in range(NFEAT):
        if (live >> j) & 1 and z[j] > tau:
            fired |= 1 << j
    return {"z": z, "s": None if s == -math.inf else s, "fired": fired, "trip": s > tau}


def build_conformal():
    cases = []
    f_std = [0.5, -2.0, 3.0, 0.25, 1.0, 0.75, -0.5, 2.5, 4.0, 0.0, -1.0, 1.5]

    def add(name, note, **kw):
        case = {
            "name": name,
            "note": note,
            "horizon_ticks": 300,
            "t_grid": 1,
            "t": 0,
            "mask": 0,
            "terms": [],
            "tau": 1.0,
            "bins": {},
            "f": list(f_std),
            "valid": 0xfff,
        }
        case.update(kw)
        case["expect"] = conformal_expect(case)
        cases.append(case)

    singles = [1 << j for j in range(8)]
    add("empty_mask", "mask 0 (Tier 1 disarmed): every z is 0, s = NEG_INFINITY (null), nothing fires, no trip.")
    add("all_invalid", "mask armed but valid = 0: no term is fully valid -> s = NEG_INFINITY, no trip.",
        mask=0xff, terms=singles, valid=0)
    add("singleton_gate_plain_max", "Eight singleton terms == a plain max over the eight channels; fired = the "
        "channels above tau.", mask=0xff, terms=singles, valid=0xff, tau=2.0)
    add("and_pair_is_min", "One AND term over tce and acc == min(z_tce, z_acc).",
        mask=0b11, terms=[0b11], valid=0b11, tau=0.0)
    add("and_pair_one_invalid", "The AND term with acc invalid: the term is disqualified -> s = NEG_INFINITY "
        "even though z_tce alone would exceed tau.", mask=0b11, terms=[0b11], valid=0b01, tau=0.0)
    add("boundary_s_eq_tau", "center 0, scale 1, f_tce == tau: s == tau must NOT trip (strict) and nothing fires.",
        mask=0b1, terms=[0b1], valid=0b1, tau=2.0, f=[2.0] + [0.0] * 11)
    add("just_above_tau", "f_tce one nanounit above tau: trips and fires.",
        mask=0b1, terms=[0b1], valid=0b1, tau=2.0, f=[2.000000001] + [0.0] * 11)
    add("dnf_mixed_partial_invalid", "Gate [tce|acm_neg, acc|path_ineff, stall|path_ineff] with path_ineff invalid: "
        "only the first term contributes.", mask=0b0110_0111, terms=[0b101, 0b10_0010, 0b110_0000], valid=0b1111_1111 & ~(1 << 5), tau=1.0)
    add("dnf_all_terms_valid", "The same gate with every channel valid: s = max of the three term minima.",
        mask=0b0110_0111, terms=[0b101, 0b10_0010, 0b110_0000], valid=0xff, tau=1.0)
    add("unmasked_channel_is_zero", "acm_neg valid but not in the mask: z_acm_neg = 0 and it never fires.",
        mask=0b11, terms=[0b1, 0b10], valid=0b111, tau=0.1)
    bins = {
        "0": {"center": [0.1 * j for j in range(NFEAT)], "scale": [1.0 + 0.5 * j for j in range(NFEAT)]},
        "50": {"center": [1.0] * NFEAT, "scale": [2.0] * NFEAT},
        "99": {"center": [-0.3 + 0.2 * j for j in range(NFEAT)], "scale": [0.25 + 0.1 * j for j in range(NFEAT)]},
    }
    add("binned_t0_bin0", "t_grid 100 over 300 ticks, t = 0 -> bin 0 (its own center/scale).",
        t_grid=100, t=0, mask=0xff, terms=singles, valid=0xff, tau=3.0, bins=bins)
    add("binned_t150_bin50", "t = 150 -> bin 50.", t_grid=100, t=150, mask=0xff, terms=singles, valid=0xff, tau=3.0, bins=bins)
    add("binned_t299_bin99", "t = 299 -> bin 99 (the last bin).", t_grid=100, t=299, mask=0xff, terms=singles, valid=0xff, tau=3.0, bins=bins)
    add("binned_t300_clamps_to_bin99", "t = 300 == horizon_ticks -> clamped to bin 99, never out of range.",
        t_grid=100, t=300, mask=0xff, terms=singles, valid=0xff, tau=3.0, bins=bins)
    add("static_uses_bin0", "t_grid 1 (static): t = 250 still lands in bin 0.",
        t_grid=1, t=250, mask=0xff, terms=singles, valid=0xff, tau=3.0, bins=bins)
    add("tau_infinite_never_trips", "tau = +INFINITY (null): finite s, nothing fires, no trip.",
        mask=0xff, terms=singles, valid=0xff, tau=None)
    add("negative_scores", "Every z negative: s is finite and negative, below tau.",
        mask=0b111, terms=singles[:3], valid=0b111, tau=-0.5, f=[-1.0, -2.0, -3.0] + [0.0] * 9)
    add("ext_channels", "Tier-2 ext channels in the gate: ext0 (z = 4) is the max.",
        mask=0xf00, terms=[1 << 8, 1 << 9, 1 << 10, 1 << 11], valid=0xfff, tau=3.5)

    bin_cases = []
    for (t, h, g) in [(0, 300, 100), (2, 300, 100), (3, 300, 100), (150, 300, 100), (299, 300, 100),
                      (300, 300, 100), (1000000, 300, 100), (0, 300, 1), (299, 300, 1), (5, 0, 100),
                      (0, 1, 100), (7, 8, 4)]:
        bin_cases.append({"t": t, "horizon_ticks": h, "t_grid": g, "expect": bin_of(t, h, g)})
    return {"cases": cases, "bin_of": bin_cases}


# --------------------------------------------------------------------------- main

def render(obj):
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the committed fixtures instead of writing them")
    args = ap.parse_args(argv)
    generated = {
        "tier1": render(build_tier1()),
        "tier0": render(build_tier0()),
        "conformal": render(build_conformal()),
    }
    if args.check:
        bad = []
        for key, text in generated.items():
            path = OUTPUTS[key]
            try:
                with open(path, "r", encoding="ascii", newline="\n") as fh:
                    current = fh.read()
            except OSError as exc:
                bad.append("%s: %s" % (path, exc))
                continue
            if current != text:
                bad.append("%s: differs from the generator output" % path)
        if bad:
            for line in bad:
                print("FAIL " + line)
            return 1
        for key in generated:
            print("OK   " + OUTPUTS[key])
        return 0
    for key, text in generated.items():
        path = OUTPUTS[key]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        print("wrote " + path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
