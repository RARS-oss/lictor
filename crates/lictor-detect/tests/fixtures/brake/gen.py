#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reference reimplementation of lictor-detect's brake feasibility (ARCHITECTURE sec 5.2) and the generator of
cases.json. Stdlib only. The PD loop is the gym-pusht agent verbatim (docs/VERIFIED_FACTS.md): per physics substep
acc = k_p*(a - p) - k_v*v ; v += acc*dt ; p += v*dt. Every expression is written in the same order as brake.rs so the
IEEE-754 results agree bit for bit; the Rust test tolerates 1e-12 relative.

    python3 gen.py            # rewrite cases.json
    python3 gen.py --check    # regenerate in memory and diff against cases.json (exit 1 on any difference)
"""
import json
import struct
import sys
from pathlib import Path

INF = float("inf")


def fmin(a, b):  # lictor_core::fmath::min -- NOT Python's min (argument order on ties)
    return a if a < b else b


def norm(v):
    s = 0.0
    for x in v:  # fixed left-to-right accumulation
        s = s + x * x
    return s ** 0.5


def dist(a, b):
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s = s + d * d
    return s ** 0.5


def clamp(x, lo, hi):  # lictor_core::fmath::clamp = max(lo, min(x, hi))
    return max(lo, min(x, hi))


def pd_feasible(c, lo, hi, p, v):
    kp, kv, dt, sub = c["k_p"], c["k_v"], c["dt"], c["substeps"]
    margin = INF

    def substep(a):
        nonlocal margin
        for i in range(len(p)):
            acc = kp * (a[i] - p[i]) - kv * v[i]
            v[i] = v[i] + acc * dt
            p[i] = p[i] + v[i] * dt
        for i in range(len(p)):
            margin = fmin(margin, fmin(p[i] - lo[i], hi[i] - p[i]))

    for row in range(c["from"], c["commit_steps"]):
        a = c["chunk"][min(row, len(c["chunk"]) - 1)]
        for _ in range(sub):
            substep(a)
    a_hold = [clamp(p[i], lo[i], hi[i]) for i in range(len(p))]
    for _ in range(c["brake_steps"] * sub):
        substep(a_hold)
    return margin, 0.0


def closed_form(c, lo, hi, p, v):
    dt = c["control_hz_den"] / c["control_hz_num"]
    react = c["react_ticks"] * dt
    g = c["k_p"] / c["k_v"]

    def radius(vv):
        nv = norm(vv)
        k = c["kind"]
        if k == "first_order_decay":
            d_stop = nv / c["k_v"]
        elif k in ("bounded_accel", "zero_velocity_hold"):
            d_stop = nv * nv / (2.0 * c["a_max"])
        elif k == "jerk_limited":
            d_stop = nv * nv / (2.0 * c["a_max"]) + nv * c["a_max"] / (2.0 * c["j_max"])
        else:
            raise ValueError(k)
        return d_stop + nv * react

    def slack_r(r):
        m = INF
        for i in range(len(p)):
            m = fmin(m, fmin(p[i] - r - lo[i], hi[i] - p[i] - r))
        return m

    margin = slack_r(radius(v))
    for row in range(c["from"], c["commit_steps"]):
        a = c["chunk"][min(row, len(c["chunk"]) - 1)]
        for i in range(len(p)):
            if c["action_kind"] == "joint_velocity":
                v[i] = a[i]
            elif c["action_kind"] == "ee_delta":
                v[i] = a[i] / dt
            else:
                v[i] = g * (a[i] - p[i])
        margin = fmin(margin, slack_r(radius(v)))
        for i in range(len(p)):
            p[i] = p[i] + v[i] * dt
    return margin, radius(v)


def evaluate(c):
    lo = [b + c["margin"] for b in c["box_lo"]]
    hi = [b - c["margin"] for b in c["box_hi"]]
    p, v = list(c["p0"]), list(c["v0"])
    fn = pd_feasible if c["kind"] == "pd_second_order" else closed_form
    margin, beyond = fn(c, lo, hi, p, v)
    return {"feasible": margin >= 0.0, "margin": margin, "stop_dist": dist(p, c["p0"]) + beyond}


BASE = dict(
    kind="pd_second_order", action_kind="ee_position", k_p=100.0, k_v=20.0, substeps=10, dt=0.01,
    commit_steps=8, brake_steps=8, react_ticks=1, control_hz_num=10, control_hz_den=1,
    a_max=20000.0, j_max=400000.0, box_lo=[15.0, 15.0], box_hi=[497.0, 497.0], margin=2.0,
    horizon=15, exec_steps=8, v0=[0.0, 0.0], **{"from": 0},
)


def ramp(x0, y0, dx, dy, n=15):
    return [[x0 + dx * (i + 1), y0 + dy * (i + 1)] for i in range(n)]


def case(name, note, **kw):
    c = dict(BASE, name=name, note=note)
    c.update(kw)
    return c


def cases():
    wall = ramp(380.0, 256.0, 20.0, 0.0)
    return [
        case("pd_straight_inside", "10 px/step in +x from the centre; nothing near a wall; from=0, v0=0",
             chunk=ramp(256.0, 256.0, 10.0, 0.0), p0=[256.0, 256.0]),
        case("pd_wall_slow", "5 px/step toward the +x wall from x=380; feasible with wide margin",
             chunk=ramp(380.0, 256.0, 5.0, 0.0), p0=[380.0, 256.0]),
        case("pd_wall_medium", "10 px/step toward the +x wall from x=380; feasible", chunk=ramp(380.0, 256.0, 10.0, 0.0),
             p0=[380.0, 256.0]),
        case("pd_wall_fast", "20 px/step toward the +x wall from x=380: the prefix ends past x=495 -> infeasible",
             chunk=wall, p0=[380.0, 256.0]),
        case("pd_wall_fast_from3", "same chunk, from=3: rows 3..7 only (async drop d=3), fewer iterations", chunk=wall,
             p0=[380.0, 256.0], **{"from": 3}),
        case("pd_wall_fast_from7", "same chunk, from=7: one committed row then the brake tail", chunk=wall,
             p0=[380.0, 256.0], **{"from": 7}),
        case("pd_v0_nonzero", "straight chunk with the simulator velocity v0 = (300, -150) px/s",
             chunk=ramp(256.0, 256.0, 10.0, 0.0), p0=[256.0, 256.0], v0=[300.0, -150.0]),
        case("pd_tail_only", "from=8 >= commit_steps: only the 80-substep brake tail from (480, 256) at 400 px/s",
             chunk=wall, p0=[480.0, 256.0], v0=[400.0, 0.0], **{"from": 8}),
        case("pd_asym_box", "asymmetric box y in [100,300], margin 5, moving -y toward the low wall",
             chunk=ramp(256.0, 130.0, 0.0, -12.0), p0=[256.0, 130.0], box_lo=[0.0, 100.0], box_hi=[512.0, 300.0],
             margin=5.0),
        case("pd_coarse_substeps", "5 substeps of dt=0.02 (same 0.1 s period); 15 px/step toward +x",
             chunk=ramp(400.0, 256.0, 15.0, 0.0), p0=[400.0, 256.0], substeps=5, dt=0.02),
        case("pd_no_brake_tail", "brake_steps=0: margin over the committed prefix only", chunk=wall, p0=[380.0, 256.0],
             brake_steps=0),
        case("first_order_decay", "closed form d_stop = ||v||/k_v with react_ticks=1; ee_position toward +x",
             kind="first_order_decay", chunk=ramp(450.0, 256.0, 10.0, 0.0), p0=[450.0, 256.0], v0=[100.0, 0.0]),
        case("bounded_accel", "closed form ||v||^2/(2 a_max), a_max=2000; ee_position toward +x",
             kind="bounded_accel", a_max=2000.0, chunk=ramp(450.0, 256.0, 10.0, 0.0), p0=[450.0, 256.0],
             v0=[100.0, 0.0]),
        case("jerk_limited", "closed form with the a_max/(2 j_max) term, a_max=2000, j_max=40000, react_ticks=2",
             kind="jerk_limited", a_max=2000.0, j_max=40000.0, react_ticks=2,
             chunk=ramp(450.0, 256.0, 10.0, 0.0), p0=[450.0, 256.0], v0=[100.0, 0.0]),
        case("zero_velocity_hold", "joint_velocity rows (150, 0) px/s, a_max=500: BoundedAccel ball, from=2",
             kind="zero_velocity_hold", action_kind="joint_velocity", a_max=500.0,
             chunk=[[150.0, 0.0]] * 15, p0=[400.0, 256.0], **{"from": 2}),
        case("bounded_accel_ee_delta", "ee_delta rows (8, 0) px/step -> 80 px/s, a_max=1000, near the +x wall",
             kind="bounded_accel", action_kind="ee_delta", a_max=1000.0, chunk=[[8.0, 0.0]] * 15,
             p0=[470.0, 256.0], v0=[40.0, 0.0]),
        case("first_order_decay_feasible", "same closed form far from every wall: feasible",
             kind="first_order_decay", chunk=ramp(300.0, 256.0, 10.0, 0.0), p0=[300.0, 256.0], v0=[50.0, 0.0]),
        case("bounded_accel_feasible", "BoundedAccel from the centre with a diagonal v0: feasible",
             kind="bounded_accel", a_max=2000.0, chunk=ramp(256.0, 256.0, 6.0, -6.0), p0=[256.0, 256.0],
             v0=[80.0, -60.0]),
        case("jerk_limited_feasible", "JerkLimited from the centre, react_ticks=0: feasible",
             kind="jerk_limited", a_max=2000.0, j_max=40000.0, react_ticks=0,
             chunk=ramp(256.0, 256.0, 10.0, 0.0), p0=[256.0, 256.0], v0=[100.0, 0.0]),
        case("zero_velocity_hold_feasible", "joint_velocity rows (60, 30) px/s from (200, 200), a_max=500: feasible",
             kind="zero_velocity_hold", action_kind="joint_velocity", a_max=500.0,
             chunk=[[60.0, 30.0]] * 15, p0=[200.0, 200.0], v0=[10.0, 0.0]),
    ]


def render():
    out = []
    for c in cases():
        c = dict(c)
        c["expect"] = evaluate(c)
        out.append(c)
    return json.dumps(out, indent=2, sort_keys=True) + "\n"


def main():
    path = Path(__file__).with_name("cases.json")
    text = render()
    if "--check" in sys.argv[1:]:
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current != text:
            sys.stderr.write("gen.py --check: cases.json differs from the regenerated text\n")
            return 1
        print(f"gen.py --check: cases.json identical ({len(json.loads(text))} cases)")
        return 0
    path.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {path} ({len(json.loads(text))} cases)")
    for c in json.loads(text):
        e = c["expect"]
        print(f"  {c['name']:<26} feasible={str(e['feasible']):<5} margin={e['margin']:+.6f} stop_dist={e['stop_dist']:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
