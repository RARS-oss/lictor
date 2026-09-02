# SPDX-License-Identifier: MIT
"""The ARCHITECTURE 10.5 arm table as data. `arm_id = <detector>-a<alpha%>-d<d>[-async][-<escalation>]`.

Every arm declares `tier1` and `alpha`; `LictorClient.hello` is always called with `expect_tier1 = arm.tier1,
expect_alpha = arm.alpha`, so a `t01-*` arm launched without a calibration cannot silently run as a `t0` arm.
Every arm uses identical paired seeds; the pool is `calib` for `calib-obs` and `eval` otherwise.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, replace

BASE_ENVELOPE = "envelopes/pusht.toml"           # fitted by `lictor envelope fit` (falls back to pusht.base.toml)
ORACLE_ENVELOPE = "envelopes/pusht.oracle.toml"  # same embodiment, demo operator key
ACKONLY_ENVELOPE = "envelopes/pusht.ackonly.toml"  # optional arm: hysteresis.rearm = "ack_only"
INJECTION_DEMO = "action_spike(p=0.02,mag=180)"


@dataclass(frozen=True)
class Arm:
    arm_id: str
    mode: str                      # "observe" | "enforce"
    tier0: str                     # "scored" (observe) | "armed" (enforce)
    tier1: bool                    # calibration loaded and armed
    alpha: tuple | None            # (num, den) or None
    d: int                         # injected staleness in control steps
    exec: str                      # "sync" | "async"
    stitch: str                    # "drop" | "freeze"
    on_escalate: str               # "terminate_fail" | "oracle_resume" | "continue" | "--"
    calibration: str | None        # calibration file name under --calibration-dir
    envelope: str                  # repo-relative envelope TOML
    injection: str | None          # inject.py spec or None
    rearm: str                     # "auto" | "ack_only" (envelope hysteresis.rearm; informational)
    pool: str = "eval"             # "eval" | "calib"
    role: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def budget(self) -> dict:
        return {"delay_steps": int(self.d), "tick_ms": 100, "exec_mode": self.exec, "stitch": self.stitch,
                "on_escalate": self.on_escalate if self.on_escalate != "--" else "terminate_fail"}


def calibration_file(alpha) -> str | None:
    """`(5, 100)` -> `calibration.a05.json` (the `lictor calibrate` naming, `bench/fixtures/calibration.a05.json`)."""
    if alpha is None:
        return None
    num, den = alpha
    if den == 100:
        return "calibration.a%02d.json" % num
    return "calibration.a%d_%d.json" % (num, den)


def _obs(arm_id, d, exec_mode="sync", role="", pool="eval", injection=None):
    return Arm(arm_id, "observe", "scored", False, None, d, exec_mode, "drop", "--", None, BASE_ENVELOPE,
               injection, "auto", pool, role)


def _enf(arm_id, alpha, d, exec_mode="sync", on_escalate="terminate_fail", envelope=BASE_ENVELOPE, injection=None,
         rearm="auto", role=""):
    return Arm(arm_id, "enforce", "armed", alpha is not None, alpha, d, exec_mode, "drop", on_escalate,
               calibration_file(alpha), envelope, injection, rearm, "eval", role)


A05 = (5, 100)

ARMS: tuple[Arm, ...] = (
    _obs("calib-obs", 0, role="calibration pool only (seeds 900000..900299)", pool="calib"),
    _obs("obs-d0", 0, role="FUSE-OFF BASELINE + labels + Layer-A eval traces"),
    _obs("obs-d1", 1, role="latency-only control"),
    _obs("obs-d2", 2, role="latency-only control"),
    _obs("obs-d3", 3, role="latency-only control"),
    _obs("obs-d5", 5, role="latency-only control"),
    _obs("obs-d8", 8, role="latency-only control"),
    _obs("obs-d2-async", 2, "async", role="RTC comparison, fuse-off"),
    _obs("obs-d5-async", 5, "async", role="RTC comparison, fuse-off"),
    _enf("t0-d0", None, 0, role="geometric enforcement alone; parity gate"),
    _enf("t01-a05-d0", A05, 0, role="HEADLINE"),
    _enf("t01-a01-d0", (1, 100), 0, role="closed-loop alpha sweep"),
    _enf("t01-a10-d0", (10, 100), 0, role="closed-loop alpha sweep"),
    _enf("t01-a20-d0", (20, 100), 0, role="closed-loop alpha sweep"),
    _enf("t01-a05-d1", A05, 1, role="THE BUDGET CURVE"),
    _enf("t01-a05-d2", A05, 2, role="THE BUDGET CURVE"),
    _enf("t01-a05-d3", A05, 3, role="THE BUDGET CURVE"),
    _enf("t01-a05-d5", A05, 5, role="THE BUDGET CURVE"),
    _enf("t01-a05-d8", A05, 8, role="THE BUDGET CURVE"),
    _enf("t01-a05-d2-async", A05, 2, "async", role="second curve"),
    _enf("t01-a05-d5-async", A05, 5, "async", role="second curve"),
    _enf("t01-a05-d0-oracle", A05, 0, on_escalate="oracle_resume", envelope=ORACLE_ENVELOPE,
         role="intervention-value counterfactual; exercises the AckToken path"),
    _enf("t01-a05-d0-ackonly", A05, 0, envelope=ACKONLY_ENVELOPE, rearm="ack_only",
         role="protective-stop-vs-e-stop accounting (optional)"),
    _obs("inj-obs-d0", 0, role="THE ENFORCEMENT DEMONSTRATION (observe)", injection=INJECTION_DEMO),
    _enf("inj-t0-d0", None, 0, injection=INJECTION_DEMO, role="THE ENFORCEMENT DEMONSTRATION (enforce)"),
)

_BY_ID = {a.arm_id: a for a in ARMS}


def arm_ids() -> list[str]:
    return [a.arm_id for a in ARMS]


def get_arm(arm_id: str) -> Arm:
    try:
        return _BY_ID[arm_id]
    except KeyError:
        raise KeyError("unknown arm %r (known: %s)" % (arm_id, ", ".join(arm_ids()))) from None


def resolve_arms(patterns) -> list[Arm]:
    """`--arms` value(s): comma-separated ids or fnmatch globs (`t01-a05-d*`), table order, de-duplicated."""
    if isinstance(patterns, str):
        patterns = [patterns]
    wanted = []
    for p in patterns:
        for part in str(p).split(","):
            part = part.strip()
            if part:
                wanted.append(part)
    if not wanted:
        raise ValueError("no arms given")
    out = []
    for pat in wanted:
        hits = [a for a in ARMS if fnmatch.fnmatchcase(a.arm_id, pat)]
        if not hits:
            raise KeyError("no arm matches %r (known: %s)" % (pat, ", ".join(arm_ids())))
        for a in hits:
            if a not in out:
                out.append(a)
    return out


def with_overrides(arm: Arm, **kw) -> Arm:
    """A copy with fields replaced (e.g. `--on-escalate`, `--envelope`)."""
    return replace(arm, **kw)


def table() -> str:
    """The 10.5 table for `run.py --list-arms`."""
    lines = ["%-20s %-8s %-6s %-6s %-7s %2s %-5s %-15s %s" % ("arm_id", "mode", "tier0", "tier1", "alpha", "d", "exec",
                                                             "on_escalate", "role")]
    for a in ARMS:
        alpha = "--" if a.alpha is None else "%d/%d" % a.alpha
        lines.append("%-20s %-8s %-6s %-6s %-7s %2d %-5s %-15s %s" % (a.arm_id, a.mode, a.tier0, str(a.tier1).lower(),
                                                                     alpha, a.d, a.exec, a.on_escalate, a.role))
    return "\n".join(lines)
