# SPDX-License-Identifier: MIT
"""Fault injection (ARCHITECTURE 10.4): five injectors on an independent `numpy.random.Philox` stream keyed by
`episode_seed ^ 0xFA17`, so injection never perturbs the policy's noise and is reproducible per episode.

Every injector returns the `fault_injection` binding sent in `episode_begin` (`kind`, string-valued `params`,
`stream_seed`); the receipt binds the host's DECLARATION of the injection (trust model: a host that injects and
declares `null` hides it). Hooks, all deterministic in call order:

* `chunk(rows, t)`   -- `action_spike(p, mag)`: every row independently, with probability p, is displaced by `mag`
                        px in a uniform random direction (a Tier-0 workspace/speed/reach violation candidate);
                        `chunk_truncate(p)`: with probability p the chunk is cut at a random row and the last valid
                        row is repeated to the horizon (the wire always carries the full 15 rows).
* `obs(pos, vel, aux, t)` -- `obs_dropout(p)`: with probability p the observation sent to the fuse is the previous
                        tick's (a stale sensor frame); `obs_noise(sigma)`: N(0, sigma) px on `pos`.
* `delay(t_req, k)`  -- `latency_spike(p, ticks)`: with probability p a request is delayed by `ticks` extra steps.

Spec strings: `action_spike(p=0.02,mag=180)`, `obs_noise(sigma=4)`, `latency_spike(p=0.1,ticks=3)` ...
"""

from __future__ import annotations

import re

import numpy as np

STREAM_KEY_XOR = 0xFA17

KINDS = ("action_spike", "chunk_truncate", "obs_dropout", "obs_noise", "latency_spike")

_DEFAULTS = {
    "action_spike": {"p": 0.02, "mag": 180.0},
    "chunk_truncate": {"p": 0.05},
    "obs_dropout": {"p": 0.05},
    "obs_noise": {"sigma": 4.0},
    "latency_spike": {"p": 0.1, "ticks": 3},
}


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)


class Injector:
    """One injector for one episode. Subclasses override the hook they act on; the others are identity."""

    kind = ""

    def __init__(self, episode_seed: int, **params) -> None:
        self.episode_seed = int(episode_seed)
        self.stream_seed = self.episode_seed ^ STREAM_KEY_XOR
        p = dict(_DEFAULTS[self.kind])
        for k, v in params.items():
            if k not in p:
                raise ValueError("%s: unknown parameter %r" % (self.kind, k))
            p[k] = type(p[k])(v)
        self.params = p
        self.rng = np.random.Generator(np.random.Philox(key=self.stream_seed))
        self.events = 0  # how many times the injector actually fired

    def binding(self) -> dict:
        return {"kind": self.kind, "params": {k: _fmt(v) for k, v in sorted(self.params.items())},
                "stream_seed": self.stream_seed}

    # hooks (identity by default)
    def chunk(self, rows: np.ndarray, t: int) -> np.ndarray:
        return rows

    def obs(self, pos, vel, aux, t: int, prev=None):
        return pos, vel, aux

    def delay(self, t_req: int, k: int) -> int:
        return 0


class ActionSpike(Injector):
    kind = "action_spike"

    def chunk(self, rows, t):
        rows = np.array(rows, dtype=np.float64, copy=True)
        h, d = rows.shape
        hits = self.rng.random(h) < self.params["p"]
        for i in np.flatnonzero(hits):
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            direction = np.zeros(d)
            direction[0] = np.cos(theta)
            if d > 1:
                direction[1] = np.sin(theta)
            rows[i] = rows[i] + self.params["mag"] * direction
            self.events += 1
        return rows


class ChunkTruncate(Injector):
    kind = "chunk_truncate"

    def chunk(self, rows, t):
        rows = np.array(rows, dtype=np.float64, copy=True)
        if self.rng.random() < self.params["p"]:
            h = rows.shape[0]
            cut = int(self.rng.integers(1, h))  # keep rows 0..cut-1, repeat row cut-1
            rows[cut:] = rows[cut - 1]
            self.events += 1
        return rows


class ObsDropout(Injector):
    kind = "obs_dropout"

    def obs(self, pos, vel, aux, t, prev=None):
        if prev is not None and self.rng.random() < self.params["p"]:
            self.events += 1
            return prev
        return pos, vel, aux


class ObsNoise(Injector):
    kind = "obs_noise"

    def obs(self, pos, vel, aux, t, prev=None):
        pos = np.asarray(pos, dtype=np.float64) + self.rng.normal(0.0, self.params["sigma"], size=np.shape(pos))
        self.events += 1
        return pos, vel, aux


class LatencySpike(Injector):
    kind = "latency_spike"

    def delay(self, t_req, k):
        if self.rng.random() < self.params["p"]:
            self.events += 1
            return int(self.params["ticks"])
        return 0


_CLASSES = {c.kind: c for c in (ActionSpike, ChunkTruncate, ObsDropout, ObsNoise, LatencySpike)}

_SPEC_RE = re.compile(r"^\s*([a-z_]+)\s*(?:\((.*)\))?\s*$")


def parse_spec(spec: str) -> tuple[str, dict]:
    """`"action_spike(p=0.02,mag=180)"` -> ("action_spike", {"p": "0.02", "mag": "180"}); `"obs_noise"` -> defaults."""
    m = _SPEC_RE.match(spec or "")
    if not m:
        raise ValueError("bad injection spec %r" % spec)
    kind, args = m.group(1), m.group(2)
    if kind not in _CLASSES:
        raise ValueError("unknown injection kind %r (known: %s)" % (kind, ", ".join(KINDS)))
    params = {}
    if args:
        for part in args.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError("injection parameter %r is not k=v" % part)
            k, v = part.split("=", 1)
            params[k.strip()] = v.strip()
    return kind, params


def make_injector(spec, episode_seed: int):
    """None for no injection, else the Injector for `spec` and this episode."""
    if spec is None or spec == "" or spec == "none":
        return None
    kind, params = parse_spec(spec)
    return _CLASSES[kind](episode_seed, **params)
