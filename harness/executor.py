# SPDX-License-Identifier: MIT
"""Delayed chunk executor: how "added latency" is realised in simulation (ARCHITECTURE 10.4).

Pure Python, no torch, deterministic: a pure function of the chunks it is given. One executor per env slot.

Time model (one tick == one `env.step`, 100 ms of simulated time on PushT):

* `sync`  -- a chunk REQUESTED at tick `t_req` (computed from the observation at `t_req`) becomes available at
  `t_req + d`. Rows 0..exec_steps-1 execute on the ticks after arrival; the next request is issued when the
  executed prefix is exhausted; until that chunk arrives the LAST EXECUTED action is repeated (hold-last) and the
  delay therefore consumes the 300-step budget. Wire labelling at delivery: `t_emit = t`, `idx = 0`.
* `async` -- the executor keeps draining the previous chunk's spare rows (rows exec_steps..horizon-1, i.e. 7 more
  on PushT, so `d <= 7` never underruns) while the request is in flight. On arrival, `stitch = "drop"` skips the new
  chunk's first `d` rows (they refer to ticks that already happened) and executes row `d` now -- wire labelling
  `t_emit = t_req`, `idx = d`; `stitch = "freeze"` executes the new chunk from row 0 anyway (RTC's freeze) --
  wire labelling `t_emit = t`, `idx = 0`. Past the spare rows the executor holds the last executed action.

Hold-last ticks are labelled with the index of the row being repeated (`idx = exec_steps - 1` in sync mode,
`horizon - 1` on an async underrun): the fuse re-checks exactly the action that reaches the environment, and the
verdict's pass-through action equals the executor's action, which is what the observe-mode identity assert needs.

The FIRST chunk of an episode is always available at `t = 0` (its request is fulfilled at `t = 0` from the reset
observation): the wire protocol requires a chunk on the first tick (a non-delivery tick with nothing to index is a
schema fault) and the environment is stationary before its first step, so a pre-episode delay would only shift the
budget without changing any state. Every LATER request is delayed by `d` (plus any injected latency spike).

At most one request is in flight; when `d` exceeds the cadence (`d >= exec_steps`, async) the next request is
issued on the tick after the arrival, so the executor never pipelines two inferences.

Driving: either pass `policy_fn(t) -> rows` (called at request time, unit-test style) or leave it None and drive
`needs_chunk(t)` / `fulfil(rows)` before every `tick(t)` (the batched rollout computes all slots' chunks in one
policy forward and fulfils each executor).
"""

from __future__ import annotations

from collections import namedtuple

Tick = namedtuple("Tick", "t idx chunk t_emit seq action src")
"""One executor tick: `chunk` is the full row list on a delivery tick and None otherwise; `t_emit`/`seq` label the
chunk on the wire (seq of the CURRENT chunk on non-delivery ticks); `action` is the row executed this tick;
`src` is "policy" (row < exec_steps of the current chunk), "drain" (a spare row, async) or "hold" (hold-last)."""

EXEC_MODES = ("sync", "async")
STITCHES = ("drop", "freeze")


class DelayedExecutor:
    """See the module docstring."""

    def __init__(self, policy_fn=None, d: int = 0, exec_mode: str = "sync", stitch: str = "drop",
                 exec_steps: int = 8, horizon: int = 15, initial_action=None, extra_delay_fn=None) -> None:
        if exec_mode not in EXEC_MODES:
            raise ValueError("exec_mode must be one of %r" % (EXEC_MODES,))
        if stitch not in STITCHES:
            raise ValueError("stitch must be one of %r" % (STITCHES,))
        if d < 0:
            raise ValueError("d must be >= 0")
        if exec_steps < 1 or horizon < exec_steps:
            raise ValueError("need 1 <= exec_steps <= horizon")
        if exec_mode == "async" and stitch == "drop" and d >= horizon:
            raise ValueError("async drop with d >= horizon would skip the whole chunk")
        self.policy_fn = policy_fn
        self.d = int(d)
        self.exec_mode = exec_mode
        self.stitch = stitch
        self.exec_steps = int(exec_steps)
        self.horizon = int(horizon)
        self.initial_action = initial_action
        self.extra_delay_fn = extra_delay_fn
        # current chunk
        self.cur = None
        self.cur_emit = 0
        self.cur_seq = -1
        self.exec_pos = 0
        # request bookkeeping
        self.next_req_t = 0
        self._req_rows = None  # rows supplied for the request due at next_req_t
        self._inflight = None  # (arrival_t, req_t, rows)
        self.chunks_requested = 0
        self.chunks_delivered = 0
        self.hold_ticks = 0
        self.last_tick = None

    # -- request side --

    def needs_chunk(self, t: int) -> bool:
        """True when a chunk request is due at tick `t` and has not been fulfilled yet."""
        return self._inflight is None and self.next_req_t == t and self._req_rows is None

    def fulfil(self, rows) -> None:
        """Supply the rows for the pending request (the policy output computed from the observation at request time)."""
        if self._req_rows is not None:
            raise RuntimeError("request already fulfilled")
        rows = [list(r) for r in rows]
        if len(rows) != self.horizon:
            raise ValueError("chunk has %d rows, executor expects %d" % (len(rows), self.horizon))
        self._req_rows = rows

    def delay_for(self, t_req: int) -> int:
        """Delay applied to the request issued at `t_req`: 0 for the first chunk of the episode, else d (+ spike)."""
        if self.chunks_requested == 0:
            return 0
        extra = 0
        if self.extra_delay_fn is not None:
            extra = int(self.extra_delay_fn(t_req, self.chunks_requested))
        return self.d + max(0, extra)

    # -- tick --

    def tick(self, t: int) -> Tick:
        if self.last_tick is not None and t != self.last_tick + 1:
            raise RuntimeError("ticks must be consecutive (got %d after %d)" % (t, self.last_tick))
        if self.last_tick is None and t != 0:
            raise RuntimeError("the first tick must be t=0")
        self.last_tick = t
        if self.policy_fn is not None and self.needs_chunk(t):
            self.fulfil(self.policy_fn(t))
        if self._inflight is None and self.next_req_t == t and self._req_rows is None:
            raise RuntimeError("chunk for the request at t=%d was not supplied" % t)

        delivered = None
        # (a0) a chunk arriving now (requested on an earlier tick)
        if self._inflight is not None and self._inflight[0] == t:
            delivered = self._deliver(t)
        # (b) move the fulfilled request into flight
        if self._req_rows is not None and self.next_req_t == t and self._inflight is None:
            delay = self.delay_for(t)
            self._inflight = (t + delay, t, self._req_rows)
            self._req_rows = None
            self.chunks_requested += 1
            # (a1) immediate arrival (d == 0 or the first chunk)
            if self._inflight[0] == t and delivered is None:
                delivered = self._deliver(t)

        if delivered is not None:
            idx = self.exec_pos
            action = self.cur[idx]
            self.exec_pos = idx + 1
            return Tick(t, idx, [list(r) for r in self.cur], self.cur_emit, self.cur_seq, list(action), "policy")

        if self.cur is None:
            # unreachable in a well-formed episode (the first request is always fulfilled at t = 0)
            if self.initial_action is None:
                raise RuntimeError("no chunk and no initial action")
            self.hold_ticks += 1
            return Tick(t, 0, None, 0, -1, list(self.initial_action), "hold")

        limit = self.exec_steps if self.exec_mode == "sync" else self.horizon
        if self.exec_pos < limit:
            idx = self.exec_pos
            self.exec_pos = idx + 1
            src = "policy" if idx < self.exec_steps else "drain"
            return Tick(t, idx, None, self.cur_emit, self.cur_seq, list(self.cur[idx]), src)
        idx = limit - 1
        self.hold_ticks += 1
        return Tick(t, idx, None, self.cur_emit, self.cur_seq, list(self.cur[idx]), "hold")

    def _deliver(self, t: int):
        arrival, req_t, rows = self._inflight
        self._inflight = None
        self.cur = rows
        self.cur_seq += 1
        self.chunks_delivered += 1
        if self.exec_mode == "sync" or self.stitch == "freeze":
            self.cur_emit = t
            self.exec_pos = 0
        else:
            self.cur_emit = req_t
            self.exec_pos = t - req_t
        if self.exec_pos >= self.horizon:
            raise RuntimeError("stale chunk: idx %d >= horizon" % self.exec_pos)
        if self.exec_mode == "sync" or self.stitch == "freeze":
            self.next_req_t = t + self.exec_steps
        else:
            self.next_req_t = max(self.cur_emit + self.exec_steps, t + 1)
        return True


def run_executor(policy_fn, n_ticks: int, **kw) -> list:
    """Convenience for tests: drive an executor for `n_ticks` ticks and return the Tick list."""
    ex = DelayedExecutor(policy_fn, **kw)
    return [ex.tick(t) for t in range(n_ticks)]
