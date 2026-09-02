# SPDX-License-Identifier: MIT
"""Python client for `lictor serve` (wire protocol `lictor-wire/v1`, see docs/wire-protocol.md).

One `LictorClient` owns one `lictor serve` child: requests go to its stdin as NDJSON, responses come back on its
stdout, one line each, strictly alternating, with a monotone `id`. The harness runs one client per env slot.

Fail-closed host contract (docs/wire-protocol.md): a read timeout, a child exit, an `id` mismatch or a fatal error
from a dead child KILLS the child, marks the client dead and raises `LictorFault` carrying
`last_safe_action = safe_action()` (the caller-supplied `clamp_box(current measured position)` callback) -- never a
stale setpoint and never the raw policy action. A fatal `error` from a LIVE child is returned as the response dict
so the harness can still send `episode_end(ended_by="fault")` and get a receipt.

Numbers: every numeric field passes through `finite_or_none` (non-finite -> None -> JSON null) and every request is
serialised with `json.dumps(..., allow_nan=False, separators=(",", ":"))`, so the non-JSON literal `NaN` can never
reach the wire. In responses exactly two fields may be `null` by design and are mapped back: `scores.s` -> -inf and
`tau` -> +inf (also `hello_ok.calibration.tau`). Nothing else is touched.

Stdlib + numpy only. Runs under WSL (the `select`-based timeout needs a POSIX pipe); a Windows-native binary is
auto-detected as a last resort.
"""

from __future__ import annotations

import glob
import json
import math
import os
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

CLIENT_VERSION = "lictor_client/0.1.0"  # sent as hello.client; bound into every receipt as body.client
PROTO = "lictor-wire/v1"
MAX_LINE = 1 << 20

__all__ = [
    "CLIENT_VERSION",
    "PROTO",
    "LictorFault",
    "LictorClient",
    "chunk_msg",
    "finite_or_none",
    "find_binary",
    "win_to_wsl",
    "wsl_to_win",
]


class LictorFault(Exception):
    """Raised on timeout / fatal error / child exit / id mismatch. The child has been KILLED and the client is dead."""

    def __init__(self, reason: str, message: str = "", last_safe_action=None) -> None:
        super().__init__("%s: %s" % (reason, message) if message else reason)
        self.reason = reason  # "timeout" | "fatal" | "exit" | "id_mismatch" | "protocol"
        self.last_safe_action = last_safe_action  # clamp_box(current measured position) from `safe_action`, or None
        self.message = message


# ---- path bridging (bulla_mcp.py style) -------------------------------------------------------------------------


def win_to_wsl(path) -> str:
    """`C:\\Users\\x\\f.txt` -> `/mnt/c/Users/x/f.txt`; anything already POSIX is returned unchanged."""
    s = str(path)
    if len(s) >= 2 and s[1] == ":" and s[0].isalpha():
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = "/" + rest
        return "/mnt/%s%s" % (drive, rest)
    return s.replace("\\", "/")


def wsl_to_win(path) -> str:
    """`/mnt/c/Users/x/f.txt` -> `C:\\Users\\x\\f.txt`; anything else is returned unchanged."""
    s = str(path)
    if s.startswith("/mnt/") and len(s) >= 6 and s[5].isalpha() and (len(s) == 6 or s[6] == "/"):
        return "%s:%s" % (s[5].upper(), s[6:].replace("/", "\\") or "\\")
    return s


def _exists(p) -> bool:
    try:
        return Path(p).is_file()
    except OSError:
        return False


def find_binary(explicit=None) -> str:
    """Binary discovery order: explicit arg, $LICTOR_BIN, $CARGO_TARGET_DIR/release/lictor,
    /mnt/d/lictor/target/release/lictor, `lictor` on PATH, a lictor.exe under /mnt/c/... (insurance only)."""
    candidates = []
    if explicit:
        candidates.append(str(explicit))
    if os.environ.get("LICTOR_BIN"):
        candidates.append(os.environ["LICTOR_BIN"])
    if os.environ.get("CARGO_TARGET_DIR"):
        candidates.append(os.path.join(os.environ["CARGO_TARGET_DIR"], "release", "lictor"))
    candidates.append("/mnt/d/lictor/target/release/lictor")
    on_path = shutil.which("lictor")
    if on_path:
        candidates.append(on_path)
    for c in candidates:
        if _exists(c):
            return c
    for pattern in ("/mnt/c/Users/*/Desktop/Robots/lictor/target/release/lictor.exe", "/mnt/[a-z]/lictor/target/release/lictor.exe"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    raise FileNotFoundError("lictor binary not found (tried: %s); set $LICTOR_BIN or pass binary=" % ", ".join(candidates))


# ---- number hygiene -----------------------------------------------------------------------------------------------


def _mask_to_none(arr: np.ndarray):
    """float64 array -> nested lists with None where not finite (shape preserved)."""
    if arr.ndim == 0:
        x = float(arr)
        return x if math.isfinite(x) else None
    out = []
    for row in arr:
        out.append(_mask_to_none(row))
    return out


def finite_or_none(x) -> list:
    """`np.asarray(x, float64)` -> nested Python lists; every non-finite entry becomes None (JSON null).
    Used by every numeric field of every request. A scalar returns a 0-d result (float or None)."""
    arr = np.asarray(x, dtype=np.float64)
    return _mask_to_none(arr)


def _finite_scalar(x):
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def chunk_msg(seq: int, t_emit: int, a: np.ndarray, exec_steps: int) -> dict:
    """a: (h, d) float64 -> {"seq","t_emit","h","d","exec","a"} with NaN -> None. The fuse always receives the
    FULL chunk (h == envelope.embodiment.horizon); the harness never trims rows."""
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("chunk must be a 2-D (h, d) array, got shape %r" % (arr.shape,))
    h, d = arr.shape
    return {"seq": int(seq), "t_emit": int(t_emit), "h": int(h), "d": int(d), "exec": int(exec_steps), "a": finite_or_none(arr)}


def _dumps(obj) -> str:
    return json.dumps(obj, allow_nan=False, separators=(",", ":"))


# ---- the client ---------------------------------------------------------------------------------------------------


class LictorClient:
    """One `lictor serve` child; see the module docstring for the fail-closed contract."""

    def __init__(self, binary, envelope: Path, calibration=None, mode: str = "observe",
                 out_dir=None, trace=None, key=None,
                 tier0=None, no_tier1: bool = False, ticks: str = "tail32", timeout_s: float = 5.0,
                 safe_action=None, latency_label=None) -> None:
        self.binary = find_binary(binary)
        self.timeout_s = float(timeout_s)
        self.safe_action = safe_action
        self._id = 0
        self._alive = False
        self.hello_ok = None
        self.last_error = None
        argv = [self.binary, "serve", "--envelope", str(envelope), "--mode", mode, "--ticks", ticks]
        if calibration is not None:
            argv += ["--calibration", str(calibration)]
        if out_dir is not None:
            argv += ["--out", str(out_dir)]
        if trace is not None:
            argv += ["--trace", str(trace)]
        if key is not None:
            argv += ["--key", str(key)]
        if tier0 is not None:
            argv += ["--tier0", str(tier0)]
        if no_tier1:
            argv.append("--no-tier1")
        if latency_label is not None:
            argv += ["--latency-label", str(latency_label)]
        self.argv = argv
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # diagnostics stay on the harness's stderr
            bufsize=1,  # line-buffered text pipes
            text=True,
            encoding="utf-8",
        )
        self._alive = True

    # -- lifecycle --

    @property
    def alive(self) -> bool:
        return self._alive and self.proc.poll() is None

    def close(self) -> None:
        """Send `bye` when the child is alive, then wait; kill on any trouble."""
        if self._alive and self.proc.poll() is None:
            try:
                self._send({"kind": "bye"})
                self._read_line(self.timeout_s)
            except Exception:
                pass
        self._kill()

    def _kill(self) -> None:
        self._alive = False
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)
            except Exception:
                pass
        for s in (self.proc.stdin, self.proc.stdout):
            try:
                if s is not None:
                    s.close()
            except Exception:
                pass

    def _fault(self, reason: str, message: str) -> LictorFault:
        self._kill()
        safe = None
        if self.safe_action is not None:
            try:
                safe = [float(x) for x in np.asarray(self.safe_action(), dtype=np.float64).ravel().tolist()]
            except Exception:
                safe = None
        return LictorFault(reason, message, last_safe_action=safe)

    # -- transport --

    def _send(self, obj: dict) -> int:
        self._id += 1
        obj = dict(obj)
        obj["id"] = self._id
        line = _dumps(obj)
        if "\n" in line or "\r" in line:
            raise self._fault("protocol", "request contains a newline")
        if len(line.encode("utf-8")) > MAX_LINE:
            raise self._fault("protocol", "request exceeds MAX_LINE")
        try:
            self.proc.stdin.write(line + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as e:
            raise self._fault("exit", "child stdin closed: %s" % e)
        return self._id

    def _read_line(self, timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        fd = self.proc.stdout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._fault("timeout", "no response within %.1f s" % timeout_s)
            try:
                ready, _, _ = select.select([fd], [], [], remaining)
            except (ValueError, OSError) as e:
                raise self._fault("exit", "child stdout unreadable: %s" % e)
            if not ready:
                continue
            line = fd.readline()
            if line == "":
                code = self.proc.poll()
                raise self._fault("exit", "child closed stdout (exit code %r)" % code)
            return line.rstrip("\n")

    def _request(self, obj: dict) -> dict:
        if not self.alive:
            raise self._fault("exit", "client is dead")
        sent = self._send(obj)
        line = self._read_line(self.timeout_s)
        try:
            resp = json.loads(line)
        except ValueError as e:
            raise self._fault("protocol", "response is not JSON: %s (%r)" % (e, line[:200]))
        if not isinstance(resp, dict) or resp.get("id") != sent:
            raise self._fault("id_mismatch", "sent id %d, got %r" % (sent, resp.get("id") if isinstance(resp, dict) else resp))
        if resp.get("kind") == "error":
            self.last_error = resp
            if resp.get("fatal") and self.proc.poll() is not None:
                raise self._fault("fatal", "%s: %s" % (resp.get("code"), resp.get("message")))
        return resp

    # -- messages --

    def hello(self, embodiment_id: str, action_dim: int, pos_dim: int, horizon: int, exec_steps: int,
              envelope_digest: str, calibration_digest, expect_tier1: bool, expect_alpha) -> dict:
        """Cross-checks the served artefacts; asserts `tier1_armed == expect_tier1` and the calibration alpha ==
        `expect_alpha` (a `(num, den)` tuple or None) and raises LictorFault(reason="protocol") otherwise."""
        resp = self._request({
            "kind": "hello", "proto": PROTO, "client": CLIENT_VERSION,
            "mode": self.argv[self.argv.index("--mode") + 1],
            "embodiment_id": embodiment_id, "action_dim": int(action_dim), "pos_dim": int(pos_dim),
            "horizon": int(horizon), "exec_steps": int(exec_steps),
            "envelope_digest": envelope_digest, "calibration_digest": calibration_digest,
        })
        if resp.get("kind") != "hello_ok":
            raise self._fault("protocol", "hello refused: %s" % _dumps(resp))
        cal = resp.get("calibration")
        if cal is not None and cal.get("tau") is None:
            cal["tau"] = math.inf
        if bool(resp.get("tier1_armed")) != bool(expect_tier1):
            raise self._fault("protocol", "tier1_armed=%r but the arm expects %r" % (resp.get("tier1_armed"), expect_tier1))
        got_alpha = (cal["alpha_num"], cal["alpha_den"]) if cal is not None else None
        want_alpha = tuple(expect_alpha) if expect_alpha is not None else None
        if got_alpha != want_alpha:
            raise self._fault("protocol", "calibration alpha %r but the arm expects %r" % (got_alpha, want_alpha))
        self.hello_ok = resp
        return resp

    def episode_begin(self, run: dict, budget: dict, binding: dict, fault_injection=None, inputs=None) -> dict:
        msg = {"kind": "episode_begin", "run": run, "budget": budget, "binding": binding, "fault_injection": fault_injection}
        if inputs:
            msg["inputs"] = {str(k): str(v) for k, v in inputs.items()}
        return self._request(msg)

    def tick(self, t: int, idx: int, pos, aux, chunk=None, vel=None, ext=(), missed_ticks: int = 0, ack=None) -> dict:
        """One env step. Maps `null -> -inf` for `scores.s` and `+inf` for `tau`; nothing else."""
        obs = {
            "pos": finite_or_none(pos),
            "vel": None if vel is None else finite_or_none(vel),
            "aux": finite_or_none(aux) if aux is not None else [],
            "ext": finite_or_none(ext) if ext is not None and len(ext) > 0 else [],
        }
        resp = self._request({"kind": "tick", "t": int(t), "idx": int(idx), "missed_ticks": int(missed_ticks),
                              "obs": obs, "chunk": chunk, "ack": ack})
        if resp.get("kind") == "verdict":
            sc = resp.get("scores")
            if isinstance(sc, dict) and sc.get("s") is None:
                sc["s"] = -math.inf
            if resp.get("tau") is None:
                resp["tau"] = math.inf
        return resp

    def episode_end(self, outcome: dict) -> dict:
        o = dict(outcome)
        for k in ("max_coverage", "final_coverage", "reward_sum"):
            o[k] = _finite_scalar(o.get(k))
        t = int(o.get("steps", 0))
        return self._request({"kind": "episode_end", "t": t, "outcome": o})


def _main(argv) -> int:  # pragma: no cover - a tiny manual smoke path
    """python -m adapters.lictor_client ENVELOPE.toml DIGEST -> hello/bye against the discovered binary."""
    if len(argv) != 3:
        print("usage: lictor_client.py ENVELOPE.toml ENVELOPE_DIGEST", file=sys.stderr)
        return 2
    c = LictorClient(None, Path(argv[1]))
    try:
        print(_dumps(c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, argv[2], None, False, None)))
    finally:
        c.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv))
