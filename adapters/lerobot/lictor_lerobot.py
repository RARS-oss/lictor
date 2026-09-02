# SPDX-License-Identifier: MIT
"""STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12.

lictor as a LeRobot v0.6 `ProcessorStep` (the `env_postprocessor` slot -- the slot whose upstream docs show a
`torch.clamp` "safety limits" example) plus a `Fuse` convenience wrapper over `adapters.lictor_client.LictorClient`.

The ten-line adoption story (LIBERO + `lerobot/smolvla_libero` is roadmap item 3 of docs/ARCHITECTURE.md sec 12;
the envelope for it does not exist yet, so the story is a shape, not a measurement):

    from adapters.lerobot.lictor_lerobot import Fuse, LictorObserveStep, LictorStep
    fuse = Fuse()                                                    # finds the `lictor` binary
    fuse.start("envelopes/libero.toml", calibration=None, out="results/libero", mode="enforce")
    env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
    env_pre = DataProcessorPipeline(steps=[*env_pre.steps, LictorObserveStep(fuse)])   # records the state
    env_post = DataProcessorPipeline(steps=[*env_post.steps, LictorStep(fuse, policy=policy)])  # fuses the action
    fuse.begin(episode=0, seed=0, policy=policy, env=env)           # once per episode, after env.reset(seed)
    ...                                                              # the stock eval loop, unchanged
    fuse.end(success=bool(info["is_success"]), steps=t, progress=float(reward))   # -> signed receipt + ledger line
    fuse.close()

What this file IS: the frozen wire client driven from inside a LeRobot pipeline. Every wire rule of
docs/ARCHITECTURE.md sec 9 is honoured by construction: one `tick` per env step, the full chunk on the delivery tick
only, `verdict.action` and nothing else is written back into the transition, the child's death or a fatal error
becomes `clamp_box(current measured position)` (position kinds) or the zero vector (velocity kinds), never a stale
setpoint and never the raw policy action.

What this file is NOT: validated. No LIBERO envelope has been fitted, no calibration has been collected through this
step, and lerobot 0.6.1's `lerobot-eval` has no flag that injects a custom step (a ~100-line custom eval script that
reuses `make_env`, `make_policy`, `make_pre_post_processors` and `make_env_pre_post_processors` is the intended
host). Three simplifications are documented rather than hidden:

  1. Chunk discovery reads `policy._queues[ACTION]`: a new chunk is assumed to have been planned on the step whose
     popped action leaves exactly `n_action_steps - 1` rows queued (the stock `select_action` cadence). Pass
     `chunk_fn` to supply the chunk explicitly (e.g. from `predict_action_chunk`) and nothing private is touched.
  2. The transition handed to the `env_postprocessor` by `lerobot_eval.py` carries ONLY the action, so the state
     reaches the fuse through `LictorObserveStep` (an `env_preprocessor` step that records `observation.state`)
     or through `state_fn`. A step without a recorded state raises instead of ticking with a guessed position.
  3. One fuse serves one episode of one environment (`B == 1`). Vectorised evaluation (`n_envs > 1`) needs one
     `Fuse` per env slot -- docs/VERIFIED_FACTS.md measures the B=32 batched forward; multiplexing episodes over a
     single child is a wire change that milestone 1 does not make.

Nothing here imports `lerobot`, `torch` or `numpy` at module import time. When the host process has ALREADY imported
`lerobot.processor`, `LictorStep` derives from the real `ProcessorStep` ABC (so `isinstance` checks pass); otherwise
it derives from a duck-typed base with the same surface. `LictorStep.bind_lerobot()` imports `lerobot.processor` on
demand and returns a registered subclass for hosts that save/load pipelines.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from typing import Any, Callable, Sequence

ADAPTER_VERSION = "lictor_lerobot/0.1.0"
ADAPTER_STATUS = "UNVERIFIED"
ZERO_DIGEST = "0" * 64
POSITION_KINDS = ("ee_position", "joint_position")
VELOCITY_KINDS = ("joint_velocity", "ee_delta")
MODES = ("observe", "enforce")
ENDED_BY = ("success", "truncated", "escalation_terminate", "fault", "abort", "retune")
ACTION_KEY = "action"           # == lerobot.utils.constants.ACTION == TransitionKey.ACTION (a str enum)
OBSERVATION_KEY = "observation"  # == TransitionKey.OBSERVATION
STATE_KEY = "observation.state"  # == lerobot.utils.constants.OBS_STATE


class LictorAdapterError(RuntimeError):
    """A host-contract violation detected on the Python side (before anything reaches the wire)."""


# ----------------------------------------------------------------------------------------------------------------
# small conversions (no numpy / torch imports; both are handled by duck typing when they show up)
# ----------------------------------------------------------------------------------------------------------------


def _is_torch(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _is_numpy(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "numpy"


def _to_nested(x: Any) -> Any:
    """torch / numpy / sequences -> nested Python lists (scalars stay scalars)."""
    if _is_torch(x):
        x = x.detach().cpu().numpy()
    if _is_numpy(x) or hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [_to_nested(v) for v in x]
    return x


def as_f64_row(x: Any) -> list[float]:
    """One action / position row as Python floats. A leading batch axis of length 1 is squeezed; B > 1 raises."""
    v = _to_nested(x)
    if isinstance(v, (int, float, bool)):
        return [float(v)]
    if not isinstance(v, list):
        raise LictorAdapterError(f"cannot read a row from {type(x).__name__}")
    while len(v) == 1 and isinstance(v[0], list):
        v = v[0]
    if any(isinstance(e, list) for e in v):
        raise LictorAdapterError("batched actions (B > 1) are not supported: one Fuse per environment slot")
    return [float(e) for e in v]


def as_f64_rows(a: Any) -> list[list[float]]:
    """An (h, d) chunk as a list of rows. A leading batch axis of length 1 is squeezed; B > 1 raises."""
    v = _to_nested(a)
    if not isinstance(v, list) or not v:
        raise LictorAdapterError("chunk must be a non-empty (h, d) array")
    if isinstance(v[0], list) and v[0] and isinstance(v[0][0], list):
        if len(v) != 1:
            raise LictorAdapterError("batched chunks (B > 1) are not supported: one Fuse per environment slot")
        v = v[0]
    rows = [as_f64_row(r) for r in v]
    d = len(rows[0])
    if d == 0 or any(len(r) != d for r in rows):
        raise LictorAdapterError("chunk rows must all have the same non-zero width")
    return rows


def finite_or_none(x: Any) -> Any:
    """Non-finite reals become None (JSON null) so a NaN can never leak as the non-JSON literal `NaN`."""
    if isinstance(x, bool) or x is None or isinstance(x, (str, int)):
        return x
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (list, tuple)):
        return [finite_or_none(v) for v in x]
    return x


def restore_like(template: Any, values: Sequence[float]) -> Any:
    """Return `values` in the container type/dtype/device/shape of `template` (the policy's own action object)."""
    vals = [float(v) for v in values]
    if _is_torch(template):
        torch = importlib.import_module("torch")
        return torch.as_tensor(vals, dtype=template.dtype, device=template.device).reshape(template.shape)
    if _is_numpy(template):
        np = importlib.import_module("numpy")
        return np.asarray(vals, dtype=template.dtype).reshape(template.shape)
    if isinstance(template, tuple):
        return tuple(vals)
    if isinstance(template, list) and template and isinstance(template[0], list):
        return [vals]
    return vals


def clamp_box(pos: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> list[float]:
    """The host-side brake for position kinds: the current measured position clamped into the box."""
    out = []
    for i, p in enumerate(pos):
        a, b = float(lo[i]), float(hi[i])
        p = float(p)
        if not math.isfinite(p):
            p = 0.5 * (a + b)
        out.append(a if p < a else b if p > b else p)
    return out


def chunk_message(seq: int, t_emit: int, rows: Sequence[Sequence[float]], exec_steps: int) -> dict:
    """The frozen wire chunk `{"seq","t_emit","h","d","exec","a"}` with NaN -> null (mirrors lictor_client.chunk_msg)."""
    a = [[finite_or_none(float(v)) for v in r] for r in rows]
    return {"seq": int(seq), "t_emit": int(t_emit), "h": len(a), "d": len(a[0]), "exec": int(exec_steps), "a": a}


def sha256_file(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def init_state_digest_of(values: Sequence[float]) -> str:
    """sha256 of the float64 little-endian bytes of the post-reset state (the wire's `init_state_digest`)."""
    import struct

    vals = [float(v) for v in as_f64_row(list(values))]
    return hashlib.sha256(struct.pack("<%dd" % len(vals), *vals)).hexdigest()


def _is_hex64(s: Any) -> bool:
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


# ----------------------------------------------------------------------------------------------------------------
# binary discovery and the envelope
# ----------------------------------------------------------------------------------------------------------------


def find_lictor_binary(explicit: os.PathLike | str | None = None) -> str | None:
    """The frozen discovery order of adapters/lictor_client.py: explicit, $LICTOR_BIN, $CARGO_TARGET_DIR/release/lictor,
    /mnt/d/lictor/target/release/lictor, `lictor` on PATH. Returns None when nothing exists."""
    client = _try_import("adapters.lictor_client")
    finder = getattr(client, "find_binary", None) if client is not None else None
    if callable(finder):
        try:
            found = finder(explicit)
            if found:
                return str(found)
        except Exception:  # noqa: BLE001 - the client's finder is allowed to raise; we fall through to our list
            pass
    candidates: list[str] = []
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
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def envelope_digest_via_cli(binary: str, envelope: os.PathLike | str) -> str:
    """`lictor envelope digest <F.toml>` prints the envelope digest on stdout (embodiment digest on stderr)."""
    r = subprocess.run([binary, "envelope", "digest", str(envelope)], capture_output=True, text=True, check=False)
    words = r.stdout.split()
    if r.returncode != 0 or not words or not _is_hex64(words[0]):
        raise LictorAdapterError(f"`lictor envelope digest` failed (exit {r.returncode}): {r.stderr.strip()[:200]}")
    return words[0]


def load_envelope(path: os.PathLike | str) -> dict:
    """Parse the envelope TOML (schema lictor-envelope/v1) and derive what the hello / brake need."""
    with open(path, "rb") as f:
        doc = tomllib.load(f)
    if doc.get("schema") != "lictor-envelope/v1":
        raise LictorAdapterError(f"{path}: schema is {doc.get('schema')!r}, expected 'lictor-envelope/v1'")
    emb = doc.get("embodiment")
    if not isinstance(emb, dict):
        raise LictorAdapterError(f"{path}: missing [embodiment] table")
    for k in ("id", "action_kind", "action_dim", "pos_dim", "horizon", "exec_steps", "control_hz_num", "control_hz_den"):
        if k not in emb:
            raise LictorAdapterError(f"{path}: [embodiment] lacks {k}")
    for k in ("box_lo", "box_hi"):
        if k not in doc:
            raise LictorAdapterError(f"{path}: lacks {k}")
    kind = str(emb["action_kind"])
    if kind not in POSITION_KINDS + VELOCITY_KINDS:
        raise LictorAdapterError(f"{path}: unknown action_kind {kind!r}")
    margin = float(doc.get("margin", 0.0))
    lo = [float(v) + margin for v in doc["box_lo"]]
    hi = [float(v) - margin for v in doc["box_hi"]]
    num, den = int(emb["control_hz_num"]), int(emb["control_hz_den"])
    return {
        "path": str(path),
        "doc": doc,
        "embodiment_id": str(emb["id"]),
        "action_kind": kind,
        "action_dim": int(emb["action_dim"]),
        "pos_dim": int(emb["pos_dim"]),
        "horizon": int(emb["horizon"]),
        "exec_steps": int(emb["exec_steps"]),
        "control_hz": (num, den),
        "tick_ms": max(1, round(1000.0 * den / num)),
        "provides_vel": bool(emb.get("provides_vel", False)),
        "aux_layout": [str(v) for v in emb.get("aux_layout", [])],
        "ext_names": [str(v) for v in emb.get("ext_names", [])],
        "box_lo": lo,
        "box_hi": hi,
    }


def load_calibration(path: os.PathLike | str) -> dict:
    """Read the self-digested calibration JSON: digest and alpha for the hello cross-check."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("schema") != "lictor-calibration/v1":
        raise LictorAdapterError(f"{path}: schema is {doc.get('schema')!r}, expected 'lictor-calibration/v1'")
    digest = doc.get("digest")
    if not _is_hex64(digest):
        raise LictorAdapterError(f"{path}: missing or malformed `digest`")
    return {"path": str(path), "digest": digest, "alpha": (int(doc["alpha_num"]), int(doc["alpha_den"]))}


def _try_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def _default_client_factory(**kw: Any) -> Any:
    client = importlib.import_module("adapters.lictor_client")
    return client.LictorClient(**kw)


def _is_lictor_fault(exc: BaseException) -> bool:
    """`adapters.lictor_client.LictorFault` by identity when importable, by shape otherwise (test doubles)."""
    client = sys.modules.get("adapters.lictor_client")
    fault = getattr(client, "LictorFault", None) if client is not None else None
    if isinstance(fault, type) and isinstance(exc, fault):
        return True
    return hasattr(exc, "reason") and hasattr(exc, "last_safe_action")


def _stringify(d: dict | None) -> dict[str, str]:
    """Binding maps carry strings only (the receipt binds the host's declaration verbatim)."""
    out: dict[str, str] = {}
    for k, v in (d or {}).items():
        if v is None:
            continue
        out[str(k)] = v if isinstance(v, str) else json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else str(v)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Fuse: start / begin / tick / end over the frozen client
# ----------------------------------------------------------------------------------------------------------------


class Fuse:
    """One `lictor serve` child, one episode at a time, driven through the frozen `LictorClient` surface.

    `client_factory(**kwargs)` replaces `adapters.lictor_client.LictorClient(**kwargs)` (tests inject a double; the
    openpi proxy's `--dry-run` injects an echo). With a custom factory and no binary the envelope digest sent in
    `hello` is `ZERO_DIGEST` unless `start(..., envelope_digest=...)` says otherwise -- a real child refuses it, which
    is the intended fail-closed outcome for a mis-wired host.
    """

    def __init__(
        self,
        binary: os.PathLike | str | None = None,
        client_factory: Callable[..., Any] | None = None,
        *,
        timeout_s: float = 5.0,
        latency_label: str | None = None,
        key: os.PathLike | str | None = None,
        tier0: str | None = None,
        no_tier1: bool = False,
        ticks: str = "tail32",
        trace: os.PathLike | str | None = None,
    ) -> None:
        self._binary_arg = binary
        self._factory = client_factory
        self._timeout_s = float(timeout_s)
        self._latency_label = latency_label
        self._key = key
        self._tier0 = tier0
        self._no_tier1 = bool(no_tier1)
        self._ticks = ticks
        self._trace = trace
        self.binary: str | None = None
        self.client: Any = None
        self.env: dict | None = None
        self.calib: dict | None = None
        self.mode: str = "observe"
        self.out: str | None = None
        self.envelope_digest: str | None = None
        self.hello: dict | None = None
        self.run_id: str = time.strftime("%Y-%m-%dT%H-%MZ", time.gmtime()) + "-lerobot"
        self._reset_episode()

    # -- episode bookkeeping ---------------------------------------------------------------------------------------

    def _reset_episode(self) -> None:
        self.open = False
        self.t = 0
        self.next_seq = 0
        self.idx: int | None = None
        self.cur_chunk: list[list[float]] | None = None
        self.faulted = False
        self.dead = False
        self.fault: dict | None = None
        self.last_verdict: dict | None = None
        self.last_pos: list[float] | None = None
        self.pending_state: dict | None = None
        self.run: dict | None = None
        self.budget: dict | None = None
        self.n_ticks = 0
        self.n_substituted = 0

    # -- properties ------------------------------------------------------------------------------------------------

    @property
    def horizon(self) -> int:
        return int(self.env["horizon"]) if self.env else 0

    @property
    def exec_steps(self) -> int:
        return int(self.env["exec_steps"]) if self.env else 0

    @property
    def action_dim(self) -> int:
        return int(self.env["action_dim"]) if self.env else 0

    @property
    def pos_dim(self) -> int:
        return int(self.env["pos_dim"]) if self.env else 0

    @property
    def alive(self) -> bool:
        c = self.client
        return bool(c is not None and not self.dead and getattr(c, "alive", True))

    # -- the host-side brake -----------------------------------------------------------------------------------------

    def safe_action(self) -> list[float]:
        """`clamp_box(current measured position)` for position kinds, the zero vector for velocity kinds."""
        if self.env is None:
            raise LictorAdapterError("safe_action before start()")
        if self.env["action_kind"] in VELOCITY_KINDS:
            return [0.0] * self.action_dim
        pos = self.last_pos if self.last_pos is not None else [math.nan] * self.action_dim
        pos = (list(pos) + [math.nan] * self.action_dim)[: self.action_dim]
        return clamp_box(pos, self.env["box_lo"], self.env["box_hi"])

    def _synthetic_hold(self, reason: str, detail: dict | None = None) -> dict:
        """What the host executes when no fuse verdict exists (dead child) or after a fatal error. Marked so it is
        never mistaken for a verdict: real verdicts carry `chain`; this carries `synthetic: true`."""
        return {
            "kind": "verdict",
            "synthetic": True,
            "status": "fault",
            "state": "fault",
            "action": self.safe_action(),
            "action_src": "hold",
            "substituted": True,
            "trips": ["schema"] if reason == "fatal" else [],
            "reason": reason,
            "reason_text": "The fuse is unavailable; the host applies its own brake and ends the episode.",
            "adapter": ADAPTER_VERSION,
            "detail": detail or {},
        }

    # -- start -------------------------------------------------------------------------------------------------------

    def start(
        self,
        envelope: os.PathLike | str,
        calibration: os.PathLike | str | None = None,
        out: os.PathLike | str | None = None,
        mode: str = "observe",
        *,
        envelope_digest: str | None = None,
        expect_tier1: bool | None = None,
        expect_alpha: tuple[int, int] | None = None,
    ) -> dict:
        """Spawn the child and `hello` it with the dims/digests read from the artefacts themselves."""
        if mode not in MODES:
            raise LictorAdapterError(f"mode must be one of {MODES}, got {mode!r}")
        if self.client is not None:
            raise LictorAdapterError("start() called twice; close() first")
        self.env = load_envelope(envelope)
        self.calib = load_calibration(calibration) if calibration is not None else None
        self.mode = mode
        self.out = str(out) if out is not None else None
        if self._factory is None:
            self.binary = find_lictor_binary(self._binary_arg)
            if self.binary is None:
                raise LictorAdapterError("no `lictor` binary found (set $LICTOR_BIN or pass binary=)")
        else:
            self.binary = find_lictor_binary(self._binary_arg)
        if envelope_digest is None:
            if self.binary is not None:
                envelope_digest = envelope_digest_via_cli(self.binary, envelope)
            elif self._factory is not None:
                envelope_digest = ZERO_DIGEST
            else:  # pragma: no cover - unreachable: a missing binary raised above
                raise LictorAdapterError("cannot compute the envelope digest without a binary")
        if not _is_hex64(envelope_digest):
            raise LictorAdapterError("envelope_digest must be 64 lowercase hex characters")
        self.envelope_digest = envelope_digest
        factory = self._factory or _default_client_factory
        self.client = factory(
            binary=self.binary,
            envelope=pathlib.Path(envelope),
            calibration=pathlib.Path(calibration) if calibration is not None else None,
            mode=mode,
            out_dir=pathlib.Path(out) if out is not None else None,
            trace=pathlib.Path(self._trace) if self._trace is not None else None,
            key=pathlib.Path(self._key) if self._key is not None else None,
            tier0=self._tier0,
            no_tier1=self._no_tier1,
            ticks=self._ticks,
            timeout_s=self._timeout_s,
            safe_action=self.safe_action,
            latency_label=self._latency_label,
        )
        if expect_tier1 is None:
            expect_tier1 = self.calib is not None and not self._no_tier1
        if expect_alpha is None and self.calib is not None and not self._no_tier1:
            expect_alpha = self.calib["alpha"]
        self.hello = self.client.hello(
            embodiment_id=self.env["embodiment_id"],
            action_dim=self.action_dim,
            pos_dim=self.pos_dim,
            horizon=self.horizon,
            exec_steps=self.exec_steps,
            envelope_digest=self.envelope_digest,
            calibration_digest=self.calib["digest"] if self.calib is not None else None,
            expect_tier1=bool(expect_tier1),
            expect_alpha=expect_alpha,
        )
        return self.hello

    # -- begin -------------------------------------------------------------------------------------------------------

    def begin(
        self,
        episode: int,
        seed: int,
        policy: Any = None,
        env: Any = None,
        *,
        run_id: str | None = None,
        arm_id: str = "lerobot-adapter",
        seed_pool: str = "eval",
        init_state_digest: str | None = None,
        budget: dict | None = None,
        binding: dict | None = None,
        fault_injection: dict | None = None,
        inputs: dict[str, str] | None = None,
    ) -> dict:
        """`episode_begin` with bindings from `harness.compat` when importable, else from `policy`/`env`/kwargs.

        `init_state_digest` is the paired-seed evidence (sha256 of the float64 post-reset state; see
        `init_state_digest_of()`); when the host does not declare it the wire carries `ZERO_DIGEST` and
        `binding.env.init_state_digest_source == "undeclared"` says so in the receipt.
        """
        if self.client is None or self.env is None:
            raise LictorAdapterError("begin() before start()")
        if self.open:
            raise LictorAdapterError("begin() while an episode is open; end() first")
        self._reset_episode()
        digest_source = "host"
        if init_state_digest is None:
            init_state_digest, digest_source = ZERO_DIGEST, "undeclared"
        if not _is_hex64(init_state_digest):
            raise LictorAdapterError("init_state_digest must be 64 lowercase hex characters")
        self.run = {
            "run_id": run_id or self.run_id,
            "arm_id": arm_id,
            "episode_index": int(episode),
            "seed": int(seed),
            "seed_pool": seed_pool,
            "init_state_digest": init_state_digest,
        }
        self.budget = {
            "delay_steps": 0,
            "tick_ms": self.env["tick_ms"],
            "exec_mode": "sync",
            "stitch": "drop",
            "on_escalate": "terminate_fail",
        }
        self.budget.update(budget or {})
        bind = self._bindings(policy, env, binding)
        bind["env"]["init_state_digest_source"] = digest_source
        manifest = self._inputs(inputs)
        r = self.client.episode_begin(self.run, self.budget, bind, fault_injection, manifest)
        self.open = True
        return r

    def _bindings(self, policy: Any, env: Any, explicit: dict | None) -> dict:
        compat = _try_import("harness.compat")
        env_b: dict = {}
        pol_b: dict = {}
        host_b: dict = {}
        if compat is not None:
            host_b = _call_first(compat, ("host_binding",)) or {}
            env_b = _call_first(compat, ("env_binding", "make_env_binding", "EnvBinding"), env) or {}
            pol_b = _call_first(compat, ("policy_binding", "make_policy_binding", "PolicyBinding"), policy) or {}
        if not env_b:
            env_b = self._env_binding_fallback(env)
        if not pol_b:
            pol_b = self._policy_binding_fallback(policy)
        if not host_b:
            host_b = self._host_binding_fallback()
        out = {"env": _stringify(env_b), "policy": _stringify(pol_b), "host": _stringify(host_b)}
        for section, values in (explicit or {}).items():
            out.setdefault(str(section), {}).update(_stringify(values))
        out["host"].setdefault("adapter", ADAPTER_VERSION)
        out["host"].setdefault("adapter_status", ADAPTER_STATUS)
        return out

    def _env_binding_fallback(self, env: Any) -> dict:
        assert self.env is not None
        spec = getattr(env, "spec", None)
        env_id = getattr(spec, "id", None) or self.env["embodiment_id"]
        num, den = self.env["control_hz"]
        b = {"env_id": env_id, "control_hz": f"{num}/{den}"}
        max_steps = getattr(spec, "max_episode_steps", None)
        if max_steps is not None:
            b["max_episode_steps"] = str(max_steps)
        b["vel_source"] = "host" if self.env["provides_vel"] else "none"
        return b

    @staticmethod
    def _policy_binding_fallback(policy: Any) -> dict:
        cfg = getattr(policy, "config", None)
        repo_id = "none" if policy is None else (getattr(policy, "name", None) or type(policy).__name__)
        b: dict = {"repo_id": repo_id}
        for k in ("horizon", "n_action_steps", "n_obs_steps", "num_inference_steps", "device", "dtype"):
            v = getattr(cfg, k, None)
            if v is not None:
                b[k] = str(v)
        weights = getattr(policy, "weights_sha256", None)
        if weights is not None:
            b["weights_sha256"] = str(weights)
        return b

    @staticmethod
    def _host_binding_fallback() -> dict:
        b = {"os": platform.platform(), "cpu": platform.processor() or "unknown", "python": platform.python_version()}
        torch = sys.modules.get("torch")
        if torch is not None:
            b["torch"] = str(getattr(torch, "__version__", "unknown"))
        lerobot = sys.modules.get("lerobot")
        if lerobot is not None:
            b["lerobot"] = str(getattr(lerobot, "__version__", "unknown"))
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONHASHSEED", "CUBLAS_WORKSPACE_CONFIG"):
            if var in os.environ:
                b[var] = os.environ[var]
        return b

    def _inputs(self, explicit: dict[str, str] | None) -> dict[str, str]:
        manifest: dict[str, str] = {}
        compat = _try_import("harness.compat")
        if compat is not None:
            manifest.update(_call_first(compat, ("inputs_manifest",)) or {})
        manifest["adapters/lerobot/lictor_lerobot.py"] = sha256_file(__file__)
        if self.env is not None:
            manifest[_repo_relative(self.env["path"])] = sha256_file(self.env["path"])
        manifest.update(explicit or {})
        return manifest

    # -- tick --------------------------------------------------------------------------------------------------------

    def note_state(self, pos: Any, vel: Any = None, aux: Any = None) -> None:
        """Record the state the next tick will carry (used by `LictorObserveStep`)."""
        self.pending_state = {"pos": as_f64_row(pos), "vel": None if vel is None else as_f64_row(vel), "aux": [] if aux is None else as_f64_row(aux)}

    def _split_state(self, state: Any) -> tuple[list[float], list[float] | None, list[float]]:
        if isinstance(state, dict):
            if "pos" not in state:
                raise LictorAdapterError("state dict needs at least `pos`")
            pos = as_f64_row(state["pos"])
            vel = None if state.get("vel") is None else as_f64_row(state["vel"])
            aux = [] if state.get("aux") is None else as_f64_row(state["aux"])
        else:
            pos, vel, aux = as_f64_row(state), None, []
        if len(pos) < self.pos_dim:
            raise LictorAdapterError(f"state.pos has {len(pos)} entries, the envelope's pos_dim is {self.pos_dim}")
        pos = pos[: self.pos_dim]
        if vel is not None:
            vel = vel[: self.pos_dim]
        return pos, vel, aux

    def tick(
        self,
        state: Any,
        action: Any,
        chunk: Any = None,
        ext: Any = None,
        *,
        missed_ticks: int = 0,
        ack: dict | None = None,
    ) -> dict:
        """Exactly one wire `tick`. `chunk` is the FULL (h, d) chunk on the delivery tick and None otherwise;
        `action` is the row the host is about to execute (checked against the chunk row the fuse will index --
        a mismatch means the host's executor and the fuse disagree on `idx`, which is a bug, not a verdict).
        Returns the verdict dict verbatim; after a fault the synthetic hold (see `_synthetic_hold`)."""
        if not self.open:
            raise LictorAdapterError("tick() before begin()")
        pos, vel, aux = self._split_state(state)
        self.last_pos = pos
        self.pending_state = None
        if self.dead:
            return self._synthetic_hold("fuse_crash", self.fault)
        if self.faulted:
            return self._synthetic_hold("fatal", self.fault)
        skip = int(missed_ticks)
        if skip < 0:
            raise LictorAdapterError("missed_ticks must be >= 0")
        t = self.t + skip
        msg = None
        if chunk is not None:
            rows = as_f64_rows(chunk)
            if len(rows) != self.horizon or len(rows[0]) != self.action_dim:
                raise LictorAdapterError(
                    f"chunk is {len(rows)}x{len(rows[0])}, the envelope needs {self.horizon}x{self.action_dim}"
                    " (the fuse always receives the FULL chunk; never trim rows)"
                )
            msg = chunk_message(self.next_seq, t, rows, self.exec_steps)
            self.next_seq += 1
            self.cur_chunk = rows
            self.idx = 0
        else:
            if self.cur_chunk is None or self.idx is None:
                raise LictorAdapterError("the first tick of an episode must deliver a chunk")
            self.idx += 1 + skip
            if self.idx >= self.horizon:
                raise LictorAdapterError("executed past the chunk horizon without a new chunk (host executor bug)")
        row = as_f64_row(action)
        if row != self.cur_chunk[self.idx]:
            raise LictorAdapterError(
                f"executor desync: the host is about to execute {row} but chunk row idx={self.idx} is {self.cur_chunk[self.idx]}"
            )
        ext_list = [] if ext is None else as_f64_row(ext)
        try:
            v = self.client.tick(t=t, idx=self.idx, pos=pos, aux=aux, chunk=msg, vel=vel, ext=ext_list, missed_ticks=skip, ack=ack)
        except Exception as exc:  # noqa: BLE001 - only LictorFault is swallowed, everything else re-raised below
            if not _is_lictor_fault(exc):
                raise
            self.dead = True
            self.faulted = True
            self.fault = {"reason": str(getattr(exc, "reason", "exit")), "t": t}
            hold = self._synthetic_hold("fuse_crash", self.fault)
            last = getattr(exc, "last_safe_action", None)
            if last is not None:
                hold["action"] = [float(x) for x in last]
            self.last_verdict = hold
            return hold
        self.t = t + 1
        self.n_ticks += 1
        if isinstance(v, dict) and v.get("kind") == "error":
            self.faulted = True
            self.fault = {"reason": "fatal", "code": v.get("code"), "message": v.get("message"), "t": t}
            hold = self._synthetic_hold("fatal", self.fault)
            self.last_verdict = hold
            return hold
        if isinstance(v, dict) and v.get("substituted"):
            self.n_substituted += 1
        self.last_verdict = v
        return v

    # -- end ---------------------------------------------------------------------------------------------------------

    def end(
        self,
        success: bool,
        steps: int,
        progress: float | None = None,
        *,
        ended_by: str | None = None,
        reward_sum: float | None = None,
        max_progress: float | None = None,
    ) -> dict:
        """`episode_end`. `progress` is the task's scalar progress at the end (coverage on PushT; `final_coverage`),
        `max_progress` its episode maximum (defaults to `progress`). A dead child gets `lictor crash-receipt`."""
        if not self.open:
            raise LictorAdapterError("end() before begin()")
        if ended_by is None:
            if success:
                ended_by = "success"
            elif self.dead:
                ended_by = "fault"
            elif self.faulted:
                ended_by = "fault"
            elif (self.last_verdict or {}).get("state") in ("escalated", "terminated"):
                ended_by = "escalation_terminate"
            else:
                ended_by = "truncated"
        if ended_by not in ENDED_BY:
            raise LictorAdapterError(f"ended_by must be one of {ENDED_BY}, got {ended_by!r}")
        self.open = False
        if self.dead:
            return self._crash_receipt()
        final = finite_or_none(None if progress is None else float(progress))
        best = finite_or_none(final if max_progress is None else float(max_progress))
        outcome = {
            "steps": int(steps),
            "success": bool(success),
            "terminated": bool(success),
            "truncated": (not success) and ended_by == "truncated",
            "max_coverage": best,
            "final_coverage": final,
            "reward_sum": finite_or_none(None if reward_sum is None else float(reward_sum)),
            "ended_by": ended_by,
        }
        return self.client.episode_end(outcome)

    def _crash_receipt(self) -> dict:
        """Host-side accounting for a child that died before `episode_end` (CLI SURFACE: `lictor crash-receipt`)."""
        info = {"kind": "crash_receipt", "written": False, "reason": (self.fault or {}).get("reason", "exit")}
        if self.binary is None or self.out is None or self.run is None or self.env is None:
            info["note"] = "no binary/out dir/run record: the crash was not recorded in a ledger"
            return info
        cmd = [self.binary, "crash-receipt", "--envelope", self.env["path"]]
        if self.calib is not None:
            cmd += ["--calibration", self.calib["path"]]
        cmd += [
            "--mode", self.mode,
            "--run-id", str(self.run["run_id"]),
            "--arm-id", str(self.run["arm_id"]),
            "--episode-index", str(self.run["episode_index"]),
            "--seed", str(self.run["seed"]),
            "--seed-pool", str(self.run["seed_pool"]),
            "--init-state-digest", str(self.run["init_state_digest"]),
            "--budget", json.dumps(self.budget, sort_keys=True, separators=(",", ":")),
            "--out", self.out,
            "--note", f"{ADAPTER_VERSION}: serve child died ({info['reason']}) at t={self.t}",
        ]
        if self._key is not None:
            cmd += ["--key", str(self._key)]
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        info["written"] = r.returncode == 0
        info["stdout"] = r.stdout.strip()
        info["stderr"] = r.stderr.strip()[-400:]
        return info

    def close(self) -> None:
        c, self.client = self.client, None
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001 - closing a dead child must never raise into the host
                pass


def _call_first(mod: Any, names: Sequence[str], *args: Any) -> Any:
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            try:
                out = fn(*args)
            except TypeError:
                continue
            if isinstance(out, dict):
                return out
            if hasattr(out, "__dict__"):
                return dict(vars(out))
    return None


def _repo_relative(path: str) -> str:
    root = pathlib.Path(__file__).resolve().parents[2]
    try:
        return pathlib.Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return pathlib.Path(path).name


# ----------------------------------------------------------------------------------------------------------------
# ProcessorStep-shaped classes
# ----------------------------------------------------------------------------------------------------------------


class _DuckProcessorStep:
    """The surface of `lerobot.processor.ProcessorStep` (0.6.1) without importing it."""

    _current_transition: Any = None

    @property
    def transition(self) -> Any:
        if self._current_transition is None:
            raise ValueError("Transition is not set. Make sure to call the step with a transition first.")
        return self._current_transition

    def __call__(self, transition: Any) -> Any:  # pragma: no cover - overridden
        return transition

    def get_config(self) -> dict[str, Any]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    def reset(self) -> None:
        return None

    def transform_features(self, features: Any) -> Any:
        return features


def _processor_step_base() -> type:
    """The real ABC when the host already imported `lerobot.processor`; the duck otherwise (never imports it)."""
    mod = sys.modules.get("lerobot.processor")
    base = getattr(mod, "ProcessorStep", None) if mod is not None else None
    return base if isinstance(base, type) else _DuckProcessorStep


_StepBase: type = _processor_step_base()


class LictorObserveStep(_StepBase):  # type: ignore[misc,valid-type]
    """`env_preprocessor` companion: records `observation.state` (first `pos_dim` entries as `pos`) so the
    action-only transition seen by `LictorStep` has a measured position to tick with. Optional `vel_key` /
    `aux_key` name observation entries used as `vel` / `aux`; `state_fn(observation)` overrides all three."""

    def __init__(
        self,
        fuse: Fuse,
        state_key: str = STATE_KEY,
        vel_key: str | None = None,
        aux_key: str | None = None,
        state_fn: Callable[[dict], dict] | None = None,
    ) -> None:
        self.fuse = fuse
        self.state_key = state_key
        self.vel_key = vel_key
        self.aux_key = aux_key
        self.state_fn = state_fn

    def __call__(self, transition: Any) -> Any:
        self._current_transition = transition
        obs = transition.get(OBSERVATION_KEY) if isinstance(transition, dict) else None
        if obs is None:
            return transition
        if self.state_fn is not None:
            st = self.state_fn(obs)
            self.fuse.note_state(st["pos"], st.get("vel"), st.get("aux"))
            return transition
        if self.state_key in obs:
            vel = obs.get(self.vel_key) if self.vel_key else None
            aux = obs.get(self.aux_key) if self.aux_key else None
            self.fuse.note_state(obs[self.state_key], vel, aux)
        return transition

    def transform_features(self, features: Any) -> Any:
        return features

    def get_config(self) -> dict[str, Any]:
        return {"state_key": self.state_key, "vel_key": self.vel_key, "aux_key": self.aux_key, "status": ADAPTER_STATUS}


class LictorStep(_StepBase):  # type: ignore[misc,valid-type]
    """`env_postprocessor` step: the transition's action goes through the fuse and comes back as `verdict.action`.

    `policy` enables chunk discovery from `policy._queues[ACTION]` (simplification 1 in the module docstring);
    `chunk_fn(action) -> (h, d) | None` supplies the chunk explicitly; `state_fn() -> {"pos", "vel", "aux"}`
    replaces `LictorObserveStep`; `ext_fn() -> [ext0..]` feeds the sidecar channels. With `auto_episodes` the
    step opens episode `k` (seed `k`) on the first action after `reset()`; hosts that know their seeds call
    `fuse.begin(...)` themselves and leave it False.
    """

    def __init__(
        self,
        fuse: Fuse,
        policy: Any = None,
        *,
        chunk_fn: Callable[[Any], Any] | None = None,
        state_fn: Callable[[], dict] | None = None,
        ext_fn: Callable[[], Sequence[float]] | None = None,
        auto_episodes: bool = False,
    ) -> None:
        self.fuse = fuse
        self.policy = policy
        self.chunk_fn = chunk_fn
        self.state_fn = state_fn
        self.ext_fn = ext_fn
        self.auto_episodes = auto_episodes
        self.episodes = 0
        self.n_calls = 0
        self.n_substituted = 0
        self.last_verdict: dict | None = None

    # -- ProcessorStep surface ---------------------------------------------------------------------------------------

    def __call__(self, transition: Any) -> Any:
        if not isinstance(transition, dict):
            raise ValueError("LictorStep expects an EnvTransition (a dict keyed by TransitionKey)")
        self._current_transition = transition
        action = transition.get(ACTION_KEY)
        if action is None:
            raise ValueError("LictorStep requires an action in the transition.")
        row = as_f64_row(action)
        fuse = self.fuse
        if self.auto_episodes and not fuse.open:
            fuse.begin(episode=self.episodes, seed=self.episodes, policy=self.policy, env=None)
        state = self.state_fn() if self.state_fn is not None else fuse.pending_state
        if state is None:
            raise LictorAdapterError(
                "no state for this tick: add LictorObserveStep to the env_preprocessor or pass state_fn"
            )
        chunk = self._chunk_for(action, row)
        ext = self.ext_fn() if self.ext_fn is not None else None
        verdict = fuse.tick(state, row, chunk=chunk, ext=ext)
        new_transition = dict(transition)
        new_transition[ACTION_KEY] = restore_like(action, verdict["action"])
        self.n_calls += 1
        if verdict.get("substituted"):
            self.n_substituted += 1
        self.last_verdict = verdict
        return new_transition

    def _chunk_for(self, action: Any, row: list[float]) -> Any:
        if self.chunk_fn is not None:
            return self.chunk_fn(action)
        if self.policy is not None:
            queues = getattr(self.policy, "_queues", None)
            cfg = getattr(self.policy, "config", None)
            n = getattr(cfg, "n_action_steps", None)
            if isinstance(queues, dict) and ACTION_KEY in queues and n is not None:
                q = queues[ACTION_KEY]
                if len(q) == int(n) - 1:
                    return [row] + [as_f64_row(x) for x in q]
                return None
        if self.fuse.cur_chunk is None and self.fuse.horizon == 1:
            return [row]  # per-action mode: an envelope with horizon = 1, exec_steps = 1
        if self.fuse.cur_chunk is None:
            raise LictorAdapterError("no chunk source: pass policy= or chunk_fn= (or use an envelope with horizon = 1)")
        return [row] if self.fuse.horizon == 1 else None

    def transform_features(self, features: Any) -> Any:
        """Identity: the action keeps its shape. When the ACTION feature is discoverable its width is cross-checked."""
        if isinstance(features, dict) and self.fuse.env is not None:
            actions = features.get("ACTION")
            if isinstance(actions, dict):
                for name, feat in actions.items():
                    shape = getattr(feat, "shape", None)
                    if shape and int(shape[-1]) != self.fuse.action_dim:
                        raise LictorAdapterError(
                            f"feature {name!r} has width {shape[-1]}, the envelope's action_dim is {self.fuse.action_dim}"
                        )
        return features

    def feature_contract(self) -> dict[str, Any]:
        """What the step expects of the pipeline, in the envelope's terms (no lerobot types needed)."""
        env = self.fuse.env or {}
        return {
            "action": {"type": "ACTION", "shape": [env.get("action_dim", 0)], "kind": env.get("action_kind")},
            "state": {"type": "STATE", "key": STATE_KEY, "pos_dim": env.get("pos_dim", 0)},
            "chunk": {"horizon": env.get("horizon", 0), "exec_steps": env.get("exec_steps", 0)},
            "embodiment_id": env.get("embodiment_id"),
            "batch": 1,
            "status": ADAPTER_STATUS,
        }

    def get_config(self) -> dict[str, Any]:
        env = self.fuse.env or {}
        return {
            "adapter": ADAPTER_VERSION,
            "status": ADAPTER_STATUS,
            "envelope": env.get("path"),
            "mode": self.fuse.mode,
            "auto_episodes": self.auto_episodes,
        }

    def state_dict(self) -> dict[str, Any]:
        """The fuse's state lives in the child process and its receipts; nothing here belongs in a checkpoint."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    def reset(self) -> None:
        """Episode boundary: an episode still open is ended as `abort` (the host reset before declaring an outcome)."""
        if self.fuse.open:
            self.fuse.end(success=False, steps=self.fuse.t, progress=None, ended_by="abort")
        self.episodes += 1
        self.n_calls = 0
        self.n_substituted = 0
        self.last_verdict = None

    # -- lerobot binding ---------------------------------------------------------------------------------------------

    @classmethod
    def bind_lerobot(cls, name: str = "lictor_step") -> type:
        """Import `lerobot.processor` now and return a subclass that is a real `ProcessorStep` and is registered
        under `name` (idempotent). Hosts that only need the step in a pipeline do not need this."""
        proc = importlib.import_module("lerobot.processor")
        base = proc.ProcessorStep
        if issubclass(cls, base):
            bound = cls
        else:
            bound = type(cls.__name__, (cls, base), {"__module__": cls.__module__, "__doc__": cls.__doc__})
        registry = getattr(proc, "ProcessorStepRegistry", None)
        if registry is not None:
            try:
                registry.register(name=name)(bound)
            except Exception:  # noqa: BLE001 - already registered (re-import) is not an error for the host
                pass
        return bound


__all__ = [
    "ADAPTER_STATUS",
    "ADAPTER_VERSION",
    "Fuse",
    "LictorAdapterError",
    "LictorObserveStep",
    "LictorStep",
    "ZERO_DIGEST",
    "as_f64_row",
    "as_f64_rows",
    "chunk_message",
    "clamp_box",
    "envelope_digest_via_cli",
    "find_lictor_binary",
    "finite_or_none",
    "init_state_digest_of",
    "load_calibration",
    "load_envelope",
    "restore_like",
    "sha256_file",
]
