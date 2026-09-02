#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12.

lictor as a websocket proxy in front of openpi's `WebsocketPolicyServer` (roadmap item 4 of docs/ARCHITECTURE.md
sec 12; the Rust `lictor proxy --upstream ws://host:8000` that this skeleton is the specification for comes later).

    python adapters/openpi/lictor_proxy.py --listen 0.0.0.0:8001 --upstream ws://host:8000 \\
        --envelope envelopes/<embodiment>.toml [--calibration <F.json>] --mode enforce --out results/openpi \\
        --exec-steps 8 --episode-boundary reset-key
    python adapters/openpi/lictor_proxy.py --dry-run --envelope envelopes/pusht.base.toml --json   # no network, no fuse child

Upstream wire format (read from openpi `packages/openpi-client/src/openpi_client/{msgpack_numpy,websocket_client_policy}.py`
and `src/openpi/serving/websocket_policy_server.py`, main branch, 2026-09-02; NOT exercised against a running server):

  * the server sends ONE msgpack frame of metadata right after the handshake; then, per request, the client sends
    one msgpack frame holding the observation dict and receives one msgpack frame holding `{"actions": (H, D), ...,
    "server_timing": {...}}`;
  * numpy arrays travel as msgpack maps with BYTES keys `{b"__ndarray__": True, b"data": <bytes>, b"dtype": "<f4",
    b"shape": [H, D]}` (scalars as `b"__npgeneric__"`); `compression=None, max_size=None` on both ends;
  * a TEXT frame from the server is an error (`WebsocketClientPolicy.infer` raises `RuntimeError` on `str`).

What the proxy does per client connection: connect upstream, forward the metadata frame (plus a `lictor` entry),
then for every observation frame: forward the bytes upstream VERBATIM, receive the chunk, lower it to the lictor
chunk IR (`horizon = H`, `exec_steps` from `--exec-steps`), drive ONE wire `tick` per action row that the client is
going to execute (rows `0 .. exec_steps-1`; the delivery tick carries the full chunk), replace those rows with the
fuse's `verdict.action`, fill rows `exec_steps .. H-1` with the last fused action (a hold: the client must never
execute rows the fuse has not ticked), and send the same response dict back with `actions` substituted and a
`lictor` summary added. A fault of any kind -- upstream down, upstream error frame, malformed frame, the fuse
child dead or a fatal wire error -- NEVER drops the client connection: the response is a hold chunk
(`clamp_box(current position)` for position kinds, zeros for velocity kinds) plus the error, and the proxy keeps
serving. The "error frame" of the plan is carried INSIDE the response dict as `lictor.error`: a separate text frame
would be read by the stock client as a server traceback and make it raise. One receipt per episode; episodes are
delimited by a truthy `--reset-key` entry in the observation dict (`--episode-boundary reset-key`, the default; the
key is forwarded upstream untouched) or by the connection itself (`--episode-boundary connection`).

The proxy-side executor gate is the simplification everything above rests on, stated once: the proxy only sees the
client at request time, so it assumes the client executes exactly `exec_steps` rows between requests
(`open_loop_horizon`/`replan_steps` in the openpi examples) and holds the delivered observation for those rows
(`t` advances by one per row). Tier 0, the brake rollout and the chunk-boundary features are evaluated at delivery
as on PushT; the per-tick trail features (`stall`, `path_ineff`) see a stationary position between requests, so a
Tier-1 calibration is only exchangeable when it was collected THROUGH this proxy. Milestone 1 ships no such
calibration: run the proxy without `--calibration` (Tier 1 disarmed) until one exists.

Nothing here imports `websockets`, `msgpack`, `numpy` or `openpi` at module import time. When `msgpack` is absent a
byte-compatible mini encoder/decoder (nil, bool, int, float64, str, bin, array, map) stands in, so `--dry-run` and
the unit tests exercise the real frame layout either way; when `numpy` is absent arrays decode to `ArrayLike`
(shape + dtype + bytes, `tolist()`), which is enough for the codec round trip and the chunk lowering.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import math
import pathlib
import struct
import sys
from typing import Any, Callable, Sequence

if __name__ == "__main__" and __package__ in (None, ""):  # `python adapters/openpi/lictor_proxy.py ...` from anywhere
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from adapters.lerobot.lictor_lerobot import (  # noqa: E402 - after the script-mode path fix above
    ADAPTER_STATUS,
    ZERO_DIGEST,
    Fuse,
    LictorAdapterError,
    as_f64_row,
    as_f64_rows,
    load_envelope,
)

PROXY_VERSION = "lictor_proxy/0.1.0"
DEFAULT_LISTEN = "0.0.0.0:8001"
DEFAULT_UPSTREAM = "ws://127.0.0.1:8000"
DEFAULT_POS_KEY = "observation/state"
DEFAULT_RESET_KEY = "reset"
BOUNDARIES = ("reset-key", "connection")
CODECS = ("auto", "msgpack", "mini")

# ----------------------------------------------------------------------------------------------------------------
# arrays without numpy
# ----------------------------------------------------------------------------------------------------------------

_STRUCT_CODES = {"f8": "d", "f4": "f", "i8": "q", "i4": "i", "i2": "h", "i1": "b", "u8": "Q", "u4": "I", "u2": "H", "u1": "B", "b1": "?"}


def _struct_fmt(dtype: str, n: int) -> str:
    order = dtype[0] if dtype and dtype[0] in "<>|=" else "<"
    kind = dtype[1:] if dtype and dtype[0] in "<>|=" else dtype
    code = _STRUCT_CODES.get(kind)
    if code is None:
        raise LictorAdapterError(f"ArrayLike cannot decode dtype {dtype!r} without numpy")
    if order in "|=":
        order = "<"
    return f"{order}{n}{code}"


def _flatten(values: Any, shape: list[int], flat: list[Any], depth: int = 0) -> None:
    if isinstance(values, (list, tuple)):
        if len(shape) <= depth:
            shape.append(len(values))
        elif shape[depth] != len(values):
            raise LictorAdapterError("ragged nested list")
        for v in values:
            _flatten(v, shape, flat, depth + 1)
    else:
        flat.append(values)


def _reshape(flat: list[Any], shape: Sequence[int]) -> Any:
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    step = 1
    for s in shape[1:]:
        step *= int(s)
    return [_reshape(flat[i * step : (i + 1) * step], shape[1:]) for i in range(int(shape[0]))]


class ArrayLike:
    """A numpy-free stand-in for an ndarray on the wire: shape, dtype string (numpy `dtype.str`) and raw bytes."""

    __slots__ = ("shape", "dtype", "data")

    def __init__(self, shape: Sequence[int], dtype: str, data: bytes) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.dtype = str(dtype)
        self.data = bytes(data)

    @classmethod
    def from_list(cls, values: Any, dtype: str = "<f8") -> "ArrayLike":
        shape: list[int] = []
        flat: list[Any] = []
        _flatten(values, shape, flat)
        fmt = _struct_fmt(dtype, len(flat))
        if fmt[-1] in "df":
            flat = [float(v) for v in flat]
        elif fmt[-1] == "?":
            flat = [bool(v) for v in flat]
        else:
            flat = [int(v) for v in flat]
        return cls(shape, dtype, struct.pack(fmt, *flat))

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def tobytes(self) -> bytes:
        return self.data

    def tolist(self) -> Any:
        flat = list(struct.unpack(_struct_fmt(self.dtype, self.size), self.data))
        return _reshape(flat, self.shape)

    def __repr__(self) -> str:
        return f"ArrayLike(shape={self.shape}, dtype={self.dtype!r}, nbytes={len(self.data)})"


def _numpy() -> Any:
    try:
        return importlib.import_module("numpy")
    except ImportError:
        return None


def _is_ndarray(x: Any) -> bool:
    return type(x).__module__.split(".")[0] == "numpy" and hasattr(x, "dtype") and hasattr(x, "shape")


def make_array(values: Any, dtype: str = "<f8") -> Any:
    """An (H, D) array of the given dtype: numpy when importable, `ArrayLike` otherwise."""
    np = _numpy()
    if np is not None:
        return np.asarray(values, dtype=np.dtype(dtype))
    return ArrayLike.from_list(values, dtype)


def raise_chunk(template: Any, rows: Sequence[Sequence[float]]) -> Any:
    """Rebuild a fused chunk in the container/dtype of the chunk the upstream sent (float32 arrays in openpi)."""
    vals = [[float(v) for v in r] for r in rows]
    if _is_ndarray(template):
        np = importlib.import_module("numpy")
        return np.asarray(vals, dtype=template.dtype)
    if isinstance(template, ArrayLike):
        return ArrayLike.from_list(vals, template.dtype)
    if template is None:
        return make_array(vals, "<f4")
    return vals


# ----------------------------------------------------------------------------------------------------------------
# msgpack-numpy codec (openpi's hooks; msgpack when importable, a byte-compatible mini encoder otherwise)
# ----------------------------------------------------------------------------------------------------------------


def pack_array(obj: Any) -> Any:
    """openpi's `msgpack_numpy.pack_array` default hook (plus `ArrayLike`)."""
    np = sys.modules.get("numpy")
    if np is not None and isinstance(obj, (np.ndarray, np.generic)):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    if isinstance(obj, ArrayLike):
        return {b"__ndarray__": True, b"data": obj.data, b"dtype": obj.dtype, b"shape": obj.shape}
    return obj


def unpack_array(obj: Any) -> Any:
    """openpi's `msgpack_numpy.unpack_array` object hook; `ArrayLike` when numpy is absent."""
    if b"__ndarray__" in obj:
        np = _numpy()
        if np is not None:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"]))
        return ArrayLike(obj[b"shape"], obj[b"dtype"], obj[b"data"])
    if b"__npgeneric__" in obj:
        np = _numpy()
        if np is not None:
            return np.dtype(obj[b"dtype"]).type(obj[b"data"])
        return obj[b"data"]
    return obj


def _mini_packb(obj: Any, default: Callable[[Any], Any] | None) -> bytes:
    out = bytearray()
    _mp_enc(obj, out, default, 0)
    return bytes(out)


def _mp_enc(o: Any, out: bytearray, default: Callable[[Any], Any] | None, depth: int) -> None:
    if depth > 128:
        raise ValueError("msgpack nesting too deep")
    if o is None:
        out.append(0xC0)
    elif o is True:
        out.append(0xC3)
    elif o is False:
        out.append(0xC2)
    elif isinstance(o, int):
        if 0 <= o < 0x80:
            out.append(o)
        elif -0x20 <= o < 0:
            out.append(o & 0xFF)
        elif 0 <= o <= 0xFF:
            out += b"\xcc" + o.to_bytes(1, "big")
        elif 0 <= o <= 0xFFFF:
            out += b"\xcd" + o.to_bytes(2, "big")
        elif 0 <= o <= 0xFFFFFFFF:
            out += b"\xce" + o.to_bytes(4, "big")
        elif 0 <= o <= 0xFFFFFFFFFFFFFFFF:
            out += b"\xcf" + o.to_bytes(8, "big")
        elif -0x80 <= o < 0:
            out += b"\xd0" + o.to_bytes(1, "big", signed=True)
        elif -0x8000 <= o < 0:
            out += b"\xd1" + o.to_bytes(2, "big", signed=True)
        elif -0x80000000 <= o < 0:
            out += b"\xd2" + o.to_bytes(4, "big", signed=True)
        elif -0x8000000000000000 <= o < 0:
            out += b"\xd3" + o.to_bytes(8, "big", signed=True)
        else:
            raise OverflowError("int out of msgpack range")
    elif isinstance(o, float):
        out += b"\xcb" + struct.pack(">d", o)
    elif isinstance(o, str):
        b = o.encode("utf-8")
        n = len(b)
        if n < 32:
            out.append(0xA0 | n)
        elif n < 0x100:
            out += b"\xd9" + n.to_bytes(1, "big")
        elif n < 0x10000:
            out += b"\xda" + n.to_bytes(2, "big")
        else:
            out += b"\xdb" + n.to_bytes(4, "big")
        out += b
    elif isinstance(o, (bytes, bytearray, memoryview)):
        b = bytes(o)
        n = len(b)
        if n < 0x100:
            out += b"\xc4" + n.to_bytes(1, "big")
        elif n < 0x10000:
            out += b"\xc5" + n.to_bytes(2, "big")
        else:
            out += b"\xc6" + n.to_bytes(4, "big")
        out += b
    elif isinstance(o, (list, tuple)):
        n = len(o)
        if n < 16:
            out.append(0x90 | n)
        elif n < 0x10000:
            out += b"\xdc" + n.to_bytes(2, "big")
        else:
            out += b"\xdd" + n.to_bytes(4, "big")
        for v in o:
            _mp_enc(v, out, default, depth + 1)
    elif isinstance(o, dict):
        n = len(o)
        if n < 16:
            out.append(0x80 | n)
        elif n < 0x10000:
            out += b"\xde" + n.to_bytes(2, "big")
        else:
            out += b"\xdf" + n.to_bytes(4, "big")
        for k, v in o.items():
            _mp_enc(k, out, default, depth + 1)
            _mp_enc(v, out, default, depth + 1)
    else:
        if default is None:
            raise TypeError(f"cannot serialize {type(o).__name__}")
        o2 = default(o)
        if o2 is o:
            raise TypeError(f"cannot serialize {type(o).__name__}")
        _mp_enc(o2, out, default, depth + 1)


def _mini_unpackb(data: bytes, object_hook: Callable[[dict], Any] | None) -> Any:
    mv = memoryview(bytes(data))
    obj, pos = _mp_dec(mv, 0, object_hook)
    if pos != len(mv):
        raise ValueError(f"{len(mv) - pos} extra bytes after the msgpack object")
    return obj


def _mp_dec(mv: memoryview, i: int, hook: Callable[[dict], Any] | None) -> tuple[Any, int]:
    if i >= len(mv):
        raise ValueError("truncated msgpack data")
    b = mv[i]
    i += 1
    if b <= 0x7F:
        return b, i
    if b >= 0xE0:
        return b - 0x100, i
    if 0x80 <= b <= 0x8F:
        return _mp_map(mv, i, b & 0x0F, hook)
    if 0x90 <= b <= 0x9F:
        return _mp_arr(mv, i, b & 0x0F, hook)
    if 0xA0 <= b <= 0xBF:
        n = b & 0x1F
        return bytes(mv[i : i + n]).decode("utf-8"), i + n
    if b == 0xC0:
        return None, i
    if b == 0xC2:
        return False, i
    if b == 0xC3:
        return True, i
    if b in (0xC4, 0xC5, 0xC6):
        w = {0xC4: 1, 0xC5: 2, 0xC6: 4}[b]
        n = int.from_bytes(mv[i : i + w], "big")
        i += w
        return bytes(mv[i : i + n]), i + n
    if b == 0xCA:
        return struct.unpack(">f", mv[i : i + 4])[0], i + 4
    if b == 0xCB:
        return struct.unpack(">d", mv[i : i + 8])[0], i + 8
    if b in (0xCC, 0xCD, 0xCE, 0xCF):
        w = 1 << (b - 0xCC)
        return int.from_bytes(mv[i : i + w], "big"), i + w
    if b in (0xD0, 0xD1, 0xD2, 0xD3):
        w = 1 << (b - 0xD0)
        return int.from_bytes(mv[i : i + w], "big", signed=True), i + w
    if b in (0xD9, 0xDA, 0xDB):
        w = {0xD9: 1, 0xDA: 2, 0xDB: 4}[b]
        n = int.from_bytes(mv[i : i + w], "big")
        i += w
        return bytes(mv[i : i + n]).decode("utf-8"), i + n
    if b in (0xDC, 0xDD):
        w = 2 if b == 0xDC else 4
        n = int.from_bytes(mv[i : i + w], "big")
        return _mp_arr(mv, i + w, n, hook)
    if b in (0xDE, 0xDF):
        w = 2 if b == 0xDE else 4
        n = int.from_bytes(mv[i : i + w], "big")
        return _mp_map(mv, i + w, n, hook)
    raise ValueError(f"unsupported msgpack type byte 0x{b:02x} (ext types are not used by openpi)")


def _mp_arr(mv: memoryview, i: int, n: int, hook: Any) -> tuple[list, int]:
    out = []
    for _ in range(n):
        v, i = _mp_dec(mv, i, hook)
        out.append(v)
    return out, i


def _mp_map(mv: memoryview, i: int, n: int, hook: Any) -> tuple[Any, int]:
    d: dict = {}
    for _ in range(n):
        k, i = _mp_dec(mv, i, hook)
        if not isinstance(k, (str, bytes)):
            raise ValueError("msgpack map keys must be str or bytes")
        v, i = _mp_dec(mv, i, hook)
        d[k] = v
    return (hook(d) if hook is not None else d), i


class Codec:
    """`pack`/`unpack` with openpi's msgpack-numpy hooks. `backend` is "auto" (msgpack when importable), "msgpack"
    or "mini"; `self.backend` says which one is in use."""

    def __init__(self, backend: str = "auto") -> None:
        if backend not in CODECS:
            raise LictorAdapterError(f"codec must be one of {CODECS}")
        self._mp = None
        if backend in ("auto", "msgpack"):
            try:
                self._mp = importlib.import_module("msgpack")
            except ImportError:
                if backend == "msgpack":
                    raise LictorAdapterError("msgpack is not installed (pip install msgpack) -- or use --codec mini") from None
        self.backend = "msgpack" if self._mp is not None else "mini"

    def pack(self, obj: Any) -> bytes:
        if self._mp is not None:
            return self._mp.packb(obj, default=pack_array, use_bin_type=True)
        return _mini_packb(obj, pack_array)

    def unpack(self, data: bytes) -> Any:
        if self._mp is not None:
            return self._mp.unpackb(data, object_hook=unpack_array, raw=False)
        return _mini_unpackb(data, unpack_array)


def same_value(a: Any, b: Any) -> bool:
    """Structural equality that treats numpy arrays and `ArrayLike` by (shape, dtype, values)."""
    if _is_ndarray(a) or isinstance(a, ArrayLike) or _is_ndarray(b) or isinstance(b, ArrayLike):
        if not (hasattr(a, "tolist") and hasattr(b, "tolist")):
            return False
        da = str(getattr(a, "dtype", ""))
        db = str(getattr(b, "dtype", ""))
        np = sys.modules.get("numpy")
        if np is not None:
            da = np.dtype(da).str if da else da
            db = np.dtype(db).str if db else db
        return tuple(a.shape) == tuple(b.shape) and da == db and a.tolist() == b.tolist()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return bool(a == b)


# ----------------------------------------------------------------------------------------------------------------
# the fuse side: observation mapping, the executor gate, an in-process echo double for --dry-run
# ----------------------------------------------------------------------------------------------------------------


class ObsMap:
    """Where `pos` / `vel` / `aux` / `ext` live in the observation dict (openpi uses flat slash-separated keys)."""

    def __init__(
        self,
        pos_key: str = DEFAULT_POS_KEY,
        vel_key: str | None = None,
        aux_key: str | None = None,
        ext_key: str | None = None,
        pos_dim: int | None = None,
        aux_dim: int | None = None,
    ) -> None:
        self.pos_key = pos_key
        self.vel_key = vel_key
        self.aux_key = aux_key
        self.ext_key = ext_key
        self.pos_dim = pos_dim
        self.aux_dim = aux_dim

    def state(self, obs: dict) -> dict:
        if not isinstance(obs, dict) or self.pos_key not in obs:
            keys = sorted(str(k) for k in obs) if isinstance(obs, dict) else []
            raise LictorAdapterError(f"observation lacks pos key {self.pos_key!r}; keys: {keys[:12]}")
        pos = as_f64_row(obs[self.pos_key])
        if self.pos_dim is not None:
            if len(pos) < self.pos_dim:
                raise LictorAdapterError(f"{self.pos_key!r} has {len(pos)} entries, the envelope's pos_dim is {self.pos_dim}")
            pos = pos[: self.pos_dim]
        vel = as_f64_row(obs[self.vel_key]) if self.vel_key and obs.get(self.vel_key) is not None else None
        aux = as_f64_row(obs[self.aux_key]) if self.aux_key and obs.get(self.aux_key) is not None else []
        if self.aux_dim is not None:
            aux = aux[: self.aux_dim]
        return {"pos": pos, "vel": vel, "aux": aux}

    def ext(self, obs: dict) -> list[float] | None:
        if self.ext_key and isinstance(obs, dict) and obs.get(self.ext_key) is not None:
            return as_f64_row(obs[self.ext_key])
        return None


class EchoClient:
    """A test double with the frozen `LictorClient` surface that returns the policy row unchanged (observe-like
    passthrough). Used by `--dry-run` and the unit tests ONLY; its verdicts carry `"echo": true` and no chain."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict]] = []
        self.alive = True
        self._chunk: list[list[float]] | None = None
        self.mode = kwargs.get("mode", "observe")

    def hello(self, **kw: Any) -> dict:
        self.calls.append(("hello", kw))
        return {
            "kind": "hello_ok",
            "proto": "lictor-wire/v1",
            "lictor": "echo",
            "mode": self.mode,
            "tier1_armed": bool(kw.get("expect_tier1")),
            "envelope_digest": kw.get("envelope_digest"),
            "calibration_digest": kw.get("calibration_digest"),
            "echo": True,
        }

    def episode_begin(self, run: dict, budget: dict, binding: dict, fault_injection: Any = None, inputs: Any = None) -> dict:
        self.calls.append(("episode_begin", {"run": run, "budget": budget, "binding": binding, "fault_injection": fault_injection, "inputs": inputs}))
        self._chunk = None
        return {"kind": "episode_ok", "state": "armed", "seq": 0, "echo": True}

    def tick(self, t: int, idx: int, pos: Any, aux: Any, chunk: dict | None = None, vel: Any = None, ext: Any = (), missed_ticks: int = 0, ack: Any = None) -> dict:
        self.calls.append(("tick", {"t": t, "idx": idx, "pos": list(pos), "aux": list(aux), "chunk": chunk, "vel": None if vel is None else list(vel), "ext": list(ext), "missed_ticks": missed_ticks, "ack": ack}))
        if chunk is not None:
            self._chunk = [[0.0 if v is None else float(v) for v in r] for r in chunk["a"]]
        if self._chunk is None or idx >= len(self._chunk):
            return {"kind": "error", "code": "state", "message": "tick without a chunk", "fatal": True, "echo": True}
        row = self._chunk[idx]
        return {
            "kind": "verdict", "t": t, "seq": t, "status": "nominal", "state": "armed", "prev_state": "armed",
            "trips": [], "trip_mask": 0, "action": list(row), "action_src": "policy", "substituted": False,
            "clamped_dims": 0, "scores": {"f": [], "z": [], "s": -math.inf, "valid": 0, "fired": 0}, "tau": math.inf,
            "window_hits": 0, "brake_margin": 0.0, "reason": "ok",
            "reason_text": "All checks passed; the policy action is applied unchanged.",
            "handoff": None, "ack_result": None, "violation_reached_env": False, "verdict_ns": 0, "chain": None,
            "echo": True,
        }

    def episode_end(self, outcome: dict) -> dict:
        self.calls.append(("episode_end", {"outcome": outcome}))
        return {"kind": "episode_receipt", "receipt_path": None, "fuse_ok": False, "fuse_notes": ["echo double: no receipt"], "echo": True}

    def close(self) -> None:
        self.calls.append(("close", {}))
        self.alive = False


class EpisodeSession:
    """The proxy-side executor gate over one started `Fuse`: one request = one chunk = `exec_steps` ticks."""

    def __init__(
        self,
        fuse: Fuse,
        obs_map: ObsMap,
        exec_steps: int | None = None,
        boundary: str = "reset-key",
        reset_key: str = DEFAULT_RESET_KEY,
        arm_id: str = "openpi-proxy",
    ) -> None:
        if fuse.env is None:
            raise LictorAdapterError("EpisodeSession needs a started Fuse")
        if boundary not in BOUNDARIES:
            raise LictorAdapterError(f"episode boundary must be one of {BOUNDARIES}")
        self.fuse = fuse
        self.obs_map = obs_map
        self.exec_steps = int(exec_steps) if exec_steps is not None else fuse.exec_steps
        if not 1 <= self.exec_steps <= fuse.horizon:
            raise LictorAdapterError(f"--exec-steps must be in 1..={fuse.horizon}, got {self.exec_steps}")
        self.boundary = boundary
        self.reset_key = reset_key
        self.arm_id = arm_id
        self.episode = 0
        self.requests = 0
        self.template: Any = None

    def describe(self) -> dict:
        env = self.fuse.env or {}
        return {
            "proxy": PROXY_VERSION,
            "status": ADAPTER_STATUS,
            "mode": self.fuse.mode,
            "embodiment_id": env.get("embodiment_id"),
            "horizon": env.get("horizon"),
            "exec_steps": self.exec_steps,
            "envelope_digest": self.fuse.envelope_digest,
            "calibration_digest": (self.fuse.calib or {}).get("digest"),
            "episode_boundary": self.boundary,
            "reset_key": self.reset_key if self.boundary == "reset-key" else None,
        }

    def begin(self, seed: int | None = None, init_state_digest: str | None = None) -> dict:
        r = self.fuse.begin(
            episode=self.episode,
            seed=self.episode if seed is None else int(seed),
            arm_id=self.arm_id,
            init_state_digest=init_state_digest,
            binding={"host": {"proxy": PROXY_VERSION, "executor_gate": str(self.exec_steps)}},
        )
        return r

    def end(self, ended_by: str = "truncated", success: bool = False) -> dict | None:
        if not self.fuse.open:
            return None
        r = self.fuse.end(success=success, steps=self.fuse.t, progress=None, ended_by=ended_by)
        self.episode += 1
        return r

    def on_request(self, obs: dict, actions: Any) -> tuple[list[list[float]], dict]:
        """Lower the upstream chunk, run the gate, return (fused rows (H, D), summary)."""
        self.requests += 1
        if self.boundary == "reset-key" and isinstance(obs, dict) and obs.get(self.reset_key) and self.fuse.open:
            self.end("truncated")
        if not self.fuse.open:
            seed = obs.get("lictor/seed") if isinstance(obs, dict) else None
            digest = obs.get("lictor/init_state_digest") if isinstance(obs, dict) else None
            self.begin(seed=None if seed is None else int(seed), init_state_digest=None if digest is None else str(digest))
        state = self.obs_map.state(obs)
        ext = self.obs_map.ext(obs)
        rows = as_f64_rows(actions)
        h, d = len(rows), len(rows[0])
        if h != self.fuse.horizon or d != self.fuse.action_dim:
            raise LictorAdapterError(
                f"upstream chunk is {h}x{d}, the envelope needs {self.fuse.horizon}x{self.fuse.action_dim}"
            )
        info: dict[str, Any] = {
            "proxy": PROXY_VERSION, "status": None, "seq": self.fuse.next_seq, "t0": self.fuse.t,
            "ticks": 0, "gate": self.exec_steps, "substituted": 0, "trips": [], "action_src": None, "error": None,
        }
        fused: list[list[float]] = []
        trips: set[str] = set()
        for i in range(self.exec_steps):
            v = self.fuse.tick(state, rows[i], chunk=rows if i == 0 else None, ext=ext)
            fused.append([float(x) for x in v["action"]])
            info["ticks"] += 1
            info["status"] = v.get("status")
            info["action_src"] = v.get("action_src")
            if v.get("substituted"):
                info["substituted"] += 1
            for name in v.get("trips") or []:
                trips.add(str(name))
            if v.get("synthetic"):
                info["error"] = json.dumps(v.get("detail") or {}, sort_keys=True)
        info["trips"] = sorted(trips)
        hold = fused[-1]
        fused.extend([list(hold) for _ in range(h - self.exec_steps)])
        self.template = actions
        return fused, info

    def hold_chunk(self) -> list[list[float]]:
        return [self.fuse.safe_action() for _ in range(self.fuse.horizon)]

    def close(self) -> None:
        self.fuse.close()


# ----------------------------------------------------------------------------------------------------------------
# the websocket proxy (websockets imported lazily)
# ----------------------------------------------------------------------------------------------------------------


def parse_listen(s: str) -> tuple[str, int]:
    host, sep, port = s.rpartition(":")
    if not sep or not port.isdigit():
        raise LictorAdapterError(f"--listen must be host:port, got {s!r}")
    return (host or "0.0.0.0", int(port))


def _websockets() -> tuple[Any, Any, type]:
    try:
        ws = importlib.import_module("websockets")
    except ImportError:
        raise LictorAdapterError("the proxy needs `websockets` (pip install websockets); --dry-run works without it") from None
    try:
        serve = importlib.import_module("websockets.asyncio.server").serve
        connect = importlib.import_module("websockets.asyncio.client").connect
    except (ImportError, AttributeError):
        serve, connect = ws.serve, ws.connect
    closed = importlib.import_module("websockets.exceptions").ConnectionClosed
    return serve, connect, closed


class Proxy:
    """One upstream connection per client connection; see the module docstring for the frame flow."""

    def __init__(self, listen: str, upstream: str, session_factory: Callable[[], EpisodeSession], codec: Codec) -> None:
        self.host, self.port = parse_listen(listen)
        self.upstream = upstream
        self.session_factory = session_factory
        self.codec = codec
        self.connections = 0

    async def serve(self) -> None:
        serve, _connect, _closed = _websockets()
        print(f"lictor_proxy {PROXY_VERSION} ({ADAPTER_STATUS}) listening on {self.host}:{self.port} -> {self.upstream}", file=sys.stderr)
        async with serve(self.handler, self.host, self.port, compression=None, max_size=None) as server:
            await server.serve_forever()

    async def handler(self, ws: Any) -> None:
        _serve, connect, closed = _websockets()
        self.connections += 1
        session = self.session_factory()
        try:
            async with connect(self.upstream, compression=None, max_size=None) as up:
                meta_raw = await up.recv()
                meta = self.codec.unpack(meta_raw) if isinstance(meta_raw, (bytes, bytearray)) else {}
                if not isinstance(meta, dict):
                    meta = {"upstream_metadata": meta}
                meta["lictor"] = session.describe()
                await ws.send(self.codec.pack(meta))
                async for raw in ws:
                    resp = await self.one(session, up, raw)
                    await ws.send(self.codec.pack(resp))
        except closed:
            pass
        except Exception as exc:  # noqa: BLE001 - a client connection must never take the proxy down
            print(f"lictor_proxy: connection ended with {type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            try:
                session.end("truncated")
            except Exception as exc:  # noqa: BLE001
                print(f"lictor_proxy: episode_end failed: {exc}", file=sys.stderr)
            session.close()

    async def one(self, session: EpisodeSession, up: Any, raw: Any) -> dict:
        if not isinstance(raw, (bytes, bytearray)):
            return self.fault_response(session, "client sent a text frame; expected a msgpack binary frame")
        try:
            obs = self.codec.unpack(bytes(raw))
        except Exception as exc:  # noqa: BLE001
            return self.fault_response(session, f"client frame is not msgpack: {exc}")
        try:
            await up.send(bytes(raw))
            resp_raw = await up.recv()
        except Exception as exc:  # noqa: BLE001
            return self.fault_response(session, f"upstream: {type(exc).__name__}: {exc}")
        if isinstance(resp_raw, str):
            return self.fault_response(session, "upstream error frame: " + resp_raw[:500])
        try:
            resp = self.codec.unpack(bytes(resp_raw))
            actions = resp["actions"]
        except Exception as exc:  # noqa: BLE001
            return self.fault_response(session, f"upstream response unusable: {exc}")
        try:
            fused, info = session.on_request(obs, actions)
        except Exception as exc:  # noqa: BLE001
            session.template = actions if session.template is None else session.template
            return self.fault_response(session, f"fuse: {type(exc).__name__}: {exc}")
        resp["actions"] = raise_chunk(actions, fused)
        resp["lictor"] = info
        return resp

    def fault_response(self, session: EpisodeSession, message: str) -> dict:
        try:
            rows = session.hold_chunk()
        except Exception:  # noqa: BLE001 - even the brake needs a started fuse; fall back to a zero chunk
            env = session.fuse.env or {}
            rows = [[0.0] * int(env.get("action_dim", 1)) for _ in range(int(env.get("horizon", 1)))]
        return {
            "actions": raise_chunk(session.template, rows),
            "lictor": {"proxy": PROXY_VERSION, "status": "fault", "action_src": "hold", "error": message, "ticks": 0},
        }


# ----------------------------------------------------------------------------------------------------------------
# --dry-run and the CLI
# ----------------------------------------------------------------------------------------------------------------


def synthetic_observation(env: dict, pos_key: str, reset_key: str) -> dict:
    pos = [float(i) * 0.25 + 1.0 for i in range(env["pos_dim"])]
    return {
        pos_key: make_array(pos, "<f4"),
        "observation/image": make_array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]], "|u1"),
        "prompt": "lictor dry run",
        reset_key: True,
        "meta": {"n": 1, "ok": True, "none": None, "blob": b"\x00\x01\xff", "f": 0.5, "neg": -7, "big": 1 << 40},
    }


def synthetic_chunk(env: dict, pos: Sequence[float]) -> Any:
    rows = [[float(pos[c % len(pos)]) + 0.5 * i for c in range(env["action_dim"])] for i in range(env["horizon"])]
    return make_array(rows, "<f4")


def dry_run(args: argparse.Namespace) -> dict:
    """Round-trip a synthetic observation and chunk through the codec and the executor gate with the echo double."""
    codec = Codec(args.codec)
    env = load_envelope(args.envelope)
    obs = synthetic_observation(env, args.pos_key, args.reset_key)
    packed = codec.pack(obs)
    back = codec.unpack(packed)
    roundtrip = same_value(obs, back)
    fuse = Fuse(client_factory=EchoClient)
    fuse.start(args.envelope, calibration=args.calibration, out=None, mode=args.mode, envelope_digest=ZERO_DIGEST)
    obs_map = ObsMap(args.pos_key, args.vel_key, args.aux_key, args.ext_key, pos_dim=env["pos_dim"], aux_dim=len(env["aux_layout"]))
    session = EpisodeSession(fuse, obs_map, args.exec_steps, args.episode_boundary, args.reset_key)
    chunk = synthetic_chunk(env, obs_map.state(back)["pos"])
    fused, info = session.on_request(back, chunk)
    resp = {"actions": raise_chunk(chunk, fused), "lictor": info, "server_timing": {"infer_ms": 0.0}}
    resp_back = codec.unpack(codec.pack(resp))
    gate = session.exec_steps
    fused_equal = as_f64_rows(resp_back["actions"])[:gate] == as_f64_rows(chunk)[:gate]
    hold_equal = all(r == fused[gate - 1] for r in as_f64_rows(resp_back["actions"])[gate:])
    session.end("truncated")
    fuse.close()
    return {
        "adapter": PROXY_VERSION,
        "status": ADAPTER_STATUS,
        "network": False,
        "fuse": "echo double (no lictor child)",
        "codec": codec.backend,
        "numpy": _numpy() is not None,
        "codec_roundtrip": bool(roundtrip),
        "obs_bytes": len(packed),
        "chunk": {"h": env["horizon"], "d": env["action_dim"], "exec": gate},
        "ticks": info["ticks"],
        "fused_equal_policy": bool(fused_equal),
        "tail_is_hold": bool(hold_equal),
        "envelope": env["path"],
        "embodiment_id": env["embodiment_id"],
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lictor_proxy", description=f"lictor websocket proxy for openpi policy servers ({ADAPTER_STATUS} roadmap adapter)")
    p.add_argument("--listen", default=DEFAULT_LISTEN, help="host:port to serve openpi clients on")
    p.add_argument("--upstream", default=DEFAULT_UPSTREAM, help="ws://host:port of the real WebsocketPolicyServer")
    p.add_argument("--envelope", required=True, help="lictor envelope TOML")
    p.add_argument("--calibration", default=None, help="calibration JSON (arms Tier 1; see the module docstring)")
    p.add_argument("--mode", default="observe", choices=("observe", "enforce"))
    p.add_argument("--out", default=None, help="receipt/ticks/ledger directory for the lictor child")
    p.add_argument("--key", default=None, help="signing key for the lictor child")
    p.add_argument("--binary", default=None, help="path to the lictor binary (default: discovery order of lictor_client)")
    p.add_argument("--exec-steps", type=int, default=None, help="rows the client executes per request (default: the envelope's exec_steps)")
    p.add_argument("--episode-boundary", default="reset-key", choices=BOUNDARIES)
    p.add_argument("--reset-key", default=DEFAULT_RESET_KEY, help="observation key whose truthy value starts a new episode")
    p.add_argument("--pos-key", default=DEFAULT_POS_KEY)
    p.add_argument("--vel-key", default=None)
    p.add_argument("--aux-key", default=None)
    p.add_argument("--ext-key", default=None)
    p.add_argument("--codec", default="auto", choices=CODECS)
    p.add_argument("--timeout-s", type=float, default=5.0)
    p.add_argument("--dry-run", action="store_true", help="codec + gate round trip with an in-process echo double; no network")
    p.add_argument("--json", action="store_true")
    return p


def make_session_factory(args: argparse.Namespace) -> Callable[[], EpisodeSession]:
    env = load_envelope(args.envelope)

    def factory() -> EpisodeSession:
        fuse = Fuse(binary=args.binary, timeout_s=args.timeout_s, key=args.key)
        fuse.start(args.envelope, calibration=args.calibration, out=args.out, mode=args.mode)
        obs_map = ObsMap(args.pos_key, args.vel_key, args.aux_key, args.ext_key, pos_dim=env["pos_dim"], aux_dim=len(env["aux_layout"]))
        return EpisodeSession(fuse, obs_map, args.exec_steps, args.episode_boundary, args.reset_key)

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.dry_run:
            out = dry_run(args)
            ok = out["codec_roundtrip"] and out["fused_equal_policy"] and out["tail_is_hold"]
            if args.json:
                print(json.dumps(out, sort_keys=True, indent=2))
            else:
                print(f"lictor_proxy dry-run  status={out['status']}  codec={out['codec']}  numpy={out['numpy']}")
                print(f"  codec round trip   {'ok' if out['codec_roundtrip'] else 'FAIL'}  ({out['obs_bytes']} bytes)")
                print(f"  chunk              h={out['chunk']['h']} d={out['chunk']['d']} exec={out['chunk']['exec']}  ticks={out['ticks']}")
                print(f"  gate               {'ok' if out['fused_equal_policy'] and out['tail_is_hold'] else 'FAIL'}  (echo double, no lictor child, no network)")
            return 0 if ok else 1
        proxy = Proxy(args.listen, args.upstream, make_session_factory(args), Codec(args.codec))
        asyncio.run(proxy.serve())
        return 0
    except LictorAdapterError as exc:
        print(f"lictor_proxy: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 0


__all__ = [
    "ArrayLike",
    "BOUNDARIES",
    "Codec",
    "EchoClient",
    "EpisodeSession",
    "ObsMap",
    "PROXY_VERSION",
    "Proxy",
    "build_parser",
    "dry_run",
    "main",
    "make_array",
    "pack_array",
    "parse_listen",
    "raise_chunk",
    "same_value",
    "synthetic_chunk",
    "synthetic_observation",
    "unpack_array",
]

if __name__ == "__main__":
    sys.exit(main())
