# SPDX-License-Identifier: MIT
"""Tests for the UNVERIFIED openpi proxy stub: import without websockets/msgpack, the msgpack-numpy codec round
trip (mini backend, and byte parity with the real msgpack when it is installed), the executor gate, the fault path
that never drops a connection, episode boundaries, and `--dry-run`."""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVELOPE = ROOT / "envelopes" / "pusht.base.toml"
STATUS_LINE = (
    "STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; "
    "end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12."
)
HEX = "0123456789abcdef" * 4

from adapters.lerobot import lictor_lerobot as ll  # noqa: E402
from adapters.openpi import lictor_proxy as px  # noqa: E402

HAS_MSGPACK = importlib.util.find_spec("msgpack") is not None
HAS_NUMPY = importlib.util.find_spec("numpy") is not None


def test_imports_without_websockets_msgpack_openpi():
    src = (ROOT / "adapters" / "openpi" / "lictor_proxy.py").read_text(encoding="utf-8")
    assert src.splitlines()[1] == "# SPDX-License-Identifier: MIT"
    assert src.splitlines()[2].startswith('"""' + STATUS_LINE)
    assert "UNVERIFIED" in src and src.isascii()
    for mod in ("websockets", "msgpack", "openpi", "openpi_client"):
        assert mod not in sys.modules, mod


SAMPLE = {
    "prompt": "pick up the block",
    "n": 1,
    "neg": -7,
    "u8": 200,
    "u16": 40000,
    "u32": 3_000_000_000,
    "u64": 1 << 40,
    "i8": -100,
    "i16": -30000,
    "i32": -2_000_000_000,
    "i64": -(1 << 40),
    "f": 0.5,
    "nan": math.nan,
    "none": None,
    "t": True,
    "blob": b"\x00\x01\xff" * 100,
    "long_str": "x" * 300,
    "list": [1, [2, 3], {"k": "v"}],
    "big_list": list(range(20)),
    "big_map": {str(i): i for i in range(20)},
    b"bytes_key": 1,
}


def test_mini_codec_roundtrip_of_plain_values():
    codec = px.Codec("mini")
    assert codec.backend == "mini"
    back = codec.unpack(codec.pack(SAMPLE))
    assert px.same_value(SAMPLE, back)
    assert isinstance(back["blob"], bytes) and isinstance(back["prompt"], str) and back[b"bytes_key"] == 1
    with pytest.raises(ValueError):
        codec.unpack(codec.pack(SAMPLE) + b"\x00")
    with pytest.raises(TypeError):
        codec.pack({"x": object()})


def test_mini_codec_roundtrip_of_arrays_with_or_without_numpy():
    codec = px.Codec("mini")
    a = px.make_array([[1.5, 2.0], [3.0, 4.25]], "<f4")
    img = px.make_array([[[0, 1, 2], [3, 4, 5]]], "|u1")
    obs = {"observation/state": a, "observation/image": img, "shape0": px.make_array([], "<f8")}
    back = codec.unpack(codec.pack(obs))
    assert px.same_value(obs, back)
    assert back["observation/state"].tolist() == [[1.5, 2.0], [3.0, 4.25]]
    assert tuple(back["observation/image"].shape) == (1, 2, 3)
    if HAS_NUMPY:
        import numpy as np

        assert isinstance(back["observation/state"], np.ndarray) and back["observation/state"].dtype == np.float32
        assert codec.unpack(codec.pack({"s": np.float32(1.5)}))["s"] == np.float32(1.5)
    else:
        assert isinstance(back["observation/state"], px.ArrayLike)


def test_arraylike_shim_is_numpy_free():
    arr = px.ArrayLike.from_list([[1, 2, 3], [4, 5, 6]], "<i4")
    assert arr.shape == (2, 3) and arr.size == 6 and arr.ndim == 2 and len(arr.tobytes()) == 24
    assert arr.tolist() == [[1, 2, 3], [4, 5, 6]]
    d = px.pack_array(arr)
    assert d[b"__ndarray__"] is True and d[b"dtype"] == "<i4" and d[b"shape"] == (2, 3)
    with pytest.raises(ll.LictorAdapterError):
        px.ArrayLike.from_list([[1, 2], [3]], "<f8")
    with pytest.raises(ll.LictorAdapterError):
        px.ArrayLike((1,), "<c16", b"\x00" * 16).tolist()


@pytest.mark.skipif(not HAS_MSGPACK, reason="msgpack not installed: byte parity with the reference encoder untested")
def test_mini_codec_is_byte_compatible_with_msgpack():
    import msgpack

    mini, real = px.Codec("mini"), px.Codec("msgpack")
    assert real.backend == "msgpack"
    payloads = [SAMPLE, {"observation/state": px.make_array([[1.5, 2.0]], "<f4"), "img": px.make_array([[[7, 8, 9]]], "|u1")}]
    for p in payloads:
        p = {k: v for k, v in p.items() if not (isinstance(v, float) and math.isnan(v))}
        assert mini.pack(p) == real.pack(p)
        assert px.same_value(mini.unpack(real.pack(p)), real.unpack(mini.pack(p)))
    assert msgpack.unpackb(mini.pack({"a": [1, 2]}), raw=False) == {"a": [1, 2]}


def chunk(h=15, d=2, base=(100.0, 200.0)):
    return px.make_array([[base[c] + 3.0 * i + c for c in range(d)] for i in range(h)], "<f4")


def make_session(exec_steps=None, boundary="reset-key", mode="observe", client=px.EchoClient):
    fuse = ll.Fuse(client_factory=client)
    fuse.start(ENVELOPE, mode=mode, envelope_digest=HEX)
    env = fuse.env
    obs_map = px.ObsMap("observation/state", aux_key="observation/aux", pos_dim=env["pos_dim"], aux_dim=len(env["aux_layout"]))
    return px.EpisodeSession(fuse, obs_map, exec_steps, boundary, "reset"), fuse.client


def test_gate_ticks_once_per_executed_row_and_holds_the_tail():
    session, client = make_session()
    obs = {"observation/state": px.make_array([100.0, 200.0, 9.0], "<f4"), "observation/aux": [1.0, 2.0, 3.0, 4.0, 5.0], "prompt": "x"}
    a = chunk()
    fused, info = session.on_request(obs, a)
    ticks = [c for c in client.calls if c[0] == "tick"]
    assert len(ticks) == 8 and [c[1]["idx"] for c in ticks] == list(range(8)) and [c[1]["t"] for c in ticks] == list(range(8))
    assert ticks[0][1]["chunk"]["h"] == 15 and ticks[0][1]["chunk"]["exec"] == 8 and all(c[1]["chunk"] is None for c in ticks[1:])
    assert ticks[0][1]["pos"] == [100.0, 200.0] and ticks[0][1]["aux"] == [1.0, 2.0, 3.0, 4.0]  # pos_dim / aux_layout widths
    rows = ll.as_f64_rows(a)
    assert fused[:8] == rows[:8] and fused[8:] == [rows[7]] * 7  # executed rows fused, the tail is a hold of the last
    assert info["ticks"] == 8 and info["seq"] == 0 and info["t0"] == 0 and info["status"] == "nominal" and info["error"] is None
    out = px.raise_chunk(a, fused)
    assert tuple(out.shape) == (15, 2) and str(out.dtype) in ("float32", "<f4")
    fused2, info2 = session.on_request(obs, chunk(base=(121.0, 221.0)))
    ticks = [c for c in client.calls if c[0] == "tick"]
    assert len(ticks) == 16 and ticks[8][1]["t"] == 8 and ticks[8][1]["chunk"]["seq"] == 1 and info2["t0"] == 8
    assert [c[0] for c in client.calls[:2]] == ["hello", "episode_begin"]
    assert client.calls[1][1]["binding"]["host"]["proxy"] == px.PROXY_VERSION
    with pytest.raises(ll.LictorAdapterError, match="upstream chunk is 14x2"):
        session.on_request(obs, chunk(h=14))
    with pytest.raises(ll.LictorAdapterError, match="lacks pos key"):
        session.on_request({"prompt": "no state"}, chunk())
    with pytest.raises(ll.LictorAdapterError, match="exec-steps"):
        px.EpisodeSession(session.fuse, session.obs_map, 16)


def test_reset_key_boundary_writes_one_receipt_per_episode():
    session, client = make_session()
    obs = {"observation/state": [100.0, 200.0], "reset": True, "lictor/seed": 41, "lictor/init_state_digest": HEX}
    session.on_request(obs, chunk())
    session.on_request({"observation/state": [100.0, 200.0], "reset": False}, chunk())
    session.on_request(obs, chunk())
    kinds = [c[0] for c in client.calls]
    assert kinds.count("episode_begin") == 2 and kinds.count("episode_end") == 1
    begins = [c[1]["run"] for c in client.calls if c[0] == "episode_begin"]
    assert begins[0]["seed"] == 41 and begins[0]["init_state_digest"] == HEX and begins[0]["episode_index"] == 0
    assert begins[1]["episode_index"] == 1 and begins[1]["seed"] == 41
    end = [c[1]["outcome"] for c in client.calls if c[0] == "episode_end"][0]
    assert end["ended_by"] == "truncated" and end["steps"] == 16
    assert session.end() is not None and session.end() is None
    assert session.describe()["episode_boundary"] == "reset-key" and session.describe()["status"] == "UNVERIFIED"


def test_connection_boundary_ignores_the_reset_key():
    session, client = make_session(boundary="connection")
    obs = {"observation/state": [100.0, 200.0], "reset": True}
    session.on_request(obs, chunk())
    session.on_request(obs, chunk())
    assert [c[0] for c in client.calls].count("episode_begin") == 1


class DyingClient(px.EchoClient):
    """Dies on the third tick the way LictorClient does: raises a LictorFault-shaped error, alive = False."""

    class Fault(Exception):
        def __init__(self, reason, last_safe_action):
            super().__init__(reason)
            self.reason = reason
            self.last_safe_action = last_safe_action

    def tick(self, t, idx, pos, aux, chunk=None, **kw):
        if t >= 2:
            self.alive = False
            raise DyingClient.Fault("timeout", self.kwargs["safe_action"]())
        return super().tick(t, idx, pos, aux, chunk, **kw)


def test_fault_returns_a_hold_chunk_and_the_session_survives():
    session, client = make_session(client=DyingClient)
    obs = {"observation/state": [-50.0, 600.0]}
    fused, info = session.on_request(obs, chunk(base=(-50.0, 600.0)))
    assert info["status"] == "fault" and info["action_src"] == "hold" and info["error"] is not None
    assert fused[2:] == [[17.0, 495.0]] * 13  # clamp_box(current measured position) from the fault on
    assert not session.fuse.alive
    hold = session.hold_chunk()
    assert hold == [[17.0, 495.0]] * 15
    fused2, info2 = session.on_request(obs, chunk())  # the dead fuse keeps answering with the brake
    assert fused2 == [[17.0, 495.0]] * 15 and info2["status"] == "fault"
    r = session.end()
    assert r["kind"] == "crash_receipt"


class FakeSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        return self.frames.pop(0)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)


def test_proxy_one_never_drops_the_connection_on_faults():
    import asyncio

    codec = px.Codec("mini")
    session, _client = make_session()
    proxy = px.Proxy("127.0.0.1:0", "ws://127.0.0.1:1", lambda: session, codec)
    assert (proxy.host, proxy.port) == ("127.0.0.1", 0)
    obs = codec.pack({"observation/state": [100.0, 200.0]})
    good = FakeSocket([codec.pack({"actions": chunk(), "server_timing": {"infer_ms": 1.0}})])
    resp = asyncio.run(proxy.one(session, good, obs))
    assert resp["lictor"]["status"] == "nominal" and resp["server_timing"]["infer_ms"] == 1.0 and tuple(resp["actions"].shape) == (15, 2)
    up_err = FakeSocket(["Traceback (most recent call last): boom"])
    resp = asyncio.run(proxy.one(session, up_err, obs))
    assert resp["lictor"]["status"] == "fault" and "upstream error frame" in resp["lictor"]["error"]
    assert tuple(resp["actions"].shape) == (15, 2) and resp["actions"].tolist() == [[100.0, 200.0]] * 15
    resp = asyncio.run(proxy.one(session, FakeSocket([]), "text frame"))
    assert resp["lictor"]["status"] == "fault" and "text frame" in resp["lictor"]["error"]
    resp = asyncio.run(proxy.one(session, FakeSocket([codec.pack({"no_actions": 1})]), obs))
    assert resp["lictor"]["status"] == "fault" and "unusable" in resp["lictor"]["error"]
    resp = asyncio.run(proxy.one(session, FakeSocket([codec.pack({"actions": chunk(h=3)})]), obs))
    assert resp["lictor"]["status"] == "fault" and "fuse:" in resp["lictor"]["error"]
    resp = asyncio.run(proxy.one(session, FakeSocket([]), b"\xc1"))
    assert resp["lictor"]["status"] == "fault" and "not msgpack" in resp["lictor"]["error"]
    codec.unpack(codec.pack(resp))  # every fault response is itself encodable


def test_dry_run_roundtrips_the_codec_without_a_network(capsys):
    rc = px.main(["--dry-run", "--envelope", str(ENVELOPE), "--json", "--codec", "mini"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["codec"] == "mini" and out["codec_roundtrip"] is True and out["fused_equal_policy"] is True and out["tail_is_hold"] is True
    assert out["network"] is False and out["status"] == "UNVERIFIED" and out["chunk"] == {"h": 15, "d": 2, "exec": 8} and out["ticks"] == 8
    rc = px.main(["--dry-run", "--envelope", str(ENVELOPE), "--exec-steps", "3"])
    text = capsys.readouterr().out
    assert rc == 0 and "codec round trip   ok" in text and "exec=3" in text and "ticks=3" in text
    assert px.main(["--dry-run", "--envelope", str(ENVELOPE), "--exec-steps", "99"]) == 2
    if not HAS_MSGPACK:
        assert px.main(["--dry-run", "--envelope", str(ENVELOPE), "--codec", "msgpack"]) == 2


def test_dry_run_as_a_script_from_the_repo_root():
    r = subprocess.run([sys.executable, "adapters/openpi/lictor_proxy.py", "--dry-run", "--envelope", "envelopes/pusht.base.toml", "--json"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["codec_roundtrip"] is True


@pytest.mark.skipif(importlib.util.find_spec("websockets") is None, reason="websockets not installed: loopback proxy path untested")
def test_loopback_proxy_against_a_fake_openpi_shaped_upstream():
    """A real websocket round trip: fake upstream (metadata frame, then one chunk per observation, a text frame on
    demand) -> proxy -> client. Still not a live openpi server; the UNVERIFIED label stays."""
    import asyncio

    serve, connect, _closed = px._websockets()
    codec = px.Codec("auto")
    clients = []

    async def fake_upstream(ws):
        await ws.send(codec.pack({"policy": "fake-pi", "version": 1}))
        async for raw in ws:
            obs = codec.unpack(raw)
            if obs.get("boom"):
                await ws.send("Traceback (most recent call last): boom")  # the stock server's error frame
                continue
            await ws.send(codec.pack({"actions": chunk(), "server_timing": {"infer_ms": 1.0}}))

    def factory():
        session, client = make_session()
        clients.append(client)
        return session

    async def run():
        async with serve(fake_upstream, "127.0.0.1", 0, compression=None, max_size=None) as up_server:
            up_port = next(iter(up_server.sockets)).getsockname()[1]
            proxy = px.Proxy("127.0.0.1:0", f"ws://127.0.0.1:{up_port}", factory, codec)
            async with serve(proxy.handler, proxy.host, proxy.port, compression=None, max_size=None) as px_server:
                px_port = next(iter(px_server.sockets)).getsockname()[1]
                async with connect(f"ws://127.0.0.1:{px_port}", compression=None, max_size=None) as client:
                    meta = codec.unpack(await client.recv())
                    assert meta["policy"] == "fake-pi" and meta["lictor"]["status"] == "UNVERIFIED" and meta["lictor"]["exec_steps"] == 8
                    await client.send(codec.pack({"observation/state": [100.0, 200.0], "reset": True}))
                    resp = codec.unpack(await client.recv())
                    assert resp["lictor"]["status"] == "nominal" and resp["lictor"]["ticks"] == 8 and resp["server_timing"]["infer_ms"] == 1.0
                    assert tuple(resp["actions"].shape) == (15, 2) and resp["actions"].tolist()[:8] == chunk().tolist()[:8]
                    await client.send(codec.pack({"observation/state": [100.0, 200.0], "boom": True}))
                    resp = codec.unpack(await client.recv())
                    assert resp["lictor"]["status"] == "fault" and "upstream error frame" in resp["lictor"]["error"]
                    assert resp["actions"].tolist() == [[100.0, 200.0]] * 15
                    await client.send(codec.pack({"observation/state": [100.0, 200.0]}))  # the connection survived
                    resp = codec.unpack(await client.recv())
                    assert resp["lictor"]["status"] == "nominal" and resp["lictor"]["t0"] == 8
                for _ in range(50):  # the handler's finally runs after the client closed
                    if clients and clients[0].calls and clients[0].calls[-1][0] == "close":
                        break
                    await asyncio.sleep(0.02)
        assert proxy.connections == 1
        kinds = [c[0] for c in clients[0].calls]
        assert kinds[:2] == ["hello", "episode_begin"] and kinds.count("episode_end") == 1 and kinds[-1] == "close"
        assert kinds.count("tick") == 16
        assert clients[0].calls[-2][1]["outcome"]["ended_by"] == "truncated"

    asyncio.run(run())


def test_parse_listen_and_serve_without_websockets():
    assert px.parse_listen("0.0.0.0:8001") == ("0.0.0.0", 8001) and px.parse_listen(":9") == ("0.0.0.0", 9)
    with pytest.raises(ll.LictorAdapterError):
        px.parse_listen("nope")
    if importlib.util.find_spec("websockets") is None:
        rc = px.main(["--envelope", str(ENVELOPE), "--listen", "127.0.0.1:0"])
        assert rc == 2
