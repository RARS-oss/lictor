# SPDX-License-Identifier: MIT
"""Tests for adapters/lictor_client.py.

Pure tests (no binary): finite_or_none / chunk_msg / allow_nan=False, path bridging, the null -> inf mapping and
the hello asserts against a fake child (a tiny Python echo server), kill-on-timeout and kill-on-exit.

Live tests (the real `lictor serve` binary, discovered like the client does): hello/bye, a 20-tick synthetic
episode with a receipt, null -> inf mapping on the first tick, and the tier1/alpha asserts. They are skipped when
no binary is found or when the served fuse is still the WP-3 stub (the probe below detects the panic).
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from adapters import lictor_client as lc
from adapters.lictor_client import CLIENT_VERSION, LictorClient, LictorFault, chunk_msg, finite_or_none

ROOT = Path(__file__).resolve().parents[2]
ENVELOPE = ROOT / "envelopes" / "pusht.base.toml"


# ---- pure ----------------------------------------------------------------------------------------------------------


def test_finite_or_none_masks_and_preserves_shape():
    assert finite_or_none([1.0, float("nan"), 3]) == [1.0, None, 3.0]
    assert finite_or_none(np.array([[1.0, np.inf], [-np.inf, 2.5]])) == [[1.0, None], [None, 2.5]]
    assert finite_or_none(np.float32(0.1)) == 0.10000000149011612  # float64 repr of the float32 value
    assert finite_or_none([]) == []
    assert finite_or_none(float("nan")) is None


def test_chunk_msg_shape_and_nan():
    a = np.arange(30, dtype=np.float64).reshape(15, 2)
    a[3, 1] = np.nan
    m = chunk_msg(4, 32, a, 8)
    assert (m["seq"], m["t_emit"], m["h"], m["d"], m["exec"]) == (4, 32, 15, 2, 8)
    assert m["a"][3] == [6.0, None]
    assert len(m["a"]) == 15
    text = json.dumps(m, allow_nan=False, separators=(",", ":"))
    assert "NaN" not in text and "null" in text
    with pytest.raises(ValueError):
        chunk_msg(0, 0, np.zeros(15), 8)


def test_dumps_refuses_nan_literals():
    with pytest.raises(ValueError):
        lc._dumps({"pos": [float("nan")]})
    assert lc._dumps({"a": [1.5, None]}) == '{"a":[1.5,null]}'


def test_path_bridging():
    assert lc.win_to_wsl("C:\\Users\\x\\f.txt") == "/mnt/c/Users/x/f.txt"
    assert lc.win_to_wsl("/mnt/d/lictor") == "/mnt/d/lictor"
    assert lc.wsl_to_win("/mnt/c/Users/x/f.txt") == "C:\\Users\\x\\f.txt"
    assert lc.wsl_to_win("/home/u/f") == "/home/u/f"
    assert lc.wsl_to_win(lc.win_to_wsl("D:\\lictor\\a.b")) == "D:\\lictor\\a.b"


def test_lictor_fault_carries_reason_and_safe_action():
    f = LictorFault("timeout", "x", last_safe_action=[1.0, 2.0])
    assert f.reason == "timeout" and f.last_safe_action == [1.0, 2.0]
    assert "timeout" in str(f)


# ---- a fake child: a Python echo server speaking just enough of the wire ---------------------------------------

FAKE_CHILD = textwrap.dedent(
    """
    import json, math, sys, time, os
    mode = os.environ.get("FAKE_MODE", "ok")
    for line in sys.stdin:
        req = json.loads(line)
        if "NaN" in line:
            raise SystemExit(9)
        k = req["kind"]
        i = req["id"]
        if mode == "hang" and k == "tick":
            time.sleep(30)
        if mode == "die" and k == "tick":
            raise SystemExit(7)
        if mode == "badid" and k == "tick":
            i = i + 1
        if k == "hello":
            cal = None
            if os.environ.get("FAKE_TIER1") == "1":
                cal = {"method": "binned", "alpha_num": 5, "alpha_den": 100, "n_calib": 137, "tau": None,
                       "kn": [3, 5], "gate": ["tce"], "digest": "e1" * 32}
            resp = {"id": i, "kind": "hello_ok", "proto": "lictor-wire/v1", "lictor": "0.1.0", "git": "nogit",
                    "lictor_sha256": "00" * 32, "envelope_digest": req["envelope_digest"],
                    "embodiment_digest": "11" * 32, "calibration_digest": req["calibration_digest"],
                    "pubkey": "22" * 32, "ephemeral_key": True, "mode": req["mode"], "tier0_armed": [],
                    "tier1_armed": cal is not None, "calibration": cal, "features": [], "trip_names": [],
                    "latency_label": "fake"}
        elif k == "episode_begin":
            resp = {"id": i, "kind": "episode_ok", "state": "armed", "seq": 0}
        elif k == "tick":
            if mode == "fatal":
                resp = {"id": i, "kind": "error", "code": "schema", "message": "boom", "fatal": True}
            else:
                pos = req["obs"]["pos"]
                resp = {"id": i, "kind": "verdict", "t": req["t"], "seq": req["t"], "status": "nominal",
                        "state": "armed", "prev_state": "armed", "trips": [], "trip_mask": 0,
                        "action": [0.0 if p is None else p for p in pos], "action_src": "policy",
                        "substituted": False, "clamped_dims": 0,
                        "scores": {"f": [0.0] * 12, "z": [0.0] * 12, "s": None, "valid": 0, "fired": 0},
                        "tau": None, "window_hits": 0, "brake_margin": 1.5, "reason": "ok", "reason_text": "ok",
                        "handoff": None, "ack_result": None, "violation_reached_env": False, "verdict_ns": 1,
                        "chain": "33" * 32}
        elif k == "episode_end":
            resp = {"id": i, "kind": "episode_receipt", "receipt_path": "", "ticks_path": "",
                    "body_digest": "44" * 32, "verdict_chain_head": "33" * 32, "timing_chain_head": "55" * 32,
                    "fuse_ok": False, "fuse_notes": ["ephemeral signing key"], "counts": {}, "ledger_seq": 0,
                    "ledger_head": "0" * 64}
        elif k == "bye":
            resp = {"id": i, "kind": "bye_ok"}
            sys.stdout.write(json.dumps(resp) + "\\n"); sys.stdout.flush()
            break
        sys.stdout.write(json.dumps(resp) + "\\n")
        sys.stdout.flush()
    """
)


@pytest.fixture
def fake_binary(tmp_path, monkeypatch):
    """A shell wrapper that ignores `serve ...` arguments and runs the fake child."""
    script = tmp_path / "fake_child.py"
    script.write_text(FAKE_CHILD)
    wrapper = tmp_path / "lictor"
    wrapper.write_text("#!/bin/sh\nexec %s %s\n" % (sys.executable, script))
    wrapper.chmod(0o755)
    monkeypatch.delenv("LICTOR_BIN", raising=False)
    return wrapper


def _client(binary, mode="observe", **kw):
    return LictorClient(binary, ENVELOPE, mode=mode, timeout_s=kw.pop("timeout_s", 2.0), **kw)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes required")
def test_fake_child_round_trip_maps_null_to_inf(fake_binary):
    c = _client(fake_binary, safe_action=lambda: np.array([12.0, 13.0]))
    try:
        h = c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, None, False, None)
        assert h["kind"] == "hello_ok" and h["id"] == 1
        assert c.episode_begin({"run_id": "r"}, {}, {})["kind"] == "episode_ok"
        v = c.tick(0, 0, [1.0, float("nan")], [0.0], chunk=chunk_msg(0, 0, np.zeros((15, 2)), 8), vel=[0.0, 0.0])
        assert v["kind"] == "verdict" and v["id"] == 3
        assert v["scores"]["s"] == -math.inf
        assert v["tau"] == math.inf
        assert v["brake_margin"] == 1.5  # other numbers untouched
        assert v["action"] == [1.0, 0.0]  # the NaN went out as null
        r = c.episode_end({"steps": 1, "success": False, "terminated": False, "truncated": True,
                           "max_coverage": float("nan"), "final_coverage": 0.5, "reward_sum": None, "ended_by": "truncated"})
        assert r["kind"] == "episode_receipt"
        assert c.alive
    finally:
        c.close()
    assert not c.alive


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes required")
def test_hello_asserts_tier1_and_alpha(fake_binary, monkeypatch):
    c = _client(fake_binary)
    with pytest.raises(LictorFault) as ei:
        c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, None, True, (5, 100))
    assert ei.value.reason == "protocol" and not c.alive
    monkeypatch.setenv("FAKE_TIER1", "1")
    c = _client(fake_binary)
    try:
        h = c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, "e1" * 32, True, (5, 100))
        assert h["calibration"]["tau"] == math.inf  # hello_ok.calibration.tau null -> +inf
    finally:
        c.close()
    c = _client(fake_binary)
    with pytest.raises(LictorFault) as ei:
        c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, "e1" * 32, True, (1, 100))
    assert ei.value.reason == "protocol"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes required")
def test_timeout_kills_the_child(fake_binary, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "hang")
    c = _client(fake_binary, timeout_s=0.5, safe_action=lambda: [100.0, 200.0])
    c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, None, False, None)
    with pytest.raises(LictorFault) as ei:
        c.tick(0, 0, [1.0, 2.0], [0.0])
    assert ei.value.reason == "timeout"
    assert ei.value.last_safe_action == [100.0, 200.0]
    assert not c.alive
    assert c.proc.poll() is not None


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes required")
def test_child_exit_and_id_mismatch(fake_binary, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "die")
    c = _client(fake_binary)
    c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, None, False, None)
    with pytest.raises(LictorFault) as ei:
        c.tick(0, 0, [1.0, 2.0], [0.0])
    assert ei.value.reason == "exit" and not c.alive
    monkeypatch.setenv("FAKE_MODE", "badid")
    c = _client(fake_binary)
    c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, None, False, None)
    with pytest.raises(LictorFault) as ei:
        c.tick(0, 0, [1.0, 2.0], [0.0])
    assert ei.value.reason == "id_mismatch" and not c.alive


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX pipes required")
def test_fatal_error_from_a_live_child_is_returned(fake_binary, monkeypatch):
    monkeypatch.setenv("FAKE_MODE", "fatal")
    c = _client(fake_binary)
    c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, "ab" * 32, None, False, None)
    r = c.tick(0, 0, [1.0, 2.0], [0.0])
    assert r["kind"] == "error" and r["fatal"] and c.alive
    assert c.last_error is r
    c.close()


# ---- live: the real binary ---------------------------------------------------------------------------------------


def _real_binary():
    try:
        return lc.find_binary(None)
    except FileNotFoundError:
        return None


def _envelope_digest(binary):
    """`lictor envelope digest`, or -- while that command is another package's stub -- the digest the server
    itself reports in its hello mismatch message (`envelope_digest <sent> != loaded <hex>`)."""
    out = subprocess.run([binary, "envelope", "digest", str(ENVELOPE)], capture_output=True, text=True, timeout=60)
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip().split()[0]
    hello = {"id": 1, "kind": "hello", "proto": "lictor-wire/v1", "client": "probe", "mode": "observe",
             "embodiment_id": "gym_pusht/PushT-v0", "action_dim": 2, "pos_dim": 2, "horizon": 15, "exec_steps": 8,
             "envelope_digest": "00" * 32, "calibration_digest": None}
    out = subprocess.run([binary, "serve", "--envelope", str(ENVELOPE), "--mode", "observe"],
                         input=json.dumps(hello) + "\n", capture_output=True, text=True, timeout=60)
    for line in out.stdout.splitlines():
        try:
            resp = json.loads(line)
        except ValueError:
            continue
        msg = resp.get("message", "")
        if "!= loaded " in msg:
            tail = msg.split("!= loaded ", 1)[1]
            return tail[:64]
    return None


@pytest.fixture(scope="module")
def live():
    binary = _real_binary()
    if binary is None or sys.platform == "win32":
        pytest.skip("no lictor binary found (set $LICTOR_BIN or build with cargo build --release -p lictor-cli)")
    digest = _envelope_digest(binary)
    if digest is None or len(digest) != 64:
        pytest.skip("pending WP-7: `lictor envelope digest` is not implemented yet")
    probe = subprocess.run([binary, "serve", "--envelope", str(ENVELOPE), "--mode", "observe"],
                           input='{"id":1,"kind":"bye"}\n', capture_output=True, text=True, timeout=60)
    if probe.returncode != 0 or "bye_ok" not in probe.stdout:
        pytest.skip("pending WP-3: lictor serve cannot start (%s)" % probe.stderr.strip().splitlines()[-1:])
    return binary, digest


def _synthetic_episode(c: LictorClient, n=20):
    hello = c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, c._digest, None, False, None)
    assert hello["proto"] == "lictor-wire/v1" and hello["tier1_armed"] is False
    assert c.episode_begin(
        {"run_id": "py-run", "arm_id": "obs-d0", "episode_index": 0, "seed": 0, "seed_pool": "eval", "init_state_digest": "3f" * 32},
        {"delay_steps": 0, "tick_ms": 100, "exec_mode": "sync", "stitch": "drop", "on_escalate": "terminate_fail"},
        {"env": {"env_id": "gym_pusht/PushT-v0"}, "policy": {"weights_sha256": "ab" * 32}, "host": {}},
        inputs={"harness/x.py": "11" * 32},
    )["kind"] == "episode_ok"
    first = None
    for t in range(n):
        pos = np.array([150.0 + t, 150.0 + 0.5 * t])
        chunk = None
        if t % 8 == 0:
            rows = np.stack([pos + np.array([1.0, 0.5]) * (i + 1) for i in range(15)])
            chunk = chunk_msg(t // 8, t, rows, 8)
        v = c.tick(t, t % 8, pos, [256.0, 256.0, 0.0, 0.1], chunk=chunk, vel=[10.0, 5.0])
        assert v["kind"] == "verdict", v
        assert v["status"] == "nominal", v
        if first is None:
            first = v
    r = c.episode_end({"steps": n, "success": False, "terminated": False, "truncated": True,
                       "max_coverage": 0.2, "final_coverage": 0.2, "reward_sum": 3.0, "ended_by": "truncated"})
    return first, r


def test_live_hello_bye(live):
    binary, digest = live
    c = LictorClient(binary, ENVELOPE, mode="observe", timeout_s=10.0)
    try:
        h = c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, digest, None, False, None)
        assert h["envelope_digest"] == digest
        assert h["ephemeral_key"] is True
        assert len(h["trip_names"]) == 16 and len(h["features"]) == 12
        assert "not a real-time environment" in h["latency_label"]
    finally:
        c.close()
    assert not c.alive


def test_live_synthetic_episode_writes_a_receipt(live, tmp_path):
    binary, digest = live
    c = LictorClient(binary, ENVELOPE, mode="observe", out_dir=tmp_path, trace=tmp_path / "trace.ndjson", timeout_s=10.0)
    c._digest = digest
    try:
        first, r = _synthetic_episode(c, 20)
        # Tier 1 disarmed: tau is null on the wire -> +inf; s is finite once a gate term is valid (the envelope
        # gate is embedded even without a calibration), so only its type is checked here
        assert first["tau"] == math.inf
        assert isinstance(first["scores"]["s"], float)
        assert r["kind"] == "episode_receipt", r
        assert r["counts"]["ticks"] == 20
        assert r["fuse_ok"] is False  # observe + ephemeral key
        assert "ephemeral signing key" in r["fuse_notes"]
        receipt = tmp_path / r["receipt_path"]
        assert receipt.is_file()
        body = json.loads(receipt.read_text())["body"]
        assert body["client"] == CLIENT_VERSION
        assert body["counts"]["ticks"] == 20 and body["verdict_events"] == 20 and body["outcome"]["steps"] == 20
        assert body["inputs"]["harness/x.py"] == "11" * 32 and "lictor:bin" in body["inputs"]
        assert (tmp_path / r["ticks_path"]).is_file()
        assert (tmp_path / "py-run" / "obs-d0" / "ledger.jsonl").is_file()
        assert (tmp_path / "py-run" / "obs-d0" / "traces" / "000000.ndjson").is_file()
        trace = (tmp_path / "trace.ndjson").read_text().splitlines()
        assert trace[0].startswith("#meta ") and len(trace) == 1 + 1 + 1 + 20 + 1
    finally:
        c.close()


def test_live_hello_tier1_assert_refuses_a_t0_server(live):
    binary, digest = live
    c = LictorClient(binary, ENVELOPE, mode="observe", timeout_s=10.0)
    with pytest.raises(LictorFault) as ei:
        c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, digest, None, True, (5, 100))
    assert ei.value.reason == "protocol" and not c.alive


def test_live_digest_mismatch_is_fatal_and_returned(live):
    binary, digest = live
    c = LictorClient(binary, ENVELOPE, mode="observe", timeout_s=10.0)
    try:
        r = c._request({"kind": "hello", "proto": "lictor-wire/v1", "client": CLIENT_VERSION, "mode": "observe",
                        "embodiment_id": "gym_pusht/PushT-v0", "action_dim": 2, "pos_dim": 2, "horizon": 15,
                        "exec_steps": 8, "envelope_digest": "00" * 32, "calibration_digest": None})
        assert r["kind"] == "error" and r["code"] == "envelope" and r["fatal"] is True
        assert c.alive  # a fatal error from a live child is returned, not raised
    finally:
        c.close()


def test_live_kill_on_timeout_with_the_real_binary(live):
    binary, digest = live
    c = LictorClient(binary, ENVELOPE, mode="observe", timeout_s=0.3, safe_action=lambda: [1.0, 2.0])
    c.hello("gym_pusht/PushT-v0", 2, 2, 15, 8, digest, None, False, None)
    # a request without a newline flush never gets an answer: emulate by writing a partial line directly
    c.proc.stdin.write('{"id":2,"kind":"bye"')
    c.proc.stdin.flush()
    with pytest.raises(LictorFault) as ei:
        c._read_line(0.3)
    assert ei.value.reason == "timeout" and ei.value.last_safe_action == [1.0, 2.0]
    assert not c.alive and c.proc.poll() is not None
