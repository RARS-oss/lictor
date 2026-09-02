# SPDX-License-Identifier: MIT
"""Tests for the UNVERIFIED LeRobot adapter stub: import without lerobot, the Fuse wrapper over a mocked
LictorClient, and LictorStep replacing the transition's action with verdict.action."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import pathlib
import struct
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENVELOPE = ROOT / "envelopes" / "pusht.base.toml"
STATUS_LINE = (
    "STATUS: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; "
    "end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12."
)
HEX = "0123456789abcdef" * 4

LEROBOT_WAS_LOADED = "lerobot" in sys.modules

from adapters.lerobot import lictor_lerobot as ll  # noqa: E402


class FakeFault(Exception):
    """Shape-compatible with adapters.lictor_client.LictorFault (which may not exist yet)."""

    def __init__(self, reason: str, last_safe_action=None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.last_safe_action = last_safe_action


class FakeClient:
    """Records every frozen-surface call; `verdict_fn(t, idx, row)` decides the returned action."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.alive = True
        self.chunk = None
        self.verdict_fn = None
        self.fail_at = None

    def hello(self, **kw):
        self.calls.append(("hello", kw))
        return {"kind": "hello_ok", "tier1_armed": kw["expect_tier1"], "mode": self.kwargs["mode"]}

    def episode_begin(self, run, budget, binding, fault_injection=None, inputs=None):
        self.calls.append(("episode_begin", {"run": run, "budget": budget, "binding": binding, "fault_injection": fault_injection, "inputs": inputs}))
        return {"kind": "episode_ok", "state": "armed", "seq": 0}

    def tick(self, t, idx, pos, aux, chunk=None, vel=None, ext=(), missed_ticks=0, ack=None):
        self.calls.append(("tick", {"t": t, "idx": idx, "pos": list(pos), "aux": list(aux), "chunk": chunk, "vel": vel, "ext": list(ext), "missed_ticks": missed_ticks, "ack": ack}))
        if self.fail_at is not None and t >= self.fail_at:
            self.alive = False
            raise FakeFault("timeout", last_safe_action=self.kwargs["safe_action"]())
        if chunk is not None:
            self.chunk = chunk["a"]
        row = [float(v) for v in self.chunk[idx]]
        action = self.verdict_fn(t, idx, row) if self.verdict_fn else row
        substituted = action != row
        return {
            "kind": "verdict", "t": t, "seq": t, "status": "clamped" if substituted else "nominal", "state": "armed",
            "prev_state": "armed", "trips": ["workspace"] if substituted else [], "trip_mask": 1 if substituted else 0,
            "action": action, "action_src": "clamp" if substituted else "policy", "substituted": substituted,
            "clamped_dims": 0, "scores": {"f": [], "z": [], "s": -math.inf, "valid": 0, "fired": 0}, "tau": math.inf,
            "window_hits": 0, "brake_margin": 0.0, "reason": "ok", "reason_text": "x", "handoff": None,
            "ack_result": None, "violation_reached_env": False, "verdict_ns": 1, "chain": HEX,
        }

    def episode_end(self, outcome):
        self.calls.append(("episode_end", {"outcome": outcome}))
        return {"kind": "episode_receipt", "fuse_ok": True, "counts": {"ticks": outcome["steps"]}}

    def close(self):
        self.calls.append(("close", {}))
        self.alive = False


def make_fuse(**kw) -> tuple[ll.Fuse, FakeClient]:
    holder = {}

    def factory(**kwargs):
        holder["client"] = FakeClient(**kwargs)
        return holder["client"]

    fuse = ll.Fuse(client_factory=factory, **kw)
    return fuse, holder


def chunk_rows(h=15, d=2, base=(100.0, 200.0)):
    return [[base[c] + 3.0 * i + c for c in range(d)] for i in range(h)]


# ----------------------------------------------------------------------------------------------------------------


def test_imports_without_lerobot_torch_numpy_at_import_time():
    src = (ROOT / "adapters" / "lerobot" / "lictor_lerobot.py").read_text(encoding="utf-8")
    assert src.startswith("# SPDX-License-Identifier: MIT\n")
    assert src.splitlines()[1].startswith('"""' + STATUS_LINE)
    assert "UNVERIFIED" in src
    assert src.isascii()
    if not LEROBOT_WAS_LOADED:
        assert "lerobot" not in sys.modules
        assert "torch" not in sys.modules
    assert ll.LictorStep.__mro__[1] is ll._DuckProcessorStep or ll.LictorStep.__mro__[1].__name__ == "ProcessorStep"


def test_conversions_and_clamp_box():
    assert ll.as_f64_row([[1, 2.5]]) == [1.0, 2.5]
    assert ll.as_f64_rows([[[1, 2], [3, 4]]]) == [[1.0, 2.0], [3.0, 4.0]]
    with pytest.raises(ll.LictorAdapterError):
        ll.as_f64_row([[1, 2], [3, 4]])  # B = 2
    assert ll.finite_or_none([1.0, math.nan, math.inf, 2]) == [1.0, None, None, 2]
    assert ll.clamp_box([-5.0, 600.0, math.nan], [0.0, 0.0, 0.0], [512.0, 512.0, 10.0]) == [0.0, 512.0, 5.0]
    msg = ll.chunk_message(3, 24, [[1.0, math.nan], [2.0, 3.0]], 8)
    assert msg == {"seq": 3, "t_emit": 24, "h": 2, "d": 2, "exec": 8, "a": [[1.0, None], [2.0, 3.0]]}
    json.dumps(msg, allow_nan=False)
    assert ll.init_state_digest_of([0, 0]) == hashlib.sha256(b"\x00" * 16).hexdigest()
    assert ll.init_state_digest_of([[1.0, 2.0]]) == hashlib.sha256(struct.pack("<2d", 1.0, 2.0)).hexdigest()
    np = pytest.importorskip("numpy")
    a = np.asarray([[1.5, 2.5]], dtype=np.float32)
    out = ll.restore_like(a, [9.0, 8.0])
    assert out.dtype == np.float32 and out.shape == (1, 2) and out.tolist() == [[9.0, 8.0]]


def test_load_envelope_reads_the_frozen_embodiment():
    env = ll.load_envelope(ENVELOPE)
    assert env["embodiment_id"] == "gym_pusht/PushT-v0"
    assert (env["action_dim"], env["pos_dim"], env["horizon"], env["exec_steps"]) == (2, 2, 15, 8)
    assert env["action_kind"] == "ee_position" and env["tick_ms"] == 100
    assert env["box_lo"] == [17.0, 17.0] and env["box_hi"] == [495.0, 495.0]  # margin-adjusted
    assert env["aux_layout"] == ["block_x", "block_y", "block_theta", "coverage"]


def test_fuse_start_begin_tick_end_over_mock_client(tmp_path):
    fuse, holder = make_fuse()
    hello = fuse.start(ENVELOPE, calibration=None, out=tmp_path, mode="enforce", envelope_digest=HEX)
    client = holder["client"]
    assert hello["kind"] == "hello_ok"
    assert client.kwargs["mode"] == "enforce" and client.kwargs["out_dir"] == tmp_path
    assert client.kwargs["envelope"] == ENVELOPE and client.kwargs["calibration"] is None
    assert callable(client.kwargs["safe_action"])
    kind, kw = client.calls[0]
    assert kind == "hello"
    assert kw == {
        "embodiment_id": "gym_pusht/PushT-v0", "action_dim": 2, "pos_dim": 2, "horizon": 15, "exec_steps": 8,
        "envelope_digest": HEX, "calibration_digest": None, "expect_tier1": False, "expect_alpha": None,
    }
    r = fuse.begin(episode=7, seed=7, init_state_digest=HEX, binding={"policy": {"weights_sha256": HEX}})
    assert r["kind"] == "episode_ok"
    kind, kw = client.calls[1]
    assert kind == "episode_begin"
    assert kw["run"] == {"run_id": fuse.run_id, "arm_id": "lerobot-adapter", "episode_index": 7, "seed": 7, "seed_pool": "eval", "init_state_digest": HEX}
    assert kw["budget"] == {"delay_steps": 0, "tick_ms": 100, "exec_mode": "sync", "stitch": "drop", "on_escalate": "terminate_fail"}
    b = kw["binding"]
    assert set(b) == {"env", "policy", "host"}
    assert all(isinstance(v, str) for sec in b.values() for v in sec.values())
    assert b["env"]["env_id"] == "gym_pusht/PushT-v0" and b["env"]["init_state_digest_source"] == "host"
    assert b["policy"]["weights_sha256"] == HEX
    assert b["host"]["adapter"] == ll.ADAPTER_VERSION and b["host"]["adapter_status"] == "UNVERIFIED"
    assert kw["inputs"]["adapters/lerobot/lictor_lerobot.py"] == ll.sha256_file(ll.__file__)
    assert kw["inputs"]["envelopes/pusht.base.toml"] == ll.sha256_file(ENVELOPE)
    json.dumps(kw, allow_nan=False, sort_keys=True)

    rows = chunk_rows()
    state = {"pos": [100.0, 200.0], "vel": [0.0, 0.0], "aux": [256.0, 256.0, 0.0, 0.5]}
    v = fuse.tick(state, rows[0], chunk=rows)
    assert v["action"] == rows[0] and v["status"] == "nominal"
    t0 = client.calls[-1][1]
    assert (t0["t"], t0["idx"], t0["pos"], t0["vel"], t0["aux"]) == (0, 0, [100.0, 200.0], [0.0, 0.0], [256.0, 256.0, 0.0, 0.5])
    assert t0["chunk"] == {"seq": 0, "t_emit": 0, "h": 15, "d": 2, "exec": 8, "a": rows}
    for i in range(1, 8):
        v = fuse.tick(state, rows[i])
        tk = client.calls[-1][1]
        assert (tk["t"], tk["idx"], tk["chunk"]) == (i, i, None)
    rows2 = chunk_rows(base=(121.0, 221.0))
    fuse.tick({"pos": [121.0, 221.0]}, rows2[0], chunk=rows2)
    tk = client.calls[-1][1]
    assert tk["chunk"]["seq"] == 1 and tk["chunk"]["t_emit"] == 8 and (tk["t"], tk["idx"]) == (8, 0)
    r = fuse.end(success=True, steps=9, progress=0.97)
    assert r["kind"] == "episode_receipt"
    outcome = client.calls[-1][1]["outcome"]
    assert outcome == {"steps": 9, "success": True, "terminated": True, "truncated": False, "max_coverage": 0.97, "final_coverage": 0.97, "reward_sum": None, "ended_by": "success"}
    assert not fuse.open
    fuse.close()
    assert client.calls[-1][0] == "close" and fuse.client is None


def test_tick_contract_violations_raise_before_the_wire():
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    with pytest.raises(ll.LictorAdapterError):
        fuse.tick({"pos": [1.0, 2.0]}, [1.0, 2.0])  # before begin
    fuse.begin(0, 0)
    rows = chunk_rows()
    with pytest.raises(ll.LictorAdapterError, match="first tick"):
        fuse.tick({"pos": [1.0, 2.0]}, rows[0])
    with pytest.raises(ll.LictorAdapterError, match="never trim rows"):
        fuse.tick({"pos": [1.0, 2.0]}, rows[0], chunk=rows[:14])
    fuse.tick({"pos": [1.0, 2.0]}, rows[0], chunk=rows)
    with pytest.raises(ll.LictorAdapterError, match="executor desync"):
        fuse.tick({"pos": [1.0, 2.0]}, rows[5])  # host executes row 5 while the fuse indexes row 1
    n = len(holder["client"].calls)
    for i in range(2, 15):
        fuse.tick({"pos": [1.0, 2.0]}, rows[i])
    with pytest.raises(ll.LictorAdapterError, match="past the chunk horizon"):
        fuse.tick({"pos": [1.0, 2.0]}, rows[14])
    assert len(holder["client"].calls) == n + 13  # the rejected ticks never reached the client
    with pytest.raises(ll.LictorAdapterError, match="mode"):
        ll.Fuse(client_factory=FakeClient).start(ENVELOPE, mode="monitor", envelope_digest=HEX)


def test_missed_ticks_advance_t_and_idx():
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    fuse.begin(0, 0)
    rows = chunk_rows()
    fuse.tick({"pos": [1.0, 2.0]}, rows[0], chunk=rows)
    fuse.tick({"pos": [1.0, 2.0]}, rows[3], missed_ticks=2)
    tk = holder["client"].calls[-1][1]
    assert (tk["t"], tk["idx"], tk["missed_ticks"]) == (3, 3, 2)
    fuse.tick({"pos": [1.0, 2.0]}, rows[4])
    assert holder["client"].calls[-1][1]["t"] == 4


def test_fault_becomes_clamp_box_and_never_the_raw_action():
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    client = holder["client"]
    client.fail_at = 2
    fuse.begin(0, 0)
    rows = chunk_rows(base=(-40.0, 700.0))  # the policy wants to leave the box
    fuse.tick({"pos": [-40.0, 700.0]}, rows[0], chunk=rows)
    fuse.tick({"pos": [-40.0, 700.0]}, rows[1])
    v = fuse.tick({"pos": [-40.0, 700.0]}, rows[2])
    assert v["synthetic"] is True and v["status"] == "fault" and v["action_src"] == "hold"
    assert v["action"] == [17.0, 495.0]  # clamp_box(current measured position) with the 2 px margin
    assert fuse.dead and not fuse.alive
    v2 = fuse.tick({"pos": [10.0, 10.0]}, rows[3])
    assert v2["action"] == [17.0, 17.0] and v2["action"] != rows[3]
    assert client.calls[-1][1]["t"] == 2  # nothing was sent to the dead child afterwards
    r = fuse.end(success=False, steps=3)
    assert r["kind"] == "crash_receipt" and r["written"] is False  # no binary/out dir -> reported, not hidden
    assert not fuse.open


def test_fatal_error_with_live_child_holds_then_ends_as_fault():
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    client = holder["client"]
    fuse.begin(0, 0)
    rows = chunk_rows()
    fuse.tick({"pos": [100.0, 200.0]}, rows[0], chunk=rows)
    client.tick = lambda **kw: {"id": 9, "kind": "error", "code": "schema", "message": "x", "fatal": True}
    v = fuse.tick({"pos": [100.0, 200.0]}, rows[1])
    assert v["synthetic"] and v["action"] == [100.0, 200.0] and "schema" in v["trips"]
    assert fuse.faulted and not fuse.dead and fuse.alive
    fuse.end(success=False, steps=2)
    assert client.calls[-1][1]["outcome"]["ended_by"] == "fault"


def test_velocity_kind_safe_action_is_zero(tmp_path):
    src = ENVELOPE.read_text(encoding="utf-8").replace('action_kind = "ee_position"', 'action_kind = "ee_delta"')
    p = tmp_path / "delta.toml"
    p.write_text(src, encoding="utf-8")
    fuse, _ = make_fuse()
    fuse.start(p, envelope_digest=HEX)
    fuse.last_pos = [300.0, 300.0]
    assert fuse.safe_action() == [0.0, 0.0]


def test_begin_uses_harness_compat_when_importable(monkeypatch):
    fake = types.ModuleType("harness.compat")
    fake.host_binding = lambda: {"os": "fake-os", "torch": "9.9"}
    fake.env_binding = lambda env: {"env_id": "fake/Env-v0", "gymnasium": "1.3.0"}
    fake.policy_binding = lambda policy: {"repo_id": "fake/policy", "weights_sha256": HEX}
    fake.inputs_manifest = lambda: {"harness/compat.py": HEX}
    monkeypatch.setitem(sys.modules, "harness.compat", fake)
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    fuse.begin(1, 1)
    kw = holder["client"].calls[-1][1]
    assert kw["binding"]["host"]["os"] == "fake-os" and kw["binding"]["env"]["env_id"] == "fake/Env-v0"
    assert kw["binding"]["policy"]["repo_id"] == "fake/policy"
    assert kw["inputs"]["harness/compat.py"] == HEX
    assert kw["run"]["init_state_digest"] == ll.ZERO_DIGEST and kw["binding"]["env"]["init_state_digest_source"] == "undeclared"


def test_calibration_json_feeds_hello(tmp_path):
    cal = tmp_path / "calibration.a05.json"
    cal.write_text(json.dumps({"schema": "lictor-calibration/v1", "alpha_num": 5, "alpha_den": 100, "digest": HEX}), encoding="utf-8")
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, calibration=cal, envelope_digest=HEX)
    kw = holder["client"].calls[0][1]
    assert kw["calibration_digest"] == HEX and kw["expect_tier1"] is True and kw["expect_alpha"] == (5, 100)
    assert holder["client"].kwargs["calibration"] == cal


# ----------------------------------------------------------------------------------------------------------------
# LictorStep / LictorObserveStep
# ----------------------------------------------------------------------------------------------------------------


class FakeQueue(list):
    pass


class FakePolicy:
    """The two private things the default chunk discovery reads: `_queues[ACTION]` and `config.n_action_steps`."""

    def __init__(self, rows, n_action_steps=8):
        self.config = types.SimpleNamespace(n_action_steps=n_action_steps, horizon=16, n_obs_steps=2, device="cpu", dtype="float32")
        self.name = "fake/policy"
        self._queues = {"action": FakeQueue()}
        self._rows = rows

    def select_action(self):
        if not self._queues["action"]:
            self._queues["action"].extend(self._rows[: self.config.n_action_steps])
        return [self._queues["action"].pop(0)]  # (1, d) like a batch of one


def test_lictor_step_replaces_the_action_with_verdict_action():
    np = pytest.importorskip("numpy")
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX, mode="enforce")
    client = holder["client"]
    client.verdict_fn = lambda t, idx, row: [min(row[0], 300.0), row[1]]  # a fake box clamp on x
    fuse.begin(0, 0)
    rows = chunk_rows(base=(295.0, 100.0))
    policy = FakePolicy(rows, n_action_steps=15)
    step = ll.LictorStep(fuse, policy=policy)
    watch = ll.LictorObserveStep(fuse)
    seen = []
    for t in range(15):
        watch({"observation": {"observation.state": np.asarray([[295.0 + t, 100.0]], dtype=np.float32)}, "action": None})
        action = np.asarray(policy.select_action(), dtype=np.float32)  # (1, 2)
        out = step({"observation": None, "action": action, "reward": 0.0, "done": False, "truncated": False, "info": {}, "complementary_data": {}})
        seen.append(out["action"])
        assert out["action"].dtype == np.float32 and out["action"].shape == (1, 2)
        assert out["action"][0, 0] <= 300.0
        assert step.transition is not None
    assert step.n_calls == 15 and step.n_substituted == sum(1 for r in rows if r[0] > 300.0) == 13
    assert step.last_verdict["chain"] == HEX
    ticks = [c for c in client.calls if c[0] == "tick"]
    assert len(ticks) == 15 and ticks[0][1]["chunk"]["h"] == 15 and all(c[1]["chunk"] is None for c in ticks[1:])
    assert [c[1]["pos"] for c in ticks][:3] == [[295.0, 100.0], [296.0, 100.0], [297.0, 100.0]]
    assert fuse.pending_state is None  # consumed by the tick
    assert step.feature_contract()["action"] == {"type": "ACTION", "shape": [2], "kind": "ee_position"}
    assert step.state_dict() == {} and step.load_state_dict({}) is None
    assert step.get_config()["status"] == "UNVERIFIED"
    feats = {"ACTION": {"action": types.SimpleNamespace(shape=(2,))}, "OBSERVATION": {}}
    assert step.transform_features(feats) is feats
    with pytest.raises(ll.LictorAdapterError):
        step.transform_features({"ACTION": {"action": types.SimpleNamespace(shape=(7,))}})
    step.reset()  # ends the open episode as abort
    assert client.calls[-1][1]["outcome"]["ended_by"] == "abort" and step.episodes == 1 and not fuse.open


def test_lictor_step_without_state_raises_and_without_action_raises():
    fuse, _ = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    fuse.begin(0, 0)
    step = ll.LictorStep(fuse, chunk_fn=lambda a: chunk_rows())
    with pytest.raises(ValueError):
        step({"action": None})
    with pytest.raises(ll.LictorAdapterError, match="no state"):
        step({"action": [[100.0, 200.0]]})


def test_lictor_step_state_fn_chunk_fn_and_list_actions():
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    rows = chunk_rows()
    calls = {"n": 0}

    def chunk_fn(action):
        calls["n"] += 1
        return rows if calls["n"] == 1 else None

    step = ll.LictorStep(fuse, chunk_fn=chunk_fn, state_fn=lambda: {"pos": [100.0, 200.0]}, ext_fn=lambda: [0.25], auto_episodes=True)
    out = step({"action": rows[0]})
    assert out["action"] == rows[0] and fuse.open and step.episodes == 0
    out = step({"action": [rows[1]]})
    assert out["action"] == [rows[1]]
    assert holder["client"].calls[-1][1]["ext"] == [0.25]
    assert holder["client"].calls[1][1]["run"]["seed"] == 0


def test_lictor_step_holds_after_a_fault_and_the_step_keeps_working():
    fuse, holder = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    holder["client"].fail_at = 1
    fuse.begin(0, 0)
    rows = chunk_rows(base=(-100.0, 50.0))
    step = ll.LictorStep(fuse, chunk_fn=lambda a: rows if fuse.cur_chunk is None else None, state_fn=lambda: {"pos": [-100.0, 50.0]})
    step({"action": rows[0]})
    out = step({"action": rows[1]})
    assert out["action"] == [17.0, 50.0]
    out = step({"action": rows[2]})
    assert out["action"] == [17.0, 50.0] and step.last_verdict["synthetic"]


@pytest.mark.skipif(
    os.environ.get("LICTOR_TEST_LEROBOT") != "1" or importlib.util.find_spec("lerobot") is None,
    reason="opt-in: LICTOR_TEST_LEROBOT=1 with lerobot installed (imports torch; ~15 s on this machine)",
)
def test_bind_lerobot_makes_a_real_processor_step():
    proc = importlib.import_module("lerobot.processor")
    bound = ll.LictorStep.bind_lerobot()
    assert issubclass(bound, proc.ProcessorStep)
    fuse, _ = make_fuse()
    fuse.start(ENVELOPE, envelope_digest=HEX)
    fuse.begin(0, 0)
    rows = chunk_rows()
    step = bound(fuse, chunk_fn=lambda a: rows if fuse.cur_chunk is None else None, state_fn=lambda: {"pos": [100.0, 200.0]})
    pipeline = proc.DataProcessorPipeline(steps=[step], to_transition=lambda x: x, to_output=lambda x: x)
    torch = importlib.import_module("torch")
    out = pipeline({proc.TransitionKey.ACTION: torch.tensor([rows[0]], dtype=torch.float32)})
    assert torch.equal(out[proc.TransitionKey.ACTION], torch.tensor([rows[0]], dtype=torch.float32))
