# adapters/

**STATUS of `lerobot/` and `openpi/`: UNVERIFIED ROADMAP ADAPTER. Imports and unit tests pass without lerobot/openpi installed; end-to-end behaviour has NOT been validated against a live policy server. See docs/ARCHITECTURE.md sec 12.**

Everything a host needs to talk to the fuse, and everything a third party needs to check what it said.

| File | Status | What backs the status |
|---|---|---|
| `lictor_client.py` | verified | the frozen Python client of `lictor-wire/v1` (docs/ARCHITECTURE.md sec 9); `tests/test_client.py` drives the real binary through a synthetic episode |
| `verify_receipt.py` | verified | stdlib-only receipt verifier; `tests/test_verify.py` plus the Rust parity test recompute the same canonical bytes and signature |
| `lerobot/lictor_lerobot.py` | UNVERIFIED | importable without lerobot; `tests/test_lerobot_stub.py` runs it over a mocked client; never run against a LeRobot policy end to end |
| `openpi/lictor_proxy.py` | UNVERIFIED | importable without websockets/msgpack; `tests/test_openpi_stub.py` covers the codec, the gate and the fault path; never run against an openpi server |

"Verified" means the tests in this repository exercise the real thing. "UNVERIFIED" means exactly what the status
line says: the code is a faithful reading of the upstream extension points at one point in time, it imports and its
unit tests pass, and nobody has watched it fuse a live policy. Both stubs are shipped so that the shape of the
adoption story can be read and criticised before milestone 2 measures it.

## The wire client -- `lictor_client.py`

`LictorClient` spawns one `lictor serve` child per host worker and speaks NDJSON on its stdin/stdout: `hello` once,
`episode_begin` / `tick` x N / `episode_end` per episode, `bye`. Every request passes through `finite_or_none`
(non-finite reals become JSON `null`, which the fuse treats as NaN and trips `nonfinite`) and
`json.dumps(..., allow_nan=False)`. The host contract is fail-closed: on a read timeout, a fatal error with a dead
child, a child exit or an `id` mismatch the client KILLS the child and raises `LictorFault` carrying
`last_safe_action` (`clamp_box(current measured position)` for position kinds, the zero vector for velocity kinds);
the harness never executes a stale setpoint or the raw policy action on that path.

```python
from adapters.lictor_client import LictorClient, chunk_msg
c = LictorClient(binary, envelope=Path("envelopes/pusht.toml"), mode="enforce", out_dir=Path("results/run/arm"))
c.hello(embodiment_id="gym_pusht/PushT-v0", action_dim=2, pos_dim=2, horizon=15, exec_steps=8,
        envelope_digest=digest, calibration_digest=None, expect_tier1=False, expect_alpha=None)
c.episode_begin(run, budget, binding, fault_injection=None, inputs=manifest)
v = c.tick(t, idx, pos=agent_pos, vel=vel, aux=aux, chunk=chunk_msg(seq, t_emit, a15, 8) if new_chunk else None)
env.step(v["action"])            # verdict.action and nothing else
c.episode_end(outcome); c.close()
```

## The receipt verifier -- `verify_receipt.py`

```
python3 adapters/verify_receipt.py RECEIPT.json --pubkey HEX [--ticks TICKS.jsonl] [--ledger LEDGER.jsonl] [--parity]
```

Prints exactly `OK` or `FAIL: <reason>` (exit 0 / 1). No third-party packages: `json`, `hashlib`, `base64`, `re`
and a pure-Python Ed25519 verify. It recomputes the canonical bytes with `json.dumps(obj, sort_keys=True,
separators=(",", ":"), ensure_ascii=False)` after re-checking the two facts that make that equal to RFC 8785 over
the float-free profile (no real numbers in the body; printable-ASCII keys), so the equality is checked on every
document rather than assumed. `--pubkey` is required: a receipt is evidence only against a pinned key. See
docs/receipt-schema.md.

## LeRobot v0.6 -- `lerobot/lictor_lerobot.py` (UNVERIFIED)

**Extension point (lerobot 0.6.1, read from the installed package).** `lerobot.envs.factory.make_env_pre_post_processors(env_cfg, policy_cfg)`
returns `(env_preprocessor, env_postprocessor)`, delegating to `env_cfg.get_env_processors()`, which is the identity
`PolicyProcessorPipeline(steps=[])` for every env except LIBERO's preprocessor. `lerobot/scripts/lerobot_eval.py`
applies the postprocessor as `action_transition = env_postprocessor({ACTION: action})` immediately before
`env.step(action.numpy())` -- the slot the upstream processors guide illustrates with a `torch.clamp` "safety
limits" example. No `lerobot-eval` flag injects a custom step, so the host is a short custom eval script that reuses
`make_env`, `make_policy`, `make_pre_post_processors` and `make_env_pre_post_processors`. `DataProcessorPipeline`
does not `isinstance`-check its steps (it calls them and duck-types `reset`), so the step works unregistered;
`LictorStep.bind_lerobot()` returns a registered subclass of the real `ProcessorStep` ABC for hosts that save
pipelines.

**The ten-line story** (the envelope for LIBERO does not exist yet; this is the shape, not a measurement):

```python
from adapters.lerobot.lictor_lerobot import Fuse, LictorObserveStep, LictorStep
fuse = Fuse()                                                    # finds the `lictor` binary
fuse.start("envelopes/libero.toml", calibration=None, out="results/libero", mode="enforce")
env_pre, env_post = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)
env_pre = DataProcessorPipeline(steps=[*env_pre.steps, LictorObserveStep(fuse)])           # records the state
env_post = DataProcessorPipeline(steps=[*env_post.steps, LictorStep(fuse, policy=policy)])  # fuses the action
fuse.begin(episode=0, seed=0, policy=policy, env=env)           # once per episode, after env.reset(seed)
...                                                              # the stock eval loop, unchanged
fuse.end(success=bool(info["is_success"]), steps=t, progress=float(reward))   # -> signed receipt + ledger line
fuse.close()
```

`Fuse` is `start(envelope, calibration, out, mode)` / `begin(episode, seed, policy, env)` / `tick(state, action,
chunk=None, ext=None)` / `end(success, steps, progress)` over the frozen client: `start` reads the embodiment from
the envelope TOML and the digest from `lictor envelope digest`; `begin` computes the `binding` maps through
`harness.compat` when it is importable (`host_binding`, `env_binding`, `policy_binding`, `inputs_manifest`) and
from `policy.config` / `env.spec` / explicit kwargs otherwise; `tick` labels chunks the sync way (`t_emit = t`,
`idx = 0` on delivery) and refuses, before anything reaches the wire, a trimmed chunk, an executor that is about to
run a row the fuse does not index, or a run past the horizon.

Three simplifications, stated rather than hidden:

1. **Chunk discovery reads `policy._queues[ACTION]`**: a new chunk is assumed on the step whose popped action leaves
   `n_action_steps - 1` rows queued (the stock `select_action` cadence). Pass `chunk_fn` (e.g. around
   `predict_action_chunk`) and nothing private is touched; an envelope with `horizon = 1, exec_steps = 1` runs the
   step per action with Tier 0 and the brake only.
2. **The transition carries only the action**, so the measured state reaches the fuse through `LictorObserveStep`
   in the `env_preprocessor` (records `observation.state`) or `state_fn`. A step without a state raises; it never
   ticks with a guessed position.
3. **One fuse, one env**: `n_envs > 1` needs one `Fuse` per env slot. docs/VERIFIED_FACTS.md measures the B=32
   batched forward; multiplexing episodes over one child is a wire change milestone 1 does not make.

On a `LictorFault` (dead child) the step writes `clamp_box(current measured position)` into the transition, keeps
doing so for the rest of the episode and `fuse.end` runs `lictor crash-receipt` so the ledger records the failure;
on a fatal wire error with a live child it holds the same way and `end` sends `ended_by: "fault"`.

Try it (no lerobot needed):

```bash
/mnt/d/lictor/venv/bin/python -m pytest adapters/tests/test_lerobot_stub.py -q
python3 -c "import adapters.lerobot.lictor_lerobot"                                 # system python, no torch
LICTOR_TEST_LEROBOT=1 /mnt/d/lictor/venv/bin/python -m pytest adapters/tests/test_lerobot_stub.py -q -k bind   # opt-in: imports lerobot + torch
```

## openpi -- `openpi/lictor_proxy.py` (UNVERIFIED)

**Wire format** (read from openpi `main`, 2026-09-02: `packages/openpi-client/src/openpi_client/msgpack_numpy.py`,
`websocket_client_policy.py`, `src/openpi/serving/websocket_policy_server.py`): the server sends one msgpack
frame of metadata after the handshake; per request the client sends one msgpack frame with the observation dict
and receives one with `{"actions": (H, D), "server_timing": {...}}`; numpy arrays are msgpack maps with BYTES keys
(`b"__ndarray__"`, `b"data"`, `b"dtype"`, `b"shape"`); a TEXT frame from the server is an error and makes
`WebsocketClientPolicy.infer` raise; `compression=None, max_size=None` on both ends.

**Port point.** `scripts/serve_policy.py` constructs `WebsocketPolicyServer(policy, host="0.0.0.0", port=8000,
metadata=...)`; every openpi example client constructs `WebsocketClientPolicy(host, port)`. Point the client at the
proxy's `--listen` and the proxy's `--upstream` at the real server. Nothing in openpi changes.

```bash
pip install websockets msgpack          # live mode only; --dry-run and the tests need neither
python adapters/openpi/lictor_proxy.py --listen 0.0.0.0:8001 --upstream ws://gpu-host:8000 \
    --envelope envelopes/<embodiment>.toml --mode enforce --out results/openpi --exec-steps 8 --episode-boundary reset-key
python adapters/openpi/lictor_proxy.py --dry-run --envelope envelopes/pusht.base.toml --json   # codec + gate, no network
```

Per client connection the proxy opens one upstream connection and one `lictor serve` child (via `Fuse`), forwards
the metadata frame with a `lictor` entry added, forwards every observation frame upstream VERBATIM, lowers the
returned `(H, D)` chunk to the chunk IR (`horizon = H`, `exec_steps` from `--exec-steps`), drives one wire `tick`
per row the client will execute (rows `0 .. exec_steps-1`, the delivery tick carrying the full chunk), substitutes
those rows with `verdict.action`, fills rows `exec_steps .. H-1` with the last fused action (a hold: the client
never executes rows the fuse has not ticked) and returns the same response dict with `actions` replaced and a
`lictor` summary. Episodes are delimited by a truthy `--reset-key` entry in the observation (`reset` by default;
`--episode-boundary connection` uses the connection instead); one receipt per episode.

**Never drops the connection on a fault.** Upstream down, an upstream error frame, a malformed frame, the fuse
child dead or a fatal wire error all produce a response with a hold chunk (`clamp_box(current position)` for
position kinds, zeros for velocity kinds) and `lictor.error`; the proxy keeps serving. The "error frame" travels
inside the response dict on purpose: a separate text frame would be read by the stock client as a server traceback
and make it raise.

**The executor gate is the simplification everything rests on.** The proxy sees the client only at request time, so
it assumes the client executes exactly `exec_steps` rows between requests (`open_loop_horizon` / `replan_steps` in
the openpi examples) and holds the delivered observation for those rows. Tier 0, the brake rollout and the
chunk-boundary features are evaluated at delivery as on PushT; the per-tick trail features see a stationary
position between requests, so a Tier-1 calibration is exchangeable only when collected THROUGH the proxy. None
exists: run without `--calibration` (Tier 1 disarmed) until one does.

When `msgpack` is absent a byte-compatible mini encoder/decoder stands in (`Codec.backend == "mini"`; the parity
test asserts identical bytes when `msgpack` is installed), and without `numpy` arrays decode to `ArrayLike`
(shape, dtype, bytes) -- enough for `--dry-run` and the tests to exercise the real frame layout on a bare
interpreter.

## Security note

LeRobot's async inference path (`policy_server` / `robot_client` over gRPC) deserialises pickled payloads;
CVE-2026-25874 is the unauthenticated deserialisation issue in that channel (upstream issue #3047 was open when this
was written; check its status before relying on the framing). lictor does not touch that path in milestone 1: the
LeRobot step runs in-process and talks to the fuse over the frozen NDJSON pipe, and the openpi proxy carries
msgpack, not pickle. The proxy does terminate a network connection: run it on localhost or inside the trust boundary
of the policy server (the upstream `Api-Key` header is not forwarded). A Rust `lictor proxy --upstream ...` that
replaces pickle with JSON/safetensors and adds signed action receipts is roadmap item 4 of docs/ARCHITECTURE.md
sec 12 -- roadmap, not shipped.

## Acceptance (WP-11)

```bash
/mnt/d/lictor/venv/bin/python -m pytest adapters/tests/test_lerobot_stub.py adapters/tests/test_openpi_stub.py -q
python3 -c "import adapters.lerobot.lictor_lerobot, adapters.openpi.lictor_proxy"     # system python, no lerobot/openpi
grep -l "UNVERIFIED" adapters/lerobot/lictor_lerobot.py adapters/openpi/lictor_proxy.py adapters/README.md | wc -l   # 3
```
