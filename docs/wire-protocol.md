# lictor wire protocol (`lictor-wire/v1`)

Owner: WP-6. Section 1 is the normative WIRE PROTOCOL section of `docs/ARCHITECTURE.md` sec 9 (copied verbatim, as
required by the plan; if the two ever differ, ARCHITECTURE wins). Sections 2-5 are the implementation notes of
`crates/lictor-runtime` (`wire.rs`, `codec.rs`, `session.rs`), `lictor serve`, `lictor crash-receipt` and
`adapters/lictor_client.py`: what the code does, with the tests that prove each claim.

Terms borrowed for readability; no conformance to any standard is claimed or tested.

## 1. The protocol (normative, verbatim)

## WIRE PROTOCOL -- `lictor-wire/v1` (normative)

Transport: one `lictor serve` child process per harness worker, spawned by Python; requests on the child's stdin, responses on its stdout, diagnostics on stderr. Rust writes NOTHING to stdout except response lines.

Framing: UTF-8, exactly one JSON object per line, `\n`-terminated, no embedded newlines, no `\r`, max line 1 MiB. Requests carry `id: u64`, strictly increasing from 1; every response echoes `id`. Strictly alternating request -> response.

Numbers: plain JSON numbers, IEEE-754 f64. Both serde_json (ryu) and CPython `repr` emit shortest-round-trip decimals and both parsers are correctly rounded, so f64 survives the boundary EXACTLY. Python MUST send `np.asarray(x, dtype=np.float64).tolist()` (never a float32 repr). A non-finite value MUST be sent as JSON `null`; the fuse treats `null` in any numeric slot as NaN -> `nonfinite` -> Fault. Python converts non-finite values to `None` explicitly (`np.isfinite`) and serialises with `json.dumps(..., allow_nan=False)`, so a NaN can never leak as the non-JSON literal `NaN` (serde_json rejects it -- still fail-closed, but with a misleading schema message). In RESPONSES exactly two real fields may be non-finite BY DESIGN and are emitted as `null`: `scores.s` (`-inf` while no gate term is fully valid, i.e. every tick before the first chunk with `L >= 2`, and always while Tier 1 is disarmed) and `tau` (`+inf` when disarmed or `k > n`); `hello_ok.calibration.tau` likewise. The client maps `null -> -math.inf` / `+math.inf` for those keys ONLY; every other response number is finite by construction (`session_episode.rs` asserts the first tick's `s` round-trips). (Design decision: the wire is never hashed, so readability wins; the receipt/tick/calibration files are hashed, so THEY are float-free.)

Error discipline (fail-closed): any malformed line, unknown field at ANY nesting level (every payload struct is `deny_unknown_fields`), dimension mismatch, non-monotonic `id`, `tick` before `episode_begin`, `idx >= horizon`, an `idx`/`t_emit` inconsistency (below), a time discontinuity (`t != prev_t + 1 + missed_ticks`; the first tick of an episode has `t == 0`) or a chunk-sequence discontinuity (`chunk.seq != next_chunk_seq`; the first chunk has `seq == 0`) -> an `error` response, `TripMask::SCHEMA`, and the fuse latches `FuseState::Fault` for the rest of the episode; every subsequent `tick` returns `status:"fault"`, `action_src:"hold"` and the hold action. It never returns the raw policy action after a fault. The two continuity checks are integer compares inside `decide()` GUARD (against `FuseRt.last_t` / `FuseRt.next_chunk_seq`), so a host that lies about time cannot silently corrupt `v_hat` or the trail. The host treats `fatal:true` as episode-abort: it stops stepping the env and MUST STILL send `episode_end` with `ended_by:"fault"`; the Session ACCEPTS `episode_end` while faulted and writes a receipt with `terminal_state:"fault"`, `fuse_ok:false`, so the episode enters the ledger as a failure instead of vanishing from `lictor curve`.

Host contract (fail-closed at the host boundary): the Python side (a) sends exactly one `tick` per `env.step`, (b) executes `verdict.action` and nothing else, (c) treats ANY of {read timeout, `error.fatal`, child exit, `id` mismatch} as a brake: for `ee_position`/`joint_position` kinds the fallback is ALWAYS `clamp_box(current measured position)` (never a stale `brake`/`hold` setpoint latched ticks ago -- on a real arm that is a lunge), for velocity kinds it is the zero vector; the episode is marked `ended_by:"fault"`. "Stale last policy action" and "raw policy action" are never the timeout path. (d) On a `LictorFault` raised for {timeout, child exit, `id` mismatch} the client KILLS the child and marks itself dead; the harness respawns a fresh `lictor serve` and re-`hello`s before the next episode (a late verdict for id N must never be read as the reply to N+1). If the child is dead the harness runs `lictor crash-receipt` (CLI SURFACE) so the episode is recorded as `ended_by:"fuse_crash"`, `success:false`; retry cap: 1 respawn per episode index, then the arm aborts with a message. (e) After `error.fatal` with a LIVE child the host sends `episode_end` (`ended_by:"fault"`) as above. `--on-fault abort` (exit 3 on the first fatal error) is a debugging aid only and is never used in an experiment arm.

**Trust model (what the fuse verifies vs what the host declares).** The fuse verifies: `mode`, dims, `horizon`/`exec_steps`, the envelope digest, the calibration digest and its `embodiment_digest`, its own binary hash, and every `tick` (schema, finiteness, continuity). EVERYTHING else is the host's declaration, signed by proxy: `binding` (package versions, policy revision, weights hash, `n_action_steps`), `run` (seed, pool, `init_state_digest`), `budget.delay_steps`/`exec_mode`/`stitch`, `fault_injection`, `inputs`, and `outcome.success`. A receipt therefore defends against post-hoc edits by anyone WITHOUT the signing key and against accidental corruption; it does not defend against the experimenter, who holds the key. External anchoring (roadmap 10) is what would change that. Every sentence in these documents of the form "the receipt binds X" means "the receipt binds the host's declaration of X" unless X is in the verified list above.

### hello (once per process)

```json
{"id":1,"kind":"hello","proto":"lictor-wire/v1","client":"lictor_client/0.1.0",
 "mode":"observe",
 "embodiment_id":"gym_pusht/PushT-v0","action_dim":2,"pos_dim":2,"horizon":15,"exec_steps":8,
 "envelope_digest":"7c4a...64hex","calibration_digest":null}
```
`mode` in {"observe","enforce"} must equal the server's `--mode`; `embodiment_id`/dims/horizon/exec_steps must equal the loaded envelope; `envelope_digest` and `calibration_digest` (or null) must equal the loaded artefacts -- any mismatch is a fatal `error` with `code:"envelope"`.

```json
{"id":1,"kind":"hello_ok","proto":"lictor-wire/v1","lictor":"0.1.0","git":"a3f1c9e","lictor_sha256":"...64hex",
 "envelope_digest":"7c4a...","embodiment_digest":"91d0...","calibration_digest":null,"pubkey":"64e8...64hex","ephemeral_key":false,
 "mode":"observe","tier0_armed":["workspace","speed","accel","jerk","reach","brake"],"tier1_armed":false,
 "calibration":null,
 "latency_label":"measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment",
 "features":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak","ext0","ext1","ext2","ext3"],
 "trip_names":["workspace","speed","accel","jerk","reach","contact","brake","nonfinite","schema","watchdog","tier1_cp","clamp_budget","handoff_timeout","operator_abort","brake_timeout","rearm_budget"]}
```
When a calibration is loaded: `"calibration":{"method":"binned","alpha_num":5,"alpha_den":100,"n_calib":137,"tau":3.7142857142857144,"kn":[3,5],"gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],"digest":"e1b8..."}` (`n_calib` = the number of scores `tau` was taken over; `tau` is `null` on the wire when `+inf`). The client asserts `hello_ok.tier1_armed == arm.tier1` and `(alpha_num, alpha_den) == arm.alpha` and raises otherwise, so a `t01-*` arm launched without `--calibration` cannot silently run as a `t0` arm.

### episode_begin (once per episode)

```json
{"id":2,"kind":"episode_begin",
 "run":{"run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","episode_index":7,"seed":7,"seed_pool":"eval",
        "init_state_digest":"3f1c...64hex"},
 "budget":{"delay_steps":0,"tick_ms":100,"exec_mode":"sync","stitch":"drop","on_escalate":"terminate_fail"},
 "binding":{"env":{"env_id":"gym_pusht/PushT-v0","gym_pusht":"<importlib.metadata.version>","gymnasium":"<importlib.metadata.version>",
                   "pymunk":"<importlib.metadata.version>","numpy":"<importlib.metadata.version>",
                   "obs_type":"pixels_agent_pos","control_hz":"10","max_episode_steps":"300","vel_source":"info.vel_agent","coverage_t0":"env.unwrapped._get_coverage()"},
            "policy":{"repo_id":"lerobot/diffusion_pusht","revision":"<HF commit sha>","weights_sha256":"<sha256 of the safetensors actually loaded>","horizon":"16",
                      "n_action_steps":"15","n_obs_steps":"2","num_inference_steps":"100","device":"cuda","dtype":"float32",
                      "normalization_migrated":"true","migration_script":"lerobot/processor/migrate_policy_normalization.py"},
            "host":{"os":"<platform.platform()>","cpu":"<platform.processor()>","gpu":"<torch.cuda.get_device_name(0) or cpu>","torch":"<torch.__version__>","lerobot":"<importlib.metadata.version>",
                    "OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONHASHSEED":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}},
 "fault_injection":null,
 "inputs":{"envelopes/pusht.toml":"<sha256>","harness/pusht_rollout.py":"<sha256>","harness/executor.py":"<sha256>","harness/compat.py":"<sha256>"}}
```
Every `binding` value is read at run time (`importlib.metadata.version`, `torch.__version__`, `platform`); the placeholders above are NOT to be pasted (on the dev machine today: gym-pusht 0.1.6, gymnasium 1.3.0, pymunk 6.11.1, numpy 2.2.6, torch 2.7.1+cu126, torchvision 0.22.1+cu126, lerobot 0.6.1 -- newer than the pins the 65.4 % number was published with, which is why the reproduction gate is hard). `fault_injection` is either `null` or `{"kind":"action_spike","params":{"p":"0.02","mag":"180"},"stream_seed":1234567}` -- the receipt binds the host's DECLARATION of injection (a host that injects and sends `null` hides it; see the trust model). `inputs` is the host's content-addressed manifest of the harness files it ran (repo-relative path -> sha256); the Session adds `lictor:bin`, `lictor:envelope` and (when loaded) `lictor:calibration`, computed by the fuse itself. `init_state_digest` = sha256 of the float64 bytes of the env's post-reset observation state (agent_pos, block_pos, block_angle) -- the paired-seed evidence (arms whose digests disagree for an episode index are refused at aggregation).

```json
{"id":2,"kind":"episode_ok","state":"armed","seq":0}
```

### tick (once per env step -- the ONLY hot message)

```json
{"id":31,"kind":"tick","t":24,"idx":0,"missed_ticks":0,
 "obs":{"pos":[213.5,301.0],"vel":[41.2,-12.7],"aux":[256.0,256.0,0.7853981633974483,0.61],"ext":[]},
 "chunk":{"seq":3,"t_emit":24,"h":15,"d":2,"exec":8,
          "a":[[214.0,300.2],[216.1,299.0],[218.0,297.7],[219.6,296.3],[221.0,294.9],[222.2,293.6],[223.3,292.4],[224.3,291.3],
               [225.2,290.3],[226.0,289.4],[226.7,288.6],[227.3,287.9],[227.8,287.3],[228.2,286.8],[228.5,286.4]]},
 "ack":null}
```
`chunk` is non-null ONLY on the tick a new chunk is delivered. On that tick: `t_emit <= t`, `idx == t - t_emit`, `idx < h`, `h == envelope.embodiment.horizon` (the fuse ALWAYS receives the FULL 15-row chunk; the harness never trims rows), `d == action_dim`, `exec == exec_steps`, `seq == next_chunk_seq`. Delivery labelling per exec mode: sync (hold-last during the delay, then execute from row 0) -> `t_emit = t`, `idx = 0`; async `stitch=freeze` (RTC freeze: the d already-planned rows are executed anyway) -> `t_emit = t`, `idx = 0`; async `stitch=drop` (time-aligned: row `d` applies now) -> `t_emit = t - d` (the generation step), `idx = d`. `brake_feasible` is evaluated from row `idx` at delivery. On non-delivery ticks `chunk` is `null` and `idx` indexes the chunk currently executing. Consequence that is DISCLOSED, not hidden: Tier-1 overlap is computed from `t_emit` differences (`s = new.t_emit - prev.t_emit`, `L = min(prev.h - s, new.h)`), so in sync mode `L = 7 - d`: `tce`/`acc` are structurally INVALID for `d >= 7` and degraded for `d in 3..6`, while async arms keep `L = 7`. `lictor curve` binds `tce_valid_frac` per arm and F1 plots it. `obs.vel` is the simulator velocity (`info["vel_agent"]`; the manifest says `provides_vel = true`); at `t = 0` the harness sends `aux[3] = env.unwrapped._get_coverage()` because `reset()` info carries no `coverage`. `ack` is `null` or an `AckToken` object (see FILE FORMATS); it is verified by the runtime before `decide()` runs.

```json
{"id":31,"kind":"verdict","t":24,"seq":24,
 "status":"nominal","state":"armed","prev_state":"armed",
 "trips":[],"trip_mask":0,
 "action":[214.0,300.2],"action_src":"policy","substituted":false,"clamped_dims":0,
 "scores":{"f":[0.012,0.31,-0.021,1.9,0.11,0.08,0.0,0.34,0.0,0.0,0.0,0.0],
           "z":[0.4,0.9,-0.2,0.1,0.7,0.3,0.0,1.1,0.0,0.0,0.0,0.0],
           "s":1.1,"valid":255,"fired":0},
 "tau":3.7142857142857144,"window_hits":0,"brake_margin":41.2,
 "reason":"ok","reason_text":"All checks passed; the policy action is applied unchanged.",
 "handoff":null,"ack_result":null,"violation_reached_env":false,
 "verdict_ns":1180,"chain":"5b7e...64hex"}
```
`scores.s` and `tau` are `null` when non-finite (see Numbers). `handoff` is non-null exactly on the tick `Escalated` is entered and carries a `HandoffRecord` (FILE FORMATS). `ack_result` is `null`, `"accepted"`, or `"rejected:<unknown_operator|bad_signature|wrong_handoff|nonce_replay|no_pending_handoff>"`. `verdict_ns` is measured around `decide()` with the clock read OUTSIDE the pure function. `chain` is the verdict-chain head after this tick.

### episode_end

```json
{"id":331,"kind":"episode_end","t":300,
 "outcome":{"steps":300,"success":false,"terminated":false,"truncated":true,
            "max_coverage":0.71,"final_coverage":0.68,"reward_sum":178.2,"ended_by":"truncated"}}
```
`ended_by` in {"success","truncated","escalation_terminate","fault","abort","retune"} (`"fuse_crash"` is written only by `lictor crash-receipt`, never sent on the wire). `episode_end` is accepted in EVERY session state after `episode_begin`, including a latched Fault; `max_coverage`/`final_coverage`/`reward_sum` may be `null` (non-finite) and are then recorded as NaN hex.

```json
{"id":331,"kind":"episode_receipt",
 "receipt_path":"results/2026-09-01T09-14Z-pilot/t01-a05-d0/receipts/000007.json",
 "ticks_path":"results/2026-09-01T09-14Z-pilot/t01-a05-d0/ticks/000007.jsonl",
 "body_digest":"be21...","verdict_chain_head":"5b7e...","timing_chain_head":"22a0...",
 "fuse_ok":true,"fuse_notes":[],
 "counts":{"ticks":300,"nominal":281,"watching":12,"clamped":0,"braking":4,"held":3,"escalated":0,"fault":0,"terminated":0,
           "substituted":7,"chunks_seen":38,"chunks_rejected":0,"clamps":0,"holds":1,"rearms":1,"escalations":0,
           "first_trip_tick":181,"first_trip_reason":"brake_tier1_cp","first_stop_tick":181,"handoff_tick":null,
           "violations_reached_env":0,"terminal_state":"armed"},
 "ledger_seq":7,"ledger_head":"aa31..."}
```

### bye / error

```json
{"id":332,"kind":"bye"}
{"id":332,"kind":"bye_ok"}
{"id":31,"kind":"error","code":"schema","message":"tick.chunk.a has 14 rows, expected h=15","fatal":true}
```
`code` in {"schema","envelope","state","protocol","internal"}. `fatal:true` for everything except an `ack` rejection (which is reported inside the verdict, not as an error).

### Tracing and replay

`lictor serve --trace FILE` appends a first line `#meta {"proto":"lictor-wire/v1","envelope_digest":"...","calibration_digest":null,"mode":"observe","lictor":"0.1.0","git":"..."}` and then EVERY request line verbatim (responses are recomputable -- that IS the determinism claim). `lictor replay FILE` re-feeds the request lines through a fresh Session and compares the verdict-chain head.

### Python client (`adapters/lictor_client.py`)

```python
CLIENT_VERSION = "lictor_client/0.1.0"          # sent as hello.client; bound into every receipt as body.client

class LictorFault(Exception):
    """Raised on timeout / fatal error / child exit / id mismatch. The child has been KILLED and the client is dead."""
    last_safe_action: list[float] | None   # clamp_box(current measured position) computed by the caller-supplied `safe_action` callback, or None
    reason: str                            # "timeout" | "fatal" | "exit" | "id_mismatch" | "protocol"

class LictorClient:
    def __init__(self, binary: str | Path, envelope: Path, calibration: Path | None = None, mode: str = "observe",
                 out_dir: Path | None = None, trace: Path | None = None, key: Path | None = None,
                 tier0: str | None = None, no_tier1: bool = False, ticks: str = "tail32", timeout_s: float = 5.0,
                 safe_action=None, latency_label: str | None = None) -> None: ...
    def hello(self, embodiment_id: str, action_dim: int, pos_dim: int, horizon: int, exec_steps: int,
              envelope_digest: str, calibration_digest: str | None,
              expect_tier1: bool, expect_alpha: tuple[int, int] | None) -> dict: ...   # asserts tier1_armed / alpha, raises otherwise
    def episode_begin(self, run: dict, budget: dict, binding: dict, fault_injection: dict | None = None,
                      inputs: dict[str, str] | None = None) -> dict: ...
    def tick(self, t: int, idx: int, pos, aux, chunk: dict | None = None, vel=None, ext=(), missed_ticks: int = 0,
             ack: dict | None = None) -> dict: ...   # maps null -> -inf for scores.s and +inf for tau; nothing else
    def episode_end(self, outcome: dict) -> dict: ...
    def close(self) -> None: ...
    @property
    def alive(self) -> bool: ...

def chunk_msg(seq: int, t_emit: int, a: np.ndarray, exec_steps: int) -> dict: ...   # a: (h, d) float64 -> {"seq","t_emit","h","d","exec","a"} with NaN -> None
def finite_or_none(x) -> list: ...                                                    # np.isfinite mask -> None; used by every numeric field
# every request is serialised with json.dumps(obj, allow_nan=False, separators=(",", ":")) after finite_or_none
```
Includes `win_to_wsl(path)` / `wsl_to_win(path)` bridging (bulla_mcp.py style) and auto-detection of a `lictor.exe` under `/mnt/c/...` when `lictor` is absent from `$PATH` and `$LICTOR_BIN` is unset (insurance only; a Windows-native build compiles because `keys.rs` guards `PermissionsExt` with `#[cfg(unix)]`).

## 2. The fail-closed host contract, as implemented

Two boundaries, two latches. The FUSE side latches `Fault` inside `decide()`; the HOST side (the Python client)
kills the child and falls back to a freshly computed safe action. Neither side ever forwards a raw policy action
once anything has gone wrong.

### 2.1 Fuse side (`Session`, `crates/lictor-runtime/src/session.rs`)

| Event | Reply | Latch | Then |
|---|---|---|---|
| malformed line (bad JSON, unknown key at any depth, `NaN`, `\r`, > 1 MiB, empty) | `error{code:"schema",fatal:true}` (the `id` is recovered from the raw line when possible) | episode faulted | next tick runs `decide()` with `schema_fault=true` -> `TripMask::SCHEMA` -> `Fault` -> hold |
| dims / `h` / `d` / `exec` / `idx >= horizon` / `t_emit > t` / `idx != t - t_emit` / `chunk.seq != next` (`Staging::stage`) | `error{code:"schema",fatal:true}` | episode faulted | the same tick is ALSO fed through `decide()` with `schema_fault=true`, so the verdict chain records the fault at that seq |
| `id` not strictly increasing | `error{code:"protocol",fatal:true}` | episode faulted | as above |
| `proto` is not `lictor-wire/v1` | `error{code:"protocol",fatal:true}` | -- | hello refused |
| hello cross-check mismatch (mode, embodiment_id, dims, horizon, exec_steps, envelope_digest, calibration_digest) | `error{code:"envelope",fatal:true}` | -- | hello refused; a later correct hello is accepted |
| `episode_begin` with `binding.policy.weights_sha256 != calibration.policy_digest` | `error{code:"envelope",fatal:true}` | -- | no episode is opened |
| `tick` before `episode_begin` | `error{code:"state",fatal:true}` on the first tick | session faulted | the fuse is reset once and every further tick without an episode returns a `fault`/`hold` VERDICT (never an action) |
| `episode_begin` while an episode is open, `hello` mid-episode, `episode_end` with no episode | `error{code:"state",fatal:true}` | episode faulted (when one is open) | -- |
| time discontinuity, chunk-seq discontinuity, `null` anywhere numeric, `missed_ticks > watchdog_ticks` | a normal `verdict` with `status:"fault"`, trips `schema` / `nonfinite` / `watchdog` | `Fault` inside `decide()` | every later tick is `fault`/`hold` |
| receipt/ticks/ledger write failure at `episode_end` | `error{code:"internal",fatal:true}` | -- | the episode is closed; nothing was appended to the ledger |

`episode_end` is accepted in every state after `episode_begin`, including a latched Fault: the receipt carries
`counts.terminal_state:"fault"`, `fuse_ok:false`, `ended_by` as sent, and a ledger entry is appended
(`tests/session_fault.rs::episode_end_after_fatal_error_still_writes_a_receipt`).

What the session verifies itself (everything else is the host's declaration, signed by proxy -- trust model in
section 1): `mode`, dims, `horizon`/`exec_steps`, the envelope digest, the calibration digest and its
`embodiment_digest` (refused at startup, exit 2), the calibration's `policy_digest` against
`binding.policy.weights_sha256`, its own binary hash (`lictor:bin`), and every tick's schema, finiteness and
continuity.

### 2.2 Host side (`adapters/lictor_client.py`)

| Event | Client action | Raised |
|---|---|---|
| no response line within `timeout_s` | `terminate()` then `kill()`; `alive = False` | `LictorFault(reason="timeout", last_safe_action=safe_action())` |
| child stdout closed / stdin broken | kill; dead | `LictorFault(reason="exit")` |
| response `id != sent id` | kill; dead | `LictorFault(reason="id_mismatch")` |
| `error{fatal:true}` and the child has EXITED | kill; dead | `LictorFault(reason="fatal")` |
| `error{fatal:true}` and the child is ALIVE | returned as the response dict; `client.last_error` | nothing -- the harness sends `episode_end(ended_by="fault")` |
| `hello_ok.tier1_armed != expect_tier1` or calibration alpha != `expect_alpha` | kill; dead | `LictorFault(reason="protocol")` |
| non-JSON response line | kill; dead | `LictorFault(reason="protocol")` |

`last_safe_action` is computed by the caller-supplied `safe_action()` callback at fault time (for `ee_position`
kinds: `clamp_box(current measured position)`); it is `None` when no callback was given. After a `LictorFault` the
harness respawns a fresh `lictor serve`, re-`hello`s, and runs `lictor crash-receipt` for the episode whose child
died (`ended_by:"fuse_crash"`, `success:false`, `terminal_state:"fault"`, `fuse_ok:false`, signed with the same key,
`fuse_notes` = evaluate_fuse notes + "fuse process died before episode_end; host-written crash receipt" + the
optional `--note`).

Serialisation: every numeric field goes through `finite_or_none` (numpy float64, non-finite -> `None`) and every
request through `json.dumps(obj, allow_nan=False, separators=(",", ":"))`; `adapters/tests/test_client.py` proves a
NaN cannot leak. Response mapping: `scores.s: null -> -inf`, `tau: null -> +inf`, `hello_ok.calibration.tau: null ->
+inf`; nothing else.

## 3. Sequence (one episode, sync exec, no escalation)

```
 harness (Python)                         lictor serve (Rust)
 ---------------                          -------------------
 spawn child, pipes line-buffered  -----> parse envelope, compile, sha256(self), key or ephemeral
 hello{id:1, digests, dims}        -----> cross-check -> hello_ok{id:1, pubkey, tier0_armed, tier1_armed, ...}
                                   <-----
 episode_begin{id:2, run, budget,  -----> policy_digest check; fuse.reset; genesis chains; per-episode paths
               binding, inputs}    <-----  episode_ok{id:2, state:"armed", seq:0}
 for t in 0..300:
   [every 8th step: policy chunk]
   tick{id, t, idx, obs, chunk?}   -----> stage -> Instant::now -> decide() -> elapsed
                                          fold tick_event (verdict chain), record decide_ns
                                   <----- verdict{id, seq, status, action, chain, verdict_ns, ...}
   env.step(verdict.action)               note_io_ns(seq, wall-clock read->flush)  (folded on the NEXT request)
 episode_end{id, t, outcome}       -----> fold pending timing; counts = Tally; evaluate_fuse; sign(body)
                                          write receipt -> ticks -> timing -> ledger (LAST)
                                   <----- episode_receipt{id, receipt_path, body_digest, chain heads, ledger_seq}
 bye{id}                           -----> bye_ok{id}; exit 0
```

Escalation variant: on the tick `Escalated` is entered the verdict carries `handoff` (a `HandoffRecord` whose
`chain_at` is that tick's event hash and whose `digest` is what an operator signs); the next tick may carry
`ack` (an `AckToken`); the runtime verifies it BEFORE `decide()` (`verify_ack`: operator listed in the envelope,
Ed25519 `verify_strict`, `handoff_digest` == the pending record, `nonce > last_nonce[operator]`) and reports
`ack_result: "accepted" | "rejected:<reason>"`; an accepted nonce is persisted to `<out>/.lictor/verifier_nonce.json`
immediately, so a fresh process rejects the same token as `nonce_replay`
(`tests/session_episode.rs::accepted_ack_persists_its_nonce_and_replays_are_rejected`). The record's
`ack`/`resolved_tick`/`outcome` (`resumed`/`aborted`/`retune`/`timeout`/`unacked`) are filled when the fuse
consumes the ack or times the handoff out, and every record is embedded in the receipt's `handoffs`.

## 4. Implementation notes (`crates/lictor-runtime`)

- `wire.rs` is exactly the frozen payload set (A.1); every request payload struct is `deny_unknown_fields`
  (`tests/roundtrip.rs` rejects an unknown key inside `obs`, `chunk`, `run`, `ack`, `budget`, `binding`,
  `fault_injection`, `outcome`, `hello`, `bye`). `Response::Verdict` is boxed (WP-0 amendment; JSON unchanged).
- `codec.rs`: `read_request` reads one line (`take(MAX_LINE + 1)`), drains an over-long line so the next line
  starts clean, rejects `\r`, empty lines, invalid UTF-8 and any serde error with `InvalidData(message)`;
  `parse_request` is the ONE wire parser (the trace reader reuses it); `peek_id` recovers the `id` of a rejected
  line; `write_response` writes compact JSON + `\n` and flushes.
- `Staging` (`session.rs`) is the ONE wire -> `TickInput` conversion: fixed buffers, `null -> NaN`, chunk rows into
  a `ChunkBuf`. `stage` copies the observation first (so a faulted tick still hands `decide()` the freshest
  position for the hold latch), then checks `pos`/`vel` length == `pos_dim`, `aux <= 16`, `ext <= 4`,
  `idx < horizon`, `h`/`d`/`exec` == the envelope, the row count and every row length, `t_emit <= t`,
  `idx == t - t_emit`, and `chunk.seq == next_chunk_seq` (its own counter, reset by a tick with `t == 0`; the same
  compare lives in `decide()` GUARD against `FuseRt.next_chunk_seq`, which is the one that latches Fault).
- Timing: `decide_ns` is measured with `Instant` around `fuse.step` only; `io_ns` arrives from the serve loop via
  `note_io_ns(seq, ns)` after the response is flushed and is folded into that seq's `TimingEvent` on the next
  request (or at `episode_end`). The verdict chain is therefore replayable and the timing chain is not, by design.
- `latency`: `Hist` = `hdrhistogram` 1 ns..10 s, 3 significant figures, reset per episode; `summary(label)` is the
  receipt's `latency` block; `to_csv` is the F5 bucket dump. `default_latency_label()` returns the WSL2 sentence
  when `/proc/version` mentions Microsoft/WSL, else `measured on <uname -sr>, non-RT kernel -- not a real-time
  environment`.
- Artefacts (`episode.rs`): `<out>/<run_id>/<arm_id>/receipts|ticks|timing/<episode:06>.*` and `ledger.jsonl`;
  the receipt is pretty JSON with sorted keys; the ticks and timing files carry the `lictor-ticks/v1` header (the
  timing one adds `"stream":"timing"`, docs/receipt-schema.md sec 2) and one canonical event per line; every file
  is fsynced before the ledger append, which is the LAST write. `ledger_prev` in the body is the hash of the
  previous ledger entry of the arm (or `null` for the first episode). The per-operator nonce table lives at
  `<out>/.lictor/verifier_nonce.json`.
- Budget binding (`budget` in the receipt): `mode` from `--mode`; `tier0_armed` = the armed Tier-0 names
  (`--tier0` overrides the envelope); `tier1_armed` = a calibration is loaded and `--no-tier1` is absent; `gate` =
  the calibration's gate (or the envelope's gate when no calibration is loaded: features are still computed and
  recorded for calibration traces, they just cannot fire); `alpha_num/alpha_den` = the calibration's, or `0/1`
  when Tier 1 is disarmed (the client's `expect_alpha=None` corresponds to `calibration: null` in `hello_ok`);
  `kn = [hysteresis.k, hysteresis.n]`.
- `hello_ok.tier1_armed` is derived from the loaded calibration (not from `FuseConfig.calib.armed()`, which is
  true whenever the envelope gate is non-empty, per WP-1's compile semantics).
- Consequence of that compile rule for the two by-design non-finite response fields: `tau` is `null` (+inf)
  whenever Tier 1 is disarmed, but `scores.s` is `null` (-inf) only while NO gate term is fully valid -- with the
  envelope gate embedded, `s` becomes finite from the first tick on which a boundary feature is valid (measured:
  the first tick of the synthetic episode carries `valid = 156`, `s = 0.0043`). `s` stays `-inf` for the whole
  episode only when the envelope's `gate = []` (`tests/session_episode.rs::empty_gate_first_tick_has_null_s`).
  The section-1 sentence "always while Tier 1 is disarmed" therefore reads "while no gate term is valid".
- `AckToken` (the frozen lictor-receipt struct reused as `tick.ack`) has no `deny_unknown_fields`; the codec
  re-checks the `ack` object of a tick against a strict private mirror so an unknown key inside `ack` is still a
  schema fault (`bad_lines.ndjson`, "unknown key inside ack"). Reported as a freeze gap.
- Response numbers other than `scores.s`/`tau` are finite by construction on the nominal path; on a Fault tick the
  fuse (WP-3) may report `brake_margin` as non-finite, which serde emits as `null` and the client leaves as `None`
  (it is not one of the two mapped keys). Recorded as an open issue for WP-3/WP-13.
- Traces: `lictor serve --trace F` writes the process-level trace (`#meta` + every request line verbatim).
  With `--out`, serve additionally writes one trace per episode at
  `<out>/<run_id>/<arm_id>/traces/<episode:06>.ndjson` (the `#meta` line, the last `hello` line, then the
  episode's request lines), so `lictor replay` and `lictor calibrate`/`load_traces` have a self-contained file per
  episode. `Session` accepts `episode_begin` without a preceding `hello` (hello is a cross-check, not a state
  requirement) so such a trace replays through a fresh session.
- `lictor serve` exit codes: 0 on `bye`/EOF; 2 on a startup refusal (envelope parse, calibration load,
  embodiment-digest mismatch, unknown `--tier0` name, unreadable `--key`); 3 on an I/O error or, under
  `--on-fault abort`, on the first fatal error (debugging aid only).

## 5. Conformance checklist

| Claim | Evidence |
|---|---|
| every golden request parses and round-trips; every golden response is the struct's JSON | `crates/lictor-runtime/tests/roundtrip.rs` + `tests/fixtures/wire/golden_*.ndjson` |
| every bad line is rejected with the expected reason (unknown nested keys, 14-row chunk, `t_emit > t`, `idx != t - t_emit`, a second `seq 0`, `NaN`) | `tests/roundtrip.rs::bad_lines_are_rejected_with_the_expected_reason` + `bad_lines.ndjson` |
| fault latch, `fault`/`hold` after any fatal error, receipt after a fatal error | `tests/session_fault.rs` |
| a 300-tick episode verifies (receipt, ticks chain head, ledger), first tick `s`/`tau` are `null`, two runs share the verdict head and not the timing head, nonce persistence | `tests/session_episode.rs` |
| `lictor serve` answers `hello` and `bye` on stdin/stdout | the WP-6 acceptance command (`printf ... | lictor serve ... | grep -q hello_ok`) |
| the Python client never emits `NaN`, maps `null -> inf` for exactly two keys, kills on timeout/exit/id mismatch, asserts tier1/alpha | `adapters/tests/test_client.py` |
