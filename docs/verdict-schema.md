# Verdict schema (lictor-wire/v1, lictor-ticks/v1)

The `SafetyVerdict` is the contract between the fuse and whoever executes actions (the PushT harness, the
LeRobot `ProcessorStep`, the openpi proxy) and between the fuse and whoever audits a run later (the ticks
file, the receipt, `lictor history`). This document is the sbx `verdict-schema.md` twin: it lists the
severity lattice, the state machine's states, every trip bit, every action source, every reason code with
its human sentence, the verdict JSON on the wire and in the ticks file, and the diff / streak semantics of
`lictor history`.

Normative sources, in order: `docs/ARCHITECTURE.md` sec 4 (the frozen types), sec 5.4 (the decision rule),
sec 6 (the transition table), sec 9 (wire and file formats). The Rust definitions live in
`crates/lictor-core/src/{verdict,state,reason,scores}.rs`; the wire struct is
`crates/lictor-runtime/src/wire.rs` (`VerdictMsg`); the file struct is
`crates/lictor-receipt/src/tick.rs` (`TickEvent`). Field names are frozen for v1; new fields, when they
come, are `#[serde(default)]` so v1 readers keep working.

One verdict is produced per control tick by the pure function `decide()`. It is a `Copy` value of about
600 bytes, allocates nothing, reads no clock, and is a function of `(state, config, calibration, input)`
only -- that is what makes the ticks file replayable byte for byte on the tested target.

## 1. `Status` -- the severity lattice

```rust
pub enum Status { Nominal, Watching, Clamped, Braking, Held, Escalated, Fault, Terminated }
```

`Ord` derives from declaration order, so `Status::Nominal < Status::Watching < ... < Status::Terminated`
and `max()` over a set of verdicts gives the worst thing that happened. On the wire and in files the
values are `snake_case` strings: `nominal`, `watching`, `clamped`, `braking`, `held`, `escalated`,
`fault`, `terminated`.

| status | meaning | action the executor receives |
|---|---|---|
| `nominal` | every check passed | the raw policy action |
| `watching` | a score is inside the warn band (`s > tau - warn_margin`) or the K-of-N window has at least one hit below K; informational, nothing substituted | the raw policy action |
| `clamped` | a soft Tier-0 limit tripped and the direction-preserving projection was applied; motion continues | the projected action |
| `braking` | a controlled stop is in progress (brake infeasible, K-of-N predictive fire, or clamp budget exhausted) | the brake action: setpoint := clamped measured position (position kinds) or ramp-to-zero (velocity kinds) |
| `held` | the agent has stopped; monitored standstill, power retained, resumable | the latched hold action |
| `escalated` | a `HandoffRecord` was emitted; a signed operator decision is awaited | the latched hold action |
| `fault` | the fail-closed latch: non-finite input, schema / continuity violation, watchdog miss, brake timeout | the hold action, for the rest of the episode |
| `terminated` | operator abort / retune or handoff timeout; absorbing | the hold action |

`status` is the severity projection of `state` after the tick. Every `FuseState` except `Idle` maps to the
`Status` of the same name (`Armed` maps to `nominal`); `Idle` never appears in a verdict because no verdict
is produced outside an episode. In Observe mode the `status` still reports what the fuse WOULD have done;
the action is the raw policy row regardless (see `violation_reached_env` below).

## 2. `FuseState` -- the escalation state machine

```rust
pub enum FuseState { Idle, Armed, Watching, Clamped, Braking, Held, Escalated, Fault, Terminated }
impl FuseState {
    pub fn is_stop(self) -> bool     // Braking | Held | Escalated | Fault | Terminated
    pub fn is_terminal(self) -> bool // Fault | Terminated
}
```

`Idle` is the default (`#[default]`) and means "no episode". `episode_begin` moves the fuse to `Armed`.
The 24-row transition table (`fsm::next`, first match wins, top-down) is in `docs/ARCHITECTURE.md` sec 6
and is fixture-tested in `crates/lictor-fuse/tests/fixtures/fsm/transitions.json`. The properties worth
remembering when reading a verdict stream:

- Tier-0 hard limits are never windowed: one violation is one trip.
- `Watching` and `Clamped` clear back to `Armed` after `clear_ticks` clean ticks (clean = no trip, no
  warn, no predictive hit); the window is cleared on that transition.
- `Braking`, `Held` and `Escalated` never clear themselves. Leaving `Held` needs either the automatic
  re-arm budget (`rearm = "auto"`: `rearm_hold` consecutive clean ticks, a fresh chunk that passed the full
  Tier-0 suite including brake feasibility on that tick, and `rearms < max_rearms`) or a signed
  `AckToken`; leaving `Escalated` always needs an ack or the handoff timeout.
- `Fault` is never re-armed within an episode. `Terminated` is absorbing.
- `stopped` (the Braking -> Held condition) is `||v_hat|| <= v_stop_eps` in Enforce mode and `true` in
  Observe mode (the observer assumes the brake would have stopped the agent).

`is_stop()` is what `lictor curve` uses for "the arm stopped this episode" (`first_stop_tick`);
`is_terminal()` is what `counts.terminal_state` reports at episode end.

## 3. `TripMask` -- every bit

`trips: u32` carries the bits raised THIS tick. The wire also carries the same information as a list of
names (`"trips": ["speed", "brake"]`) beside the integer (`"trip_mask": 66`).

| bit | name | value | tier | raised when | clampable |
|---|---|---|---|---|---|
| 0 | `workspace` | 1 | Tier-0 soft | a chunk row leaves the margin-adjusted box | yes: componentwise clamp |
| 1 | `speed` | 2 | Tier-0 soft | `||a_i - q_{i-1}|| / dt > v_max` | yes: sequential leash |
| 2 | `accel` | 4 | Tier-0 soft | second difference over `dt^2` exceeds `a_max` | yes (via the leash) |
| 3 | `jerk` | 8 | Tier-0 soft | third difference over `dt^3` exceeds `j_max` | yes (via the leash) |
| 4 | `reach` | 16 | Tier-0 soft | `||a_0 - p_t|| > reach_max` (teleport guard) | yes: pull `a_0` toward `p_t` |
| 5 | `contact` | 32 | Tier-0 soft, off by default | commanded speed near the object exceeds `contact.v_max` (perception-assisted) | yes: leash |
| 6 | `brake` | 64 | Tier-0 hard | the committed prefix plus a braking manoeuvre cannot stay inside the box (sec 5.2) | no -> `Braking` |
| 7 | `nonfinite` | 128 | Tier-0 hard, always armed | any non-finite value in `obs` or the chunk (`null` on the wire) | no -> `Fault` |
| 8 | `schema` | 256 | Tier-0 hard, always armed | dims / horizon / idx / id violation, time discontinuity, chunk-sequence discontinuity, or a runtime-raised schema fault | no -> `Fault` |
| 9 | `watchdog` | 512 | Tier-0 hard, always armed | `missed_ticks > watchdog_ticks` | no -> `Fault` |
| 10 | `tier1_cp` | 1024 | FSM | K-of-N predictive fire (`popcount(window) >= K`) | no -> `Braking` |
| 11 | `clamp_budget` | 2048 | FSM | `clamp_streak >= clamp_streak_to_brake` or `clamps > max_clamps_per_episode` | no -> `Braking` |
| 12 | `handoff_timeout` | 4096 | FSM | `escalated_ticks >= handoff_timeout_ticks` | no -> `Terminated` |
| 13 | `operator_abort` | 8192 | FSM | an accepted `Abort` ack | no -> `Terminated` |
| 14 | `brake_timeout` | 16384 | FSM | `brake_ticks >= brake_timeout_ticks` without a confirmed stop | no -> `Fault` |
| 15 | `rearm_budget` | 32768 | FSM | `Held` escalates because `rearms >= max_rearms` | no -> `Escalated` |

Derived constants: `TIER0_SOFT = workspace | speed | accel | jerk | reach | contact` (63),
`TIER0_HARD = brake | nonfinite | schema | watchdog` (960). `tier0_enabled` in the envelope can disarm
soft bits and `brake`; `nonfinite`, `schema` and `watchdog` are always armed. Soft trips whose projection
changed nothing still count as trips. A Tier-1 feature never raises a trip bit by itself: the decision is
on the single aggregate `s` against `tau`, windowed, and surfaces as `tier1_cp`.

`counts.trips_by_bit` in the receipt is a 16-entry array indexed by this table.

## 4. `ActionSource`

```rust
pub enum ActionSource { Policy, Clamped, Brake, Hold }
```

| value | produced in state | what `action` is |
|---|---|---|
| `policy` | `Armed`, `Watching` (and always in Observe mode) | `cur.action(idx)` -- the policy's row for this tick, unchanged |
| `clamped` | `Clamped` | the projected row (`clamped_dims` says which dims moved) |
| `brake` | `Braking` | `brake_action`: clamp_box(measured position) for position kinds, ramp-to-zero for velocity kinds; recomputed every tick |
| `hold` | `Held`, `Escalated`, `Fault`, `Terminated` | the hold setpoint latched on the tick the stop state was entered; never chases the object |

`substituted` is `action_src != policy` and is always `false` in Observe mode. `violation_reached_env` is
the Observe-mode counterpart: `true` when the would-be `action_src` was not `policy`, i.e. a Tier-0
violation (or a predictive stop) was let through on purpose. The receipt sums these into
`counts.violations_reached_env`, and `evaluate_fuse` sets `fuse_ok = false` whenever that count is
positive -- a signed, intact receipt can honestly say the fuse did nothing.

## 5. `ReasonCode` and `reason_text` -- the human-escalation glossary

Every verdict carries one `ReasonCode` and, on the wire, its `reason_text`: one plain-English sentence
(at most 120 characters, ending with a period) explaining WHY the fuse acted. `HandoffRecord` carries the
same pair for the tick on which `Escalated` was entered, which is what an operator reads before signing
an `AckToken`. `crates/lictor-core/src/reason.rs` (`reason_text`) is the normative text and is tested
by `crates/lictor-core/tests/reason_glossary.rs` (every code has a sentence, length, final period, no
banned vocabulary); the table below mirrors it.

| code | emitted on | reason_text |
|---|---|---|
| `ok` | a clean tick in `Armed` (Enforce) | All checks passed; the policy action is applied unchanged. |
| `observe_only` | a clean tick in `Armed` (Observe) | Observe mode: every check ran, but the raw policy action is passed through regardless of the result. |
| `watch_band` | entering or staying in `Watching` | A score is close to its threshold or the window has a hit; the policy action is applied unchanged. |
| `clamp_workspace` | `Clamped`, first soft bit = `workspace` | The chunk left the workspace box; it was projected back inside and motion continues. |
| `clamp_speed` | `Clamped`, first soft bit = `speed` | A commanded step exceeded the speed limit; it was shortened along its own direction. |
| `clamp_accel` | `Clamped`, first soft bit = `accel` | The chunk exceeded the acceleration limit; the offending step was shortened. |
| `clamp_jerk` | `Clamped`, first soft bit = `jerk` | The chunk exceeded the jerk limit; the offending step was shortened. |
| `clamp_reach` | `Clamped`, first soft bit = `reach` | The first chunk row was too far from the measured position; it was pulled toward it. |
| `clamp_contact` | `Clamped`, first soft bit = `contact` | Commanded speed near the object exceeded the contact limit; the step was shortened. |
| `brake_infeasible` | `-> Braking` on `brake` | The committed motion could not be stopped inside the workspace; the fuse is braking. |
| `brake_tier1_cp` | `-> Braking` on `tier1_cp` | The predictive score exceeded its calibrated threshold K of N times; the fuse is braking. |
| `brake_clamp_budget` | `-> Braking` on `clamp_budget` | Too many chunks needed projection in a row or in this episode; the fuse is braking. |
| `stopping` | staying in `Braking` | Braking is in progress; the setpoint is held at the measured position until the agent stops. |
| `held_standstill` | `Braking -> Held` and staying in `Held` | The agent has stopped; the fuse holds it in place until it is re-armed. |
| `escalate_hold_timeout` | `Held -> Escalated` after `escalate_after_hold_ticks` | Held 30 ticks without a clean re-arm window; a human must decide. |
| `escalate_rearm_budget` | `Held -> Escalated` with `rearms >= max_rearms` | The automatic re-arm budget for this episode is used up; a human must decide. |
| `escalate_tier1_persistent` | `Held -> Escalated` while `predictive` | The predictive alarm persisted through the hold; a human must decide. |
| `fault_non_finite` | `-> Fault` on `nonfinite` | A non-finite value reached the fuse; the hold action is applied for the rest of the episode. |
| `fault_schema` | `-> Fault` on `schema` | A malformed or discontinuous input reached the fuse; the hold action is applied for the rest of the episode. |
| `fault_watchdog` | `-> Fault` on `watchdog` | Too many ticks were missed; the hold action is applied for the rest of the episode. |
| `fault_brake_timeout` | `Braking -> Fault` on `brake_timeout` | Braking did not bring the agent to a stop in time; the hold action is applied for the rest of the episode. |
| `fault_internal` | reserved for an internal inconsistency | An internal inconsistency was detected; the hold action is applied for the rest of the episode. |
| `rearmed_auto` | `Held -> Armed` under `rearm = "auto"` | The agent stayed clean through the hold and a fresh chunk passed every check; the fuse re-armed itself. |
| `rearmed_ack` | `Held`/`Escalated -> Armed` on an accepted `Resume` ack | A listed operator signed a resume decision for this handoff; the fuse re-armed. |
| `terminated_abort` | `-> Terminated` on an accepted `Abort` ack | A listed operator signed an abort decision; the episode is over. |
| `terminated_timeout` | `Escalated -> Terminated` on `handoff_timeout` | No operator decision arrived within the handoff timeout; the episode is over. |
| `terminated_retune` | `-> Terminated` on an accepted `Retune` ack | A listed operator asked for a retune; the episode is over and a new envelope is expected. |
| `episode_end` | the receipt's final record | The episode ended; no further action is produced. |

The number in `escalate_hold_timeout` is the PushT base envelope's `escalate_after_hold_ticks`; the
sentence is fixed text, the envelope carries the actual value. The SS1 / SS2 / STO / monitored-standstill
words used elsewhere in the documentation are borrowed for readability; no conformance is claimed or
tested.

`reason_text`, `AckToken.note`, `fuse_notes` and the run / arm identifiers are stripped of ASCII control
characters and escape sequences by `crates/lictor-cli/src/render.rs` before any terminal rendering, so a
signed note cannot inject terminal escapes.

## 6. The verdict on the wire (`kind: "verdict"`, `lictor-wire/v1`)

Exactly one per `tick` request; `id` echoes the request. Numbers are plain JSON f64 (the wire is never
hashed, so readability wins); `null` in `scores.s` means `-inf` (no gate term fully valid, or Tier 1
disarmed) and `null` in `tau` means `+inf` (disarmed, or the degenerate `k > n_calib` case). The Python
client maps those two keys, and only those, to `-math.inf` / `+math.inf`.

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

| field | type | meaning |
|---|---|---|
| `t` | u32 | absolute env step (echoed from the request) |
| `seq` | u32 | tick sequence within the episode, from 0; `t == seq` unless ticks were missed |
| `status` | `Status` | severity after this tick (sec 1) |
| `state`, `prev_state` | `FuseState` | state after / before this tick; the pair is the per-tick diff |
| `trips`, `trip_mask` | names, u32 | the bits raised THIS tick (sec 3), twice |
| `action` | f64[action_dim] | the action the executor MUST apply verbatim |
| `action_src` | `ActionSource` | where `action` came from (sec 4) |
| `substituted` | bool | `action_src != policy`; always `false` in Observe |
| `clamped_dims` | u32 | bit c set when the projection moved dimension c |
| `scores.f` | f64[12] | raw Tier-1 features in `Feat` order: `tce acc acm_neg njr reach path_ineff stall speed_peak ext0 ext1 ext2 ext3`; `0.0` where not valid |
| `scores.z` | f64[12] | standardised `(f - center[bin t]) / scale[bin t]`; `0.0` where not valid or not in the gate mask |
| `scores.s` | f64 or null | the gated aggregate `max over terms of min over the term`; `null` = `-inf` |
| `scores.valid` | u32 | bit j set when feature j was computable this tick |
| `scores.fired` | u32 | bit j set when `z_j > tau` (informational; the decision is on `s`) |
| `tau` | f64 or null | the split-conformal threshold; `null` = `+inf` |
| `window_hits` | u8 | popcount of the K-of-N ring after this tick |
| `brake_margin` | f64 | slack of the braking check in env units; `< 0` means infeasible |
| `reason`, `reason_text` | `ReasonCode`, string | sec 5 |
| `handoff` | `HandoffRecord` or null | non-null exactly on the tick `Escalated` is entered |
| `ack_result` | string or null | `"accepted"`, or `"rejected:<unknown_operator|bad_signature|wrong_handoff|nonce_replay|no_pending_handoff>"`; an ack rejection is reported here, never as an `error` |
| `violation_reached_env` | bool | Observe mode: the would-be `action_src` was not `policy` |
| `verdict_ns` | u64 | wall-clock around `decide()`, clock read outside the pure function; WSL2-labelled in the receipt |
| `chain` | hex64 | the verdict-chain head after this tick (`hash` of this tick's `TickEvent`) |

The `SafetyVerdict` struct additionally carries `window: u64` (the raw ring), `action_dim`,
`handoff_seq` and `ack_consumed`; the wire projects them into `window_hits`, the `action` length,
`handoff` and `ack_result`.

After a fatal `error` (schema, protocol, envelope) the session latches `Fault`; every later `tick` of
that episode still goes through `decide()` with `schema_fault = true` and returns `status:"fault"`,
`action_src:"hold"`. The fuse never returns the raw policy action after a fault, and `episode_end` is
accepted while faulted so the episode enters the ledger as a failure.

## 7. The verdict in the ticks file (`TickEvent`, `lictor-ticks/v1`)

`results/<run>/<arm>/ticks/<episode:06>.jsonl`: a header line
`{"schema":"lictor-ticks/v1","run_id":"...","arm_id":"...","episode_index":7,"genesis":"0000...0000"}`
followed by one `TickEvent` per tick in the canonical `jcs-floatfree/v1` encoding (compact, keys sorted,
every real number as `{"f64":"<16 hex>"}` and every real array as `{"f64a":"<base64 LE>","shape":[..]}`):

```json
{"action":{"f64a":"AAAAAADAKkBmZmZmZsZyQA==","shape":[2]},"action_src":"policy","brake_margin":{"f64":"404499999999999a"},"clamped_dims":0,"f":{"f64a":"...","shape":[12]},"fired":0,"handoff_seq":null,"hash":"5b7e...","prev":"0000...","prev_state":"armed","reason":"ok","s":{"f64":"3ff199999999999a"},"seq":24,"state":"armed","status":"nominal","substituted":false,"t":24,"tau":{"f64":"400db6db6db6db6e"},"trips":0,"valid":255,"violation_reached_env":false,"window_hits":0,"z":{"f64a":"...","shape":[12]}}
```

Rules:

- `hash = sha256(canon(event with "hash":""))`; `prev` of seq 0 is the genesis (64 zeros); the receipt
  signs `verdict_chain_head` (the last `hash`) and `verdict_events` (the count).
- The event contains ONLY what a replay reproduces: no `reason_text` (derivable from `reason`), no
  `HandoffRecord` (it lives in `receipt.body.handoffs`), no `verdict_ns` (that is the separate, honestly
  non-replayable timing chain in `timing/<episode:06>.jsonl`: `{seq, decide_ns, io_ns, prev, hash}`),
  no `ack_result` (the ack itself is recorded in the handoff record), no raw `window` (only
  `window_hits`).
- `s` and `tau` are stored as their exact bit patterns, so `-inf` is `fff0000000000000` and `+inf` is
  `7ff0000000000000`; nothing is `null` in the file.
- `trips` is the integer mask; the names are recovered through `TripMask::names`.
- The receipt embeds the last 32 events (`ticks_policy: "tail32"`, or `all` / `none`); the full stream is
  the file. `lictor verify --ticks` recomputes the whole chain, compares the recomputed head with the
  signed head (`HEAD MISMATCH` when they differ), and cross-checks the header and the embedded tail's
  first `prev` against the file.

`lictor replay` re-feeds the recorded requests through a fresh session and compares only the verdict
chain head; it prints `timing chain head varies (by design -- wall-clock is not replayed)` every time.

## 8. Diff and streak semantics (`lictor history`)

Two levels of "what changed", both derived from verdicts and receipts, neither hashed:

**Per tick (the diff).** Every verdict carries `prev_state -> state`. A transition row of the state
machine fires exactly when they differ, and `reason` names the row. The receipt's `counts` sums the ticks
spent in each state (`nominal`, `watching`, `clamped`, `braking`, `held`, `escalated`, `fault`,
`terminated`), the substitutions, the chunks seen and rejected, `clamps`, `holds`, `rearms`,
`escalations`, and the first-event ticks (`first_trip_tick`, `first_trip_reason`, `first_stop_tick`,
`handoff_tick`).

**Per episode (the run memory, sbx `history.rs` lineage).** After every episode the runtime appends one
`EpisodeRecord` to `$LICTOR_KEYS/history.jsonl` (default `$HOME/.lictor/history.jsonl`; `--dir`
overrides), capped at 200 lines (the file is rewritten when the cap is exceeded):

```json
{"ts":"2026-09-01T09:41:02Z","run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","seed":7,"success":false,"first_trip_reason":"brake_tier1_cp","trips":["tier1_cp"],"stopped":true,"escalated":false}
```

`lictor history [--dir DIR] [--n 20]` calls `summarize(dir, n)` over the last `n` records and prints
`LoopSignals`:

| signal | definition | printed as |
|---|---|---|
| `tail_streak: Option<(reason, count)>` | the most recent record's `first_trip_reason` and how many of the last `n` records share it; `None` when fewer than 2 share it or the last record has no trip | `tail-streak: brake_tier1_cp in 4 of the last 6 episodes on t01-a05-d0` |
| `identical_run: u32` | the length of the run of consecutive most-recent records with the same `(arm_id, trips)` tail | `identical: the last 3 episodes tripped the same bits on t01-a05-d0` |
| `last_n: u32` | how many records were actually examined (fewer than `n` on a fresh directory) | `episodes examined: 6` |

The streak lines are the fuse's anti-loop signal for whoever is tuning an envelope or a calibration: the
same `first_trip_reason` episode after episode means the threshold, not the policy, is what the operator
is looking at. `--json` prints the `LoopSignals` object instead. `history.jsonl` is a convenience, never a
source of truth; the ledger and the receipts are.
