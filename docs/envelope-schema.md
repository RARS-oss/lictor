# Envelope schema (`lictor-envelope/v1`)

The envelope is the safety configuration: the embodiment manifest, the geometric / kinematic limits, the
braking model, the projection mode, the hysteresis of the state machine, which Tier-0 checks are armed,
the default Tier-1 gate, and the operator keys allowed to re-arm. It is a TOML file, parsed by
`SafetyEnvelope::from_toml` (`crates/lictor-core/src/envelope.rs`), validated by `validate()`, compiled
once, off the hot path, into the fixed-array `FuseConfig` by `compile()`, and digested into every receipt.
The frozen Rust definitions are in `docs/ARCHITECTURE.md` sec 4; the base file for PushT is
`envelopes/pusht.base.toml` (its fenced text in `docs/ARCHITECTURE.md` sec 9 is normative, byte for
byte). Unknown keys anywhere are a parse error: an envelope never silently ignores a misspelt limit.

Three envelope files exist on PushT:

| file | produced by | limits | operators |
|---|---|---|---|
| `envelopes/pusht.base.toml` | hand-written (WP-1) | PLACEHOLDERS, loose enough not to touch the baseline | `[]` |
| `envelopes/pusht.toml` | `lictor envelope fit` (committed by WP-13) | empirical p99.9 x slack over calibration successes | `[]` |
| `envelopes/pusht.oracle.toml` | `lictor envelope fit ... --operator <hex>` | the same | the demo operator key |

All three share one `[embodiment]` table and therefore one `embodiment_digest`, which is what a
calibration binds to (sec 6 below); their `envelope_digest`s differ and are printed by `lictor verify`.

## 1. Top-level keys

| key | type | PushT base | units | meaning and validation |
|---|---|---|---|---|
| `schema` | string | `"lictor-envelope/v1"` | -- | must be exactly this |
| `envelope_id` | string | `"pusht-base-v1"` | -- | free identifier; bound into the digest |
| `box_lo` | f64[pos_dim] | `[15.0, 15.0]` | env units (px) | workspace box, lower corner; agent radius 15 px on a `[0,512]^2` table |
| `box_hi` | f64[pos_dim] | `[497.0, 497.0]` | env units | upper corner; `box_lo[c] + margin < box_hi[c] - margin` for every c |
| `margin` | f64 | `2.0` | env units | subtracted from the box once at compile (`lo = box_lo + margin`, `hi = box_hi - margin`); `>= 0` |
| `v_max` | f64 | `1000.0` | units/s | commanded speed limit `||a_i - q_{i-1}|| / dt`; `> 0`; the per-step leash is `step_max = v_max * dt` (PushT: 100 px per 0.1 s) |
| `a_max` | f64 | `20000.0` | units/s^2 | commanded acceleration limit (second difference over `dt^2`); `> 0` |
| `j_max` | f64 | `400000.0` | units/s^3 | commanded jerk limit (third difference over `dt^3`); `> 0` |
| `reach_max` | f64 | `150.0` | env units | teleport guard `||a_0 - p_t||`; `> 0` |
| `clamp_mode` | `"off"` or `"project"` | `"project"` | -- | `project`: soft trips are corrected by the direction-preserving leash and motion continues (`Clamped`); `off`: any soft trip brakes (FSM row 17b) |
| `tier0_enabled` | string[] | `["workspace","speed","accel","jerk","reach","brake"]` | -- | subset of `workspace speed accel jerk reach contact brake`; `nonfinite`, `schema`, `watchdog` are always armed; unknown names are rejected; `contact` requires a `[contact]` table |
| `gate` | string[] | `["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"]` | -- | the default Tier-1 gate: a fixed-order DNF over feature names, `|` inside a term means AND (`"acc|path_ineff"`); at most 8 terms; unknown feature names are rejected; overridden by `calibration.json` when a calibration is loaded |
| `operators` | string[] | `[]` | hex64 | Ed25519 public keys allowed to sign an `AckToken`; slot = index; at most 8; each exactly 64 lowercase hex characters; NEVER edited at run time (the oracle arm uses `pusht.oracle.toml`) |
| `fail_closed` | bool | `true` | -- | MUST be `true` in v1; recorded so a future `false` is auditable |
| `[embodiment]` | table | -- | -- | sec 2 |
| `[brake]` | table | -- | -- | sec 3 |
| `[contact]` | table, optional | absent | -- | sec 4; disarmed unless `"contact"` is in `tier0_enabled` |
| `[hysteresis]` | table | -- | -- | sec 5 |
| `[fit]` | table, optional | absent | -- | sec 7; written by `lictor envelope fit` |

Base values are placeholders. Hand-picked limits are either decorative or dishonest; the fitted file
(sec 7) is what every curve point runs under.

## 2. `[embodiment]` -- the manifest (`EmbodimentManifest`)

Everything a Tier-1 feature may depend on lives here and ONLY here. Fitting the limits above or changing
the operator list does not touch this table, so a calibration fitted under the base envelope stays valid
under the fitted one; changing anything in this table changes `embodiment_digest` and invalidates every
calibration bound to it.

| key | type | PushT | meaning and validation |
|---|---|---|---|
| `id` | string | `"gym_pusht/PushT-v0"` | must equal `hello.embodiment_id` |
| `units` | string | `"px"` | documentation (`px`, `rad`, `m`); bound into the digest |
| `action_kind` | enum | `"ee_position"` | `ee_position`, `ee_delta`, `joint_position`, `joint_velocity`, `other`; selects the brake action (position kinds: setpoint := clamped measured position; velocity kinds: ramp to zero) |
| `action_dim` | u16 | `2` | `1..=32` (`MAX_D`); must equal `hello.action_dim` and every chunk's `d` |
| `pos_dim` | u16 | `2` | `1..=32` (`MAX_POS`); the length of `obs.pos` / `obs.vel`, `box_lo`, `box_hi` |
| `horizon` | u16 | `15` | `1..=64` (`MAX_H`); the chunk length the adapter DELIVERS (PushT: `n_action_steps = 16 - 2 + 1 = 15`); every chunk's `h` must equal it |
| `exec_steps` | u16 | `8` | `1..=horizon`; the irrevocably committed prefix; overlap `L = horizon - exec_steps` (7) |
| `control_hz_num`, `control_hz_den` | u32, u32 | `10`, `1` | control period `dt = den / num` seconds, exact rational; both `> 0` |
| `provides_vel` | bool | `true` | the host sends the plant's own velocity as `obs.vel` (PushT: `info["vel_agent"]`); `false` -> finite-difference `v_hat` and the fuse note `v_hat finite-differenced` in every receipt |
| `horizon_ticks` | u32 | `300` | the episode time base for calibration binning (`bin(t) = min(99, t*100/300)`); `> 0`; MUST equal the env's `max_episode_steps` (`lictor calibrate` refuses a mismatch) |
| `norm_center` | f64[action_dim] | `[256.0, 256.0]` | Tier-1 uses `abar = (a - center) / scale` |
| `norm_scale` | f64[action_dim] | `[256.0, 256.0]` | every entry `> 0`; `norm_scale_iso = min_c norm_scale[c]` feeds `reach`, `speed_peak` and the stall reference speed `0.05 * norm_scale_iso / dt` (128 px/s) |
| `aux_layout` | string[] | `["block_x","block_y","block_theta","coverage"]` | names of the privileged auxiliary scalars in `obs.aux`; at most 16 (`MAX_AUX`); PushT sends `coverage` on every tick (`_get_coverage()` at t = 0) |
| `ext_names` | string[] | `[]` | names of the Tier-2 scalars `ext0..ext3` in `obs.ext`; at most 4 (`MAX_EXT`) |

## 3. `[brake]` -- the braking model (`BrakeModel`)

| key | type | PushT | meaning |
|---|---|---|---|
| `kind` | enum | `"pd_second_order"` | `pd_second_order` (exact forward rollout of the env's second-order PD agent), `first_order_decay` (`d_stop = ||v|| / k_v`), `bounded_accel` (`||v||^2 / (2 a_max)`), `jerk_limited` (`||v||^2 / (2 a_max) + ||v|| a_max / (2 j_max)`), `zero_velocity_hold` (velocity-command embodiments; feasibility as `bounded_accel`, brake = ramp to zero) |
| `k_p`, `k_v` | f64 | `100.0`, `20.0` | the plant's PD gains; must equal the env's (`gym_pusht/envs/pusht.py`: `k_p, k_v = 100, 20`) |
| `substeps` | u16 | `10` | physics substeps per control tick; must equal the env's inner loop |
| `dt` | f64 | `0.01` | the PHYSICS substep, not the control period; `substeps * dt` should equal the control period |
| `commit_steps` | u16 | `8` | rows of the irrevocable prefix simulated forward (`exec_steps` on PushT) |
| `brake_steps` | u16 | `8` | extra hold steps simulated after the prefix (`pd_second_order` only) |
| `react_ticks` | u16 | `1` | dead time before a brake bites (closed-form kinds only) |

`pd_second_order` runs EXACTLY `(commit_steps - from + brake_steps) * substeps` iterations (at most 160
on PushT) of `acc = k_p (a - p) - k_v v; v += acc dt; p += v dt` from the measured position and the
simulator's own velocity, tracking the minimum distance to the margin-adjusted box; `feasible` means that
minimum never went negative. The model is contact-free: contact only decelerates the agent in this
environment, so the prediction is conservative for the box constraint -- a modelling assumption, stated
as such. From a finite-difference `v0` (`provides_vel = false`) the rollout is approximate.

## 4. `[contact]` -- reduced speed near the object (`ContactLimit`, optional, off by default)

| key | type | example | meaning |
|---|---|---|---|
| `radius` | f64 | `80.0` | env units; the check applies when `||p_t - c_block|| <= radius` |
| `v_max` | f64 | `400.0` | units/s; commanded speed limit inside the radius |
| `aux_center` | [u8; 2] | `[0, 1]` | indices into `obs.aux` holding the object centre (`block_x`, `block_y`) |

Perception-assisted: it reads privileged simulator state through `aux`. When it is shown it is a
separate, labelled line, never folded into the headline curve.

## 5. `[hysteresis]` -- the state machine's knobs (`Hysteresis`)

| key | type | PushT | meaning and validation |
|---|---|---|---|
| `k`, `n` | u8, u8 | `3`, `5` | K-of-N on the predictive channel; `1 <= k <= n <= 63` (`window_mask = (1 << n) - 1` must fit in a u64) |
| `warn_margin` | f64 | `0.5` | z units; `s > tau - warn_margin` enters `Watching` |
| `clear_ticks` | u8 | `5` | clean ticks to leave `Watching` / `Clamped` |
| `clamp_streak_to_brake` | u8 | `3` | consecutive clamped CHUNKS before `Braking` (`clamp_budget`) |
| `max_clamps_per_episode` | u16 | `60` | clamped ticks per episode before `Braking` (`clamp_budget`) |
| `stop_confirm_ticks` | u8 | `2` | `Braking -> Held` once `||v_hat|| <= v_stop_eps` for this many consecutive ticks |
| `v_stop_eps` | f64 | `5.0` | units/s |
| `brake_timeout_ticks` | u16 | `30` | `Braking` that never confirms a stop -> `Fault` (`brake_timeout`) |
| `rearm` | enum | `"auto"` | `auto` (protective-stop semantics: the fuse may re-arm itself from `Held`) or `ack_only` (only a signed `AckToken` leaves `Held` / `Escalated`) |
| `rearm_hold` | u16 | `10` | clean ticks in `Held` before an automatic re-arm |
| `max_rearms` | u8 | `2` | automatic re-arms per episode; exhausting it escalates (`rearm_budget`) |
| `escalate_after_hold_ticks` | u16 | `30` | `Held` this long -> `Escalated` (a `HandoffRecord` is emitted) |
| `handoff_timeout_ticks` | u32 | `200` | `Escalated` with no ack -> `Terminated` (`handoff_timeout`) |
| `watchdog_ticks` | u8 | `2` | `missed_ticks` above this -> `Fault` (`watchdog`) |

## 6. Digests

Two digests are computed from the parsed structure, never from the file's bytes:

- `envelope_digest = sha256(canon(floatify(json(envelope))))`, where `json(envelope)` is serde's JSON of
  `SafetyEnvelope` (struct field names as keys, TOML tables as nested objects, an absent `contact` /
  `fit` as `null`), `floatify` replaces every real number by `{"f64":"<16 hex>"}` and every real array by
  `{"f64a":"<base64 LE>","shape":[..]}`, and `canon` is RFC 8785 JCS (`jcs-floatfree/v1`, see
  `docs/receipt-schema.md`). Comments, key order and whitespace in the TOML do not enter the digest;
  every value does, including `operators` (WHO may re-arm is signed) and `fit`.
- `embodiment_digest` = the same over `json(envelope.embodiment)` alone.

Consequences: `lictor envelope fit` (which changes only `v_max`, `a_max`, `j_max`, `reach_max`, `fit`)
and `--operator` (which changes only `operators`) change `envelope_digest` and leave `embodiment_digest`
unchanged. `calibration.json` binds `embodiment_digest` (plus `policy_digest`); `lictor serve` and
`lictor replay` refuse at startup (exit 2) a calibration whose `embodiment_digest` differs from the served
envelope's, and `episode_begin` refuses a `weights_sha256` that differs from the calibration's
`policy_digest`. The receipt embeds the floatified envelope in full and `lictor verify` recomputes
`envelope_digest` from it (`envelope ok digest recomputed ... embodiment ...`). `hello` must carry the
digest of the envelope the server loaded, or the session refuses with `error{code:"envelope"}`.

`lictor envelope digest F.toml` prints the envelope digest on stdout and `embodiment=<hex>` on stderr;
`lictor envelope check F.toml` validates and prints both. Under `no_std` the compiled `FuseConfig` carries
zero digests (the MCU path takes them out of band; roadmap item 9).

## 7. The fit procedure (`lictor envelope fit`) -- summary

Hand-picked limits are decorative; the procedure is:

1. Run the calibration pool (seeds `900000..900299`; 100 episodes in the pilot) in Observe mode under
   the BASE envelope (arm `calib-obs`). Observe mode never changes a trajectory, so this one pass is at the
   same time the calibration source (`lictor calibrate`) and the fit source.
2. `lictor envelope fit --run <DIR> --arm calib-obs --base envelopes/pusht.base.toml --quantile 0.999
   --slack 1.25 -o envelopes/pusht.toml --report docs/envelope_fit_report.md`: over the SUCCESSFUL
   calibration episodes only, recompute the commanded speed, acceleration, jerk and reach of every
   delivered chunk with the same formulas Tier 0 uses, take the empirical nearest-rank quantile of each
   (0.999), multiply by the slack (1.25), and write a new envelope whose `[embodiment]` table is
   byte-identical to the base (a test asserts `embodiment_digest` unchanged), whose four limits are the
   fitted values, and whose `[fit]` table records `source_run`, `quantile`, `slack`, `n_episodes`,
   `fitted_utc` and a `note`. The report lists, per quantity, `n`, p50, p99, p99.9, the chosen limit and
   the base limit.
3. Once more with `--operator <demo pubkey hex>` -> `envelopes/pusht.oracle.toml` for the
   `oracle_resume` arm: identical embodiment, identical calibration, a different `envelope_digest`.
4. Parity gate before any curve point is reported: `t0-d0` vs `obs-d0` on the first 60 paired eval seeds
   must give McNemar exact `p > 0.05` with overlapping Clopper-Pearson intervals. If the fitted envelope
   costs measurable success it is too tight: raise the slack, refit, and record every refit in the report.

The `[fit]` keys: `source_run` (string), `quantile` (f64 in `(0, 1)`), `slack` (f64 `>= 1`),
`n_episodes` (u32), `fitted_utc` (RFC 3339 string; documentation only, never affects a verdict), `note`
(string). The Tier-1 side of calibration (robust standardisation, the split-conformal quantile, the
K-of-N caveat, the worked numbers for `n_calib` = 137 / 59 / 45) is `docs/calibration.md`.

## 8. CLI

```
lictor envelope init  --profile pusht -o <F.toml> [--operator <hex>]...   # emits the base file
lictor envelope check <F.toml>                                            # validate + both digests
lictor envelope digest <F.toml>                                           # envelope digest (stdout), embodiment (stderr)
lictor envelope show <F.toml>                                             # human rendering; --json for the structure
lictor envelope fit   --run <DIR> --arm <ARM> --base <F.toml> [--quantile 0.999] [--slack 1.25] [--operator <hex>]... -o <F.toml> [--report <F.md>]
```

Every command accepts `--json`. Validation failures are `EnvelopeError::{Parse, Invalid, Unsupported}`
with the offending key named; `lictor serve` exits 2 on any of them.
