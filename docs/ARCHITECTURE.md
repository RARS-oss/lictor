# lictor -- ARCHITECTURE (milestone 1: the PushT safety-vs-latency curve)

**A deterministic real-time safety fuse over learned robot policies: failure prediction + geometric limit enforcement + human escalation, with Ed25519-signed, hash-chained, replayable receipts.**

Author: RARS-oss (author of bulla / sbx). Companion to `docs/ANALYSIS.md`. This document is the design freeze for milestone 1; `docs/IMPLEMENTATION_PLAN.md` carries the same interface freeze plus the parallel work packages. Everything in the FROZEN sections is normative: if an interface is missing here it is a bug in this document, not a licence to improvise.

Merged from two design lenses -- experiment-first (design 1) and audit-first (design 2). Section 13 records every conflict and which side won.

---

## 0. Ground truth about the target (verified, not assumed)

| Fact | Value | Source |
|---|---|---|
| Env | `gym_pusht/PushT-v0`, `max_episode_steps=300`, `render_fps=10` -> control period 0.1 s | gym_pusht `__init__.py`, `pusht.py` |
| Action | `Box(0, 512, (2,), float32)` -- 2-D end-effector position setpoint in px | `pusht.py` |
| Agent dynamics | per env step 10 substeps of `acc = k_p (a - p) - k_v v; v += acc dt; p += v dt`, `k_p=100`, `k_v=20`, `dt=0.01` | `pusht.py` |
| Success | `coverage > 0.95`; `info["coverage"]`; reward `clip(coverage/0.95, 0, 1)` | `pusht.py` |
| Velocity | `info["vel_agent"] = np.array(self.agent.velocity)` on EVERY step (`_get_info`, called from `reset` and `step`): the simulator's own instantaneous velocity, so `provides_vel = true` and the PD brake rollout starts from the true `v0`. `info["coverage"]` is set ONLY in `step()` (never in `reset()`), so the harness sends `env.unwrapped._get_coverage()` at `t = 0` | gym_pusht 0.1.6 `pusht.py:263, 414-419` (installed) |
| Policy | `lerobot/diffusion_pusht`: `horizon=16`, `n_action_steps=8`, `n_obs_steps=2`, action `[2]`, state `[2]`, image `[3,96,96]`, DDPM 100 train steps, `num_inference_steps=null` (-> 100), MIN_MAX normalisation | `config.json` |
| Published baseline | `pc_success = 65.4` over 500 episodes, `eval_ep_s = 1.46` on the reference GPU | `eval_info.json` |
| Checkpoint | last modified 2025-03-06 -- PRE normalisation migration; on lerobot 0.6.1 a direct `from_pretrained` raises `ProcessorMigrationError`; the fix is `python <site-packages>/lerobot/processor/migrate_policy_normalization.py --pretrained-path lerobot/diffusion_pusht --output-dir $LICTOR_MODELS/diffusion_pusht_migrated` and loading policy + processors from the migrated path (done once on this machine; the harness binds `normalization_migrated` and `migration_script` into the receipt) | HF hub; VERIFIED_FACTS 2026-08-31 |
| Chunk slicing | `generate_actions` runs `conditional_sample` over the FULL horizon (prior `torch.randn(B, horizon, action_dim)` and per-step scheduler noise both shaped by `horizon`, never by `n_action_steps`) and returns `actions[:, start:start+n_action_steps]` with `start = n_obs_steps - 1 = 1`; `DiffusionConfig.__post_init__` does NOT validate `n_action_steps <= horizon - n_obs_steps + 1` | `modeling_diffusion.py` (installed 0.6.1) |
| Queues | `select_action` re-plans only when its ACTION deque (`maxlen = n_action_steps`, set in `reset()`) is empty -- with the override it would execute 15 before re-planning. `populate_queues` DUPLICATES the first observation to fill `n_obs_steps` at `t = 0`, afterwards `[obs_{t-1}, obs_t]` | `modeling_diffusion.py`, `lerobot/policies/utils.py` |

**The single most consequential finding (design 1).** Stock `predict_action_chunk()` returns only the 8 executed actions, so consecutive chunks have ZERO overlap and Temporal Consistency Error (the backbone of black-box action-space detection: ActProbe / VLA-FAIL ACC / STAC) would be unavailable. The fix is exact and non-invasive:

> Set `policy.config.n_action_steps = horizon - n_obs_steps + 1 = 15` and let the HARNESS own the execution cadence at `exec_steps = 8`.

Then `generate_actions` slices `actions[:, 1:16]` -> H = 15 actions; chunk k covers `[t_k, t_k+14]`, chunk k+1 (emitted at `t_k+8`) covers `[t_k+8, t_k+22]`; overlap `L = H - S = 7`. The executed sequence is byte-identical to stock LeRobot because the executed actions are exactly the first 8 of the 15 (same RNG state, same batch -> same 16-row sample -> same slice prefix). HOW the harness must drive it (verified against the installed 0.6.1 source): it must NOT call `select_action` (its deque would re-plan every 15 steps); it calls `policy.predict_action_chunk(batch)` on the OFFLINE path (all `policy._queues` empty, asserted before every call) with a hand-built `(B=1, n_obs_steps=2, ...)` history batch that replicates `populate_queues` exactly (the observation DUPLICATED at `t = 0`, `[obs_{t-1}, obs_t]` afterwards), and executes the first 8 rows itself. `harness/tests/test_h15.py` asserts, on the same seed: `policy.diffusion.config is policy.config`, `n_action_steps == horizon - n_obs_steps + 1 == 15`, the manual batch `torch.equal` to what `populate_queues` builds, chunk shapes `(1,15,2)` vs stock `(1,8,2)`, `torch.equal(post(a8), post(a15)[:, :8])` through the same postprocessor, `num_inference_steps == 100`, fp32 parameters, and `torch.are_deterministic_algorithms_enabled()` -- so the published 65.4 % baseline claim survives.

**Policy determinism.** `conditional_sample` draws initial noise and per-step variance noise from the global torch RNG. The harness calls `torch.manual_seed(hash64(episode_seed, chunk_index))` before every `predict_action_chunk`, making the policy output a pure function of `(episode_seed, chunk_index, observation)`. "Paired" means: same reset state (`init_state_digest`) + same per-chunk noise; trajectories diverge after the first substitution or delayed action BY DESIGN, and that divergence is the measurement. DO NOT batch episodes across envs to gain throughput: DDPM's `scheduler.step` draws `(B, 16, 2)` variance noise from the global RNG per batch, so batching is incompatible with per-episode seeding -- nobody should "optimise" this later.

---

## 1. Overview and data flow

Two planes (Simplex / LITHE split), one byte stream between them. The best-effort plane may allocate, GC, page-fault and stall; the fuse plane may not.

```
+============ BEST-EFFORT PLANE -- Python 3.12 @ /mnt/d/lictor/venv, torch 2.7.1+cu126 fp32 (GTX 1060 3GB) or CPU ============+
|                                                                                                                              |
|  gym_pusht/PushT-v0            lerobot DiffusionPolicy             harness/pusht_rollout.py                                   |
|  10 Hz . 300 steps max         H=15 (n_action_steps override)      arm . seed . delay injection (sync|async)                 |
|  reset(seed=S)                 S=8 executed . DDPM x100            resumable JSONL index . paired seeds                       |
|       |                              ^          |                        |                                                    |
|       | obs {pixels 96^2, agent_pos} |          | chunk 15x2 px          v                                                    |
|       +------------------------------+          +-------------------> harness/executor.py                                     |
|                                                                        delay queue d (sync: hold-last; async: drain prev)   |
|                                                                        per env step t: {obs_t, chunk?, idx}  ---------+     |
+========================================================================================================================|=====+
                                     lictor-wire/v1 . NDJSON . one line per tick . stdin/stdout                          |
+============ FUSE PLANE -- one `lictor serve` process per worker (Rust, panic=abort, no_std core) =======================|=====+
|                                                                                                                        v     |
|  stdin -> codec -> wire::Request -> stage ChunkBuf / ObsView --->  +------------------- decide() -------------------+        |
|  (alloc OK here -- off the measured path)                          | PURE. no alloc . no clock . no lock . no       |        |
|                                                                    | syscall . no panic . fixed iteration bounds    |        |
|                                                                    |                                                |        |
|                                                                    |  0 GUARD     nonfinite | schema | watchdog     |        |
|                                                                    |  1 INGEST    pos, v_hat, trail, chunk copy     |        |
|                                                                    |  2 TIER 0    box|speed|accel|jerk|reach|contact|        |
|                                                                    |              -> project (leash) or hard trip   |        |
|                                                                    |              BRAKE FEASIBILITY: 160-step exact |        |
|                                                                    |              PD rollout of the committed prefix|        |
|                                                                    |  3 TIER 1    tce acc acm_neg njr reach         |        |
|                                                                    |              path_ineff stall speed_peak ext*  |        |
|                                                                    |  4 CONFORMAL z=(f-c[bin t])/s[bin t]; s=DNF;   |        |
|                                                                    |              s > tau  (split-CP quantile)      |        |
|                                                                    |  5 WINDOW    K-of-N u64 popcount               |        |
|                                                                    |  6 FSM       Armed..Terminated (table, sec 6)  |        |
|                                                                    |  7 ACTION    policy | clamped | brake | hold   |        |
|                                                                    +------------------------+-----------------------+        |
|                                                                                             | SafetyVerdict (Copy, ~600 B)   |
|  stdout <- codec <- wire::Response::Verdict <---------------------------------------------+                                 |
|                                                                                             v                                |
|  verdict_chain: sha256(canon(TickEvent))  -- replayable      timing_chain: sha256(canon(TimingEvent)) -- honestly not        |
|  (folded on the I/O thread AFTER decide() returns; never inside the measured region)                                        |
|                                                                                             |                                |
|  episode_end -> ReceiptBody{run, budget, envelope, calibration, counts, outcome, handoffs, chain heads, latency, ticks tail} |
|                     -> Ed25519 sign ONCE -> results/<run>/<arm>/receipts/NNNNNN.json  -> ledger.jsonl (hash-chained)         |
+==============================================================================================================================+
                                                        |
        ------------------- OFFLINE LANE (CPU, seconds, no GPU, fully deterministic) -------------------
        results/<run>/<arm>/{traces,ticks,receipts,ledger}
              |-> lictor envelope fit  -> envelopes/pusht.toml     (empirical p99.9 x slack; McNemar parity gate)
              |-> lictor calibrate     -> calibration.<alpha>.json (split-CP over binned robust z; K-of-N)
              |-> lictor sweep         -> sweep.jsonl              (Layer A: TPR/FPR/lead/ROC/AUCPDT for EVERY alpha x detector)
              |-> lictor curve         -> curve/<arm>.json (signed) + summary.csv   (Layer B: from receipts ONLY; refuses on broken ledger)
              |-> lictor replay --repeat 40 -> "replays 40/40 byte-identical"        (determinism evidence: 40 replays, one machine)
              |-> adapters/verify_receipt.py (stdlib) -> OK | FAIL                  (third-party verification, no Rust)
              +-> harness/analyze.py + figures.py -> docs/figures/*.svg
```

**Why the fuse owns the executor gate.** The harness sends `{obs, chunk?, idx}` and executes `verdict.action` VERBATIM. The fuse holds the currently-executing chunk in its own arena and returns one action per tick. In `observe` mode it always returns the raw policy action (asserted per tick by the harness) while computing every score, transition and tally, so the fuse-off baseline, the calibration source and the Layer-A trace source are ONE run -- and "observe is dynamically identical to no-lictor" is an asserted invariant, not an assumption.

**Fail-closed at both boundaries.** Inside: any non-finite input, schema/continuity violation or watchdog miss latches `Fault` and substitutes the hold action for the rest of the episode. Outside: the host's default output for a tick is `clamp_box(current measured position)`; only a verdict arriving with a matching `id` replaces it; a dead child is killed, respawned and the episode is booked as `fuse_crash` (a failure), never silently skipped. A latency outlier degrades liveness (false-stop rate on the curve), never safety.

---

## 2. The two-plane split

| | Best-effort plane | Fuse plane |
|---|---|---|
| Language | Python 3.12 (lerobot 0.6.1, torch 2.7.1+cu126) | Rust 2021, `panic = "abort"` |
| May | allocate, block on GPU, page-fault, be nondeterministic across machines | none of these on the decision path |
| Owns | policy inference, env stepping, delay injection, run bookkeeping, figures | `decide()`, chains, receipt assembly, signing, ledger |
| Time | wall-clock | integer ticks only (`t`, `idx`, `missed_ticks`), with `t` and chunk `seq` continuity checked in GUARD; the clock is read only around `decide()` for the timing chain. Wall-clock enters ONLY `TimingEvent` (monotonic `Instant`, never negative), `created_epoch`, `fitted_utc` and the index `ts`; none of them affects any verdict or verification. A WSL2 clock jump can at most cause a spurious client timeout, recorded as `ended_by: fault` |
| Interface | `lictor-wire/v1` NDJSON over the child's stdin/stdout | same |
| Trust | untrusted input (schema-checked, dimension-checked, finite-checked, continuity-checked) | trusted; a panic aborts the process, which the host treats as a brake and books as `fuse_crash` |

Why a subprocess and not pyo3 (both designs agreed): pyo3 forbids `panic = "abort"`, drags the fuse into torch's allocator/GIL/GC and contaminates every latency measurement; the subprocess makes the Simplex split an OS boundary, makes the wire log the replay fixture for free, needs zero build system beyond cargo on this machine, and (because no crate depends on `nix`/`libc`/`seccompiler`) a Windows-native `lictor.exe` can be driven from WSL Python as insurance. IPC cost is reported separately (`io_ns` in the timing chain) from the decision cost (`decide_ns`): the headline "verdict costs N us" is never contaminated by pipe latency, and the pipe latency (tens of us per ~1 KB line) is stated, not hidden.

Deferred (roadmap, not built): a logger thread fed by a preallocated SPSC ring (design 2) so chain folding leaves the I/O thread. At 10 Hz the per-tick sha256 over ~600 canonical bytes is irrelevant to the tick budget, and the measured region already excludes it.

---

## 3. Cargo workspace layout

```
lictor/
|-- Cargo.toml  Cargo.lock  rust-toolchain.toml  rustfmt.toml  .cargo/config.toml  Makefile  pyproject.toml  .gitignore
|-- LICENSE(MIT)  README.md  SECURITY.md  CONTRIBUTING.md
|-- crates/
|   |-- lictor-core/      no_std-capable. Frozen types: chunk IR, envelope, scores/calibration table, verdict, state, reason, ack. No logic beyond ctor/validate/compile.
|   |-- lictor-canon/     RFC 8785 JCS over the float-free profile; F64Hex / F64Array; floatify; sha256.
|   |-- lictor-detect/    no_std-capable. tier0 (limits + projection), brake (feasibility + brake/hold actions), window, tier1 (features), conformal.
|   |-- lictor-fuse/      no_std-capable. decide(), FuseRt, fsm::next, Tally. Zero-alloc hot path.
|   |-- lictor-receipt/   std. Tick/timing chains, ReceiptBody, sign/verify (Ed25519), ledger, curve receipt, handoff/ack, history, keys.
|   |-- lictor-runtime/   std. Wire schema + codec, Session (serve/replay/bench share it), episode writer, latency histogram, trace files.
|   |-- lictor-calib/     std, offline. Trace loading, split-CP calibration, envelope fit, metrics, Layer-A sweep, Layer-B curve.
|   +-- lictor-cli/       the `lictor` binary (clap).
|-- adapters/   lictor_client.py  verify_receipt.py  lerobot/lictor_lerobot.py (ProcessorStep, roadmap)  openpi/lictor_proxy.py (websocket proxy, roadmap)
|-- harness/    compat.py seeds.py pusht_rollout.py executor.py inject.py policy_replay.py arms.py run.py microbench.py analyze.py stats.py figures.py env.sh requirements.txt tests/
|-- envelopes/  pusht.base.toml  pusht.toml  pusht.oracle.toml  (the latter two are written by `lictor envelope fit`; committed by WP-13)
|-- bench/      run_all.sh  latency/  determinism/  tamper/  workspace-breach/  brake/  failure-prediction/  fixtures/
|-- docs/       ANALYSIS.md ARCHITECTURE.md IMPLEMENTATION_PLAN.md wire-protocol.md receipt-schema.md verdict-schema.md envelope-schema.md calibration.md
|               EXPERIMENT.md REPRODUCE.md envelope_fit_report.md banner.svg figures/
|-- scripts/    demo.sh ci_python.sh
|-- research/   README.md (+ memos behind every cited number)
+-- .github/workflows/ci.yml
```

Crate dependency graph (acyclic):

```
lictor-core <- lictor-detect <- lictor-fuse
lictor-core(std) -> lictor-canon (digest_hex)            lictor-canon <- lictor-receipt <- lictor-runtime <- lictor-calib <- lictor-cli
lictor-runtime -> {core, detect, fuse, canon, receipt}   (it does NOT depend on lictor-calib: the CLI loads calibration.json and hands
                                                          the runtime a compiled `CalibrationLoaded`)
lictor-calib   -> {core, detect, fuse, canon, receipt, runtime}   (wire + trace types come from runtime: ONE wire parser)
dev-only: lictor-fuse [dev-dependencies] lictor-runtime  (Staging + TraceReader for tests/alloc.rs, tests/determinism.rs; a dev-dep cycle is legal)
```

Workspace Cargo.toml (WP-0 writes it verbatim):

```toml
[workspace]
resolver = "2"
members = ["crates/lictor-core","crates/lictor-canon","crates/lictor-detect","crates/lictor-fuse",
           "crates/lictor-receipt","crates/lictor-runtime","crates/lictor-calib","crates/lictor-cli"]

[workspace.package]
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/RARS-oss/lictor"
description = "lictor: a deterministic safety fuse over learned robot policies -- failure prediction, geometric limit enforcement, human escalation, signed replayable receipts"

[workspace.dependencies]
lictor-core    = { path = "crates/lictor-core",    version = "0.1.0" }
lictor-canon   = { path = "crates/lictor-canon",   version = "0.1.0" }
lictor-detect  = { path = "crates/lictor-detect",  version = "0.1.0" }
lictor-fuse    = { path = "crates/lictor-fuse",    version = "0.1.0" }
lictor-receipt = { path = "crates/lictor-receipt", version = "0.1.0" }
lictor-runtime = { path = "crates/lictor-runtime", version = "0.1.0" }
lictor-calib   = { path = "crates/lictor-calib",   version = "0.1.0" }
serde       = { version = "1", default-features = false, features = ["derive", "alloc"] }
serde_json  = "1"
toml        = "0.8"
sha2        = "0.10"
hex         = "0.4"
base64      = "0.22"
ed25519-dalek = "2"          # keygen goes through getrandom; the rand_core feature is not used
getrandom   = "0.2"
libm        = "0.2"          # unconditional dependency of lictor-core (no feature); std uses the hardware sqrt
clap        = { version = "4", features = ["derive"] }
anyhow      = "1"
thiserror   = "2"
hdrhistogram = "7"
tempfile    = "3"            # dev-dependency of receipt, runtime, calib, cli (pre-declared: only WP-0 edits Cargo.toml)

[profile.release]
opt-level = 3
lto = "fat"
codegen-units = 1
panic = "abort"

[profile.bench-debug]     # release-vs-debug determinism cross-check target
inherits = "dev"
opt-level = 0
```

Build discipline: `CARGO_TARGET_DIR` is NEVER hardcoded; `.cargo/config.toml` sets only `[build] rustflags = ["-D", "warnings"]`; every documented command is run with `CARGO_TARGET_DIR=/mnt/d/lictor/target` (C: is full; `harness/env.sh` exports it). `rust-toolchain.toml`: `channel = "stable"`, components `rustfmt, clippy`. `Cargo.lock` is committed (WP-0). `crates/lictor-cli/build.rs` emits `cargo:rustc-env=LICTOR_GIT=<short sha|nogit>`, `cargo:rustc-env=LICTOR_BUILD_UTC=...` and `cargo:rerun-if-changed=.git/HEAD` so `env!("LICTOR_GIT")` never fails on a fresh checkout without git. no_std check: `cargo check -p lictor-core -p lictor-detect -p lictor-fuse --no-default-features` (detect/fuse depend on core with `default-features = false` and forward `std`, so no_std is actually exercised).

---

## 4. Frozen shared types

The complete public surface every crate codes against. Bodies are the owning work package's; signatures are frozen.

## FROZEN RUST TYPES (normative; verbatim in the code)

Rules that apply to every definition below:

- Edition 2021, `#![forbid(unsafe_code)]` in every library crate. The ONLY `unsafe` in the workspace is the counting allocator: `crates/lictor-cli/src/main.rs` carries `#![deny(unsafe_code)]` plus `#[allow(unsafe_code)] mod alloc_count;` (a `GlobalAlloc` wrapper over `System` with an `AtomicBool` gate, always installed, counting only while the gate is set), and `crates/lictor-fuse/tests/alloc.rs` (an integration-test crate root, outside the library's `forbid`) carries `#![allow(unsafe_code)]` with a header comment. Nothing else.
- Clippy policy: `cargo clippy --workspace --all-targets -- -D warnings` must be green on the WP-0 `todo!()` stubs. Therefore every frozen `pub fn new() -> Self` has a frozen `impl Default` beside it (`ChunkBuf`, `Trail`, `Tier1Rt`, `FuseRt`, `Hist`, `Staging`) and every frozen `len()` has a frozen `is_empty()` beside it (`Trail`, `F64Array`). No `#[allow(clippy::...)]` on frozen items and no `[workspace.lints]` table.
- `lictor-core`, `lictor-detect`, `lictor-fuse` are `#![cfg_attr(not(feature = "std"), no_std)]`. They contain no I/O, no crypto, no clock, no allocation on the decision path.
- Float discipline in `lictor-core` / `lictor-detect` / `lictor-fuse` (the decision path): `f64` only; the ONLY operations are `+ - * /`, `sqrt`, `min`, `max`, `abs`, comparisons. All are correctly rounded by IEEE-754, so verdicts are bit-identical on the tested target (x86_64 SSE2, release and debug) and are expected to be on any target where Rust `f64` is strict binary64 with no extended-precision intermediates (i586/x87 excluded) and no FMA contraction; untested elsewhere. FORBIDDEN: `f32` in the decision path, `mul_add`, `powi`, `powf`, `exp`, `ln`, `sin`, `cos`, `atan2`, `hypot`, `HashMap` iteration, `rayon`, SIMD reductions, `Instant`/`SystemTime`, RNG, `f64::min`/`f64::max`/`f64::clamp` (NaN semantics differ from `fmath`), and `as` casts FROM `f64`. Norms use `sqrt(sum x*x)` with fixed left-to-right accumulation (index ascending). `min`/`max`/`abs`/`sqrt` go through `lictor_core::fmath` so std and no_std are identical. SCOPE: this deny list applies to `lictor-core`, `lictor-detect`, `lictor-fuse` only; `lictor-calib` and `lictor-receipt` are off the decision path and MAY use `ln`/`exp`/`lgamma`-style functions (Clopper-Pearson, McNemar, bootstrap) and `f64::total_cmp` sorting.
- Any NaN/Inf reaching `decide()` -> `TripMask::NONFINITE` -> `FuseState::Fault` -> hold action (fail-closed). The fuse never panics on data; `debug_assert!` only.
- Every serde enum on a wire/file surface is `#[serde(rename_all = "snake_case")]`.
- Signatures below are frozen. Bodies are the owning WP's. Private fields may be added; public surface may not change without a WP-0 amendment.

### crates/lictor-core/src/lib.rs (WP-0 owns; declarations + re-exports only)

```rust
#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
//! lictor-core: frozen types for the deterministic safety fuse. No logic beyond
//! constructors, validation and trivially-derived accessors. See docs/ARCHITECTURE.md.
//! DECISION-PATH DENY LIST (also enforced by review): f32, mul_add, powi, powf, exp, ln,
//! sin, cos, atan2, hypot, HashMap iteration, rayon, SIMD reductions, Instant, SystemTime, RNG.
#[cfg(feature = "std")] extern crate std;
extern crate alloc;

pub mod fmath;
pub mod chunk;
pub mod envelope;
pub mod scores;
pub mod verdict;
pub mod state;
pub mod reason;
pub mod ack;

pub use ack::{AckDecision, VerifiedAck};
pub use chunk::{ActionKind, ChunkBuf, ChunkError, ChunkView, ObsView, MAX_AUX, MAX_D, MAX_EXT, MAX_H, MAX_POS};
pub use envelope::{BrakeKind, BrakeModel, ClampMode, ContactLimit, EmbodimentManifest, EnvelopeError,
                   FitRecord, FuseConfig, GateSpec, Hysteresis, RearmPolicy, SafetyEnvelope, MAX_OPERATORS, MAX_TERMS};
pub use reason::{reason_text, ReasonCode};
pub use scores::{bin_of, CalMethod, CalibrationC, Feat, Scores, NFEAT, T_GRID};
pub use state::{EpisodeInit, ExecMode, FuseMode, FuseState};
pub use verdict::{ActionSource, SafetyVerdict, Status, TripMask};

pub const LICTOR_VERSION: &str = env!("CARGO_PKG_VERSION");
```

### crates/lictor-core/src/fmath.rs

```rust
//! The only float helpers allowed on the decision path. Identical results under std and no_std.
//! `libm` is an UNCONDITIONAL dependency of lictor-core (there is no `libm` feature): under `std` the hardware sqrt is
//! used, under no_std `libm::sqrt`; both are correctly rounded, hence bit-identical.
#[inline] pub fn sqrt(x: f64) -> f64 { #[cfg(feature = "std")] { x.sqrt() } #[cfg(not(feature = "std"))] { libm::sqrt(x) } }
#[inline] pub fn abs(x: f64) -> f64 { f64::from_bits(x.to_bits() & !(1u64 << 63)) }
/// `if a < b { a } else { b }` -- deliberately NOT f64::min (NaN never reaches here; fail-closed earlier).
#[inline] pub fn min(a: f64, b: f64) -> f64 { if a < b { a } else { b } }
#[inline] pub fn max(a: f64, b: f64) -> f64 { if a > b { a } else { b } }
#[inline] pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 { max(lo, min(x, hi)) }
/// Euclidean norm with fixed left-to-right accumulation over `v[..n]`.
#[inline] pub fn norm(v: &[f64], n: usize) -> f64 { let mut s = 0.0; let mut i = 0; while i < n { s = s + v[i] * v[i]; i += 1; } sqrt(s) }
/// ||a[..n] - b[..n]|| with fixed order.
#[inline] pub fn dist(a: &[f64], b: &[f64], n: usize) -> f64 { let mut s = 0.0; let mut i = 0; while i < n { let d = a[i] - b[i]; s = s + d * d; i += 1; } sqrt(s) }
#[inline] pub fn all_finite(v: &[f64]) -> bool { v.iter().all(|x| x.is_finite()) }
```

### crates/lictor-core/src/chunk.rs -- the action-chunk IR

```rust
use serde::{Deserialize, Serialize};

pub const MAX_H: usize = 64;    // pi0 chunks are 50 -> headroom
pub const MAX_D: usize = 32;    // pi0 padded action_dim is 32 -> headroom
pub const MAX_POS: usize = 32;  // proprioceptive position dim
pub const MAX_AUX: usize = 16;  // embodiment-declared auxiliary scalars (privileged sim state)
pub const MAX_EXT: usize = 4;   // Tier-2 external scalars

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind { EePosition, EeDelta, JointPosition, JointVelocity, Other }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkError { TooLong, TooWide, LenMismatch, ExecExceedsHorizon, Empty }

/// Pre-allocated, owned chunk storage (16.5 KB). Lives inside `FuseRt`; never allocated on the hot path.
#[derive(Clone)]
pub struct ChunkBuf {
    data: [f64; MAX_H * MAX_D],
    seq: u32, t_emit: u32, horizon: u16, dim: u16, exec_steps: u16, filled: bool,
}
impl ChunkBuf {
    pub const fn new() -> Self;
    /// Row-major copy of `src` (len == h*d). Non-finite values are accepted here and caught by `decide()`.
    pub fn fill(&mut self, seq: u32, t_emit: u32, h: u16, d: u16, exec: u16, src: &[f64]) -> Result<(), ChunkError>;
    pub fn copy_from(&mut self, v: ChunkView<'_>);
    pub fn view(&self) -> Option<ChunkView<'_>>;          // None when !filled
    pub fn row_mut(&mut self, i: usize) -> &mut [f64];     // len == dim; i clamped to horizon-1
    pub fn clear(&mut self);
    pub fn is_filled(&self) -> bool;
}
impl Default for ChunkBuf { fn default() -> Self { Self::new() } }

/// Borrowed, Copy view. THE chunk IR passed to detectors.
#[derive(Clone, Copy, Debug)]
pub struct ChunkView<'a> {
    pub seq: u32,         // monotone chunk index within the episode, from 0
    pub t_emit: u32,      // absolute env step at which index 0 applies
    pub horizon: u16,     // H (PushT: 15)
    pub dim: u16,         // d (PushT: 2)
    pub exec_steps: u16,  // S -- the irrevocably committed prefix (PushT: 8)
    pub data: &'a [f64],  // len == horizon*dim, row-major: data[i*dim + c]
}
impl<'a> ChunkView<'a> {
    /// Row `i`, clamped to `horizon-1` (never panics in release; debug_assert in debug).
    #[inline] pub fn action(&self, i: usize) -> &'a [f64];
    #[inline] pub fn overlap_len(&self) -> usize { (self.horizon - self.exec_steps) as usize }
    pub fn all_finite(&self) -> bool;
}

/// Everything the fuse is told about the world this tick.
#[derive(Clone, Copy, Debug)]
pub struct ObsView<'a> {
    pub t: u32,                 // absolute env step
    pub pos: &'a [f64],         // proprioceptive position, len == manifest.pos_dim (mandatory)
    pub vel: Option<&'a [f64]>, // proprioceptive velocity when manifest.provides_vel (PushT: Some -- gym_pusht info["vel_agent"] every step); None only for embodiments that provide none
    pub aux: &'a [f64],         // layout declared by manifest.aux_layout (PushT: block_x, block_y, block_theta, coverage)
    pub ext: &'a [f64],         // Tier-2 scalars ext0..ext3 from the policy process; len <= MAX_EXT; may be empty
}
```

### crates/lictor-core/src/envelope.rs -- the safety configuration

```rust
use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};
use crate::{chunk::{ActionKind, MAX_D, MAX_POS}, scores::CalibrationC, state::FuseMode};

pub const MAX_TERMS: usize = 8;
pub const MAX_OPERATORS: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbodimentManifest {
    pub id: String,                 // "gym_pusht/PushT-v0"
    pub units: String,              // "px" | "rad" | "m" -- documentation, bound into the digest
    pub action_kind: ActionKind,    // ee_position
    pub action_dim: u16,            // 2
    pub pos_dim: u16,               // 2
    pub horizon: u16,               // 15   (the chunk length the adapter DELIVERS)
    pub exec_steps: u16,            // 8    (the committed prefix)
    pub control_hz_num: u32,        // 10   control period dt = den/num seconds, exact rational
    pub control_hz_den: u32,        // 1
    pub provides_vel: bool,         // true (PushT): the harness sends the simulator velocity as obs.vel; false -> finite-difference v_hat (recorded as a fuse_note)
    pub horizon_ticks: u32,         // 300  episode time base for calibration binning; MUST equal the env's max_episode_steps (checked by `lictor calibrate`)
    pub norm_center: Vec<f64>,      // len == action_dim; Tier-1 uses abar = (a - center)/scale
    pub norm_scale: Vec<f64>,       // len == action_dim; all > 0
    pub aux_layout: Vec<String>,    // ["block_x","block_y","block_theta","coverage"]
    pub ext_names: Vec<String>,     // <= 4 names for ext0..ext3; [] when unused
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrakeKind {
    /// Exact forward rollout of the env's 2nd-order PD agent (gym-pusht: k_p=100, k_v=20, dt=0.01, 10 substeps).
    PdSecondOrder,
    /// Closed form d_stop = ||v|| / k_v (setpoint-held first-order decay).
    FirstOrderDecay,
    /// Closed form d_stop = ||v||^2 / (2 a_max).
    BoundedAccel,
    /// Closed form d_stop = ||v||^2/(2 a_max) + ||v|| a_max/(2 j_max) (Ruckig two-phase upper bound).
    JerkLimited,
    /// Velocity-command embodiments: brake = ramp to zero; feasibility uses BoundedAccel.
    ZeroVelocityHold,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrakeModel {
    pub kind: BrakeKind,
    pub k_p: f64,          // 100.0 (PushT)
    pub k_v: f64,          // 20.0
    pub substeps: u16,     // 10 -- must equal the env's inner loop
    pub dt: f64,           // 0.01 -- physics substep, NOT the control period
    pub commit_steps: u16, // 8 -- irrevocable prefix simulated forward
    pub brake_steps: u16,  // 8 -- extra hold steps simulated after the prefix (PdSecondOrder only)
    pub react_ticks: u16,  // 1 -- dead time before a brake bites (closed-form kinds only)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContactLimit { pub radius: f64, pub v_max: f64, pub aux_center: [u8; 2] } // aux indices of the object centre

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClampMode { Off, Project }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RearmPolicy { Auto, AckOnly }

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Hysteresis {
    pub k: u8,                          // 3   K-of-N on the predictive channel
    pub n: u8,                          // 5   1 <= K <= N <= 63 (window_mask = (1 << n) - 1 must fit in u64)
    pub warn_margin: f64,               // 0.5 (z units): s > tau - warn_margin -> Watching
    pub clear_ticks: u8,                // 5   clean ticks to leave Watching/Clamped
    pub clamp_streak_to_brake: u8,      // 3   consecutive clamped CHUNKS -> Braking
    pub max_clamps_per_episode: u16,    // 60
    pub stop_confirm_ticks: u8,         // 2   Braking -> Held once ||v|| <= v_stop_eps this many ticks
    pub v_stop_eps: f64,                // 5.0 units/s
    pub brake_timeout_ticks: u16,       // 30  Braking that never stops -> Fault
    pub rearm: RearmPolicy,             // auto (PushT default) | ack_only
    pub rearm_hold: u16,                // 10  clean ticks in Held before auto re-arm
    pub max_rearms: u8,                 // 2   auto re-arms per episode
    pub escalate_after_hold_ticks: u16, // 30  Held this long -> Escalated (handoff)
    pub handoff_timeout_ticks: u32,     // 200 Escalated with no ack -> Terminated
    pub watchdog_ticks: u8,             // 2   missed_ticks above this -> Fault
}

/// Fixed-order DNF over Tier-1 features: s = max_i min_{j in terms[i]} z_j. A singleton term is a plain channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GateSpec { pub n_terms: u8, pub terms: [u32; MAX_TERMS] }
impl GateSpec {
    pub const DISARMED: Self = Self { n_terms: 0, terms: [0; MAX_TERMS] };
    /// Parse ["tce","acc|path_ineff"] -> terms (each '|' = AND within a term). Unknown name -> Err.
    pub fn parse(terms: &[String]) -> Result<Self, EnvelopeError>;
    pub fn mask(&self) -> u32;   // OR of all terms
    pub fn names(&self) -> Vec<String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FitRecord { pub source_run: String, pub quantile: f64, pub slack: f64, pub n_episodes: u32, pub fitted_utc: String, pub note: String }

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SafetyEnvelope {
    pub schema: String,             // "lictor-envelope/v1"
    pub envelope_id: String,        // "pusht-base-v1"
    pub embodiment: EmbodimentManifest,
    pub box_lo: Vec<f64>,           // len == pos_dim, env units
    pub box_hi: Vec<f64>,
    pub margin: f64,                // subtracted from the box before EVERY check
    pub v_max: f64,                 // units/s
    pub a_max: f64,                 // units/s^2
    pub j_max: f64,                 // units/s^3
    pub reach_max: f64,             // units: ||a_0 - p_t||
    pub contact: Option<ContactLimit>,
    pub brake: BrakeModel,
    pub clamp_mode: ClampMode,
    pub hysteresis: Hysteresis,
    pub tier0_enabled: Vec<String>, // subset of ["workspace","speed","accel","jerk","reach","contact","brake"]; nonfinite/schema/watchdog always on
    pub gate: Vec<String>,          // default gate terms, overridable by calibration.json
    pub operators: Vec<String>,     // Ed25519 pubkeys (hex64) allowed to sign an AckToken; slot = index
    pub fit: Option<FitRecord>,     // present when produced by `lictor envelope fit`
    pub fail_closed: bool,          // MUST be true in v1; recorded so a future `false` is auditable
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvelopeError { Parse(String), Invalid(String), Unsupported(String) }
impl SafetyEnvelope {
    #[cfg(feature = "std")] pub fn from_toml(s: &str) -> Result<Self, EnvelopeError>;
    #[cfg(feature = "std")] pub fn to_toml(&self) -> Result<String, EnvelopeError>;
    /// sha256 over lictor_canon::canon(floatify(to_value(self))). std only.
    #[cfg(feature = "std")] pub fn digest_hex(&self) -> String;
    /// sha256 over canon(floatify(to_value(&self.embodiment))) -- the sub-digest a calibration binds to. Tier-1 features
    /// depend on the EmbodimentManifest ONLY (never on v_max/a_max/j_max/reach_max/operators), so `envelope fit` and an
    /// operator-list change do not invalidate a calibration; a manifest change does. std only.
    #[cfg(feature = "std")] pub fn embodiment_digest(&self) -> String;
    pub fn validate(&self) -> Result<(), EnvelopeError>;   // dims, positivity, 1 <= k <= n <= 63, <= MAX_OPERATORS, horizon_ticks > 0, fail_closed == true
    pub fn dt(&self) -> f64 { self.embodiment.control_hz_den as f64 / self.embodiment.control_hz_num as f64 }
    /// One-shot, off-loop: validate + flatten Vec -> fixed arrays + precompute derived constants.
    pub fn compile(&self, mode: FuseMode, calib: Option<CalibrationC>) -> Result<FuseConfig, EnvelopeError>;
    pub fn tier0_mask(&self) -> Result<u32, EnvelopeError>;
}

/// The immutable, hot-path-ready configuration. No Vec, no String; every derived constant baked.
/// `envelope_digest` / `embodiment_digest` are filled by `compile` under `std` (via `digest_hex` / `embodiment_digest`);
/// under no_std they are `[0; 32]` (the MCU path carries digests out-of-band -- roadmap 9).
#[derive(Clone, Debug)]
pub struct FuseConfig {
    pub envelope_digest: [u8; 32],
    pub embodiment_digest: [u8; 32],
    pub calib_digest: Option<[u8; 32]>,
    pub mode: FuseMode,
    pub kind: ActionKind,
    pub dim: usize, pub pos_dim: usize, pub horizon: usize, pub exec: usize, pub overlap: usize,
    pub dt: f64, pub inv_dt: f64, pub inv_dt2: f64, pub inv_dt3: f64,
    pub norm_center: [f64; MAX_D], pub inv_norm_scale: [f64; MAX_D], pub norm_scale_iso: f64,
    pub box_lo: [f64; MAX_POS], pub box_hi: [f64; MAX_POS],   // ALREADY margin-adjusted
    pub v_max: f64, pub a_max: f64, pub j_max: f64, pub reach_max: f64,
    pub step_max: f64,          // v_max * dt -- the per-step leash
    pub contact: Option<ContactLimit>,
    pub brake: BrakeModel,
    pub clamp: ClampMode,
    pub hyst: Hysteresis,
    pub window_mask: u64,       // (1 << n) - 1, n <= 63
    pub tier0_enabled: u32,     // TripMask bits armed
    pub n_operators: u8,
    pub calib: CalibrationC,    // CalibrationC::DISARMED when Tier 1 is off
}
```

### crates/lictor-core/src/scores.rs -- features and the runtime calibration table

```rust
use serde::{Deserialize, Serialize};
use crate::envelope::GateSpec;

pub const NFEAT: usize = 12;
pub const T_GRID: usize = 100;

/// Frozen feature ids. EVERY channel is oriented LARGER = MORE ANOMALOUS (acm is stored negated).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[repr(usize)]
pub enum Feat { Tce = 0, Acc = 1, AcmNeg = 2, Njr = 3, Reach = 4, PathIneff = 5, Stall = 6, SpeedPeak = 7, Ext0 = 8, Ext1 = 9, Ext2 = 10, Ext3 = 11 }
impl Feat {
    pub const NAMES: [&'static str; NFEAT] = ["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak","ext0","ext1","ext2","ext3"];
    pub const CHUNK_BOUNDARY_MASK: u32 = 0b0000_1001_1111; // tce acc acm_neg njr reach speed_peak: updated at a chunk boundary, held between
    pub const PER_TICK_MASK: u32      = 0b1111_0110_0000; // path_ineff stall ext0..3: updated every tick
    pub fn from_name(s: &str) -> Option<Feat>;
    #[inline] pub fn bit(self) -> u32 { 1u32 << (self as u32) }
    pub fn name(self) -> &'static str;
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scores {
    pub f: [f64; NFEAT],   // raw, dimensionless (0.0 where !valid)
    pub z: [f64; NFEAT],   // standardised (0.0 where !valid or unmasked)
    pub s: f64,            // aggregate nonconformity = max over gate terms of min over the term; NEG_INFINITY when nothing valid
    pub valid: u32,        // bit j set <=> feature j computable this tick
    pub fired: u32,        // bit j set <=> z_j > tau (informational; the decision is on s)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalMethod { Static, Binned }   // Static == Binned with t_grid = 1

/// Hot-path calibration artefact: fixed arrays (~19 KB), no allocation, no I/O. Built once from calibration.json.
#[derive(Clone, Copy, Debug)]
pub struct CalibrationC {
    pub method: CalMethod,
    pub alpha_num: u32, pub alpha_den: u32,
    pub n_calib: u32,            // the number of per-episode scores tau was taken over (== calibration.json n_calib; 2-way split: n_fit)
    pub horizon_ticks: u32,      // the time base used for binning (PushT: 300; == manifest.horizon_ticks)
    pub t_grid: u16,             // 1 (static) or T_GRID (binned)
    pub center: [[f64; NFEAT]; T_GRID],   // robust median per (bin, feature)
    pub scale:  [[f64; NFEAT]; T_GRID],   // 1.4826*MAD per (bin, feature), floored at 1e-9
    pub mask: u32,               // features entering s (== gate.mask())
    pub gate: GateSpec,
    pub tau: f64,                // split-CP quantile on per-episode max s; +INFINITY = never fires
    pub digest: [u8; 32],
}
impl CalibrationC {
    pub const DISARMED: Self;    // mask 0, gate DISARMED, tau +INFINITY, digest zero
    #[inline] pub fn bin(&self, t: u32) -> usize { bin_of(t, self.horizon_ticks, self.t_grid) }
    pub fn armed(&self) -> bool { self.mask != 0 }
}
/// INTEGER arithmetic only; shared by runtime and offline calibrator. A mismatch is a determinism trap.
#[inline] pub fn bin_of(t: u32, horizon_ticks: u32, t_grid: u16) -> usize {
    let g = t_grid as u64; let h = if horizon_ticks == 0 { 1 } else { horizon_ticks as u64 };
    let b = (t as u64) * g / h; if b >= g { (g - 1) as usize } else { b as usize }
}
```

### crates/lictor-core/src/verdict.rs

```rust
use serde::{Deserialize, Serialize};
use crate::{chunk::MAX_D, reason::ReasonCode, scores::Scores, state::FuseState};

/// Severity lattice (sbx feedback lineage). Ord derives from declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status { Nominal, Watching, Clamped, Braking, Held, Escalated, Fault, Terminated }

pub struct TripMask;
impl TripMask {
    pub const NONE: u32            = 0;
    pub const WORKSPACE: u32       = 1 << 0;
    pub const SPEED: u32           = 1 << 1;
    pub const ACCEL: u32           = 1 << 2;
    pub const JERK: u32            = 1 << 3;
    pub const REACH: u32           = 1 << 4;
    pub const CONTACT: u32         = 1 << 5;
    pub const BRAKE: u32           = 1 << 6;   // braking infeasible
    pub const NONFINITE: u32       = 1 << 7;
    pub const SCHEMA: u32          = 1 << 8;
    pub const WATCHDOG: u32        = 1 << 9;
    pub const TIER1_CP: u32        = 1 << 10;  // K-of-N predictive fire
    pub const CLAMP_BUDGET: u32    = 1 << 11;
    pub const HANDOFF_TIMEOUT: u32 = 1 << 12;
    pub const OPERATOR_ABORT: u32  = 1 << 13;
    pub const BRAKE_TIMEOUT: u32   = 1 << 14;
    pub const REARM_BUDGET: u32    = 1 << 15;
    pub const TIER0_SOFT: u32 = Self::WORKSPACE | Self::SPEED | Self::ACCEL | Self::JERK | Self::REACH | Self::CONTACT;
    pub const TIER0_HARD: u32 = Self::BRAKE | Self::NONFINITE | Self::SCHEMA | Self::WATCHDOG;
    pub const NAMES: [&'static str; 16] = ["workspace","speed","accel","jerk","reach","contact","brake","nonfinite",
        "schema","watchdog","tier1_cp","clamp_budget","handoff_timeout","operator_abort","brake_timeout","rearm_budget"];
    pub fn names(m: u32) -> impl Iterator<Item = &'static str>;
    pub fn from_name(s: &str) -> Option<u32>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionSource { Policy, Clamped, Brake, Hold }

/// Copy, no String, no allocation. ~600 bytes.
#[derive(Clone, Copy, Debug)]
pub struct SafetyVerdict {
    pub seq: u32,                  // tick sequence within the episode, from 0
    pub t: u32,                    // absolute env step
    pub status: Status,
    pub state: FuseState,          // state AFTER this tick
    pub prev_state: FuseState,
    pub trips: u32,                // TripMask bits raised THIS tick
    pub action: [f64; MAX_D],      // the action the executor MUST apply (only [..action_dim] meaningful)
    pub action_dim: u16,
    pub action_src: ActionSource,
    pub substituted: bool,         // action != raw policy action (always false in Observe mode)
    pub clamped_dims: u32,         // dims moved by the Tier-0 projection
    pub scores: Scores,
    pub tau: f64,
    pub window: u64,               // K-of-N ring AFTER this tick
    pub window_hits: u8,           // popcount(window)
    pub brake_margin: f64,         // units of slack in the braking check; < 0 == infeasible
    pub reason: ReasonCode,
    pub handoff_seq: Option<u32>,  // Some on the tick Escalated is entered
    pub ack_consumed: bool,        // a VerifiedAck was applied this tick
    pub violation_reached_env: bool, // Observe mode: the would-be action_src was != Policy
}
```

### crates/lictor-core/src/state.rs

```rust
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuseState { #[default] Idle, Armed, Watching, Clamped, Braking, Held, Escalated, Fault, Terminated }
impl FuseState { pub fn is_stop(self) -> bool { matches!(self, Self::Braking | Self::Held | Self::Escalated | Self::Fault | Self::Terminated) }
                 pub fn is_terminal(self) -> bool { matches!(self, Self::Fault | Self::Terminated) } }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuseMode { Observe, Enforce }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecMode { Sync, Async }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpisodeInit { pub episode_index: u32, pub seed: u64, pub delay_steps: u16, pub exec: ExecMode }
```

### crates/lictor-core/src/reason.rs -- the human-escalation glossary (sbx `hint_for` lineage)

```rust
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    Ok, ObserveOnly, WatchBand,
    ClampWorkspace, ClampSpeed, ClampAccel, ClampJerk, ClampReach, ClampContact,
    BrakeInfeasible, BrakeTier1Cp, BrakeClampBudget, Stopping, HeldStandstill,
    EscalateHoldTimeout, EscalateRearmBudget, EscalateTier1Persistent,
    FaultNonFinite, FaultSchema, FaultWatchdog, FaultBrakeTimeout, FaultInternal,
    RearmedAuto, RearmedAck, TerminatedAbort, TerminatedTimeout, TerminatedRetune, EpisodeEnd,
}
/// One sentence per code, <= 120 chars, plain English, explaining WHY the fuse acted.
pub fn reason_text(r: ReasonCode) -> &'static str;
```

### crates/lictor-core/src/ack.rs -- crypto-free ack types (verification lives in lictor-receipt)

```rust
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AckDecision { Resume, Abort, Retune }
/// An AckToken whose signature, operator membership and handoff digest were ALREADY verified by the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifiedAck { pub decision: AckDecision, pub operator_slot: u8, pub nonce: u64, pub handoff_seq: u32 }
```

### crates/lictor-detect (lib.rs: `pub mod tier0; pub mod brake; pub mod window; pub mod tier1; pub mod conformal;`)

```rust
// tier0.rs
use lictor_core::*;
#[derive(Clone, Copy, Debug, Default)]
pub struct Tier0Out { pub trips: u32, pub clamped_dims: u32, pub peak_speed: f64 }
/// Whole-chunk check at a chunk boundary. Writes the projected chunk into `out` (== ch when nothing clamped).
/// Exactly `horizon` iterations. Zero allocation.
pub fn check_chunk(cfg: &FuseConfig, obs: ObsView<'_>, ch: ChunkView<'_>, out: &mut ChunkBuf) -> Tier0Out;
/// Intra-chunk per-tick check of the single committed action `a` against predecessor `prev` (box, speed, reach, contact).
/// Writes the (possibly leashed) action into `out[..dim]`.
pub fn check_action(cfg: &FuseConfig, obs: ObsView<'_>, prev: &[f64], a: &[f64], out: &mut [f64]) -> Tier0Out;

// brake.rs
#[derive(Clone, Copy, Debug, Default)]
pub struct BrakeOut { pub feasible: bool, pub margin: f64, pub stop_dist: f64 }
/// Braking feasibility of the committed prefix of `ch` starting at row `from`, from plant state (p0, v0).
/// `from` = the row executed THIS tick (`idx`): at a chunk boundary `from == idx` (0 for sync/freeze delivery, `d` for async drop).
/// `v0` = obs.vel when the manifest provides it (PushT), else the finite-difference v_hat.
/// PdSecondOrder: EXACTLY (commit_steps + brake_steps) * substeps iterations of the PD map.
/// Closed-form kinds: EXACTLY commit_steps iterations of Euler + stopping ball. No allocation.
pub fn brake_feasible(cfg: &FuseConfig, p0: &[f64], v0: &[f64], ch: ChunkView<'_>, from: usize) -> BrakeOut;
/// The Braking-phase action: setpoint := clamp_box(p) (position kinds) or ramp-to-zero (velocity kinds).
pub fn brake_action(cfg: &FuseConfig, p: &[f64], v: &[f64], out: &mut [f64]);
/// The Held action: the latched hold setpoint (already clamped) copied into out.
pub fn hold_action(cfg: &FuseConfig, p_latch: &[f64], out: &mut [f64]);

// window.rs
#[derive(Clone, Copy, Debug, Default)]
pub struct Window { pub bits: u64 }
impl Window { #[inline] pub fn push(&mut self, hit: bool, mask: u64); #[inline] pub fn hits(&self) -> u8; pub fn clear(&mut self); }
pub const TRAIL_W: usize = 32;
#[derive(Clone, Copy)]
pub struct Trail { buf: [[f64; MAX_POS]; TRAIL_W], head: u8, len: u8 }
impl Trail { pub const fn new() -> Self; pub fn clear(&mut self); pub fn push(&mut self, p: &[f64], d: usize);
             pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool; pub fn net(&self, d: usize, w: usize) -> f64;  /* ||p_t - p_{t-w+1}|| */
             pub fn path(&self, d: usize, w: usize) -> f64; /* sum of consecutive distances over the last w */ }
impl Default for Trail { fn default() -> Self { Self::new() } }

// tier1.rs
#[derive(Clone)]
pub struct Tier1Rt { pub prev: ChunkBuf, pub have_prev: bool, pub trail: Trail, pub held: [f64; NFEAT], pub held_valid: u32 }
impl Tier1Rt { pub fn new() -> Self; pub fn reset(&mut self); }
impl Default for Tier1Rt { fn default() -> Self { Self::new() } }
pub const PE_WINDOW: usize = 20;   // ticks (2.0 s at 10 Hz)
/// stall reference speed v_ref = STALL_VREF_FRAC * norm_scale_iso / dt (PushT: 0.05 * 256 / 0.1 = 128 px/s). Depends on the
/// manifest only -- NEVER on v_max, which `envelope fit` changes after calibration.
pub const STALL_VREF_FRAC: f64 = 0.05;
/// Updates `rt` and writes raw features into `sc.f` / `sc.valid`. Chunk-boundary features recompute only when `chunk.is_some()`.
/// CONTRACT: `decide()` pushes `rt.trail` BEFORE calling this. `features` is the SOLE writer of `rt.prev`/`rt.have_prev`: after
/// computing tce/acc it copies the RAW incoming chunk into `rt.prev`, so TCE is always raw-vs-raw in Observe and Enforce alike
/// (calibration features == enforcement features). Every feature depends on the EmbodimentManifest only.
pub fn features(cfg: &FuseConfig, rt: &mut Tier1Rt, obs: ObsView<'_>, chunk: Option<ChunkView<'_>>, idx: u16, sc: &mut Scores);

// conformal.rs
/// z_j = (f_j - center[b][j]) / scale[b][j] for masked & valid j; b = cal.bin(t). Others 0.0.
pub fn standardise(cal: &CalibrationC, t: u32, sc: &mut Scores);
/// s = max over terms of min over the term's channels (all masked & valid), NEG_INFINITY if no term is fully valid.
/// Also sets sc.fired (z_j > tau per channel, informational). Returns s.
pub fn aggregate(cal: &CalibrationC, sc: &mut Scores) -> f64;
/// STRICT: s > tau.
#[inline] pub fn trip(cal: &CalibrationC, sc: &Scores) -> bool;
```

### crates/lictor-fuse (lib.rs: `pub mod fuse; pub mod fsm; pub mod tally; pub use fuse::{decide, Fuse, FuseRt, TickInput}; pub use fsm::{next, FsmInput}; pub use tally::Tally;`)

```rust
// fuse.rs
use lictor_core::*; use lictor_detect::{tier1::Tier1Rt, window::Window};
pub struct TickInput<'a> {
    pub obs: ObsView<'a>,
    pub chunk: Option<ChunkView<'a>>,   // Some ONLY on the tick a new chunk arrives
    pub idx: u16,                        // index into the CURRENT chunk for this tick
    pub missed_ticks: u8,                // the ONLY time-like input; an integer, supplied by the host
    pub ack: Option<VerifiedAck>,
    pub schema_fault: bool,             // set by the runtime on a wire/schema violation -> TripMask::SCHEMA in GUARD
}
/// Pre-allocated runtime state (~58 KB: three 16.5 KB ChunkBufs + an 8 KB Trail; `lictor bench` prints the exact size_of).
/// Constructed once per process; `reset()` per episode. Never allocates.
pub struct FuseRt {
    pub cur: ChunkBuf,                   // chunk currently executing (post-projection in Enforce; RAW in Observe, see WP-3)
    pub scratch: ChunkBuf,               // projection target
    pub t1: Tier1Rt,
    pub window: Window,
    pub state: FuseState,
    pub seq: u32,
    pub last_t: u32,                     // obs.t of the previous tick (time-continuity guard: t == last_t + 1 + missed_ticks; first tick t == 0)
    pub next_chunk_seq: u32,             // the seq the next delivered chunk MUST carry (0 after reset; chunk.seq continuity guard)
    pub pos: [f64; MAX_POS], pub prev_pos: [f64; MAX_POS], pub vel: [f64; MAX_POS], pub have_prev_pos: bool,
    pub last_cmd: [f64; MAX_D],          // last action actually emitted
    pub hold: [f64; MAX_POS],            // latched hold setpoint (Held/Escalated/Fault/Terminated)
    pub clean_run: u16, pub clamp_streak: u8, pub clamps: u16,
    pub brake_ticks: u16, pub stopped_ticks: u8, pub held_ticks: u16, pub held_clean: u16,
    pub escalated_ticks: u32, pub rearms: u8, pub handoff_seq: u32, pub handoff_pending: bool,
    pub chunk_ok_this_tick: bool,
    pub last_nonce: [u64; MAX_OPERATORS],
    pub tally: Tally,
}
impl FuseRt { pub fn new() -> Self; /// clears everything; state = Armed; tally = Tally::default() (max_s/max_z = NEG_INFINITY)
              pub fn reset(&mut self, init: EpisodeInit); }
impl Default for FuseRt { fn default() -> Self { Self::new() } }

/// THE CONTRACT. Pure: no allocation, no clock read, no lock, no syscall, no panic path, fixed iteration bounds.
/// Same (cfg, rt-before, inp) => bit-identical verdict and rt-after on any IEEE-754 platform.
#[inline(never)]
pub fn decide(cfg: &FuseConfig, rt: &mut FuseRt, inp: &TickInput<'_>) -> SafetyVerdict;

pub struct Fuse { cfg: FuseConfig, rt: FuseRt }
impl Fuse {
    pub fn new(cfg: FuseConfig) -> Self;
    pub fn cfg(&self) -> &FuseConfig;
    pub fn rt(&self) -> &FuseRt;
    pub fn reset(&mut self, init: EpisodeInit);
    #[inline] pub fn step(&mut self, inp: &TickInput<'_>) -> SafetyVerdict { decide(&self.cfg, &mut self.rt, inp) }
    pub fn finish(&self) -> Tally;
}

// fsm.rs
#[derive(Clone, Copy, Debug)]
pub struct FsmInput { pub trips: u32, pub predictive: bool, pub warn: bool, pub soft_clampable: bool,
                      pub stopped: bool, pub chunk_boundary: bool, pub chunk_ok: bool, pub ack: Option<VerifiedAck> }
/// Pure transition per the ARCHITECTURE table (rows evaluated top-down, first match wins). Updates counters in `rt`,
/// returns (new_state, extra_trips, reason).
pub fn next(cfg: &FuseConfig, rt: &mut FuseRt, inp: FsmInput) -> (FuseState, u32, ReasonCode);

// tally.rs -- Copy, fixed arrays; converted to receipt VerdictCounts by lictor-receipt
#[derive(Clone, Copy, Debug)]
pub struct Tally {
    pub ticks: u32, pub nominal: u32, pub watching: u32, pub clamped: u32, pub braking: u32, pub held: u32,
    pub escalated: u32, pub fault: u32, pub terminated: u32, pub substituted: u32,
    pub chunks_seen: u32, pub chunks_rejected: u32, pub clamps: u32, pub holds: u32, pub rearms: u32, pub escalations: u32,
    pub trips_by_bit: [u32; 16], pub fired_by_feat: [u32; NFEAT],
    pub first_trip_tick: Option<u32>, pub first_trip_reason: Option<ReasonCode>,
    pub first_stop_tick: Option<u32>, pub handoff_tick: Option<u32>,
    pub violations_reached_env: u32, pub max_s: f64, pub max_z: [f64; NFEAT], pub terminal_state: FuseState,
}
/// Hand-written (NOT derived): zeros/None everywhere, `terminal_state = Idle`, `max_s = f64::NEG_INFINITY`,
/// `max_z = [f64::NEG_INFINITY; NFEAT]` -- `s` is NEG_INFINITY while nothing is valid or Tier 1 is disarmed, and 0.0 would
/// bias the per-episode `max_s` that feeds ROC-AUC. F64Hex encodes +-inf exactly; the wire encodes it as `null`.
impl Default for Tally { fn default() -> Self; }
```

### crates/lictor-canon (lib.rs re-exports everything below)

```rust
use serde_json::Value;
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CanonError {
    #[error("float in canonical body at {0}")] FloatInBody(String),
    #[error("integer out of +-2^53-1 at {0}")] IntegerOutOfRange(String),
    #[error("key not [a-z0-9_]+: {0}")] BadKey(String),
    #[error("bad f64 encoding: {0}")] BadF64(String),
    #[error("serialize: {0}")] Serialize(String),
}
pub const CANONICAL_ID: &str = "jcs-floatfree/v1";
/// RFC 8785: keys sorted by UTF-16 code units, no whitespace, minimal escaping; integers only (|v| <= 2^53-1).
pub fn canon(v: &Value) -> Result<Vec<u8>, CanonError>;
pub fn check_keys(v: &Value) -> Result<(), CanonError>;
pub fn sha256_hex(bytes: &[u8]) -> String;
pub fn sha256_jcs(v: &Value) -> Result<String, CanonError>;
/// Replace every non-integer JSON number x by {"f64": f64_to_hex(x)} recursively (for TOML-sourced config trees).
pub fn floatify(v: Value) -> Value;
pub fn f64_to_hex(x: f64) -> String;            // 16 lowercase hex of x.to_bits() (big-endian nibbles)
pub fn f64_from_hex(s: &str) -> Result<f64, CanonError>;
/// Serializes as {"f64":"<16hex>"}; deserializes from the same.
#[derive(Clone, Copy, Debug, PartialEq)] pub struct F64Hex(pub f64);
/// Serializes as {"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[..]}.
#[derive(Clone, Debug, PartialEq)] pub struct F64Array { pub shape: Vec<u32>, pub data: Vec<f64> }
impl F64Array { pub fn from_slice(d: &[f64], shape: &[u32]) -> Self; pub fn len(&self) -> usize; pub fn is_empty(&self) -> bool; }
/// to_value + check_keys + reject bare floats -> canonical bytes.
pub fn canon_of<T: serde::Serialize>(t: &T) -> Result<Vec<u8>, CanonError>;
pub fn digest_of<T: serde::Serialize>(t: &T) -> Result<String, CanonError>;
```

### crates/lictor-receipt (lib.rs: `pub mod tick; pub mod body; pub mod sign; pub mod ledger; pub mod curve; pub mod handoff; pub mod history; pub mod keys;` + re-exports)

```rust
use std::collections::BTreeMap; use lictor_canon::{F64Array, F64Hex}; use lictor_core::*;
pub const RECEIPT_SCHEMA: &str = "lictor-receipt/v1";
pub const TICKS_SCHEMA: &str = "lictor-ticks/v1";
pub const LEDGER_SCHEMA: &str = "lictor-ledger/v1";
pub const CURVE_SCHEMA: &str = "lictor-curve/v1";
pub const HANDOFF_SCHEMA: &str = "lictor-handoff/v1";
pub const ZERO_HASH: &str = "0000000000000000000000000000000000000000000000000000000000000000";

// tick.rs -- the VERDICT chain (replayable) and the TIMING chain (honestly not)
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TickEvent {
    pub seq: u32, pub t: u32,
    pub state: FuseState, pub prev_state: FuseState, pub status: Status,
    pub trips: u32, pub action_src: ActionSource, pub substituted: bool, pub clamped_dims: u32,
    pub action: F64Array,      // [action_dim]
    pub f: F64Array,           // [NFEAT] raw features (what `lictor sweep` re-scores)
    pub z: F64Array,           // [NFEAT]
    pub valid: u32, pub fired: u32,
    pub s: F64Hex, pub tau: F64Hex, pub window_hits: u8, pub brake_margin: F64Hex,
    pub reason: ReasonCode, pub handoff_seq: Option<u32>, pub violation_reached_env: bool,
    pub prev: String, pub hash: String,   // hash = sha256(canon(self with hash = "")) -- see docs/receipt-schema.md
}
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TimingEvent { pub seq: u32, pub decide_ns: u64, pub io_ns: u64, pub prev: String, pub hash: String }
pub fn tick_event(prev: &str, v: &SafetyVerdict) -> TickEvent;
pub fn timing_event(prev: &str, seq: u32, decide_ns: u64, io_ns: u64) -> TimingEvent;
#[derive(Clone, Debug, PartialEq)] pub struct ChainReport { pub ok: bool, pub break_at: Option<u32>, pub head: String, pub n: u32 }
pub fn verify_tick_chain(events: &[TickEvent], genesis_prev: &str) -> ChainReport;
pub fn verify_timing_chain(events: &[TimingEvent]) -> ChainReport;

// body.rs
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RunBinding { pub run_id: String, pub arm_id: String, pub episode_index: u32, pub seed: u64,
    pub seed_pool: String, pub init_state_digest: String,
    pub env: BTreeMap<String, String>, pub policy: BTreeMap<String, String>, pub host: BTreeMap<String, String> }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BudgetBinding { pub mode: FuseMode, pub delay_steps: u16, pub tick_ms: u32, pub exec_mode: ExecMode, pub stitch: String,
    pub on_escalate: String, pub tier0_armed: Vec<String>, pub tier1_armed: bool, pub gate: Vec<String>,
    pub alpha_num: u32, pub alpha_den: u32, pub kn: [u8; 2] }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FaultBinding { pub kind: String, pub params: BTreeMap<String, String>, pub stream_seed: u64 }
#[derive(Clone, Debug, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub struct VerdictCounts { pub ticks: u32, pub nominal: u32, pub watching: u32, pub clamped: u32, pub braking: u32, pub held: u32,
    pub escalated: u32, pub fault: u32, pub terminated: u32, pub substituted: u32, pub chunks_seen: u32, pub chunks_rejected: u32,
    pub clamps: u32, pub holds: u32, pub rearms: u32, pub escalations: u32, pub trips_by_bit: Vec<u32>, pub fired_by_feat: Vec<u32>,
    pub first_trip_tick: Option<u32>, pub first_trip_reason: Option<String>, pub first_stop_tick: Option<u32>,
    pub handoff_tick: Option<u32>, pub violations_reached_env: u32, pub terminal_state: FuseState }
impl From<&lictor_fuse::Tally> for VerdictCounts { fn from(t: &lictor_fuse::Tally) -> Self; }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeOutcome { pub steps: u32, pub success: bool, pub terminated: bool, pub truncated: bool,
    pub max_coverage: F64Hex, pub final_coverage: F64Hex, pub reward_sum: F64Hex,
    pub ended_by: String,          // "success"|"truncated"|"escalation_terminate"|"fault"|"fuse_crash"|"abort"|"retune"
    pub max_s: F64Hex, pub max_z: F64Array }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LatencySummary { pub n: u32, pub p50_ns: u64, pub p90_ns: u64, pub p99_ns: u64, pub p999_ns: u64, pub max_ns: u64, pub label: String }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReceiptBody {
    pub schema: String, pub canonical: String, pub created_epoch: u64,
    pub lictor_version: String, pub lictor_git: String, pub lictor_sha256: String,
    pub client: String,                    // the `hello.client` string (host-declared)
    pub run: RunBinding, pub budget: BudgetBinding, pub fault_injection: Option<FaultBinding>,
    pub envelope: serde_json::Value,       // floatify(envelope) -- the full config, float-free
    pub envelope_digest: String, pub calibration_digest: Option<String>,
    pub inputs: BTreeMap<String, String>,  // content-addressed: key -> sha256. Host-declared entries (from episode_begin.inputs) use repo-relative paths;
                                           // fuse-computed entries use the keys "lictor:bin", "lictor:envelope", "lictor:calibration" (absent when no calibration)
    pub counts: VerdictCounts, pub outcome: EpisodeOutcome, pub handoffs: Vec<HandoffRecord>,
    pub verdict_events: u32, pub verdict_chain_head: String, pub timing_events: u32, pub timing_chain_head: String,
    pub latency: LatencySummary,
    pub ticks_policy: String,              // "tail32"|"all"|"none"
    #[serde(default)] pub ticks: Vec<TickEvent>,   // embedded per ticks_policy; the full stream is in the ticks file
    pub fuse_ok: bool, pub fuse_notes: Vec<String>,
    pub ledger_prev: Option<String>,
}
impl ReceiptBody { pub fn canonical(&self) -> Result<Vec<u8>, lictor_canon::CanonError>; pub fn digest_hex(&self) -> Result<String, lictor_canon::CanonError>; }
/// The honest verdict on the RECORD (not on the run). intact != fuse_ok. `ephemeral_key`: the receipt was signed by a key generated
/// at `lictor serve` startup (no `--key`); `verify` recomputes it as `body.fuse_notes.iter().any(|n| n == "ephemeral signing key")`.
pub fn evaluate_fuse(budget: &BudgetBinding, counts: &VerdictCounts, calibration_digest: Option<&str>, ephemeral_key: bool) -> (bool, Vec<String>);

// sign.rs
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedReceipt { pub body: ReceiptBody, pub body_digest: String, pub pubkey: String, pub sig: String }
pub fn sign(body: ReceiptBody, seed: &[u8; 32]) -> Result<SignedReceipt, ReceiptError>;
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct VerifyReport { pub schema_ok: bool, pub sig_ok: bool, pub digest_ok: bool, pub ticks_ok: bool, pub pubkey_ok: bool,
                          pub envelope_digest_ok: bool, pub counts_ok: bool, pub fuse_ok: bool, pub break_at: Option<u32>, pub notes: Vec<String> }
/// pubkey_ok = (expect_pubkey.is_none() || sr.pubkey == expected). envelope_digest_ok = recomputed sha256(canon(body.envelope)) == body.envelope_digest.
/// counts_ok = counts.ticks == verdict_events == timing_events == outcome.steps.
impl VerifyReport { pub fn intact(&self) -> bool { self.schema_ok && self.sig_ok && self.digest_ok && self.ticks_ok && self.pubkey_ok && self.envelope_digest_ok && self.counts_ok } }
pub fn verify(sr: &SignedReceipt, expect_pubkey: Option<&str>) -> VerifyReport;
/// Recompute the verdict chain from a full ticks file. `ChainReport.ok` is CHAIN INTEGRITY ONLY (a re-chained file passes it);
/// the caller MUST compare `report.head` with `sr.body.verdict_chain_head` and report `HEAD MISMATCH` when they differ, and MUST
/// cross-check the ticks header (run_id, arm_id, episode_index, genesis) against the body and the embedded tail's first `prev`.
pub fn verify_ticks_file(sr: &SignedReceipt, ticks: &[TickEvent]) -> ChainReport;
/// The two committed TEST keys (their pubkeys), so `lictor verify` can warn "signed with the committed test key".
pub const TEST_PUBKEYS: [&str; 2];   // filled in by WP-4 from tests/fixtures/receipt/key.hex and bench/fixtures/key.hex
#[derive(Debug, thiserror::Error)] pub enum ReceiptError { #[error("canon: {0}")] Canon(#[from] lictor_canon::CanonError), #[error("key: {0}")] Key(String), #[error("io: {0}")] Io(String) }

// ledger.rs -- a hash chain per arm: dropping, reordering or editing an entry WITHOUT the signing key breaks the chain at a reported seq.
// It does not defend against the key-holder (who can rebuild and re-sign); tail truncation of a DECLARED pool is detected by `lictor curve`.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LedgerEntry { pub seq: u32, pub run_id: String, pub arm_id: String, pub episode_index: u32, pub seed: u64,
    pub init_state_digest: String, pub receipt_digest: String, pub verdict_chain_head: String,
    pub success: bool, pub fuse_ok: bool, pub tripped: bool, pub stopped: bool, pub escalated: bool,
    pub prev: String, pub hash: String }
pub fn ledger_entry(prev: &str, seq: u32, sr: &SignedReceipt) -> LedgerEntry;
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LedgerReport { pub chain_ok: bool, pub break_at: Option<u32>, pub episodes: u32, pub successes: u32, pub trips: u32, pub stops: u32, pub escalations: u32, pub head: String }
pub fn verify_ledger(entries: &[LedgerEntry]) -> LedgerReport;
pub fn read_ledger(path: &std::path::Path) -> Result<Vec<LedgerEntry>, ReceiptError>;
pub fn append_ledger(path: &std::path::Path, sr: &SignedReceipt) -> Result<LedgerEntry, ReceiptError>;   // reads head, appends, single writer

// handoff.rs -- the human-escalation surface
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AckToken { pub schema: String, pub handoff_digest: String, pub decision: AckDecision, pub operator: String,
                      pub nonce: u64, pub note: String, pub sig: String }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HandoffRecord { pub schema: String, pub run_id: String, pub arm_id: String, pub episode_index: u32,   // replay-binding: a captured ack
    pub seq: u32, pub tick: u32, pub reason: ReasonCode, pub reason_text: String,                                   // from another run/arm/episode cannot verify
    pub reasons: Vec<String>, pub trips: u32, pub fired: u32, pub window_hits: u8, pub top_z: Vec<(String, F64Hex)>,
    pub chain_at: String,    // == the `hash` of the TickEvent of the tick on which Escalated was entered (the verdict-chain head after that tick)
    pub envelope_digest: String, pub calibration_digest: Option<String>,
    pub digest: String,  // sha256(canon(self with digest="", ack=None, resolved_tick=None, outcome="")) -- WHAT THE OPERATOR SIGNS
    pub ack: Option<AckToken>, pub resolved_tick: Option<u32>, pub outcome: String }  // "resumed"|"aborted"|"retune"|"timeout"|"unacked"
pub fn ack_signing_bytes(t: &AckToken) -> Result<Vec<u8>, ReceiptError>;   // canon({handoff_digest,decision,operator,nonce,note,schema})
pub fn sign_ack(handoff_digest: &str, decision: AckDecision, nonce: u64, note: &str, seed: &[u8; 32]) -> Result<AckToken, ReceiptError>;
#[derive(Debug, Clone, PartialEq, Eq)] pub enum AckError { UnknownOperator, BadSignature, WrongHandoff, NonceReplay, NoPendingHandoff, TooLongNote }
/// Pure verification: operator listed in `operators` (slot = index), Ed25519 verify_strict, digest == pending, nonce > last_nonce[slot].
/// `last_nonce` lives in the Session ACROSS episodes (never reset per episode) and is persisted per pubkey in
/// `<out>/.lictor/verifier_nonce.json` when `--out` is set; without that file acks are replayable across processes (SECURITY.md).
pub fn verify_ack(t: &AckToken, operators: &[String], pending_digest: Option<&str>, pending_seq: u32, last_nonce: &[u64]) -> Result<VerifiedAck, AckError>;
pub fn load_nonces(path: &std::path::Path) -> Result<BTreeMap<String, u64>, ReceiptError>;   // pubkey -> last nonce; missing file == empty
pub fn save_nonces(path: &std::path::Path, m: &BTreeMap<String, u64>) -> Result<(), ReceiptError>;

// curve.rs
/// A paired difference with its bootstrap CI (10 000 resamples, splitmix64 seed 20260830) and exact McNemar on the same pairs.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeltaCi { pub diff: F64Hex, pub lo: F64Hex, pub hi: F64Hex, pub mcnemar_p: F64Hex, pub mcnemar_b: u32, pub mcnemar_c: u32 }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveMetrics { pub n: u32, pub success_rate: F64Hex, pub success_ci: [F64Hex; 2], pub averted_rate: F64Hex, pub flagged_rate: F64Hex,
    pub flagged_ci: [F64Hex; 2], pub false_trip_rate: F64Hex, pub false_trip_ci: [F64Hex; 2], pub intervention_rate: F64Hex,
    pub intervention_tick_frac: F64Hex, pub escalation_rate: F64Hex, pub lead_mean: F64Hex, pub lead_p50: F64Hex, pub lead_p10: F64Hex,
    pub aucpdt: F64Hex, pub roc_auc: F64Hex, pub bacc: F64Hex, pub violations_reached_env: u32,
    pub delta_vs_baseline: DeltaCi,                    // success(arm) - success(obs-d0), paired by episode index
    pub delta_vs_latency_control: Option<DeltaCi>,     // success(arm) - success(obs-d{d}); None when no latency-control arm exists in the run
    pub tce_valid_frac: F64Hex,                        // fraction of ticks with Feat::Tce valid (sync d >= 7 has NO chunk overlap -> 0.0)
    pub latency_p50_ns: u64, pub latency_p99_ns: u64, pub latency_max_ns: u64, pub latency_label: String,
    pub n_fail_baseline: u32, pub n_succ_baseline: u32 }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CurveReceiptBody { pub schema: String, pub canonical: String, pub created_epoch: u64, pub run_id: String, pub arm_id: String,
    pub baseline_arm_id: String, pub latency_control_arm_id: Option<String>, pub ledger_head: String, pub n_episodes: u32, pub ledger_chain_ok: bool,
    /// sha256(canon({"envelope_digest": String, "calibration_digest": Option<String>, "budget": BudgetBinding})) of the arm's receipts (all identical, else refused)
    pub arm_config_digest: String,
    pub receipt_pubkey: String,                        // the ONE pubkey every receipt of the arm carries (mixed keys or an ephemeral note -> refused)
    pub run_json_sha256: String,                       // sha256 of results/<run>/run.json (the unsigned harness manifest, bound here)
    pub n_declared: u32, pub n_present: u32, pub n_missing: u32, pub missing_indices: Vec<u32>,   // declared pool (run.json + calibration seed_pool) vs ledger
    pub partial: bool,                                 // true only under --partial (index sets differ from the baseline's or from the declared pool)
    pub small_n: bool,                                 // true only under --allow-small (n < 100): a pilot point, never a headline
    pub cross_run_mismatches: Vec<u32>,                // episode indices whose verdict_chain_head differs from the same (arm, seed) in --compare-run
    pub pair_mismatches: Vec<u32>, pub seed_overlap_with_calibration: Vec<u64>, pub metrics: CurveMetrics, pub eps_prog: F64Hex }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SignedCurve { pub body: CurveReceiptBody, pub body_digest: String, pub pubkey: String, pub sig: String }
pub fn sign_curve(body: CurveReceiptBody, seed: &[u8; 32]) -> Result<SignedCurve, ReceiptError>;
pub fn verify_curve(sc: &SignedCurve, expect_pubkey: Option<&str>) -> VerifyReport;   // ticks_ok/counts_ok/envelope_digest_ok = true (n/a)

// history.rs -- sbx run-memory lineage -> .lictor/history.jsonl, capped at 200
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EpisodeRecord { pub ts: String, pub run_id: String, pub arm_id: String, pub seed: u64, pub success: bool,
                           pub first_trip_reason: Option<String>, pub trips: Vec<String>, pub stopped: bool, pub escalated: bool }
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct LoopSignals { pub tail_streak: Option<(String, u32)>, pub identical_run: u32, pub last_n: u32 }
pub fn record_episode(dir: &std::path::Path, r: &EpisodeRecord) -> std::io::Result<()>;
pub fn summarize(dir: &std::path::Path, n: usize) -> std::io::Result<LoopSignals>;

// keys.rs
pub fn keygen() -> [u8; 32];                                    // getrandom
pub fn pubkey_hex(seed: &[u8; 32]) -> String;
pub fn load_seed(path: &std::path::Path) -> Result<[u8; 32], ReceiptError>;   // 64 hex chars
/// mode 0600 on unix (behind `#[cfg(unix)]` via std::os::unix::fs::PermissionsExt; a no-op with a stderr warning elsewhere).
/// Refuses (ReceiptError::Key) a path under `/mnt/[a-z]/` (WSL DrvFs does not enforce 0600) unless `allow_drvfs`; refuses to
/// overwrite an existing file unless `force`.
pub fn save_seed(path: &std::path::Path, seed: &[u8; 32], force: bool, allow_drvfs: bool) -> Result<(), ReceiptError>;
/// `$LICTOR_KEYS` if set, else `$HOME/.lictor` (ext4 on this machine) -- NEVER the repo directory.
pub fn default_key_dir() -> std::path::PathBuf;
```

### crates/lictor-runtime (lib.rs: `pub mod wire; pub mod codec; pub mod session; pub mod episode; pub mod latency; pub mod trace;`)

```rust
// wire.rs -- serde types for lictor-wire/v1 (see the WIRE PROTOCOL section).
// `deny_unknown_fields` on the internally-tagged `Request` enum does NOT reject unknown keys inside the variant payloads, so it
// is on EVERY payload struct below as well (fail-closed: an unknown key anywhere in a request is a schema fault).
// Every real-valued REQUEST field is Option<f64> (JSON null == non-finite -> NaN -> GUARD NONFINITE). Real-valued RESPONSE
// fields are f64; serde_json emits `null` for a non-finite f64, and exactly two response fields can be non-finite BY DESIGN:
// `scores.s` (NEG_INFINITY while no gate term is fully valid or Tier 1 is disarmed) and `tau` (+INFINITY when disarmed or k > n),
// same for `hello_ok.calibration.tau`. The Python client maps null -> -inf / +inf for those keys ONLY.
pub const PROTO: &str = "lictor-wire/v1";
#[derive(Debug, Clone, serde::Deserialize)] #[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request { Hello(HelloReq), EpisodeBegin(EpisodeBeginReq), Tick(TickReq), EpisodeEnd(EpisodeEndReq), Bye(ByeReq) }
#[derive(Debug, Clone, serde::Serialize)] #[serde(tag = "kind", rename_all = "snake_case")]
pub enum Response { HelloOk(HelloOk), EpisodeOk(EpisodeOk), Verdict(VerdictMsg), EpisodeReceipt(EpisodeReceiptMsg), ByeOk { id: u64 }, Error(ErrorMsg) }
// ---- request payloads (all `#[derive(Debug, Clone, serde::Deserialize)] #[serde(deny_unknown_fields)]`; keys == the WIRE PROTOCOL JSON)
pub struct HelloReq { pub id: u64, pub proto: String, pub client: String, pub mode: FuseMode, pub embodiment_id: String, pub action_dim: u16,
    pub pos_dim: u16, pub horizon: u16, pub exec_steps: u16, pub envelope_digest: String, pub calibration_digest: Option<String> }
pub struct RunMsg { pub run_id: String, pub arm_id: String, pub episode_index: u32, pub seed: u64, pub seed_pool: String, pub init_state_digest: String }
pub struct BudgetMsg { pub delay_steps: u16, pub tick_ms: u32, pub exec_mode: ExecMode, pub stitch: String, pub on_escalate: String }
pub struct BindingMsg { pub env: BTreeMap<String, String>, pub policy: BTreeMap<String, String>, pub host: BTreeMap<String, String> }
pub struct FaultInjectionMsg { pub kind: String, pub params: BTreeMap<String, String>, pub stream_seed: u64 }
pub struct EpisodeBeginReq { pub id: u64, pub run: RunMsg, pub budget: BudgetMsg, pub binding: BindingMsg, pub fault_injection: Option<FaultInjectionMsg>,
    #[serde(default)] pub inputs: BTreeMap<String, String> }   // host-declared content hashes: repo-relative path -> sha256 (absent == empty)
pub struct ObsMsg { pub pos: Vec<Option<f64>>, pub vel: Option<Vec<Option<f64>>>, pub aux: Vec<Option<f64>>, pub ext: Vec<Option<f64>> }
pub struct ChunkMsg { pub seq: u32, pub t_emit: u32, pub h: u16, pub d: u16, pub exec: u16, pub a: Vec<Vec<Option<f64>>> }
pub struct TickReq { pub id: u64, pub t: u32, pub idx: u16, pub missed_ticks: u8, pub obs: ObsMsg, pub chunk: Option<ChunkMsg>, pub ack: Option<AckToken> }
pub struct OutcomeMsg { pub steps: u32, pub success: bool, pub terminated: bool, pub truncated: bool, pub max_coverage: Option<f64>,
    pub final_coverage: Option<f64>, pub reward_sum: Option<f64>, pub ended_by: String }
pub struct EpisodeEndReq { pub id: u64, pub t: u32, pub outcome: OutcomeMsg }
pub struct ByeReq { pub id: u64 }
// ---- response payloads (all `#[derive(Debug, Clone, serde::Serialize)]`)
pub struct CalibrationInfo { pub method: CalMethod, pub alpha_num: u32, pub alpha_den: u32, pub n_calib: u32, pub tau: f64, pub kn: [u8; 2], pub gate: Vec<String>, pub digest: String }
pub struct HelloOk { pub id: u64, pub proto: String, pub lictor: String, pub git: String, pub lictor_sha256: String, pub envelope_digest: String,
    pub embodiment_digest: String, pub calibration_digest: Option<String>, pub pubkey: String, pub ephemeral_key: bool, pub mode: FuseMode,
    pub tier0_armed: Vec<String>, pub tier1_armed: bool, pub calibration: Option<CalibrationInfo>, pub features: Vec<String>, pub trip_names: Vec<String>,
    pub latency_label: String }
pub struct EpisodeOk { pub id: u64, pub state: FuseState, pub seq: u32 }
pub struct ScoresMsg { pub f: Vec<f64>, pub z: Vec<f64>, pub s: f64, pub valid: u32, pub fired: u32 }
pub struct VerdictMsg { pub id: u64, pub t: u32, pub seq: u32, pub status: Status, pub state: FuseState, pub prev_state: FuseState, pub trips: Vec<String>,
    pub trip_mask: u32, pub action: Vec<f64>, pub action_src: ActionSource, pub substituted: bool, pub clamped_dims: u32, pub scores: ScoresMsg, pub tau: f64,
    pub window_hits: u8, pub brake_margin: f64, pub reason: ReasonCode, pub reason_text: String, pub handoff: Option<HandoffRecord>,
    pub ack_result: Option<String>, pub violation_reached_env: bool, pub verdict_ns: u64, pub chain: String }
pub struct EpisodeReceiptMsg { pub id: u64, pub receipt_path: String, pub ticks_path: String, pub body_digest: String, pub verdict_chain_head: String,
    pub timing_chain_head: String, pub fuse_ok: bool, pub fuse_notes: Vec<String>, pub counts: VerdictCounts, pub ledger_seq: u32, pub ledger_head: String }
pub struct ErrorMsg { pub id: u64, pub code: String, pub message: String, pub fatal: bool }

// codec.rs
pub const MAX_LINE: usize = 1 << 20;
pub fn read_request(r: &mut impl std::io::BufRead, buf: &mut String) -> std::io::Result<Option<Request>>;  // None at EOF; Err(InvalidData) on malformed
pub fn write_response(w: &mut impl std::io::Write, resp: &Response) -> std::io::Result<()>;              // one line + flush

// session.rs -- one fuse, one episode at a time; shared by serve, replay, bench, selftest, and the fuse crate's alloc/determinism tests
/// A calibration ALREADY loaded and compiled by the caller (lictor-cli via lictor-calib). lictor-runtime does NOT depend on
/// lictor-calib (lictor-calib depends on lictor-runtime for the wire/trace types).
pub struct CalibrationLoaded { pub c: CalibrationC, pub digest: String, pub embodiment_digest: String, pub file_sha256: String, pub path: String }
pub struct SessionConfig { pub envelope: SafetyEnvelope, pub envelope_toml_sha: String, pub calibration: Option<CalibrationLoaded>,
    pub mode: FuseMode, pub tier0_override: Option<u32>, pub tier1: bool, pub ticks_policy: String, pub key: Option<[u8; 32]>,
    pub out_dir: Option<std::path::PathBuf>, pub trace: Option<std::path::PathBuf>, pub lictor_git: String, pub lictor_sha256: String,
    pub latency_label: String }   // default: `default_latency_label()`
/// Off-path staging: converts a TickReq into a TickInput with buffers allocated ONCE (null -> NaN; chunk rows -> ChunkBuf).
/// Used by Session and by bench/alloc/determinism tests, so there is exactly ONE wire -> TickInput conversion in the workspace.
pub struct Staging { /* pos/vel/aux/ext: [f64; MAX_*] + lengths, have_vel, chunk: ChunkBuf, have_chunk */ }
impl Staging {
    pub fn new() -> Self;
    /// Copies the request into the buffers. Err(message) on a dimension / horizon / idx / seq violation (the caller raises SCHEMA).
    pub fn stage(&mut self, cfg: &FuseConfig, req: &TickReq) -> Result<(), String>;
    /// Borrow the staged buffers as the pure input. `schema_fault` forces TripMask::SCHEMA in GUARD.
    pub fn input(&self, req: &TickReq, ack: Option<VerifiedAck>, schema_fault: bool) -> TickInput<'_>;
}
impl Default for Staging { fn default() -> Self { Self::new() } }
pub struct Session { /* Fuse + Staging + chains + timing + episode assembly + pending handoff + last_nonce (kept ACROSS episodes) */ }
impl Session {
    /// Refuses (Err) a calibration whose `embodiment_digest` != `cfg.envelope.embodiment_digest()` -- fatal at startup, never a note.
    pub fn new(cfg: SessionConfig) -> anyhow::Result<Self>;
    /// Handle one request; measures decide_ns around `decide()` and folds the chains AFTER it returns (never inside).
    /// `episode_end` is ACCEPTED while faulted (writes a receipt with terminal_state = fault, fuse_ok = false, ended_by as sent).
    pub fn handle(&mut self, req: Request) -> Response;
    /// Called by the serve loop after the response is flushed: wall-clock from request-line read to response flush, recorded into
    /// the timing chain entry of tick `seq` (the entry is emitted on the NEXT request or at episode_end; never inside decide()).
    pub fn note_io_ns(&mut self, seq: u32, ns: u64);
    pub fn fault_latched(&self) -> bool;
    pub fn verdict_chain_head(&self) -> &str;
    pub fn latency(&self) -> &latency::Hist;
}
/// WSL2_LABEL when /proc/version contains "microsoft" or "WSL", else "measured on <uname -sr>, non-RT kernel -- not a real-time environment".
pub fn default_latency_label() -> String;

// episode.rs -- assemble ReceiptBody from bindings + Tally + chains; sign; write receipt/ticks/timing; append ledger LAST
pub struct EpisodePaths { pub receipt: std::path::PathBuf, pub ticks: std::path::PathBuf, pub timing: std::path::PathBuf, pub ledger: std::path::PathBuf }
pub fn episode_paths(out_dir: &std::path::Path, run_id: &str, arm_id: &str, episode_index: u32) -> EpisodePaths;
pub fn write_episode(paths: &EpisodePaths, sr: &SignedReceipt, ticks: &[TickEvent], timing: &[TimingEvent]) -> anyhow::Result<LedgerEntry>;
/// Host-side crash accounting (`lictor crash-receipt`): when the serve child died before `episode_end`, write a minimal SIGNED receipt
/// with zero counts, `outcome = {steps: 0, success: false, ended_by: "fuse_crash"}`, `terminal_state: fault`, `fuse_ok: false`,
/// `fuse_notes: ["fuse process died before episode_end; host-written crash receipt"]`, empty ticks/timing files, and append the ledger.
pub fn write_crash_episode(cfg: &SessionConfig, run: RunBinding, budget: BudgetBinding, client: &str, note: &str) -> anyhow::Result<LedgerEntry>;

// latency.rs
pub struct Hist { /* hdrhistogram::Histogram<u64>, 1ns..10s, 3 sig figs */ }
impl Hist { pub fn new() -> Self; pub fn record(&mut self, ns: u64); pub fn summary(&self, label: &str) -> LatencySummary; pub fn to_csv(&self) -> String; }
impl Default for Hist { fn default() -> Self { Self::new() } }
pub const WSL2_LABEL: &str = "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment";

// trace.rs -- request trace files: "#meta {json}" first line, then every request line VERBATIM
pub struct TraceWriter; impl TraceWriter { pub fn open(p: &std::path::Path, meta: &serde_json::Value) -> std::io::Result<Self>; pub fn line(&mut self, raw: &str) -> std::io::Result<()>; }
pub struct TraceReader; impl TraceReader { pub fn open(p: &std::path::Path) -> std::io::Result<(Self, serde_json::Value)>; }
impl Iterator for TraceReader { type Item = std::io::Result<(u64 /*line_no*/, String /*raw*/, Request)>; }
```

### crates/lictor-calib (lib.rs: `pub mod file; pub mod traces; pub mod calibrate; pub mod envfit; pub mod metrics; pub mod sweep; pub mod curve;`)

```rust
// file.rs -- calibration.json (schema lictor-calibration/v1), float-free; see FILE FORMATS
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SeedPool { pub name: String, pub lo: u64, pub hi: u64 }   // inclusive range
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tier0Percentiles { pub v_max: F64Hex, pub a_max: F64Hex, pub j_max: F64Hex, pub reach_max: F64Hex, pub q: F64Hex }
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CalibrationFile { pub schema: String, pub canonical: String, pub method: CalMethod, pub alpha_num: u32, pub alpha_den: u32,
    pub n_total: u32,      // successful calibration episodes available
    pub n_scale: u32,      // episodes used to fit center/scale
    pub n_calib: u32,      // per-episode scores tau was taken over (2-way: == n_scale; 3-way: a disjoint set) -- k = ceil((n_calib+1)(1-alpha))
    pub n_holdout: u32, pub split: String,   // "2way" | "3way"
    pub seed_pool: SeedPool, pub source_run: String, pub source_arm: String,
    pub envelope_digest: String, pub embodiment_digest: String, pub policy_digest: String, pub lictor_git: String,
    pub horizon_ticks: u32, pub t_grid: u16, pub feature_ids: Vec<String>, pub mask: u32, pub gate: Vec<String>,
    pub center: F64Array, pub scale: F64Array, pub tau: F64Hex, pub kn: [u8; 2], pub warn_margin: F64Hex,
    pub holdout_fpr: F64Hex, pub holdout_fpr_k1: F64Hex, pub tier0_percentiles: Option<Tier0Percentiles>, pub notes: Vec<String>, pub digest: String }
impl CalibrationFile { pub fn load(p: &std::path::Path) -> anyhow::Result<Self>; pub fn save(&self, p: &std::path::Path) -> anyhow::Result<()>;
                       pub fn compile(&self) -> anyhow::Result<CalibrationC>; pub fn digest_hex(&self) -> anyhow::Result<String>;
                       /// -> lictor_runtime::session::CalibrationLoaded (file_sha256 over the bytes on disk)
                       pub fn loaded(&self, path: &std::path::Path) -> anyhow::Result<lictor_runtime::session::CalibrationLoaded>; }
// traces.rs -- one Trace per episode, built from the ticks file + receipt (labels) + the request trace (coverage), pool-filtered.
// Trace lines are parsed with lictor_runtime::trace::TraceReader (the ONE wire parser); never a second serde_json::Value parser.
pub struct Trace { pub arm_id: String, pub seed: u64, pub episode_index: u32, pub success: bool, pub steps: u32,
                   pub coverage: Vec<f64>, pub f: Vec<[f64; NFEAT]>, pub valid: Vec<u32>, pub first_stop_tick: Option<u32>, pub init_state_digest: String }
pub fn load_traces(run_dir: &std::path::Path, arm_id: &str) -> anyhow::Result<Vec<Trace>>;   // verifies each receipt + ticks chain first; refuses broken ones
// calibrate.rs
pub struct CalibrateOpts { pub alpha_num: u32, pub alpha_den: u32, pub method: CalMethod, pub gate: Vec<String>, pub holdout_num: u32, pub holdout_den: u32,
                           pub kn: [u8; 2], pub warn_margin: f64, pub n_calib_cap: Option<u32>, pub horizon_ticks: u32,
                           pub split: u8 }   // 2: center/scale AND tau on the fit subset (bound approximate); 3: center/scale on A (40 %), tau on B (30 %), holdout C (30 %) (bound exact at K = 1)
pub fn robust_center_scale(fit: &[Trace], method: CalMethod, horizon_ticks: u32) -> ([[f64; NFEAT]; T_GRID], [[f64; NFEAT]; T_GRID], u16);
pub fn episode_max_s(tr: &Trace, cal: &CalibrationC) -> f64;
pub fn split_quantile(sorted_max_s: &[f64], alpha_num: u32, alpha_den: u32) -> (f64, Option<String>);  // (tau, note); +INF when k > n
pub fn calibrate(traces: &[Trace], opts: &CalibrateOpts, envelope: &SafetyEnvelope, policy_digest: &str, source_run: &str) -> anyhow::Result<CalibrationFile>;   // binds envelope.digest_hex() AND envelope.embodiment_digest()
// envfit.rs
pub struct FitOpts { pub quantile: f64, pub slack: f64 }
pub fn fit_envelope(base: &SafetyEnvelope, traces: &[Trace], opts: &FitOpts, operators: &[String]) -> anyhow::Result<(SafetyEnvelope, String /*report md*/)>;   // `operators` replaces base.operators (the oracle envelope); embodiment untouched
// metrics.rs
pub fn clopper_pearson(k: u32, n: u32, conf: f64) -> (f64, f64);
pub fn mcnemar_exact(b: u32, c: u32) -> f64;
pub fn ponr_tick(coverage: &[f64], eps_prog: f64) -> u32;      // t_fail = min{t : c*_T - c*_t < eps}
pub fn roc_auc(pos: &[f64], neg: &[f64]) -> f64;
pub fn aucpdt(leads: &[i64], horizon: u32) -> f64;
pub fn paired_bootstrap_diff(a: &[bool], b: &[bool], n_resamples: u32, seed: u64) -> (f64, f64, f64);   // (diff, lo, hi); RNG = splitmix64 (in-crate)
// sweep.rs
pub struct DetectorSpec { pub name: String, pub gate: Vec<String>, pub kn: [u8; 2], pub tier0: bool }
pub struct SweepPoint { /* exactly the sweep.jsonl keys in FILE FORMATS */ }
/// `artefacts`: calibration.<alpha>.json files already produced by `lictor calibrate`; when an (alpha, gate, kn, method) matches one,
/// its center/scale/tau are used VERBATIM (`tau_source: "artefact"`) so the Layer-A point is comparable to the Layer-B arm;
/// otherwise a full fit with no holdout is done (`tau_source: "fit"`).
pub fn sweep(calib: &[Trace], eval: &[Trace], alphas: &[(u32, u32)], dets: &[DetectorSpec], method: CalMethod, horizon_ticks: u32, eps_prog: f64,
             artefacts: &[CalibrationFile]) -> Vec<SweepPoint>;
// curve.rs -- closed-loop metrics from receipts ONLY
pub struct CurveOpts { pub eps_prog: f64, pub latency_control: Option<String>, pub calib_seed_override: Option<(u64, u64)>, pub allow_small: bool,
                       pub partial: bool, pub compare_run: Option<std::path::PathBuf>, pub n_boot: u32, pub boot_seed: u64 }   // 10_000, 20260830
/// Refuses (Err, exit 1 in the CLI) when: the ledger is broken; the arm's receipts carry >1 pubkey or an ephemeral-key note; the arm's
/// budget.tier1_armed / alpha disagree with run.json; the arm's episode-index set != the baseline's or != the declared pool (unless
/// `partial`); n < 100 (unless `allow_small`). Missing indices count as success = false, stopped = true (terminate_fail accounting).
pub fn curve(run_dir: &std::path::Path, arm: &str, baseline: &str, opts: &CurveOpts) -> anyhow::Result<CurveReceiptBody>;
```

## 5. Detector math (exact)

Notation. `dt = control_hz_den/control_hz_num` (PushT 0.1 s). `p_t` proprioceptive position at step t (px). Chunk k emitted at `t_k` with `H = 15`, `d = 2`, `S = exec_steps = 8`, `L = H - S = 7`. `a_i` = row i of the current chunk; predecessor `q_{-1} = p_{t_k}` (current position), `q_i = a_i`. `abar_i[c] = (a_i[c] - norm_center[c]) / norm_scale[c]`. `v_hat_t = obs.vel` -- on PushT the simulator's own `info["vel_agent"]` (`provides_vel = true`); the finite-difference fallback `(p_t - p_{t-1}) / dt` (zero on the first tick) is used only for embodiments with `provides_vel = false` and is recorded as the fuse_note `v_hat finite-differenced` in every receipt of such a run. The finite difference is the MEAN velocity over the last control period, not the instantaneous state the PD integrator needs, so a rollout from it is approximate; from the simulator velocity it is exact. All norms Euclidean with fixed left-to-right accumulation; `eps = 1e-9`.

### 5.1 Tier 0 -- geometric / kinematic. Hard limits, never windowed, fire with K = 1. `trip iff quantity > limit` (strict).

Box is margin-adjusted once at compile: `lo = box_lo + margin`, `hi = box_hi - margin`.

| bit | check | formula (chunk boundary: all i in [0, H); intra-chunk tick: the single committed a_idx with q = last_cmd) | PushT base | clampable |
|---|---|---|---|---|
| `workspace` | box | `lo[c] <= a_i[c] <= hi[c]` | [17,17]..[495,495] | yes: componentwise clamp |
| `speed` | commanded speed | `v_i = ||a_i - q_{i-1}|| / dt <= v_max` | 1000 px/s | yes: sequential leash |
| `accel` | commanded accel | `||a_{i+1} - 2 a_i + a_{i-1}|| / dt^2 <= a_max`, i in [0, H-1) | 20000 | yes (via leash on the offending step) |
| `jerk` | commanded jerk | `||a_{i+2} - 3 a_{i+1} + 3 a_i - a_{i-1}|| / dt^3 <= j_max`, i in [0, H-2) | 4e5 | yes (via leash) |
| `reach` | teleport guard | `||a_0 - p_{t_k}|| <= reach_max` | 150 px | yes: pull a_0 toward p_t along the ray |
| `contact` | reduced speed near the object (perception-assisted, OFF by default) | if `||p_t - c_block|| <= R` then `v_i <= v_contact` | R=80, v=400 | yes: leash with step_max_contact |
| `brake` | the checked stopping condition | see 5.2 | -- | NO -> Braking |
| `nonfinite` | fail-closed | any non-finite in obs, ext or chunk | always armed | NO -> Fault |
| `schema` | fail-closed | dims/horizon/idx/id violations (raised by the runtime, applied by decide) | always armed | NO -> Fault |
| `watchdog` | fail-closed | `missed_ticks > watchdog_ticks` | always armed | NO -> Fault |

Base limits are placeholders. `lictor envelope fit --quantile 0.999 --slack 1.25` sets `v_max`, `a_max`, `j_max`, `reach_max` to the empirical p99.9 of the same quantities over calibration SUCCESSES times a declared slack, writes `envelopes/pusht.toml` with a `[fit]` record (and `envelopes/pusht.oracle.toml` with `--operator`), and the McNemar parity gate (10.3) must pass before any curve point is reported. Hand-picked limits are either decorative or dishonest. Fitting changes ONLY these four Tier-0 limits; no Tier-1 feature depends on them (5.3), so a calibration fitted under the base envelope remains valid under the fitted one -- `lictor serve` enforces that by binding the calibration to the `embodiment_digest`, not the envelope digest.

**Projection (`ClampMode::Project`), sequential direction-preserving leash, i = 0..H-1, q_{-1} = p_t:**

```
a'_i <- clamp(a_i, lo, hi)                       componentwise
delta <- a'_i - q_{i-1};  n <- ||delta||
if n > step_max (= v_max * dt):  a'_i <- q_{i-1} + delta * (step_max / n)
if i == 0 and ||a'_0 - p_t|| > reach_max:  a'_0 <- p_t + (a'_0 - p_t) * (reach_max / ||a'_0 - p_t||)
q_i <- a'_i
```
Uses only `+ - * / sqrt`. `clamped_dims` records which dims moved. Soft trips whose projection changed nothing (accel/jerk limits satisfied after the leash) still count as trips. `TIER0_HARD` bits are never clampable.

### 5.2 Brake feasibility -- the checked stopping condition (ARMTD / UR-style)

*At every tick, the committed prefix plus a braking manoeuvre must jointly satisfy all limits.* This is a checked condition under the model assumptions below (known PD gains and substep, contact-free free-space motion, the simulator's own `v0`); violations of those assumptions are not detected.

`BrakeKind::PdSecondOrder` (PushT; exact for the free-space agent GIVEN the simulator's `v0` -- which is why `obs.vel` is mandatory on PushT; from a finite-difference `v0` it is approximate):

```
p <- p_t ; v <- v_hat_t
for i in from .. commit_steps-1:                       # the irrevocable prefix (8 rows at a sync boundary, 8-d at an async-drop delivery, fewer intra-chunk)
    for _ in 0 .. substeps-1:                          # 10
        acc = k_p*(a_i - p) - k_v*v ;  v += acc*dt ;  p += v*dt     # dt = 0.01 (physics), NOT the control period
        margin = min(margin, min_c(min(p[c] - lo[c], hi[c] - p[c])))
a_hold = clamp(p, lo, hi)
for _ in 0 .. brake_steps*substeps-1:                  # 8*10 = 80
    acc = k_p*(a_hold - p) - k_v*v ;  v += acc*dt ;  p += v*dt
    margin = min(margin, ...)
feasible = margin >= 0 ; stop_dist = ||p - p_t||
```
EXACTLY `(commit_steps - from + brake_steps) * substeps` iterations (<= 160) of 2-D `+ - *` -- expected sub-microsecond (measured by the `brake` line of `lictor bench`), allocation-free, bit-reproducible on the tested target. Contact with the T-block only decelerates the agent in this env, so the contact-free model is conservative for the box constraint -- stated as a modelling assumption, not a proof.

Closed-form kinds (`FirstOrderDecay`, `BoundedAccel`, `JerkLimited`, `ZeroVelocityHold`), for embodiments whose plant is not known exactly:

```
p_0 = p_t ; v_0 = v_hat_t
for j in from .. commit_steps-1:   v_j = g*(a_j - p_j) [ee_position/joint_position with g = k_p/k_v] | a_j [joint_velocity] | a_j/dt [ee_delta]
                                   p_{j+1} = p_j + v_j*dt
d_stop(v) = ||v||/k_v  |  ||v||^2/(2 a_max)  |  ||v||^2/(2 a_max) + ||v||*a_max/(2 j_max)
R_j = d_stop(v_j) + ||v_j|| * react_ticks * dt
slack_{j,c} = min(p_j[c] - R_j - lo[c], hi[c] - p_j[c] - R_j) ;  margin = min over j,c
```
Braking action: position kinds -> `a_brake = clamp(p_t, lo, hi)` (setpoint := measured position; under the PD law acceleration becomes exactly `-k_v v`, the same decay the closed form assumed); velocity kinds -> `v_hat * max(0, 1 - a_max*dt/(||v_hat||+eps))` (ramp to zero). Hold action: the latched `a_hold` from the tick Held was entered. Braking decays (recomputed each tick) then Held latches (never chases the block). SS1/SS2-like: controlled stop, power retained, resumable; "controlled retreat" is deliberately not implemented (the SS1/SS2/STO terms are borrowed from IEC 60204-1 / ISO 10218 for readability; no conformance to any clause is claimed or tested).

### 5.3 Tier 1 -- action-space failure signals. Black-box, no extra policy passes, all dimensionless, all oriented LARGER = MORE ANOMALOUS.

| j | id | formula | cadence |
|---|---|---|---|
| 0 | `tce` | `sqrt( (1/(L d)) sum_{i<L} sum_c (abar^{k}_{S+i,c} - abar^{k+1}_{i,c})^2 )`, chunk k rows S..H-1 vs chunk k+1 rows 0..L-1 | boundary; held |
| 1 | `acc` | `sqrt( num / (den + eps) )`, `num` = the TCE sum, `den = sum_{i=1}^{L-1} sum_c (abar^{k+1}_{i,c} - abar^{k+1}_{i-1,c})^2` (VLA-FAIL velocity normalisation: "disagreeing" vs "merely moving fast") | boundary; held |
| 2 | `acm_neg` | `-(1/(H-1)) sum_{i=1}^{H-1} ||abar_i - abar_{i-1}||` (negated chunk magnitude: frozen policy = high score) | boundary; held |
| 3 | `njr` | `[ (1/(H-3)) sum_{i<H-3} ||abar_{i+3} - 3abar_{i+2} + 3abar_{i+1} - abar_i||^2 ] / [ eps + (1/(H-1)) sum_{i<H-1} ||abar_{i+1} - abar_i||^2 ]` | boundary; held |
| 4 | `reach` | `||a_0 - p_{t_k}|| / norm_scale_iso` (the soft twin of the hard reach check) | boundary; held |
| 5 | `path_ineff` | `1 - N_t / max(P_t, eps)`, `N_t = ||p_t - p_{t-W+1}||`, `P_t = sum ||p_s - p_{s-1}||` over the last W = 20 ticks | per tick, t >= W-1 |
| 6 | `stall` | `1 - min(1, N_t / (v_ref W dt))`, `v_ref = STALL_VREF_FRAC * norm_scale_iso / dt` (PushT: 0.05 * 256 / 0.1 = 128 px/s) | per tick, t >= W-1 |
| 7 | `speed_peak` | `max_i ||a_i - q_{i-1}|| / norm_scale_iso` over the current chunk (the largest normalised step; `= max_i v_i dt / norm_scale_iso`) | boundary; held |
| 8-11 | `ext0..ext3` | `obs.ext[j-8]` verbatim -- Tier-2 white-box scalars (logpZO / RND / SAFE probe / privileged progress) | per tick |

INVARIANT: every Tier-1 feature is a function of the EmbodimentManifest (`norm_center`, `norm_scale`, `dt`, `horizon`, `exec_steps`) and the data ONLY -- never of `v_max`/`a_max`/`j_max`/`reach_max` or `operators`, which change after calibration. `tce`/`acc` compare the incoming chunk with the previous RAW chunk (`Tier1Rt.prev`, written only by `features`), in Observe and Enforce alike, so calibration features and enforcement features are the same function. `valid` bit j is set only when the feature is computable: `tce`/`acc` need a previous chunk with `L >= 2`, where `L = min(prev.h - (new.t_emit - prev.t_emit), new.h)` -- in sync mode `L = 7 - d`, so they are INVALID for `d >= 7` and degraded for `d in 3..6` (disclosed on F1 via `tce_valid_frac`; async arms keep `L = 7`); `path_ineff`/`stall` need `t >= W-1`; `ext*` need `obs.ext.len() > j-8`; boundary features are invalid until the first chunk. TCE is a LAGGING feature (chunk k's disagreement is known only when chunk k+1 arrives, S steps later) -- accounted for in the lead-time analysis, not hidden. The privileged `coverage` aux is never a feature by default; a `cov_stall` ext channel is a clearly-labelled separate Pareto line.

### 5.4 The runtime decision rule (exactly, in order, inside `decide()`)

```
0. GUARD    nonfinite(obs.pos|vel|aux|ext) or chunk.!all_finite -> NONFINITE  (aux NaN faults even when no aux consumer is armed: deliberate) ;
            idx >= horizon or dims != cfg or inp.schema_fault -> SCHEMA ;
            time continuity: (seq == 0 && t != 0) || (seq > 0 && t != last_t + 1 + missed_ticks) -> SCHEMA ;
            chunk continuity: chunk.is_some() && (chunk.seq != next_chunk_seq || chunk.t_emit > t || idx != t - chunk.t_emit) -> SCHEMA ;
            missed_ticks > watchdog_ticks -> WATCHDOG.  Any of these: state = Fault (latched), action = hold, RETURN.
1. INGEST   prev_pos <- pos ; pos <- obs.pos ; v_hat = obs.vel or finite difference ; trail.push(pos) ; last_t <- t ;
            if chunk.is_some(): have_prev handled by features() ; next_chunk_seq++ ; chunks_seen++      (decide() NEVER writes t1.prev)
2. TIER 0   boundary: (trips0, clamped) = check_chunk(cfg, obs, chunk, &mut scratch)
                      brake = brake_feasible(cfg, pos, v_hat, scratch.view(), idx)     (from the row executed THIS tick: 0 sync/freeze, d async-drop)
                      if brake.!feasible: trips0 |= BRAKE
                      if trips0 & TIER0_HARD == 0: cur <- scratch  (accepted, possibly projected)  else chunks_rejected++
            intra:    (trips0, clamped) = check_action(cfg, obs, last_cmd, cur.action(idx), &mut scratch.row(idx))
                      brake = brake_feasible(cfg, pos, v_hat, cur.view(), idx) ; BRAKE bit likewise
            trips0 &= (cfg.tier0_enabled | TIER0_HARD)
3. TIER 1   features(cfg, &mut t1, obs, chunk /*RAW*/, idx, &mut sc)   (only when cfg.calib.armed(); else sc = default)
            features() is the sole writer of t1.prev: raw-vs-raw TCE in both modes
4. CONFORMAL b = cal.bin(t) ; z_j = (f_j - center[b][j]) / scale[b][j]  for j in mask & valid
            s = max over gate terms T of min_{j in T} z_j, over terms whose channels are ALL valid ; NEG_INFINITY if none
            hit = (s > tau)              STRICT
            warn = !hit && (s > tau - warn_margin)
5. WINDOW   window = ((window << 1) | hit) & window_mask ; predictive = popcount(window) >= K
6. FSM      (state', extra, reason) = fsm::next(cfg, rt, FsmInput{trips0, predictive, warn, soft_clampable, stopped, chunk_boundary, chunk_ok, ack})
            stopped = if mode == Observe { true } else { ||v_hat|| <= v_stop_eps }     (observe mode assumes the brake would have stopped)
7. ACTION   Armed|Watching -> cur.action(idx) [policy] ; Clamped -> scratch row [clamped] ; Braking -> brake_action [brake] ;
            Held|Escalated|Fault|Terminated -> hold (latched) [hold]
            if mode == Observe: action = raw policy row, action_src = policy, substituted = false,
                                violation_reached_env = (would-be src != policy) ; tally.violations_reached_env += 1
8. TALLY    counts, first_trip, first_stop, max_s = max(max_s, s) (max_s starts at NEG_INFINITY), max_z likewise, last_cmd <- action ; seq++
```
Everything in steps 0-8 is integer/float arithmetic on fixed-size arrays. `gate.eval` and the window are integer bit operations.

### 5.5 Hysteresis

- Tier-0 hard limits are never windowed. A limit is a limit; windowing it would be a lie.
- The predictive channel is windowed K-of-N (u64 ring, popcount). PushT default (3, 5). With K = 1 the split-CP bound `P(fire | nominal) <= alpha` holds exactly; with K > 1 the rule is strictly harder to trip, so alpha is a conservative UPPER bound and the empirical held-out rate is reported next to alpha in every figure and in `calibration.json` (`holdout_fpr`, `holdout_fpr_k1`).
- Clear-down is asymmetric: `Watching`/`Clamped` -> `Armed` after `clear_ticks` clean ticks (clean = no trip, no warn, no predictive) and the window is cleared. `Braking`/`Held`/`Escalated` never clear themselves: leaving them requires either the auto re-arm budget (`RearmPolicy::Auto`, protective-stop semantics) or a signed `AckToken` (`RearmPolicy::AckOnly`, or always for `Escalated`).

---

## 6. Escalation state machine

States: `Idle` (no episode) -> `Armed` (pass-through) / `Watching` (warn band or window hits < K; pass-through; informational) / `Clamped` (Tier-0 projection applied; motion continues) / `Braking` (controlled stop in progress; SS1/SS2-like) / `Held` (monitored standstill -- ISO 10218-1:2025 5.5.5 vocabulary, borrowed for readability; no conformance to any clause is claimed or tested; power retained; resumable) / `Escalated` (HandoffRecord emitted; awaiting a signed operator decision) / `Fault` (the fail-closed latch: non-finite input, schema/continuity violation, watchdog miss, brake timeout -- host/plant anomalies, not fuse-internal ones; STO-like, same caveat; never re-armed within an episode) / `Terminated` (ack Abort/Retune or handoff timeout; absorbing).

Transition table -- `fsm::next` implements EXACTLY this, top-down, first match wins. "clean" = no trip AND !warn AND !predictive this tick. Counters live in `FuseRt`.

| # | From | Condition | To | extra trips / reason / side effects |
|---|---|---|---|---|
| 1 | any | `trips & (NONFINITE|SCHEMA|WATCHDOG) != 0` | Fault | latch hold = clamp(pos or last finite pos); reason Fault* |
| 2 | Terminated | -- | Terminated | |
| 3 | Fault | -- | Fault | never re-armable in-episode |
| 4 | Escalated | ack Abort | Terminated | OPERATOR_ABORT; TerminatedAbort; handoff outcome "aborted" |
| 5 | Escalated | ack Retune | Terminated | TerminatedRetune; outcome "retune" (harness restarts with a new envelope) |
| 6 | Escalated | ack Resume | Armed | rearms++; window=0; clean_run=0; clamp_streak=0; held_ticks=0; last_nonce[slot]=nonce; RearmedAck |
| 7 | Escalated | `escalated_ticks >= handoff_timeout_ticks` | Terminated | HANDOFF_TIMEOUT; TerminatedTimeout; outcome "timeout" |
| 8 | Escalated | -- | Escalated | escalated_ticks++ |
| 9 | Held | ack Resume | Armed | as row 6 |
| 10 | Held | ack Abort | Terminated | as row 4 |
| 11 | Held | `rearm == Auto && held_clean >= rearm_hold && chunk_ok && rearms < max_rearms` | Armed | rearms++; window=0; RearmedAuto (chunk_ok = a fresh chunk passed the full Tier-0 suite incl. brake THIS tick) |
| 12 | Held | `held_ticks >= escalate_after_hold_ticks` | Escalated | handoff_seq++, handoff_pending; REARM_BUDGET if rearms >= max_rearms (EscalateRearmBudget) else EscalateTier1Persistent if predictive else EscalateHoldTimeout |
| 13 | Held | -- | Held | held_ticks++; held_clean = clean ? held_clean+1 : 0 |
| 14 | Braking | `brake_ticks >= brake_timeout_ticks` | Fault | BRAKE_TIMEOUT; FaultBrakeTimeout |
| 15 | Braking | `stopped` for `stop_confirm_ticks` consecutive ticks | Held | holds++; latch hold = clamp(pos); HeldStandstill |
| 16 | Braking | -- | Braking | brake_ticks++; stopped_ticks = stopped ? +1 : 0; Stopping |
| 17 | Armed/Watching/Clamped | `trips & BRAKE` | Braking | BrakeInfeasible; first_stop |
| 17b | Armed/Watching/Clamped | `trips & TIER0_SOFT != 0 && !soft_clampable` (clamp_mode == Off) | Braking | no extra trip (the soft bit stays); reason Clamp<first soft bit>; first_stop |
| 18 | Armed/Watching/Clamped | `predictive` | Braking | TIER1_CP; BrakeTier1Cp |
| 19 | Armed/Watching/Clamped | `clamp_streak >= clamp_streak_to_brake || clamps > max_clamps_per_episode` | Braking | CLAMP_BUDGET; BrakeClampBudget |
| 20 | Armed/Watching/Clamped | `trips & TIER0_SOFT != 0 && soft_clampable` | Clamped | clamps++ (per tick); clamp_streak++ on chunk_boundary; Clamp<first bit> |
| 21 | Watching/Clamped | `clean_run >= clear_ticks` | Armed | window=0; clamp_streak=0; Ok |
| 22 | Armed/Watching/Clamped | `warn || popcount(window) >= 1` | Watching | WatchBand |
| 23 | Armed | -- | Armed | Ok (ObserveOnly when mode == Observe) |

Notes: `soft_clampable = (clamp_mode == Project)`; row 17b is the explicit `Off` case (Braking with the clamp reason and the soft trip bit; `BrakeInfeasible` is NOT used) so `fixtures/fsm/transitions.json` covers it. `clean_run` increments on clean ticks and resets otherwise, in every state. The `Watching` state substitutes nothing -- its count is a reported metric ("detector fired below K, envelope intact").

**What "escalate to human" concretely does.** On entering `Escalated` the runtime builds a `HandoffRecord` (reason code + plain-English `reason_text`, trip names, top-3 z-scores, state digest, chain head, envelope/calibration digests, self-digest) and returns it inside the verdict. The harness acts per `--on-escalate`:

- `terminate_fail` (DEFAULT; the only accounting used for headline numbers): the episode stops immediately, counted NOT a success, `ended_by="escalation_terminate"`. An escalation costs you the task.
- `oracle_resume`: the harness's designated operator key signs a `Resume` AckToken against the handoff digest (nonce = handoff_seq+1) and the policy continues -- models "a human took over and did as well as the policy would have". Scored in a separate "success with assistance" column, never folded into plain success. Exercises the full ack path end-to-end so the receipt shows which key authorised which resume against which digest.
- `continue`: recorded but ignored (pure detector measurement without closed-loop coupling).

Re-arm semantics: `Held -> Armed` under `Auto` requires ALL of `rearm_hold` consecutive clean ticks in Held, a fresh chunk passing the full Tier-0 suite including brake feasibility, and `rearms < max_rearms`; exhausting the budget escalates. Under `AckOnly`, only an `AckToken` from a key listed in the signed envelope, with a strictly increasing per-operator nonce, leaves Held/Escalated. Fault is never re-armable. Every ack (accepted or rejected) is recorded in the ticks file and the receipt.

---

## 7. Split-conformal calibration (offline) and the deterministic runtime rule

Data hygiene, non-negotiable: calibration rollouts come from a seed pool DISJOINT from eval (calib = `900000..900299`; eval = `0..499`; pilot eval = `0..99`) and use ONLY episodes the policy succeeded on (no failure labels required -- the FIPER / surgical-detector economy). `lictor curve` and the harness's `--check-pools` assert disjointness; `lictor calibrate` refuses if `horizon_ticks` differs from `envelope.embodiment.horizon_ticks` or from the receipts' `run.env.max_episode_steps`, and binds `embodiment_digest` + `policy_digest` so `serve` / `episode_begin` can refuse a calibration for a different manifest or different weights.

1. Traces: run the calibration pool in `observe` mode (arm `calib-obs`). Each episode's ticks file stores `f[t]`, `valid[t]`; the receipt stores `success` and `max_coverage`; the trace stores the request lines. `load_traces` verifies each receipt and ticks chain before use.
2. Keep successes only: `n_total ~ 196` at the published 65.4 % (300 calibration episodes). Deterministic 70/30 fit/held-out split by sorted episode index (fit = the first 70 %, `n_scale = n_calib = 137`, `n_holdout = 59`); the held-out subset never touches tau. `--split 3` instead uses 40/30/30: center/scale on A (`n_scale = 78`), tau on B (`n_calib = 59`), holdout on C (59) -- the exact-bound variant, with fewer scores per part (alpha = 1/100 is then degenerate: k = 60 > 59).
3. Time base: `T_GRID = 100` bins over `horizon_ticks = 300`, `bin(t) = min(99, t*100/300)` in integer arithmetic (`lictor_core::bin_of`, shared by runtime and calibrator). `method = static` is the same procedure with `t_grid = 1`.
4. Robust standardisation per (bin b, feature j) over all valid samples in the fit subset, in fixed order (episode index ascending, then t ascending): `center = median`, `scale = 1.4826 * MAD`, floored at `1e-9`; sorting by `f64::total_cmp`. Empty bins are filled deterministically: backward from the nearest lower populated bin, else forward; a feature with no samples at all is dropped from the mask with a note.
5. Per-episode nonconformity: `M_i = max_t s_t^{(i)}` where `s_t` is the gated aggregate of sec 5.4 with the chosen gate, over the `n = n_calib` episodes tau is taken from (2-way: the 137 fit episodes; NOT the 196 successes). Sort ascending. With `k = ceil((n+1)(1-alpha))` computed in exact rational arithmetic on `alpha_num/alpha_den`: `tau = M_(k)` if `k <= n`, else `+INFINITY` (never fires -- recorded as a note, and the figures show the point as degenerate, not as data). Worked numbers for n = 137: alpha = 1/100 -> k = 137 (tau = the maximum: barely non-degenerate); 2/100 -> 136; 5/100 -> 132; 10/100 -> 125; 20/100 -> 111; 5/1000 -> 138 > 137 -> +INF.
6. Guarantee, stated honestly: for an exchangeable nominal episode, `P(max_t s_t > tau) <= alpha` at K = 1 holds EXACTLY only when the score function (center/scale) was fitted on episodes disjoint from those tau was taken over (`--split 3`). Under the default 2-way split the standardisation is fitted on the same 137 episodes whose `M_i` set tau, so their scores are not exchangeable with a fresh episode's and the bound is APPROXIMATE; `holdout_fpr_k1` on the untouched 30 % is the empirical check and F7 is the evidence. With K-of-N (K > 1) alpha is additionally a conservative upper bound. The gate's AND-terms (FIPER-style) suppress benign-OOD false alarms without a union bound because the threshold is on the single aggregate `s`, not per channel.
7. Validity check (figure F7): the held-out 30 % measures the empirical episode-level firing rate at the configured K-of-N (`holdout_fpr`) and at K = 1 (`holdout_fpr_k1`); plotted against nominal alpha with the diagonal. If the diagonal is badly violated, exchangeability is broken and we say so.
8. Economy ablation (figure F8): `n_calib in {25, 50, 100, all}` -> threshold stability.
9. Output: one `calibration.<alpha>.json` per alpha (float-free, self-digested). The runtime binds the digest into every receipt; `hello` cross-checks it. `lictor sweep --calibration-dir` reuses these SAME artefacts (identical tau) for the Layer-A points it compares with Layer-B arms; where it fits its own tau the point is labelled `tau_source: "fit"` and is not used for that comparison.

Why one aggregate threshold and not per-channel taus (design conflict, resolved): per-channel split quantiles carry only a union-bound guarantee (`n_ch * alpha`); the single quantile on the DNF aggregate carries the exact marginal bound at K = 1 and still expresses FIPER's AND-gating. Design 1's lambda-grid functional band is dropped in favour of design 2's integer time-binned standardisation with the same single quantile -- simpler, and identical in what it guarantees.

---

## 8. Canonical encoding, chains, receipts

**Canonical bytes = RFC 8785 JCS over a float-free JSON profile** (`jcs-floatfree/v1`): keys sorted by UTF-16 code units, no whitespace, integers only; every real number is `{"f64":"<16hex bits>"}`, every real array `{"f64a":"<base64 LE>","shape":[..]}`; every key matches `[a-z0-9_]+`. Consequence: Python's `json.dumps(json.loads(f), sort_keys=True, separators=(",",":"), ensure_ascii=False)` is byte-identical to lictor's canonical bytes, which is what makes `adapters/verify_receipt.py` a stdlib-only verifier rather than a promise. bulla's serde-field-order canonicalisation is broken NOW, while nothing external verifies lictor receipts. TOML-sourced config (the envelope) is `floatify`-ed before digesting.

**Two hash chains, deliberately** (design 2):

| Chain | Covers | Replayable? | Bound into the receipt as |
|---|---|---|---|
| `verdict_chain` | `TickEvent` = {seq, t, state, prev_state, status, trips, action_src, substituted, clamped_dims, action, f, z, valid, fired, s, tau, window_hits, brake_margin, reason, handoff_seq, violation_reached_env, prev} -- ONLY what a replay reproduces | YES, byte-identical | `verdict_chain_head`, `verdict_events` |
| `timing_chain` | `TimingEvent` = {seq, decide_ns, io_ns, prev} -- measured wall-clock | No, by design | `timing_chain_head`, `timing_events` |

`h_0 = ZERO_HASH`; `h_i = sha256(canon(event_i with hash=""))` where the event carries `prev = h_{i-1}`. Per tick: hashing only, on the I/O thread AFTER `decide()` returns. Per episode: ONE Ed25519 signature over `canon(body)`; `body_digest = sha256(canon(body))`. `lictor replay` asserts equality of the verdict head ONLY and prints, verbatim, `timing chain head varies (by design -- wall-clock is not replayed)` -- the exact analogue of bulla's "the outputs reproduce; the signed body does not".

**Trust model.** The fuse VERIFIES: `mode`, dims, horizon/exec_steps, the envelope digest, the calibration digest and its embodiment/policy bindings, its own binary hash, and every tick's schema, finiteness and continuity. Everything in `binding`, `run`, `budget.delay_steps`/`exec_mode`/`stitch`, `fault_injection`, `inputs` (host part) and `outcome.success` is the HOST'S DECLARATION, signed by proxy. A receipt defends against post-hoc edits by anyone without the signing key and against accidental corruption; it does not defend against the experimenter, who holds the key and can rebuild a ledger and re-sign it. External anchoring (roadmap 10) is what would change that. Below, "binds X" means "binds the host's declaration of X" unless X is in the verified list.

**What the receipt binds (the audit claim, enumerated):** `envelope` + `envelope_digest` (including `operators`, so WHO may re-arm is signed) and, through the calibration, `embodiment_digest`; `calibration_digest` (alpha, gate, K-of-N, n_calib, seed pool: a curve point cannot silently use another alpha or a calibration fitted on its own eval seeds); `run.policy` (the host's declaration of repo, revision, weights sha256, horizon/n_action_steps/n_obs_steps -- the overlap TCE assumed -- device, dtype, `normalization_migrated`; the fuse checks only that `weights_sha256` matches the calibration's `policy_digest`); `run.env` + `run.host` (every pin and thread/env var as declared; lerobot #4390-class hazards are pinned); `run.seed` + `run.init_state_digest` (the paired-seed evidence: arms whose digests disagree for an episode index cannot be aggregated); `budget` (delay, exec mode, stitch, tiers, on_escalate: the x-axis is signed); `fault_injection` (the host's declaration; a host that injects and declares `null` is not caught -- trust model); `client` (the harness client version); `counts`, `outcome`, `handoffs` (the numbers, every handoff with its ack); both chain heads (editing one tick in an unsigned 300-line ticks file breaks verification when the file is checked with `--ticks`); `mode`, `fuse_ok`, `fuse_notes`, `violations_reached_env` (intact != fuse_ok: a valid receipt can honestly attest the fuse was observing and 137 violations reached the environment); `ledger_prev`; `inputs` (host-declared harness file hashes plus the fuse-computed `lictor:bin`, `lictor:envelope`, `lictor:calibration`).

**Ledger** (`ledger.jsonl`, hash-chained per arm): dropping, reordering or editing an episode WITHOUT the signing key breaks the chain at a reported `seq`. `lictor curve` recomputes EVERY reported metric from the receipts, refuses on a broken ledger, on mixed or ephemeral keys, on an index set that differs from the baseline's or from the DECLARED pool (run.json + calibration `seed_pool`; `--partial` instead binds `n_declared/n_present/n_missing/missing_indices`), lists `pair_mismatches`, `cross_run_mismatches` and calibration/eval seed overlap in the signed `CurveReceipt`, and counts missing episodes as failures. Episodes that fault, crash or hit a fatal schema error still produce a receipt (`terminal_state: fault`, or a host-written `fuse_crash` receipt), so they cannot vanish from the curve. The harness's `index.jsonl` is a convenience index, never a source of truth. Honest gap: truncation of a declared pool is detected; a key-holder who also rewrites run.json is not; Rekor / RFC 3161 anchoring is roadmap.

---

## 9. Wire protocol, file formats, CLI surface

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

## FILE FORMATS (normative)

### Canonical encoding profile `jcs-floatfree/v1` (applies to every file below EXCEPT the envelope TOML, the wire trace and the harness index)

1. RFC 8785 JCS: object keys sorted by UTF-16 code-unit sequence, no whitespace, minimal escaping, integers rendered as plain digits.
2. Float-free: the JSON text contains NO non-integer numbers. A real number is `{"f64":"<16 lowercase hex of IEEE-754 bits>"}`; an array of reals is `{"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[...]}`. Integers must satisfy |v| <= 2^53-1.
3. Every key matches `[a-z0-9_]+`.

Consequence: Python's `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()` over `json.loads(file)` is BYTE-IDENTICAL to lictor's canonical bytes. That is what makes `adapters/verify_receipt.py` a stdlib-only verifier. Files on disk are written pretty-printed (2-space indent, keys sorted) for humans; the canonical bytes are always recomputed, never trusted from disk.

Digests: `sha256(canon(value))`, hex64. `body_digest` of a signed object = sha256 over canon(body). Ed25519 signature = over the same canonical bytes (`verify_strict`). Self-digesting objects (calibration.json, HandoffRecord) compute their digest over canon(self with the digest field set to "").

### Envelope TOML -- `envelopes/pusht.base.toml` (schema `lictor-envelope/v1`)

```toml
schema = "lictor-envelope/v1"
envelope_id = "pusht-base-v1"
# Base values are PLACEHOLDERS: loose enough not to touch the baseline. `lictor envelope fit` writes envelopes/pusht.toml.
box_lo = [15.0, 15.0]          # agent radius 15 px; env box is [0,512]^2
box_hi = [497.0, 497.0]
margin = 2.0
v_max = 1000.0                 # px/s   (100 px per 0.1 s control step)
a_max = 20000.0                # px/s^2
j_max = 400000.0               # px/s^3
reach_max = 150.0              # px, ||a_0 - p_t||
clamp_mode = "project"
tier0_enabled = ["workspace", "speed", "accel", "jerk", "reach", "brake"]   # "contact" is perception-assisted; off by default
gate = ["tce", "acc", "acm_neg", "njr", "reach", "path_ineff", "stall", "speed_peak"]   # plain max over 8 channels (calibration.json may override)
operators = []                 # hex64 Ed25519 pubkeys allowed to sign an AckToken. NEVER edited at run time by Python: the oracle arm
                               # uses envelopes/pusht.oracle.toml, produced by `lictor envelope fit ... --operator <hex>` (same embodiment,
                               # hence the same embodiment_digest and the same calibration; a different envelope_digest, printed by verify)
fail_closed = true

[embodiment]
id = "gym_pusht/PushT-v0"
units = "px"
action_kind = "ee_position"
action_dim = 2
pos_dim = 2
horizon = 15                   # n_action_steps = horizon(16) - n_obs_steps(2) + 1 = 15 (see ARCHITECTURE sec 0)
exec_steps = 8                 # stock n_action_steps -- the executed prefix is identical to published LeRobot
control_hz_num = 10
control_hz_den = 1
provides_vel = true            # gym_pusht info["vel_agent"] every step -> obs.vel; the PD brake rollout starts from the true v0
horizon_ticks = 300            # == max_episode_steps; the calibration time base (`lictor calibrate` refuses a mismatch)
norm_center = [256.0, 256.0]
norm_scale = [256.0, 256.0]
aux_layout = ["block_x", "block_y", "block_theta", "coverage"]
ext_names = []

[brake]
kind = "pd_second_order"       # gym-pusht agent: acc = k_p (a - p) - k_v v ; v += acc dt ; p += v dt ; 10 substeps of dt = 0.01
k_p = 100.0
k_v = 20.0
substeps = 10
dt = 0.01
commit_steps = 8
brake_steps = 8
react_ticks = 1

[hysteresis]
k = 3
n = 5
warn_margin = 0.5
clear_ticks = 5
clamp_streak_to_brake = 3
max_clamps_per_episode = 60
stop_confirm_ticks = 2
v_stop_eps = 5.0
brake_timeout_ticks = 30
rearm = "auto"
rearm_hold = 10
max_rearms = 2
escalate_after_hold_ticks = 30
handoff_timeout_ticks = 200
watchdog_ticks = 2

# [contact]  (optional, disarmed unless listed in tier0_enabled)
# radius = 80.0
# v_max = 400.0
# aux_center = [0, 1]

# [fit]  (written by `lictor envelope fit`)
# source_run = "2026-09-01T09-14Z-pilot"
# quantile = 0.999
# slack = 1.25
# n_episodes = 196
# fitted_utc = "2026-09-01T11:02:00Z"
# note = "p99.9 of |v|,|a|,|j|,reach over calibration successes x 1.25"
```
Envelope digest = sha256(canon(floatify(json(envelope)))) where `json(envelope)` is serde's JSON of `SafetyEnvelope` (struct field names as keys, TOML tables become nested objects, `contact`/`fit` absent -> `null`). Embodiment digest = the same over `json(envelope.embodiment)` alone; a calibration binds THAT, so `envelope fit` (which changes `v_max`..`reach_max`) and the oracle operator list do not invalidate it.

### calibration.json (schema `lictor-calibration/v1`, float-free)

```json
{"schema":"lictor-calibration/v1","canonical":"jcs-floatfree/v1",
 "method":"binned","alpha_num":5,"alpha_den":100,
 "n_total":196,"n_scale":137,"n_calib":137,"n_holdout":59,"split":"2way",
 "seed_pool":{"name":"calib","lo":900000,"hi":900299},
 "source_run":"2026-09-01T09-14Z-pilot","source_arm":"calib-obs",
 "envelope_digest":"7c4a...","embodiment_digest":"91d0...","policy_digest":"<weights sha256>","lictor_git":"a3f1c9e",
 "horizon_ticks":300,"t_grid":100,
 "feature_ids":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak","ext0","ext1","ext2","ext3"],
 "mask":255,
 "gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],
 "center":{"f64a":"...","shape":[100,12]},
 "scale":{"f64a":"...","shape":[100,12]},
 "tau":{"f64":"400db6db6db6db6e"},
 "kn":[3,5],
 "warn_margin":{"f64":"3fe0000000000000"},
 "holdout_fpr":{"f64":"3fa47ae147ae147b"},
 "holdout_fpr_k1":{"f64":"3fb0000000000000"},
 "tier0_percentiles":null,
 "notes":[],
 "digest":"<sha256 over canon(self with digest = \"\")>"}
```
`n_calib` is ALWAYS the number of per-episode scores `tau` was taken over (`k = ceil((n_calib + 1)(1 - alpha))`; a 2-way split of 196 successes gives 137, not 196); `n_scale` is the number of episodes center/scale were fitted on; `n_total` the successes available. `split` is `"2way"` (default: `n_scale == n_calib`, the bound is approximate because the standardisation was fitted on the same episodes) or `"3way"` (`--split 3`: standardise on A, `tau` on B, holdout on C; exact bound at K = 1, fewer scores per part). `tau` is `{"f64":"7ff0000000000000"}` (+inf) when `k > n_calib` -- the file records the degenerate point honestly. `holdout_fpr` is the empirical episode-level firing rate on the held-out subset with the configured K-of-N; `holdout_fpr_k1` with K=1 (the rate alpha actually bounds). `tier0_percentiles` is non-null only when produced together with `envelope fit` (a `Tier0Percentiles` object: `{"v_max":{"f64":..},"a_max":..,"j_max":..,"reach_max":..,"q":..}`). `embodiment_digest` MUST equal the served envelope's (`lictor serve`/`replay` refuse otherwise) and `policy_digest` MUST equal `binding.policy.weights_sha256` at `episode_begin` (fatal `error{code:"envelope"}` otherwise).

### Ticks file -- `results/<run>/<arm>/ticks/<episode:06>.jsonl` (schema `lictor-ticks/v1`)

First line: `{"schema":"lictor-ticks/v1","run_id":"...","arm_id":"...","episode_index":7,"genesis":"0000...0000"}`. Then one `TickEvent` per line, canonical (compact, sorted keys), e.g.:

```json
{"action":{"f64a":"AAAAAADAKkBmZmZmZsZyQA==","shape":[2]},"action_src":"policy","brake_margin":{"f64":"404499999999999a"},"clamped_dims":0,"f":{"f64a":"...","shape":[12]},"fired":0,"handoff_seq":null,"hash":"5b7e...","prev":"0000...","prev_state":"armed","reason":"ok","s":{"f64":"3ff199999999999a"},"seq":24,"state":"armed","status":"nominal","substituted":false,"t":24,"tau":{"f64":"400db6db6db6db6e"},"trips":0,"valid":255,"violation_reached_env":false,"window_hits":0,"z":{"f64a":"...","shape":[12]}}
```
`hash = sha256(canon(event with "hash":""))`; `prev` of seq 0 is the genesis (64 zeros). Companion `timing/<episode:06>.jsonl` holds `TimingEvent`s with the same rule.

### Receipt -- `results/<run>/<arm>/receipts/<episode:06>.json` (schema `lictor-receipt/v1`)

```json
{"body":{
   "schema":"lictor-receipt/v1","canonical":"jcs-floatfree/v1","created_epoch":1788327242,
   "lictor_version":"0.1.0","lictor_git":"a3f1c9e","lictor_sha256":"...","client":"lictor_client/0.1.0",
   "run":{"run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","episode_index":7,"seed":7,"seed_pool":"eval","init_state_digest":"3f1c...",
          "env":{"env_id":"gym_pusht/PushT-v0","gym_pusht":"<from importlib.metadata>","gymnasium":"<from importlib.metadata>","pymunk":"<from importlib.metadata>","numpy":"<from importlib.metadata>","obs_type":"pixels_agent_pos","control_hz":"10","max_episode_steps":"300","vel_source":"info.vel_agent","coverage_t0":"env.unwrapped._get_coverage()"},
          "policy":{"repo_id":"lerobot/diffusion_pusht","revision":"...","weights_sha256":"...","horizon":"16","n_action_steps":"15","n_obs_steps":"2","num_inference_steps":"100","device":"cuda","dtype":"float32","normalization_migrated":"true","migration_script":"lerobot/processor/migrate_policy_normalization.py"},
          "host":{"os":"<platform.platform()>","cpu":"...","gpu":"NVIDIA GeForce GTX 1060 3GB","torch":"2.7.1+cu126","lerobot":"0.6.1","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","PYTHONHASHSEED":"0","CUBLAS_WORKSPACE_CONFIG":":4096:8"}},
   "budget":{"mode":"enforce","delay_steps":0,"tick_ms":100,"exec_mode":"sync","stitch":"drop","on_escalate":"terminate_fail",
             "tier0_armed":["workspace","speed","accel","jerk","reach","brake"],"tier1_armed":true,
             "gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],"alpha_num":5,"alpha_den":100,"kn":[3,5]},
   "fault_injection":null,
   "envelope":{ "...floatified SafetyEnvelope..." },
   "envelope_digest":"7c4a...","calibration_digest":"e1b8...",
   "inputs":{"envelopes/pusht.toml":"...","harness/pusht_rollout.py":"...","harness/executor.py":"...","harness/compat.py":"...",
             "lictor:bin":"...","lictor:envelope":"...","lictor:calibration":"..."},
   "counts":{"ticks":300,"nominal":281,"watching":12,"clamped":0,"braking":4,"held":3,"escalated":0,"fault":0,"terminated":0,"substituted":7,
             "chunks_seen":38,"chunks_rejected":0,"clamps":0,"holds":1,"rearms":1,"escalations":0,
             "trips_by_bit":[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],"fired_by_feat":[3,1,0,0,0,9,11,0,0,0,0,0],
             "first_trip_tick":181,"first_trip_reason":"brake_tier1_cp","first_stop_tick":181,"handoff_tick":null,
             "violations_reached_env":0,"terminal_state":"armed"},
   "outcome":{"steps":300,"success":false,"terminated":false,"truncated":true,"max_coverage":{"f64":"3fe6b851eb851eb8"},
              "final_coverage":{"f64":"3fe5c28f5c28f5c3"},"reward_sum":{"f64":"4066466666666666"},"ended_by":"truncated",
              "max_s":{"f64":"4010a3d70a3d70a4"},"max_z":{"f64a":"...","shape":[12]}},
   "handoffs":[],
   "verdict_events":300,"verdict_chain_head":"5b7e...","timing_events":300,"timing_chain_head":"22a0...",
   "latency":{"n":300,"p50_ns":1180,"p90_ns":1400,"p99_ns":2600,"p999_ns":4400,"max_ns":61200,
              "label":"measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment"},
   "ticks_policy":"tail32","ticks":[ "...last 32 TickEvents..." ],
   "fuse_ok":true,"fuse_notes":[],
   "ledger_prev":"9c02..."},
 "body_digest":"be21...","pubkey":"64e8...","sig":"c4a1...128hex"}
```
`fuse_ok` rules (`evaluate_fuse`): false with a note when `mode == observe` ("the fuse observed but did not enforce -- this receipt does not attest protection"); when `violations_reached_env > 0`; when `tier1_armed` with no `calibration_digest`; when `terminal_state == fault` (including host-written `fuse_crash` receipts); when `delay_steps > 0` and `exec_mode` missing; when the signing key was ephemeral ("ephemeral signing key"). A valid, signed, chain-intact receipt can honestly say the fuse did nothing (intact != fuse_ok).

### Ledger -- `results/<run>/<arm>/ledger.jsonl` (schema `lictor-ledger/v1`)

First line `{"schema":"lictor-ledger/v1","run_id":"...","arm_id":"...","genesis":"0000...0000"}`, then one entry per episode:

```json
{"arm_id":"t01-a05-d0","episode_index":7,"escalated":false,"fuse_ok":true,"hash":"aa31...","init_state_digest":"3f1c...","prev":"9c02...","receipt_digest":"be21...","run_id":"2026-09-01T09-14Z-pilot","seed":7,"seq":7,"stopped":true,"success":false,"tripped":true,"verdict_chain_head":"5b7e..."}
```
Known gap, partially closed: a backward chain cannot by itself detect tail truncation. The eval pool is DECLARED (run.json `seeds`, calibration.json `seed_pool`), so `lictor curve` reads the declared pool from `run.json` + the arm's calibration (a `--calib-seeds` flag is only an override and is recorded), asserts the ledger's index set == the declared pool, and otherwise refuses -- or, under `--partial`, binds `partial: true, n_declared, n_present, n_missing, missing_indices` into the signed curve receipt. Truncation of a declared pool is therefore detected; truncation by the key-holder who also rewrites run.json is not (trust model). External anchoring (Rekor / RFC 3161) is roadmap.

### HandoffRecord and AckToken (schema `lictor-handoff/v1`)

```json
{"schema":"lictor-handoff/v1","run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0-oracle","episode_index":7,
 "seq":0,"tick":97,"reason":"escalate_hold_timeout","reason_text":"Held 30 ticks without a clean re-arm window; a human must decide.",
 "reasons":["tier1_cp","rearm_budget"],"trips":33792,"fired":96,"window_hits":4,
 "top_z":[["stall",{"f64":"4010a3d70a3d70a4"}],["path_ineff",{"f64":"400a666666666666"}],["tce",{"f64":"3ff8000000000000"}]],
 "chain_at":"5b7e...","envelope_digest":"7c4a...","calibration_digest":"e1b8...",
 "digest":"9a41...","ack":null,"resolved_tick":null,"outcome":"unacked"}
```
```json
{"schema":"lictor-handoff/v1","handoff_digest":"9a41...","decision":"resume","operator":"ab12...64hex","nonce":7,"note":"oracle resume","sig":"c4...128hex"}
```
Signed bytes = canon({"schema","handoff_digest","decision","operator","nonce","note"}). `run_id`/`arm_id`/`episode_index` are digested, so an ack captured on one run/arm/episode does not verify on a paired re-run of the same seed; per-operator `last_nonce` is kept across episodes in the Session and persisted in `<out>/.lictor/verifier_nonce.json`. `note` and `reason_text` are stripped of control characters before any terminal rendering.

### Curve receipt -- `results/<run>/curve/<arm>.json` (schema `lictor-curve/v1`) + `results/<run>/curve/summary.csv`

Body per `CurveReceiptBody`; the CSV has one row per arm with columns: `pilot,arm_id,n,n_missing,partial,delay_steps,exec_mode,alpha,detector,success_rate,success_lo,success_hi,delta_vs_baseline,delta_vs_baseline_lo,delta_vs_baseline_hi,mcnemar_p,mcnemar_b,mcnemar_c,latency_control_arm,delta_vs_latency_control,delta_vs_latency_control_lo,delta_vs_latency_control_hi,mcnemar_p_lc,mcnemar_b_lc,mcnemar_c_lc,averted_rate,flagged_rate,flagged_lo,flagged_hi,false_trip_rate,false_trip_lo,false_trip_hi,intervention_rate,intervention_tick_frac,escalation_rate,lead_mean,lead_p50,lead_p10,aucpdt,roc_auc,bacc,violations_reached_env,tce_valid_frac,latency_p50_ns,latency_p99_ns,latency_max_ns,latency_label,ledger_chain_ok,pair_mismatches,cross_run_mismatches,receipt_pubkey`. `pilot` is `1` when `small_n` (n < 100) and `0` otherwise; `figures.py` hollows pilot points and never draws a line through them.

### Sweep -- `results/<run>/sweep.jsonl` (Layer A, one line per (alpha, detector))

```json
{"alpha_num":5,"alpha_den":100,"detector":"t01","gate":["tce","acc","acm_neg","njr","reach","path_ineff","stall","speed_peak"],"k":3,"n":5,"tier0":true,"method":"binned","tau":3.7142857142857144,"tau_source":"artefact","n_calib":137,"n_fail":35,"n_succ":65,"tpr":0.6,"tpr_ci":[0.42,0.76],"fpr":0.06,"fpr_ci":[0.02,0.15],"lead_mean":41.2,"lead_p50":37.0,"lead_p10":4.0,"aucpdt":0.51,"roc_auc":0.71,"bacc":0.77,"fire_frac_mean":0.08,"holdout_fpr":0.05,"holdout_fpr_k1":0.07,"eps_prog":0.02}
```
(Plain JSON floats: sweep.jsonl is an analysis product, not a signed artefact. `tau` is `null` when `+inf` (degenerate alpha). `tau_source` is `"artefact"` when the point reused a `calibration.<alpha>.json` produced by `lictor calibrate` -- the Layer-A-vs-Layer-B validity comparison is made ONLY on such points -- and `"fit"` when `sweep` fitted a full-sample tau itself.)

### Harness index -- `results/<run>/<arm>/index.jsonl` (convenience only; rebuilt from the ledger at startup)

```json
{"run_id":"2026-09-01T09-14Z-pilot","arm_id":"t01-a05-d0","episode_index":7,"seed":7,"seed_pool":"eval","init_state_digest":"3f1c...","success":false,"steps":300,"ended_by":"truncated","max_coverage":0.71,"final_coverage":0.68,"receipt":"receipts/000007.json","receipt_digest":"be21...","first_stop_tick":181,"tripped":true,"stopped":true,"escalated":false,"policy_action_equal_ticks":293,"wall_s":18.4,"ts":"2026-09-01T09:41:02Z"}
```

### Trace -- `results/<run>/<arm>/traces/<episode:06>.ndjson`

`#meta {...}` then every request line verbatim (wire format, plain floats). Replay fixture and calibration source.

### Results tree

```
results/<run_id>/
  run.json                      # arms, seeds, pins, commands, lictor build (harness-written)
  sweep.jsonl                   # Layer A
  curve/<arm>.json  summary.csv # Layer B, from receipts
  calibration.<alpha>.json      # per alpha
  <arm_id>/
    ledger.jsonl  index.jsonl
    receipts/NNNNNN.json  ticks/NNNNNN.jsonl  timing/NNNNNN.jsonl  traces/NNNNNN.ndjson
  .lictor/verifier_nonce.json   # per-pubkey last ack nonce (under --out)
$LICTOR_KEYS/ (default $HOME/.lictor/):  key.hex  operator.hex  history.jsonl   # keys NEVER live in the repo or under /mnt/[a-z]/ (DrvFs ignores 0600)
```
Heavy results live on D: (`LICTOR_RESULTS=/mnt/d/lictor/results`, symlinked or passed via `--out`); the repo commits only `bench/fixtures/`, `docs/figures/`, the fitted `envelopes/pusht.toml` + `envelopes/pusht.oracle.toml`, `docs/envelope_fit_report.md` and the `research/*.md` memos (all generated-in-repo files are owned by WP-13).

## CLI SURFACE (normative; clap derive; every command accepts `--json`)

```
lictor serve      --envelope <F.toml> [--calibration <F.json>] [--mode observe|enforce]
                  [--key <F.hex>] [--out <DIR>] [--trace <F.ndjson>] [--ticks tail32|all|none]
                  [--tier0 <csv of trip names>] [--no-tier1] [--on-fault hold|abort] [--latency-label <S>]
                    NDJSON server on stdin/stdout (lictor-wire/v1). Exit 0 on bye, 3 on internal error.
                    Refuses at startup (exit 2) a calibration whose embodiment_digest != the envelope's. Without --key an ephemeral
                    key is generated and every receipt carries fuse_ok=false + note "ephemeral signing key" (lictor curve refuses such arms).
                    --on-fault abort is a debugging aid only. --latency-label defaults to default_latency_label() (WSL2 sentence here).
lictor verify     <RECEIPT.json> [--ticks <F.jsonl>] [--timing <F.jsonl>] [--ledger <F.jsonl>] [--curve <C.json>]
                  [--calibration <F.json>] [--pubkey <hex>] [--json]
                    prints the aligned block below; exit 0 intact AND fuse held, 1 otherwise. Also recomputes envelope_digest from the
                    embedded body.envelope, checks counts.ticks == verdict_events == timing_events == outcome.steps, and with --ticks
                    cross-checks the ticks header (run_id, arm_id, episode_index, genesis) and the embedded tail's first `prev`
                    against the file, and compares the recomputed head with the signed head (HEAD MISMATCH). With --calibration it
                    checks budget.alpha/gate/kn against the file. With --pubkey a mismatch is `pubkey        MISMATCH` and intact NO.
                    Prints `WARNING: signed with the committed test key` when the pubkey is one of TEST_PUBKEYS.
lictor replay     <TRACE.ndjson> --envelope <F> [--calibration <F>] [--mode observe|enforce] [--repeat N] [--expect <hex64>]
                    "replays N/N byte-identical  verdict_chain=<hex>" ; "timing chain head varies (by design -- wall-clock is not replayed)"
lictor calibrate  --run <DIR> --arm <ARM> --envelope <F> --alpha <num/den>[,...] [--method static|binned]
                  [--gate <csv terms>] [--kn 3,5] [--holdout 3/10] [--split 2|3] [--n-calib N] [--warn-margin 0.5] -o <DIR>
                    writes <DIR>/calibration.<alpha>.json per alpha (float-free, self-digested). Refuses eval-pool seeds, and a
                    horizon_ticks that differs from envelope.embodiment.horizon_ticks or from run.env.max_episode_steps.
lictor sweep      --run <DIR> --calib-arm <ARM> --eval-arm <ARM> --alphas <csv num/den> [--calibration-dir <DIR>]
                  --detectors t0,t1_tce,t1_stall,t1_full,t01,t01_and [--method binned] [--eps-prog 0.02] -o <F.jsonl>
                    with --calibration-dir, points whose (alpha, gate, kn, method) match a calibration.<alpha>.json reuse its tau verbatim.
lictor curve      --run <DIR> --baseline <ARM> [--arms a,b,c] [--latency-control <ARM>] [--eps-prog 0.02] [--calib-seeds 900000-900299]
                  [--allow-small] [--partial] [--compare-run <DIR>] [--key <F.hex>] -o <DIR>
                    recomputes every metric FROM RECEIPTS; refuses on a broken ledger, on mixed/ephemeral receipt keys, on a
                    tier1/alpha disagreement with run.json, on an index set != the baseline's or != the declared pool (unless
                    --partial), and on n < 100 (unless --allow-small; then small_n=true and the CSV row starts with pilot=1).
                    --latency-control defaults to obs-d<d> derived from the arm's delay_steps when that arm exists in the run.
                    Missing indices count as success=false, stopped=true. Writes <arm>.json (signed) + summary.csv.
lictor envelope   init  --profile pusht -o <F.toml> [--operator <hex>]...
                  check <F.toml>                       -> validate + digest + embodiment digest
                  digest <F.toml>                      -> "<envelope digest>" (and "embodiment=<hex>" on stderr)
                  show <F.toml>
                  fit   --run <DIR> --arm <ARM> --base <F.toml> [--quantile 0.999] [--slack 1.25] [--operator <hex>]... -o <F.toml> [--report <F.md>]
                    --operator writes the oracle envelope (identical embodiment, hence identical embodiment_digest and calibration).
lictor bench      [--n 200000] [--trace <F.ndjson>] [--tier t0|t0t1] [--csv <F>] [--json]
                    "verdict  p50 X us  p99 X us  p99.9 X us  p99.99 X us  max X us  (n=N)" + per-tier lines + "allocations   0"
                    the allocation count comes from the always-installed gated counting allocator (alloc_count.rs); the "arena"
                    figure is size_of::<FuseRt>() + size_of::<FuseConfig>() printed at run time, never a constant.
lictor selftest   [--json]    -> one line per check, final "SELFTEST PASS" | "SELFTEST FAIL"
lictor ledger     verify <F.jsonl> | append --ledger <F.jsonl> --receipt <R.json>
lictor crash-receipt --envelope <F.toml> [--calibration <F.json>] --mode observe|enforce --run-id <S> --arm-id <S> --episode-index <N>
                  --seed <N> --seed-pool <S> --init-state-digest <hex> --budget <json> [--key <F.hex>] --out <DIR> [--note <S>]
                    host-side accounting for a serve child that died before episode_end: writes a signed receipt with
                    ended_by="fuse_crash", success=false, terminal_state=fault, fuse_ok=false and appends the ledger. Exit 0.
lictor key        init [-o <F.hex>] [--role signer|operator] [--force] [--i-know] | pub [--key <F.hex>]
                    default path $LICTOR_KEYS/key.hex (or $HOME/.lictor/key.hex); refuses to overwrite without --force; refuses a
                    path under /mnt/[a-z]/ without --i-know; prints "pubkey <hex>", "stored <path> (mode 0600)" and one custody line.
lictor ack        --handoff <digest|record.json> --decision resume|abort|retune --key <operator.hex> [--note S] [--nonce N] [-o <F.json>]
lictor history    [--dir <DIR>] [--n 20]
lictor version    -> "lictor 0.1.0 (<git>) sha256=<binary sha>"
```

Exit codes: `0` ok; `1` verification failed / fuse not held / determinism mismatch; `2` usage; `3` internal. NOTE for scripts: `lictor verify` exits 1 on an honest Observe receipt (intact but not enforced), so CI and bench scripts MUST use `--json` and test the `intact` field rather than the exit code when they mean "is the record intact".

Reference output (frozen strings -- CI greps them; do not reword):

```
$ lictor verify results/r/t01-a05-d0/receipts/000007.json --ticks results/r/t01-a05-d0/ticks/000007.jsonl --pubkey 64e8...
  schema         ok    lictor-receipt/v1
  signature      ok    (key 64e8281d05cf...)
  pubkey         ok    (matches --pubkey)
  body digest    ok    be21...
  envelope       ok    digest recomputed 7c4a...  embodiment 91d0...
  counts         ok    ticks=300 verdict_events=300 timing_events=300 steps=300
  verdict chain  ok    head=5b7e...  (300 ticks)
  timing chain   ok    head=22a0...  (not replayable -- by design)
  intact         YES
  FUSE HELD      (7 substitutions, 0 violations reached the environment)
  bindings       envelope=7c4a... calibration=e1b8...(alpha=5/100,n_calib=137) policy=lerobot/diffusion_pusht@a3f1...
                 seeds pool=eval seed=7 init_state=3f1c...  client=lictor_client/0.1.0

$ lictor verify r.json --ticks rechained.jsonl          # attacker recomputed hashes 17..299: the chain links, the head does not
  verdict chain  HEAD MISMATCH recomputed=9d02... signed=5b7e...
  intact         NO

$ lictor verify r.json --pubkey 0000...
  pubkey         MISMATCH  (receipt 64e8281d05cf..., expected 0000...)
  intact         NO

$ lictor verify crates/lictor-receipt/tests/fixtures/receipt/receipt_000007.json
  WARNING: signed with the committed test key

$ lictor verify observe.json
  ...
  intact         YES
  FUSE NOT ENFORCED
  notes:  - the fuse observed but did not enforce -- this receipt does not attest protection

$ lictor verify tampered.json
  schema         ok    lictor-receipt/v1
  signature      FAIL  (key 64e8281d05cf...)
  body digest    FAIL  be21...
  intact         NO
  notes:  - Ed25519 signature does not verify against the body

$ lictor verify r.json --ticks edited.jsonl
  verdict chain  BROKEN at seq=17

$ lictor ledger verify results/r/t01-a05-d0/ledger.jsonl
  episodes 500   chain ok    success 341/500 (68.2%)   stops 61   escalated 22   fuse_ok 500/500
$ ... after deleting 43 entries ...
  ledger BROKEN at seq=3 -- entries are missing or edited; this curve point cannot be trusted

$ lictor replay --repeat 40 bench/fixtures/traces/pusht_000007.ndjson --envelope envelopes/pusht.toml --calibration bench/fixtures/calibration.a05.json --mode enforce
  ticks=300  repeats=40
  replays 40/40 byte-identical  verdict_chain=5b7e...
  timing chain head varies (by design -- wall-clock is not replayed)
  states nominal=281 watching=12 clamped=0 braking=4 held=3 escalated=0 fault=0
  RESULT  DETERMINISTIC

$ lictor bench --n 200000                              # X = measured value; CI greps the SHAPE of these lines, never the numbers
  tier=t0t1  horizon=15  exec=8  dim=2  n=200000
  verdict  p50 X us  p99 X us  p99.9 X us  p99.99 X us  max X us  (n=200000)
  tier0    p50 X us  p99 X us
  brake    p50 X us  p99 X us
  tier1    p50 X us  p99 X us
  allocations   0  (state preallocated X KiB = size_of FuseRt + FuseConfig)
  environment   WSL2 (Hyper-V utility VM, non-RT host) -- NOT a real-time measurement

$ lictor curve --run results/r --baseline obs-d0 --arms t01-a05-d0 -o results/r/curve
  t01-a05-d0  n=500  success 0.xxx [0.xxx,0.xxx]  delta_vs_obs-d0 +0.xxx [-0.xxx,+0.xxx] p=0.xx  delta_vs_obs-d0(lc) ...  tce_valid 1.000  pubkey 64e8...
$ lictor curve --run results/r --baseline obs-d0 --arms t01-a05-d3 -o results/r/curve
  t01-a05-d3  ... REFUSED: episode index set differs from baseline (missing 3 indices: 17,211,340); re-run them or pass --partial
```

## 10. The experiment protocol -- the safety-vs-latency curve on PushT

### 10.1 Day-0 gate (`harness/microbench.py`; nothing else runs until it prints PASS)

| Probe | Pass condition | If it fails |
|---|---|---|
| `import lerobot.policies` in the venv (the cv2 circular-import failure seen on 2026-08-31 came from `opencv-python` + `opencv-python-headless` installed together; only `opencv-python-headless` may be present, pinned in `requirements.txt`) | imports in < 60 s | `smoke/fix_cv2.sh`; stop until green |
| lerobot 0.6.1 loads `lerobot/diffusion_pusht` from the MIGRATED path (`lerobot/processor/migrate_policy_normalization.py` -> `$LICTOR_MODELS/diffusion_pusht_migrated`, already present on this machine; `compat.py` names the script, never searches for it); config prints `horizon=16 n_action_steps=8 n_obs_steps=2` | prints the chunk spec + `weights_sha256` | stop; fix the environment |
| `n_action_steps=15` override: `test_h15` (the six runtime asserts of sec 0) | first 8 of 15 == stock 8, bit-exact, through the offline `predict_action_chunk` path | fallback: call `policy.diffusion.conditional_sample()` directly (documented coupling); never change `num_inference_steps` |
| seconds per episode, observe mode, 10 episodes, single env B=1 (GPU fp32; then CPU with `OMP_NUM_THREADS=1`) | recorded in `microbench.json` with an `environment` field; `run.py --plan` uses it and REFUSES any step whose projection exceeds `--budget-min` | if > 25 s/ep on GPU: cut the pilot to the three arms of 10.8 at 40 seeds; CPU is documented as unusable for any pilot arm (single-threaded, minutes/episode) |
| 1 vs 2 GPU workers on the 10-episode smoke (fp32 UNet ~1 GB + ~300 MB CUDA context each on 3 GB is marginal; time-slicing one launch-bound GPU gives at most ~1.5x) | measured episodes/min for both | `run.py --plan` picks the faster; never assume 2 |
| GPU determinism: `obs-d0` twice on seeds 0, 1, 2 | byte-identical action sequences on all three | fall back to CPU inference for headline arms (McNemar is invalid without pairing) |
| observe-mode identity: `obs-d0` vs a no-lictor rollout, seed 0..2 | byte-identical action sequences | bug in the harness/fuse -- stop |
| render backend | `MUJOCO_GL`-style probe irrelevant for PushT (pygame/pymunk), but `SDL_VIDEODRIVER=dummy` must render `rgb_array` headless | pin the working setting into `run.host` |
| `lictor bench --n 200000` | `allocations   0`; p99 <= 50 us | fix before proceeding |
| `lictor replay --repeat 40` on `bench/fixtures/traces/pusht_000007.ndjson` | `replays 40/40 byte-identical` | fix before proceeding |
| 65.4 % reproduction (smoke check): `obs-d0` x 60 (pilot; CI about +-12 pp, so it accepts roughly 53-77 %) -- the n=500 `obs-d0` run is the real reproduction; gym-pusht 0.1.6 / gymnasium 1.3.0 / pymunk 6.11.1 are NEWER than the pins the number was published with, so this gate stays HARD | 95 % Clopper-Pearson CI covers 0.654 | investigate the pins before trusting any curve; report what WE measure either way |

Reference: `eval_ep_s = 1.46` on the reference GPU is lerobot's BATCHED eval number (~50 envs per batch), not a single-env figure. ~37 chunks x 100 DDPM steps ~ 3 700 launch-bound UNet forwards per episode at B = 1; on a GTX 1060 3GB fp32 the honest expectation is 10-20 s/episode (i.e. 500 episodes ~ 1.5-3 h per arm), and CPU with the required `OMP_NUM_THREADS=1` is minutes per episode -- UNVERIFIED until measured. Batching across envs is the only large throughput lever and it is INCOMPATIBLE with per-episode `torch.manual_seed(hash64(seed, chunk))` (10.6); it is not an option. Every number below scales off the measurement; `harness/run.py --plan` prints the projected wall-clock before running anything and refuses a step over `--budget-min`.

### 10.2 Two-layer design (design 1) -- what makes today feasible

**Layer A -- open-loop detection sweep. Free, CPU, deterministic, seconds.** Observe-mode runs never change the trajectory, so one set of observe rollouts yields the complete raw-feature trace of every episode. `lictor sweep` recomputes TPR / FPR / lead time / ROC-AUC / AUCPDT for EVERY alpha and EVERY detector subset offline, with no re-rollout. This is exactly how the detector literature evaluates, and it gives the full detection Pareto from a single GPU pass.

**Layer B -- closed-loop arms. Expensive, GPU.** Only a handful of `(detector, alpha, d, exec_mode)` points run with `mode=enforce`, to measure what Layer A structurally cannot: task-success delta (Y2) and intervention rate (Y3), including the failures the fuse itself causes. Closed-loop TPR vs the Layer-A prediction at the same alpha is reported as a validity test of Layer A. All Layer-B numbers come from `lictor curve` over signed receipts.

### 10.3 Envelope fitting and the parity gate (design 2) -- the step that decides whether Tier 0 is real or decorative

1. Run the calibration pool in observe mode with the BASE envelope (`calib-obs`, 100 episodes in the pilot, 300 overnight).
2. `lictor envelope fit --quantile 0.999 --slack 1.25` -> `envelopes/pusht.toml` + `docs/envelope_fit_report.md` (empirical distributions of |v|, |a|, |j|, reach over successes), and once more with `--operator <demo pubkey>` -> `envelopes/pusht.oracle.toml` (identical embodiment). The calibration fitted in step 1 stays valid under both: Tier-1 features depend on the manifest only, and `serve` binds the calibration to the `embodiment_digest`.
3. Parity gate: `t0-d0` vs `obs-d0` on the first 60 paired eval seeds. Require McNemar exact `p > 0.05` and overlapping Clopper-Pearson CIs. If the fitted envelope costs measurable success it is too tight: increase slack, refit, record every refit in the report. No curve point is reported until parity passes.

### 10.4 How "added latency" is realised in sim (both designs; RTC / arXiv 2605.08168 methodology)

x-axis = `d` control steps of injected staleness, converted to ms for display (PushT: 100 ms/step; a second top axis at a VLA-realistic 20 ms clock, captioned as a re-scaling, not a measurement). This makes the curve independent of WSL2 wall-clock -- the only way an honest curve is measurable on this machine.

- `sync`: a chunk requested at step t becomes available at t+d; for those d steps the env is stepped with the last executed action repeated (hold-last); the delay consumes the 300-step budget. Wire labelling at delivery: `t_emit = t`, `idx = 0`, the full 15 rows. Consequence, disclosed: consecutive deliveries are 8+d steps apart, so the Tier-1 overlap is `L = 7 - d` -- `tce`/`acc` are invalid for `d >= 7` and degraded for `d in 3..6`; the curve partly measures feature loss there and says so (`tce_valid_frac` on F1).
- `async`: the executor keeps draining the previous chunk (7 spare actions beyond the 8 executed, so d <= 7 is fully coverable); on arrival `stitch=drop` skips the new chunk's first d rows (they refer to already-executed steps) -- the fuse still receives all 15 rows with `t_emit = t - d`, `idx = d`, and brake feasibility starts from row d; `stitch=freeze` executes them (RTC's freeze) with `t_emit = t`, `idx = 0`; beyond the spare actions, hold-last. Async keeps `L = 7`.
- The fuse sees EVERY tick either way (`chunk:null` on non-boundary ticks). The fuse's own measured cost enters as `d_total = d + ceil(decide_ms / tick_ms)` = `d` at 10 Hz for all tiers -- honestly: at a 100 ms control period the measured decide + IPC cost rounds to zero additional control steps (a statement about simulated time, not about wall-clock guarantees); the budget axis is a what-if for higher-rate embodiments; Tier 2 is where it will bite. The measured `decide_ns` histogram is overlaid on F1 so readers see what the fuse costs versus what the budget buys.
- Fault injection (`harness/inject.py`, independent `Philox(episode_seed ^ 0xFA17)` stream, always bound into `fault_injection`): `action_spike(p, mag)`, `obs_dropout(p)`, `obs_noise(sigma)`, `chunk_truncate(p)`, `latency_spike(p, ticks)`.

### 10.5 Arms (`harness/arms.py`) -- `arm_id = <detector>-a<alpha%>-d<d>[-async][-<escalation>]`; every arm uses identical paired seeds

| arm_id | mode | tier0 | tier1 alpha | d | exec | on_escalate | role |
|---|---|---|---|---|---|---|---|
| `calib-obs` | observe | scored | scored | 0 | sync | -- | calibration pool only (seeds 900000..900299) |
| `obs-d0` | observe | scored | scored | 0 | sync | -- | FUSE-OFF BASELINE + labels + Layer-A eval traces |
| `obs-d{1,2,3,5,8}` | observe | scored | scored | d | sync | -- | LATENCY-ONLY CONTROLS: separate "latency hurt me" from "the fuse hurt me" |
| `obs-d2-async`, `obs-d5-async` | observe | scored | scored | 2,5 | async | -- | RTC comparison, fuse-off |
| `t0-d0` | enforce | armed | -- | 0 | sync | terminate_fail | geometric enforcement alone; parity gate |
| `t01-a05-d0` | enforce | armed | 5/100 | 0 | sync | terminate_fail | HEADLINE |
| `t01-a01-d0`, `t01-a10-d0`, `t01-a20-d0` | enforce | armed | 1,10,20 /100 | 0 | sync | terminate_fail | closed-loop alpha sweep |
| `t01-a05-d{1,2,3,5,8}` | enforce | armed | 5/100 | d | sync | terminate_fail | THE BUDGET CURVE |
| `t01-a05-d2-async`, `t01-a05-d5-async` | enforce | armed | 5/100 | 2,5 | async | terminate_fail | second curve |
| `t01-a05-d0-oracle` | enforce (`envelopes/pusht.oracle.toml`: same embodiment, demo operator key) | armed | 5/100 | 0 | sync | oracle_resume | intervention-value counterfactual; exercises the AckToken path |
| `t01-a05-d0-ackonly` | enforce (rearm=ack_only) | armed | 5/100 | 0 | sync | terminate_fail | protective-stop-vs-e-stop accounting (optional) |
| `inj-obs-d0`, `inj-t0-d0` | observe / enforce | scored / armed | -- | 0 | sync | terminate_fail | `action_spike(p=0.02, mag=180)`: THE ENFORCEMENT DEMONSTRATION (expect ~100+ vs 0 violations reaching the env) |

Every `Arm` declares `tier1: bool` and `alpha: (num, den) | None`; `LictorClient.hello` asserts `hello_ok.tier1_armed` and the calibration's alpha against them, and `lictor curve` refuses an arm whose receipts' `budget.tier1_armed`/alpha disagree with `run.json` -- a `t01-*` arm can never silently run as `t0`.

Detector subsets for `lictor sweep`: `t0` (Tier-0 only), `t1_tce` ([tce]), `t1_stall` ([stall, path_ineff]), `t1_full` (all 8), `t01` (Tier 0 + all 8), `t01_and` ([tce|acm_neg, acc|path_ineff, stall|path_ineff]).

### 10.6 Seeds, pools, pairing, determinism

- calib pool `900000..900299`; eval pool `0..499`; pilot `0..99`. Provably disjoint; asserted by `run.py --check-pools`, `lictor calibrate` (refuses eval seeds), `lictor curve` (lists overlap in the signed curve receipt).
- Paired across every arm: `env.reset(seed=s)` + `torch.manual_seed(hash64(s, chunk_idx))` with `hash64(seed, chunk) = splitmix64(((seed << 32) ^ chunk) mod 2^64)` (Python ints are unbounded, so every step is reduced mod 2^64; `RunBinding.seed` is u64; test vectors: `hash64(0,0) = 0xe220a8397b1dcdaf`, `hash64(7,3) = 0xfd323448a4497c68`, `hash64(900000,37) = 0xaf19a08d78d230a5`); `init_state_digest` bound into every receipt and ledger entry; arms disagreeing on an episode index are refused at aggregation. Paired = same reset state + same per-chunk noise; trajectories diverge after the first substitution by design. Since `obs-d0` runs in both the pilot (0..59) and the full run (0..499), `run.py --summary` and `lictor curve --compare-run` compare `verdict_chain_head` for every (arm, seed) shared between runs and report `cross_run_mismatches` -- a free multi-seed policy-determinism check. One episode per env, never batched (see 10.1).
- Determinism env (`harness/env.sh`): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 SDL_VIDEODRIVER=dummy CARGO_TARGET_DIR=/mnt/d/lictor/target LICTOR_RESULTS=/mnt/d/lictor/results`; `torch.use_deterministic_algorithms(True, warn_only=True)`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, fp32, `policy.eval()`; exact pins `lerobot==0.6.1`, `torch==2.7.1+cu126`, `gym-pusht`, `gymnasium`, `pymunk`, `numpy`, checkpoint revision sha -- all bound into every receipt.
- Resumability: `results/<run>/<arm>/ledger.jsonl` is the source of truth; `index.jsonl` is rebuilt from it at startup; the ledger append is the LAST write per episode; `--resume` is the default; `--fresh` refuses a non-empty directory; arm-major, seed-ascending order so a truncated run is always a complete prefix of paired seeds.

### 10.7 Metrics -- three lines per budget point, or the curve is dishonest (computed by `lictor curve` from receipts)

Let `F` / `S` = eval seeds that fail / succeed in `obs-d0`.

| id | metric | definition |
|---|---|---|
| Y1a `flagged_rate` (TPR) | of `F`, fraction where the arm reached `Braking`/`Held`/`Escalated` (`first_stop_tick` non-null) | episode-level recall |
| Y1a' `averted_rate` | of `F`, fraction the arm converted to SUCCESS | reported separately -- "flagged" and "averted" are different goods |
| Y1b `false_trip_rate` (FPR) | of `S`, fraction where the arm stopped | the empirical rate alpha is supposed to bound |
| Y1c `lead` | `t_fail - first_stop_tick` over `F`, with `t_fail = min{t : c*_T - c*_t < eps_prog}`, `c*_t = max_{u<=t} coverage_u`, `eps_prog = 0.02` from the paired `obs-d0` trace (point-of-no-return PROXY, labelled as such everywhere; sensitivity over eps in {0.01, 0.02, 0.05}); mean, p50, p10; AUCPDT (VLA-FAIL); F1-timeliness hypervolume (ActProbe) | detection timeliness |
| Y2 `success_rate` | under `terminate_fail`, vs `obs-d0` (fuse cost, `delta_vs_baseline`) AND vs `obs-d{d}` (latency-only control at the same d, `delta_vs_latency_control`) -- always both deltas, each with a paired-bootstrap CI and exact McNemar, computed by `lictor curve` and written to `summary.csv` (analyze.py only re-checks them) | captures latency-induced NEW failures |
| Y3 `intervention_rate`, `intervention_tick_frac`, `escalation_rate` | episodes with >= 1 substitution; substituted ticks / ticks; episodes reaching Escalated (Sirius definition) | intervention cost |
| M8 `violations_reached_env` | Tier-0 violations not substituted (observe / injection arms) | the enforcement pillar's own number |
| M9-M10 `roc_auc`, `bacc` | over per-episode `max_s` vs the `obs-d0` label; balanced accuracy at tau | field-standard comparability |
| M11 `latency` | p50/p90/p99/p99.9/max of `decide_ns` and `io_ns` from timing chains, WSL2-labelled | what the fuse actually costs |
| M12 `delta_vs_*` | exact McNemar on paired success(baseline) vs success(arm) and a paired bootstrap CI (10 000 resamples, splitmix64 seed 20260830) -- both inside `DeltaCi`, in the signed curve receipt | validity |
| M13 `tce_valid_frac` | fraction of ticks with a valid `tce`; sync `d >= 7` -> 0 | the feature-loss confound, disclosed per point |

Statistics: Clopper-Pearson 95 % CIs on every rate (n=60 -> about +-12 pp; n=100 -> about +-10 pp; n=500 -> about +-4 pp; all stated). `lictor curve` refuses n < 100 without `--allow-small`; with it the point is bound `small_n: true`, the CSV row starts with `pilot=1`, and `figures.py` draws it hollow and never runs a line through it. No headline number ever comes from a pilot point. Every latency number carries `latency_label` (the WSL2 sentence). Expect and report the field's ~72-88 % detection ceiling.

### 10.8 Schedule (pilot today, full overnight), gated on measured s/episode

The one-hour pilot is 220 episodes -- nothing more fits at the honest 12-20 s/ep single-env rate on the 1060 (the earlier 970-episode plan was a 3-5x overrun and is gone). `run.py --plan --budget-min 60` refuses to start any step whose projection from `microbench.json` exceeds the budget.

| # | step | episodes | at 12 s/ep | at 20 s/ep | deliverable |
|---|---|---|---|---|---|
| 0 | day-0 gate + smoke `obs-d0` x 10 + `lictor bench` + the 1-vs-2-worker probe | 10 | 3 min | 4 min | plumbing green, s/episode measured, F5 (latency histogram) -- LANDS FIRST |
| 1 | `calib-obs` x 100 (seeds 900000..900099) | 100 | 20 min | 33 min | `envelope fit` (+ oracle toml), `calibration.<alpha>.json` for alpha in {5,10,20}/100 (~65 successes -> n_calib ~45: alpha 1/100 and 2/100 are degenerate at this n and are marked so), F7, F8 |
| 2 | `obs-d0` x 60 (seeds 0..59) | 60 | 12 min | 20 min | Layer-A eval traces; 65.4 % smoke check; the fuse-off baseline for the pilot point |
| 3 | `lictor sweep` (every alpha x detector) | 0 | seconds | seconds | F2 (ROC/Pareto), F3 (lead time), F4 (score bands) -- the science lands here |
| 4 | `t01-a05-d0` x 60 (seeds 0..59) | 60 | 12 min | 20 min | the ONE closed-loop pilot point (`--allow-small`, hollow on F1), F6 (McNemar vs `obs-d0`) |
| 5 | `lictor replay --repeat 40`, tamper vector, alloc test, `verify_receipt.py` | 0 | 2 min | 2 min | determinism + tamper evidence for the README |

Pilot total ~50 min at 12 s/ep, ~80 min at 20 s/ep (the parity gate and everything else move to the overnight run; the pilot point is labelled pilot everywhere). Overnight, in priority order and each gated by `--plan`: `calib-obs` to 300 (refit + recalibrate at every alpha) -> `obs-d0` x 500 (the real 65.4 % reproduction) -> `t0-d0` x 60 parity gate (refit if it fails) -> `t01-a05-d0` x 500 -> `obs-d2`, `t01-a05-d2` -> `t01-a05-d{1,3,5,8}` + `obs-d{1,3,5,8}` -> alpha sweep -> async pairs -> oracle -> injection arms. At 12-20 s/ep one arm of 500 is 1.7-2.8 h, so night 1 realistically covers `obs-d0`, `t01-a05-d0`, `t0-d0`(60) and the d=2 pair; the rest is night 2 and 3. Workers: whichever of 1 or 2 the day-0 probe measured faster; CPU workers are not used for arms (minutes/episode).

### 10.9 Figures (`docs/figures/*.svg`, matplotlib only, committed; every point carries n and a CI)

- F1 `curve-safety-latency.svg` -- x = d (twin top axes: ms at 100 ms and at 20 ms, the latter captioned as a re-scaling). Three stacked panels: (a) flagged / averted / false-trip, (b) success for `t01-a05-d*`, `t0-d*`, and the `obs-d*` latency control, (c) intervention + escalation rate, with `tce_valid_frac` per d as a thin bar under (a). Sync solid, async dashed; pilot (small_n) points hollow with no connecting line. Caption states: "sync d >= 7 has no chunk overlap; tce/acc are invalid there and the per-d valid fraction is shown". `decide_ns` histogram inset.
- F2 `pareto-detection.svg` -- Layer A: TPR vs FPR over the alpha grid, one line per detector subset, AUC in the legend; held-out-FPR-vs-alpha panel.
- F3 `lead-time.svg` -- CDF of lead (ticks before the PoNR proxy), per alpha, proxy definition in the caption, plus the eps sensitivity and the artefact that an episode with no progress at all has `t_fail = 0`, so every detector shows a negative lead on it.
- F4 `score-bands.svg` -- median +- IQR of `s_t` for successful vs failing episodes with `tau` overlaid.
- F5 `latency-hist.svg` -- `decide_ns` HdrHistogram, log-x, p50/p99/p99.9/p99.99/max annotated; `io_ns` beside it; cyclictest environment baseline panel; captioned "measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) -- not a real-time environment".
- F6 `success-delta.svg` -- paired McNemar contingency per arm vs `obs-d0` and vs `obs-d{d}`.
- F7 `cp-validity.svg` -- nominal alpha vs empirical held-out FPR (K-of-N and K=1) with the diagonal.
- F8 `calib-economy.svg` -- tau stability vs `n_calib`.
- F9 `enforcement.svg` -- `violations_reached_env` for `inj-obs-d0` vs `inj-t0-d0`, with the tamper-demo terminal capture.

---

## 11. Honest boundary (README section -- hard editorial rule)

What lictor cannot catch, stated before anyone asks:

1. **Semantic task failure inside the geometric envelope.** On PushT the natural failures are semantic -- the block does not reach the goal -- and happen entirely inside a legal workspace. Tier 0 is a guard, not the main event; its targets are (a) zero violations reaching the environment under injected action spikes and (b) no measurable success cost once the envelope is fitted at empirical p99.9 x 1.25 (McNemar p > 0.05); both are measured and reported whichever way they come out. The predictive tier carries the curve.
2. **Conformal guarantees are conditional.** alpha bounds the episode-level false-alarm rate only under exchangeability with the calibration pool -- this policy, this task, this distribution. A shift breaks it. With K-of-N, alpha is a conservative upper bound; we report the empirical held-out rate beside it.
3. **Perception-assisted checks (`contact`, any `cov_stall` ext channel) use privileged simulator state** and are off by default; when shown they are a separate, labelled line.
4. **The braking model is contact-free.** Contact only decelerates the agent in PushT, so the prediction is conservative for the box constraint -- a modelling assumption, not a proof. The rollout starts from the simulator's own velocity (`info["vel_agent"]`); an embodiment without a velocity signal falls back to a finite difference, which is recorded in the receipt and makes the rollout approximate.
5. **WSL2 measurements are not real-time measurements.** Algorithmic determinism, zero allocations, fixed iteration bounds and byte-identical replay (on the tested target) are verifiable properties. Latency histograms are statistics under virtualisation and every one of them carries the WSL2 label. The d-axis is simulated time and therefore virtualisation-independent. lictor never says "safety-rated", "hard real-time" (any spelling), "certified", "engineered to ... principles", or implies a PL/SIL rating; it claims algorithmic determinism, a bounded-by-construction hot path and measured latency histograms. It borrows two design ideas from the functional-safety literature -- fail-closed defaults and a bounded, deterministic decision path -- and claims no conformance to IEC 61508, ISO 13849 or ISO 10218; the SS1/SS2/STO/monitored-standstill words are used for readability only.
6. **The receipt is evidence against outsiders, not against the experimenter.** The fuse verifies mode, dims, digests, its own hash and every tick; everything in `binding`, `run`, `budget.delay_steps`, `fault_injection`, `inputs` and `outcome.success` is the host's signed-by-proxy declaration, and the key-holder can rebuild and re-sign a ledger. Truncation of a DECLARED pool is detected by `lictor curve`; an external witness (Rekor / RFC 3161) is what would bind the experimenter and is roadmap.
7. **65.4 % is the published PushT number.** We re-measure it with our pins and report what we get.
8. **Tier 1 is a detector we ported (TCE / ACC / ACM / NJR / path efficiency / stall), not one we invented.** The contribution is the deterministic, replayable, signed decision plane and the honest budget-to-safety curve.
9. **Sensors that lie and a calibration set contaminated with failures** defeat the predictive tier; the receipt records the calibration pool and its success labels so the contamination is at least auditable.
10. **The IPC round trip (tens of us) is larger than the verdict itself.** Both numbers are reported; neither is hidden in the other.

---

## 12. Roadmap (README order, value-first; nothing below changes `lictor-core`, `lictor-detect`, `lictor-fuse`, the receipt schema or the wire schema -- only a new manifest, envelope and adapter)

1. **PushT + diffusion policy** (this document): the curve, the receipts, the tamper demo.
2. **ALOHA sim + ACT**: `ActionKind::JointPosition`, `action_dim=14`, box = joint limits, `BrakeKind::JerkLimited`; a non-VLA policy class; `envelopes/aloha.toml` + `harness/aloha_rollout.py`.
3. **LIBERO + `lerobot/smolvla_libero`** via `adapters/lerobot/lictor_lerobot.py` (`LictorStep`, a v0.6 `env_postprocessor` ProcessorStep -- the slot whose docs show a `torch.clamp` safety example); `action_dim=7`, `control_hz=20`, `horizon=50`, `exec=25`; the first policy whose per-suite success is unpublished and must be measured; 20 Hz makes the budget axis bite. Shipped as an importable, clearly-labelled UNVERIFIED stub in milestone 1.
4. **openpi websocket proxy** (`adapters/openpi/lictor_proxy.py`, then a Rust `lictor proxy --upstream ws://host:8000`): pi0.5 on a rented 4090 with WAN latency as a controlled variable; msgpack in, same chunk IR, same NDJSON internally. Shipped as an importable UNVERIFIED stub in milestone 1.
5. **Tier-2 sidecar contract**: `ext0..ext3` are already plumbed; a patched `BasePolicy` attaches logpZO / RND / SAFE-probe scalars and the mask flips on.
6. **`RearmPolicy::AckOnly` operator tooling**: `lictor ack` exists; a real operator console is roadmap.
7. **Logger thread over a preallocated SPSC ring** (chain folding off the I/O thread) for high-rate embodiments.
8. **Native Linux / PREEMPT_RT histograms** (mlockall + SCHED_FIFO + isolcpus): the credible worst-case latency claim; the same binary is expected in the 50-200 us class -- stated conditionally, not measured here.
9. **`no_std` MCU fuse**: `lictor-core`/`-detect`/`-fuse` compile without std today (libm sqrt fallback); the cheapest path to a bounded-latency measurement on a bare-metal target later.
10. **External witness (Rekor / RFC 3161)**: closes the ledger tail-truncation gap.
11. **`lictor-zk`**: Bulletproofs range proof that a commanded action was inside the signed envelope limits without revealing it (bulla-zk lineage, isolated dalek-ng fork).
12. **cu29-task / dora-node wrappers**: distribution channels into Copper-rs and dora-rs.

Language that is never used, anywhere, by anyone on this project as a claim about lictor: "safety-rated", "hard real-time" / "hard-real-time", "PL d", "SIL 2", "certified", "engineered to <standard>", "<standard> principles", "certified limits", and any "-like"/"-style" reference to a standard's clause without the disclaimer "terms borrowed for readability; no conformance is claimed or tested". WP-12's acceptance grep covers every spelling.

---

## 13. Conflict log -- what was merged and which design won each point

| Point | Design 1 (experiment-first) | Design 2 (audit-first) | Decision and why |
|---|---|---|---|
| Python-Rust IPC | subprocess NDJSON | subprocess NDJSON | both; no conflict |
| Wire numbers | plain JSON f64 | float-free hex/base64 | D1 for the WIRE (exact anyway, readable traces; `null` = non-finite); D2 for every HASHED file |
| Canonical bytes | serde field order (JCS roadmap) | RFC 8785 JCS, float-free profile | D2 -- break bulla's canonicalisation now while nothing external verifies; stdlib Python verifier is the adoption proof |
| Float encoding in files | q6 fixed-point ints in a binary tick core | `_f64` / `_f64a` suffixed fields | D2 mechanism, made self-describing: `{"f64":hex}` / `{"f64a":b64,"shape"}` objects; `floatify` for TOML-sourced config; no key-suffix convention |
| Tick chain | fixed-width LE binary, single chain | JCS per tick, two chains (verdict + timing) | D2 -- one canonical mechanism for everything the Python verifier touches; timing honesty |
| Chain folding | I/O thread after decide() | logger thread + SPSC ring | D1 for milestone 1 (buildable today; measured region already excludes it); ring is roadmap |
| Crates | core/detect/fuse/receipt/wire/calib/cli | core(all logic)/canon/receipt/fuse(server)/calib/cli | D1's detect/fuse split (better parallel ownership) + D2's canon crate; server runtime becomes `lictor-runtime` |
| Chunk IR | `ChunkBuf` + Copy `ChunkView`/`ObsView` | owned `ActionChunk` arrays + `TickInput` | D1's borrowed views (cheaper, no 16 KB copies) + D2's `TickInput` with `missed_ticks` and `ack` |
| Chunk spec | `n_action_steps=15` override, L=7 | assumed stock L=8 | D1 -- verified against the slicing code; D2's assumption was wrong |
| Feature set | 8 (tce, acm, reach, jerk, path_ineff, stall, speed_peak, sidecar) | 6 + 4 ext (trackerr, tce, acc, acm(neg), njr, patheff) | union: 12 = tce, acc, acm_neg, njr, reach, path_ineff, stall, speed_peak, ext0..3; D2's orientation rule (larger = more anomalous) |
| Tier-0 as scores? | trip bits | score channels with s > 0 | D1 -- hard limits are trip bits, not calibrated channels (D2 agreed they are never windowed) |
| Calibration | pooled median/MAD; single split quantile on aggregate s; lambda-grid functional band | per-bin mean/std; per-channel taus; DNF gate | single quantile on the DNF aggregate (D1's exact bound + D2's gate) over D2's integer time-binned standardisation with D1's robust median/MAD; lambda grid dropped; 70/30 holdout (D2) |
| K-of-N | u32 ring | u64 popcount | D2 |
| FSM | Idle/Armed/Warned/Clamped/Holding/Escalated/Fault; auto re-arm budget | Armed/Watching/Clamped/Braking/Held/Escalated/Faulted/Terminated; ack-only re-arm; Watching -> Braking timer | union: Idle + D2 states; Watching = D1's warn band (informational only; K-of-N goes straight to Braking -- no second timer); `RearmPolicy {Auto (D1, PushT default), AckOnly (D2)}`; Escalated always needs an ack or the harness policy |
| Brake feasibility | 160-iteration exact PD rollout | closed-form stopping ball per BrakeModel | both: `PdSecondOrder` (PushT, exact) + closed-form kinds for other embodiments |
| Brake action | latched hold at p_trip | setpoint := measured p_t each tick | Braking = D2 (decay, the verified model); Held = D1 (latched, never chases the block) |
| v0 for the rollout | 0 when vel unavailable | finite-difference v_hat | NEITHER after review: gym_pusht returns `info["vel_agent"]` every step, so v0 is the simulator velocity (`provides_vel = true`); the finite difference is the recorded fallback for other embodiments |
| Baseline arm | `mon-d0` (monitor) | `off` (observe-only) | same idea; named `obs-d0`, plus D1's byte-identity assertion and D2's `violations_reached_env` |
| Latency control arms | `mon-d{1,2,3,5,8}` | none | D1 -- without them a success drop is unattributable |
| Envelope numbers | placeholders + `calibrate --tier0` percentiles | `envelope fit` p99.9 x 1.25 + McNemar parity gate | D2's procedure with D1's placeholder base file |
| Seeds | eval 0..499, calib 900000..900299 | hashed eval seeds | D1 (transparent), plus D2's `init_state_digest` pairing proof |
| Metrics source | harness CSVs + `lictor sweep` | `lictor curve` from receipts only | both: `sweep` (Layer A, from verified ticks) + `curve` (Layer B, from receipts, refuses on broken ledger); harness index is never a source of truth |
| Escalation accounting | `terminate_fail` default; `oracle_resume` counterfactual | `terminate` default; scripted oracle + signed resume | D1 naming, D2's signed-ack oracle arm |
| Injection arms | none (natural failures) | `inj_off`/`inj_on` action spikes | D2 -- the enforcement pillar needs a demonstration PushT's natural failures cannot give |
| Receipt ticks | embedded head32/all/none + ticks digest | separate trace + chain head | both: separate ticks file + `ticks_policy` tail32 embedding for escalation forensics |
| CLI naming | `serve`, `key`, `ledger`, `history`, `sweep` | `fuse`, `keygen`, `ack`, `curve`, `report` | union under D1 names; `ack` and `curve` added; `report` replaced by `harness/figures.py` |
| Python layout | `sim/` + `adapters/lictor_client.py` | `harness/` + `adapters/lerobot/` | `harness/` (D2 name) + D1's transport client + D2's ProcessorStep as a roadmap stub |
| Float discipline | `+ - * / sqrt min max abs` only, no libm | libm for transcendentals | D1 (none are needed); libm only as the no_std `sqrt` fallback (D2's no_std goal kept) |
| Windows-native build | fallback | fallback | both; WSL is working in this session (cargo 1.98, target dir on D:) so it is insurance, not the plan |
