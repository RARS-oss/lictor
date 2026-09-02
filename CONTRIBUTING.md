# Contributing

lictor follows the bulla / sbx house rules. They are short, they are enforced by CI and by a grep, and a
pull request that does not meet them is not reviewed for content. The normative design documents are
`docs/ARCHITECTURE.md` (the interface freeze, the math, the experiment) and `docs/IMPLEMENTATION_PLAN.md`
(the same freeze plus the work packages and their file ownership); `docs/VERIFIED_FACTS.md` records what was
measured on the development machine and wins over the plan wherever the two disagree.

## 1. Environment

All Rust and Python work runs under WSL2 (Ubuntu) on the development machine; a native Linux box works the
same way. Drive C: is full, so **`CARGO_TARGET_DIR` is mandatory** -- the Makefile's `guard` target refuses
to run without it, `.cargo/config.toml` deliberately sets no target directory, and nothing is ever built
into the repository.

```bash
export PATH="$HOME/.cargo/bin:$PATH"                 # cargo 1.98 stable (rust-toolchain.toml pins stable + rustfmt + clippy)
export CARGO_TARGET_DIR=/mnt/d/lictor/target         # any writable directory off the repository
export HF_HOME=/mnt/d/lictor/hf                      # checkpoints never land on C:
cd /mnt/c/Users/Daniil/Desktop/Robots/lictor
source harness/env.sh                                # exports every pin and determinism variable (OMP_NUM_THREADS=1, PYTHONHASHSEED=0, ...)
```

Python is `/mnt/d/lictor/venv/bin/python` (3.12; lerobot 0.6.1, torch 2.7.1+cu126, gym_pusht, diffusers;
`opencv-python-headless` only -- `opencv-python` beside it breaks the `cv2` import). Python is always run
as a module from the repository root: `python -m pytest harness/tests adapters/tests -q`,
`python harness/run.py ...`; `pyproject.toml` sets `pythonpath = ["."]`.

When several agents or people build in the same tree at once, use a private target directory
(`CARGO_TARGET_DIR=/mnt/d/lictor/target-<you>`) and crate-scoped commands (`cargo build -p <crate>`), and
never run `cargo fmt --all` over files you do not own.

## 2. Before a pull request

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
make nostd-check                                      # lictor-core / -detect / -fuse without std
cargo build --release -p lictor-cli && $CARGO_TARGET_DIR/release/lictor selftest
bash bench/run_all.sh $CARGO_TARGET_DIR/release/lictor
bash scripts/ci_python.sh                             # no GPU, no torch: the harness unit tests and the stdlib verifier
```

CI (`.github/workflows/ci.yml`) runs exactly these, plus `python3 adapters/verify_receipt.py` on the
committed receipt fixture. `RUSTFLAGS=-D warnings` is set globally; a warning is a failure.

## 3. Rust style

- Edition 2021. `#![forbid(unsafe_code)]` in every library crate. The only `unsafe` in the workspace is
  the counting allocator (`crates/lictor-cli/src/alloc_count.rs`, behind `#![deny(unsafe_code)]` +
  `#[allow(unsafe_code)] mod alloc_count;` in `main.rs`) and its twin in
  `crates/lictor-fuse/tests/alloc.rs`. Nothing else, ever.
- `cargo clippy --workspace --all-targets -- -D warnings` clean. No `#[allow(clippy::...)]` on frozen
  items and no `[workspace.lints]` table: every `pub fn new() -> Self` has a `Default` implementation
  beside it, every `len()` an `is_empty()`. If a lint cannot be satisfied without changing a frozen signature, the
  freeze is amended in both design documents, not the lint.
- `// SPDX-License-Identifier: MIT` is the first line of every source file (Rust, Python, shell, SVG
  comment, TOML, YAML). The licence is MIT for everything.
- ASCII only, in code and in docs. No emojis. No typographic dashes or quotes; `--`, `+-`, `x`, `<=`.
- Every serde enum on a wire or file surface is `#[serde(rename_all = "snake_case")]`; every wire payload
  struct is `#[serde(deny_unknown_fields)]`.
- Human output is deterministic and capped (fixed column widths, at most N lines, control characters
  stripped); every command accepts `--json`. CI greps the frozen output strings listed in
  `docs/ARCHITECTURE.md` sec 9 -- do not reword them.
- Every JSON artefact written to disk is pretty-printed with sorted keys. Canonical bytes
  (`jcs-floatfree/v1`) are always recomputed from the parsed structure and never trusted from disk.

## 4. The decision path (`lictor-core`, `lictor-detect`, `lictor-fuse`)

These three crates are `#![cfg_attr(not(feature = "std"), no_std)]` and contain no I/O, no crypto, no
clock and no allocation on the decision path. Inside them the following are forbidden and reviewed for by
grep: `f32`, `mul_add`, `powi`, `powf`, `exp`, `ln`, `sin`, `cos`, `atan2`, `hypot`, `f64::min`,
`f64::max`, `f64::clamp` (NaN semantics differ from `fmath`), `as` casts from `f64`, `HashMap` iteration,
`rayon`, SIMD reductions, `Instant` / `SystemTime`, any RNG, any allocation, and any panic on data
(`debug_assert!` only). Use `lictor_core::fmath` for `sqrt`, `min`, `max`, `abs`, `clamp`, `norm`, `dist`.
Sums are accumulated left to right, index ascending. Any non-finite value reaching `decide()` is
`TripMask::NONFINITE` -> `Fault` -> hold action.

`lictor-calib` and `lictor-receipt` are off the decision path and may use `ln`, `exp`, `lgamma`-style
functions and `f64::total_cmp` sorting.

## 5. Tests and fixtures

- Unit tests live inline (`#[cfg(test)]`) in the module they test; integration tests in `tests/*.rs`.
- Fixture tests are the norm: JSON / TOML fixtures under `tests/fixtures/<area>/`, generated by a named
  script (`gen.py` beside the fixture, numpy for the numeric ones, tolerance `1e-12`), pretty-printed with
  sorted keys, and never edited by hand. A fixture that changes needs its generator changed in the same
  commit.
- Every `include_str!` must resolve at all times; a fixture that is not yet real is an empty `[]` and the
  consuming test reports `[skip]` rather than failing.
- Tests that depend on another work package's unfinished code are kept compiling and marked
  `#[ignore = "pending WP-n: <reason>"]` (Rust) or `pytest.mark.skip(reason="pending WP-n")` (Python), and
  listed in the package report. They are un-ignored by the integration package.
- Numbers in tests are measured, never asserted from memory: a latency bound in a test is a regression
  tripwire on the tested machine, labelled as such.

## 6. Banned vocabulary

The following are never used as claims about lictor, anywhere -- code, comments, docs, commit messages,
figure captions, issue text. Each line below is a term that is not claimed, with the wording to use instead:

- "safety-rated" -- not claimed; say "a software fuse with a measured, WSL2-labelled latency histogram".
- "hard real-time" / "hard-real-time" -- not claimed; say "algorithmic determinism, zero allocations, fixed iteration bounds".
- "PL d", "SIL 2", or any performance / integrity level -- not claimed; lictor implies no rating.
- "certified", "certified limits" -- not claimed; say "limits fitted at empirical p99.9 x slack, recorded in `[fit]`".
- "engineered to <standard>", "<standard> principles" -- not claimed; say "two ideas borrowed from the functional-safety literature; no conformance claimed or tested".
- "determinism proof" -- not claimed; the 40-replay run is determinism evidence on the tested target.
- "bit-identical on every conforming platform" -- not claimed; say "on the tested target".
- "a receipt can never hide ..." -- not claimed; say "the receipt binds the host's declaration of ...".
- any "-like" / "-style" reference to a standard's clause -- never without the disclaimer "terms borrowed for readability; no conformance is claimed or tested".

Every latency number carries the WSL2 label (`latency_label`) or the label of the machine it was measured
on.

The grep that enforces it (it is WP-12's acceptance command and must stay empty):

```bash
! grep -rniE "safety-rated|hard[ -]real[ -]time|PL d|SIL 2|\bcertified\b|engineered to|principles of|[0-9]{4,5} principles|certified limits|determinism proof|every conforming platform|can never hide" \
    README.md SECURITY.md CONTRIBUTING.md docs/*.md research/*.md harness/*.py adapters/*.py crates \
  | grep -viE "never|not claim|banned|does not|no claim|not-claim|borrowed for readability"
```

Numbers that have not been measured are written as explicit placeholders ("TBD after the pilot"), never
estimated; every number that is cited has a memo under `research/` (see `research/README.md`).

## 7. The interface freeze

`docs/ARCHITECTURE.md` sec 4 and `docs/IMPLEMENTATION_PLAN.md` PART A are the frozen public surface: types,
wire messages, file formats, CLI flags and the frozen output strings. Bodies belong to the owning work
package; signatures do not change without an amendment recorded in both documents (PART A.0 lists the ones
made so far). If a frozen signature is wrong, do not patch around it in files you do not own: implement the
best faithful version inside your files and report the problem. `docs/work-packages.json` is the
authoritative ownership list; exactly one owner writes a given path.

## 8. Keys and results

- Keys never live in the repository (`*.hex` is git-ignored) and never under `/mnt/[a-z]/` (DrvFs does
  not enforce `0600`; `lictor key init` refuses such a path without `--i-know`). The default location is
  `$LICTOR_KEYS`, else `$HOME/.lictor`. The two committed test keys are listed by public key in
  `SECURITY.md`; `lictor verify` warns whenever a receipt was signed with one of them, and no experiment
  arm is ever signed with one.
- Experiment outputs (`results/`) are never committed; heavy artefacts live on D:
  (`LICTOR_RESULTS=/mnt/d/lictor/results`). The repository commits only `bench/fixtures/`,
  `docs/figures/`, the fitted envelopes, `docs/envelope_fit_report.md` and the `research/` memos.

## 9. Commits and branches

- Branch names `wp-<n>-<slug>` during the parallel phase, `<topic>` afterwards. Commit only when asked to
  during the parallel phase.
- One fix per commit during integration, with a message naming the package it corrects.
- No `--no-verify`, no history rewriting on `main`.

## 10. Reporting

Open a GitHub issue on `RARS-oss/lictor`. For anything that concerns verification, attach the receipt, the
ticks file and the output of `lictor version`. For anything sensitive, see `SECURITY.md`.
