# lictor -- REPRODUCE (the exact command sequence)

This is the recipe for the PushT experiment of `docs/EXPERIMENT.md`, in the order ARCHITECTURE 10.8 runs it:
the 220-episode pilot (steps 0-5), then the overnight priority list. Every command is meant to be pasted.
Heavy artefacts (the venv, the checkpoint, the results, the cargo target directory) live on `D:`; the
repository never receives them. Every wall-clock number here comes from `docs/VERIFIED_FACTS.md` or from
`microbench.json`; none is an estimate that predates the measurement.

Every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement.

## 0. Machine, paths, environment

| item | value |
|---|---|
| host | Windows 11 + WSL2 Ubuntu (kernel 6.6), GTX 1060 3GB (Pascal, fp32 only), 7.9 GB RAM in the guest |
| repository | `/mnt/c/Users/Daniil/Desktop/Robots/lictor` (Windows: `C:\Users\Daniil\Desktop\Robots\lictor`) |
| toolchain | cargo 1.98 / rustc 1.98 stable at `$HOME/.cargo/bin` (not on `PATH` in non-login shells) |
| Python | `/mnt/d/lictor/venv` (3.12): `lerobot 0.6.1`, `torch 2.7.1+cu126`, `gymnasium 1.3.0`, `gym_pusht`, `diffusers`, `opencv-python-headless` (never `opencv-python` beside it), `numpy`, `matplotlib`, `pytest` |
| checkpoint | `/mnt/d/lictor/models/diffusion_pusht_migrated` (the migrated `lerobot/diffusion_pusht`; see 0.2) |
| results | `LICTOR_RESULTS=/mnt/d/lictor/results` |
| cargo target | `CARGO_TARGET_DIR=/mnt/d/lictor/target` (drive `C:` is full: never build into the repo or `C:`) |
| HF cache | `HF_HOME=/mnt/d/lictor/hf` |
| keys | `$LICTOR_KEYS` (default `$HOME/.lictor/`): `key.hex`, `operator.hex`, `history.jsonl` -- never in the repository, never under `/mnt/[a-z]/` (DrvFs ignores mode 0600) |

Running WSL commands from Windows: write a script file and run
`MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash /mnt/d/<...>/script.sh` (inline quoting through PowerShell or
Git-Bash breaks `$PATH`).

### 0.1 Shell setup (every session)

```bash
cd /mnt/c/Users/Daniil/Desktop/Robots/lictor
source harness/env.sh          # PATH, CARGO_TARGET_DIR, LICTOR_RESULTS, LICTOR_MODELS, HF_HOME,
                               # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
                               # CUBLAS_WORKSPACE_CONFIG=:4096:8 SDL_VIDEODRIVER=dummy,
                               # LICTOR_BIN=$CARGO_TARGET_DIR/release/lictor, activates /mnt/d/lictor/venv
export RUN=$(date -u +%Y-%m-%dT%H-%MZ)-pilot
```

### 0.2 The migrated checkpoint (once per machine)

Loading `lerobot/diffusion_pusht` directly on lerobot 0.6.1 raises `ProcessorMigrationError` (the hub
checkpoint predates the 0.6 processor pipeline). The stock migration script
`lerobot/processor/migrate_policy_normalization.py` then crashes at `save_pretrained` ("Couldn't encode 84":
draccus cannot encode JSON lists into the tuple-typed `crop_shape` / `down_dims` / `optimizer_betas`).
`harness/migrate_checkpoint.py` drives the stock script with that fix:

```bash
python harness/migrate_checkpoint.py --pretrained-path lerobot/diffusion_pusht \
    --output-dir $LICTOR_MODELS/diffusion_pusht_migrated
ls -l $LICTOR_MODELS/diffusion_pusht_migrated
# config.json (2099 B)  model.safetensors (1050861448 B)  policy_preprocessor.json + step_3_normalizer safetensors
# policy_postprocessor.json + step_0_unnormalizer safetensors  README.md
```

`compat.py` loads the policy AND the processors from that directory (`LICTOR_MODEL` overrides the path) and
binds `weights_sha256`, `normalization_migrated = true` and `migration_script` into every receipt.

### 0.3 Build, self-test, keys

```bash
cargo build --release -p lictor-cli                       # ~35 s cold on this machine (WSL2, not RT)
$LICTOR_BIN version                                       # lictor 0.1.0 (<git>) sha256=<binary sha>
$LICTOR_BIN selftest                                      # ... SELFTEST PASS
$LICTOR_BIN key init                                      # -> $HOME/.lictor/key.hex ; prints "pubkey <hex>", "stored <path> (mode 0600)"
$LICTOR_BIN key init --role operator -o $HOME/.lictor/operator.hex
$LICTOR_BIN key pub | tee /mnt/d/lictor/pubkey.txt        # pin the expected key OUT OF BAND; --pubkey uses it below
export PUBKEY=$(sed -n 's/^pubkey //p' /mnt/d/lictor/pubkey.txt)
export OPERATOR_PUB=$($LICTOR_BIN key pub --key $HOME/.lictor/operator.hex | sed -n 's/^pubkey //p')
```

`key init` refuses to overwrite without `--force` and refuses any path under `/mnt/[a-z]/` without
`--i-know`. `lictor serve` without a key generates an ephemeral one and marks every receipt
`fuse_ok = false` with the note `ephemeral signing key`; `lictor curve` refuses such arms, so the key step is
not optional.

## 1. Measured costs (why the schedule looks the way it does)

Every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement.

| quantity | measured | where |
|---|---|---|
| `import lerobot, torch` from the drvfs venv | 12-16 s | VERIFIED_FACTS |
| policy load from the migrated directory | 8.3 s | VERIFIED_FACTS |
| chunk inference (100 DDPM steps), B = 1 | mean 3.46 s, max 6.3 s | VERIFIED_FACTS |
| one 300-step episode at B = 1 (~38 chunk inferences) | 153.9 s | VERIFIED_FACTS (seed 0, observe) |
| ms per env-chunk vs batch size | B=1 2843 / B=2 1389 / B=4 705 / B=8 376 / B=16 213 / B=32 123 | VERIFIED_FACTS `bench_batch` |
| peak VRAM at B = 32 | 2387 MiB (3 GB cap; B = 64 untested -- do not exceed 32 without re-measuring) | VERIFIED_FACTS |
| GPU time per episode at B = 32 | 38 x 123 ms = 4.7 s | derived |

Projections from those measurements (GPU time only; environment stepping and fuse IPC are recorded by
`microbench.json` when the harness runs).
Every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement.

| step | episodes | at B = 1 (153.9 s/ep) | at B = 32 (4.7 s/ep GPU) |
|---|---|---|---|
| `calib-obs` x 100 | 100 | 4.3 h | ~8 min |
| `obs-d0` x 60 | 60 | 2.6 h | ~5 min |
| `t01-a05-d0` x 60 | 60 | 2.6 h | ~5 min |
| pilot total | 220 | 9.4 h | ~17 min |
| one full arm x 500 | 500 | 21.4 h | ~40 min |

The 12-20 s/episode assumption behind ARCHITECTURE 10.8's one-hour table is superseded by the measurement:
at B = 1 `run.py --plan --budget-min 60` refuses every pilot step (intended), and the pilot fits an hour only
with `B = 32` batched environments. Batching and paired seeding interact (EXPERIMENT section 9); the day-0
determinism probe (`obs-d0` twice on seeds 0..2) and `cross_run_mismatches` are the checks that the harness's
batching kept pairing exact. If they fail, the fallback is B = 1 with the left-hand column.

## 2. Day-0 gate (step 0; nothing else runs until it prints PASS)

```bash
python harness/run.py --check-pools                       # pools disjoint: ok
python harness/microbench.py --out $LICTOR_RESULTS/$RUN   # writes microbench.json; first row: import lerobot.policies
cat $LICTOR_RESULTS/$RUN/microbench.json                  # environment = the WSL2 sentence; s/episode GPU B=1 and CPU;
                                                          # 1-vs-2-worker episodes/min; chunk latency; h15 PASS;
                                                          # determinism PASS on seeds 0..2; bench p99
$LICTOR_BIN bench --n 200000                              # allocations   0 ; p99 <= 50 us (WSL2, not RT)
$LICTOR_BIN bench --n 200000 --csv $LICTOR_RESULTS/$RUN/bench.csv
$LICTOR_BIN replay --repeat 40 bench/fixtures/traces/pusht_000007.ndjson \
    --envelope envelopes/pusht.toml --calibration bench/fixtures/calibration.a05.json --mode enforce
#   replays 40/40 byte-identical  verdict_chain=<hex>
#   timing chain head varies (by design -- wall-clock is not replayed)
#   RESULT  DETERMINISTIC
python harness/run.py --run-id $RUN --arms obs-d0 --seeds 0-9 --out $LICTOR_RESULTS --workers 1   # smoke x 10
python harness/run.py --run-id $RUN --out $LICTOR_RESULTS --summary
python harness/run.py --plan --budget-min 60 --arms calib-obs,obs-d0,t01-a05-d0 --seeds pilot \
    --out $LICTOR_RESULTS --run-id $RUN                   # prints the 220-episode projection; REFUSES over-budget steps
```

`--plan` reads `microbench.json`, picks 1 or 2 workers from the measured probe and refuses to start a step
whose projection exceeds `--budget-min`. Deliverable of step 0: plumbing green, s/episode measured, F5
(`latency-hist.svg`) renderable from `bench.csv` + `microbench.json` -- it lands first.

## 3. The pilot (steps 1-5; 220 episodes)

Seeds: pilot calibration `900000..900099` (`--pool calib`), pilot evaluation `0..59` (`--pool eval`);
full pools `900000..900299` and `0..499`. Every command below is resumable (`--resume` is the default; the
ledger is the source of truth) and refuses to aggregate arms whose `init_state_digest` disagree.

### Step 1 -- calibration pool, envelope fit, calibration artefacts

```bash
python harness/run.py --run-id $RUN --arms calib-obs --seeds 900000-900099 --pool calib \
    --envelope envelopes/pusht.base.toml --out $LICTOR_RESULTS
$LICTOR_BIN envelope fit --run $LICTOR_RESULTS/$RUN --arm calib-obs --base envelopes/pusht.base.toml \
    --quantile 0.999 --slack 1.25 -o envelopes/pusht.toml --report docs/envelope_fit_report.md
$LICTOR_BIN envelope fit --run $LICTOR_RESULTS/$RUN --arm calib-obs --base envelopes/pusht.base.toml \
    --quantile 0.999 --slack 1.25 --operator $OPERATOR_PUB -o envelopes/pusht.oracle.toml
$LICTOR_BIN envelope check envelopes/pusht.toml           # validate + digest + embodiment digest (identical for both files)
$LICTOR_BIN calibrate --run $LICTOR_RESULTS/$RUN --arm calib-obs --envelope envelopes/pusht.toml \
    --alpha 5/100,10/100,20/100,1/100,2/100 --method binned --kn 3,5 --holdout 3/10 --split 2 \
    -o $LICTOR_RESULTS/$RUN
ls $LICTOR_RESULTS/$RUN/calibration.*.json               # one float-free, self-digested file per alpha
```

With ~65 successes out of 100 the 2-way split leaves `n_calib ~ 45`: alpha 1/100 and 2/100 are degenerate at
this n (`tau = +inf`, recorded as a note in the file and drawn as degenerate on F2 / F8); 5/100 gives
k = 44. The calibration binds `embodiment_digest` (identical for `pusht.toml` and `pusht.oracle.toml`) and
`policy_digest` (= `weights_sha256` of the migrated checkpoint); `serve` refuses a mismatch at startup.
Deliverables: `envelopes/pusht.toml`, `envelopes/pusht.oracle.toml`, `docs/envelope_fit_report.md`,
`calibration.<alpha>.json`, F7 and F8.

### Step 2 -- fuse-off baseline (labels, Layer-A eval traces, the 65.4 % smoke check)

```bash
python harness/run.py --run-id $RUN --arms obs-d0 --seeds 0-59 --pool eval \
    --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --out $LICTOR_RESULTS --summary   # success k/60 and its CP interval
```

The 65.4 % gate is a smoke check here (n = 60 -> CI about +-12 pp, accepting roughly 53-77 %); the n = 500
`obs-d0` run overnight is the reproduction. `obs-d0` runs in observe mode with the calibration loaded, so its
ticks carry `f`, `z`, `s`, `valid` for every tick: that is what Layer A re-scores.

### Step 3 -- Layer A (seconds, CPU)

```bash
$LICTOR_BIN sweep --run $LICTOR_RESULTS/$RUN --calib-arm calib-obs --eval-arm obs-d0 \
    --alphas 1/100,2/100,5/100,10/100,20/100 --calibration-dir $LICTOR_RESULTS/$RUN \
    --detectors t0,t1_tce,t1_stall,t1_full,t01,t01_and --method binned --eps-prog 0.02 \
    -o $LICTOR_RESULTS/$RUN/sweep.jsonl
# economy ablation (F8): re-calibrate at n_calib in {25, 50} and re-sweep into the same file
for N in 25 50; do
  $LICTOR_BIN calibrate --run $LICTOR_RESULTS/$RUN --arm calib-obs --envelope envelopes/pusht.toml \
      --alpha 5/100,10/100,20/100 --n-calib $N -o $LICTOR_RESULTS/$RUN/calib-n$N
  $LICTOR_BIN sweep --run $LICTOR_RESULTS/$RUN --calib-arm calib-obs --eval-arm obs-d0 \
      --alphas 5/100,10/100,20/100 --calibration-dir $LICTOR_RESULTS/$RUN/calib-n$N --detectors t01 \
      -o $LICTOR_RESULTS/$RUN/sweep-n$N.jsonl
  cat $LICTOR_RESULTS/$RUN/sweep-n$N.jsonl >> $LICTOR_RESULTS/$RUN/sweep.jsonl
done
```

`sweep.jsonl` is a plain-float analysis product (not signed). Points whose (alpha, gate, kn, method) match a
`calibration.<alpha>.json` reuse its tau verbatim (`tau_source: "artefact"`); the Layer-A-vs-B comparison
uses only those. Deliverables: F2, F3 (from the sweep quantiles; the full lead CDF needs per-episode leads,
see section 5), F8.

### Step 4 -- the one closed-loop pilot point

```bash
python harness/run.py --run-id $RUN --arms t01-a05-d0 --seeds 0-59 --pool eval \
    --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --on-escalate terminate_fail \
    --out $LICTOR_RESULTS
$LICTOR_BIN curve --run $LICTOR_RESULTS/$RUN --baseline obs-d0 --arms t01-a05-d0 --eps-prog 0.02 \
    --allow-small --key $HOME/.lictor/key.hex -o $LICTOR_RESULTS/$RUN/curve
#   t01-a05-d0  n=60  success 0.xxx [0.xxx,0.xxx]  delta_vs_obs-d0 +0.xxx [-0.xxx,+0.xxx] p=0.xx ... tce_valid 0.xxx  pubkey <hex>
head -1 $LICTOR_RESULTS/$RUN/curve/summary.csv            # pilot,arm_id,n,... (the frozen column list); the row starts with pilot=1
```

`--allow-small` binds `small_n: true` into the signed curve receipt and makes the CSV row `pilot = 1`;
`figures.py` draws it hollow and never runs a line through it; it is never a headline. Without
`--allow-small`, `lictor curve` refuses n < 100. It also refuses a broken ledger (`ledger BROKEN at seq=N --
entries are missing or edited; this curve point cannot be trusted`), mixed or ephemeral keys, a
`tier1`/`alpha` disagreement with `run.json`, and an episode-index set that differs from the baseline's or
from the declared pool (`REFUSED: episode index set differs from baseline (missing 3 indices: 17,211,340);
re-run them or pass --partial`). Deliverable: the pilot point on F1, F6.

### Step 5 -- determinism and tamper evidence

```bash
$LICTOR_BIN replay --repeat 40 $LICTOR_RESULTS/$RUN/obs-d0/traces/000007.ndjson \
    --envelope envelopes/pusht.toml --calibration $LICTOR_RESULTS/$RUN/calibration.a05.json --mode observe \
    --expect $(python - <<'EOF'
import json,os; r=json.load(open(os.path.expandvars("$LICTOR_RESULTS/$RUN/obs-d0/receipts/000007.json"))); print(r["body"]["verdict_chain_head"])
EOF
)
cargo test -p lictor-fuse --release --test alloc -- --nocapture | grep "decide: 10000 ticks, 0 allocations, 0 deallocations"
bash bench/tamper/run_experiment.sh $LICTOR_BIN          # signature FAIL / chain BROKEN at seq=17 / HEAD MISMATCH / pubkey MISMATCH
bash scripts/demo.sh | tee $LICTOR_RESULTS/$RUN/demo.txt # the README capture; feeds F9 via --tamper-log
```

## 4. Analysis and figures

```bash
python harness/analyze.py --run $LICTOR_RESULTS/$RUN --bench-csv $LICTOR_RESULTS/$RUN/bench.csv
#   arms / layer A vs B / latency tables, then: stats cross-check: ok  (or the diffs, exit 1)
ls $LICTOR_RESULTS/$RUN/analysis/      # arms.csv sweep.csv layer_ab.csv lead.csv tce_valid.csv latency.csv cross_check.json
python harness/analyze.py scores --run $LICTOR_RESULTS/$RUN --arm obs-d0 -o $LICTOR_RESULTS/$RUN/analysis/score_bands.csv
python harness/figures.py --run $LICTOR_RESULTS/$RUN --out docs/figures \
    --bench-csv $LICTOR_RESULTS/$RUN/bench.csv --score-bands $LICTOR_RESULTS/$RUN/analysis/score_bands.csv \
    --tamper-log $LICTOR_RESULTS/$RUN/demo.txt
ls docs/figures/*.svg | wc -l          # 9
```

`analyze.py` consumes `curve/summary.csv`, the signed `curve/<arm>.json` bodies, `sweep.jsonl`,
`microbench.json`, the bench bucket dump and (optionally) `calibration.<alpha>.json` -- never the harness
`index.jsonl`, never an episode receipt. The deltas are copied verbatim from `summary.csv`; the
cross-check recomputes Clopper-Pearson / McNemar / bootstrap / bacc with `harness/stats.py`. `scores` is the
one documented exception (ledger + ticks of one observe arm, for F4's per-tick bands). CI renders the same
nine figures without a run: `python harness/figures.py --from-fixture --out /tmp/figs` (samples, never
committed).

Latency table sentence (printed by `analyze.py` above its latency table and carried by every row as
`latency_label`):
every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement.

| `analysis/latency.csv` row | meaning |
|---|---|
| `source = curve` | `latency_p50_ns / p99 / max` of `decide_ns` per arm from the signed curve receipts, n = episodes |
| `source = bench` | percentiles (bucket upper bounds) of `lictor bench --csv` per series: verdict, tier0, brake, tier1 |
| `source = microbench` | every `*_s` / `*_ms` / `*_us` / `*_ns` field of `microbench.json`, scaled to ns |

## 5. Overnight (priority order; each step gated by `--plan`)

```bash
export RUN=$(date -u +%Y-%m-%dT%H-%MZ)-full
python harness/run.py --plan --budget-min 480 --arms calib-obs --seeds 900000-900299 --out $LICTOR_RESULTS --run-id $RUN
python harness/run.py --run-id $RUN --arms calib-obs --seeds 900000-900299 --pool calib --envelope envelopes/pusht.base.toml --out $LICTOR_RESULTS
# refit + recalibrate at every alpha exactly as in step 1 (2-way AND --split 3, so F7 shows both bounds)
python harness/run.py --run-id $RUN --arms obs-d0 --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms t0-d0 --seeds 0-59 --pool eval --envelope envelopes/pusht.toml --out $LICTOR_RESULTS      # parity gate
$LICTOR_BIN curve --run $LICTOR_RESULTS/$RUN --baseline obs-d0 --arms t0-d0 --allow-small --partial --key $HOME/.lictor/key.hex -o $LICTOR_RESULTS/$RUN/curve
#   require McNemar p > 0.05 and overlapping CP intervals; otherwise refit with more slack and record it in docs/envelope_fit_report.md
python harness/run.py --run-id $RUN --arms t01-a05-d0 --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms obs-d2,t01-a05-d2 --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms "t01-a05-d*,obs-d*" --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms t01-a01-d0,t01-a10-d0,t01-a20-d0 --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms "*-async" --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --calibration-dir $LICTOR_RESULTS/$RUN --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms t01-a05-d0-oracle --seeds 0-499 --pool eval --envelope envelopes/pusht.oracle.toml --calibration-dir $LICTOR_RESULTS/$RUN --on-escalate oracle_resume --out $LICTOR_RESULTS
python harness/run.py --run-id $RUN --arms inj-obs-d0,inj-t0-d0 --seeds 0-499 --pool eval --envelope envelopes/pusht.toml --out $LICTOR_RESULTS
$LICTOR_BIN curve --run $LICTOR_RESULTS/$RUN --baseline obs-d0 --key $HOME/.lictor/key.hex -o $LICTOR_RESULTS/$RUN/curve   # every arm, latency controls resolved by d
python harness/run.py --run-id $RUN --out $LICTOR_RESULTS --summary --compare-run $LICTOR_RESULTS/<pilot run>   # cross-run lines
```

Order and rationale: `calib-obs` to 300 (n_calib = 137, no degenerate alpha above 5/1000) -> `obs-d0` x 500
(the real 65.4 % reproduction) -> the parity gate -> the headline point -> the d = 2 pair -> the rest of the
budget curve with its latency controls -> the alpha sweep -> the async pairs -> the oracle arm -> the
injection arms. At B = 32 each 500-episode arm is ~40 min of GPU time; at B = 1 it is ~21 h and only
`obs-d0`, `t01-a05-d0` and the parity gate are realistic for a first night. Workers: whichever of 1 or 2 the
day-0 probe measured faster; CPU workers are never used for arms (minutes per episode).

## 6. Verifying any receipt (always with `--pubkey`)

```bash
python adapters/verify_receipt.py $LICTOR_RESULTS/$RUN/t01-a05-d0/receipts/000007.json --pubkey $PUBKEY \
    --ticks $LICTOR_RESULTS/$RUN/t01-a05-d0/ticks/000007.jsonl --ledger $LICTOR_RESULTS/$RUN/t01-a05-d0/ledger.jsonl
#   OK ...            (stdlib only: json + hashlib + a pure-Python Ed25519 verify; no lictor binary needed)
$LICTOR_BIN verify $LICTOR_RESULTS/$RUN/t01-a05-d0/receipts/000007.json \
    --ticks $LICTOR_RESULTS/$RUN/t01-a05-d0/ticks/000007.jsonl --pubkey $PUBKEY --json | python -c "import json,sys; r=json.load(sys.stdin); print('intact', r['intact'])"
$LICTOR_BIN ledger verify $LICTOR_RESULTS/$RUN/t01-a05-d0/ledger.jsonl
#   episodes 500   chain ok    success 341/500 (68.2%)   stops 61   escalated 22   fuse_ok 500/500
$LICTOR_BIN verify $LICTOR_RESULTS/$RUN/curve/t01-a05-d0.json --curve $LICTOR_RESULTS/$RUN/curve/t01-a05-d0.json --pubkey $PUBKEY
```

Pin the key out of band: `--pubkey` is what turns a re-signed receipt into `pubkey MISMATCH`; without it a
verifier only proves the file is consistent with whatever key it embeds. `lictor verify` exits 1 on an honest
observe receipt (intact but not enforced), so scripts test the `intact` field of `--json`, never the exit
code, when they mean "is the record intact". A receipt signed with one of the two committed test keys prints
`WARNING: signed with the committed test key`. What a receipt does not defend against: the holder of the
signing key (trust model, EXPERIMENT threat 11).

## 7. Expected artefacts

```
/mnt/d/lictor/
  venv/  models/diffusion_pusht_migrated/  hf/  target/
  results/<run>/
    run.json                 microbench.json  bench.csv  sweep.jsonl  demo.txt
    calibration.<alpha>.json calib-n25/ calib-n50/
    curve/<arm>.json  curve/summary.csv
    analysis/ arms.csv sweep.csv layer_ab.csv lead.csv tce_valid.csv latency.csv cross_check.json score_bands.csv
    <arm_id>/ ledger.jsonl index.jsonl receipts/NNNNNN.json ticks/NNNNNN.jsonl timing/NNNNNN.jsonl traces/NNNNNN.ndjson
    .lictor/verifier_nonce.json
$HOME/.lictor/ key.hex operator.hex history.jsonl
repo: envelopes/pusht.toml envelopes/pusht.oracle.toml docs/envelope_fit_report.md docs/figures/*.svg research/*.md
```

`index.jsonl` is a convenience index rebuilt from the ledger at startup; nothing in the analysis reads it.

## 8. Troubleshooting (all seen on this machine)

| symptom | cause | fix |
|---|---|---|
| `import lerobot.policies` fails with a cv2 circular import | `opencv-python` installed beside `opencv-python-headless` | `pip uninstall opencv-python`; only the headless wheel may be present (pinned in `harness/requirements.txt`) |
| `ProcessorMigrationError` on `from_pretrained("lerobot/diffusion_pusht")` | pre-0.6 checkpoint | section 0.2: migrate once, load from the migrated directory |
| migration crashes with "Couldn't encode 84" | draccus vs tuple-typed config fields | use `harness/migrate_checkpoint.py`, not the stock script directly |
| `No space left on device` on `C:` | drive `C:` is full | everything heavy on `/mnt/d/lictor`; `CARGO_TARGET_DIR`, `HF_HOME`, `LICTOR_RESULTS` set by `harness/env.sh` |
| CUDA out of memory at B > 32 | 3 GB card; 2387 MiB peak measured at B = 32 | do not exceed B = 32 without re-measuring |
| `lictor serve` exits 2 at startup | calibration `embodiment_digest` != envelope's | recalibrate against the envelope you serve (`envelope check` prints both digests) |
| `error{code:"envelope"}` at `episode_begin` | `weights_sha256` != calibration `policy_digest` | the calibration was fitted with another checkpoint; recalibrate |
| `lictor curve` prints `REFUSED: episode index set differs ...` | a truncated or partial arm | re-run the missing indices (`--resume`) or pass `--partial` (missing indices count as failures) |
| receipts carry `ephemeral signing key` | `serve` ran without `--key` and no `$LICTOR_KEYS/key.hex` | section 0.3; `curve` refuses such arms |
| `run.py --plan` refuses a step | projection over `--budget-min` at the measured s/episode | raise the budget or shrink the step; never "optimise" by disabling the plan |
| `stats cross-check: N diffs` | a CSV number disagrees with the Python recomputation beyond tolerance | read `analysis/cross_check.json`; a bootstrap diff within 0.02 is Monte-Carlo, anything else is a bug in one of the two implementations -- do not publish the curve until resolved |
