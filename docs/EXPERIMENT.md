# lictor -- the experiment (protocol, metrics, statistics, threats to validity)

This is the paper-shaped description of the one experiment lictor makes: **the safety-vs-latency curve on
PushT**. It is written before the numbers exist and is not edited to fit them. Every number that will appear
in `docs/figures/*.svg` is produced by `lictor curve` / `lictor sweep` from signed receipts and re-checked by
`harness/analyze.py`; nothing in this document is a result. Companion documents: `docs/ARCHITECTURE.md`
sections 5-8 (the math), section 10 (the protocol this document expands), section 11 (the honest boundary);
`docs/REPRODUCE.md` (the exact command sequence); `docs/VERIFIED_FACTS.md` (the measured facts about the
machine, the checkpoint and the simulator that every cost figure below is derived from).

Vocabulary note: "protective stop", "monitored standstill", SS1/SS2/STO are terms borrowed for readability;
no conformance is claimed or tested. Wall-clock numbers in this document are statistics measured under
virtualisation; see section 9.

## 1. Question and claims

**Question.** For a frozen learned policy (`lerobot/diffusion_pusht`, a diffusion policy with a 16-step
horizon) driven through a deterministic decision plane (the fuse), how much episode-level failure does a
black-box action-space detector plus a geometric envelope catch, at what cost in task success and
interventions, as a function of the control-latency budget `d` the fuse is allowed to consume?

**What is claimed.** Three things, each with its own evidence:

1. *Enforcement.* Under injected action spikes, Tier 0 lets zero violations reach the environment in enforce
   mode (`inj-t0-d0`) versus the count scored in observe mode (`inj-obs-d0`). Figure F9.
2. *Prediction.* The conformal Tier-1 aggregate flags a fraction of the failures the fuse-off baseline
   suffers, at an episode-level false-alarm rate that the calibration's alpha bounds under exchangeability.
   The open-loop sweep (Layer A, figure F2) gives the full detector Pareto; the closed-loop arms (Layer B,
   figure F1) give the task-success delta and the intervention rate that Layer A structurally cannot.
3. *Auditability.* Every closed-loop number is recomputed from Ed25519-signed receipts whose verdict chain
   replays byte-identically on the tested target; a curve point whose ledger is broken, whose keys are mixed
   or ephemeral, or whose episode set is not the declared pool is refused, not silently reported.

**What is not claimed.** No safety rating of any kind; no real-time guarantee (section 9); no detection
ceiling above the field's; no reproduction of the published 65.4 % beyond re-measuring it under our pins
(section 10, threat 7); no defence against the experimenter who holds the signing key (section 10, threat 11).

## 2. Conditions (arms)

`arm_id = <detector>-a<alpha%>-d<d>[-async][-<escalation>]` (ARCHITECTURE 10.5). Every arm uses the same
paired seeds. `mode = observe` never substitutes an action (the trajectory is the policy's own; the fuse
scores and records); `mode = enforce` substitutes (clamp / brake / hold) and stops the episode on escalation.

| arm_id | mode | Tier 0 | Tier 1 alpha | d | exec | on_escalate | role |
|---|---|---|---|---|---|---|---|
| `calib-obs` | observe | scored | scored | 0 | sync | -- | calibration pool only (seeds 900000..900299) |
| `obs-d0` | observe | scored | scored | 0 | sync | -- | fuse-off baseline, the success labels, the Layer-A eval traces |
| `obs-d{1,2,3,5,8}` | observe | scored | scored | d | sync | -- | latency-only controls: "latency hurt me" separated from "the fuse hurt me" |
| `obs-d2-async`, `obs-d5-async` | observe | scored | scored | 2, 5 | async | -- | RTC-style comparison, fuse off |
| `t0-d0` | enforce | armed | -- | 0 | sync | terminate_fail | geometric enforcement alone; the parity gate |
| `t01-a05-d0` | enforce | armed | 5/100 | 0 | sync | terminate_fail | headline |
| `t01-a{01,10,20}-d0` | enforce | armed | 1, 10, 20 /100 | 0 | sync | terminate_fail | closed-loop alpha sweep |
| `t01-a05-d{1,2,3,5,8}` | enforce | armed | 5/100 | d | sync | terminate_fail | the budget curve |
| `t01-a05-d{2,5}-async` | enforce | armed | 5/100 | 2, 5 | async | terminate_fail | the second curve |
| `t01-a05-d0-oracle` | enforce (`envelopes/pusht.oracle.toml`) | armed | 5/100 | 0 | sync | oracle_resume | intervention-value counterfactual; exercises the AckToken path |
| `t01-a05-d0-ackonly` | enforce (`rearm = ack_only`) | armed | 5/100 | 0 | sync | terminate_fail | protective-stop-vs-e-stop accounting (optional) |
| `inj-obs-d0`, `inj-t0-d0` | observe / enforce | scored / armed | -- | 0 | sync | terminate_fail | `action_spike(p = 0.02, mag = 180)`: the enforcement demonstration |

Detector subsets for the open-loop sweep: `t0` (Tier 0 only), `t1_tce` ([tce]), `t1_stall`
([stall, path_ineff]), `t1_full` (all 8 Tier-1 features), `t01` (Tier 0 + all 8), `t01_and`
([tce|acm_neg, acc|path_ineff, stall|path_ineff]). The runtime rule is one split-conformal threshold `tau`
on the gated aggregate `s_t` with a K-of-N window (PushT default 3-of-5); Tier-0 limits are never windowed.

Every `Arm` declares `tier1: bool` and `alpha`; `hello` asserts them against the served calibration and
`lictor curve` refuses an arm whose receipts' `budget` disagrees with `run.json`, so a `t01-*` arm can never
silently run as `t0`.

## 3. Seeds, pools, pairing

- Calibration pool `900000..900299` (pilot: `900000..900099`); evaluation pool `0..499` (pilot: `0..59`).
  Provably disjoint; asserted by `run.py --check-pools` (prints `pools disjoint: ok`), refused by
  `lictor calibrate` (eval seeds) and listed by `lictor curve` (`seed_overlap_with_calibration` in the signed
  curve receipt).
- Pairing: `env.reset(seed = s)` plus `torch.manual_seed(hash64(s, chunk_idx))` before every
  `predict_action_chunk`, with `hash64(seed, chunk) = splitmix64(((seed << 32) ^ chunk) mod 2^64)`
  (test vectors `hash64(0, 0) = 0xe220a8397b1dcdaf`, `hash64(7, 3) = 0xfd323448a4497c68`,
  `hash64(900000, 37) = 0xaf19a08d78d230a5`, `hash64(499, 0) = 0xb267f5be46d03c23`). "Paired" means the
  same reset state (`init_state_digest`, bound into every receipt) and the same per-chunk noise; trajectories
  diverge after the first substituted or delayed action by design -- that divergence is the measurement.
- Two arms that disagree on `init_state_digest` for an episode index are refused at aggregation
  (`pair_mismatches`); `obs-d0` runs in the pilot and in the full run, so `verdict_chain_head` is compared for
  every (arm, seed) shared between runs (`cross_run_mismatches`) -- a free multi-seed policy-determinism check.
- Order: arm-major, seed-ascending; `ledger.jsonl` is the source of truth and its append is the last write of
  an episode, so a truncated run is always a complete prefix of paired seeds and `--resume` never re-runs an
  index with a ledger entry (crash receipts included).

## 4. Delay realisation (the x-axis)

The x-axis is `d` control steps of injected staleness, simulated time: 100 ms per step on PushT (control
period 0.1 s = 10 pymunk substeps of 0.01 s). A second top axis on F1 re-scales `d` to a 20 ms clock; it is a
re-scaling, not a measurement. Because `d` is simulated, the curve is independent of WSL2 wall-clock -- the
only way the curve is honestly measurable on this machine.

- `sync`: a chunk requested at step `t` becomes available at `t + d`; the environment is stepped with the
  last executed action repeated (hold-last) meanwhile; the delay consumes the 300-step budget. Wire labelling
  at delivery: `t_emit = t`, `idx = 0`, all 15 rows. Consequence, disclosed: consecutive deliveries are
  `8 + d` steps apart, so the Tier-1 overlap is `L = 7 - d`; `tce`/`acc` are invalid for `d >= 7` and
  degraded for `d in 3..6`. The curve partly measures feature loss there and says so: `tce_valid_frac` is
  bound into every curve receipt and drawn as the thin bar under F1(a).
- `async`: the executor keeps draining the previous chunk (7 spare actions beyond the 8 executed, so
  `d <= 7` is fully coverable); on arrival `stitch = drop` skips the new chunk's first `d` rows (the fuse
  still receives all 15 rows with `t_emit = t - d`, `idx = d`; brake feasibility starts from row `d`);
  `stitch = freeze` executes them (`t_emit = t`, `idx = 0`); beyond the spare actions, hold-last.
  Async keeps `L = 7`, so it is a second curve, drawn dashed.
- The fuse's own cost enters as `d_total = d + ceil(decide_ms / tick_ms) = d` at 10 Hz for every tier: at a
  100 ms control period the measured decide + IPC cost rounds to zero additional control steps. That is a
  statement about simulated time, not a wall-clock guarantee; the budget axis is a what-if for
  higher-rate embodiments. The measured `decide_ns` histogram is inset on F1 so the reader sees what the
  fuse costs next to what the budget buys.
- Fault injection (`harness/inject.py`, an independent `Philox(episode_seed ^ 0xFA17)` stream, always
  declared in the receipt's `fault_injection`): `action_spike(p, mag)`, `obs_dropout(p)`, `obs_noise(sigma)`,
  `chunk_truncate(p)`, `latency_spike(p, ticks)`. The headline experiment uses `action_spike(0.02, 180)` only.

## 5. Two layers

**Layer A -- open-loop detection sweep (free, CPU, deterministic, seconds).** Observe-mode runs never change
the trajectory, so the `obs-d0` traces carry the complete raw feature trace `f[t]`, `valid[t]` of every
evaluation episode and `calib-obs` the calibration pool's. `lictor sweep` recomputes TPR / FPR / lead time /
ROC-AUC / AUCPDT for every alpha and every detector subset offline, with no re-rollout, writing one line per
(alpha, detector) to `sweep.jsonl`. Points that reuse a `calibration.<alpha>.json` artefact verbatim carry
`tau_source: "artefact"`; the Layer-A-vs-Layer-B comparison is made only on those. Points where the sweep
fitted its own full-sample tau carry `tau_source: "fit"` and are never used for that comparison.

**Layer B -- closed-loop arms (expensive, GPU).** Only the arms of section 2 run in enforce mode, to measure
what Layer A structurally cannot: the task-success delta (Y2) and the intervention rate (Y3), including the
failures the fuse itself causes. The closed-loop flagged rate at alpha is reported next to the Layer-A TPR at
the same alpha as a validity test of Layer A (`analysis/layer_ab.csv`). All Layer-B numbers come from
`lictor curve` over signed receipts.

## 6. Metric definitions (per budget point; ARCHITECTURE 10.7)

Let `F` / `S` be the evaluation seeds that fail / succeed in `obs-d0` (the labels). `n_fail_baseline = |F|`
and `n_succ_baseline = |S|` are bound into every curve receipt. "Three lines per budget point, or the curve
is dishonest": every point on F1 carries a detection line, a task-success line and an intervention line.

| id | metric | definition |
|---|---|---|
| Y1a | `flagged_rate` (TPR) | of `F`, the fraction where the arm reached `Braking` / `Held` / `Escalated` (`first_stop_tick` non-null) |
| Y1a' | `averted_rate` | of `F`, the fraction the arm converted to success -- reported separately: "flagged" and "averted" are different goods |
| Y1b | `false_trip_rate` (FPR) | of `S`, the fraction where the arm stopped -- the empirical rate alpha is supposed to bound |
| Y1c | `lead` | `t_fail - first_stop_tick` over `F`, with `t_fail = min{t : c*_T - c*_t < eps_prog}`, `c*_t = max_{u <= t} coverage_u`, `eps_prog = 0.02`, from the paired `obs-d0` trace; mean, p50, p10; AUCPDT (VLA-FAIL: `(1/H) sum_{L=0}^{H-1} |{lead >= L}| / |F|`); F1-timeliness hypervolume (ActProbe) |
| Y2 | `success_rate` | under `terminate_fail`, with BOTH deltas: `delta_vs_baseline` (vs `obs-d0`: the fuse cost) and `delta_vs_latency_control` (vs `obs-d{d}`: the latency-only control at the same d), each a `DeltaCi` = paired difference, paired-bootstrap 95 % CI, exact McNemar p and the discordant counts |
| Y3 | `intervention_rate`, `intervention_tick_frac`, `escalation_rate` | episodes with >= 1 substitution; substituted ticks / ticks; episodes reaching `Escalated` |
| M8 | `violations_reached_env` | Tier-0 violations not substituted (observe / injection arms): the enforcement pillar's own number |
| M9-10 | `roc_auc`, `bacc` | over per-episode `max_s` vs the `obs-d0` label; balanced accuracy at tau |
| M11 | latency | p50 / p90 / p99 / p99.9 / max of `decide_ns` and `io_ns` from the timing chains, every number carrying `latency_label` |
| M12 | `delta_vs_*` | see Y2; both inside the signed curve receipt and `summary.csv` |
| M13 | `tce_valid_frac` | fraction of ticks with a valid `tce`; sync `d >= 7` -> 0 (the feature-loss confound, disclosed per point) |

**The point-of-no-return proxy.** `t_fail` is a PROXY for the moment an episode became unrecoverable: the
first tick after which the running maximum of coverage never improves by more than `eps_prog`. It is labelled
as a proxy everywhere and its sensitivity over `eps_prog in {0.01, 0.02, 0.05}` is drawn on F3. Known
artefact, disclosed on F3: an episode with no progress at all has `t_fail = 0`, so every detector shows a
negative lead on it; that is a property of the proxy, not a late detection. TCE is a lagging feature (chunk
k's disagreement is known only when chunk k + 1 arrives, 8 steps later); the lag is inside the reported leads,
not subtracted.

## 7. Statistics

- **Rates.** Every rate carries an exact two-sided 95 % Clopper-Pearson interval computed by `lictor curve`
  / `lictor sweep` from the counts (n = 60 -> about +-13 pp at p = 0.5; n = 100 -> about +-10 pp;
  n = 500 -> about +-4.5 pp; all stated on the figure next to the point).
- **Paired deltas.** `success(arm) - success(control)` paired by episode index, with a percentile bootstrap
  over pairs (10 000 resamples, splitmix64 seed 20260830 in `lictor-calib`) and the exact two-sided McNemar
  test on the discordant pairs `(b, c)` (two-sided binomial tail, `p = min(1, 2 P(X <= min(b, c)))`,
  `X ~ Bin(b + c, 1/2)`). Both deltas -- vs `obs-d0` and vs `obs-d{d}` -- are always reported: the first is
  what the fuse costs, the second separates the fuse's cost from the staleness the budget itself injects.
- **Cross-check, not source.** `harness/stats.py` recomputes Clopper-Pearson (regularised incomplete beta
  via a Lentz continued fraction on `math.lgamma`, quantile by bisection), McNemar, the bootstrap
  (`numpy.random.default_rng(20260830)`, 10 000 resamples, on the 2x2 table reconstructed from the
  CSV counts) and the sweep's `bacc`; `harness/analyze.py` prints `stats cross-check: ok` or the diffs. CP
  and McNemar must agree to 1e-9; the two bootstraps draw from different RNG streams and are compared within
  a Monte-Carlo band of 0.02. analyze.py never computes a delta itself: `delta_vs_baseline*` and
  `delta_vs_latency_control*` are copied verbatim from `summary.csv`.
- **Pilot points.** `lictor curve` refuses n < 100 without `--allow-small`; with it the point is bound
  `small_n: true`, the CSV row starts with `pilot = 1`, `figures.py` draws it hollow and never runs a line
  through it, and no headline number is ever taken from it.
- **Degenerate alphas.** With `k = ceil((n_calib + 1)(1 - alpha))` in integer arithmetic, `k > n_calib`
  means the quantile does not exist and `tau = +inf` (the detector never fires). Such points are marked
  degenerate on F2 / F8 and are facts about the calibration size, not data. Worked numbers: n_calib = 137
  (2-way split of 196 successes): alpha 1/100 -> k = 137 (tau = the maximum), 2/100 -> 136, 5/100 -> 132,
  10/100 -> 125, 20/100 -> 111, 5/1000 -> 138 > 137 -> +inf; the pilot's n_calib ~ 45 makes 1/100 and 2/100
  degenerate.
- **Conformal validity.** The bound `P(fire | nominal) <= alpha` holds exactly at K = 1 only when the
  standardisation (center / scale) was fitted on episodes disjoint from those tau was taken over
  (`--split 3`); under the default 2-way split it is approximate. The empirical held-out firing rate at the
  configured K-of-N and at K = 1 is bound into `calibration.<alpha>.json` and plotted against nominal alpha
  with the diagonal on F7. A point far above the diagonal means exchangeability is broken; it is reported,
  not tuned away.
- **Multiple comparisons.** The alpha grid `{1, 2, 5, 10, 20} / 100`, the d grid `{0, 1, 2, 3, 5, 8}` and
  the detector subsets are fixed here, before any data; no correction is applied and none is claimed. The
  headline point is `t01-a05-d0`, chosen a priori.
- **The parity gate** (ARCHITECTURE 10.3): `t0-d0` vs `obs-d0` on the first 60 paired evaluation seeds must
  give McNemar `p > 0.05` and overlapping Clopper-Pearson intervals before any curve point is reported; a
  fitted envelope that costs measurable success is too tight and is refit with more slack, every refit
  recorded in `docs/envelope_fit_report.md`.

## 8. Figures (`docs/figures/*.svg`, `harness/figures.py`, matplotlib only)

| id | file | inputs | what it shows |
|---|---|---|---|
| F1 | `curve-safety-latency.svg` | `curve/summary.csv`, `curve/<arm>.json`, `lictor bench --csv` | three stacked panels vs d: (a) flagged / averted / false-trip with `tce_valid_frac` bar, (b) success for `t01-a05-d*`, `t0-d*`, `obs-d*`, (c) intervention + escalation; twin top axes (100 ms and 20 ms re-scaling); sync solid, async dashed; pilot hollow; `decide_ns` inset |
| F2 | `pareto-detection.svg` | `sweep.jsonl` | Layer A: TPR vs FPR over the alpha grid per detector subset, ROC-AUC in the legend; held-out FPR vs alpha |
| F3 | `lead-time.svg` | per-episode leads (`--leads`) or the sweep quantiles | CDF of lead before the PoNR proxy per alpha, the proxy definition, the eps sensitivity, the `t_fail = 0` artefact |
| F4 | `score-bands.svg` | `analyze.py scores` (ledger + ticks of one observe arm) | median +- IQR of `s_t` for successful vs failing episodes, tau overlaid |
| F5 | `latency-hist.svg` | `lictor bench --csv`, `microbench.json`, `summary.csv` | `decide_ns` histogram, log-x, p50 / p99 / p99.9 / p99.99 / max; `io_ns` beside it; cyclictest baseline panel; the WSL2 sentence verbatim |
| F6 | `success-delta.svg` | `summary.csv`, `curve/<arm>.json` | McNemar contingency per arm vs `obs-d0` and vs `obs-d{d}` |
| F7 | `cp-validity.svg` | `calibration.<alpha>.json`, `sweep.jsonl` | nominal alpha vs empirical held-out FPR (K-of-N and K = 1) with the diagonal |
| F8 | `calib-economy.svg` | `sweep.jsonl` rows at n_calib in {25, 50, 100, all} | tau stability vs n_calib, degenerate alphas marked |
| F9 | `enforcement.svg` | `summary.csv`, the demo capture | `violations_reached_env` for `inj-obs-d0` vs `inj-t0-d0`; the tamper-demo terminal capture |

`figures.py --from-fixture` renders all nine from a synthetic sample embedded in the script so CI renders
without a run; those renders are samples, never results, and are not committed under `docs/figures/`.

## 9. Compute budget -- MEASURED, and what it does to the schedule

Every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement.

| quantity (GTX 1060 3GB, fp32, lerobot 0.6.1, torch 2.7.1+cu126) | measured | source |
|---|---|---|
| `import lerobot, torch` (drvfs venv) | 12-16 s | VERIFIED_FACTS |
| policy load from the migrated checkpoint | 8.3 s | VERIFIED_FACTS |
| one chunk inference (100 DDPM steps), B = 1 | mean 3.46 s, max 6.3 s | VERIFIED_FACTS smoke |
| one 300-step episode, B = 1, ~38 chunk inferences | **153.9 s** | VERIFIED_FACTS smoke (seed 0) |
| denoiser batch scaling, ms per env-chunk | B=1 2843, B=2 1389, B=4 705, B=8 376, B=16 213, **B=32 123** | VERIFIED_FACTS bench_batch |
| peak VRAM at B = 32 | 2387 MiB (cap ~3 GB; B = 64 untested) | VERIFIED_FACTS bench_batch |
| lerobot's published `eval_ep_s` | 1.46 s (a batched-eval number, not single-env) | eval_info.json |

Consequences, stated plainly:

- At B = 1 the pilot of 220 episodes (`calib-obs` x 100, `obs-d0` x 60, `t01-a05-d0` x 60) costs
  220 x 153.9 s = 9.4 h, and one 500-episode arm costs ~21 h. The 12-20 s/episode assumption behind
  ARCHITECTURE 10.8's one-hour pilot table is superseded by the measurement; `run.py --plan --budget-min 60`
  will refuse every pilot step at B = 1, which is the intended behaviour.
- At B = 32 the denoiser is launch-bound and batching gives ~23x throughput: 38 x 123 ms = 4.7 s of GPU time
  per episode, so a 100-episode arm is ~8 min and a 500-episode arm ~40 min of GPU time, before environment
  stepping and fuse IPC (unmeasured at the time of writing; `microbench.json` records them when the harness
  runs). VERIFIED_FACTS therefore requires the harness to batch 32 environments per policy forward.
- Batching interacts with pairing (section 3): DDPM draws the prior and every scheduler step's variance noise
  from the global torch RNG per batch, so with a plain batched forward the noise an episode receives depends
  on the batch it shares. Paired seeding survives batching only if the per-episode noise is made explicit
  (the prior via `predict_action_chunk(batch, noise = ...)` and the scheduler noise via an explicit generator
  per episode) or if the batch composition is identical across arms at every chunk boundary -- and arms
  terminate episodes at different times. Which of these the harness does is WP-8's implementation decision;
  the experiment does not trust it: `init_state_digest`, `pair_mismatches` and `cross_run_mismatches` make
  any residual divergence visible in the signed curve receipt, and the day-0 gate runs `obs-d0` twice on
  seeds 0..2 and requires byte-identical action sequences. If batching breaks pairing, McNemar is invalid and
  the fallback is B = 1 with the schedule above.
- The frozen wire protocol is single-episode per `lictor serve` child; multiplexing B episodes is done by
  running one child per environment slot (the wire section already says "one serve child per harness
  worker"), which keeps every receipt, ticks file and ledger exactly as specified.

## 10. Threats to validity

1. **The PoNR proxy.** `t_fail` is defined from the coverage running maximum, not from ground truth about
   recoverability; a different `eps_prog` moves every lead. Mitigation: the sensitivity band on F3, the
   `t_fail = 0` artefact disclosed, AUCPDT reported next to the quantiles.
2. **Exchangeability.** alpha bounds the episode-level false-alarm rate only under exchangeability with the
   calibration pool -- this policy, this task, this distribution; the 2-way split makes the bound
   approximate; K-of-N makes it a conservative upper bound. Mitigation: F7 with the held-out rates at K-of-N
   and K = 1, `--split 3` available, the seed pools disjoint and declared.
3. **WSL2.** As stated above every timing table: every wall-clock figure in this document was measured under WSL2 virtualisation (Hyper-V utility VM, non-RT host) and is not a real-time measurement.
   Latency histograms are statistics under
   virtualisation and every one of them carries the `latency_label` sentence; the d-axis is simulated time
   and therefore virtualisation-independent. Algorithmic determinism, zero allocations, fixed iteration
   bounds and byte-identical replay are verifiable properties on the tested target (x86_64 SSE2); they are
   not claims about other targets.
4. **A single policy, a single task.** One checkpoint (`lerobot/diffusion_pusht`), one simulator, 2-D
   end-effector position control; nothing here transfers by itself to another embodiment. The Tier-1 features
   depend on the chunk specification (below), and the envelope on this workspace.
5. **Simulator fidelity and pins.** gym-pusht 0.1.6 / gymnasium 1.3.0 / pymunk 6.11.1 / numpy are newer than
   the pins the published 65.4 % was measured under; the 65.4 % gate is a smoke check on 60 seeds and the
   n = 500 `obs-d0` run is the real reproduction, reported whichever way it comes out. The braking model is
   contact-free (contact only decelerates the agent in PushT; conservative for the box constraint -- a
   modelling assumption, not a proof); it starts from the simulator's own `info["vel_agent"]`.
6. **Chunk-spec dependence.** TCE/ACC exist only because `n_action_steps` is overridden to
   `horizon - n_obs_steps + 1 = 15` while the harness executes 8, giving an overlap `L = 7`; the executed
   8-prefix is asserted bit-exact against the stock configuration (`test_h15`, the six runtime asserts of
   ARCHITECTURE section 0). Sync delay reduces `L` to `7 - d`, disclosed via `tce_valid_frac`. A policy with a
   different horizon or no overlap has no TCE.
7. **The published baseline.** 65.4 % is lerobot's number; we re-measure it. If our `obs-d0` differs, the
   curve is still internally valid (every delta is paired against our own baseline) but the absolute level
   is ours, not theirs.
8. **Batching vs pairing** (section 9): if the harness batches and pairing is not exact, the paired tests
   are invalid; the receipts make the breakage visible and the fallback is unbatched.
9. **Small pilot n.** The pilot point (n = 60) has a Clopper-Pearson half-width of about 13 pp; it is drawn
   hollow, labelled pilot, and never a headline.
10. **The envelope fit and the parity gate.** A too-tight envelope destroys the baseline; a too-loose one makes
    Tier 0 decorative. The fit at empirical p99.9 x 1.25 and the parity gate (`t0-d0` vs `obs-d0`) are the
    mitigation; the injection arms carry the enforcement demonstration whichever way parity comes out.
11. **Trust model.** The fuse verifies mode, dims, digests, its own binary hash and every tick; everything in
    `binding`, `run`, `budget.delay_steps`, `fault_injection`, `inputs` and `outcome.success` is the host's
    declaration, signed by proxy. A receipt defends against post-hoc edits by anyone without the signing key
    and against accidental corruption; it does not defend against the experimenter, who holds the key and can
    rebuild and re-sign a ledger. Truncation of a declared pool is detected by `lictor curve`; an external
    witness (Rekor / RFC 3161) is roadmap.
12. **Contaminated calibration.** Calibration uses successes only; a sensor that lies or a calibration set
    contaminated with failures defeats the predictive tier; the receipt records the pool and its labels so
    the contamination is at least auditable.
13. **Semantic failures inside the envelope.** PushT's natural failures are semantic (the block does not reach
    the goal) and happen inside a legal workspace; Tier 0 is a guard, not the main event, and the predictive
    tier carries the curve. Expect and report the field's ~72-88 % detection ceiling.

## 11. Reproducibility appendix

- Pins (bound into every receipt's `run.env` / `run.host` / `binding.policy`): `lerobot == 0.6.1`,
  `torch == 2.7.1+cu126`, `gym-pusht`, `gymnasium`, `pymunk`, `numpy` from `importlib.metadata`; the
  checkpoint's HF revision and the sha256 of the migrated `model.safetensors` (`weights_sha256`, which must
  equal the calibration's `policy_digest` at `episode_begin`); `normalization_migrated = true`,
  `migration_script = lerobot/processor/migrate_policy_normalization.py` (driven by
  `harness/migrate_checkpoint.py`, which carries the tuple-typing fix the stock script needs).
- Determinism environment (`harness/env.sh`): `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
  CUBLAS_WORKSPACE_CONFIG=:4096:8 SDL_VIDEODRIVER=dummy`; `torch.use_deterministic_algorithms(True,
  warn_only=True)`, `cudnn.deterministic = True`, `cudnn.benchmark = False`, fp32, `policy.eval()`.
- Artefacts per run (`$LICTOR_RESULTS/<run>/`): `run.json` (arms with `tier1` / `alpha`, seeds, pins,
  commands, lictor build; its sha256 is bound into every curve receipt), `microbench.json`, `sweep.jsonl`,
  `calibration.<alpha>.json`, `curve/<arm>.json` + `curve/summary.csv`, and per arm `ledger.jsonl`,
  `receipts/`, `ticks/`, `timing/`, `traces/`. Analysis products under `<run>/analysis/`.
- Verification of any receipt, with the key pinned out of band: `python adapters/verify_receipt.py
  <receipt> --pubkey <hex> [--ticks <ticks.jsonl>] [--ledger <ledger.jsonl>]` (stdlib only), or
  `lictor verify <receipt> --ticks ... --pubkey <hex>`; `lictor ledger verify <ledger.jsonl>`;
  `lictor replay --repeat 40 <trace> --envelope ... --calibration ... --mode enforce --expect <head>`.
- What is committed: `bench/fixtures/`, `docs/figures/*.svg` (from a real run, never from the fixture),
  `envelopes/pusht.toml` + `envelopes/pusht.oracle.toml`, `docs/envelope_fit_report.md`, the `research/`
  memos. Keys never live in the repository (the two committed keys are test keys, listed in `SECURITY.md`).
- The exact command sequence, with the measured costs above every timing table: `docs/REPRODUCE.md`.
