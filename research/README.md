# research -- the memos behind every number cited in the README

Every number the README cites has a memo in this directory: one file per number, written by the integration
package (WP-13) from the sources listed here, and never edited by hand afterwards. A memo states the claim
exactly as the README cites it, the primary source with the date it was fetched, the figure with its
denominator (episodes, trials, tasks), who measured it (third party, self-reported, or this machine), and
what would falsify it. Anything that does not have a memo does not get cited.

The URLs below are the ones recorded in `docs/ANALYSIS.md` sec 9 (fetched 2026-08-29 .. 2026-08-31).
The `status` column is the honesty label the README carries next to the number.

## 1. Numbers cited in "The hole"

| as cited in the README | memo | primary sources | status |
|---|---|---|---|
| pi0.5, the strongest policy on RoboChallenge, averages 43.7 % success on a real-robot benchmark | `robochallenge-pi05.md` | <https://arxiv.org/pdf/2510.17950> | third-party |
| pi0-FAST-DROID in the wild: 42.3 % average task progress over 300+ trials; freezing, spatial misjudgement / collisions, prompt sensitivity; calls for runtime monitoring | `penn-pi0-in-the-wild.md` | <https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/> | third-party |
| RoboArena: seven institutions, 600+ double-blind paired real-robot episodes; self-run evaluations misrank generalist policies | `roboarena.md` | <https://arxiv.org/abs/2506.18123> | third-party |
| GEN-1.5 (19.08.2026): 59 % +- 10 % average success over 10 real-robot tasks (~41 % failure); one-shot in-context; closed weights, partnership only; makers say they cannot explain why it works | `gen-1.5.md` | <https://generalistai.com/blog/gen-1.5>, <https://the-decoder.com/gen-1-5-generalist-ai-teaches-robots-new-tasks-from-a-single-demo/>, <https://www.marktechpost.com/2026/08/24/generalist-ai-releases-gen-1-5-a-robot-foundation-model-that-learns-new-tasks-from-one-3-12-second-demo/> | self-reported, unverified (see sec 5) |
| the ~90 % problem: models at ~95 % on benchmarks land at 60-80 % on real tasks; interventions paper over the gap | `ninety-percent-problem.md` | <https://adamohq.com/blog/the-90-percent-problem-in-humanoid-deployment> | industry commentary |
| 3Laws Supervisor, the one shipping policy-agnostic runtime safety product, acquired by Amazon on 2026-05-05 and taken off-market | `3laws-acquisition.md` | <https://pitchbook.com/profiles/company/501876-73>, <https://escalatepr.com/case-study/robotics-pr-3laws-robotics-amazon-acquisition/>, <https://docs.nav2.org/tutorials/docs/using_isaac_perceptor.html> | third-party record |
| NVIDIA Halos for Robotics (June 2026) is IGX-locked infrastructure that does not wrap policies; GR00T's model card puts guardrails on the deployer | `nvidia-halos.md` | <https://developer.nvidia.com/blog/inside-nvidia-halos-for-robotics-a-full-stack-functional-safety-system-for-physical-ai/>, <https://nvidianews.nvidia.com/news/nvidia-announces-halos-for-robotics-the-industrys-first-full-stack-safety-system-for-physical-ai>, <https://huggingface.co/nvidia/GR00T-N1.7-3B> | vendor documentation |
| LeRobot's official safety story is a Python `torch.clamp`: the `EEBoundsAndSafety` clipping step plus a joint-limit discovery script | `lerobot-safety-story.md` | <https://huggingface.co/docs/lerobot/hilserl>, <https://huggingface.co/docs/lerobot/en/env_processor> | vendor documentation |
| every published VLA failure detector is research Python (SAFE, FAIL-Detect, Sentinel, FIPER, ActProbe, VLA-FAIL, Hide-and-Seek, SAFECAST, PATCH, RC-NF, Pre-VLA, TAIL-Safe); the field's detection ceiling is ~72-88 % bACC / ROC-AUC | `detector-landscape.md` | <https://arxiv.org/abs/2506.09937>, <https://arxiv.org/abs/2503.08558>, <https://arxiv.org/abs/2410.04640>, <https://arxiv.org/abs/2510.09459>, <https://arxiv.org/abs/2606.08508>, <https://arxiv.org/abs/2606.21386>, <https://arxiv.org/abs/2605.30834>, <https://arxiv.org/abs/2608.04246>, <https://arxiv.org/html/2606.16690>, <https://arxiv.org/html/2603.11106>, <https://arxiv.org/html/2605.22446>, <https://arxiv.org/html/2605.01195> | papers; ceiling is a reading across them, not a single source |
| a Feb 2026 position paper with Amazon / NVIDIA co-authors argues modular external guardrails for foundation-model robots are missing | `guardrails-position-paper.md` | <https://arxiv.org/html/2602.04056> | paper |

## 2. Numbers and lineages cited in "What it is" and the honest boundary

| as cited | memo | primary sources | status |
|---|---|---|---|
| temporal consistency error (chunk-overlap disagreement) and chunk magnitude as black-box, action-space failure signals; ~3 ms/step; split-CP quantile | `actprobe-tce.md` | <https://arxiv.org/html/2606.08508>, <https://github.com/air-embodied-brain/actprobe> | paper + code (Apache-2.0) |
| ACC: velocity-normalised chunk-overlap disagreement; AUCPDT as the timeliness metric | `vla-fail-acc-aucpdt.md` | <https://arxiv.org/abs/2606.21386> | UNVERIFIED (anonymous project page; see sec 5) |
| STAC / Sentinel: statistical distance between overlapping segments of consecutive chunks; sampling-dependent | `sentinel-stac.md` | <https://arxiv.org/abs/2410.04640>, <https://github.com/agiachris/sentinel> | paper + code (MIT) |
| FIPER: AND-gating orthogonal scores to suppress benign-OOD false alarms; calibrates on 10-50 successful rollouts | `fiper-and-gate.md` | <https://arxiv.org/abs/2510.09459>, <https://github.com/utiasDSL/fiper> | paper + code (MIT) |
| FAIL-Detect: logpZO / RND scores with functional conformal bands (the Tier-2 `ext` channel contract) | `fail-detect.md` | <https://arxiv.org/abs/2503.08558>, <https://github.com/CXU-TRI/FAIL-Detect> | paper + code (licence unconfirmed) |
| SAFE: tiny probe on VLA hidden states, +0.73 ms, functional CP (the white-box sidecar the `ext` slots are for) | `safe-probe.md` | <https://arxiv.org/abs/2506.09937>, <https://github.com/vla-safe/SAFE> | paper + code |
| conformal thresholds calibrated offline make the runtime decision a deterministic comparison (the calibration contract) | `conformal-calibration-contract.md` | <https://arxiv.org/abs/2506.09937>, <https://arxiv.org/html/2503.08558v3>, <https://arxiv.org/html/2501.04823> | papers |
| "every plan carries a verified fail-safe braking manoeuvre" (ARMTD) and the predictive stopping-distance cap of the UR e-Series are the ancestors of the brake-feasibility invariant | `brake-invariant-lineage.md` | <https://arxiv.org/abs/2002.01591>, <https://www.universal-robots.com/media/1804320/e-series-functional-safety.pdf>, <https://www.universal-robots.com/manuals/EN/HTML/SW10_6/Content/prod-usr-man/hardware/arm_UR20/stopping_time_n_distances/stoppingTimeAndDistance_UR20_en.htm> | paper + vendor documentation; lineage only, no conformance claimed |
| Ruckig as the canonical jerk-limited stop generator (`BrakeKind::JerkLimited` closed form) | `ruckig.md` | <https://arxiv.org/abs/2105.04830> | paper |
| RTC: actions inside the committed window are frozen because they are guaranteed to execute; the latency-injection methodology behind the d-axis | `rtc-latency-injection.md` | <https://arxiv.org/abs/2506.07339>, <https://www.pi.website/research/real_time_chunking>, <https://arxiv.org/abs/2605.08168> | papers |
| per-action CBF filtering causes jitter; chunk-level aggregate constraints are the 2026 direction | `chunk-level-verification.md` | <https://arxiv.org/abs/2607.29569>, <https://arxiv.org/html/2607.26789>, <https://arxiv.org/html/2605.22446>, <https://arxiv.org/abs/2606.23686> | papers |
| ISO 10218-1:2025 vocabulary (monitored standstill 5.5.5, stopping-time / stopping-distance limiting 5.5.6 / 5.5.7, clause 7.5.12 documentation duty) and the SS1 / SS2 / STO stop categories -- terms borrowed for readability; no conformance is claimed or tested | `standards-vocabulary.md` | <https://www.automate.org/robotics/blogs/updated-iso-10218-faq>, <https://arxiv.org/pdf/2602.17822>, <https://eshield.pl/en/knowledge/sto-ss1-ss2-safely-stop-machine/>, <https://www.therobotreport.com/now-available-full-403-page-ansi-a3-r15-06-2025-robot-safety-standard/> | secondary sources; clause details UNVERIFIED against the purchased text (sec 5) |
| the SSM protective-distance equation makes reaction time a first-class safety parameter (why fuse latency is reported in us, never hidden) | `ssm-reaction-time.md` | <https://pmc.ncbi.nlm.nih.gov/articles/PMC5117641/>, <https://www.diag.uniroma1.it/deluca/pHRI_elective/ISO_TS_15066_2016_en.pdf>, <https://www.iso.org/standard/80590.html> | paper + standard mirrors |
| EU AI Act Article 9 continuous-monitoring obligations (2027) and procurement guidance that asks for an intervention log | `regulatory-pull.md` | <https://arxiv.org/html/2605.28726>, <https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/> | paper + industry guidance |
| the pickle-over-gRPC deserialisation issue in LeRobot's async inference path (CVE-2026-25874), which lictor does not touch in milestone 1 | `lerobot-cve-2026-25874.md` | <https://chocapikk.com/posts/2026/lerobot-pickle-rce/>, <https://github.com/huggingface/lerobot/issues/3047> | third-party write-up + open issue |

## 3. Numbers cited in "Determinism", "Build" and the WSL2 label

| as cited | memo | primary sources | status |
|---|---|---|---|
| Rust `f64` semantics (RFC 3514): strict binary64, no FMA contraction unless asked; the basis of "bit-identical on the tested target" | `float-determinism.md` | <https://rust-lang.github.io/rfcs/3514-float-semantics.html>, <https://github.com/rust-lang/rust/issues/128288>, <https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/>, <https://gafferongames.com/post/floating_point_determinism/> | RFC + engineering write-ups |
| the `libm` / Rapier `enhanced-determinism` pattern for cross-target reproducibility | `libm-rapier-pattern.md` | <https://docs.rs/libm/latest/libm/>, <https://rapier.rs/docs/user_guides/rust/determinism/> | library documentation |
| WSL2 is a Hyper-V utility VM scheduled by a non-RT host; a PREEMPT_RT guest kernel removes guest-side latency only | `wsl2-not-real-time.md` | <https://github.com/microsoft/WSL2-Linux-Kernel>, <https://github.com/Locietta/xanmod-kernel-WSL2/issues/88>, <https://github.com/pep248/WSL_realtime_kernel>, <https://en.wikipedia.org/wiki/PREEMPT_RT> | kernel sources + community builds |
| coordinated omission: why the latency histogram is sampled open-loop against the tick schedule | `coordinated-omission.md` | <https://highscalability.com/your-load-generator-is-probably-lying-to-you-take-the-red-pi/>, <https://www.scylladb.com/2021/04/22/on-coordinated-omission/>, <https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/cyclictest/start> | engineering write-ups + tool docs |
| the 50-200 us class expected on PREEMPT_RT with mlockall + SCHED_FIFO + isolcpus (conditional, not measured) | `preempt-rt-expectation.md` | <https://shuhaowu.com/blog/2022/04-linux-rt-appdev-part4.html>, <https://people.redhat.com/williams/latency-howto/rt-latency-howto.txt>, <https://arxiv.org/abs/2603.07442>, <https://www.codethink.co.uk/articles/2026/copper-rs-real-time-robotics.html> | write-ups + prior art (LITHE < 100 us WCET on a Pi 4B; Copper-rs hexapod 730 us max); lictor's own number is TBD |
| Copper-rs and dora-rs as the distribution channels of roadmap item 12 | `copper-dora.md` | <https://github.com/copper-project/copper-rs>, <https://github.com/dora-rs/dora>, <https://fosdem.org/2026/schedule/event/SK8EGJ-copper-rust-robotics-runtime/> | repositories |
| bulla / sbx: the receipt, verdict, history and rendering lineage credited in "Authorship" | `bulla-sbx-lineage.md` | <https://github.com/RARS-oss/bulla>, <https://github.com/RARS-oss/sbx> (private repositories, pushed 2026-08-23) | the author's own prior work |

## 4. Numbers measured on this machine

| as cited | where | status |
|---|---|---|
| `lerobot/diffusion_pusht` publishes `pc_success = 65.4` over 500 episodes and `eval_ep_s = 1.46` on the reference GPU (batched eval); config `horizon=16 n_action_steps=8 n_obs_steps=2`, 100 DDPM steps | `hf-diffusion-pusht.md` (WP-13), from <https://huggingface.co/lerobot/diffusion_pusht> `eval_info.json` + `config.json`, fetched 2026-08-30 | published by the checkpoint's authors; re-measured by the `obs-d0` x 500 run |
| one 300-step episode at B=1: 153.9 s wall; 44 chunk inferences, mean 3462 ms, max 6278 ms; policy load 8.3 s; `import lerobot, torch` 12-16 s | [`day0-smoke/smoke.log`](day0-smoke/smoke.log), produced by [`day0-smoke/smoke_pusht.py`](day0-smoke/smoke_pusht.py) | measured here, 2026-08-31, GTX 1060 3GB fp32, WSL2, one seed |
| denoiser ms per env-chunk vs batch size: 2843 (B=1) .. 123 (B=32), peak VRAM 2179-2387 MiB | [`day0-smoke/bench_batch.log`](day0-smoke/bench_batch.log), produced by [`day0-smoke/bench_batch.py`](day0-smoke/bench_batch.py) | measured here, 2026-08-31 |
| the `n_action_steps = 15` override leaves the executed 8-row prefix bitwise identical to the stock chunk (max diff 0.0) and the stock chunk is bit-reproducible under `torch.manual_seed` | same log | measured here, 2026-08-31 |
| the stock lerobot 0.6.1 migration script crashes on tuple-typed config fields; the patched driver works | [`day0-smoke/migrate_patched.py`](day0-smoke/migrate_patched.py), `docs/VERIFIED_FACTS.md` | reproduced here, 2026-08-31 |
| gym_pusht PD constants `k_p = 100`, `k_v = 20`, `dt = 0.01`, 10 substeps; `info["vel_agent"]` every step; `coverage` only in `step()` | `docs/ARCHITECTURE.md` sec 0, read from the installed `gym_pusht/envs/pusht.py` (0.1.6) | read from source here, 2026-08-31 |
| everything the README marks **TBD after the pilot**: verdict p99, `replays 40/40`, the curve, the parity gate, `violations_reached_env` | `results/<run>/` via `lictor bench`, `lictor replay`, `lictor curve`; memos `pilot-*.md` written by WP-13 after the run | not yet measured; never estimated |

The day-0 directory carries its own [README](day0-smoke/README.md). Those files are the raw evidence and are not edited.

## 5. UNVERIFIED -- claims that appear in the sources and are NOT cited as fact

Carried over from `docs/ANALYSIS.md` (its own UNVERIFIED markers), plus what the plan added. None of these is used in the README as a number.

- GEN-1.5's 59 % +- 10 % is self-reported by Generalist AI; no independent verification exists (the-decoder says so explicitly); what the std dev is computed over (tasks vs trials) is not stated. The README quotes it once, labelled self-reported and unverified.
- Press-reported ~99 % success for the earlier GEN-0 / GEN-1 on trained tasks.
- UR10e Cat 0 stop time of about 400 ms (snippet-level only).
- ISO 10218-1:2025 robot-class thresholds and clause details rest on one secondary source (<https://arxiv.org/pdf/2602.17822>); confirm against the purchased standard before quoting any clause normatively. lictor quotes no clause normatively.
- NVIDIA developer-forum latencies for pi0.5 / GR00T on Thor and RTX 5090 (<https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788>).
- VLA-FAIL (ACC, AUCPDT): anonymous project page only; the metric definition is used, the reported numbers are not.
- SAFECAST's exact gains over SAFE (extracted figures look misparsed).
- Voyage Robotics (one VC listicle; details unknown).
- The RTX 3050 Ti SmolVLA fine-tuning anecdote (low confidence).
- The absence of published failure-detection results on SmolVLA specifically (absence of evidence).
- The plan's pre-measurement expectation of 12-20 s per PushT episode at B=1 on the GTX 1060 -- superseded by the measured 153.9 s (sec 4); recorded here so the correction is visible.
- The 1-vs-2-GPU-worker throughput, the GPU determinism probe (seeds 0..2 twice), observe-mode identity, the 65.4 % reproduction under the installed pins, and every latency number -- all gated on `harness/microbench.py` and TBD until it prints PASS.

## 6. How a memo is written (for WP-13)

One file per row above, ASCII only, no emojis, named as in the `memo` column. Each memo has exactly these headings: `Claim as cited` (the README sentence, verbatim), `Source` (URL, title, authors or organisation, date fetched), `Figure` (the number, its denominator, the exact table or sentence it comes from), `Independence` (third-party / self-reported / measured here), `Caveats` (what the source itself says against the number), `Falsifier` (what observation would make the README sentence wrong). A memo never rounds a number the source did not round, and never adds a number the source did not give. The banned-vocabulary grep in `CONTRIBUTING.md` runs over `research/*.md` too.
