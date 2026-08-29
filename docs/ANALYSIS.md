# lictor — FULL ANALYSIS

**A deterministic real-time safety fuse over any learned robot policy: failure prediction + geometric limit enforcement + human escalation.**

Author: RARS-oss (author of bulla/sbx). Synthesis of 9 research reports. Date: 2026-08-29.

Honesty convention used throughout: every number is cited inline; claims that could not be traced to a primary source are marked **UNVERIFIED**. Self-reported vendor numbers are labeled as such. This document seeds the dissertation; nothing below should require retraction.

---

## 1. Executive summary

Frontier vision-language-action (VLA) policies are being pushed toward deployment at reliability levels that would be unacceptable in any other safety-relevant domain, by vendors who by their own admission cannot explain how their models work, through access models (closed weights, partnership-only) that make inspection impossible. The best third-party number available — π0.5, the strongest model on the RoboChallenge real-robot benchmark — is **43.7% average success** (<https://arxiv.org/pdf/2510.17950>). Penn's in-the-wild study of pi0-FAST-DROID measured **42.3% average task progress** over 300+ trials and explicitly calls for runtime monitoring and safety layers, noting the policy "can accidentally strike humans" (<https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/>).

The research world has responded with failure *detectors* — SAFE, FAIL-Detect, Sentinel, FIPER, and a dense 2026 wave — all research Python, none productized. The one shipping policy-agnostic runtime safety product, 3Laws Supervisor, was acquired by Amazon on 2026-05-05 and taken off-market (<https://pitchbook.com/profiles/company/501876-73>). NVIDIA's Halos for Robotics (June 2026) is a hardware/OS/infrastructure play locked to IGX silicon; it does not wrap policies (<https://developer.nvidia.com/blog/inside-nvidia-halos-for-robotics-a-full-stack-functional-safety-system-for-physical-ai/>). The precise unoccupied slice: **an independent, hardware-agnostic, deterministic real-time runtime that combines (a) policy failure prediction, (b) geometric/kinematic limit enforcement, and (c) human escalation, at the policy-serving boundary, with auditable signed decisions.**

lictor occupies that slice. Its architecture is a two-plane Simplex/LITHE split: a best-effort plane (VLA inference, learned failure scores) and a deterministic fuse plane (pure Rust, zero-alloc hot path, pure verdict function, fail-closed by construction) connected by lock-free rings. Its decision math reuses the field's now-standard trick — conformal-prediction thresholds calibrated offline, making the runtime decision a deterministic comparison — and its audit layer ports the author's bulla/sbx machinery (Ed25519-signed receipts, hash-chained ledgers, typed verdicts) into a domain where ISO 10218-1:2025 §7.5.12 now *requires* per-safety-function documentation of response times and failure metrics.

The first deliverable experiment is the **safety-vs-latency curve**: % of failures averted, end-to-end task success, and intervention rate, as functions of per-tick verification budget, measured on lerobot sim benchmarks (PushT diffusion → ALOHA ACT → LIBERO + SmolVLA) on the available hardware (GTX 1060 3GB / CPU / WSL2 — which cannot claim real-time, and this document says exactly what it *can* claim). No prior paper plots this exact curve; the pieces (RTC's latency-injection results, conformal-α sweeps, cost-constrained-monitor theory) exist separately. Quantifying the budget→safety mapping, deterministically and with replayable signed evidence, *is* the contribution.

---

## 2. The hole — the reliability crisis of deployed VLA policies in 2026

### 2.1 The GEN-1.5 story, told accurately

On **2026-08-19**, Generalist AI (founded 2024 by ex-DeepMind researchers Pete Florence and Andy Zeng and ex-Boston Dynamics engineer Andrew Barry) released GEN-1.5 (<https://generalistai.com/blog/gen-1.5>). Its headline capability is one-shot in-context "physical prompting": a single 3–12-second sensorimotor demonstration inserted into a 30-second context window, no gradient updates, emitting 100 Hz action trajectories. The headline result: **59% (±10% std dev) average success across 10 diverse real-robot tasks** (zippers, jars, pulling money from wallets) — i.e. **~41% failure**. Few-shot adaptation (10 gradient steps on ~5 minutes of data) raises this to 83% (±9%); a one-step-adaptation variant reaches 66.5%.

What must be said precisely, because the "41% failure rate of a deployed black box" one-liner needs two corrections to survive diligence:

1. **The 41% is a self-reported capability number, not a field failure rate.** All GEN-1.5 results come from the company; none have been independently verified — the-decoder states this explicitly (<https://the-decoder.com/gen-1-5-generalist-ai-teaches-robots-new-tasks-from-a-single-demo/>). The tasks are described by Generalist itself as "simple and short-horizon". What the ±10% std dev is computed over (tasks vs trials) is not stated, so per-task failure could be far worse than 41%. **UNVERIFIED**: press-reported ~99% success claims for the earlier GEN-0/GEN-1 on trained tasks.
2. **GEN-1.5 is not mass-deployed.** No public weights, no API, no self-serve product; it runs on Generalist's own fleet, available via direct partnership only (<https://www.marktechpost.com/2026/08/24/generalist-ai-releases-gen-1-5-a-robot-foundation-model-that-learns-new-tasks-from-one-3-12-second-demo/>). "Deployed by people who cannot see inside it" should be reframed as "released to partners who receive behavior, not weights — and headed toward deployment fast": Generalist raised $400M at $2B in June 2026 (NVIDIA, Bezos Expeditions, Fei-Fei Li; <https://siliconangle.com/2026/06/04/generalist-ai-raises-400m-2b-valuation-build-general-intelligence-real-world/>) and ~$200M more at $3B roughly six weeks later (<https://techfundingnews.com/ex-deepmind-founders-robotics-startup-generalist-hits-3b-valuation-with-200m-funding/>).

What **is** fully accurate and citable — and is actually the stronger half of the claim:

- **Opacity is admitted by the makers.** The Generalist blog itself: "Why this capability emerges from pretraining is difficult to pinpoint," and in-context skills "are currently more brittle than finetuned models." Closed weights + partnership-only access + zero independent verification = a policy that must be treated as an untrusted black box even by its own customers.
- **The industry's frontier feature ships policies in their least reliable mode.** The 59%→83% one-shot→few-shot delta shows adaptation-free in-context deployment — the fastest, most brittle mode — is exactly what is being marketed, because it is fastest. That is precisely when a deterministic outer envelope matters most.

Recommended honest framing for the project README: *"GEN-1.5 (19.08.2026): the frontier one-shot capability its makers advertise fails ~4 times in 10 (59%±10%, self-reported, unverified) on simple short-horizon tasks — and it is closed-weights, partnership-only, from makers who say they cannot explain why it works."*

### 2.2 The better-armored, third-party numbers

Do not anchor the narrative solely to one funding-cycle press number. The independent picture is worse and more durable:

| Model | Result | Source | Independence |
|---|---|---|---|
| π0.5 (strongest tested) | 43.7% avg success, RoboChallenge real-robot benchmark | <https://arxiv.org/pdf/2510.17950> | Third-party |
| π0.5 on AgiBot G2, unseen settings | ~24% success | RhinoVLA report <https://arxiv.org/pdf/2606.07383> | Third-party |
| pi0-FAST-DROID in the wild | 42.3% avg progress over 300+ trials; 24% pick-and-place; 8% coffee machine; failure modes: freezing, spatial misjudgment/collisions, prompt sensitivity | <https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/> | Third-party |
| GR00T N1.5 | 38.3% on 12 novel DreamGen tasks; 15% zero-shot novel-object pick-and-place | <https://research.nvidia.com/labs/gear/gr00t-n1_5/> | Self-reported |
| SmolVLA | ~78.3% avg on three simple real SO-100 tasks | <https://arxiv.org/pdf/2506.01844> | Self-reported |
| Gemini Robotics-ER 1.5 | >65% real-world few-shot success; success-*detection* accuracy only 0.79–0.80 multiview | <https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/> | Self-reported |

Two structural facts amplify the hole:

- **Self-reported evals are known to misrank policies.** RoboArena (7 institutions, 600+ double-blind paired real-robot episodes) exists precisely because centralized/self-run evaluations misrank generalist policies (<https://arxiv.org/abs/2506.18123>). Industry commentary (Adamo, "The 90 Percent Problem in Humanoid Deployment," 2026): models hitting ~95% on benchmarks hit "60 to 80 percent on real tasks in real environments," with teleoperation interventions papering over the gap (<https://adamohq.com/blog/the-90-percent-problem-in-humanoid-deployment>). This directly motivates lictor's bulla-lineage answer: tamper-evident, third-party-checkable safety-eval receipts — an honest-evaluation story no VLA vendor offers.
- **Regulation is arriving.** "How VLAs Fail Differently" ties runtime monitoring to EU AI Act Article 9 continuous-monitoring obligations for high-risk AI landing 2027 (<https://arxiv.org/html/2605.28726>). 2026 humanoid procurement guidance already tells buyers to "require an intervention log" (<https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/>).

Positioning note for the failure-prediction pillar: even frontier labs' *learned* failure detectors sit at ~0.79–0.80 accuracy (Gemini ER 1.5 multiview success detection). A deterministic geometric/kinematic checker does not compete with that — it complements it. That is the defensible statement.

---

## 3. Landscape

### 3.1 Failure-detection research (through Aug 2026)

The field has converged on a common recipe: **distill each control step into one or a few scalar risk scores, then compare against thresholds calibrated OFFLINE via (split or functional/time-varying) conformal prediction (CP)** — making the runtime decision a deterministic comparison, which is exactly what a Rust fuse needs. Reported accuracy has a consistent ceiling of ~72–88% (bACC/ROC-AUC) on unseen tasks and real hardware; false-positive rate is directly controlled by the CP significance level α, and sweeping α yields the accuracy-vs-detection-time curve that SAFE, Hide-and-Seek, VLA-FAIL (AUCPDT) and ActProbe (F1-timeliness Pareto hypervolume) already publish.

| Method (venue) | Signal | Black/white-box | Deterministic at runtime? | Cost | Code |
|---|---|---|---|---|---|
| **FAIL-Detect** (RSS 2025, <https://arxiv.org/abs/2503.08558>) | logpZO flow-matching likelihood / RND on obs embeddings; also STAC, SPARC, DER | White-box (embeddings) | Yes once CP-calibrated (functional bands, α=0.05) | logpZO 33–40 ms/step (A6000); their STAC reimpl. 1.45 s/step (256 sampled chunks) | <https://github.com/CXU-TRI/FAIL-Detect> (license unconfirmed) |
| **Sentinel/STAC** (CoRL 2024, <https://arxiv.org/abs/2410.04640>) | Statistical distance (MMD/KL) between overlapping segments of consecutive *sampled* chunk batches + VLM tier for task-progression failures | Black-box weights, but needs batch sampling | STAC: no (sampling, seed-dependent); threshold itself yes | Sampling-dominated (~256 chunks/step); VLM ~2.3 s/query | MIT, <https://github.com/agiachris/sentinel> (+319GB rollouts, 82GB checkpoints) |
| **SAFE** (NeurIPS 2025, <https://arxiv.org/abs/2506.09937>) | Tiny MLP/LSTM (2.3M params) on VLA final-layer hidden states | White-box | Yes (functional CP) | +0.73 ms (<1% of VLA inference) | <https://github.com/vla-safe/SAFE> — tested on OpenVLA/pi0/pi0-FAST |
| **FIPER** (NeurIPS 2025, <https://arxiv.org/abs/2510.09459>) | RND on obs embeddings AND action-chunk entropy (AND-gated to suppress benign-OOD false alarms) | Gray (embeddings + sampled chunks) | Threshold yes; ACE needs batch sampling (seed-dependent) | Sampling-dependent; calibrates on 10–50 successful rollouts | MIT, <https://github.com/utiasDSL/fiper> |
| **ActProbe** (Jun 2026, <https://arxiv.org/html/2606.08508>) | Temporal Consistency Error (overlap MSE of consecutive chunks) + chunk magnitude → ~24K-param LSTM-MLP, task-language-conditioned | **Pure black-box, action-space only** | Yes (split-CP quantile) | ~3 ms/step, CPU-viable; probe trains in ~1 min | Apache-2.0, <https://github.com/air-embodied-brain/actprobe>; tested on OpenVLA/pi0/pi0.5/GR00T; real-robot generalization weak (6/12, 5/12 unseen picks) |
| **VLA-FAIL** (Jun 2026, <https://arxiv.org/abs/2606.21386>) | LLMD (last-layer Mahalanobis, white-box) + ACC (single-sample velocity-normalized chunk-overlap disagreement, training-free black-box) | Both variants | Yes (time-constant CP band) | "Marginal" overhead; introduces AUCPDT metric | **UNVERIFIED** (anonymous project page only) |
| **Hide-and-Seek** (May 2026, <https://arxiv.org/html/2605.30834>) | Contrastive probe over VLA action embeddings, trajectory-level labels only | White-box | Yes (functional CP) | ~1 ms/step (A6000); ~2000× faster than VLM monitor (2.343 s/step) | Unconfirmed |
| **SAFECAST** (Aug 2026, <https://arxiv.org/abs/2608.04246>) | Pre-final-layer hidden-state probe + contrast-set training for deployment-shift robustness | White-box | Yes (functional CP) | Small probe | No release mentioned; exact gains over SAFE **UNVERIFIED** (extracted figures look misparsed) |
| **PATCH** (Jun 2026, <https://arxiv.org/html/2606.16690>) | Latent patch "innovation" (visual residuals unexplained by self-motion) in action-conditioned corridor | Gray (RGB + state + chunks) | Yes (fixed p99.5 percentile thresholds) | 18.1 ms/frame (RTX 5090) | Unconfirmed |
| **RC-NF** (Mar 2026, <https://arxiv.org/html/2603.11106>) | Normalizing-flow likelihood over proprio + SAM2 object points | Gray | Yes (static quantile) | ~100 ms/step (RTX 3090) | Unconfirmed |
| **Surgical FoMo detector** (Jul 2026, <https://arxiv.org/html/2607.27511>) | Flow-matching world model over DINOv2/VAE latents, inverse-transport nonconformity | Gray | Yes (CP from just 19 rollouts) | 13.98 Hz (RTX 5090); 96.6% FDR @ 1.3% FAR | Promised "upon acceptance" |
| **Pre-VLA** (May 2026, <https://arxiv.org/html/2605.22446>) | Pre-execution chunk verification via frozen world-model backbone + safety/advantage heads | White-box, heavyweight | Threshold yes | ~184 ms/candidate chunk, up to ~1.1 s with resampling | None |
| **TAIL-Safe** (May 2026, <https://arxiv.org/html/2605.01195>) | Visibility/graspability scores → Lipschitz Q-function | Gray; deterministic (ODE) policies only | Yes | 2.8 ms/step (RTX 4090) | None |
| **VLM judges** (AHA <https://arxiv.org/abs/2410.00371>, Code-as-Monitor <https://arxiv.org/pdf/2412.04455>) | Semantic task-progression judgment | Black-box | **No** (non-deterministic, cloud) | ~2.3 s/query | AHA partial |
| Ensembles (Diff-DAgger critique, <https://arxiv.org/html/2410.14868v1>) | Disagreement | — | — | N× inference; conflates multimodality with uncertainty | Effectively deprecated for large policies |

Key structural conclusions:

- **Cheap-and-deterministic exists**: pure action-space features (TCE/ACC/ACM/SPARC — arithmetic on chunks the policy already emits, microseconds, no model, no sampling) and tiny probes (0.73–3 ms). **Avoid sampling-dependent scores** (original STAC, FIPER's ACE) in the deterministic tier — the cost is policy sampling, and results are seed-dependent.
- **Conformal prediction is the calibration contract.** Ship per-task calibration artifacts (time-indexed mu_t + band_t arrays, or a single split quantile, plus α) as signed, hashed files; the runtime decision becomes bit-reproducible from logged scores. Calibration-data economy: FIPER needs 10–50 successful rollouts, the surgical detector 19, SAFE ~100+ *plus* failure rollouts — default lictor tooling to "a few dozen successful rollouts, no failure data" with supervised probes as an upgrade path.
- **Three-tier fuse architecture follows directly**: Tier 0 (pure Rust, microseconds, training-free, deterministic): chunk-overlap disagreement, chunk magnitude, smoothness/jerk, plus geometric/kinematic limit checks. Tier 1 (Rust, ~1–3 ms, deterministic): ActProbe-style ~24K-param probe over Tier-0 features, portable to burn/candle/tract, CPU-runnable. Tier 2 (optional scalar channel): white-box scores (logpZO/RND, SAFE/LLMD probes) computed in the Python policy process, streamed to the fuse as floats — the Rust side owns only threshold comparison, windowing/hysteresis, and escalation. VLM judging goes to the human-escalation tier, never the fuse. Replicate FIPER's AND-gating between orthogonal scores to suppress benign-OOD false alarms.
- **Two open lanes confirmed**: no published failure-detection results on SmolVLA specifically (**UNVERIFIED** as absence-of-evidence), and nobody ships a policy-agnostic deterministic *runtime* with signed/auditable decisions — the papers ship detectors, not runtimes. lictor's novelty claim is the certified/reproducible decision plane and the honest cross-detector benchmark, **not** a new detection score. Cross-validating a Rust port against the Python reference implementations (same rollout in → same score out) is itself a publishable engineering claim.

### 3.2 Geometric safety and functional-safety vocabulary

The standards ground shifted in 2025 and lictor should speak the new language natively:

- **ISO 10218-1/-2:2025** (published March 2025, in force 2025-04-01) absorbed ISO/TS 15066, which no longer exists standalone (<https://www.automate.org/robotics/blogs/updated-iso-10218-faq>). Terminology: "collaborative robot" → "collaborative application"; "safety-rated monitored stop" → "monitored standstill." New named safety functions: monitored standstill (5.5.5), stopping-time limiting (5.5.6), stopping-distance limiting (5.5.7). **Clause 7.5.12 now requires manufacturers to document every safety function's PL, SIL, PFH, DC, HFT and response times** (Annex D template) — the bulla-style signed safety-case log maps onto this almost verbatim. Robot Classes: Class I (≤10 kg, ≤50 N, ≤250 mm/s) needs PL b/SIL 1; Class II needs PL d/SIL 2. Caveat: class thresholds and clause details rest on one secondary source (<https://arxiv.org/pdf/2602.17822>) — **confirm against the purchased standard before quoting normatively**. The US adopted the text as ANSI/A3 R15.06-2025 (<https://www.therobotreport.com/now-available-full-403-page-ansi-a3-r15-06-2025-robot-safety-standard/>). Cybersecurity is now part of robot safety.
- **Stop-category vocabulary** (IEC 60204-1 / IEC 61800-5-2): Cat 0/STO = immediate power removal; Cat 1/SS1 = controlled stop then power removal; Cat 2/SS2 = controlled stop with power retained (monitored standstill). lictor's escalation ladder maps: chunk rejected pre-execution → SS2-like hold (resumable); in-flight violation prediction → SS1-like Ruckig-computed brake; fault in the fuse itself → STO-like disable. "Controlled retreat" is **not** a standard certified response — treat as optional research feature; ISO 10218-1:2025 5.1.8's position-maintaining/stored-energy requirement is the reason hold-under-power is the default. Escalation-to-human maps onto protective-stop (auto-resumable after reset) vs emergency-stop semantics; re-arm after N consecutive safe-stops requires a human acknowledgment token — where bulla's Ed25519 machinery ports directly.
- **Certified precedent for predictive enforcement**: Universal Robots e-Series — all safety functions PL d Cat 3, TÜV NORD-certified, each with published PFHd (mostly 1.2E-07/h) — enforces stopping-time/stopping-distance limits **predictively**: current speed is continuously capped so the configured stop can never be exceeded (<https://www.universal-robots.com/media/1804320/e-series-functional-safety.pdf>). This is the certified ancestor of lictor's core invariant: *at every cycle, the committed action prefix + a braking trajectory must jointly satisfy all limits.* UR also gives the mode-switch template (reduced-mode limit-set swap within 500 ms / 2 cm of a trigger plane) and the versioned, checksummed safety-configuration pattern lictor's SafetyEnvelope should copy. **UNVERIFIED**: UR10e Cat 0 stop ≈ 400 ms (snippet-level).
- **The safety-vs-latency bridge is a standards formula**: the ISO/TS 15066 SSM protective-distance equation S(t0) ≥ ∫vH(TR+TS) + ∫vR(TR) + ∫vS(TS) + C + ZS + ZR makes reaction time TR a first-class safety parameter — every millisecond of fuse latency has a quantifiable cost in millimeters of required separation (defaults: human 1600/2000 mm/s per ISO 13855; NIST recommends ≥100 Hz evaluation, <https://pmc.ncbi.nlm.nih.gov/articles/PMC5117641/>). Commercial safety controllers publish computed *worst-case* response times per configuration, never "typical" (SICK Flexi Soft worked example: 19.0 ms).
- **Enforcement math**: prefer closed-form/fixed-iteration over general QPs. CBF-QP filters are fast enough (2.6–22.5 kHz on a 7-DOF Franka, <https://arxiv.org/pdf/2503.06736>) but iterative solvers have data-dependent iteration counts; execution-time-certified QP is research, not product (<https://arxiv.org/html/2510.21773>). If a QP is used: cap iterations and treat non-convergence as a trip condition (fail-safe brake). **Ruckig** (jerk-limited time-optimal OTG, mean 19.8 µs/cycle for 7 DoF, robust over 1e9 trajectories, <https://arxiv.org/abs/2105.04830>) is the canonical safe-stop-trajectory generator. ARMTD's discipline — every plan carries a verified fail-safe braking maneuver (<https://arxiv.org/abs/2002.01591>) — is the design invariant. Keep learned components entirely out of the trusted path.
- **Chunk-level verification is a live 2026 subfield validating the thesis**: Pre-VLA scores chunks pre-execution; barrier-enhanced flow matching enforces a Log-Sum-Exp aggregate CBF over the whole chunk *because per-action CBF filtering causes jitter/overreaction* (<https://arxiv.org/abs/2607.29569>); CheckVLA verifies in-flight with an action-conditioned world model (<https://arxiv.org/html/2607.26789>); LIBERO-Safety (ECCV 2026, 19,664 demos) benchmarks physical+semantic VLA safety and establishes the "generalization-safety tension" framing (<https://arxiv.org/abs/2606.23686>). RTC's "actions frozen because guaranteed to execute" defines the irrevocable commitment window lictor must pre-verify (<https://arxiv.org/abs/2506.07339>). At 50 Hz with 50-step chunks (1 s of committed motion), chunk lookahead is the dominant safety mechanism; a ~1–2 ms per-chunk verification budget (kinematic forward simulation of 50 steps + braking feasibility at each) is achievable and should be profiled and published as the core latency claim.
- **Honest positioning language** (non-negotiable): a userspace layer on PREEMPT_RT/WSL2 may claim "deterministic algorithmic behavior — same inputs → same verdict, bounded-by-construction hot path, measured worst-case latency histograms," and "engineered to IEC 61508/ISO 13849 principles." It may **never** claim "safety-rated," "hard real-time," or imply PL/SIL ratings — certification requires redundant hardware channels (Cat 3), certified RTOS scope, IEC 61508-3 lifecycle evidence, MISRA-style coding standards, and T2/T3 tool qualification (<https://ldra.com/iec-61508/>). PFHd-style vocabulary is used as *design targets*. AI functional safety guidance: ISO/IEC TR 5469:2024 and its successor TS (<https://www.iso.org/standard/81283.html>).

### 3.3 Competitive gap — what exists, what does not

The gap claim is ~90% validated, with two corrections that make it *stronger* when stated precisely:

- **3Laws Supervisor existed and is gone.** A CBF-based software layer between autonomy stack and robot, rewriting unsafe motion commands in real time, with an official ROS 2 Nav2 integration — acquired **2026-05-05 by Amazon** (PitchBook, <https://pitchbook.com/profiles/company/501876-73>), taking the only shipping policy-agnostic runtime safety product off-market and internal. Note it corrected velocity commands for AMRs/manipulators; it did **not** do learned-policy failure prediction or VLA-specific monitoring.
- **NVIDIA Halos for Robotics** (June 2026, "the industry's first full-stack safety system for Physical AI") is infrastructure: IGX Thor safety island at IEC 61508 SIL 3, proprietary Halos Core safety OS in early access, an open-source facility-mounted "Outside-In Safety Blueprint," and an ANAB-accredited certification lab — NVIDIA-silicon-locked, and it explicitly does not wrap VLA policies, predict policy failures, or enforce kinematic limits on policy outputs (<https://nvidianews.nvidia.com/news/nvidia-announces-halos-for-robotics-the-industrys-first-full-stack-safety-system-for-physical-ai>). NVIDIA's own GR00T model card pushes guardrail responsibility onto deployers (<https://huggingface.co/nvidia/GR00T-N1.7-3B>).
- **Every published VLA failure detector is research Python** (§3.1). Runtime-assurance tooling (AFRL act3-ace/run-time-assurance, SOTER-on-ROS, RTAEval) is research-grade and aerospace-oriented. A Feb 2026 position paper with Amazon/NVIDIA co-authors *explicitly argues modular external guardrails for foundation-model robots are missing*, citing latency overheads and correlated FM failures — with no released system (<https://arxiv.org/html/2602.04056>). A deterministic non-learned enforcement layer answers both of that paper's criticisms.
- **Big labs are closed and internal**: DeepMind ships model-level safety + the ASIMOV-Agentic benchmark — which scores *proactively requesting human intervention*, industry validation of the escalation pillar; design lictor's escalation API to be measurable in that framing (<https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/>). Physical Intelligence has published nothing on runtime safety beyond HAL force/speed limits; Figure runs an in-house safety center; 1X leans on compliant hardware + Redwood AI; Agility is adopting NVIDIA IGX + Halos Core.
- **Adjacent players are complements, not competitors**: Foxglove/InOrbit/Formant (data infra/incident review — integration targets for intervention events), evals players (Andon Labs Butter-Bench, ReSim, RoboDojo — offline), classical safety vendors (Fort Robotics SIL 3 wireless e-stop; Veo FreeMove acquired by Symbotic Jul 2024 and internalized). AI-observability vendors have not entered robotics (WhyLabs acquired by Apple Jan 2025, discontinued).
- **The only startup claiming the space** is Voyage Robotics (London, pre-seed, founded 2024, "neural safety system for Physical AI" — details **UNVERIFIED**, one VC listicle, <https://viewpoints.fov.ventures/p/10-early-stage-startups-to-watch-in-2026>). Its *learned* approach concedes the deterministic niche.
- **The baseline to beat is embarrassingly weak and citable**: LeRobot's official safety story is a Python `EEBoundsAndSafety` clipping step plus a joint-limit discovery script (<https://huggingface.co/docs/lerobot/hilserl>).

**The precise unoccupied slice**: independent + hardware-agnostic + CPU-real-time + policy-level (wraps VLA action chunks at the serving boundary) + deterministic (non-learned trusted path) + failure prediction + geometric enforcement + escalation + signed auditable receipts. Threat model: (a) NVIDIA extending Halos down-stack; (b) Amazon productizing 3Laws via AWS; (c) Voyage maturing. Being open-source and hardware-agnostic is the wedge against all three. Beware architecture-specific failure signatures (<https://arxiv.org/pdf/2605.28726>): truly policy-agnostic *detection* is empirically hard — report curves per-policy, which fits the honesty framing.

---

## 4. Integration surfaces — where the fuse intercepts action chunks

Two ecosystems, two wire protocols, one internal design rule: **lictor defines its own chunk IR and ships thin adapters** (the protocols are not shared between openpi, LeRobot, and GR00T stacks).

### 4.1 openpi (pi0 / pi0-FAST / pi0.5) — Apache-2.0, active through Aug 2026

- **Serving boundary**: `scripts/serve_policy.py` starts a `WebsocketPolicyServer` (default port 8000) wrapping any `BasePolicy`; each msgpack_numpy-serialized observation dict is answered with a full action chunk (<https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py>). The client (`openpi-client`, Python 3.8-compatible) calls `WebsocketClientPolicy(host, port).infer(obs) -> {"actions": (action_horizon, action_dim)}`. Observation keys: `observation/image`, `observation/wrist_image` (uint8 224×224 resize_with_pad), `observation/state`, `prompt`.
- **Two intercept granularities**: (a) a **transparent websocket proxy** speaking the msgpack wire protocol — the robot points at the fuse's port instead of 8000; zero changes to client or server; policy-agnostic by construction; chunk-level decisions (one per 0.5–1 s) — failure prediction and escalation live here; (b) the **client-side execution deque** (`examples/libero/main.py`: extend with `action_chunk[:replan_steps]`, `popleft()` per `env.step`) for per-action gating at 10–50 Hz — deterministic geometric/kinematic gating lives here. Plan both interfaces.
- **Chunking/latency reality**: pi0 base = 50-action chunks = 1 s at 50 Hz; LIBERO pi0.5 uses action_horizon=10, replan_steps=5; DROID = 15 Hz, 8-dim joint-velocity actions; ALOHA 14-dim joint position at 50 Hz. Official pi0 inference on RTX 4090: 43.8/53.7/67.6 ms for 1/2/3 views (Dexmal optimized to 20.0–36.8 ms; <https://arxiv.org/html/2510.26742v1>); PI's RTC deployment measured 108–139 ms end-to-end remote and survives +200 ms injected latency via chunk-stitching (freeze first ~3 actions) (<https://www.pi.website/research/real_time_chunking>); DROID docs call 0.5–1 s/chunk over a network "normal". A 1–2 ms fuse check is <2% overhead — the expensive knobs on the curve are detector tiers, not geometric checks.
- **Escalation plumbing exists**: the server reports errors as a text frame + INTERNAL_ERROR close; the fuse escalates by substituting a safe-stop chunk (hold/zero-velocity, dynamically smooth per RTC's freeze insight) plus a typed verdict channel. Never just drop the connection — `WebsocketClientPolicy` retries forever silently.
- **Constraints**: inference needs >8 GB VRAM (README) — pi0-family cannot run on the local GPU; the documented cloud-GPU-server + local-client topology matches our hardware split exactly. pi0.6/π*0.6 are closed (issue #791 unanswered) but the model card confirms the same chunked flow-matching interface across generations — the interception surface is stable. LIBERO is the only fully dockerized in-repo sim benchmark, with published pi0.5 results (96.85% avg — so failure injection is required to populate the failure class there). **UNVERIFIED**: padded action_dim=32; native RTC support in serve_policy.py at HEAD; scope of the Jun 2026 batching change.
- **Embodiment binding**: action semantics differ per checkpoint. The fuse needs an **embodiment manifest** (joint/vel/accel limits, workspace geometry, control frequency) bound — and signed — at the websocket metadata handshake. Internal-feature (Tier 2) monitoring requires an optional sidecar contract: a patched `BasePolicy` attaches feature scalars to the msgpack response; the verdict schema must accept both black-box and sidecar inputs.

### 4.2 LeRobot / SmolVLA (v0.6.1, Aug 2026)

- **Blessed insertion point**: the v0.6 processors pipeline — `preprocess_observation → env_preprocessor → policy preprocessor → policy.select_action → policy postprocessor → env_postprocessor → env.step`. The `env_postprocessor` is currently **identity for all envs and the docs literally show a "safety limits" torch.clamp example as its intended use** (<https://huggingface.co/docs/lerobot/en/env_processor>). Implement the fuse as a registered `ProcessorStep`; ship a ~100-line custom eval script reusing `make_env/make_policy/make_pre_post_processors` (no CLI flag injects custom steps into lerobot-eval — an upstream `--env.postprocessor_steps` PR is a high-visibility contribution vehicle).
- **Chunk-level enforcement primitive is ready-made**: `PreTrainedPolicy` exposes `predict_action_chunk()` (whole chunk) and **`drop_queued_actions()`** — validate the upcoming chunk, and on violation drop the queue, substitute a safe action, escalate (<https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py>). This maps 1:1 onto the three pillars.
- **Async inference is the production-shaped surface**: gRPC `policy_server`/`robot_client`, `actions_per_chunk=50`, `chunk_size_threshold=0.7`, overlapping chunks merged by an `aggregate_fn` — the fuse sits as a proxy between server and client or as a custom aggregate_fn (<https://huggingface.co/docs/lerobot/main/async>). The 2-process async split is also the VRAM workaround (below).
- **The security hook**: **CVE-2026-25874** — unauthenticated pickle-deserialization RCE in exactly that gRPC channel (`add_insecure_port` + `pickle.loads`); issue #3047 still open, fix PR #3048 unmerged at last check (<https://chocapikk.com/posts/2026/lerobot-pickle-rce/>). The policy→actuator channel is today both *unsafe* (no limits) and *insecure* (RCE). A Rust proxy that replaces pickle with safetensors/JSON and adds authenticated, Ed25519-signed action receipts fixes a documented hole while adding safety. (Verify patch status at implementation time; if merged, reframe as "we harden further.")
- **Adjacent LeRobot assets**: v0.6.0's reward-models API (Robometer, TOPReward) is a candidate learned-failure-predictor baseline; `lerobot-rollout`'s DAgger-style human-in-the-loop corrections are prior art adjacent to the escalation pillar; RTC is integrated in LeRobot docs and applies to pi0/pi0.5/SmolVLA (<https://huggingface.co/docs/lerobot/rtc>).
- **Environment constraints**: Python ≥3.12 only (host 3.14 unsupported — pin 3.12 in a uv venv under WSL2); torch pinned 2.7.x+cu126 for Pascal (cu128/129 dropped sm_61 — do not let uv resolve a newer torch); mujoco version must be pinned (issue #4390: ≥3.4.0 silently breaks a libero_spatial initial state — a determinism hazard directly relevant to honest curves); API churn is real (0.6.1 renamed `lerobot.types`) — pin the exact version, integrate against public factories.

---

## 5. What we reuse from bulla/sbx — concrete inventory

Both repos (RARS-oss/bulla, RARS-oss/sbx, MIT, Rust 2021 workspaces, pushed 2026-08-23) are unpublished to crates.io, so the reuse path is **vendor-with-provenance-comment** — the precedent bulla itself set vendoring hermit-core from sbx.

**Code assets:**

1. **`bulla-core/src/lib.rs`** (one 32 KB platform-independent file; deps: serde, serde_json, sha2 0.10, hex 0.4, ed25519-dalek 2.x, getrandom) → vendored as `crates/lictor-receipt`. Contains the complete signed-receipt machinery: `ReceiptBody`/`SignedReceipt` (canonical bytes = serde_json field order; sha256 body_digest; Ed25519 verify_strict), `seal_chain` hash-chained event log (ZERO_HASH genesis), content-addressed input `Manifest`, append-only anti-cherry-pick run ledger (`verify_ledger` with break_at detection), and `evaluate_seal` — the verifier recomputes the verdict from attested facts. Mechanical renaming: `EvalPolicy` → SafetyEnvelope (limits, watchdog period); `AppliedSummary` → EnforcedSummary (which limits actually held this episode — hermit-core's demanded-vs-applied doctrine); `OutcomeSummary` → InterventionOutcome; `evaluate_seal` → evaluate_fuse; run ledger → episode ledger (dropping the episode where the fuse tripped breaks the chain at a reported seq). **Keep the "intact ≠ seal_ok" doctrine**: a valid receipt can honestly attest that limits were violated. **Keep crypto off the real-time path**: hash-chain per-tick events (the `seal_chain`/`egress_summary` pattern: sha256 over {seq,kind,detail,prev}), sign once per episode/intervention.
2. **sbx `crates/feedback` — the shape, not the crate**: serde-stable `SafetyVerdict` with severity-ordered Status (Nominal < Clamped < PredictedFailure < EStop < Escalated), `diff(prev,cur)` deltas between episodes, deterministic capped render with a `hint_for`-style glossary explaining *why* the fuse tripped (the human-escalation channel), and `history.rs` (std+serde only, copyable nearly verbatim) → `.lictor/history.jsonl` with `tail_streak` signals ("policy hit the same wrist-limit 4 episodes running"). The tree-sitter/compiler-output parsers are irrelevant. Document the contract in a `docs/verdict-schema.md` twin.
3. **bulla-zk** (Bulletproofs range proof over Ristretto25519) — a plausible *later* differentiator (prove a commanded action was within certified limits without revealing the policy's action); keep in a separate optional crate with the -ng dalek fork isolated; not milestone-1.
4. **Do not reuse**: hermit-core (Linux sandbox — wrong layer), feedback's parse.rs/scope.rs, bulla's git-seal/oracle/egress-broker.

**Discipline assets (the author's brand — imitate verbatim):**

- Workspace skeleton: `[workspace.package]`, path+version internal deps, release profile `opt-level=3, lto="fat", codegen-units=1` (**revisit `panic="abort"`** if lictor-core is ever embedded via pyo3, which needs unwinding), members `crates/lictor-core` (pure: envelope + verdict + receipt) / `crates/lictor-fuse` (deterministic runtime: limit checks, failure-predictor interface, escalation state machine) / `crates/lictor-cli` + `adapters/` (thin single-file Python shims in the `bulla_mcp.py` style, incl. its Windows→WSL path translation — needed on this exact setup) + `bench/` + `docs/` + `research/`.
- CI: `dtolnay/rust-toolchain@stable`, `RUSTFLAGS=-D warnings`, fmt+clippy gates, and **every bench experiment re-run in CI with grep-asserted expected result lines**.
- bench/: bulla-style controlled A/B per failure vector (`bench/overspeed/`, `bench/workspace-breach/`, `bench/failure-prediction/` — same rollout, fuse-off vs fuse-on, asserting results *from the signed receipts* via python3 heredocs), plus sbx-style episodes × conditions × seeds JSONL harness with resumability, `--summary`, and committed SVG figures for the curve.
- CLI: `lictor bench` (per-step fuse overhead in µs is the headline number, the way 4.6 ms sandbox startup was sbx's), `lictor selftest`, and sbx's `--repeat N` semantic-identity check as the determinism demonstration (same observation stream → byte-identical intervention decisions across N replays; sbx's 40/40-byte-identical result is the rhetorical template: "determinism is the precondition, not decoration").
- Docs: README with banner SVG, badges, "See it in 30 seconds," **"honest boundary" section** (what the fuse cannot catch: semantic task failure inside the geometric envelope, sensor lies, contaminated calibration data), roadmap ordered by value, authorship section crediting Claude; `docs/REPORT.md` paper-shaped with Threats to validity + Reproducibility appendix + `research/` directory holding the memos behind every cited number; the determinism experiment stated precisely (decisions reproduce; timestamped signed bodies honestly do not); the ledger's honest gap stated (backward chain cannot detect tail truncation — needs an external witness/Rekor anchor).

**Two flagged upgrades before receipts proliferate**: (1) bulla's "canonical bytes" are serde-field-order JSON, not RFC 8785 JCS — decide now whether lictor breaks compatibility for a real canonical encoding (JCS/canonical CBOR) while nothing external verifies receipts yet; a third-party verifier in Python must otherwise replicate serde's field order exactly. (2) bulla-core structs are String/Vec-heavy and allocate per event — the fuse hot path needs a pre-allocated/no-alloc verdict path with the receipt built off-loop.

---

## 6. Feasibility on our hardware (GTX 1060 3GB / CPU / WSL2)

### 6.1 What runs

| Target | Feasibility | Notes |
|---|---|---|
| **gym-pusht + lerobot/diffusion_pusht** | Yes — CPU-viable | 65.4% success over 500 episodes → ~35% *natural* failures, ideal detector fodder, no injection needed (<https://huggingface.co/lerobot/diffusion_pusht>); needs the shipped normalization-migration script (pre-PR-#1452 checkpoint); validate the 65.4% reproduces (gym-pusht/gymnasium pinning, issue #470) |
| **gym-aloha + ACT (transfer_cube)** | Yes — small fast transformer | 83%/500 episodes; adds real 14-D joint-limit/velocity enforcement |
| **LIBERO + lerobot/smolvla_libero** | Yes (headline, with care) | SmolVLA 450M, ~2 GB inference (LeRobot async docs), documented to run on CPU; Pascal has no bf16 → fp32 or CPU; the checkpoint's per-suite success rate is unpublished — **must be measured** (a ~400-episode baseline run is needed anyway) |
| pi0 / pi0.5 locally | No | >8 GB VRAM floor (openpi README; ~14 GB per LeRobot async docs; one report ~37 GB to load a finetuned policy server, openpi issue #599). Path: rented 4090-class GPU running `serve_policy.py --env LIBERO`, local sim + fuse over websocket — exactly openpi's documented deployment topology, so the experiment doubles as a realistic latency study with WAN latency as a controlled variable |
| SimplerEnv | No | CUDA ≥11.8, Vulkan, "ideally RTX"; not officially supported by openpi anyway (issue #799) |
| isaaclab_arena | No | Out of local reach; roadmap only |

VRAM contention is documented even on 8 GB cards (lerobot issue #3098: PyTorch model + MuJoCo EGL context): adopt the 2-process split — policy in one process (GPU fp32 or CPU), env+rendering in another (`MUJOCO_GL=osmesa` on CPU if EGL fails under WSL2; WSL2 commonly forces GLX/D3D12 paths). Conveniently, the async server/client split already *is* that architecture, and the fuse naturally lives between the two processes. Day-1 microbenchmarks required (**all wall-clock estimates are UNVERIFIED until then**): SmolVLA fp32 chunk latency on the 1060 and on CPU (expect ~0.5–5 s per 50-action chunk; the only published points are ~101 ms on an RTX 5080 laptop and a low-confidence ~10 s on an RTX 3050 Ti); EGL-vs-osmesa under WSL2.

Offline fallback that is also how the field benchmarks: evaluate detectors over recorded rollouts (Sentinel's ~319 GB rollout datasets, ActProbe's precomputed features — probe trains in ~1 min; FAIL-Detect's logged-rollout protocol). Driving the fuse from recorded chunk streams makes the safety-vs-latency sweeps perfectly reproducible and CPU-only.

### 6.2 WSL2 honesty — what we can and cannot claim

WSL2 runs a guest kernel in a Hyper-V utility VM whose vCPUs are scheduled by a non-RT Windows host: even a custom PREEMPT_RT guest kernel (community builds exist: Locietta/xanmod-kernel-WSL2, pep248/WSL_realtime_kernel) only removes guest-side latency — **host preemption can stall the whole VM for milliseconds. Nothing hard-real-time is claimable in WSL2. Ever.**

What **is** honestly claimable, and how milestone 1 reports it:

1. **Algorithmic determinism** — verifiable properties, not measurements: zero allocations, zero syscalls, zero locks, fixed iteration bounds on the hot path (chunk length is a constant); pure verdict function `(state, chunk, scores, config) → verdict` with no clock reads inside the decision; CI-enforced via an allocation-counting global allocator and `--repeat N` byte-identical replay.
2. **Bit-reproducible verdicts across platforms** via the float-determinism recipe: f64 under Rust RFC 3514 IEEE semantics (<https://rust-lang.github.io/rfcs/3514-float-semantics.html>), no fma contraction (explicit `mul_add` only where intended), all transcendentals through the `libm` crate (the Rapier `enhanced-determinism` pattern, <https://rapier.rs/docs/user_guides/rust/determinism/>), fixed evaluation order (no rayon/SIMD reductions, no HashMap iteration in the decision path), any NaN → fail-closed verdict. `fixed`-point is a documented escape hatch, not the default. This yields a marketable "replay the exact verdict anywhere" guarantee matching the signed-receipt aesthetic.
3. **Statistical latency distributions, labeled honestly**: full HdrHistogram p50/p99/p99.9/p99.99/max, sampled open-loop against the intended tick schedule (Gil Tene's coordinated-omission discipline), with a cyclictest environment baseline published alongside, labeled "measured under WSL2 virtualization (Hyper-V utility VM, non-RT host), not an RT environment." No published cyclictest-in-WSL2 histograms were found — producing one is original content for the writeup. Deployment claim stated conditionally: "on PREEMPT_RT (mainline since kernel 6.12) with mlockall + SCHED_FIFO + isolcpus, the same binary is expected in the 50–200 µs class" — deferred to real hardware. Prior art for the class: LITHE's deterministic 1 kHz spine on a Raspberry Pi 4B, <100 µs WCET, <4 µs release jitter (<https://arxiv.org/abs/2603.07442>); Copper-rs hexapod at 730 µs max end-to-end (<https://www.codethink.co.uk/articles/2026/copper-rs-real-time-robotics.html>).
4. **Fail-closed design makes soft-RT measurement acceptable for milestone 1**: structure the actuator-side consumer so the *default* output for a tick is the safe-stop ramp; only a verdict arriving before the deadline AND carrying valid approval replaces it. Never let "stale last action" or "raw policy action" be the timeout path. A latency outlier therefore degrades liveness (false-stop rate on the curve), never safety. And in sim, wall-clock and control-rate are decoupled: the safety-vs-latency curve is computed against *simulated* time budgets (LIBERO fps=20 → 50 ms/tick), which is virtualization-independent.

Architecture: two-plane LITHE/Simplex split — best-effort plane (VLA inference, learned detectors, logging) and deterministic fuse plane (single pinned thread) connected only by lock-free SPSC rings (rtrb / heapless::spsc; basedrop-style return-ring for retiring chunk buffers off-loop). Gate per control TICK, not per chunk inference. Hot-path rules: pre-allocated buffers, debug_assert + anomaly→safe-stop (never panic), heartbeat watchdog following ros-safety/software_watchdogs. Ecosystem positioning: interoperate with, don't compete against, Copper-rs (deterministic robot runtime, stable July 2026 — study its `freeze` replay and .copper log before designing lictor's replay format) and dora-rs; cu29-task and dora-node wrappers are cheap follow-on distribution channels. Design `lictor-core` no_std-compatible (heapless + libm) from the start to keep the MCU-fuse endgame open — the cheapest path to a genuinely hard-real-time claim later. Skip RTIC/Embassy for milestone 1 (bare-metal targets).

---

## 7. The experiment — the safety-vs-latency curve

### 7.1 The gap the curve fills

Nobody plots **"% failures averted (y) vs added per-chunk/per-tick verification latency (x) with task success as a constraint."** The pieces exist separately: RTC shows injected +100/+200 ms latency degrades synchronous execution sharply while async is flat (<https://www.pi.website/research/real_time_chunking>) — proving the *cost* side of the x-axis: verification latency itself creates failures unless hidden by async execution; arXiv 2605.08168 sweeps injected delays up to d=20 control steps on **LIBERO with SmolVLA** — a directly reusable harness on our exact stack; "Combining Cost-Constrained Runtime Monitors" gives the formal budget framework, theory-only (<https://arxiv.org/abs/2507.15886>); a safety-gating paper reports reliability-vs-intervention-efficiency trade-offs under domain shift (57.5%→77.2% with gating; abstract-level verification only, <https://pmc.ncbi.nlm.nih.gov/articles/PMC13210885/>); streaming-perception literature supplies the Pareto formalism; SAFE/Hide-and-Seek plot accuracy vs detection-*time* (an α-sweep), not compute. The 2026 monitor wave all advertises "lightweight" but rarely quantifies the budget axis. Quantifying the budget→safety mapping is the paper.

### 7.2 Design

- **Environments/policies (ladder)**: Phase 0 (week 1, CPU/1060) — PushT + diffusion policy (~35% natural failures; ~6 budget points × 500 episodes ≈ a weekend, **UNVERIFIED**). Phase 1 — ALOHA sim + ACT (real joint-limit enforcement, 14-D, covers a non-VLA policy class). Phase 2 (headline) — LIBERO libero_10 + `lerobot/smolvla_libero` via a custom eval script on the lerobot processors pipeline (50 episodes/task × 10 tasks per point; ~8 h per 500-episode point at ~1 min/episode fp32, **UNVERIFIED** — ~1 week for a 5-point curve). Later: pi0.5 replication on a rented GPU via the openpi websocket proxy (pi05_libero baseline 96.85%), demonstrating policy-agnosticism across diffusion/ACT/SmolVLA/pi0.5.
- **Failure sources**: natural failures across seeds as the primary source (SAFE precedent — trajectory-level success/failure labels only; step-level labels unnecessary per the literature); LIBERO-plus's 7 perturbation axes (~10,000 variants: object layout, camera, robot init, language, lighting, background, sensor noise; <https://arxiv.org/abs/2510.13626>) as a graded OOD severity dial, choosing 2–3 tiers yielding informative failure rates (~30–60%); seen/unseen task split à la SAFE. For high-success baselines (pi0.5-LIBERO ~97%), a failure-injection harness (observation corruption, prompt perturbation, injected latency, chunk truncation — DRTC/RTC precedent) populates the failure class.
- **Sweep axis**: per-tick verification budget ∈ {0, 1, 5, 10, 25, 50, 100 ms}, each point realized **both** as injected dead-time (RTC/2605.08168 methodology) **and** as a real detector suite that fits the budget: Tier 0 = deterministic Rust geometric/kinematic chunk verification + Ruckig braking-feasibility lookahead (µs); Tier 1 = action-space signals needing no extra policy passes (TCE/ACM/ACC, ~1–3 ms); Tier 2 = learned OOD scores (logpZO/RND, one small forward pass) and STAC-style multi-sample consistency where sample count is the compute knob. Within each point, sweep conformal α for the internal Pareto and report the envelope. Run **sync and async execution modes as separate curves** (RTC proved the difference is first-order).
- **Metrics — three lines per budget point, or the curve is dishonest**: Y1 = % of failing episodes where the fuse fires before the failure point (TPR), with FPR alongside; Y2 = end-to-end task success with the fuse in the loop (captures latency-*induced* new failures and dropped-chunk re-inference cost); Y3 = intervention/escalation rate (Sirius definition, <https://arxiv.org/abs/2211.08416>). Plus the field-standard stack for comparability: ROC-AUC on max per-rollout score, balanced accuracy, detection time via CP bands, AUCPDT (VLA-FAIL) and F1-timeliness hypervolume (ActProbe) as single-number summaries. Expect and report the field's ~72–88% detection ceiling; frame the y-axis alternative in ISO/TS 15066 terms (fuse latency → required-separation inflation in mm) for the standards-grounded version of the plot.
- **Episode counts & statistics**: 500 episodes per curve point (95% Clopper-Pearson CI ≈ ±4pp; TRI guidance: 70 rollouts at 90% success spans 15.4pp; ±2pp needs 1,030 — <https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d>), **paired seeds** across all budget arms (`--env.init_states=true`, fixed `--seed`) so McNemar/paired-bootstrap comparisons are valid; STEP sequential stopping for fuse-on/off comparisons (up to 32% fewer trials, <https://arxiv.org/abs/2503.10966>); 3 seeds per LeRobot guidance; never report a point from the default 10 episodes/task. Every point resumable.
- **Determinism engineering**: pin mujoco exactly (issue #4390), pin dataset revisions and lerobot version, hard resets only for reported numbers, `per_episode_seed=true`, `num_steps_wait=50`, fixed `MUJOCO_GL`, `OMP/MKL_NUM_THREADS=1`, fp32 on Pascal. **Every fuse verdict emitted as an entry in a hash-chained per-tick log, folded into an Ed25519-signed per-episode receipt carrying the full seed/version manifest** — any curve point replayable bit-for-bit. No published robotics eval does this; it is the bulla/sbx signature applied to safety evaluation.

---

## 8. Risks and open questions

**Risks (ranked):**

1. **Latency estimates are all unverified on our hardware.** SmolVLA on the 1060/CPU has no published numbers; if chunks take ≥5 s, the LIBERO phase timeline doubles. Mitigation: day-1 microbenchmark gates the plan; PushT phase is CPU-safe regardless; offline evaluation over recorded rollouts is a fallback that is *also* how the field benchmarks.
2. **Black-box-only detection may be too weak.** ActProbe's real-robot generalization was poor (6/12, 5/12 unseen picks); architecture-specific failure signatures (<https://arxiv.org/pdf/2605.28726>) mean truly policy-agnostic detection is empirically hard; transfer to SmolVLA's flow-matching chunks is untested. Mitigation: the verdict schema accepts an optional white-box sidecar scalar channel from day one; report curves per-policy.
3. **Narrative decay.** GEN-1.5 may be revised or debunked (single self-reported figure in a funding cycle); CVE-2026-25874 may be patched. Mitigation: the hole is anchored on third-party numbers (RoboChallenge, Penn, RoboArena); the CVE story degrades gracefully to "we harden further."
4. **Baseline success too high for ROC curves** on LIBERO with pi0.5 (~97%): rely on smolvla_libero's (unknown, likely lower) rate, LIBERO-10, LIBERO-plus tiers, and injection.
5. **Competitive compression**: NVIDIA extending Halos down-stack, Amazon productizing 3Laws via AWS, Voyage maturing, LeRobot upstreaming a real safety step. Mitigation: speed, open source, hardware-agnosticism, and the receipts/audit layer nobody else has.
6. **Standards-claim overreach is a one-way credibility door**: a single "safety-rated"-sounding sentence poisons the project. The honest-boundary language in §3.2/§6.2 is a hard editorial rule.
7. **WSL2 measurement credibility**: worst-case histograms from a VM invite reviewer attack. Mitigation: label everything, lean on simulated-time budgets and algorithmic-determinism claims, plan native-Linux/dual-boot for credible worst-case histograms later.

**Open questions (consolidated):**

- *Incident*: exact GEN-1.5 task list/platform; what the ±10% std dev is over; any independent replication since 2026-08-24; named commercial deployment partners (currently: none found).
- *openpi*: exact msgpack wire schema (capture a session before writing the Rust proxy); padded action_dim=32 confirmation; native RTC at HEAD; batching-change scope (multi-client serving affects proxy design); LIBERO env.step rate for wall-clock conversion; SAFE's exact feature tap and patch invasiveness (determines whether the sidecar contract is milestone-1); would π*0.6 RECAP value functions be usable as failure signals if released.
- *lerobot*: smolvla_libero per-suite success (must measure); EGL vs osmesa under WSL2 with the 1060 (must test); CVE #3047/PR #3048 status at build time; Python 3.13 wheel availability for sim extras; hub migration status of legacy checkpoints; whether torch 2.9–2.11 keep cu126 wheels (Pascal viability horizon).
- *Detection*: minimum calibration-rollout count for stable functional-CP bands vs split quantiles (papers range 10–100+ — needs our own ablation); cross-platform bitwise determinism of score computation (no paper addresses it — a lane for lictor); VLA-FAIL/SAFECAST/Hide-and-Seek/PATCH/RC-NF code releases and licenses; FAIL-Detect license; whether pi0 embeddings for logpZO/RND can be extracted without running the full model (3 GB constraint); whether any commercial runtime shipped a comparable monitor in 2026 (no public evidence — **UNVERIFIED**).
- *Standards*: normative text of ISO 10218-1:2025 clauses 5.5/7.5.12 and Annexes C/D/E/M/N (paywalled — purchase before quoting); whether 2025 standards address learned/non-deterministic control (likely silent; check ISO/IEC TR 5469 successor TS exact designation); per-class mandatory-vs-optional safety functions (Annex C); current UR SW 10.x PFHd tables (extracted table is dated 2018); closed-form chunk-level CBF for box joint limits + kinematic chains; defensible human-speed model for simulated SSM experiments; whether controlled retreat appears anywhere as an allowed automatic response.
- *RT engineering*: WSL2 cyclictest histograms under host load (none published — original-content opportunity); libm bit-reproducibility across ISAs at source level; copper-rs per-task deadline hooks (fuse-as-cu29-task option) and exact stable-release version; dora 1.0 final status; process-over-IPC vs pyo3 embedding (no bulla/sbx prior art for in-process Rust↔Python; pyo3 kills panic=abort); pre-allocated no-alloc verdict-path design; whether LeRobot's RTC exposes a hook between chunk generation and action emission (the natural gate point) without forking.
- *Methodology*: ground truth for "failure averted before impact" (point-of-no-return labeling in sim — SAFE et al. only use end-of-episode labels; per-timestep labels would strengthen the claim and need custom LIBERO/PushT instrumentation); can the safe-stop itself be verified not to cause harm mid-contact (certified backup controller for a manipulation VLA is open); LIBERO-plus severity → failure-probability mapping for smolvla_libero; gym-pusht/gymnasium pin validation (issue #470); final literature sweep at submission for any Aug-2026 paper plotting our exact curve.
- *Reuse*: extract the receipt model into one shared published crate vs a third vendored copy (drift risk grows per copy); canonical encoding decision (§5); one assertion language for bench experiments (Python-heavy harness argues for python3 heredocs, per bulla precedent); actual crates.io/Release status of sbx v0.1.0.

---

## 9. Full bibliography

### Incident / reliability crisis
- <https://generalistai.com/blog/gen-1.5> — GEN-1.5 release blog (primary), 2026-08-19
- <https://the-decoder.com/gen-1-5-generalist-ai-teaches-robots-new-tasks-from-a-single-demo/> — The Decoder, 2026-08-20
- <https://www.marktechpost.com/2026/08/24/generalist-ai-releases-gen-1-5-a-robot-foundation-model-that-learns-new-tasks-from-one-3-12-second-demo/> — MarkTechPost, 2026-08-24
- <https://www.techtimes.com/articles/325174/20260821/generalist-ai-gen-15-learns-new-robot-tasks-single-demo-no-retraining.htm> — TechTimes, 2026-08-21
- <https://theaiinsider.tech/2026/08/22/generalist-ai-releases-gen-1-5-robot-foundation-model-that-learns-from-a-single-demonstration/> — The AI Insider, 2026-08-22
- <https://siliconangle.com/2026/06/04/generalist-ai-raises-400m-2b-valuation-build-general-intelligence-real-world/> — SiliconANGLE, 2026-06-04
- <https://techfundingnews.com/ex-deepmind-founders-robotics-startup-generalist-hits-3b-valuation-with-200m-funding/> — Tech Funding News, ~2026-08
- <https://generalistai.com/blog/nov-04-2025-GEN-0> — GEN-0 blog, 2025-11-04
- <https://www.humanoidsdaily.com/news/generalist-ai-unveils-gen-0-claims-scaling-laws-for-robotics-backed-by-270-000-hours-of-real-world-data> — Humanoids Daily, 2025-11
- <https://arxiv.org/pdf/2510.17950> — RoboChallenge (π0.5 43.7%)
- <https://arxiv.org/abs/2506.18123> — RoboArena
- <https://research.nvidia.com/labs/gear/gr00t-n1_5/> — GR00T N1.5 eval numbers
- <https://arxiv.org/pdf/2506.01844> / <https://arxiv.org/abs/2506.01844> — SmolVLA paper
- <https://deepmind.google/blog/gemini-robotics-15-brings-ai-agents-into-the-physical-world/> — Gemini Robotics 1.5, 2025-09
- <https://arxiv.org/pdf/2606.07383> — RhinoVLA technical report (independent π0.5 numbers)
- <https://adamohq.com/blog/the-90-percent-problem-in-humanoid-deployment> — Adamo, 2026
- <https://note.com/kagawatomo/n/n0b4652d40093?hl=en> — GEN-1.5 verification caveats
- <https://tgstat.ru/en/channel/%40your_event_horizon> — Russian-language coverage path
- <https://penn-pal-lab.github.io/Pi0-Experiment-in-the-Wild/> — Penn PAL in-the-wild pi0 study

### openpi / pi0
- <https://github.com/Physical-Intelligence/openpi> — openpi repo
- <https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md> — remote inference docs
- <https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/websocket_client_policy.py> — client source
- <https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/serving/websocket_policy_server.py> — server source
- <https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md> — LIBERO eval + pi0.5 results
- <https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py> — client loop
- <https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/README.md> — DROID example
- <https://github.com/Physical-Intelligence/openpi/commits/main> — 2026 activity
- <https://github.com/Physical-Intelligence/openpi/issues/791> — pi0.6 open-sourcing (unanswered)
- <https://github.com/Physical-Intelligence/openpi/issues/799> — SimplerEnv unofficial
- <https://github.com/Physical-Intelligence/openpi/issues/599> — inference VRAM report
- <https://website.pi-asset.com/pi06star/PI06_model_card.pdf> — pi0.6 model card, 2025-11-17
- <https://www.pi.website/blog/openpi> — Open Sourcing pi0
- <https://www.pi.website/research/real_time_chunking> — RTC research page
- <https://arxiv.org/html/2506.07339v1> / <https://arxiv.org/abs/2506.07339> — RTC paper
- <https://github.com/Physical-Intelligence/real-time-chunking-kinetix> — RTC sim code
- <https://arxiv.org/html/2510.26742v1> — Running VLAs at Real-time Speed (4090 latencies)
- <https://github.com/Dexmal/realtime-vla> — optimized pi0 inference
- <https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html> — SAFE (NeurIPS proceedings)
- <https://arxiv.org/pdf/2606.15021> — Action Token Intervention steering
- <https://github.com/jackvial/drtc> — DRTC (fault-injected distributed RTC)
- <https://docs.openedgeplatform.intel.com/2026.0/edge-ai-suites/robotics-ai-suite/embodied/sample_pipelines/pi05_with_rtc.html> — Intel pi0.5+RTC pipeline
- <https://arxiv.org/pdf/2512.16881> — PolaRiS real-to-sim evaluation
- <https://huggingface.co/blog/pi0> — HF pi0 blog
- <https://github.com/huggingface/lerobot/issues/3226> — openpi vs lerobot LIBERO settings
- <https://federicosarrocco.com/blog/pi-star-06-recap> — π*0.6/RECAP coverage
- <https://www.therobotreport.com/physical-intelligence-open-sources-pi0-robotics-foundation-model/> — Robot Report
- <https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788> — NVIDIA forum latencies (UNVERIFIED)
- <https://github.com/allenzren/open-pi-zero> — pi0 reimplementation with SimplerEnv
- <https://github.com/DelinQu/SimplerEnv-OpenVLA> — SimplerEnv fork
- <https://www.jetson-ai-lab.com/tutorials/openpi_on_thor/> — openpi on Jetson Thor

### LeRobot / SmolVLA
- <https://github.com/huggingface/lerobot/blob/main/pyproject.toml> — pins
- <https://pypi.org/project/lerobot/#history> — release history
- <https://huggingface.co/docs/lerobot/main/async> — async inference
- <https://huggingface.co/docs/lerobot/main/libero> — LIBERO docs
- <https://huggingface.co/docs/lerobot/en/introduction_processors> — processors
- <https://huggingface.co/docs/lerobot/en/env_processor> — env processors (safety-limits example)
- <https://huggingface.co/docs/lerobot/en/backwardcomp> — normalization migration
- <https://github.com/huggingface/lerobot/blob/main/src/lerobot/scripts/lerobot_eval.py> — eval loop
- <https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py> — policy API
- <https://github.com/huggingface/lerobot/blob/main/src/lerobot/envs/configs.py> — env registry
- <https://huggingface.co/blog/smolvla> — SmolVLA blog
- <https://huggingface.co/blog/lerobot-release-v060> — v0.6.0 blog
- <https://github.com/huggingface/lerobot/releases/tag/v0.6.0> — v0.6.0 release
- <https://huggingface.co/blog/nvidia/nvidia-isaac-teleop-and-gr00t17-in-lerobot> — GR00T 1.7 partnership, 2026-07-06
- <https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128> — Pascal removal
- <https://github.com/pytorch/pytorch/issues/160575> — Pascal wheels discussion
- <https://chocapikk.com/posts/2026/lerobot-pickle-rce/> — CVE-2026-25874
- <https://github.com/huggingface/lerobot/issues/3047> — pickle RCE issue (open)
- <https://github.com/huggingface/lerobot/issues/3098> — low-VRAM 2-process eval
- <https://github.com/huggingface/lerobot/issues/4390> — mujoco ≥3.4.0 breaks LIBERO state
- <https://github.com/huggingface/lerobot/issues/1530> — Docker GPU rendering issue
- <https://github.com/huggingface/lerobot/issues/2114> — pi0 LIBERO reproduction discrepancy
- <https://huggingface.co/lerobot/smolvla_libero> — sim checkpoint
- <https://huggingface.co/lerobot/diffusion_pusht> — PushT checkpoint (65.4%)
- <https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human> — ALOHA ACT checkpoint (83%)
- <https://arxiv.org/pdf/2509.23224> — Leave No Observation Behind (SmolVLA 101 ms/RTX 5080)
- <https://arxiv.org/abs/2605.08168> — Async inference methods for VLAs (LIBERO+SmolVLA delay sweeps)
- <https://arxiv.org/pdf/2512.01031> — VLASH
- <https://arxiv.org/pdf/2606.08094> — vla.cpp
- <https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632> — RTX 3050 Ti anecdote (low confidence)
- <https://docs.pytorch.org/rl/main/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html> — MuJoCo rendering backends
- <https://arxiv.org/abs/2602.22818> — LeRobot library paper

### Failure detection
- <https://arxiv.org/abs/2503.08558> / <https://arxiv.org/html/2503.08558v3> — FAIL-Detect (RSS 2025)
- <https://github.com/CXU-TRI/FAIL-Detect> — code
- <https://arxiv.org/abs/2410.04640> — Sentinel/STAC (CoRL 2024)
- <https://github.com/agiachris/sentinel> — code (MIT)
- <https://sites.google.com/stanford.edu/sentinel> — project page
- <https://arxiv.org/abs/2506.09937> / <https://arxiv.org/html/2506.09937v2> — SAFE (NeurIPS 2025)
- <https://vla-safe.github.io/> — SAFE project page
- <https://github.com/vla-safe/SAFE> — code
- <https://arxiv.org/abs/2510.09459> — FIPER (NeurIPS 2025)
- <https://tum-lsy.github.io/fiper_website/> — FIPER project page
- <https://github.com/utiasDSL/fiper> — code (MIT)
- <https://arxiv.org/html/2606.08508> / <https://arxiv.org/abs/2606.08508> — ActProbe
- <https://github.com/air-embodied-brain/actprobe> — code (Apache-2.0)
- <https://arxiv.org/abs/2606.21386> / <https://arxiv.org/pdf/2606.21386> — VLA-FAIL (AUCPDT)
- <https://arxiv.org/abs/2608.04246> / <https://arxiv.org/html/2608.04246> — SAFECAST
- <https://arxiv.org/html/2605.30834> / <https://arxiv.org/abs/2605.30834> — Hide-and-Seek
- <https://arxiv.org/html/2605.22446> / <https://arxiv.org/abs/2605.22446> — Pre-VLA
- <https://arxiv.org/html/2606.16690> / <https://arxiv.org/abs/2606.16690> — PATCH
- <https://arxiv.org/html/2603.11106> — RC-NF
- <https://arxiv.org/html/2607.27511> — surgical flow-matching world-model detector
- <https://arxiv.org/html/2605.01195> — TAIL-Safe
- <https://arxiv.org/html/2410.14868v1> / <https://arxiv.org/abs/2410.14868> — Diff-DAgger (ensemble critique)
- <https://arxiv.org/abs/2410.00371> — AHA (ICLR 2025)
- <https://arxiv.org/pdf/2412.04455> — Code-as-Monitor (CVPR 2025)
- <https://arxiv.org/html/2501.04823> — conformal warning systems from sparse human feedback
- <https://arxiv.org/pdf/2501.10561> — surgical ensemble UQ
- <https://arxiv.org/html/2605.28726> / <https://arxiv.org/pdf/2605.28726> — How VLAs Fail Differently (black-box action monitoring; EU AI Act Art. 9)

### Geometric safety / standards
- <https://www.automate.org/robotics/blogs/updated-iso-10218-faq> — A3 FAQ on ISO 10218:2025
- <https://arxiv.org/pdf/2602.17822> — ISO 10218 2011-vs-2025 comparative analysis (secondary source)
- <https://www.therobotreport.com/iso-10218-industrial-robot-safety-standard-receives-major-overhaul/> — Robot Report
- <https://www.evsint.com/collaborative-robot-safety-standards-2026-iso-10218-2025-ts-15066/> — EVS
- <https://www.automate.org/robotics/news/new-ansi-a3-r15-06-2025-american-national-standard-for-industrial-robot-safety-now-available-for-purchase> — ANSI/A3 R15.06-2025
- <https://www.therobotreport.com/now-available-full-403-page-ansi-a3-r15-06-2025-robot-safety-standard/> — full standard release
- <https://www.universal-robots.com/media/1804320/e-series-functional-safety.pdf> — UR e-Series functional safety
- <https://www.universal-robots.com/manuals/EN/HTML/SW5_19/Content/prod-usr-man/complianceUR5e/H_g5_sections/safetyFunctionsAndinterfaces/safety_functions_table1_g5_en.htm> — UR safety functions table
- <https://www.universal-robots.com/manuals/EN/HTML/SW10_6/Content/prod-usr-man/hardware/arm_UR20/stopping_time_n_distances/stoppingTimeAndDistance_UR20_en.htm> — UR stopping time/distance
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC5117641/> — NIST SSM implementation (Marvel & Norcross)
- <https://www.diag.uniroma1.it/deluca/pHRI_elective/ISO_TS_15066_2016_en.pdf> — ISO/TS 15066 full-text mirror
- <https://www.iso.org/standard/80590.html> — ISO 13855:2024
- <https://eshield.pl/en/knowledge/sto-ss1-ss2-safely-stop-machine/> — STO/SS1/SS2
- <https://www.sick.com/be/en/safe-motion-drive-safety-functions/s/safe-motion-drive-safety-functions> — SICK safe motion
- <https://www.sick.com/us/en/what-are-performance-levels/w/blog-safety-standard-performance-levels> — PL/PFHd
- <https://www.sick.com/media/content/h13/h46/9693001941022.pdf> — Flexi Soft operating instructions
- <https://www.pilz.com/en-US/products/small-controllers> — PNOZmulti (PL e/SIL 3)
- <https://ldra.com/iec-61508/> — IEC 61508 overview
- <https://www.qa-systems.com/solutions/iec-61508/> — tool qualification
- <https://www.elektrobit.com/tech-corner/linux-iec-61508-automotive-safety-environment-sil-2/> — Linux SIL 2
- <https://www.elektrobit.com/products/ecu/eb-corbos/linux-for-safety-applications/> — EB corbos
- <https://arxiv.org/abs/2105.04830> — Ruckig
- <https://arxiv.org/pdf/2503.06736> — Operational-space CBFs on Franka (2.6–22.5 kHz)
- <https://github.com/danielpmorton/cbfpy> — CBFpy
- <https://github.com/bardhh/cbfkit> — CBFKit (paper <https://arxiv.org/abs/2404.07158>)
- <https://arxiv.org/html/2510.21773> — Real-Time QP Solvers review
- <https://arxiv.org/pdf/2304.11576> — exact WCET for implicit MPC
- <https://arxiv.org/html/2403.18235v1> — execution-time-certified QP
- <https://arxiv.org/abs/2002.01591> — ARMTD (RSS 2020)
- <https://huggingface.co/docs/lerobot/rtc> / <https://huggingface.co/docs/lerobot/en/rtc> — RTC in LeRobot
- <https://huggingface.co/nvidia/GR00T-N1.5-3B> — GR00T N1.5 model card (also <https://arxiv.org/abs/2503.14734>)
- <https://arxiv.org/abs/2607.29569> — barrier-enhanced flow matching (chunk-level LogSumExp CBF)
- <https://arxiv.org/html/2607.26789> — CheckVLA
- <https://arxiv.org/abs/2606.09749> — attention-guided safety filter
- <https://arxiv.org/abs/2606.23686> — LIBERO-Safety (ECCV 2026)
- <https://arxiv.org/html/2604.23775v1> — VLA safety survey

### Competitive landscape
- <https://www.therobotreport.com/3laws-secures-4-1m-in-seed-funding-to-improve-robot-safety/> — 3Laws seed
- <https://pitchbook.com/profiles/company/501876-73> — 3Laws acquisition record (Amazon, 2026-05-05)
- <https://escalatepr.com/case-study/robotics-pr-3laws-robotics-amazon-acquisition/> — acquisition case study
- <https://docs.nav2.org/tutorials/docs/using_isaac_perceptor.html> — 3Laws Supervisor Nav2 tutorial
- <https://www.prnewswire.com/news-releases/3laws-secures-4-1m-in-seed-funding-to-enable-safe-unsupervised-robot-operation-in-dynamic-environments-302266960.html> — 3Laws PR
- <https://developer.nvidia.com/blog/inside-nvidia-halos-for-robotics-a-full-stack-functional-safety-system-for-physical-ai/> — Halos technical blog
- <https://nvidianews.nvidia.com/news/nvidia-announces-halos-for-robotics-the-industrys-first-full-stack-safety-system-for-physical-ai> — Halos announcement
- <https://siliconangle.com/2026/06/22/nvidia-introduces-halos-robotics-bridge-physical-ai-safety-gap/> — SiliconANGLE on Halos
- <https://huggingface.co/nvidia/GR00T-N1.7-3B> — GR00T N1.7 model card (guardrails = deployer's responsibility)
- <https://arxiv.org/html/2602.04056> — Modular Safety Guardrails position paper
- <https://github.com/act3-ace/run-time-assurance> — AFRL RTA library
- <https://arxiv.org/pdf/2110.03506> — RTA survey (Simplex)
- <https://arxiv.org/pdf/2008.09707> — SOTER on ROS
- <https://link.springer.com/article/10.1007/s11334-024-00553-6> — Black-box Simplex (journal)
- <https://viewpoints.fov.ventures/p/10-early-stage-startups-to-watch-in-2026> — Voyage Robotics mention
- <https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/> — Gemini Robotics 2, 2026-07-30
- <https://www.automate.org/ai/industry-insights/google-deepmind-announces-gemini-robotics-2-new-safety-measures-for-humanoids> — ASIMOV-Agentic coverage
- <https://www.therobotreport.com/physical-intelligence-raises-600m-advance-robot-foundation-models/> — PI $600M
- <https://www.advancedmanufacturing.org/news-desk/putting-robot-safety-front-and-center/article_c8da04a8-df07-11ef-bd84-4b3909cc30da.html> — Figure safety center
- <https://www.1x.tech/discover/redwood-ai> — 1X Redwood AI
- <https://huggingface.co/docs/lerobot/hilserl> — LeRobot safety tools (EEBoundsAndSafety)
- <https://huggingface.co/docs/lerobot/inference> — lerobot-rollout
- <https://www.automate.org/ai/industry-insights/how-foxglove-is-navigating-the-robotics-data-gap> — Foxglove
- <https://www.inorbit.ai/automate/developers> — InOrbit
- <https://kanopylabs.com/blog/langsmith-vs-arize-vs-whylabs> — WhyLabs/Apple
- <https://andonlabs.com/evals/butter-bench> — Butter-Bench
- <https://www.startus-insights.com/innovators-guide/embodied-ai-startups/> — ReSim mention
- <https://techxplore.com/news/2026-08-scientists-robodojo-platform-embodied-ai.html> — RoboDojo
- <https://www.prnewswire.com/news-releases/fort-robotics-launches-wireless-e-stop-pro-real-time-wireless-safety-for-complex-industrial-environments-302669307.html> — Fort Robotics SIL 3 e-stop
- <https://ir.symbotic.com/news-releases/news-release-details/symbotic-acquires-veo-robotics-enhance-efficiency-and-safety> — Veo/Symbotic
- <https://www.iso.org/standard/81283.html> — ISO/IEC TR 5469:2024
- <https://etech.iec.ch/node/897> — TR 5469 successor TS
- <https://kraneshares.com/humanoid-robotics-in-2026-the-race-from-pilot-to-platform/> — intervention-log procurement guidance
- <https://index.ros.org/p/nav2_collision_monitor/> — nav2 collision monitor
- <https://github.com/ros-safety/software_watchdogs> — ROS 2 watchdogs

### bulla/sbx (reuse inventory)
- <https://github.com/RARS-oss/bulla> — repo (private, pushed 2026-08-23)
- <https://github.com/RARS-oss/bulla/blob/main/Cargo.toml> — workspace config (hermit-core vendor comment)
- <https://github.com/RARS-oss/bulla/blob/main/crates/bulla-core/src/lib.rs> — receipt machinery
- <https://github.com/RARS-oss/bulla/blob/main/Cargo.lock> — resolved crypto versions
- <https://github.com/RARS-oss/bulla/blob/main/crates/bulla-zk/src/lib.rs> — Bulletproofs range proof
- <https://github.com/RARS-oss/bulla/blob/main/.github/workflows/ci.yml> — CI
- <https://github.com/RARS-oss/bulla/blob/main/bench/README.md> — bench vectors
- <https://github.com/RARS-oss/bulla/blob/main/bench/determinism/run_experiment.sh> — determinism experiment
- <https://github.com/RARS-oss/bulla/blob/main/docs/REPORT.md> — report structure
- <https://github.com/RARS-oss/bulla/blob/main/README.md> — house style
- <https://github.com/RARS-oss/bulla/blob/main/adapters/mcp/bulla_mcp.py> — MCP adapter + WSL bridging
- <https://github.com/RARS-oss/sbx> — repo (private, pushed 2026-08-23)
- <https://github.com/RARS-oss/sbx/blob/main/Cargo.toml> — workspace config
- <https://github.com/RARS-oss/sbx/blob/main/crates/feedback/src/lib.rs> — Verdict model
- <https://github.com/RARS-oss/sbx/blob/main/crates/feedback/src/delta.rs> — Verdict diff
- <https://github.com/RARS-oss/sbx/blob/main/crates/feedback/src/history.rs> — JSONL run-memory
- <https://github.com/RARS-oss/sbx/blob/main/crates/feedback/src/render.rs> — deterministic render
- <https://github.com/RARS-oss/sbx/blob/main/docs/verdict-schema.md> — verdict contract
- <https://github.com/RARS-oss/sbx/blob/main/crates/sbx-cli/src/main.rs> — CLI shape
- <https://github.com/RARS-oss/sbx/blob/main/.github/workflows/ci.yml> — CI
- <https://github.com/RARS-oss/sbx/blob/main/.github/workflows/release.yml> — release workflow
- <https://github.com/RARS-oss/sbx/blob/main/docs/RELEASING.md> — releasing docs
- <https://github.com/RARS-oss/sbx/blob/main/bench/README.md> — bench harness
- <https://github.com/RARS-oss/sbx/blob/main/README.md> — house style

### Methodology / statistics
- <https://arxiv.org/abs/2604.16683> — Rewind-IL
- <https://arxiv.org/abs/2607.01804> — VLA-Corrector
- <https://arxiv.org/abs/2507.15886> — Cost-Constrained Runtime Monitors
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC13210885/> — uncertainty-calibrated safety gating (abstract-level verification only)
- <https://arxiv.org/abs/2005.10420> — Towards Streaming Perception
- <https://arxiv.org/abs/2409.11542> — VALO
- <https://arxiv.org/abs/2401.13585> — latency-precision scheduling
- <https://huggingface.co/docs/lerobot/libero_plus> — LIBERO-plus docs
- <https://arxiv.org/abs/2510.13626> — LIBERO-plus paper
- <https://arxiv.org/abs/2510.03827> — LIBERO-PRO
- <https://github.com/simpler-env/SimplerEnv> — SimplerEnv requirements
- <https://medium.com/toyotaresearch/statistical-thinking-for-robot-policy-evaluation-from-rigorous-a-b-testing-to-effective-0ae886fbd68d> — TRI statistics guidance
- <https://arxiv.org/abs/2503.10966> — STEP sequential stopping
- <https://arxiv.org/abs/2510.04354> — imperfect-simulator policy evaluation
- <https://arxiv.org/abs/2211.08416> — Sirius (intervention rate)
- <https://arxiv.org/pdf/2510.02298> — ARMADA
- <https://huggingface.co/blog/lerobot-release-v040> — v0.4.0 eval hub
- <https://github.com/huggingface/blog/blob/fb37fc1c2501b86d9f837e9aa4e3ff20ec41eb8d/lerobot-release-v060.md> — v0.6.0 notes
- <https://github.com/huggingface/lerobot/blob/4aaff99be4a1d81568c08c8f0296b41b40c99ec4/docs/source/molmoact2.mdx> — determinism flags
- <https://arxiv.org/abs/2407.08735> — fast/slow LLM anomaly monitor

### Real-time Rust / determinism
- <https://github.com/copper-project/copper-rs> — Copper-rs
- <https://www.codethink.co.uk/articles/2026/copper-rs-real-time-robotics.html> — Codethink hexapod case study
- <https://fosdem.org/2026/schedule/event/SK8EGJ-copper-rust-robotics-runtime/> — FOSDEM 2026
- <https://www.copper-robotics.com/> — Copper Robotics
- <https://github.com/dora-rs/dora> — dora-rs
- <https://dora-rs.ai/> — dora site
- <https://github.com/dora-rs/adora> — adora (archived 2026-05-12)
- <https://arxiv.org/abs/2603.07442> — LITHE
- <https://arxiv.org/pdf/2102.12981> — Black-Box Simplex
- <https://rust-lang.github.io/rfcs/3514-float-semantics.html> — RFC 3514 float semantics
- <https://github.com/rust-lang/rust/issues/128288> — float-semantics tracking issue
- <https://randomascii.wordpress.com/2013/07/16/floating-point-determinism/> — Dawson
- <https://gafferongames.com/post/floating_point_determinism/> — Gaffer On Games
- <https://docs.rs/libm/latest/libm/> — libm crate
- <https://rust-random.github.io/book/crate-reprod.html> — Rand reproducibility
- <https://rapier.rs/docs/user_guides/rust/determinism/> — Rapier enhanced-determinism
- <https://docs.rs/fixed> — fixed crate
- <https://wiki.linuxfoundation.org/realtime/documentation/howto/tools/cyclictest/start> — cyclictest
- <https://oneuptime.com/blog/post/2026-03-04-measure-benchmark-latency-cyclictest-rhel-real-time/view> — cyclictest methodology
- <https://people.redhat.com/williams/latency-howto/rt-latency-howto.txt> — Red Hat RT latency
- <https://en.wikipedia.org/wiki/PREEMPT_RT> — PREEMPT_RT mainlined (6.12)
- <https://github.com/Locietta/xanmod-kernel-WSL2/issues/88> — RT kernel in WSL2
- <https://github.com/pep248/WSL_realtime_kernel> — WSL RT kernel
- <https://github.com/microsoft/WSL2-Linux-Kernel> — WSL2 kernel
- <https://highscalability.com/your-load-generator-is-probably-lying-to-you-take-the-red-pi/> — coordinated omission
- <https://www.scylladb.com/2021/04/22/on-coordinated-omission/> — coordinated omission
- <https://shuhaowu.com/blog/2022/04-linux-rt-appdev-part4.html> — Linux RT programming (mlockall/SCHED_FIFO)
- <https://eci.intel.com/docs/3.3/development/performance/rt_scheduling.html> — RT scheduling
- <https://lib.rs/crates/linux-rtic-macros> — RTIC on Linux
- <https://rtic.rs/2/book/en/rtic_and_embassy.html> — RTIC/Embassy
- <https://embassy.dev/> — Embassy
- <https://github.com/mgeier/rtrb> — rtrb SPSC ring
- <https://github.com/rust-embedded/heapless> — heapless
- <https://micahrj.github.io/posts/basedrop/> — Basedrop
- <https://docs.rs/rubato/latest/rubato/> — rubato RT-safe API contract
- <https://www.gt-engineering.it/en/insights/machinery-safety/stop-functions/> — IEC 60204-1 stop categories
- <https://ez.analog.com/ez-blogs/b/engineerzone-spotlight/posts/cobots-safety-rated-monitored-stop> — SRMS
- <https://arxiv.org/html/2607.12659v3> — Jetson-PI (pi0 76 ms/chunk)
