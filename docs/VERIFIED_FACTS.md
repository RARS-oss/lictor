# VERIFIED FACTS for lictor implementation agents (measured/observed on this machine, 2026-08-30/31)

## Environment (all verified by running)
- WSL2 Ubuntu, kernel 6.6.114; cargo 1.98.0 / rustc 1.98.0 stable at $HOME/.cargo/bin (NOT on PATH in non-login shells: `export PATH="$HOME/.cargo/bin:$PATH"`).
- Drive C: is FULL. Use `export CARGO_TARGET_DIR=/mnt/d/lictor/target` (verified: hello-world builds in 2 s). Repo: /mnt/c/Users/Daniil/Desktop/Robots/lictor (Windows: C:\Users\Daniil\Desktop\Robots\lictor).
- Python venv: /mnt/d/lictor/venv (Python 3.12.14). Installed: lerobot 0.6.1, torch 2.7.1+cu126, torchvision 0.22.1+cu126, gymnasium 1.3.0, gym_pusht, opencv-python-headless 5.0.0 (opencv-python was REMOVED — both installed together breaks cv2 import), diffusers (lerobot[diffusion] extra, installing).
- CUDA works on the GTX 1060 3GB (capability (6,1)); fp32 only. `import lerobot, torch` takes ~12 s from the drvfs venv.
- Set `HF_HOME=/mnt/d/lictor/hf` (checkpoints must not land on C:). Set `MUJOCO_GL=egl` (not needed for PushT, harmless).
- Run WSL commands from Windows via a script file: `MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash /mnt/d/.../script.sh` (inline quoting through PowerShell/Git-Bash breaks `$PATH`).
- uv: `~/.local/bin/uv`, with `UV_CACHE_DIR=/mnt/d/lictor/uv-cache UV_LINK_MODE=copy`. Installs on drvfs are SLOW (minutes per large wheel) — install only what is needed.

## lerobot/diffusion_pusht (from HF config.json + eval_info.json, fetched 2026-08-30)
- config: horizon=16, n_action_steps=8, n_obs_steps=2, input_features observation.image [3,96,96] VISUAL + observation.state [2] STATE, output action [2]; noise_scheduler DDPM, num_train_timesteps=100, num_inference_steps=null (=> 100 denoising steps), prediction_type=epsilon, clip_sample=true range 1.0; normalization ACTION/STATE MIN_MAX, VISUAL MEAN_STD; crop_shape [84,84] crop_is_random=true (train only); vision_backbone resnet18, spatial_softmax_num_keypoints=32; down_dims [512,1024,2048]; kernel_size 5; n_groups 8; use_film_scale_modulation true; drop_n_last_frames 7.
- eval_info.json aggregated: pc_success=65.4 (500 episodes), avg_max_reward=0.955, eval_ep_s=1.46 s/episode on the reference GPU.
- Checkpoint files: config.json, model.safetensors, train_config.json (last modified 2025-03-06 — predates lerobot 0.6 processors; `make_pre_post_processors(policy.config, pretrained_path=...)` may need the normalization-migration path; handle both).

## lerobot 0.6.1 DiffusionPolicy mechanics (read from installed source, modeling_diffusion.py)
- `DiffusionModel.generate_actions`: runs `conditional_sample` over the full horizon (16) then slices `start = n_obs_steps - 1 = 1; end = start + n_action_steps; actions[:, start:end]`. Default => indices 1..8 (8 actions). With `config.n_action_steps = horizon - n_obs_steps + 1 = 15` => indices 1..15 (15 actions); the first 8 are IDENTICAL to the default slice given the same sample. Docstring asserts the constraint `n_action_steps <= horizon - n_obs_steps + 1`.
- `DiffusionPolicy.select_action`: `populate_queues(self._queues, batch)` every call (obs deques of maxlen n_obs_steps; first step copies the obs n_obs_steps times), and ONLY when `len(self._queues[ACTION]) == 0` calls `predict_action_chunk` and extends the action deque with `actions.transpose(0,1)`; then `popleft()`.
  => TRAP: with n_action_steps=15 and plain select_action, the policy would execute 15 actions before re-planning, changing the baseline. The harness MUST drive the cadence itself: each step call `populate_queues` (or select_action-equivalent obs handling), every 8 executed steps call `policy.predict_action_chunk(batch)` (it stacks from the populated obs queues), execute the first 8 of the 15 returned actions, hand all 15 to the fuse. Assert on a fixed seed that the executed 8-prefix equals the default-config chunk.
- `predict_action_chunk(batch, noise=None)`: online mode stacks from `self._queues` when any queue is populated; `noise` may be passed explicitly => for paired-seed reproducibility, seed torch per episode (`torch.manual_seed(seed)`) or pass explicit noise of shape (B, horizon, action_dim).
- Constants: `from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE`.
- `drop_queued_actions` does NOT exist on DiffusionPolicy in 0.6.1 (0 occurrences) — clear `policy._queues[ACTION].clear()` directly or manage your own deque.

## gym_pusht (verified by running)
- `gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=300)`; obs keys: `pixels` (H,W,3 uint8, 96x96 at default) and `agent_pos` (2,) float in [0,512]; action_space Box [0,512]^2 (end-effector position target). `info["is_success"]` available; reward = goal coverage in [0,1]; success = coverage >= 0.95.
- Agent dynamics (gym_pusht/envs/pusht.py): PD control on the agent body per substep: k_p=100, k_v=20, dt=0.01, 10 substeps per env step (i.e. control period 0.1 s). Verify constants by reading the installed file before hardcoding.

## gym_pusht PD dynamics — VERIFIED from installed pusht.py (lines 164-258)
- `self.k_p, self.k_v = 100, 20`; `self.dt = 0.01`; `self.control_hz = self.metadata["render_fps"]` (see grep output: render_fps value); substeps per env step `n_steps = int(1 / (self.dt * self.control_hz))` (=> 10 when control_hz == 10, i.e. one env step = 0.1 s of sim time).
- Per substep: `acceleration = k_p * (action - agent.position) + k_v * (Vec2d(0,0) - agent.velocity)`; `agent.velocity += acceleration * dt`; `space.step(dt)` (pymunk; space.damping = 0.0 unless configured). This is a SECOND-ORDER PD system (not first-order decay): brake feasibility must forward-roll these exact equations (Euler, dt=0.01) — the closed-form ||v||/k_v is only an approximation.
- Reward: `coverage = _get_coverage()`; `reward = clip(coverage / 0.95, 0, 1)`; `terminated = is_success = coverage > 0.95`. `info` carries `coverage`, `pos_agent`, `vel_agent`, `block_pose` (block x, y, angle) — the harness should log `vel_agent` per step (ground-truth velocity for v_hat validation) and `block_pose`.
- Action space: Box [0,512]^2 float32 (target agent position). Agent radius / walls: read `pusht.py` `_setup` for the exact collision geometry if a workspace box tighter than [0,512] is needed.

## Checkpoint migration (VERIFIED 2026-08-31)
- Loading `lerobot/diffusion_pusht` directly on lerobot 0.6.1 FAILS with `ProcessorMigrationError` (hub has no policy_preprocessor.json). Fix: `python <site-packages>/lerobot/processor/migrate_policy_normalization.py --pretrained-path lerobot/diffusion_pusht --output-dir /mnt/d/lictor/models/diffusion_pusht_migrated`. The harness must load the policy AND processors from the MIGRATED LOCAL PATH: `DiffusionPolicy.from_pretrained(MODEL)` and `make_pre_post_processors(policy.config, pretrained_path=MODEL)`; expose it as an env var / CLI flag (`LICTOR_MODEL`, default that path) and record the sha256 of the migrated model.safetensors + processor JSONs in receipts as `policy_digest`.
- Extras needed beyond lerobot[pusht]: `lerobot[diffusion]` (diffusers 0.39.0). Installed.
- The stock migration script CRASHES at `policy.save_pretrained` ("Couldn't encode 84": draccus cannot encode JSON lists into tuple-typed DiffusionConfig fields crop_shape/down_dims/optimizer_betas). Working driver: /mnt/d/lictor/smoke/migrate_patched.py (exec's a patched `main()` that tuple-ifies numeric lists in cleaned_config before make_policy_config). Migrated checkpoint VERIFIED at /mnt/d/lictor/models/diffusion_pusht_migrated: config.json 2099 B, model.safetensors 1050861448 B, policy_preprocessor.json + step_3_normalizer safetensors, policy_postprocessor.json + step_0_unnormalizer safetensors, README.md. The harness WP must ship this driver as harness/migrate_checkpoint.py (same fix) so REPRODUCE.md works from a clean machine.

## MEASURED policy cost on this machine (smoke, 2026-08-31, GTX 1060 3GB fp32, B=1 single env)
- Policy load from migrated dir: 8.3 s. `import lerobot, torch`: 12-16 s (drvfs).
- One 300-step episode (seed 0, no success, max_reward 0.989): **153.9 s wall**. select_action median 0.10 ms (queue pop); each chunk inference (100 DDPM steps) mean **3.46 s**, max 6.3 s; ~38 chunk inferences per 300-step episode.
- => At B=1: 500 episodes ≈ 21 h per arm. The 1.46 s/episode in eval_info.json is lerobot's BATCHED eval (many envs per forward). The harness MUST batch environments (vectorized gym envs, one policy forward for B episodes) and the fuse server MUST multiplex B concurrent episodes (episode_id on every message). See bench_batch.log for ms-per-env-chunk vs B and peak VRAM (3 GB cap).
- Pilot sizing must be derived from the measured per-env-chunk cost at the chosen B, not from eval_info.json.

## MEASURED denoiser batch scaling (bench_batch.py, GTX 1060 3GB fp32, cudnn.benchmark=True, 100 DDPM steps)
| B (envs per forward) | ms per chunk-batch | ms per env-chunk | peak VRAM |
|---|---|---|---|
| 1 | 2843 | 2843 | 2179 MiB |
| 2 | 2779 | 1389 | 2180 MiB |
| 4 | 2820 | 705 | 2182 MiB |
| 8 | 3009 | 376 | 2189 MiB |
| 16 | 3416 | 213 | 2244 MiB |
| 32 | 3921 | 123 | 2387 MiB |
=> launch-bound: batching 32 envs gives ~23x throughput. HARNESS REQUIREMENT: run B=32 vectorized gym_pusht envs (gymnasium SyncVectorEnv or manual list; keep per-env seeds = episode ids for paired arms), one policy forward per chunk for all B envs, and multiplex B concurrent episodes through ONE fuse process (every wire message carries episode_id; the fuse keeps B independent FuseRt/session states). Expected: 38 chunks × 123 ms ≈ 4.7 s GPU per episode => 500 episodes ≈ 40 min per arm; a 100-episode pilot arm ≈ 8 min. Do NOT exceed B=32 without re-measuring VRAM (3 GB cap; B=64 untested).
- Preprocessor output keys: `observation.image` (1,3,96,96) and `observation.state` (1,2) — the pre-pipeline does NOT rename keys; `select_action` stacks images into OBS_IMAGES internally.

## VERIFIED overlap trick (bench_batch.py, bitwise)
- With `policy.config.n_action_steps = 8` (default): first select_action + remaining queue = chunk of shape (8, 2). With `n_action_steps = 15`: shape (15, 2). `torch.equal(c8, c15[:8]) == True` (max |diff| = 0.0) and the default chunk is bit-reproducible across calls under `torch.manual_seed(seed)` before select_action. So: executed prefix unchanged, published baseline preserved; overlap between consecutive 15-chunks issued every 8 steps = 7 actions (indices 8..14 of chunk k vs 0..6 of chunk k+1... precisely: chunk k covers steps t+1..t+15, chunk k+1 covers t+9..t+23 => overlap = steps t+9..t+15 = 7 steps).
