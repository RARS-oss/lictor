"""Day-1 microbenchmark: import time, checkpoint load, chunk latency, n_action_steps=15 overlap check."""
import os, sys, time, json, traceback, inspect
os.environ.setdefault("HF_HOME", "/mnt/d/lictor/hf")
os.environ.setdefault("MUJOCO_GL", "egl")
MODEL = os.environ.get("LICTOR_MODEL", "/mnt/d/lictor/models/diffusion_pusht_migrated")
t0 = time.perf_counter()
import torch
import lerobot
t_import = time.perf_counter() - t0
print(f"[import] lerobot+torch: {t_import:.1f}s  torch={torch.__version__} cuda={torch.cuda.is_available()}")

# --- 1. source-level check of the overlap mechanics -------------------------------
from lerobot.policies.diffusion import modeling_diffusion as md
src = inspect.getsource(md.DiffusionPolicy)
for needle in ("def predict_action_chunk", "def select_action", "n_action_steps", "n_obs_steps - 1", "_queues", "drop_queued_actions"):
    print(f"[src] '{needle}':", src.count(needle), "occurrences")
# print the predict_action_chunk / select_action bodies for the record
for name in ("predict_action_chunk", "select_action"):
    fn = getattr(md.DiffusionPolicy, name, None)
    if fn:
        print(f"----- DiffusionPolicy.{name} -----")
        print(inspect.getsource(fn))
# also the model-level generate_actions slicing
if hasattr(md, "DiffusionModel"):
    gs = inspect.getsource(md.DiffusionModel.generate_actions)
    print("----- DiffusionModel.generate_actions -----"); print(gs)

# --- 2. load checkpoint ---------------------------------------------------------------
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
t0 = time.perf_counter()
try:
    policy = DiffusionPolicy.from_pretrained(MODEL)
    print(f"[load] ok in {time.perf_counter()-t0:.1f}s; horizon={policy.config.horizon} n_action_steps={policy.config.n_action_steps} n_obs_steps={policy.config.n_obs_steps} num_inference_steps={policy.config.num_inference_steps}")
except Exception:
    traceback.print_exc(); sys.exit(2)
device = "cuda" if torch.cuda.is_available() else "cpu"
policy.to(device); policy.eval()

# --- 3. processors (lerobot 0.6 pipeline) ----------------------------------------------
try:
    from lerobot.policies.factory import make_pre_post_processors
    pre, post = make_pre_post_processors(policy.config, pretrained_path=MODEL)
    print("[proc] pre/post processors:", type(pre).__name__, type(post).__name__)
except Exception:
    traceback.print_exc(); pre = post = None

# --- 4. env ---------------------------------------------------------------------------
import gymnasium as gym, gym_pusht, numpy as np
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=300)
obs, info = env.reset(seed=0)
print("[env] obs keys:", list(obs.keys()), "pixels", obs["pixels"].shape, "agent_pos", obs["agent_pos"], "action_space", env.action_space)

def to_batch(obs):
    img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1) / 255.0   # HWC uint8 -> CHW float [0,1]
    st = torch.from_numpy(obs["agent_pos"]).float()
    return {"observation.image": img.unsqueeze(0).to(device), "observation.state": st.unsqueeze(0).to(device)}

def run_episode(seed, n_steps_cap=300, label=""):
    policy.reset()
    obs, _ = env.reset(seed=seed)
    lat, chunk_calls, steps, done = [], 0, 0, False
    max_reward = 0.0
    while not done and steps < n_steps_cap:
        batch = to_batch(obs)
        if pre is not None:
            batch = pre(batch)
        t = time.perf_counter()
        with torch.inference_mode():
            a = policy.select_action(batch)
        if device == "cuda": torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
        if post is not None:
            a = post(a)
        a = a.squeeze(0).detach().cpu().numpy()
        obs, r, term, trunc, info = env.step(a)
        max_reward = max(max_reward, float(r)); done = term or trunc; steps += 1
    lat = np.array(lat)
    big = lat[lat > np.median(lat) * 3]   # the calls that ran the denoiser
    print(f"[ep{label}] seed={seed} steps={steps} max_reward={max_reward:.3f} success={info.get('is_success')} "
          f"select_action: median={np.median(lat)*1e3:.2f}ms  inference-calls={len(big)} mean={big.mean()*1e3 if len(big) else 0:.0f}ms max={lat.max()*1e3:.0f}ms")
    return steps, max_reward, lat

t0 = time.perf_counter()
run_episode(0, label="-default")
print(f"[wall] episode: {time.perf_counter()-t0:.1f}s")

# --- 5. overlap check: n_action_steps=15 -> predict_action_chunk shape ----------------------
try:
    policy.reset(); obs, _ = env.reset(seed=1)
    b = to_batch(obs)
    if pre is not None: b = pre(b)
    # feed n_obs_steps observations so the queue is full
    with torch.inference_mode():
        for _ in range(policy.config.n_obs_steps):
            policy._queues["observation.image"].append(b["observation.image"]) if hasattr(policy, "_queues") else None
        chunk_default = None
        if hasattr(policy, "predict_action_chunk"):
            from lerobot.utils.constants import OBS_IMAGE, OBS_STATE
            bb = {k: torch.stack(list(v), dim=1) for k, v in policy._queues.items() if k in b} if hasattr(policy, "_queues") else b
            try:
                chunk_default = policy.predict_action_chunk(bb)
                print("[overlap] default n_action_steps=%d -> predict_action_chunk shape %s" % (policy.config.n_action_steps, tuple(chunk_default.shape)))
            except Exception as e:
                print("[overlap] predict_action_chunk(default) failed:", type(e).__name__, e)
    old = policy.config.n_action_steps
    policy.config.n_action_steps = policy.config.horizon - policy.config.n_obs_steps + 1
    policy.reset(); obs, _ = env.reset(seed=1)
    b = to_batch(obs)
    if pre is not None: b = pre(b)
    with torch.inference_mode():
        a1 = policy.select_action(b)
        qlen = len(policy._queues["action"]) if hasattr(policy, "_queues") and "action" in policy._queues else None
    print(f"[overlap] override n_action_steps={policy.config.n_action_steps}: first select_action ok, action queue length after call = {qlen} (expect {policy.config.n_action_steps-1})")
    policy.config.n_action_steps = old
except Exception:
    traceback.print_exc()
print("[done]")
