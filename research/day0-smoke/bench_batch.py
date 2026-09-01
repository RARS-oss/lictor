"""(1) conditional_sample throughput vs batch size on the 1060; (2) n_action_steps=15 overlap check via select_action."""
import os, time, torch, numpy as np
os.environ.setdefault("HF_HOME", "/mnt/d/lictor/hf")
MODEL = "/mnt/d/lictor/models/diffusion_pusht_migrated"
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
policy = DiffusionPolicy.from_pretrained(MODEL).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=MODEL)
import gymnasium as gym, gym_pusht
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array", max_episode_steps=300)
obs, _ = env.reset(seed=0)
def batch_of(obs):
    img = torch.from_numpy(obs["pixels"]).float().permute(2, 0, 1) / 255.0
    st = torch.from_numpy(obs["agent_pos"]).float()
    return pre({"observation.image": img.unsqueeze(0).cuda(), "observation.state": st.unsqueeze(0).cuda()})
b = batch_of(obs)
print("[keys after pre]", {k: tuple(v.shape) for k, v in b.items() if torch.is_tensor(v)})

# ---- (1) batch scaling of the denoiser ----
torch.backends.cudnn.benchmark = True
n_obs = policy.config.n_obs_steps
for B in (1, 2, 4, 8, 16, 32):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        bb = {OBS_STATE: b[OBS_STATE].repeat(B, 1).unsqueeze(1).repeat(1, n_obs, 1)}
        imgk = "observation.image" if "observation.image" in b else [k for k in b if "image" in k][0]
        im = b[imgk]
        if im.ndim == 4: im = im.unsqueeze(1)             # (B,1,C,H,W)
        bb[OBS_IMAGES] = im.repeat(B, n_obs, 1, 1, 1).unsqueeze(2)   # (B, n_obs, n_cam=1, C, H, W)
        with torch.inference_mode():
            policy.diffusion.generate_actions(bb); torch.cuda.synchronize()   # warmup
            t = time.perf_counter(); N = 2
            for _ in range(N): policy.diffusion.generate_actions(bb)
            torch.cuda.synchronize(); dt = (time.perf_counter() - t) / N
        print(f"[batch] B={B:2d}: {dt*1e3:7.0f} ms/chunk-batch  = {dt/B*1e3:6.0f} ms per env-chunk  | peak VRAM {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
    except Exception as e:
        print(f"[batch] B={B}: FAILED {type(e).__name__}: {str(e)[:120]}"); break

# ---- (2) overlap check ----
def first_chunk(n_action_steps, seed=123):
    policy.config.n_action_steps = n_action_steps
    policy.reset(); o, _ = env.reset(seed=0); bb = batch_of(o)
    torch.manual_seed(seed)
    with torch.inference_mode():
        a0 = policy.select_action(bb)
    rest = torch.stack(list(policy._queues[ACTION]), 0).squeeze(1) if len(policy._queues[ACTION]) else torch.empty(0, 2)
    return torch.cat([a0, rest], 0).cpu()
c8 = first_chunk(8); c15 = first_chunk(15); c8b = first_chunk(8)
print(f"[overlap] default chunk shape {tuple(c8.shape)}; override chunk shape {tuple(c15.shape)}")
print(f"[overlap] default reproducible with same seed: {torch.equal(c8, c8b)}")
print(f"[overlap] executed 8-prefix identical (bitwise): {torch.equal(c8, c15[:8])}   max|diff|={ (c8 - c15[:8]).abs().max().item():.3e}")
policy.config.n_action_steps = 8
print("[done]")
