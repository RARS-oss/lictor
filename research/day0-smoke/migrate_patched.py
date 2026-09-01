"""Run lerobot's migrate_policy_normalization.main() with a one-line fix:
draccus cannot encode JSON lists into tuple-typed config fields (crop_shape=[84,84] -> "Couldn't encode 84").
We tuple-ify numeric lists in cleaned_config right before make_policy_config."""
import os, sys, inspect
os.environ.setdefault("HF_HOME", "/mnt/d/lictor/hf")
OUT = "/mnt/d/lictor/models/diffusion_pusht_migrated"
from lerobot.processor import migrate_policy_normalization as mig
src = inspect.getsource(mig.main)
needle = "policy_config = make_policy_config(policy_type, **cleaned_config)"
assert needle in src, "migration script changed; needle not found"
fix = ("cleaned_config = {k: (tuple(v) if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v) else v) "
       "for k, v in cleaned_config.items()}\n    " + needle)
patched = src.replace(needle, fix)
ns = dict(vars(mig)); exec(compile(patched, mig.__file__, "exec"), ns)
sys.argv = ["migrate", "--pretrained-path", "lerobot/diffusion_pusht", "--output-dir", OUT]
ns["main"]()
print("=== migrated dir ===")
for f in sorted(os.listdir(OUT)):
    print(f"{os.path.getsize(os.path.join(OUT, f)):>12}  {f}")
