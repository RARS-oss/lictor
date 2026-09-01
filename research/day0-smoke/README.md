# Day-0 smoke measurements (2026-08-31, GTX 1060 3GB, WSL2)

Raw evidence behind the numbers in `docs/VERIFIED_FACTS.md`:

- `smoke_pusht.py` / `smoke.log` — first end-to-end episode of `lerobot/diffusion_pusht` (migrated) on gym-pusht: 153.9 s wall at B=1, 3.46 s per 100-step DDPM chunk.
- `bench_batch.py` / `bench_batch.log` — denoiser throughput vs batch size (B=1..32: 2843 -> 123 ms per env-chunk, peak VRAM 2.4 GB) and the bitwise check that `n_action_steps=15` leaves the executed 8-action prefix identical to the default configuration.
- `migrate_patched.py` — the working driver for lerobot 0.6.1's checkpoint migration (stock script crashes on tuple-typed config fields).

These are measurements, not claims: WSL2 wall-clock, single machine, one seed. They size the experiment; they are not results.
