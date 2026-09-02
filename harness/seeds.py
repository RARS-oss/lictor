# SPDX-License-Identifier: MIT
"""Seed pools, the per-chunk noise hash and range parsing (ARCHITECTURE 10.6).

Pools are provably disjoint: the eval pool is 0..499 and the calibration pool 900000..900299; `run.py --check-pools`
asserts it, `lictor calibrate` refuses eval seeds and `lictor curve` lists any overlap. `hash64(seed, chunk)` seeds
the policy's noise for chunk `chunk` of episode `seed`; every intermediate value is reduced mod 2^64 because Python
integers are unbounded. Stdlib only.
"""

from __future__ import annotations

MASK64 = (1 << 64) - 1

EVAL = range(0, 500)
CALIB = range(900000, 900300)
PILOT_EVAL = range(0, 60)
PILOT_CALIB = range(900000, 900100)

POOLS = {"eval": EVAL, "calib": CALIB}
PILOTS = {"eval": PILOT_EVAL, "calib": PILOT_CALIB}

# Frozen test vectors (ARCHITECTURE 10.6 / IMPLEMENTATION_PLAN WP-8).
VECTORS = {
    (0, 0): 0xE220A8397B1DCDAF,
    (7, 3): 0xFD323448A4497C68,
    (900000, 37): 0xAF19A08D78D230A5,
    (499, 0): 0xB267F5BE46D03C23,
}


def splitmix64(x: int) -> int:
    """SplitMix64 output for state `x` (one step: add the golden gamma, then the two xor-shift-multiplies)."""
    z = (x + 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return (z ^ (z >> 31)) & MASK64


def hash64(seed: int, chunk: int) -> int:
    """`splitmix64(((seed << 32) ^ chunk) mod 2^64)`; every step reduced mod 2^64."""
    if seed < 0 or chunk < 0:
        raise ValueError("seed and chunk must be non-negative")
    return splitmix64(((seed << 32) ^ chunk) & MASK64)


def pool_of(seed: int) -> str | None:
    """`"eval"`, `"calib"` or None for a seed outside both pools."""
    if seed in EVAL:
        return "eval"
    if seed in CALIB:
        return "calib"
    return None


def pools_disjoint() -> bool:
    """True when the eval and calibration ranges share no seed (checked as ranges, not by enumeration)."""
    return EVAL.stop <= CALIB.start or CALIB.stop <= EVAL.start


def parse_range(text: str, pool: str | None = None) -> list[int]:
    """`"0-59"` -> [0..59]; `"3"`; `"0-3,7,10-12"`; `"pilot"` / `"all"` resolve against `pool` (default eval).
    Ranges are inclusive on both ends, ascending, de-duplicated."""
    text = str(text).strip()
    if not text:
        raise ValueError("empty seed range")
    if text == "pilot":
        return list(PILOTS[pool or "eval"])
    if text == "all":
        return list(POOLS[pool or "eval"])
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError("descending range %r" % part)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("no seeds in %r" % text)
    if min(out) < 0:
        raise ValueError("negative seed in %r" % text)
    return sorted(out)


def format_range(seeds) -> str:
    """Compact inverse of parse_range: [0,1,2,5,7,8] -> "0-2,5,7-8"."""
    s = sorted(set(int(x) for x in seeds))
    if not s:
        return ""
    parts = []
    start = prev = s[0]
    for x in s[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append("%d" % start if start == prev else "%d-%d" % (start, prev))
        start = prev = x
    parts.append("%d" % start if start == prev else "%d-%d" % (start, prev))
    return ",".join(parts)
