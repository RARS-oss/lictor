#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Stdlib-only third-party verifier for lictor receipts (schema lictor-receipt/v1, profile jcs-floatfree/v1).

Usage: verify_receipt.py RECEIPT.json --pubkey HEX [--ticks TICKS.jsonl] [--ledger LEDGER.jsonl] [--parity]

Prints exactly `OK` or `FAIL: <reason>` (exit 0 / 1). `--parity` additionally prints `canonical bytes: IDENTICAL`
when the recomputed canonical bytes hash to the receipt's body_digest, and `signature: ok`. `--pubkey` is required:
a receipt is evidence only against a pinned key. No third-party packages: json, hashlib, base64, sys, pathlib, re,
plus the pure-Python Ed25519 reference verify at the bottom (slow is fine).

Why this works with `json.dumps` alone: the profile has no real numbers (they are {"f64": hex} objects), and every
object key is printable ASCII, for which Python's code-point key order equals RFC 8785's UTF-16 code-unit order.
Both facts are re-checked below on every document, never assumed. See docs/receipt-schema.md.
"""

import base64  # noqa: F401  (part of the declared stdlib surface; f64a payloads are opaque to this verifier)
import hashlib
import json
import pathlib
import re
import sys

KEY_RE = re.compile(r"^[!-~]+$")  # printable ASCII: the class that keeps Python's key sort == JCS UTF-16 sort
MAX_INT = 2**53 - 1
ZERO = "0" * 64
RECEIPT_SCHEMA = "lictor-receipt/v1"
TICKS_SCHEMA = "lictor-ticks/v1"
LEDGER_SCHEMA = "lictor-ledger/v1"
CANONICAL_ID = "jcs-floatfree/v1"


class Fail(Exception):
    pass


def check_tree(v, path="$"):
    """Every key printable ASCII, no real numbers, integers within +-2^53-1 (the jcs-floatfree/v1 profile)."""
    if isinstance(v, dict):
        for k, c in v.items():
            if not KEY_RE.match(k):
                raise Fail("bad key at %s.%r" % (path, k))
            check_tree(c, "%s.%s" % (path, k))
    elif isinstance(v, list):
        for i, c in enumerate(v):
            check_tree(c, "%s[%d]" % (path, i))
    elif isinstance(v, bool) or v is None or isinstance(v, str):
        return
    elif isinstance(v, float):
        raise Fail("float in canonical body at %s" % path)
    elif isinstance(v, int):
        if abs(v) > MAX_INT:
            raise Fail("integer out of +-2^53-1 at %s" % path)
    else:
        raise Fail("unsupported value at %s" % path)


def canon(v):
    check_tree(v)
    return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(b):
    return hashlib.sha256(b).hexdigest()


def chained_hash(event):
    e = dict(event)
    e["hash"] = ""
    return sha256_hex(canon(e))


def walk_chain(events, genesis, what):
    """Recompute a hash chain (seq continuity, prev linkage, hash); returns the head."""
    prev, first = genesis, events[0]["seq"] if events else 0
    for i, e in enumerate(events):
        if e["seq"] != first + i or e["prev"] != prev or chained_hash(e) != e["hash"]:
            raise Fail("%s chain broken at seq %d" % (what, first + i))
        prev = e["hash"]
    return prev


def read_jsonl(path):
    lines = [ln for ln in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    docs = [json.loads(ln) for ln in lines]
    if not docs or "schema" not in docs[0]:
        raise Fail("%s has no header line" % path)
    return docs[0], docs[1:]


def verify(receipt_path, pubkey, ticks_path=None, ledger_path=None):
    """Returns the list of parity lines; raises Fail with the reason otherwise."""
    try:
        sr = json.loads(pathlib.Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Fail("cannot read receipt: %s" % e)
    if not isinstance(sr, dict) or not all(k in sr for k in ("body", "body_digest", "pubkey", "sig")):
        raise Fail("malformed receipt")
    body = sr["body"]
    if body.get("schema") != RECEIPT_SCHEMA:
        raise Fail("schema mismatch")
    if body.get("canonical") != CANONICAL_ID:
        raise Fail("canonical profile mismatch")
    canonical = canon(body)
    if sha256_hex(canonical) != sr["body_digest"]:
        raise Fail("body_digest mismatch")
    if sr["pubkey"] != pubkey:
        raise Fail("pubkey mismatch")
    if not ed25519_verify(bytes.fromhex(sr["sig"]), canonical, bytes.fromhex(sr["pubkey"])):
        raise Fail("signature invalid")
    if sha256_hex(canon(body["envelope"])) != body["envelope_digest"]:
        raise Fail("envelope_digest mismatch")
    n = body["verdict_events"]
    if not (body["counts"]["ticks"] == n == body["timing_events"] == body["outcome"]["steps"]):
        raise Fail("counts mismatch")
    tail = body.get("ticks", [])
    expected_len = {"all": n, "tail32": min(32, n), "none": 0}.get(body["ticks_policy"])
    if expected_len is None or len(tail) != expected_len:
        raise Fail("embedded ticks do not match ticks_policy")
    if tail:
        genesis = ZERO if body["ticks_policy"] == "all" else tail[0]["prev"]
        if walk_chain(tail, genesis, "embedded tick") != body["verdict_chain_head"]:
            raise Fail("embedded tick chain head mismatch")
    if ticks_path is not None:
        header, ticks = read_jsonl(ticks_path)
        run = body["run"]
        if (header.get("schema"), header.get("run_id"), header.get("arm_id"), header.get("episode_index"), header.get("genesis")) != (
            TICKS_SCHEMA, run["run_id"], run["arm_id"], run["episode_index"], ZERO,
        ):
            raise Fail("ticks header mismatch")
        head = walk_chain(ticks, ZERO, "verdict")
        if len(ticks) != n:
            raise Fail("verdict chain length mismatch")
        if head != body["verdict_chain_head"]:
            raise Fail("verdict chain head mismatch")
    if ledger_path is not None:
        header, entries = read_jsonl(ledger_path)
        if header.get("schema") != LEDGER_SCHEMA or header.get("genesis") != ZERO:
            raise Fail("ledger header mismatch")
        walk_chain(entries, ZERO, "ledger")
        run = body["run"]
        mine = [e for e in entries if (e["run_id"], e["arm_id"], e["episode_index"]) == (run["run_id"], run["arm_id"], run["episode_index"])]
        if not mine:
            raise Fail("receipt not in ledger")
        if mine[-1]["receipt_digest"] != sr["body_digest"]:
            raise Fail("ledger receipt_digest mismatch")
    if not body["fuse_ok"]:
        for note in body["fuse_notes"]:
            sys.stderr.write("note: fuse_ok=false: %s\n" % note)
    return ["canonical bytes: IDENTICAL", "signature: ok"]


def main(argv):
    receipt, ticks, ledger, pubkey, parity = None, None, None, None, False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--ticks":
            ticks, i = argv[i + 1], i + 1
        elif a == "--ledger":
            ledger, i = argv[i + 1], i + 1
        elif a == "--pubkey":
            pubkey, i = argv[i + 1], i + 1
        elif a == "--parity":
            parity = True
        elif a.startswith("-") or receipt is not None:
            print("usage: verify_receipt.py RECEIPT.json --pubkey HEX [--ticks T.jsonl] [--ledger L.jsonl] [--parity]")
            return 2
        else:
            receipt = a
        i += 1
    if receipt is None or pubkey is None:
        print("FAIL: --pubkey HEX and a receipt path are required (pin the expected signing key)")
        return 1
    try:
        lines = verify(receipt, pubkey.strip().lower(), ticks, ledger)
    except Fail as e:
        print("FAIL: %s" % e)
        return 1
    except (KeyError, TypeError, ValueError, IndexError, OSError) as e:
        print("FAIL: malformed input (%s: %s)" % (type(e).__name__, e))
        return 1
    if parity:
        for ln in lines:
            print(ln)
    print("OK")
    return 0


# --- Ed25519 verify: the classic public-domain reference (Bernstein, Duif, Lange, Schwabe, Yang), ported to
# --- Python 3, plus the canonical-encoding checks (S < l, y < p) that make an honest signature the only one
# --- that passes. Pure integer arithmetic; ~1 s per signature; used only for verification.
_q = 2**255 - 19
_l = 2**252 + 27742317777372353535851937790883648493
_d = (-121665 * pow(121666, _q - 2, _q)) % _q
_I = pow(2, (_q - 1) // 4, _q)


def _xrecover(y):
    xx = (y * y - 1) * pow(_d * y * y + 1, _q - 2, _q)
    x = pow(xx, (_q + 3) // 8, _q)
    if (x * x - xx) % _q != 0:
        x = (x * _I) % _q
    return _q - x if x % 2 != 0 else x


_By = (4 * pow(5, _q - 2, _q)) % _q
_B = (_xrecover(_By), _By)


def _edwards(P, Q):
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * pow(1 + _d * x1 * x2 * y1 * y2, _q - 2, _q)
    y3 = (y1 * y2 + x1 * x2) * pow(1 - _d * x1 * x2 * y1 * y2, _q - 2, _q)
    return (x3 % _q, y3 % _q)


def _scalarmult(P, e):
    Q = (0, 1)
    while e:
        if e & 1:
            Q = _edwards(Q, P)
        P = _edwards(P, P)
        e >>= 1
    return Q


def _encodepoint(P):
    x, y = P
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _decodepoint(s):
    raw = int.from_bytes(s, "little")
    y, sign = raw & ((1 << 255) - 1), raw >> 255
    if y >= _q:
        return None  # non-canonical encoding
    x = _xrecover(y)
    if x & 1 != sign:
        x = _q - x
    if (-x * x + y * y - 1 - _d * x * x * y * y) % _q != 0:
        return None
    return (x, y)


def ed25519_verify(sig, msg, pk):
    if len(sig) != 64 or len(pk) != 32:
        return False
    R, A = _decodepoint(sig[:32]), _decodepoint(pk)
    S = int.from_bytes(sig[32:], "little")
    if R is None or A is None or S >= _l:
        return False
    h = int.from_bytes(hashlib.sha512(sig[:32] + pk + msg).digest(), "little") % _l
    return _scalarmult(_B, S) == _edwards(R, _scalarmult(A, h))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
