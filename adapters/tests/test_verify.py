# SPDX-License-Identifier: MIT
"""adapters/verify_receipt.py against the committed Rust-generated fixtures: parity, every FAIL reason, and the
tamper cases (edited tick, re-chained ticks file, deleted ledger entry, whitespace-only edit, wrong key)."""

import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "adapters" / "verify_receipt.py"
FIX = ROOT / "crates" / "lictor-receipt" / "tests" / "fixtures" / "receipt"
RECEIPT = FIX / "receipt_000007.json"
TICKS = FIX / "ticks_000007.jsonl"
LEDGER = FIX / "ledger.jsonl"
PUBKEY = (FIX / "pubkey.txt").read_text().strip()

sys.path.insert(0, str(ROOT / "adapters"))
import verify_receipt as vr  # noqa: E402


def run(*args):
    p = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, cwd=ROOT)
    return p.returncode, p.stdout.strip().splitlines()


def test_ok_with_ticks_ledger_and_parity():
    code, out = run(RECEIPT, "--ticks", TICKS, "--ledger", LEDGER, "--pubkey", PUBKEY, "--parity")
    assert code == 0, out
    assert out == ["canonical bytes: IDENTICAL", "signature: ok", "OK"]


def test_ok_receipt_only():
    assert run(RECEIPT, "--pubkey", PUBKEY) == (0, ["OK"])


def test_pubkey_is_required_and_pinned():
    code, out = run(RECEIPT)
    assert code == 1 and out[0].startswith("FAIL: --pubkey")
    assert run(RECEIPT, "--pubkey", "ab" * 32) == (1, ["FAIL: pubkey mismatch"])


def test_canonical_bytes_match_rust_digest_directly():
    sr = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert vr.sha256_hex(vr.canon(sr["body"])) == sr["body_digest"]
    # The Rust-computed tick hashes recompute from Python's canonical bytes (the same rule, the same bytes).
    for tick in sr["body"]["ticks"]:
        assert vr.chained_hash(tick) == tick["hash"]
    assert vr.ed25519_verify(bytes.fromhex(sr["sig"]), vr.canon(sr["body"]), bytes.fromhex(sr["pubkey"]))
    assert not vr.ed25519_verify(bytes.fromhex(sr["sig"]), vr.canon(sr["body"]) + b" ", bytes.fromhex(sr["pubkey"]))


def test_profile_checks():
    with pytest.raises(vr.Fail, match="float in canonical body"):
        vr.canon({"a": [1, {"b": 1.0}]})
    with pytest.raises(vr.Fail, match="bad key"):
        vr.canon({"a b": 1})
    with pytest.raises(vr.Fail, match="bad key"):
        vr.canon({"caf\u00e9": 1})
    with pytest.raises(vr.Fail, match="integer out of"):
        vr.canon({"n": 2**53})
    assert vr.canon({"lictor:bin": "x", "OMP_NUM_THREADS": "1", "envelopes/pusht.toml": "y", "z": [True, None, -1]}) == (
        b'{"OMP_NUM_THREADS":"1","envelopes/pusht.toml":"y","lictor:bin":"x","z":[true,null,-1]}'
    )
    raw = "q\"b\\s\b\f\n\r\t\x1f\x7f/\u00e9"
    assert vr.canon({"s": raw}) == '{"s":"q\\"b\\\\s\\b\\f\\n\\r\\t\\u001f\x7f/\u00e9"}'.encode("utf-8")


def test_whitespace_only_edit_still_verifies(tmp_path):
    sr = json.loads(RECEIPT.read_text(encoding="utf-8"))
    p = tmp_path / "r.json"
    p.write_text(json.dumps(sr, indent=4, sort_keys=False), encoding="utf-8")
    assert run(p, "--pubkey", PUBKEY) == (0, ["OK"])


def test_edited_body_field_fails_digest(tmp_path):
    sr = json.loads(RECEIPT.read_text(encoding="utf-8"))
    sr["body"]["run"]["seed"] += 1
    p = tmp_path / "r.json"
    p.write_text(json.dumps(sr), encoding="utf-8")
    assert run(p, "--pubkey", PUBKEY) == (1, ["FAIL: body_digest mismatch"])
    sr = json.loads(RECEIPT.read_text(encoding="utf-8"))
    sr["sig"] = "00" + sr["sig"][2:]
    p.write_text(json.dumps(sr), encoding="utf-8")
    assert run(p, "--pubkey", PUBKEY) == (1, ["FAIL: signature invalid"])
    sr = json.loads(RECEIPT.read_text(encoding="utf-8"))
    sr["body"]["ticks"][3]["t"] += 1
    sr["body_digest"] = vr.sha256_hex(vr.canon(sr["body"]))
    p.write_text(json.dumps(sr), encoding="utf-8")
    assert run(p, "--pubkey", PUBKEY) == (1, ["FAIL: signature invalid"])


def _ticks_lines():
    return [ln for ln in TICKS.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_edited_tick_breaks_the_chain_at_its_seq(tmp_path):
    lines = _ticks_lines()
    e = json.loads(lines[18])
    assert e["seq"] == 17
    e["trips"] ^= 1
    lines[18] = json.dumps(e, sort_keys=True, separators=(",", ":"))
    p = tmp_path / "ticks.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run(RECEIPT, "--ticks", p, "--pubkey", PUBKEY) == (1, ["FAIL: verdict chain broken at seq 17"])


def test_rechained_ticks_file_fails_on_the_head(tmp_path):
    lines = _ticks_lines()
    header, events = lines[0], [json.loads(ln) for ln in lines[1:]]
    events[17]["trips"] ^= 1
    prev = events[16]["hash"]
    for e in events[17:]:
        e["prev"] = prev
        e["hash"] = vr.chained_hash(e)
        prev = e["hash"]
    p = tmp_path / "ticks.jsonl"
    p.write_text("\n".join([header] + [vr.canon(e).decode() for e in events]) + "\n", encoding="utf-8")
    assert run(RECEIPT, "--ticks", p, "--pubkey", PUBKEY) == (1, ["FAIL: verdict chain head mismatch"])


def test_ticks_header_and_length_are_checked(tmp_path):
    lines = _ticks_lines()
    p = tmp_path / "ticks.jsonl"
    p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert run(RECEIPT, "--ticks", p, "--pubkey", PUBKEY) == (1, ["FAIL: verdict chain length mismatch"])
    h = json.loads(lines[0])
    h["episode_index"] = 8
    p.write_text("\n".join([json.dumps(h)] + lines[1:]) + "\n", encoding="utf-8")
    assert run(RECEIPT, "--ticks", p, "--pubkey", PUBKEY) == (1, ["FAIL: ticks header mismatch"])


def test_deleted_ledger_entry_breaks_at_three(tmp_path):
    lines = [ln for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 6
    del lines[4]  # header + entries 0..4: line 4 is seq 3
    p = tmp_path / "ledger.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert run(RECEIPT, "--ledger", p, "--pubkey", PUBKEY) == (1, ["FAIL: ledger chain broken at seq 3"])
    # The receipt must be the one the ledger recorded.
    lines = [ln for ln in LEDGER.read_text(encoding="utf-8").splitlines() if ln.strip()]
    p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert run(RECEIPT, "--ledger", p, "--pubkey", PUBKEY) == (1, ["FAIL: receipt not in ledger"])


def test_malformed_inputs_fail_cleanly(tmp_path):
    p = tmp_path / "r.json"
    p.write_text("{not json", encoding="utf-8")
    code, out = run(p, "--pubkey", PUBKEY)
    assert code == 1 and out[0].startswith("FAIL: cannot read receipt")
    p.write_text("{}", encoding="utf-8")
    assert run(p, "--pubkey", PUBKEY) == (1, ["FAIL: malformed receipt"])
    assert run(tmp_path / "missing.json", "--pubkey", PUBKEY)[0] == 1
