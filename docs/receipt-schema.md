# lictor receipt schema (`lictor-receipt/v1`) and the canonical profile (`jcs-floatfree/v1`)

Owner: WP-4. Normative companion to `docs/ARCHITECTURE.md` sec 8 and the frozen types in `crates/lictor-receipt`.
Every claim below is what the code does; the tests named in each section are the evidence.

Terms borrowed for readability; no conformance to any standard is claimed or tested.

## 1. The canonical profile `jcs-floatfree/v1`

Canonical bytes are RFC 8785 (JSON Canonicalization Scheme) over a JSON subset with no real numbers:

1. Object keys sorted by their UTF-16 code-unit sequence; no whitespace; `true` / `false` / `null` as literals.
2. Strings escaped minimally: `\"`, `\\`, `\b`, `\f`, `\n`, `\r`, `\t`; every other U+0000..U+001F as `\u00xx`
   with lowercase hex; everything else (including `/`, U+007F and all non-ASCII) emitted verbatim as UTF-8.
3. Integers as plain digits, |v| <= 2^53 - 1 (`IntegerOutOfRange` beyond). NO non-integer number may appear:
   a JSON text carrying `1.0`, `0.5` or `1e3` is rejected with `FloatInBody(path)`.
4. A real number travels as `{"f64":"<16 lowercase hex of the IEEE-754 bits>"}` (`F64Hex`); an array of reals as
   `{"f64a":"<standard padded base64 of little-endian f64 bytes>","shape":[..]}` (`F64Array`; the product of
   `shape` must equal the element count). `+inf`, `-inf`, `-0.0` and NaN payloads round-trip bit-exactly.
5. Every object key matches `^[!-~]+$` (non-empty printable ASCII; `check_keys` reports the first offender with
   its path).

Rule 5 is what makes `adapters/verify_receipt.py` a stdlib program: for printable ASCII keys, Python's code-point
key sort equals the UTF-16 code-unit sort, so `json.dumps(obj, sort_keys=True, separators=(",", ":"),
ensure_ascii=False).encode()` over `json.loads(file)` is byte-identical to lictor's canonical bytes. The Python
verifier re-checks rules 3 and 5 on every document before it hashes anything; it never assumes them.

Freeze note (reported, not patched around): the frozen text says keys match `[a-z0-9_]+`, but the frozen receipt
binds `inputs` keys such as `envelopes/pusht.toml` and `lictor:bin` and `run.host` keys such as `OMP_NUM_THREADS`.
The enforced class is therefore the one the rule exists for -- the ASCII range where the two sort orders agree --
and the `CanonError::BadKey` message names that class. `tests/rfc8785.rs`, `tests/key_charset.rs`.

Digests: `sha256(canon(value))`, hex64. `body_digest` of a signed object = sha256 over `canon(body)`; the Ed25519
signature is over the same bytes and is checked with `verify_strict`. Self-digesting objects (`HandoffRecord`,
calibration.json) digest `canon(self with the digest field = "")`.

Files on disk are pretty-printed (2-space indent, keys sorted) for humans; the canonical bytes are ALWAYS
recomputed and never trusted from disk -- a whitespace-only edit of a receipt changes nothing
(`tests/tamper.rs::whitespace_only_edits_do_not_matter`).

## 2. The two hash chains

| Chain | Event | Replayable | Bound into the receipt as |
|---|---|---|---|
| verdict | `TickEvent` -- only what a replay reproduces | yes, byte-identical on the tested target | `verdict_chain_head`, `verdict_events` |
| timing | `TimingEvent` -- measured wall-clock | no, by design | `timing_chain_head`, `timing_events` |

Rule (both chains): `h_0 = ZERO_HASH` (64 zeros); event `i` carries `prev = h_{i-1}` and
`hash = sha256(canon(event_i with hash = ""))`. `verify_tick_chain(events, genesis_prev)` and
`verify_timing_chain` walk a slice, checking seq continuity from the first event's seq, `prev` linkage and every
recomputed hash; `break_at` is the seq of the first bad event (the EXPECTED seq when an event is missing, so a
dropped event is reported at its own position). `head` is the last hash seen, or the genesis for an empty slice.

`TickEvent` fields (all bound): `seq`, `t`, `state`, `prev_state`, `status`, `trips` (TripMask bits raised this
tick), `action_src`, `substituted`, `clamped_dims`, `action` (F64Array `[action_dim]`, the action the executor
must apply), `f` (raw features `[NFEAT]`, what `lictor sweep` re-scores), `z` (standardised), `valid`, `fired`,
`s`, `tau`, `window_hits`, `brake_margin`, `reason`, `handoff_seq`, `violation_reached_env`, `prev`, `hash`.
`TimingEvent`: `seq`, `decide_ns`, `io_ns`, `prev`, `hash`.

Ticks file `ticks/<episode:06>.jsonl`: first line
`{"arm_id":..,"episode_index":..,"genesis":"00..00","run_id":..,"schema":"lictor-ticks/v1"}`, then one
canonical `TickEvent` per line. The companion `timing/<episode:06>.jsonl` uses the same rule and the same header
with an extra `"stream":"timing"` key (there is no separate frozen timing schema id).

## 3. The receipt (`SignedReceipt` = `{body, body_digest, pubkey, sig}`)

`body` is `ReceiptBody`. What each field binds (ARCHITECTURE sec 8 trust model: "binds X" means "binds the
host's declaration of X" unless the fuse verifies X itself):

| Field | Meaning | Why it is bound |
|---|---|---|
| `schema`, `canonical` | `lictor-receipt/v1`, `jcs-floatfree/v1` | a verifier refuses what it does not understand (`schema_ok`) |
| `created_epoch` | unix seconds at signing | ordering evidence; not replayed |
| `lictor_version`, `lictor_git`, `lictor_sha256` | the fuse build (sha256 of its own binary, fuse-verified) | the decision code is identified |
| `client` | the `hello.client` string | the harness client version |
| `run` | `run_id`, `arm_id`, `episode_index`, `seed`, `seed_pool`, `init_state_digest`, `env`/`policy`/`host` pin maps | paired-seed evidence: arms whose `init_state_digest` disagree for an index cannot be aggregated; every pin (lerobot #4390-class hazards) is declared |
| `budget` | `mode`, `delay_steps`, `tick_ms`, `exec_mode`, `stitch`, `on_escalate`, `tier0_armed`, `tier1_armed`, `gate`, `alpha_num/den`, `kn` | the x-axis of the curve is signed |
| `fault_injection` | `{kind, params, stream_seed}` or null | the host's declaration (an undeclared injection is not caught: trust model) |
| `envelope`, `envelope_digest` | `floatify(envelope)` and `sha256(canon(envelope))` (fuse-verified) | the full safety configuration incl. `operators` (who may re-arm) |
| `calibration_digest` | digest of calibration.json (fuse-verified) or null | a curve point cannot silently use another alpha or a calibration fitted on its own eval seeds |
| `inputs` | content-addressed map: host-declared repo-relative paths plus `lictor:bin`, `lictor:envelope`, `lictor:calibration` | what ran, by hash |
| `counts` | `VerdictCounts` (from the fuse `Tally`) | the numbers `lictor curve` recomputes from |
| `outcome` | steps, success, terminated, truncated, coverages, reward_sum, `ended_by`, `max_s`, `max_z` | the y-axis; `max_s` feeds ROC-AUC (NEG_INFINITY when nothing was valid, encoded exactly) |
| `handoffs` | every `HandoffRecord` with its ack | the human-escalation history is signed |
| `verdict_events`, `verdict_chain_head` | length and head of the verdict chain | editing one tick in the unsigned 300-line ticks file breaks `lictor verify --ticks` |
| `timing_events`, `timing_chain_head` | length and head of the timing chain | tamper-evidence for the latency data |
| `latency` | n, p50/p90/p99/p999/max ns, `label` | the histogram summary always carries its WSL2 label |
| `ticks_policy`, `ticks` | `tail32` / `all` / `none`; the embedded events | a receipt alone shows the last 32 verdicts |
| `fuse_ok`, `fuse_notes` | the honest verdict on the record (sec 5) | intact != fuse_ok |
| `ledger_prev` | hash of the previous ledger entry of this arm, or null | the receipt names its place in the ledger |

`counts_ok` requires `counts.ticks == verdict_events == timing_events == outcome.steps`.

### `evaluate_fuse` (the honest verdict on the RECORD, not on the run)

`fuse_ok` is false, with one note per cause, when: `budget.mode == observe` ("the fuse observed but did not
enforce -- this receipt does not attest protection"); `counts.violations_reached_env > 0` ("N tier-0 violations
reached the environment"); `tier1_armed` with no calibration digest ("tier 1 armed without a calibration
digest"); `terminal_state == fault` ("fuse latched Fault", including host-written `fuse_crash` receipts);
`delay_steps > 0` with no exec mode ("delay injected without an exec mode" -- unreachable from the frozen
non-optional `exec_mode`, kept as the grep target); the signing key was generated at `lictor serve` startup
("ephemeral signing key"). `verify` recomputes it from the body (`ephemeral_key` = the note is present) and
reports a disagreement with the stored flag in `notes`; `VerifyReport.fuse_ok` is true only when both agree on
true. A valid, signed, chain-intact receipt can honestly say the fuse did nothing
(`tests/tamper.rs::observe_receipt_is_intact_but_not_fuse_ok`).

### `verify(sr, expect_pubkey)` -> `VerifyReport`

`schema_ok` (schema and profile ids), `digest_ok` (recomputed `body_digest`), `sig_ok` (`verify_strict` over the
canonical bytes), `pubkey_ok` (`expect_pubkey.is_none() || sr.pubkey == expected`), `envelope_digest_ok`,
`counts_ok`, `ticks_ok` (the embedded ticks: length per `ticks_policy`, chain-consistent from `ZERO_HASH` under
`all` or from the tail's first `prev` under `tail32`, the tail ends at `verdict_events`, head ==
`verdict_chain_head`), `fuse_ok`, `break_at`, `notes`. `intact()` = the first seven. A receipt re-signed with
another key is `sig_ok` and `digest_ok` but not `pubkey_ok` when the original key is expected; every scalar leaf
of the body is bound (`tests/tamper.rs::every_scalar_of_the_body_is_bound` mutates all of them).

`verify_ticks_file(sr, ticks)` is CHAIN INTEGRITY ONLY: a re-chained file (tick 17 edited, hashes 17..299
recomputed) passes it with a different `head`; `lictor verify --ticks` compares `head` with
`verdict_chain_head` and reports `HEAD MISMATCH`, and cross-checks the ticks header against the body.

`TEST_PUBKEYS`: the pubkeys of the two committed TEST seeds -- `crates/lictor-receipt/tests/fixtures/receipt/key.hex`
(seed bytes `00..1f`, pubkey `03a107bf...31b8`) and `bench/fixtures/key.hex` (seed bytes `40..5f`, pubkey
`2543b92f...559d`; WP-10 commits that file with exactly that seed). `lictor verify` warns when a receipt carries
either.

## 4. The ledger (`lictor-ledger/v1`)

`ledger.jsonl` per arm: header `{"arm_id","genesis","run_id","schema"}`, then one canonical `LedgerEntry` per
episode: `seq` (0-based position), `run_id`, `arm_id`, `episode_index`, `seed`, `init_state_digest`,
`receipt_digest`, `verdict_chain_head`, `success`, `fuse_ok`, `tripped` (`first_trip_tick` present), `stopped`
(`first_stop_tick` present), `escalated` (`escalations > 0` or `handoff_tick` present), `prev`, `hash` =
`sha256(canon(entry with hash = ""))`. `append_ledger` reads the current head, appends one line, fsyncs (single
writer); `read_ledger` skips the header (any line with a `schema` key) and blank lines; `verify_ledger` checks
`seq == position`, linkage from `ZERO_HASH` and every hash, and tallies episodes/successes/trips/stops/escalations.

Dropping, reordering or editing an entry without the signing key breaks the chain at a reported `seq`
(`tests/tamper.rs::deleting_ledger_entry_three_breaks_at_three`). Known gap, partially closed: a backward chain
cannot by itself detect TAIL TRUNCATION (the same test shows `entries[..4]` verifies). The eval pool is declared
(run.json `seeds`, calibration `seed_pool`), so `lictor curve` asserts the ledger's index set equals the declared
pool and otherwise refuses or, under `--partial`, binds `n_declared/n_present/n_missing/missing_indices` into the
signed curve receipt. Truncation by the key-holder who also rewrites run.json is not detected (trust model);
external anchoring is roadmap.

## 5. Handoff and ack (`lictor-handoff/v1`)

`HandoffRecord.digest = sha256(canon(self with digest = "", ack = None, resolved_tick = None, outcome = ""))` --
what the operator signs; attaching the ack later does not move it. `run_id`, `arm_id` and `episode_index` are
digested, so an ack captured on one run/arm/episode does not verify on a paired re-run of the same seed;
`chain_at` is the `hash` of the `TickEvent` of the tick on which Escalated was entered (the verdict-chain head at
that moment), so the operator signs the exact fuse history they were shown. There is no `state_digest`.

`AckToken` signed bytes = `canon({schema, handoff_digest, decision, operator, nonce, note})` (the six fields; `sig`
excluded). `verify_ack` is pure: operator must be listed (slot = index) else `UnknownOperator`; `note` <= 200
characters else `TooLongNote`; schema and `verify_strict` else `BadSignature`; a pending digest must exist else
`NoPendingHandoff`; it must equal the token's else `WrongHandoff`; `nonce > last_nonce[slot]` (0 when the table
is shorter) else `NonceReplay`. `last_nonce` lives in the Session across episodes and is persisted per pubkey in
`<out>/.lictor/verifier_nonce.json` (pretty JSON `{pubkey: nonce}`); without that file acks are replayable across
processes (SECURITY.md). `note` and `reason_text` are stripped of control characters before terminal rendering.

## 6. Curve receipt (`lictor-curve/v1`)

`SignedCurve = {body: CurveReceiptBody, body_digest, pubkey, sig}` with the same mechanism; `verify_curve` reports
`ticks_ok` / `counts_ok` / `envelope_digest_ok` / `fuse_ok` as `true` (not applicable). The body binds the
ledger head and chain status, `arm_config_digest`, the one `receipt_pubkey`, `run_json_sha256`, the declared-pool
accounting, `partial` / `small_n`, mismatch lists and the `CurveMetrics` with their paired `DeltaCi`s.

## 7. History and keys

`history.jsonl` (`$LICTOR_KEYS/` or `$HOME/.lictor/`): one `EpisodeRecord` per line, capped at 200 (rewritten via
a temp file when exceeded; a torn last line is never glued to the next record). `summarize(dir, n)` reports
`tail_streak` (the most recent `first_trip_reason` and how many of the last n share it, when >= 2) and
`identical_run` (consecutive most-recent records with the same `(arm_id, trips)`).

Seed files: 64 hex characters; `#` comment lines allowed (the committed TEST keys say what they are). `save_seed`
writes 0600 on unix (`#[cfg(unix)]`), refuses `/mnt/[a-z]/` paths unless `--i-know` (DrvFs ignores 0600) and
refuses to overwrite unless `--force`. `default_key_dir` = `$LICTOR_KEYS` else `$HOME/.lictor`.

## 8. The stdlib Python verifier

`adapters/verify_receipt.py RECEIPT.json --pubkey HEX [--ticks T.jsonl] [--ledger L.jsonl] [--parity]`: checks
the profile (rules 3 and 5) on every document, `body_digest`, the pinned key (`FAIL: pubkey mismatch`), the
signature (pure-Python Ed25519 reference verify with canonical-encoding checks), `envelope_digest`, the counts, the
embedded tail, the full ticks chain and its head (`FAIL: verdict chain broken at seq N` /
`FAIL: verdict chain head mismatch`), the ledger chain and this receipt's presence in it. Prints exactly `OK` or
`FAIL: <reason>`; `--parity` adds `canonical bytes: IDENTICAL` and `signature: ok`. Exit 0 / 1. `--pubkey` is
required. `crates/lictor-receipt/tests/python_parity.rs` and `adapters/tests/test_verify.py`.

## 9. Fixtures

`crates/lictor-receipt/tests/chain.rs` regenerates every fixture under `tests/fixtures/receipt` when
`LICTOR_WRITE_FIXTURES=1` and otherwise asserts the committed files equal a fresh regeneration: a 300-tick
synthetic episode 7 (281 nominal, 12 watching, 4 braking, 3 held, first trip `brake_tier1_cp` at tick 181), its
ticks and timing files, a 5-entry ledger (episodes 3..7), `key.hex` / `pubkey.txt` / `operator.hex`,
`handoff.json` (escalation at tick 97 of the same chain) and `ack.json`.

## 10. Divergences from bulla-core (so a future shared crate is mechanical)

Vendored from RARS-oss/bulla @ 173e1cd65fb43353a9752f95077937aa4bb0f8e2, `crates/bulla-core/src/lib.rs`.

| bulla-core | lictor-receipt |
|---|---|
| canonical bytes = `serde_json::to_vec` in serde field order | RFC 8785 JCS over `jcs-floatfree/v1` (`lictor_canon::canon_of`) |
| floats as JSON numbers | no floats: `F64Hex` / `F64Array` |
| `seal_chain` hashes an `EventCore {seq, kind, detail, prev}` | events hash themselves with `hash = ""` (`TickEvent`, `TimingEvent`, `LedgerEntry`) |
| one event chain | two chains (verdict: replayable; timing: not) |
| `sign` panics on serialisation failure | `sign` returns `Result<_, ReceiptError>` |
| `verify(sr)` | `verify(sr, expect_pubkey)` adds `pubkey_ok`, `envelope_digest_ok`, `counts_ok`, `ticks_ok`, `fuse_ok`, `break_at` |
| `evaluate_seal` | `evaluate_fuse` with the six fixed notes |
| `LedgerEntry {seq, receipt_digest, seal_ok, solve_exit, grade_exit}` | per-episode fields (run/arm/index/seed/digests/flags) plus a header line, `read_ledger`, `append_ledger` |
| `generate_seed` / `seed_to_hex` / `seed_from_hex` | `keys::{keygen, load_seed, save_seed}` with comment lines, DrvFs and overwrite refusals |
| no handoff / ack / history / curve | `handoff.rs`, `history.rs`, `curve.rs` |
