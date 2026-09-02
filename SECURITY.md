# Security model and trust boundary

lictor sits between a learned policy and an actuator, and it signs what it decided. Its boundary is
stated plainly: what the fuse verifies, what it merely records, who a receipt defends against, where the
keys live, and which claims are NOT made.

## Trust model

The following paragraph is normative (`docs/ARCHITECTURE.md` sec 8 and the wire section, verbatim):

> **Trust model.** The fuse VERIFIES: `mode`, dims, horizon/exec_steps, the envelope digest, the
> calibration digest and its embodiment/policy bindings, its own binary hash, and every tick's schema,
> finiteness and continuity. Everything in `binding`, `run`, `budget.delay_steps`/`exec_mode`/`stitch`,
> `fault_injection`, `inputs` (host part) and `outcome.success` is the HOST'S DECLARATION, signed by
> proxy. A receipt defends against post-hoc edits by anyone without the signing key and against
> accidental corruption; it does not defend against the experimenter, who holds the key and can rebuild a
> ledger and re-sign it. External anchoring (roadmap 10) is what would change that. Below, "binds X"
> means "binds the host's declaration of X" unless X is in the verified list.

In one sentence, the sentence the README uses: receipts bind the host's declarations and defend against
edits by anyone without the key.

| the fuse VERIFIES (computed or checked by lictor itself) | the HOST DECLARES (recorded and signed by proxy) |
|---|---|
| `mode` (observe / enforce) equals `--mode` | `binding.env`, `binding.policy`, `binding.host` (package versions, checkpoint revision, `weights_sha256`, `n_action_steps`, device, dtype, thread and env pins) |
| `embodiment_id`, `action_dim`, `pos_dim`, `horizon`, `exec_steps` equal the loaded envelope | `run` (`run_id`, `arm_id`, `episode_index`, `seed`, `seed_pool`, `init_state_digest`) |
| `envelope_digest` and `calibration_digest` in `hello` equal the loaded artefacts | `budget.delay_steps`, `exec_mode`, `stitch`, `on_escalate`, `tick_ms` |
| the calibration's `embodiment_digest` equals the envelope's (refused at startup otherwise) | `fault_injection` (a host that injects and declares `null` is not caught) |
| the calibration's `policy_digest` equals `binding.policy.weights_sha256` (refused at `episode_begin` otherwise) | `inputs` -- the host's content-addressed manifest of the harness files it ran |
| its own binary sha256 (`lictor_sha256`, `inputs["lictor:bin"]`) and the sha256 of the envelope / calibration files it loaded (`lictor:envelope`, `lictor:calibration`) | `outcome` (`success`, `steps`, `max_coverage`, `ended_by`, ...) |
| every `tick`: schema (`deny_unknown_fields` at every nesting level), dimensions, finiteness (`null` = NaN), `t` continuity, chunk `seq` continuity, `idx == t - t_emit`, `id` monotonicity | `client` (the harness client version string) |
| every ack: Ed25519 `verify_strict` against a listed operator, digest equality with the pending handoff, nonce monotonicity | the `--latency-label` sentence (defaults to the WSL2 sentence when `/proc/version` says so) |

Every count in `counts`, both chain heads, `fuse_ok`, `fuse_notes` and `violations_reached_env` are
computed by the fuse from its own verdict stream; `intact` and `fuse_ok` are different verdicts on a
receipt (a valid, signed, chain-intact receipt can honestly attest that the fuse was observing and that
137 violations reached the environment).

## What a receipt, a ticks file and a ledger defend against -- and what they do not

- **Body edit without the key.** One byte of the signed body changes `body_digest` and the Ed25519
  signature no longer verifies (`signature FAIL`, `body digest FAIL`, `intact NO`).
- **Tick edit.** Editing one `TickEvent` in the (unsigned) ticks file breaks the hash chain at that seq
  when the file is checked with `lictor verify --ticks` (`verdict chain BROKEN at seq=17`). Re-computing
  the hashes from that tick onward makes the chain link again but the recomputed head no longer equals
  the head the receipt signed (`verdict chain HEAD MISMATCH`). A ticks-file edit outside the embedded
  tail is NOT detected when `--ticks` is not passed; the receipt embeds only the last 32 events by
  default (`ticks_policy: "tail32"`).
- **Whitespace edit** of the pretty-printed receipt: nothing happens, by design -- canonical bytes are
  recomputed from the parsed structure, never trusted from disk.
- **Re-signing with another key.** The receipt verifies against its embedded `pubkey`; only `--pubkey`
  (or the `receipt_pubkey` bound into the signed curve receipt) turns that into `pubkey MISMATCH`. Pin the
  expected key out of band; `adapters/verify_receipt.py --pubkey` is required by the demo, the bench
  vectors and CI for exactly this reason.
- **Ledger.** Dropping, reordering or editing an entry without the key breaks the per-arm hash chain at
  a reported seq (`ledger BROKEN at seq=3`). A backward chain does not by itself detect truncation of the
  most recent entries. The eval pool is DECLARED (`run.json` seeds, `calibration.json` `seed_pool`), so
  `lictor curve` asserts the ledger's index set equals the declared pool and refuses otherwise, or under
  `--partial` binds `n_declared` / `n_present` / `n_missing` / `missing_indices` into the signed curve
  receipt and counts missing episodes as failures. Limit of that check: a key-holder who also rewrites
  `run.json` is not caught. An external witness (Rekor / RFC 3161) is roadmap item 10.
- **Episodes that fault, crash or hit a fatal schema error** still produce a receipt (`terminal_state:
  fault`, or a host-written `fuse_crash` receipt signed by the same key through `lictor crash-receipt`),
  so they cannot vanish from the curve. `lictor curve` refuses arms whose receipts carry more than one
  public key or any ephemeral-key note.
- **The experimenter.** Holds the key; can rebuild and re-sign everything above. The receipts are
  evidence against outsiders and against accidents, not against the person running the experiment.

## Key custody

- Keys are 32-byte Ed25519 seeds stored as 64 hex characters. `lictor key init [-o F] [--role
  signer|operator]` generates one through `getrandom` and writes it with mode `0600` (unix; a warning
  elsewhere). The default directory is `$LICTOR_KEYS`, else `$HOME/.lictor` (`key.hex`, `operator.hex`;
  `history.jsonl` and `ack_nonce.json` live beside them).
- Keys never live in the repository (`*.hex` is git-ignored; see the two exceptions below) and never
  under `/mnt/[a-z]/`: WSL DrvFs does not enforce `0600`, so `key init` refuses such a path unless
  `--i-know` is passed. It also refuses to overwrite an existing file unless `--force`.
- `key init` prints `pubkey <hex>`, `stored <path> (mode 0600)` and the custody line: `custody: this file
  signs every receipt you produce; back it up off the repo and never commit it`.
- **Rotation.** `lictor key init --force` replaces the key. Old receipts stay verifiable against the
  pubkey embedded in each of them; `lictor curve` refuses an arm signed with mixed keys, so a rotation
  mid-arm means re-running the arm, not mixing the ledgers. Keep the old public key on record.
- **Ephemeral keys.** `lictor serve` without `--key` (and with no `$LICTOR_KEYS/key.hex`) generates a
  key at startup, warns on stderr, and every receipt it writes carries `fuse_ok = false` with the note
  `ephemeral signing key`. `lictor curve` refuses such arms. Ephemeral keys exist so `replay`, `bench`,
  `selftest` and tests run without touching a real key; they never produce a reportable number.

## The two committed TEST keys

Two private keys ARE committed, on purpose, so that the fixtures and the bench vectors are reproducible
and the demo runs on a fresh clone. Both are force-added past `.gitignore` with a `# test key` comment:

| path | role | pubkey |
|---|---|---|
| `crates/lictor-receipt/tests/fixtures/receipt/key.hex` (pubkey in `pubkey.txt` beside it; `operator.hex` is the matching test operator key) | signs the receipt fixtures (`receipt_000007.json`, `ledger.jsonl`, `handoff.json`, `ack.json`) | TBD after WP-4 lands the fixture (WP-13 pastes `pubkey.txt` here) |
| `bench/fixtures/key.hex` | signs the receipts produced by `bench/*/run_experiment.sh` and `scripts/demo.sh` | TBD after WP-10 lands the fixture (WP-13 pastes the pubkey here) |

`lictor_receipt::TEST_PUBKEYS` lists both public keys, and `lictor verify` prints
`WARNING: signed with the committed test key` whenever a receipt was signed by either. A receipt signed
with a test key proves nothing about who produced it; do not sign an experiment arm with one.

## Operators, acks, and what a forged ack would need

The envelope's `operators` list (at most 8 hex64 public keys; slot = index) is inside the signed
envelope, so WHO may re-arm is bound into every receipt. Leaving `Held` (under `rearm = "ack_only"`) or
`Escalated` (always) requires an `AckToken`:

```json
{"schema":"lictor-handoff/v1","handoff_digest":"9a41...","decision":"resume","operator":"ab12...64hex","nonce":7,"note":"oracle resume","sig":"c4...128hex"}
```

`verify_ack` accepts a token only when ALL of: the `operator` pubkey is listed (slot found); the
Ed25519 `verify_strict` signature over `canon({schema, handoff_digest, decision, operator, nonce, note})`
checks; `handoff_digest` equals the digest of the CURRENTLY pending `HandoffRecord`; `nonce >
last_nonce[slot]`; `note` is at most 200 characters. A forged ack therefore needs the operator's private
key, the pending handoff digest (which digests `run_id`, `arm_id`, `episode_index`, the escalation tick
and the chain head at that tick), and a fresh nonce. A rejection is reported inside the verdict
(`ack_result: "rejected:<reason>"`), never as a fatal error, and every ack -- accepted or rejected -- is
recorded in the ticks file and the receipt.

**Ack replay.** Because `run_id` / `arm_id` / `episode_index` are digested, an ack captured on one
run / arm / episode does not verify on a paired re-run of the same seed. Within a process the
per-operator `last_nonce` is kept ACROSS episodes; with `--out` it is persisted per pubkey in
`<out>/.lictor/verifier_nonce.json` and reloaded at startup. Without that file (no `--out`, or the file
deleted) acks ARE replayable across processes -- stated here because it is the one replay path that
exists. `note` and `reason_text` are stripped of control characters before rendering.

## Input handling and surface

- **Fail-closed at both boundaries.** Inside: any non-finite input, schema or continuity violation, or
  watchdog miss latches `Fault` and substitutes the hold action for the rest of the episode; the fuse
  never returns the raw policy action after a fault, and `episode_end` is accepted while faulted so the
  episode is booked as a failure. Outside: the host's default output for a tick is
  `clamp_box(current measured position)` (position kinds) or the zero vector (velocity kinds); a late
  verdict, a fatal error, a dead child or an `id` mismatch is treated as a brake, the child is killed
  and respawned, and the episode is booked `ended_by: "fault"` or `"fuse_crash"`.
- **Wire hygiene.** One JSON object per line, at most 1 MiB, UTF-8, no embedded `\r`; every payload
  struct is `deny_unknown_fields`; `id` strictly increasing; requests and responses strictly alternate;
  `NaN` never appears as a literal (`null` is the only non-finite spelling; the Python client serialises
  with `allow_nan=False`).
- **No network surface in milestone 1.** `lictor serve` speaks NDJSON on its own stdin / stdout to a
  parent process it did not choose; it opens no socket and reads no file it was not given on the command
  line (`--envelope`, `--calibration`, `--key`, `--trace`, `--out`).
- **Adapters.** `adapters/lictor_client.py` and `adapters/verify_receipt.py` are the verified surface.
  `adapters/lerobot/lictor_lerobot.py` (a LeRobot `ProcessorStep`) and `adapters/openpi/lictor_proxy.py`
  (a websocket proxy for openpi's policy server) are UNVERIFIED roadmap stubs: importable, unit-tested
  without their upstreams, never validated end to end against a live policy server, and the proxy
  necessarily terminates a network connection. LeRobot's async inference path carries a pickle-over-gRPC
  deserialisation issue (CVE-2026-25874); lictor does not touch that path in milestone 1.
- **Terminal rendering.** `render.rs` strips ASCII control characters and escape sequences from every
  host- or operator-supplied string (`AckToken.note`, `reason_text`, `fuse_notes`, run / arm ids).

## Claims that are NOT made

The following words are banned as claims about lictor anywhere in this project (code, comments, docs,
commit messages, figure captions), and none of them is claimed here:

- "safety-rated" -- not claimed. lictor is a software fuse with a measured, WSL2-labelled latency
  histogram; no rating of any kind is claimed.
- "hard real-time" / "hard-real-time" -- not claimed. WSL2 measurements are statistics under
  virtualisation; the honest claims are algorithmic determinism, zero allocations, fixed iteration bounds
  and byte-identical replay on the tested target.
- "PL d", "SIL 2", or any performance level / integrity level -- not claimed.
- "certified", "certified limits" -- not claimed; the limits are empirical quantiles times a declared
  slack, recorded in `[fit]`.
- "engineered to <standard>", "<standard> principles" -- not claimed. Two design ideas are borrowed
  from the functional-safety literature (fail-closed defaults; a bounded, deterministic decision path);
  no conformance to IEC 61508, ISO 13849 or ISO 10218 is claimed or tested.
- SS1 / SS2 / STO / "monitored standstill" -- terms borrowed for readability; no conformance is claimed
  or tested.
- "determinism proof" -- not claimed; `replay --repeat 40` is determinism EVIDENCE on one machine.
- "bit-identical on every conforming platform" -- not claimed; bit-identical on the tested target
  (x86_64 SSE2, release and debug) and expected on targets with strict binary64 and no FMA contraction.
- "a receipt can never hide ..." -- not claimed; a receipt binds the host's declaration of what it
  records.

## Reporting

This is a research / portfolio project. If you find an issue, open a GitHub issue on `RARS-oss/lictor`
or, for something sensitive, contact the maintainer through the GitHub profile rather than posting
exploit details publicly. Include the receipt, ticks file and `lictor version` output when the issue
concerns verification.
