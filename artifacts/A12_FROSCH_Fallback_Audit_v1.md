# A12 — Fallback Audit

**Project:** FROSCH Bottle Inspection Pipeline
**Handbook reference:** §5.1.6

**Requirement:** No prohibited pattern; every failure path returns an explicit
absent status — never a silently substituted measurement value.

**Method:** Static regex scan (`fallback_audit.py`) of `live_inference.py`.
**Note:** the pattern-ID taxonomy below reflects categories actually observed
in the tool's output (P1, P2, P4, P5, P6). No P3 or P7 findings were seen in
any run reviewed for this audit — if the script defines those categories,
they produced zero matches here and are not otherwise verified against the
script's source.

**Raw scan result:** 61 pattern matches across 14 enumerated failure paths.
Raw tool verdict: FAIL (the scanner has no execution context and cannot
distinguish a genuine silent fallback from guarded/benign code).

### Manual triage

| Pattern | Hits | Verdict | Basis |
|---|---|---|---|
| P1 (except → return constant) | 3 | False positive | Inside `_poll_async_ocr`; returning `None` on OCR failure is the correct explicit-absent signal |
| P2 (except: pass) | 3 | False positive | `ResourceMonitor` GPU-stat polling fallback feeds an on-screen "N/A" display string, not a measurement path |
| P4 (`list[0]` unchecked) | 3 | False positive | All three guarded by an `if list:` check on an adjacent line the regex can't see |
| P5 (`.get(key, default)`) | 49 | False positive | Internal per-bottle track-state bookkeeping (counters, history, cached flags) — `create_track()` pre-initializes the same defaults |
| P6 (`np.clip`) | 3 | False positive | Sigmoid numerical-stability clipping and bbox-to-frame-edge coordinate clamps — geometry, not measurement-result clamping |

60 of 61 raw findings verified individually as false positives.

### Open finding — NOT resolved

`finalize_measurements()` backfills `track["capacity"]` to the operator's
`--expected-capacity` CLI value whenever OCR never confirms a capacity
reading, instead of reporting the measurement as absent.

**This matches the prohibited pattern this audit exists to catch** —
§5.1.6's own language: *"A measurement that cannot be produced is reported as
absent — never replaced by a default."* An unconfirmed OCR capacity silently
becoming the expected value is a substituted measurement.

**Status: OPEN, not accepted-as-design.** Recommended fix: report capacity as
`unconfirmed` when OCR never locks a value; keep `--expected-capacity` as
recorded batch metadata rather than a substitute for the measurement. Not yet
implemented.

*(Separately, still out of scope: OCR successfully reading a capacity that
disagrees with `--expected-capacity` — e.g. a mis-sorted bottle — is not
currently flagged either. Noted for future audit, not blocking this one.)*

### Final disposition

60 of 61 raw findings: false positive. **1 of 61: open defect, unresolved.**

**Evidence file:** `A12_fallback_audit.json`
**Commit:** 3df16e135e83c1008eedefe02114e98f426eab47
**Environment:** RTX 4090, Ubuntu 22.04, CUDA 12.4, TensorRT 10.x


---

## A12 Overall: **FAIL** — one unresolved prohibited-pattern finding (capacity backfill). Returns to implementation per §5.1.6/§7.3 before this can close as PASS.
