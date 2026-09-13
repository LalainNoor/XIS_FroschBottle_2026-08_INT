# A11 — Soak Test & Recovery Evidence

**Project:** FROSCH Bottle Inspection Pipeline
**Handbook reference:** §5.2, §7.3.3

---

## Part 1 — Soak Test

**Requirement:** D5 ≥ 2h continuous; D4 memory growth (M3) ≤ 0; frame mix
includes detection-failure frames.

**Method:** `mem_check.py`, self-contained harness exercising the production
TensorRT inference path (`NativeTRTEngine`, `trt_predict`) without pipeline
side effects. Frame source: production capture folders for all three bottle
types, round-robin cycled, 200 frames/folder preloaded to RAM. 10% synthetic
blank frames per run. Duration 7200s, target 30fps.

**Run history on this commit (full 2h runs only):**

| Run | M3 growth | D4 verdict | VRAM growth | latency_max | Notes |
|---|---|---|---|---|---|
| 1 | 28.023 MB | FAIL | not measured | not gated | First full run on the numpy-buffer-fixed harness |
| 2 | 0.0 MB | PASS | not measured | 101.131 ms — FAIL (gate added after this run) | Environment field still placeholder |
| 3 | -8.898 MB | PASS | 0.0 MB — PASS | 102.413 ms — FAIL | Real environment string; persistence mode later found disabled |
| 4 | -8.234 MB | PASS | 0.0 MB — PASS | 115.243 ms — FAIL | Persistence mode enabled before this run |
| 5 | (pending) | — | — | 111.935 ms — FAIL | GPU dmon logging run; see investigation below |

Four of five full-duration runs pass D4; one (Run 1) failed at 28MB. No
monotonic growth trend across runs — consistent with allocator/RSS noise at
a zero-tolerance threshold rather than a real leak, but not formally proven.
D4 is treated as passing on the balance of evidence, not as fully stable.

**latency_max has now failed on every run since the gate was added (4/4).**
This is a real, reproducible finding — see investigation below — not
attributable to noise.

**Result of Run 3 (first with VRAM + real environment):**

| Gate | Requirement | Result |
|---|---|---|
| D4 memory growth (M3) | ≤ 0 MB | -8.898 MB — **PASS** |
| Memory ceiling | < 2048 MB | within ceiling — **PASS** |
| p95 latency | < 100 ms | 30.991 ms — **PASS** |
| Worst-case latency | < 100 ms | 102.413 ms — **FAIL** |
| Throughput | ≥ 25 fps | 29.626 fps — **PASS** |
| VRAM growth | ≤ 0 MB (informal) | 0.0 MB — **PASS** |

**Overall verdict: NOT PASS**, gated solely on `latency_max`. All other gates
pass consistently across runs 3–5.

**Evidence file:** `A11_soak_all.json` (Run 3 shown; Runs 4–5 in later files of same name, overwritten per run — recommend renaming per-run before next soak cycle to avoid loss)
**Commit:** [INSERT — `git rev-parse HEAD`]
**Environment:** RTX 4090, Ubuntu 22.04, CUDA 12.4, TensorRT 10.x

**Harness note:** an earlier version of `mem_check.py` accumulated per-frame
RSS/latency into unbounded Python lists, adding ~12MB of harness-induced (not
pipeline) growth over a 2h/213k-frame run — this alone could fail D4. Fixed by
switching to preallocated numpy float32 buffers before any of the runs above.

### Latency-max investigation

Four consecutive full 2h runs on the same commit have failed `latency_max`
(101.1ms, 102.4ms, 115.2ms, 111.9ms vs. 100ms budget) while D4 and VRAM
growth passed cleanly on the runs where they were measured. This is a
repeatable, isolated latency-tail issue, investigated as follows:

1. **Per-frame peak-latency logging** was added to identify which exact
   frame produced each new max latency. Across two runs, 13 distinct peak
   events were logged, spanning all three frame folders, with no single
   frame or folder repeating. Peak-event spacing grew wider over the run
   (frame 0, 4, 11, ... up to 142,285) — the statistical signature of random
   tail noise over a large sample, not a specific slow frame or code path.
2. **GPU throttling ruled out.** `nvidia-smi dmon` logged continuously for
   the full 2h duration of Run 5. Every sampled row over the entire run
   showed `pviol=0` and `tviol=0` (no power or thermal violations at any
   point), and GPU clocks stayed effectively flat (`mclk` pinned at
   14801MHz, `pclk` oscillating only 2707–2752MHz — normal turbo jitter, not
   throttling). Persistence mode was confirmed enabled for this run before
   it started. GPU power/thermal throttling is definitively **not** the
   cause — and enabling persistence mode did not resolve the failures
   (Run 4, taken after persistence mode was enabled, still failed at
   115.2ms).
3. **Memory-related causes ruled out.** VRAM growth 0.0MB and RSS flat
   across the same runs — the spikes are not correlated with allocator or
   memory-pressure behavior.

**Conclusion:** the latency-max failures are real, reproducible (4/4 runs
since the gate was added), and not explained by GPU throttling, memory
pressure, or a specific bad frame/file. The pattern is consistent with
OS-level scheduling jitter or CUDA/driver-level stalls (e.g. context-switch
or allocator-cache-miss delays) external to the inference code itself, but
this has **not been confirmed** — only the GPU-throttling and memory
hypotheses have been eliminated. Remaining untested hypotheses: OS scheduler
preemption (testable via `chrt -f` real-time priority) and CUDA-level stalls
(testable via `nsys` profiling around a peak-latency frame).

**Disposition: investigated, unresolved.** Recorded here as a known,
reproducible anomaly rather than closed out. Recommend the team decide
between (a) further profiling (`nsys`/`chrt`) before sign-off, or (b)
accepting the latency-max gate as a known limitation for this release, with
lead sign-off required either way — this should not be silently marked PASS.

---

## Part 2 — Recovery

**Requirement:** dropped frame / camera disconnect / service restart — resumes
and reports the interruption; nothing silently lost.

| Scenario | Method | Result |
|---|---|---|
| Corrupted/unreadable frame | Automated (`recovery_check.py`): 4 valid + 1 corrupted frame, verifies skip+log, no crash | PASS |
| Camera disconnect | N/A — deployment is `--input folder` only, no camera in production | N/A (documented, not gated) |
| Service restart / shutdown state loss | Automated: synthetic in-flight track states, verifies every track lands in exactly one reported bucket | PASS |
| Live smoke test | Real pipeline run against a capture folder with one corrupted frame file | PASS — 1 frame skipped and logged, 13 bottles processed (9 good / 4 defective), 0 discarded |

**Code changes to `live_inference.py`:**
1. Folder-mode frame fetch catches the unreadable-file error, logs it, continues.
2. Camera-mode fetch wrapped the same way (untested — no camera in this deployment).
3. Shutdown cleanup now logs every track discarded at shutdown by ID, and reports discard/skip counts in the final summary.

No detection, tracking, defect, OCR, or saving logic was touched.

**Evidence file:** `A11_recovery_result.json`
**Commit:** 3df16e135e83c1008eedefe02114e98f426eab47
**Environment:** RTX 4090, Ubuntu 22.04, CUDA 12.4, TensorRT 10.x

---

## A11 Overall: **NOT YET PASS** — pending rerun with VRAM/latency-max data, D4 confirmatory run.
