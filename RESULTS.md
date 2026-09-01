# Frosch Live Inference Results

## 1. Purpose

This document records observed runtime behavior of the Frosch live-inspection pipeline (`live_inference.py`).

These are **runtime inspection counts**, not formal model-accuracy metrics. Formal accuracy requires manually verified ground truth for each bottle.

## 2. Final Validation Runs

Final recorded frame-folder validation was performed on **31 August 2026** for all three supported bottle capacities using `live_inferenc.py`.

| Bottle type | Frame folder | Frames | GOOD | DEFECTIVE | INCOMPLETE | Total |
|---|---|---:|---:|---:|---:|---:|
| 500 ml | `Capture_2026-07-15_07h31m58s` | 2,760 | 25 | 2 | 2 | 29 |
| 300 ml | `Capture_2026-07-15_07h50m00s` | 1,270 | 6 | 4 | 0 | 10 |
| 100 ml | `Capture_2026-07-15_07h53m14s` | 1,292 | 9 | 4 | 0 | 13 |

### 500 ml final counts

```text
Total: 29 | Good: 25 | Defective: 2 | Incomplete: 2
```

### 300 ml final counts

```text
Total: 10 | Good: 6 | Defective: 4 | Incomplete: 0
```

### 100 ml final counts

```text
Total: 13 | Good: 9 | Defective: 4 | Incomplete: 0
```

## 3. Observed Final-Result Behavior

### GOOD

A GOOD result requires:

```text
Orientation : PASS
H Center    : PASS
V Center    : PASS
Defects     : None
Status      : GOOD
```

### DEFECTIVE — defect detection

The runtime correctly records a bottle as DEFECTIVE when a confirmed bump or damage defect is present even when orientation and centricity pass. Example from the 100 ml run:

```text
Orientation : 0.420 deg (PASS)
H Center    : 0.011 (PASS)
V Center    : 0.119 (PASS)
Defects     : damage
Status      : DEFECTIVE
```

### DEFECTIVE — failed inspection measurement

A bottle can also be DEFECTIVE when an inspection measurement fails, even when no defect box is confirmed.

### INCOMPLETE

An INCOMPLETE result is used when required measurements remain unavailable at finalization:

```text
Orientation : Pending
H Center    : Pending
V Center    : Pending
Defects     : None
Status      : INCOMPLETE
```

The 500 ml run produced 2 INCOMPLETE bottles. These are not treated as physical defects.

## 4. Runtime Frame and Mask Information

- 500 ml run: 2,760 frames
- 300 ml run: 1,270 frames
- 100 ml run: 1,292 frames

Camera frame resolution: `2048 × 2448`

## 5. Trigger-Line Behavior

```text
TRIGGER_LINE_X_RATIO     = 0.40
TRIGGER_LINE_TOLERANCE   = 20 px
```

For the 2448-pixel-wide validation frames, the runtime trigger line is at approximately `x = 979`.

## 6. Defect Confirmation

```text
DEFECT_OVERLAP_THRESH        = 0.30
DEFECT_CONFIRMATION_FRAMES   = 1
DEFECT_MAX_MISSING_FRAMES    = 1
```

One-frame confirmation is intentional; some real defect detections are intermittent.

## 7. Centricity Stabilization

```text
CENTRICITY_SPATIAL_TOLERANCE = 0.08
CENTRICITY_MIN_HISTORY       = 3
CENTRICITY_HISTORY_WINDOW    = 5
```

Final H/V values are derived from accumulated measurements; sudden transient jumps are rejected with a log message.

## 8. Capacity OCR

OCR supports 100 ml, 300 ml, and 500 ml. If OCR does not produce a valid result before finalization, `--expected-capacity` is used as a fallback.

## 9. CSV Output

The pipeline creates a capacity-specific CSV at startup:

```text
results_100ml.csv
results_300ml.csv
results_500ml.csv
```

Columns: `Bottle`, `Capacity`, `Orientation`, `H_Center`, `V_Center`, `Defects`, `Timestamp`

## 10. Performance Optimization

### Observed FPS

| Condition | FPS range |
|---|---|
| Idle (no bottle in frame) | 83 – 96 FPS |
| Active (bottle present, full pipeline) | 22 – 35 FPS |

Active FPS includes dual TensorRT inference (detection + segmentation), mask geometry, centricity, defect validation, tracking, and image saving.

### How Performance Was Achieved

#### 1. Native TensorRT backend — single engine load at startup

Both the RF-DETR detection engine (`rfdetr-medium.trt`, input 576×576) and the segmentation engine (`rfdetr-seg-medium.trt`, input 432×432) are loaded once at process startup and reused for every frame. There is no model reload per bottle or per frame.

#### 2. Persistent CUDA buffers

GPU input/output buffers for each TensorRT engine are allocated once and reused across all frames (PERFORMANCE OPTIMIZATION 1 in the source). This eliminates per-frame GPU memory allocation and deallocation, which was a significant latency contributor in earlier versions.

#### 3. Dedicated CUDA stream per engine

Each engine runs on its own non-default CUDA stream (`execute_async_v3`). This removes TensorRT's default-stream synchronization warning and allows cleaner stream management without cross-stream blocking.

#### 4. Async OCR via ThreadPoolExecutor

EasyOCR is the slowest single operation in the pipeline (GPU-based text recognition). In the script it is submitted to a background `ThreadPoolExecutor(max_workers=1)` thread and polled non-blockingly each frame. OCR therefore runs in parallel with TensorRT inference and geometry processing rather than blocking the main loop. This was the primary change that moved active FPS from ~4–10 FPS (earlier versions with synchronous OCR) to the 22–35 FPS range observed.
#### 5. TensorRT engines initialized once

Both engines are constructed before the frame loop begins and shared across all bottles processed during a run. There is no lazy or per-bottle initialization.

### Summary of Gains

| Optimization | Impact |
|---|---|
| Persistent CUDA buffers | Eliminates per-frame GPU alloc overhead |
| Async OCR (ThreadPoolExecutor) | Primary bottleneck removed from critical path |
| Single engine load + dedicated streams | Stable baseline, no reload latency |

## 11. Formal Evaluation

This document intentionally does not claim formal model accuracy metrics. A formal evaluation requires a manually verified ground-truth table per bottle (capacity, orientation, defect label) compared against the system's final result.

## 12. Status

The final live pipeline (`live_inference.py`) is implemented and has been validated with recorded runs for all three supported bottle types. Performance satisfies the active-FPS requirement demonstrated by the validation runs.
