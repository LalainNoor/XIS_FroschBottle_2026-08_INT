# Frosch Live Inference Results

## 1. Purpose

This document records observed runtime behavior of the final Frosch live-inspection pipeline.

These are **runtime inspection counts**, not formal model-accuracy metrics. Formal accuracy requires manually verified ground truth for each bottle and comparison against the system's final result.

## 2. Final Validation Runs

Final recorded frame-folder validation was performed on **21 August 2026** for all three supported bottle capacities.

| Bottle type | Frame folder | Frames | GOOD | DEFECTIVE | INCOMPLETE | Total |
|---|---|---:|---:|---:|---:|---:|
| 500 ml | `Capture_2026-07-15_07h31m58s` | 2,760 | 45 | 3 | 5 | 53 |
| 300 ml | `Capture_2026-07-15_07h50m00s` | 1,270 | 8 | 7 | 0 | 15 |
| 100 ml | `Capture_2026-07-15_07h53m14s` | 1,292 | 14 | 6 | 0 | 20 |

These totals are the final runtime results emitted by the current inference script for the three validation runs.

### 500 ml final counts

```text
Total: 53 | Good: 45 | Defective: 3 | Incomplete: 5
```

### 300 ml final counts

```text
Total: 15 | Good: 8 | Defective: 7 | Incomplete: 0
```

### 100 ml final counts

```text
Total: 20 | Good: 14 | Defective: 6 | Incomplete: 0
```

## 3. Interpretation

The above counts represent bottles finalized by the runtime pipeline during recorded validation runs.

They should **not** be interpreted as:

- formal classification accuracy,
- precision,
- recall,
- F1,
- mAP,
- defect rate,
- or ground-truth correctness.

A ground-truth evaluation set is required before those metrics can be reported.

## 4. Observed Final-Result Behavior

### GOOD

A GOOD result requires:

```text
Orientation : PASS
H Center    : PASS
V Center    : PASS
Defects     : None
Status      : GOOD
```

The final 500 ml run included multiple GOOD examples with orientation values around 4–6 degrees and H/V values within the configured thresholds.

### DEFECTIVE — defect detection

The runtime correctly records a bottle as DEFECTIVE when a confirmed bump or damage defect is present even when orientation and centricity pass.

Examples in the final logs include:

```text
Orientation : 5.766 deg (PASS)
H Center    : 0.093 (PASS)
V Center    : 0.029 (PASS)
Defects     : bump
Status      : DEFECTIVE
```

and 100 ml examples such as:

```text
Orientation : 0.212 deg (PASS)
H Center    : 0.029 (PASS)
V Center    : 0.120 (PASS)
Defects     : bump
Status      : DEFECTIVE
```

### DEFECTIVE — failed inspection measurement

A bottle can also be DEFECTIVE when an inspection measurement fails, even when no defect box is confirmed.

This is separate from the defect-model path.

### INCOMPLETE

An INCOMPLETE result is used when required measurements remain unavailable:

```text
Orientation : Pending
H Center    : Pending
V Center    : Pending
Defects     : None
Status      : INCOMPLETE
```

The final 500 ml validation produced 5 INCOMPLETE bottles. These should be reviewed separately from DEFECTIVE bottles because missing measurements are not treated as physical defects.

## 5. Runtime Frame and Mask Information

The three validation runs used recorded camera frames.

The logs show:

- 500 ml run: 2,760 frames
- 300 ml run: 1,270 frames
- 100 ml run: 1,292 frames

The camera frame resolution used by the final pipeline is:

```text
2048 x 2448
```

The bottle segmentation mask is resized to the same frame resolution before mask-based geometry is calculated.

## 6. Trigger-Line Behavior

The final script uses:

```text
TRIGGER_LINE_X_RATIO = 0.40
TRIGGER_LINE_TOLERANCE = 20 px
```

For the 2448-pixel-wide validation frames, the runtime trigger line is reported at approximately:

```text
x = 979
```

Crossing the line finalizes the bottle without waiting for the full missing-frame fallback.

## 7. Defect Confirmation

The final script uses:

```text
DEFECT_OVERLAP_THRESH = 0.30
DEFECT_CONFIRMATION_FRAMES = 1
DEFECT_MAX_MISSING_FRAMES = 1
```

A candidate defect must overlap the actual bottle segmentation mask sufficiently before it can affect the final result.

The final pipeline intentionally uses one-frame confirmation because some real defect detections can be intermittent.

## 8. Centricity Stabilization

The runtime logs show the stabilization logic rejecting sudden label-position changes with messages such as:

```text
centricity sudden jump ignored
```

The current stabilization configuration is:

```text
CENTRICITY_SPATIAL_TOLERANCE = 0.08
CENTRICITY_MIN_HISTORY = 3
CENTRICITY_HISTORY_WINDOW = 5
```

Final H/V values are taken from accumulated measurements rather than blindly using a single transient frame.

## 9. Capacity OCR

Capacity OCR supports:

```text
100 ml
300 ml
500 ml
```

During a run, capacity may initially be unavailable:

```text
Capacity : Not detected ml
```

and become available later:

```text
Bottle #N capacity updated: 500 ml
```

The current finalization logic uses `--expected-capacity` as a fallback if OCR never produces a valid capacity for the track.

## 10. CSV Output

The final script creates a capacity-specific CSV at startup:

```text
results_100ml.csv
results_300ml.csv
results_500ml.csv
```

Columns:

```text
Bottle
Capacity
Orientation
H_Center
V_Center
Defects
Timestamp
```

The CSV is intended for traceability and runtime result logging.

## 11. Validation Status

The final pipeline was exercised on all three supported bottle capacities:

- 500 ml
- 300 ml
- 100 ml

The final implementation includes:

- bottle detection,
- bottle segmentation,
- bottle tracking,
- capacity OCR,
- capacity-specific H/V centricity references,
- mask-based orientation,
- defect validation against the bottle mask,
- defect confirmation,
- final GOOD/DEFECTIVE/INCOMPLETE classification,
- complete-bottle saving,
- annotated output preservation,
- CSV logging.

## 12. Formal Evaluation

This document intentionally does not claim formal model accuracy metrics.

A formal project evaluation should compare, for every tested bottle:

| Bottle | Ground Truth | System Result | Correct? | Capacity Correct? | Orientation Correct? | H Correct? | V Correct? | Defect Correct? |
|---|---|---|---|---|---|---|---|---|

From that table, formal metrics can be calculated after manual review.

## 13. Status

The final live pipeline is implemented and has been validated with recorded runs for all three supported bottle types.

The results document distinguishes:

- verified runtime behavior,
- observed validation counts,
- implementation behavior,
- and formal metrics that require a separate ground-truth evaluation.
