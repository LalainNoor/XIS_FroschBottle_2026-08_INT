# Frosch Live Inference Results

## 1. Purpose

This document records observed runtime behavior of the current Frosch live-inspection pipeline.

These are **runtime inspection counts**, not formal model-accuracy metrics.

A formal accuracy report requires manually verified ground truth for every bottle and comparison against the system's final result.

## 2. Recorded Validation Run

A recorded inference run on 12 August 2026 reached:

| Final category | Count |
|---|---:|
| GOOD | 44 |
| DEFECTIVE | 4 |
| INCOMPLETE | 3 |
| **Total finalized** | **51** |

The runtime log shows the final count after Bottle #52 was finalized:

```text
Final counts -> Total: 51 | Good: 44 | Defective: 4 | Incomplete: 3
```

The same run also shows that the pipeline saved original and annotated images under the corresponding result directory.

## 3. Interpretation

The 51 bottles represent bottles finalized by the runtime pipeline during that recorded run.

They should **not** be interpreted as:

- 44 true positives for GOOD,
- 4 true positives for DEFECTIVE,
- 3 true positives for INCOMPLETE,
- 86.3% model accuracy,
- 7.8% defect rate,
- or any other accuracy statistic.

Those interpretations require a manually reviewed ground-truth result for each bottle.

## 4. Examples of Observed Final Results

### GOOD

The runtime produced examples such as:

```text
Orientation : PASS
H Center    : PASS
V Center    : PASS
Defects     : None
Status      : GOOD
```

and saved both:

```text
original.jpg
annotated.jpg
```

under the `good` directory.

### DEFECTIVE — Centricity Failure

A recorded example showed:

```text
Orientation : PASS
H Center    : FAIL
V Center    : PASS
Defects     : None
Status      : DEFECTIVE
```

This demonstrates that a centricity failure alone can make the final result DEFECTIVE.

### DEFECTIVE — Confirmed Defect

A recorded example showed:

```text
Orientation : PASS
H Center    : PASS
V Center    : PASS
Defects     : bump
Status      : DEFECTIVE
```

This demonstrates that a confirmed bump can make an otherwise passing bottle DEFECTIVE.

### INCOMPLETE

Recorded runs also produced:

```text
Orientation : Pending
H Center    : Pending
V Center    : Pending
Defects     : None
Status      : INCOMPLETE
```

The bottle was saved under:

```text
saved_bottles/incomplete/
```

## 5. Runtime Frame and Mask Information

The camera stream used in recorded runs produced frames of:

```text
2048 x 2448
```

The segmentation mask was resized to the same frame resolution:

```text
2048 x 2448
```

The logs repeatedly confirmed matching frame/mask dimensions before orientation processing.

## 6. Trigger-Line Behavior

The current script uses:

```text
TRIGGER_LINE_X_RATIO = 0.40
```

For the recorded 2448-pixel-wide frames, the runtime trigger line was reported at:

```text
x = 979
```

A bottle crossing the trigger line caused finalization without waiting for the full missing-frame fallback.

This was observed in recorded runs with messages such as:

```text
Bottle #31 crossed trigger line; finalizing stable result: GOOD
```

and:

```text
Bottle #32 crossed trigger line; finalizing stable result: DEFECTIVE
```

## 7. Defect Confirmation

The current script requires a defect to remain valid for two consecutive frames before it is accepted:

```text
DEFECT_CONFIRMATION_FRAMES = 2
```

The defect also needs sufficient overlap with the actual bottle segmentation mask:

```text
DEFECT_OVERLAP_THRESH = 0.30
```

This is intended to reduce isolated one-frame false positives and detections outside the actual bottle.

## 8. Centricity Stabilization Observations

Runtime logs show the stabilization logic rejecting sudden label-position changes.

Examples include messages such as:

```text
centricity sudden jump ignored
```

The current implementation therefore does not blindly accept every frame's H/V measurement.

The stored H/V history is used to obtain a more stable final result before the bottle is finalized.

## 9. Capacity OCR Observations

The runtime initially reports capacity as not detected when OCR has not yet succeeded.

For example:

```text
Capacity : Not detected
```

Later frames can update the track:

```text
Bottle #1 capacity updated: 500 ml
```

This is why the live pipeline keeps the bottle active across multiple frames instead of finalizing immediately after the first detection.

## 10. CSV Output

Each run creates a timestamped CSV:

```text
results_YYYYMMDD_HHMMSS.csv
```

with:

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

## 11. Current Known Issues / Areas for Further Validation

The observed runtime behavior indicates that the system is functional, but several areas still require proper validation before claiming production-level accuracy:

### 11.1 H/V centricity

Some bottles can receive DEFECTIVE status from an H/V FAIL even when no physical defect is detected.

This may be a genuine centricity failure or a label/bounding-box association issue. Ground-truth review is required to distinguish the two.

### 11.2 Incomplete bottles

INCOMPLETE results occur when required measurements remain unavailable.

These cases should be reviewed to determine whether the root cause is:

- missing segmentation,
- missing label detection,
- short track duration,
- partial bottle visibility,
- or another association issue.

### 11.3 Segmentation alignment

The current script includes explicit mask resizing and bounding-box constraints to reduce mask/frame misalignment.

The saved annotated images should still be visually reviewed to confirm that the segmentation contour follows the actual bottle boundary.

### 11.4 Defect quality

The defect confirmation logic reduces one-frame false positives, but it does not guarantee defect-model accuracy.

Damage/bump detections should be compared with manually reviewed samples.

### 11.5 Formal accuracy metrics

The current results document does not claim precision, recall, F1, mAP, or classification accuracy for the live inspection pipeline.

Those metrics should be calculated from a ground-truth evaluation set.

## 12. Recommended Final Evaluation

For a final project-level evaluation, create a manually reviewed table with at least:

| Bottle | Ground Truth | System Result | Correct? | Capacity Correct? | Orientation Correct? | H Correct? | V Correct? | Defect Correct? |
|---|---|---|---|---|---|---|---|---|

Then calculate:

- overall result accuracy,
- GOOD precision/recall,
- DEFECTIVE precision/recall,
- INCOMPLETE rate,
- defect-type precision/recall,
- capacity recognition accuracy,
- orientation accuracy,
- H-centering accuracy,
- V-centering accuracy.

## 13. Status

The live pipeline is implemented and has been exercised on recorded camera runs.

The current documentation distinguishes:

- verified runtime behavior,
- observed counts,
- known implementation limitations,
- and metrics that still require formal ground-truth evaluation.

## New Bottle Type Validation — 13 August 2026

A recorded frame-folder test was performed using:

- Frame source: `Capture_2026-07-15_07h53m14s`
- Total frames: 1,292
- Frame resolution: 2048 × 2448
- Bottle type: 100 ml new bottle type

### Final runtime results

| Final category | Count |
|---|---:|
| GOOD | 10 |
| DEFECTIVE | 4 |
| INCOMPLETE | 0 |
| **Total** | **14** |

These are runtime inspection counts and are not formal accuracy metrics.

### Validated behavior

The new bottle type was validated for:

- bottle detection
- bottle segmentation and mask alignment
- bottle tracking
- orientation
- H centricity stabilization
- V centricity for the new bottle geometry
- defect confirmation
- complete-bottle saving
- defect annotation preservation
- 100 ml capacity reporting

The new bottle type required a separate V-centricity tolerance because its normal label position produced a larger normalized vertical offset than the original bottle type.

The final saved annotations were visually reviewed for representative GOOD and DEFECTIVE bottles.

### Capacity

The new bottle type is a 100 ml product. Capacity OCR was observed to produce ambiguous readings on some frames, so the configured expected capacity for this bottle type is used to prevent transient OCR misclassification.


