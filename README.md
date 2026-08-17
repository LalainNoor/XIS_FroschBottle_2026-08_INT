# Frosch Bottle Inspection Pipeline

Computer vision pipeline for automated Frosch bottle inspection using RF-DETR detection and segmentation, mask-based orientation analysis, label centricity checks, defect detection, OCR-based capacity recognition, bottle tracking, and automated result saving.

---

## Project Overview

The pipeline processes bottle images from either:

1. A live GenICam-compatible camera / camera simulator
2. A recorded folder of image frames

Both modes use the same inference and inspection pipeline.

The system performs:

- Bottle detection
- Bottle segmentation
- Bottle tracking
- Bottle mask alignment
- Mask-based orientation estimation
- Label detection and association
- Horizontal and vertical label centricity checks
- Damage detection
- Bump detection
- Multi-frame defect confirmation
- Capacity OCR
- Bottle finalization
- GOOD / DEFECTIVE / INCOMPLETE classification
- Annotated bottle image saving
- Original bottle image saving
- CSV result logging
- Live visualization

---

## Repository Structure

    .
    ├── live_inference.py
    ├── README.md
    ├── RESULTS.md
    ├── FROSCH_TRAINING_REPORT.md
    ├── FROSCH_BOTTLE_IMAGE_SAVING.md
    ├── requirements.txt
    ├── .gitignore
    ├── runs/
    │   ├── frosch_medium/
    │   └── frosch_seg_medium/
    └── saved_bottles/
        ├── good/
        ├── defective/
        └── incomplete/

Generated result CSV files and other runtime outputs may also be created during inference.

---

## Models

The pipeline uses two RF-DETR models.

### Bottle Detection

    runs/frosch_medium/checkpoint_best_regular.pth

### Bottle Segmentation

    runs/frosch_seg_medium/checkpoint_best_total.pth

The segmentation model provides the bottle mask used for:

- Bottle-mask alignment
- Bottle orientation
- Defect-overlap validation
- Saved annotation visualization

---

## Requirements

Install the dependencies using:

    pip install -r requirements.txt

For GPU-based execution, make sure the appropriate CUDA-enabled PyTorch environment is available.

---

## Running Inference

The project uses a single `live_inference.py` script for both live camera inference and recorded-frame testing.

No separate `live_inference_frames.py` script is required.

### Live Camera / Simulator

Run the default live inference mode:

    python live_inference.py

The default input mode is:

    camera

The input mode can also be specified explicitly:

    python live_inference.py --input camera

The camera source is configured through the GenICam/Harvester setup in `live_inference.py`.

The live window can be closed by pressing:

    q

---

### Recorded Frame Testing

Recorded frames can be processed directly using the same script:

    python live_inference.py \
        --input folder \
        --frame-dir /path/to/frame/folder \
        --expected-capacity 100

For the currently validated 100 ml bottle type:

    python live_inference.py \
        --input folder \
        --frame-dir /home/xisai/Downloads/Capture_2026-07-15_07h53m14s \
        --expected-capacity 100

The folder mode is intended for:

- Repeatable testing
- New bottle-type validation
- Regression testing
- Testing previously captured image sequences

The source folder is scanned for supported image formats including:

    .jpg
    .jpeg
    .png
    .bmp
    .tif
    .tiff

Frames are processed in natural filename order.

---

## Runtime Arguments

| Argument | Options / Example | Description |
|---|---|---|
| `--input` | `camera` / `folder` | Selects the inference input source. Default: `camera` |
| `--frame-dir` | `/path/to/frames` | Folder containing recorded frames when `--input folder` is selected |
| `--expected-capacity` | `100`, `300`, `500` | Expected bottle capacity for the bottle type being tested |

Example:

    python live_inference.py \
        --input folder \
        --frame-dir /home/xisai/Downloads/Capture_2026-07-15_07h53m14s \
        --expected-capacity 100

The same inspection pipeline is used regardless of the selected input mode.

---

## Input Processing Pipeline

    Input
      │
      ├── Camera
      │     └── GenICam / Harvester
      │
      └── Folder
            └── Recorded image frames
                  │
                  ▼
           RF-DETR Bottle Detection
                  │
                  ▼
           RF-DETR Bottle Segmentation
                  │
                  ▼
            Bottle Mask Alignment
                  │
                  ▼
              Bottle Tracking
                  │
                  ├── Orientation
                  ├── Label Association
                  ├── H Centricity
                  ├── V Centricity
                  ├── Capacity OCR
                  └── Defect Detection
                           │
                           ▼
                    Multi-frame Confirmation
                           │
                           ▼
                      Final Bottle State
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
               GOOD     DEFECTIVE  INCOMPLETE
                 │         │         │
                 └─────────┼─────────┘
                           ▼
                      Save Results

---

## Bottle Segmentation and Mask Alignment

The segmentation model provides the bottle mask.

The mask is:

1. Resized to the current frame resolution when necessary.
2. Constrained to the detected bottle bounding box.
3. Used as the geometry source for orientation.
4. Used to validate whether detected defects overlap the actual bottle region.

This prevents segmentation padding or coordinate differences from producing mask regions outside the detected bottle.

---

## Orientation Detection

Bottle orientation is calculated from the segmentation mask pixels.

The system derives the bottle's principal axis using the mask geometry and determines the orientation angle.

The orientation status is:

    PASS
    FAIL

The configured orientation limit is:

    45 degrees

The saved annotations can display the bottle's orientation axes.

---

## Label Detection and Centricity

The pipeline associates the most spatially consistent label box with the detected bottle.

The label is clipped to the bottle boundary before centricity is evaluated.

Two centricity measurements are calculated.

### H Center

Horizontal label-to-bottle center offset.

Default threshold:

    0.15

### V Center

Vertical label-to-bottle center offset.

Default threshold:

    0.15

For the currently validated narrow/tall new bottle type, a larger V tolerance is used:

    0.20

The new bottle type also uses an aspect-ratio rule to distinguish it from the original bottle geometry.

---

## Centricity Stabilization

Centricity is accumulated over the bottle's lifetime rather than relying on a single frame.

The pipeline:

- Keeps a short history of accepted normalized label offsets.
- Rejects sudden spatial jumps.
- Uses accumulated H/V results for final bottle classification.
- Prevents temporary edge-of-frame failures from unnecessarily determining the final result.

This is particularly important when the bottle is entering or leaving the camera view.

---

## Defect Detection

The pipeline detects:

- Damage
- Bump

A detected defect is only accepted when it sufficiently overlaps the actual bottle segmentation mask.

Configured defect overlap threshold:

    0.30

A defect must also be detected in consecutive frames before it is accepted.

Current confirmation requirement:

    2 consecutive frames

This reduces isolated one-frame false positives.

---

## Capacity Detection

Capacity is initially estimated using OCR.

The OCR process uses multiple preprocessing variants, including:

- Original grayscale
- CLAHE-enhanced grayscale
- Otsu thresholding
- Adaptive thresholding

The accepted capacities are:

    100 ml
    300 ml
    500 ml

For a known bottle type, the expected capacity can be provided explicitly:

    --expected-capacity 100

For example:

    python live_inference.py \
        --input folder \
        --frame-dir /path/to/frames \
        --expected-capacity 100

This allows the OCR process to be used for recognition while preventing transient OCR errors from changing the final product capacity.

When testing another bottle type, set `--expected-capacity` to the correct product capacity.

---

## Bottle Tracking and Finalization

Each detected bottle receives a track ID.

Bottle matching uses:

- Bounding-box IoU
- Conservative center-distance fallback

The fallback is intended for normal bottle motion between frames.

A bottle is finalized after it has disappeared from the stream for the configured number of missing frames or when the processing sequence ends.

The current missing-frame threshold is:

    20 frames

A bottle must have a valid complete saved frame before it can be finalized and saved.

---

## Final Classification

The system produces one of three final states.

### GOOD

A bottle is GOOD when:

- Orientation passes
- H centricity passes
- V centricity passes
- No confirmed defects are present
- Required measurements are available

### DEFECTIVE

A bottle is DEFECTIVE when:

- A confirmed damage defect is present
- A confirmed bump defect is present
- Orientation fails
- H centricity fails
- V centricity fails

### INCOMPLETE

A bottle is INCOMPLETE when required measurements remain unavailable.

Missing measurements are not automatically converted into DEFECTIVE.

---

## Saved Bottle Images

Finalized bottles are saved under:

    saved_bottles/

with categories:

    saved_bottles/
    ├── good/
    ├── defective/
    └── incomplete/

Each saved bottle can contain:

    original.jpg
    annotated.jpg

### Annotated Images

The annotated image may show:

- Bottle bounding box
- Bottle segmentation mask
- Orientation axes
- Label bounding box
- H/V centricity lines
- Damage boxes
- Bump boxes
- Final status
- Capacity
- Orientation result
- H Center result
- V Center result
- Defect list

For defective bottles, the pipeline prefers a complete bottle frame that also contains the confirmed defect annotation.

If the exact defect-confirmation frame is partially clipped, the pipeline uses a complete defect snapshot when available and can reconstruct the confirmed defect position relative to the selected bottle frame.

This prevents defective bottles from being unnecessarily cropped while preserving the visual defect annotation.

---

## CSV Logging

Each run creates a timestamped CSV file:

    results_YYYYMMDD_HHMMSS.csv

The CSV contains:

    Bottle
    Capacity
    Orientation
    H_Center
    V_Center
    Defects
    Timestamp

Example:

    Bottle,Capacity,Orientation,H_Center,V_Center,Defects,Timestamp
    1,100,PASS,PASS,PASS,None,12:34:56

Runtime result CSV files are intended for validation and inspection records.

---

## Current New Bottle Type Validation

A recorded-frame validation was performed on the current new bottle type using:

    Frame source:
    /home/xisai/Downloads/Capture_2026-07-15_07h53m14s

    Frames:
    1292

    Bottle type:
    100 ml

Final runtime result:

| Category | Count |
|---|---:|
| GOOD | 10 |
| DEFECTIVE | 4 |
| INCOMPLETE | 0 |
| **Total** | **14** |

The validation confirmed the following behaviors for the new bottle type:

- Bottle detection
- Bottle segmentation
- Bottle-mask alignment
- Bottle tracking
- Orientation
- H centricity
- V centricity using the new-bottle tolerance
- Defect confirmation
- Complete bottle saving
- Defect-preserving saved annotations
- 100 ml capacity reporting
- GOOD / DEFECTIVE / INCOMPLETE finalization

These counts are runtime validation observations and are not formal model accuracy metrics.

---

## Runtime Outputs

Runtime-generated files may include:

    results_*.csv
    saved_bottles/

Temporary debug and test outputs should not be treated as model evaluation metrics.

---

## Validation Notes

The recorded-frame mode is recommended when validating a new bottle type because it provides a repeatable input sequence.

A typical workflow is:

    1. Capture a representative frame sequence.
    2. Run the sequence using --input folder.
    3. Set --expected-capacity for the bottle type.
    4. Review GOOD results.
    5. Review DEFECTIVE results.
    6. Check segmentation-mask alignment.
    7. Check H/V centricity.
    8. Check orientation.
    9. Check capacity.
    10. Review saved annotated images.
    11. Record final validation results.

When a new bottle type is introduced, its normal geometry and label position should be validated before changing the configured centricity thresholds.
---

## Configuration Summary

| Parameter | Current value | Purpose |
|---|---:|---|
| Detection threshold | `0.40` | RF-DETR detection confidence |
| Segmentation threshold | `0.30` | RF-DETR segmentation confidence |
| Tracking IoU | `0.40` | Primary bottle matching |
| Missing frames | `20` | Track finalization fallback |
| Trigger line | `40%` of frame width | Bottle finalization |
| Trigger tolerance | `20 px` | Minimum movement across trigger |
| H centricity | `0.15` | Horizontal centricity threshold |
| V centricity | `0.15` | Default vertical centricity threshold |
| New-bottle V centricity | `0.20` | Vertical threshold for current new bottle type |
| New-bottle aspect ratio | `0.45` | Identifies narrow/tall bottle geometry |
| Centricity jump tolerance | `0.08` | Reject sudden normalized-offset jumps |
| Centricity minimum history | `3` | Warm-up before spatial jump filtering |
| Centricity history window | `5` | Recent centricity samples retained |
| Orientation limit | `45°` | Orientation PASS limit |
| Defect mask overlap | `0.30` | Required defect/mask overlap |
| Defect confirmation | `2 frames` | Consecutive-frame confirmation |
| Minimum bottle area | `20000 px²` | Prevent tiny/partial saves |
| Save padding | `20 px` | Crop padding |

---

## Important Current Limitations

The following points describe the current implementation:

1. The script does not currently call `cv2.undistort()`. Camera calibration/undistortion therefore needs to be handled upstream if required by the deployment setup.
2. The script does not explicitly set RF-DETR inference to FP16. Runtime logs have indicated that the loaded RF-DETR models are not optimized for inference, so latency can be higher than an optimized deployment.
3. The CTI path is machine-specific and must be changed on another system.
4. The current code assumes the expected detector class names are present.
5. The final GOOD / DEFECTIVE / INCOMPLETE counts are runtime inspection counts, not model accuracy metrics. A proper accuracy evaluation requires a ground-truth test set and sample-level comparison.
6. Capacity OCR is used for recognition, while the final expected capacity can be supplied through `--expected-capacity` for a known bottle type.
7. The segmentation model is used to obtain the bottle mask and orientation. Detector boxes remain the main source for bottle tracking and class-specific defect, label, and capacity candidates.

---

## Training and Evaluation

Model-training details are documented separately in:

    FROSCH_TRAINING_REPORT.md

The project documentation should not claim formal precision, recall, F1, mAP, or similar evaluation metrics unless those values have been measured and verified.

Runtime bottle counts from recorded sequences are validation observations, not formal model accuracy measurements.

---

## Bottle Image Saving Documentation

Detailed bottle image-saving behavior is documented in:

    FROSCH_BOTTLE_IMAGE_SAVING.md

This document describes:

- Complete-frame selection
- Defect-frame selection
- Cropping
- Annotation preservation
- Saved-image organization

---

## Git Notes

The canonical inference script is:

    live_inference.py

No separate inference script is required for folder testing.

Use:

    python live_inference.py --input camera

for live camera inference and:

    python live_inference.py \
        --input folder \
        --frame-dir /path/to/frames \
        --expected-capacity 100

for recorded-frame validation.
