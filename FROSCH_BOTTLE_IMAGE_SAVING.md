# Frosch Bottle Image Saving

## 1. Purpose

The final live inference pipeline saves each finalized bottle as a pair of images:

1. `original.jpg` — clean selected bottle crop.
2. `annotated.jpg` — the same selected bottle crop with inspection annotations.

Bottles are organized first by capacity and then by final inspection status.

## 2. Output Structure

```text
saved_bottles/
├── 100ml/
│   ├── good/
│   │   └── bottle_001/
│   │       ├── original.jpg
│   │       └── annotated.jpg
│   ├── defective/
│   └── incomplete/
├── 300ml/
│   ├── good/
│   ├── defective/
│   └── incomplete/
└── 500ml/
    ├── good/
    ├── defective/
    └── incomplete/
```

The status category is derived from the final result:

```text
GOOD       -> <capacity>/good/
DEFECTIVE  -> <capacity>/defective/
INCOMPLETE -> <capacity>/incomplete/
```

## 3. Exactly-Once Saving

Each track maintains a saved state.

After the bottle's image pair is successfully written, the track is marked as saved and is not written again.

This prevents repeated image pairs for the same tracked bottle.

## 4. Bottle Selection

The pipeline does not simply save the first frame in which a bottle is detected.

It prefers a complete and valid bottle snapshot that:

- has a valid bottle bounding box,
- is sufficiently inside the frame,
- satisfies the minimum bottle-area requirement,
- preserves the corresponding annotations,
- retains valid segmentation/orientation information when available.

The current minimum bottle area is:

```text
MIN_BOTTLE_AREA = 20000 px²
```

A complete-frame snapshot is retained separately from defect snapshots so that the final saved crop can remain visually useful.

## 5. Finalization Conditions

A bottle can be finalized in two ways.

### Trigger-line crossing

```text
TRIGGER_LINE_X_RATIO = 0.40
TRIGGER_LINE_TOLERANCE = 20 px
```

When the tracked bottle center crosses the trigger line, its accumulated result is finalized.

### Missing-frame fallback

If the bottle is not detected for:

```text
MAX_MISSING_FRAMES = 20
```

frames, the track is finalized through the missing-frame fallback.

## 6. Defect Snapshot Preference

For a DEFECTIVE bottle, the saving logic prefers a frame that contains:

1. the complete bottle,
2. the confirmed defect information,
3. the corresponding label/bottle geometry,
4. the segmentation/orientation data.

If the exact defect-confirmation frame is partially clipped, the implementation falls back to a complete saved bottle snapshot and/or preserved defect boxes.

This keeps the final bottle crop complete while preserving defect visualization where possible.

## 7. Crop Padding

The selected crop is expanded by:

```text
SAVE_PADDING = 20 px
```

where the frame boundaries allow it.

The crop is clipped to the actual frame. No image content is fabricated.

## 8. Original Image

`original.jpg` is generated before annotations are drawn.

It is intended to remain a clean sample of the selected bottle crop.

## 9. Annotated Image

`annotated.jpg` is generated from the same selected crop.

Depending on the available information, the annotation can contain:

- bottle bounding box,
- bottle ID,
- final status,
- capacity,
- orientation,
- H Center,
- V Center,
- defect list,
- bottle segmentation contour,
- orientation axes,
- label bounding box,
- label center marker,
- H/V centricity guide lines,
- damage boxes,
- bump boxes.

The bottle segmentation contour and geometry are converted from full-frame coordinates into crop-local coordinates before drawing.

## 10. Defective Samples

A bottle is stored under `defective/` when:

- orientation fails,
- H centricity fails,
- V centricity fails,
- or a confirmed defect exists.

Examples:

```text
Orientation: PASS
H Center: PASS
V Center: PASS
Defects: bump
Status: DEFECTIVE
```

or:

```text
Orientation: PASS
H Center: FAIL
V Center: PASS
Defects: None
Status: DEFECTIVE
```

## 11. Incomplete Samples

An INCOMPLETE bottle is not automatically defective.

It is used when a required measurement remains unavailable/Pending at finalization.

Example:

```text
Orientation: Pending
H Center: Pending
V Center: Pending
Defects: None
Status: INCOMPLETE
```

The sample is stored under the capacity-specific `incomplete/` directory.

## 12. Capacity-Specific Saving

The current inference script creates:

```text
saved_bottles/<expected-capacity>ml/
```

using the `--expected-capacity` argument.

Supported values:

```text
100
300
500
```

For example:

```text
saved_bottles/100ml/good/
saved_bottles/300ml/defective/
saved_bottles/500ml/incomplete/
```

## 13. One Bottle = Two Images

For every successfully saved bottle:

```text
1 bottle -> 2 image files
```

Example:

```text
bottle_018/
├── original.jpg
└── annotated.jpg
```

## 14. Runtime Verification

The final recorded validation logs show the expected saving pattern for all three bottle types, including:

```text
saved_bottles/500ml/good/
saved_bottles/500ml/defective/
saved_bottles/500ml/incomplete/
```

```text
saved_bottles/300ml/good/
saved_bottles/300ml/defective/
```

```text
saved_bottles/100ml/good/
saved_bottles/100ml/defective/
```

## 15. Current Saving Parameters

| Parameter | Value |
|---|---:|
| Save root | `./saved_bottles` |
| Save padding | `20 px` |
| Minimum bottle area | `20000 px²` |
| Missing-frame finalization | `20 frames` |
| Trigger line | `40%` frame width |
| Trigger tolerance | `20 px` |

## 16. Verification Checklist

- [x] Save complete bottle crop
- [x] Save clean/original image
- [x] Save annotated image
- [x] Add crop padding
- [x] Separate GOOD images
- [x] Separate DEFECTIVE images
- [x] Separate INCOMPLETE images
- [x] Organize output by bottle capacity
- [x] Save each track only once
- [x] Preserve selected frame with annotation data
- [x] Preserve segmentation contour for annotated output
- [x] Prefer a complete defect snapshot for defective samples
- [x] Keep missing measurements distinct from FAIL measurements
