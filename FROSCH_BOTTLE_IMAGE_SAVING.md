# Frosch Bottle Image Saving

## 1. Purpose

The live inference pipeline saves a finalized bottle as a pair of images:

1. `original.jpg` — clean bottle crop.
2. `annotated.jpg` — the same selected bottle crop with inspection annotations.

The current implementation saves bottles into separate directories according to their final inspection status.

## 2. Output Structure

```text
saved_bottles/
├── good/
│   └── bottle_001/
│       ├── original.jpg
│       └── annotated.jpg
├── defective/
│   └── bottle_002/
│       ├── original.jpg
│       └── annotated.jpg
└── incomplete/
    └── bottle_003/
        ├── original.jpg
        └── annotated.jpg
```

The category is derived from the final status:

```text
GOOD       -> saved_bottles/good/
DEFECTIVE  -> saved_bottles/defective/
INCOMPLETE -> saved_bottles/incomplete/
```

## 3. Exactly-Once Saving

Each track contains:

```python
track["saved"]
```

The save function immediately returns when the track has already been saved.

After both image files are successfully written:

```python
track["saved"] = True
```

The track is then removed from the active tracking list.

This prevents the same active bottle track from generating repeated image pairs across subsequent frames.

## 4. Bottle Selection

The pipeline does not simply save the first frame in which a bottle is detected.

A complete bottle snapshot is required.

The selected detection must:

- have a valid complete bounding box,
- be inside the frame,
- satisfy the minimum bottle-area requirement,
- have a valid segmentation/orientation snapshot when one is available,
- preserve the frame and associated annotation data together.

The current minimum bottle area is:

```text
MIN_BOTTLE_AREA = 20000 px²
```

The best complete detection is updated when a better valid bottle-area snapshot is available.

## Complete Defect Snapshot

For defective bottles, the pipeline prefers a saved frame that contains:

1. the complete bottle,
2. the confirmed defect annotation,
3. the corresponding bottle/label geometry,
4. the associated segmentation/orientation information.

If the exact confirmation frame is partially clipped, a complete defect snapshot or complete bottle snapshot is preferred for the final saved annotation.

This prevents saved defective bottles from being unnecessarily cropped while preserving the visual defect annotation.

## 5. Finalization Conditions

A bottle can be finalized in two ways.

### Trigger-line crossing

The current trigger line is:

```text
40% of the frame width
```

with:

```text
TRIGGER_LINE_TOLERANCE = 20 px
```

When the tracked bottle center crosses this line, the bottle is finalized immediately.

### Missing-frame fallback

If the bottle is no longer detected for:

```text
MAX_MISSING_FRAMES = 20
```

frames, the bottle is finalized through the missing-frame fallback.

This allows the image-saving logic to continue working if the trigger-line event is not reached.

## 6. Crop Padding

The selected bottle crop is expanded by:

```text
SAVE_PADDING = 20 px
```

where image boundaries allow it.

Therefore, the saved crop contains a small amount of context around the bottle.

No missing image content is fabricated; the crop is clipped to the actual frame boundaries.

## 7. Original Image

`original.jpg` is generated from the selected crop before annotation.

It does not intentionally add:

- bottle bounding boxes,
- bottle ID text,
- status text,
- label graphics,
- centricity lines,
- defect labels,
- orientation axes.

It is intended to be the clean sample corresponding to the annotated result.

## 8. Annotated Image

`annotated.jpg` is generated from the same selected crop.

Depending on the available data, the annotation contains:

- Bottle ID
- final status
- capacity
- orientation
- H Center
- V Center
- defect list
- bottle bounding box
- bottle segmentation contour
- orientation axes
- label bounding box
- label center marker
- H/V centricity guide lines
- damage boxes
- bump boxes

The segmentation contour is transformed from full-frame coordinates into crop-local coordinates before drawing.

## 9. Defective Samples

A bottle is classified as DEFECTIVE when at least one required inspection check fails or a confirmed defect exists.

Examples:

```text
Orientation: FAIL
H Center: PASS
V Center: PASS
Defects: None
Status: DEFECTIVE
```

or:

```text
Orientation: PASS
H Center: PASS
V Center: PASS
Defects: bump
Status: DEFECTIVE
```

or:

```text
Defects: bump, damage
Status: DEFECTIVE
```

The image is stored under:

```text
saved_bottles/defective/
```

## 10. Incomplete Samples

An INCOMPLETE bottle is not automatically treated as defective.

The current finalization logic uses INCOMPLETE when required measurements remain unavailable/Pending.

For example:

```text
Orientation: Pending
H Center: Pending
V Center: Pending
Defects: None
Status: INCOMPLETE
```

The image is stored under:

```text
saved_bottles/incomplete/
```

This distinction is important because missing measurements and failed measurements represent different conditions.

## 11. Defect Frame Preference

When a confirmed defect exists, the saving logic prefers a frame that actually contains the confirmed defect information.

If a defect frame is unavailable, the implementation falls back to the best complete bottle snapshot and/or preserved defect boxes where possible.

For GOOD bottles, the best complete snapshot is used.

## 12. One Bottle = Two Images

For every successfully saved bottle:

```text
1 bottle -> 2 image files
```

For example:

```text
bottle_024/
├── original.jpg
└── annotated.jpg
```

This makes the saved dataset convenient for later visual review, debugging, and sample collection.

## 13. Runtime Verification

Recorded inference runs show the expected output pattern.

For example, a GOOD bottle was saved as:

```text
saved_bottles/good/bottle_031/original.jpg
saved_bottles/good/bottle_031/annotated.jpg
```

A DEFECTIVE bottle was saved as:

```text
saved_bottles/defective/bottle_052/original.jpg
saved_bottles/defective/bottle_052/annotated.jpg
```

An INCOMPLETE bottle was also observed under:

```text
saved_bottles/incomplete/
```

This confirms that the three-way status directory structure is used by the current implementation.

## 14. Current Saving Parameters

| Parameter | Value |
|---|---:|
| Save directory | `./saved_bottles` |
| Save padding | `20 px` |
| Minimum bottle area | `20000 px²` |
| Missing-frame finalization | `20 frames` |
| Trigger line | `40%` frame width |
| Trigger tolerance | `20 px` |

## 15. Verification Checklist

- [x] Save complete bottle crop
- [x] Save clean/original image
- [x] Save annotated image
- [x] Add crop padding
- [x] Separate GOOD images
- [x] Separate DEFECTIVE images
- [x] Separate INCOMPLETE images
- [x] Save each active track only once
- [x] Preserve selected frame with its annotation data
- [x] Preserve segmentation contour for annotated output
- [x] Prefer a confirmed-defect frame for defective samples
- [x] Keep missing measurements distinct from FAIL measurements

