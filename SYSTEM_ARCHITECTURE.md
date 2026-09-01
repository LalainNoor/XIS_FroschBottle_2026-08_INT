# System / Pipeline Architecture

## Input

The pipeline accepts:

1. GenICam/Vimba X camera input.
2. Vimba X camera simulator input.
3. Recorded image-folder input.

## Detection

RF-DETR Medium detection runs through TensorRT. The detector class map is:

1. `Frosch-bottle-UTNY-aUbJ-XBXs`
2. `bottle`
3. `bump`
4. `capacity`
5. `damage`
6. `label`
7. `scratch`

The inspection logic primarily consumes bottle, capacity, label, damage and bump.

## Segmentation

RF-DETR Medium segmentation provides bottle masks. Masks are used for orientation, centricity, defect validation and annotated visualization.

## OCR

EasyOCR processes the detected capacity region. Multiple preprocessing variants are used. If OCR does not return a valid supported capacity, expected-capacity fallback is available.

## Tracking

Bottle detections are associated across frames. Track state stores capacity, geometry, defect evidence and selected image data until finalization.

## Geometry

### Orientation

Orientation is calculated from bottle-mask pixels. Maximum accepted angle is 45 degrees.

### Horizontal centricity

```text
H = (label_center_x - bottle_center_x) / bottle_width
```

Threshold: `0.15`.

### Vertical centricity

```text
V = (label_center_y - bottle_center_y) / bottle_height
```

Expected values:

| Capacity | Expected V |
|---|---:|
| 100 ml | 0.12 |
| 300 ml | 0.07 |
| 500 ml | 0.01 |

Allowed deviation: `0.05`.

## Defect Validation

Damage and bump candidates are checked against the bottle mask. The documented defect overlap threshold is `0.30`.

## Finalization

A track is finalized when it crosses the trigger line or when missing-frame conditions force fallback finalization.

- Trigger line: 40% of frame width
- Trigger tolerance: 20 px
- Maximum missing frames: 20

## Final Status

`GOOD` requires passing orientation and H/V centricity with no confirmed defect.

`DEFECTIVE` is produced for a failed required geometry condition or a confirmed damage/bump defect.

`INCOMPLETE` is produced when required inspection information remains unresolved at finalization.

## Output

Each finalized bottle produces:

- a CSV record,
- an original crop,
- an annotated crop.
