# Measurement Methodology / Accuracy Report

## Status

Pixel-to-millimetre measurement was **not implemented**.

The Frosch runtime performs image-space inspection rather than calibrated real-world dimension measurement.

## Handbook Measurement Workflow

The handbook specifies:

- calibrated images,
- `cv2.undistort()`,
- a calibrated reference object,
- pixels-per-mm derivation,
- mask/bounding-box dimensions,
- width and height in millimetres,
- physical validation on 10+ instances,
- MAE and MPE.

These stages were not completed for this project.

## Implemented Image-Space Measurements

The runtime calculates normalized geometry.

### Horizontal centricity

```text
H = (label_center_x - bottle_center_x) / bottle_width
```

### Vertical centricity

```text
V = (label_center_y - bottle_center_y) / bottle_height
```

These values are not millimetres.

## Reference Object

No calibrated physical reference object was used.

## Accuracy

No physical ruler/calliper measurement set was collected for metric measurement.

Therefore:

- MAE: N/A
- MPE: N/A
- width_mm: N/A
- height_mm: N/A

The separate 52-bottle inspection validation is a status-classification validation and is not a millimetre-measurement accuracy study.

## Future Work

Complete camera calibration first, then derive and validate pixels-per-mm using a known reference object before reporting metric dimensions.
