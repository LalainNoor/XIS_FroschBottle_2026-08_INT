# Design Decisions

## RF-DETR

RF-DETR was selected as the model family for detection and segmentation.

## Separate Detection and Segmentation

Detection provides class association and tracking information. Segmentation provides bottle masks for geometry and defect validation.

## TensorRT

TensorRT provides the optimized deployment path. Engines are initialized once and GPU buffers are reused.

## Per-Class Confidence Thresholds

Different classes have different evaluation behavior, so a single universal threshold was replaced with class-specific thresholds.

## Mask-Based Geometry

Bottle and label geometry is primarily derived from segmentation masks.

## Defect Mask Validation

Damage/bump detections must overlap the bottle mask sufficiently before influencing the final bottle result.

## Finalization Logic

Results are accumulated over frames and finalized at the trigger line or through tracking-loss conditions.

## INCOMPLETE

INCOMPLETE is distinct from DEFECTIVE. It represents unresolved required inspection information rather than confirmed physical damage.

## Image Storage

Images are organized by final status with separate original and annotated folders. Capacity remains available in CSV/annotations and is not used as the top-level image folder.
