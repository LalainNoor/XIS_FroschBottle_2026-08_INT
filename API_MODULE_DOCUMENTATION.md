# API / Module Documentation

The Frosch project is a Python application rather than a REST service. The handbook API requirement is therefore addressed through the command-line interface and module responsibilities.

## `live_inference.py`

Primary runtime entry point.

### CLI

```bash
python live_inference.py --input folder --frame-dir /path/to/frames --expected-capacity 100
python live_inference.py --input folder --frame-dir /path/to/frames --expected-capacity 300
python live_inference.py --input folder --frame-dir /path/to/frames --expected-capacity 500
```

### Input

- camera or frame-folder source,
- expected capacity when using recorded capacity-specific runs.

### Output

- live annotated view,
- finalized status,
- CSV record,
- original bottle crop,
- annotated bottle crop.

## TensorRT Runtime

The runtime loads:

```text
output/rfdetr-medium.trt
output/rfdetr-seg-medium.trt
```

It initializes the engines once and reuses GPU buffers.

## `export_frosch_tensorrt.py`

Exports detection and segmentation models to TensorRT with FP16 enabled.

Outputs:

```text
output/rfdetr-medium.trt
output/rfdetr-seg-medium.trt
```

## `count_bottles.py`

Validation utility used to count/inspect saved bottle results. It is not the primary inspection decision engine.

## CSV Interface

The result record contains:

```text
Bottle
Capacity
Orientation
H_Center
V_Center
Defects
Timestamp
```

## Image-Saving Interface

Each finalized bottle saves:

```text
original.jpg
annotated.jpg
```

under its final status.

## Example GOOD Output

```text
Bottle #25 FINAL RESULT
Orientation : 5.126 deg (PASS)
H Center    : 0.006 (PASS)
V Center    : 0.006 (PASS)
Defects     : None
Status      : GOOD
```

## Example INCOMPLETE Output

```text
Orientation : Pending
H Center    : Pending
V Center    : Pending
Status      : INCOMPLETE
```

The distinction is intentional: unresolved inspection is not equivalent to a confirmed physical defect.
