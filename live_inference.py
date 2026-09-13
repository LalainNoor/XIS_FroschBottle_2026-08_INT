import torch
import supervision as sv
import cv2
import numpy as np
import re
import os
import argparse
import time
import subprocess
import psutil

try:
    import pynvml
except ImportError:
    pynvml = None

import easyocr
import tensorrt as trt
from torchvision.transforms import functional as TVF
from harvesters.core import Harvester

import csv
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# -----------------------------
# Runtime arguments / configuration
# -----------------------------
def parse_runtime_args():
    parser = argparse.ArgumentParser(
        description="Frosch bottle live/folder inference pipeline"
    )
    parser.add_argument(
        "--input",
        "--input-mode",
        dest="input_mode",
        choices=("camera", "folder"),
        default="camera",
        help="Input source: camera (default) or folder",
    )
    parser.add_argument(
        "--frame-dir",
        default="/home/xisai/Downloads/Capture_2026-07-15_07h53m14s",
        help="Folder containing recorded frames when --input folder is used",
    )
    parser.add_argument(
        "--expected-capacity",
        type=int,
        choices=(100, 300, 500),
        default=100,
        help="Expected bottle capacity for the currently tested bottle type",
    )
    return parser.parse_args()

ARGS = parse_runtime_args()

LOG_FILE = f"results_{ARGS.expected_capacity}ml.csv"
with open(LOG_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Bottle",
        "Capacity",
        "Orientation",
        "H_Center",
        "V_Center",
        "Defects",
        "Timestamp",
    ])

# -----------------------------
# Configuration
# -----------------------------
INPUT_MODE = ARGS.input_mode
FRAME_DIR = ARGS.frame_dir

CHECKPOINT = "runs/frosch_medium/checkpoint_best_regular.pth"
SEG_CHECKPOINT = "runs/frosch_seg_medium/checkpoint_best_total.pth"
SEG_THRESHOLD = 0.30

# Per-class confidence thresholds (overrides universal THRESHOLD / SEG_THRESHOLD).
# Lower  → more recall, more false positives.
# Higher → fewer false positives, more misses.
PER_CLASS_CONF = {
    "bottle":   0.70,   # was 0.40 - filters person FP (conf~0.43)
    "label":    0.35,
    "capacity": 0.35,
    "bump":     0.50,   # was 0.25, and than 0.35 - raise to cut 500ml #17 FP
    "damage":   0.30,   # was 0.45 - lower to catch 100ml #17 missed
    "scratch":  0.30,   # undertrained – lower to catch more
}

BOTTLE_CLASS_NAME = "bottle"
ORIENTATION_MAX_ANGLE_DEG = 45.0
CTI_PATH = "/home/xisai/Downloads/VimbaX_2026-2/cti/VimbaCameraSimulatorTL.cti"
THRESHOLD = 0.4
IOU_THRESHOLD = 0.4
MAX_MISSING_FRAMES = 20

# Trigger line for bottle finalization
TRIGGER_LINE_X_RATIO = 0.40
TRIGGER_LINE_TOLERANCE = 20

# A centricity measurement is considered reliable only when
# the bottle is sufficiently away from the left/right frame edges.
COMPLETE_X_MARGIN = 20

H_CENTRICITY_THRESH = 0.15

# Expected V centricity for each bottle type.
BOTTLE_TYPE1_EXPECTED_V = 0.01   # 500 ml
BOTTLE_TYPE2_EXPECTED_V = 0.12   # 100 ml
BOTTLE_TYPE3_EXPECTED_V = 0.07   # 300 ml

V_DEVIATION_THRESH = 0.05

# Expected capacity for the bottle type currently being tested.
EXPECTED_CAPACITY = ARGS.expected_capacity

# Preserve the existing camera/folder workflow while selecting
# the correct V reference for each bottle capacity.
if EXPECTED_CAPACITY == 500:
    BOTTLE_TYPE = 1
    EXPECTED_V = BOTTLE_TYPE1_EXPECTED_V

elif EXPECTED_CAPACITY == 100:
    BOTTLE_TYPE = 2
    EXPECTED_V = BOTTLE_TYPE2_EXPECTED_V

elif EXPECTED_CAPACITY == 300:
    BOTTLE_TYPE = 3
    EXPECTED_V = BOTTLE_TYPE3_EXPECTED_V

else:
    raise ValueError(
        f"Unsupported bottle capacity: {EXPECTED_CAPACITY}"
    )

LABEL_CONTAINMENT_TOLERANCE = 10

# Centricity stabilization:
# relative label position is expected to remain spatially consistent
# while the bottle moves through the camera frame.
CENTRICITY_SPATIAL_TOLERANCE = 0.08
CENTRICITY_MIN_HISTORY = 3
CENTRICITY_HISTORY_WINDOW = 5

# Minimum number of reliable complete-frame observations
# required before H/V centricity can determine the result.
MIN_RELIABLE_CENTRICITY_MEASUREMENTS = 3
# -----------------------------
# Classification Rules
# -----------------------------
# GOOD      : orientation PASS, h_center PASS, v_center PASS, no defects, label present
# DEFECTIVE : any check FAIL OR damage/bump detected inside bottle mask
# INCOMPLETE: any required measurement still Pending (label never detected, mask unavailable)
DEFECT_OVERLAP_THRESH = 0.3   # defect box must overlap bottle mask by this fraction to count
# A defect must be detected in consecutive frames before it is accepted.
# This reduces one-frame false positives from the defect detector.
DEFECT_CONFIRMATION_FRAMES = 1

# Allow a confirmed candidate to survive a short detector miss.
# This handles intermittent bump/damage detections between frames.
DEFECT_MAX_MISSING_FRAMES = 1
# FIX 1: minimum bottle area to avoid saving spurious/partial detections
MIN_BOTTLE_AREA = 20000

# A track must be genuinely re-detected/matched across at least this many
# separate frames before it is allowed to be saved/counted. Filters out
# single-frame false-positive detections (empty conveyor, noise) that would
# otherwise sit unmatched for MAX_MISSING_FRAMES and get flushed as a bogus
# INCOMPLETE entry. Real bottles are matched dozens of times and clear this
# trivially.
MIN_MATCHED_FRAMES_TO_SAVE = 3

# Mask-based duplicate track threshold:
# if an unmatched bottle detection's mask overlaps an existing track's mask
# by more than this fraction, it is the same physical bottle — skip new track.
MASK_IOU_DUPLICATE_THRESH = 0.30

SAVE_ROOT_DIR = os.path.join(os.getcwd(), "saved_bottles")
SAVE_DIR = SAVE_ROOT_DIR

SAVE_PADDING = 20

os.makedirs(SAVE_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=True)

# V23: run the expensive OCR call in a single background worker. Only one
# OCR job is allowed at a time so EasyOCR/GPU access remains serialized.
_ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='frosch-ocr')
_ocr_future = None
_ocr_future_track_id = None
_ocr_future_box = None
# Step 7: native TensorRT inference.
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class NativeTRTEngine:
    def __init__(self, engine_path):
        self.engine_path = engine_path

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"Failed to create TensorRT context: {engine_path}")

        # Use a dedicated non-default CUDA stream for TensorRT execution.
        # This removes TensorRT's default-stream synchronization warning while
        # keeping the existing inference/data flow unchanged.
        self.stream = torch.cuda.Stream(device="cuda")

        self.input_name = None
        self.output_names = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_names.append(name)

        shape = tuple(self.engine.get_tensor_shape(self.input_name))
        if len(shape) != 4 or shape[0] != 1 or shape[1] != 3:
            raise RuntimeError(f"Unexpected input shape: {shape}")

        self.input_h = int(shape[2])
        self.input_w = int(shape[3])

        # PERFORMANCE OPTIMIZATION 1:
        # Reuse TensorRT GPU input/output buffers across frames instead of
        # allocating new CUDA tensors on every inference.
        # This does not change model inputs, outputs, thresholds, or decoding.
        self._input_tensor = torch.empty(
            (1, 3, self.input_h, self.input_w),
            dtype=torch.float32,
            device="cuda",
        )
        self._output_tensors = {}
        for name in self.output_names:
            out_shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in out_shape):
                raise RuntimeError(
                    f"Dynamic output shape is unsupported: {name} {out_shape}"
                )

            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if np_dtype == np.float32:
                torch_dtype = torch.float32
            elif np_dtype == np.float16:
                torch_dtype = torch.float16
            elif np_dtype == np.int32:
                torch_dtype = torch.int32
            elif np_dtype == np.int64:
                torch_dtype = torch.int64
            else:
                raise RuntimeError(
                    f"Unsupported TensorRT output dtype: {name} {np_dtype}"
                )

            out = torch.empty(
                out_shape,
                dtype=torch_dtype,
                device="cuda",
            )
            self._output_tensors[name] = out

        self.context.set_tensor_address(
            self.input_name,
            self._input_tensor.data_ptr(),
        )
        for name, out in self._output_tensors.items():
            self.context.set_tensor_address(name, out.data_ptr())

        print(
            f"[OPTIMIZATION] TensorRT loaded: {engine_path} | "
            f"input={self.input_h}x{self.input_w} | outputs={self.output_names}"
        )
        print("[OPTIMIZATION] Persistent TensorRT CUDA buffers: enabled")

    def infer(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_w, self.input_h),
            interpolation=cv2.INTER_LINEAR,
        )

        normalized = resized.astype(np.float32) / 255.0
        normalized -= np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32,
        )
        normalized /= np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32,
        )

        chw = np.ascontiguousarray(
            np.transpose(normalized, (2, 0, 1))
        )

        with torch.cuda.stream(self.stream):
            self._input_tensor.copy_(
                torch.from_numpy(chw).unsqueeze(0),
                non_blocking=True,
            )

            if not self.context.execute_async_v3(
                self.stream.cuda_stream
            ):
                raise RuntimeError(
                    f"TensorRT execution failed: {self.engine_path}"
                )

        self.stream.synchronize()

        return {
            name: tensor.detach().float().cpu().numpy()
            for name, tensor in self._output_tensors.items()
        }


def _load_class_names():
    """Load the exact trained class-name order from the RF-DETR checkpoint."""
    try:
        import torch as _torch
        for ckpt_path in (
            CHECKPOINT,
            SEG_CHECKPOINT,
        ):
            ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
            args = ckpt.get("args") if isinstance(ckpt, dict) else None
            if args is not None:
                names = getattr(args, "classes", None)
                if names:
                    return list(names)

            if isinstance(ckpt, dict):
                for key in ("class_names", "classes", "names"):
                    names = ckpt.get(key)
                    if names:
                        return list(names)
    except Exception as exc:
        print(f"[WARNING] Could not read class names from checkpoint: {exc}")

    CLASS_NAMES = [
        "Frosch-bottle-UTNY-aUbJ-XBXs",
        "bottle",
        "bump",
        "capacity",
        "damage",
        "label",
        "scratch",
    ]

CLASS_NAMES = [
    "Frosch-bottle-UTNY-aUbJ-XBXs",
    "bottle",
    "bump",
    "capacity",
    "damage",
    "label",
    "scratch",
]

print(f"[OPTIMIZATION] TensorRT class map: {CLASS_NAMES}")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def decode_rfdetr_outputs(raw, frame_shape, score_threshold, keep_masks=False):
    if "dets" not in raw or "labels" not in raw:
        raise RuntimeError(f"Expected dets/labels outputs, got {list(raw)}")

    boxes_cwh = raw["dets"][0]
    logits_all = raw["labels"][0]

    logits = logits_all[:, :-1]
    probs = _sigmoid(logits)

    flat = probs.reshape(-1)
    k = min(boxes_cwh.shape[0], flat.size)
    order = np.argsort(-flat, kind="stable")[:k]

    scores = flat[order]
    num_classes = probs.shape[1]
    query_idx = order // num_classes
    class_ids = order % num_classes

    keep = scores > score_threshold
    scores = scores[keep]
    query_idx = query_idx[keep]
    class_ids = class_ids[keep]

    boxes = boxes_cwh[query_idx]
    h, w = frame_shape[:2]

    cx, cy, bw, bh = boxes.T
    xyxy = np.stack(
        [
            (cx - bw / 2) * w,
            (cy - bh / 2) * h,
            (cx + bw / 2) * w,
            (cy + bh / 2) * h,
        ],
        axis=1,
    )

    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)

    detections = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=scores.astype(np.float32),
        class_id=class_ids.astype(int),
    )

    detections.data["class_name"] = np.array(
        [
            CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES)
            else f"class_{int(cid)}"
            for cid in class_ids
        ],
        dtype=object,
    )

    detections.data["_rfdetr_query_idx"] = query_idx.astype(np.int32)

    if keep_masks and "masks" in raw:
        raw_masks = raw["masks"][0]
        selected_masks = raw_masks[query_idx]

        mask_tensor = torch.from_numpy(selected_masks).float().unsqueeze(1)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        selected_masks_full = (mask_tensor.sigmoid() > 0.5).cpu().numpy()
        detections.mask = selected_masks_full

    return detections


def trt_predict(runtime_model, frame, threshold, keep_masks=False):
    raw = runtime_model.infer(frame)
    return decode_rfdetr_outputs(
        raw,
        frame.shape,
        threshold,
        keep_masks=keep_masks,
    )


model = NativeTRTEngine("output/rfdetr-medium.trt")
seg_model = NativeTRTEngine("output/rfdetr-seg-medium.trt")

print("[OPTIMIZATION] Native TensorRT backend: enabled for detection + segmentation.")
print("[OPTIMIZATION] TensorRT engines initialized once: enabled.")

class ResourceMonitor:
    """Low-overhead FPS/CPU/GPU monitor for the live display."""

    def __init__(self, poll_interval=0.5):
        self.poll_interval = poll_interval
        self.last_poll = 0.0
        self.cpu_percent = 0.0
        self.gpu_percent = None
        self.gpu_mem_used = None
        self.gpu_mem_total = None
        self.nvml_handle = None

        psutil.cpu_percent(interval=None)

        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                self.gpu_mem_total = mem.total
            except Exception:
                self.nvml_handle = None

    def update(self):
        now = time.perf_counter()
        if now - self.last_poll < self.poll_interval:
            return

        self.last_poll = now
        self.cpu_percent = psutil.cpu_percent(interval=None)

        if self.nvml_handle is not None:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                self.gpu_percent = util.gpu
                self.gpu_mem_used = mem.used
                self.gpu_mem_total = mem.total
                return
            except Exception:
                pass

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=0.15,
            )
            if result.returncode == 0 and result.stdout.strip():
                gpu, used, total = result.stdout.strip().splitlines()[0].split(", ")
                self.gpu_percent = float(gpu)
                self.gpu_mem_used = float(used) * 1024 * 1024
                self.gpu_mem_total = float(total) * 1024 * 1024
        except Exception:
            pass

    def text(self, fps):
        gpu_text = f"{self.gpu_percent:.0f}%" if self.gpu_percent is not None else "N/A"
        if self.gpu_mem_used is not None and self.gpu_mem_total:
            vram_text = (
                f"{self.gpu_mem_used / (1024**3):.1f}/"
                f"{self.gpu_mem_total / (1024**3):.1f}GB"
            )
        else:
            vram_text = "N/A"
        return f"FPS: {fps:.1f} | CPU: {self.cpu_percent:.0f}% | GPU: {gpu_text} | VRAM: {vram_text}"

    def close(self):
        if self.nvml_handle is not None and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


resource_monitor = ResourceMonitor()
fps = 0.0
active_fps = 0.0  # smoothed FPS for active-bottle frames only

recently_saved_boxes = []   # [(box, ttl_remaining)]
RECENTLY_SAVED_TTL = 40     # frames to block re-detection after save

bottle_count = 0
completed_count = 0
good_count = 0
defective_count = 0
incomplete_count = 0
skipped_frame_count = 0  # frames dropped due to unreadable file / fetch failure (§5.2 recovery)

next_track_id = 0
tracked = []


# -----------------------------
# Helpers
# -----------------------------
def extract_capacity(text):
    digits = re.sub(r"\D", "", text.strip())
    if digits in {"100", "300", "500"}:
        return int(digits)

    return None

def run_ocr(image, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = image.shape[:2]

    pad_x = 24
    pad_y = 24

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    _ocr_pre_start = time.perf_counter()

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    gray = cv2.resize(
        gray,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC,
    )

    clahe = cv2.createCLAHE(
        clipLimit=3.0,
        tileGridSize=(8, 8),
    )
    enhanced = clahe.apply(gray)

    variants = [
        gray,
        enhanced,
        cv2.threshold(
            enhanced,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[1],
        cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        ),
    ]

    ocr_batch_v13["preprocessing"] += time.perf_counter() - _ocr_pre_start

    def _extract_valid_candidates(results):
        valid_candidates = []
        for (_, text_value, conf) in results:
            if conf < 0.15:
                continue
            cap = extract_capacity(text_value)
            if cap in {100, 300, 500}:
                valid_candidates.append((cap, float(conf)))
        return valid_candidates

    def _select_capacity(valid_candidates):
        if not valid_candidates:
            return None
        counts = Counter(cap for cap, _ in valid_candidates)
        return max(
            counts,
            key=lambda cap: (
                counts[cap],
                max(
                    confidence
                    for candidate, confidence in valid_candidates
                    if candidate == cap
                ),
            ),
        )

    _ocr_v1_start = time.perf_counter()
    first_results = reader.readtext(
        variants[0],
        allowlist="0123456789",
        paragraph=False,
    )
    ocr_variant_times["variant1"] += time.perf_counter() - _ocr_v1_start
    ocr_variant_counts["variant1"] += 1

    first_candidates = _extract_valid_candidates(first_results)
    if first_candidates:
        return _select_capacity(first_candidates)

    remaining_variants = variants[1:]

    _prep_batch_start = time.perf_counter()
    _batch_dims = [(int(v.shape[1]), int(v.shape[0])) for v in remaining_variants]
    ocr_batch_v13["batch_prepare"] += time.perf_counter() - _prep_batch_start
    ocr_batch_v13["batch_calls"] += 1
    ocr_batch_v13["images_total"] += len(remaining_variants)
    for _bw, _bh in _batch_dims:
        ocr_batch_v13["width_sum"] += _bw
        ocr_batch_v13["height_sum"] += _bh
        ocr_batch_v13["width_min"] = _bw if ocr_batch_v13["width_min"] is None else min(ocr_batch_v13["width_min"], _bw)
        ocr_batch_v13["width_max"] = _bw if ocr_batch_v13["width_max"] is None else max(ocr_batch_v13["width_max"], _bw)
        ocr_batch_v13["height_min"] = _bh if ocr_batch_v13["height_min"] is None else min(ocr_batch_v13["height_min"], _bh)
        ocr_batch_v13["height_max"] = _bh if ocr_batch_v13["height_max"] is None else max(ocr_batch_v13["height_max"], _bh)

    OCR_BATCH_MAX_DIM_V14 = 512
    _resize_start = time.perf_counter()
    resized_variants = []
    for _img in remaining_variants:
        _h, _w = _img.shape[:2]
        _max_dim = max(_w, _h)
        if _max_dim > OCR_BATCH_MAX_DIM_V14:
            _scale = OCR_BATCH_MAX_DIM_V14 / float(_max_dim)
            _new_w = max(1, int(round(_w * _scale)))
            _new_h = max(1, int(round(_h * _scale)))
            _img = cv2.resize(_img, (_new_w, _new_h), interpolation=cv2.INTER_AREA)
        resized_variants.append(_img)
    ocr_batch_v13["preprocessing"] += time.perf_counter() - _resize_start

    if hasattr(reader, "readtext_batched"):
        _ocr_batch_start = time.perf_counter()
        batched_results = reader.readtext_batched(
            resized_variants,
            batch_size=len(resized_variants),
            allowlist="0123456789",
            paragraph=False,
        )
        _batch_elapsed = time.perf_counter() - _ocr_batch_start
        ocr_batch_v13["inference"] += _batch_elapsed
        ocr_variant_times["variants2_4_batch"] += _batch_elapsed
        ocr_variant_counts["variants2_4_batch"] += 1
    else:
        _ocr_batch_start = time.perf_counter()
        batched_results = [
            reader.readtext(
                variant,
                allowlist="0123456789",
                paragraph=False,
            )
            for variant in remaining_variants
        ]
        _batch_elapsed = time.perf_counter() - _ocr_batch_start
        ocr_batch_v13["inference"] += _batch_elapsed
        ocr_variant_times["variants2_4_batch"] += _batch_elapsed
        ocr_variant_counts["variants2_4_batch"] += 1

    _parse_start = time.perf_counter()
    all_candidates = []
    for results in batched_results:
        candidates = _extract_valid_candidates(results)
        if candidates:
            all_candidates.extend(candidates)
    ocr_batch_v13["parsing"] += time.perf_counter() - _parse_start

    return _select_capacity(all_candidates)


def _ocr_box_is_still_compatible(last_box, current_box):
    if last_box is None or current_box is None:
        return False
    lc = box_center(last_box)
    cc = box_center(current_box)
    lw = max(1.0, float(last_box[2] - last_box[0]))
    lh = max(1.0, float(last_box[3] - last_box[1]))
    cw = max(1.0, float(current_box[2] - current_box[0]))
    ch = max(1.0, float(current_box[3] - current_box[1]))
    center_shift = ((cc[0] - lc[0]) ** 2 + (cc[1] - lc[1]) ** 2) ** 0.5
    size_change = max(abs(cw - lw) / lw, abs(ch - lh) / lh)
    return (
        center_shift <= max(12.0, 0.20 * max(lw, lh))
        and size_change <= 0.12
    )


def _submit_async_ocr(track, frame, cap_box):
    global _ocr_future, _ocr_future_track_id, _ocr_future_box
    if _ocr_future is not None and not _ocr_future.done():
        return False
    job_frame = frame.copy()
    job_box = tuple(cap_box)
    _ocr_future = _ocr_executor.submit(run_ocr, job_frame, job_box)
    _ocr_future_track_id = track["id"]
    _ocr_future_box = job_box
    return True


def _poll_async_ocr(track, current_cap_box=None):
    global _ocr_future, _ocr_future_track_id, _ocr_future_box
    if _ocr_future is None or not _ocr_future.done():
        return None

    future = _ocr_future
    future_track_id = _ocr_future_track_id
    future_box = _ocr_future_box
    _ocr_future = None
    _ocr_future_track_id = None
    _ocr_future_box = None

    if future_track_id != track.get("id"):
        try:
            future.result()
        except Exception as exc:
            print(f"[WARNING] Async OCR failed: {exc}")
        return None

    if current_cap_box is not None and not _ocr_box_is_still_compatible(future_box, current_cap_box):
        try:
            future.result()
        except Exception as exc:
            print(f"[WARNING] Async OCR failed: {exc}")
        return None

    try:
        return future.result()
    except Exception as exc:
        print(f"[WARNING] Async OCR failed: {exc}")
        return None

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


def mask_iou(mask_a, mask_b):
    if mask_a is None or mask_b is None:
        return 0.0
    inter = float(np.logical_and(mask_a, mask_b).sum())
    if inter == 0.0:
        return 0.0
    union = float(np.logical_or(mask_a, mask_b).sum())
    return inter / max(1.0, union)


def deduplicate_boxes(boxes, iou_thresh=0.65):
    if len(boxes) <= 1:
        return list(boxes)
    kept = []
    for box in boxes:
        if not any(iou(box, k) > iou_thresh for k in kept):
            kept.append(box)
        else:
            print(
                f"[DEDUP] Dropped duplicate bottle box "
                f"({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])})"
            )
    return kept


def stable_capacity(values):
    clean = [
        int(v)
        for v in values
        if v in {100, 300, 500}
    ]

    if not clean:
        return None

    counts = Counter(clean)
    return max(counts, key=counts.get)


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def bottle_track_match(current_box, previous_box):
    if iou(current_box, previous_box) > IOU_THRESHOLD:
        return True

    cx, cy = box_center(current_box)
    px, py = box_center(previous_box)

    current_w = max(1.0, current_box[2] - current_box[0])
    current_h = max(1.0, current_box[3] - current_box[1])
    previous_w = max(1.0, previous_box[2] - previous_box[0])
    previous_h = max(1.0, previous_box[3] - previous_box[1])

    max_w = max(current_w, previous_w)
    max_h = max(current_h, previous_h)

    horizontal_distance = abs(cx - px)
    vertical_distance = abs(cy - py)

    return (
        horizontal_distance <= max_w * 0.75
        and vertical_distance <= max_h * 0.45
    )


def get_mask_orientation(mask):
    if mask is None:
        return None

    _v16_t = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None

    _mask_u8 = mask.astype(np.uint8, copy=False)
    _points = cv2.findNonZero(_mask_u8)
    if _points is None or len(_points) < 20:
        if ACTIVE_PROFILE_THIS_FRAME:
            orientation_profile_times["mask_pixels"] += time.perf_counter() - _v16_t
        return None
    _points = _points.reshape(-1, 2)
    xs = _points[:, 0]
    ys = _points[:, 1]
    if len(xs) < 20:
        if ACTIVE_PROFILE_THIS_FRAME:
            orientation_profile_times["mask_pixels"] += time.perf_counter() - _v16_t
        return None

    xs = xs.astype(np.float32, copy=False)
    ys = ys.astype(np.float32, copy=False)
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["mask_pixels"] += time.perf_counter() - _v16_t

    _v16_t = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    if ACTIVE_PROFILE_THIS_FRAME: _v19_t = time.perf_counter()
    center_x = float(xs.mean())
    center_y = float(ys.mean())
    n = float(len(xs))
    if ACTIVE_PROFILE_THIS_FRAME: orientation_profile_times["centering_mean"] += time.perf_counter() - _v19_t

    if ACTIVE_PROFILE_THIS_FRAME: _v19_t = time.perf_counter()
    sum_x2 = float(np.dot(xs, xs))
    sum_xy = float(np.dot(xs, ys))
    sum_y2 = float(np.dot(ys, ys))
    if ACTIVE_PROFILE_THIS_FRAME: orientation_profile_times["centering_demean"] += time.perf_counter() - _v19_t

    if ACTIVE_PROFILE_THIS_FRAME: _v19_t = time.perf_counter()
    denom = max(1.0, n - 1.0)
    cov_xx = (sum_x2 - n * center_x * center_x) / denom
    cov_xy = (sum_xy - n * center_x * center_y) / denom
    cov_yy = (sum_y2 - n * center_y * center_y) / denom
    covariance = np.array(
        [[cov_xx, cov_xy], [cov_xy, cov_yy]],
        dtype=np.float64,
    )
    if ACTIVE_PROFILE_THIS_FRAME: orientation_profile_times["centering_covariance_ops"] += time.perf_counter() - _v19_t
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["centering_covariance"] += time.perf_counter() - _v16_t

    _v16_t = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        if ACTIVE_PROFILE_THIS_FRAME:
            orientation_profile_times["eigen_angle"] += time.perf_counter() - _v16_t
        return None

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    norm = float(np.linalg.norm(major))
    if norm < 1e-6:
        if ACTIVE_PROFILE_THIS_FRAME:
            orientation_profile_times["eigen_angle"] += time.perf_counter() - _v16_t
        return None

    major /= norm
    if major[1] < 0:
        major = -major

    minor = np.array([-major[1], major[0]], dtype=np.float32)

    angle_deg = float(
        np.degrees(
            np.arctan2(
                abs(float(major[0])),
                abs(float(major[1])),
            )
        )
    )
    status = "PASS" if angle_deg <= ORIENTATION_MAX_ANGLE_DEG else "FAIL"
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["eigen_angle"] += time.perf_counter() - _v16_t

    _v16_t = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    if ACTIVE_PROFILE_THIS_FRAME: _v20_t = time.perf_counter()
    major_x = float(major[0])
    major_y = float(major[1])
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["projection_center"] += time.perf_counter() - _v20_t

    if ACTIVE_PROFILE_THIS_FRAME: _v20_t = time.perf_counter()
    projections = (xs - center_x) * major_x + (ys - center_y) * major_y
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["projection_dot"] += time.perf_counter() - _v20_t

    if ACTIVE_PROFILE_THIS_FRAME: _v20_t = time.perf_counter()
    half_length = max(20.0, float(np.max(np.abs(projections))))
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["projection_max"] += time.perf_counter() - _v20_t
        orientation_profile_times["projection"] += time.perf_counter() - _v16_t

    _v16_t = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    if ACTIVE_PROFILE_THIS_FRAME: _v18_t = time.perf_counter()
    mask_u8 = mask.astype(np.uint8) * 255
    if ACTIVE_PROFILE_THIS_FRAME: orientation_profile_times["contour_prepare"] += time.perf_counter() - _v18_t
    if ACTIVE_PROFILE_THIS_FRAME: _v18_t = time.perf_counter()
    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if ACTIVE_PROFILE_THIS_FRAME: orientation_profile_times["contour_find"] += time.perf_counter() - _v18_t
    if ACTIVE_PROFILE_THIS_FRAME: _v18_t = time.perf_counter()
    mask_contour = None
    if contours:
        mask_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(mask_contour) < 20:
            mask_contour = None
    if ACTIVE_PROFILE_THIS_FRAME: orientation_profile_times["contour_select"] += time.perf_counter() - _v18_t
    if ACTIVE_PROFILE_THIS_FRAME:
        orientation_profile_times["contour"] += time.perf_counter() - _v16_t

    return {
        "status": status,
        "angle_deg": angle_deg,
        "center": (center_x, center_y),
        "major": (float(major[0]), float(major[1])),
        "minor": (float(minor[0]), float(minor[1])),
        "half_length": half_length,
        "mask_contour": mask_contour.reshape(-1, 2).tolist() if mask_contour is not None else None,
    }


def find_bottle_mask(bottle_box, segmentation_detections):
    if segmentation_detections is None or segmentation_detections.mask is None:
        return None

    best_mask = None
    best_score = 0.0
    class_names = segmentation_detections.data.get("class_name", [])

    for i, cls in enumerate(class_names):
        if cls != BOTTLE_CLASS_NAME:
            continue
        if float(segmentation_detections.confidence[i]) < PER_CLASS_CONF.get(BOTTLE_CLASS_NAME, SEG_THRESHOLD):
            continue

        seg_box = tuple(map(int, segmentation_detections.xyxy[i]))
        score = iou(bottle_box, seg_box)

        if score > best_score:
            best_score = score
            best_mask = segmentation_detections.mask[i]

    return best_mask

def get_mask_centroid(mask):
    if mask is None:
        return None
    binary = mask.astype(np.uint8, copy=False)
    moments = cv2.moments(binary, binaryImage=True)
    area = moments["m00"]
    if area <= 0.0:
        return None
    return (float(moments["m10"] / area), float(moments["m01"] / area))

def find_label_mask(bottle_box, seg_detections, frame):
    if seg_detections is None or seg_detections.mask is None:
        return None
    best_mask, best_score = None, 0.0
    for i, cls in enumerate(seg_detections.data.get("class_name", [])):
        if cls != "label":
            continue
        if float(seg_detections.confidence[i]) < PER_CLASS_CONF.get("label", SEG_THRESHOLD):
            continue
        score = iou(bottle_box, tuple(map(int, seg_detections.xyxy[i])))
        if score > best_score:
            best_score = score
            best_mask = seg_detections.mask[i]
    if best_mask is None:
        return None
    best_mask = resize_mask_to_frame(best_mask, frame)
    return constrain_mask_to_box(best_mask, bottle_box)

def resize_mask_to_frame(mask, frame):
    if mask is None:
        return None
    fh, fw = frame.shape[:2]
    mh, mw = mask.shape[:2]
    if (mh, mw) == (fh, fw):
        return mask
    resized = cv2.resize(mask.astype(np.uint8), (fw, fh), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def constrain_mask_to_box(mask, box):
    if mask is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    mh, mw = mask.shape[:2]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(mw, x2); y2 = min(mh, y2)
    constrained = np.zeros_like(mask)
    if x2 > x1 and y2 > y1:
        constrained[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return constrained


def draw_mask_contour(image, orientation_data, color=(0, 255, 0), thickness=3, fill=False):
    if orientation_data is None:
        return

    contour = orientation_data.get("mask_contour")
    if contour is None or len(contour) < 3:
        return

    pts = np.asarray(contour, dtype=np.float32).reshape(-1, 1, 2)
    pts = np.round(pts).astype(np.int32)

    if fill:
        overlay = image.copy()
        cv2.fillPoly(overlay, [pts], color)
        image[:] = cv2.addWeighted(overlay, 0.14, image, 0.86, 0)

    cv2.polylines(image, [pts], True, color, thickness, cv2.LINE_AA)


def draw_mask_orientation(image, orientation_data, color=(255, 0, 0), thickness=3):
    if orientation_data is None:
        return

    cx, cy = orientation_data["center"]
    dx, dy = orientation_data["major"]
    mx, my = orientation_data["minor"]
    half_length = orientation_data["half_length"]

    cx, cy = int(round(cx)), int(round(cy))

    p1 = (
        int(round(cx - dx * half_length)),
        int(round(cy - dy * half_length)),
    )
    p2 = (
        int(round(cx + dx * half_length)),
        int(round(cy + dy * half_length)),
    )
    cv2.line(image, p1, p2, color, thickness, cv2.LINE_AA)

    minor_half = max(15.0, half_length * 0.25)
    q1 = (
        int(round(cx - mx * minor_half)),
        int(round(cy - my * minor_half)),
    )
    q2 = (
        int(round(cx + mx * minor_half)),
        int(round(cy + my * minor_half)),
    )
    cv2.line(image, q1, q2, color, max(1, thickness - 1), cv2.LINE_AA)
    cv2.drawMarker(image, (cx, cy), color, cv2.MARKER_CROSS, 18, 2)

def is_reliable_bottle_frame(box, frame_shape):
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)

    return (
        x1 >= COMPLETE_X_MARGIN
        and x2 <= frame_w - COMPLETE_X_MARGIN
        and y1 >= 0
        and y2 <= frame_h
    )

def check_centricity(
    bottle_box,
    label_box,
    bottle_mask=None,
    label_mask=None,
):
    bottle_c = get_mask_centroid(bottle_mask)
    label_c = get_mask_centroid(label_mask)

    bx, by = bottle_c if bottle_c is not None else box_center(bottle_box)
    lx, ly = label_c if label_c is not None else box_center(label_box)

    bw = max(1.0, float(bottle_box[2] - bottle_box[0]))
    bh = max(1.0, float(bottle_box[3] - bottle_box[1]))

    h_offset = (lx - bx) / bw
    v_offset = (ly - by) / bh

    h_ok = abs(h_offset) <= H_CENTRICITY_THRESH

    v_ok = (
        abs(v_offset - EXPECTED_V)
        <= V_DEVIATION_THRESH
    )

    return (
        "PASS" if h_ok else "FAIL",
        "PASS" if v_ok else "FAIL",
        h_offset,
        v_offset,
        (bx, by),
        (lx, ly),
    )

def has_crossed_trigger_line(track, box, frame_width):
    line_x = int(frame_width * TRIGGER_LINE_X_RATIO)

    current_center_x = (box[0] + box[2]) / 2.0
    previous_center_x = track.get("previous_center_x")

    crossed = False

    if previous_center_x is not None:
        previous_side = previous_center_x - line_x
        current_side = current_center_x - line_x

        if (
            previous_side * current_side <= 0
            and abs(current_center_x - previous_center_x) >= TRIGGER_LINE_TOLERANCE
        ):
            crossed = True

    track["previous_center_x"] = current_center_x

    if crossed:
        track["trigger_crossed"] = True
        print(
            f"Bottle #{track['id'] + 1} crossed trigger line "
            f"at x={line_x}"
        )

    return crossed


def create_track(box):
    global next_track_id
    track = {
        "id": next_track_id,
        "box": box,
        "capacity": None,
        "defects": [],
        "defect_streaks": {
            "damage": 0,
            "bump": 0,
        },
        "defect_missing_frames": {
            "damage": 0,
            "bump": 0,
        },
        "orientation": None,
        "orientation_data": None,
        "_current_bottle_mask": None,
        "best_complete_orientation_data": None,
        "best_defect_orientation_data": None,
        "h_center": None,
        "v_center": None,
        "missing": 0,
        "saved": False,
        "frames_seen": 0,
        # Counts frames where this track was actually matched to a
        # detection (not merely frames elapsed since creation).
        "matched_frames": 1,
        "previous_center_x": None,
        "trigger_crossed": False,
        "best_box": box,
        "best_frame": None,
        "best_complete_box": None,
        "best_complete_frame": None,
        "best_complete_label_box": None,
        "best_complete_damage_boxes": [],
        "best_complete_bump_boxes": [],
        "best_defect_frame": None,
        "best_defect_box": None,
        "best_defect_label_box": None,
        "best_defect_damage_boxes": [],
        "best_defect_bump_boxes": [],

        "best_complete_defect_frame": None,
        "best_complete_defect_box": None,
        "best_complete_defect_label_box": None,
        "best_complete_defect_damage_boxes": [],
        "best_complete_defect_bump_boxes": [],
        "best_complete_defect_orientation_data": None,

        "best_defect_damage_relative": [],
        "best_defect_bump_relative": [],

        "best_valid_box": None,
        "best_valid_frame": None,
        "annotation_label_box": None,
        "annotation_damage_boxes": [],
        "annotation_bump_boxes": [],
        "best_valid_h_center": None,
        "best_valid_v_center": None,
        "best_valid_centricity_error": None,
        "centricity_offset_history": [],
        "h_history": [],
        "v_history": [],
        "orientation_history": [],
        "h_value_history": [],
        "v_value_history": [],
        "h_px_history": [],
        "v_px_history": [],
        "orientation_angle_history": [],
        "final_h_value": None,
        "final_v_value": None,
        "final_orientation_angle": None,
        "capacity_history": [],
        "ocr_capacity_locked": False,
        "ocr_last_capacity": None,
        "ocr_same_capacity_count": 0,
        "ocr_last_box": None,
        "ocr_cached_capacity": None,
        "ocr_cache_age": 0,
        "ocr_frames_since_inference": 0,
        "ocr_temporal_interval": 3,
        "ocr_async_pending": False,
        "finalized": False,
        "final_status": None,
    }
    next_track_id += 1
    return track


def majority_result(values):
    clean = [v for v in values if v in {"PASS", "FAIL"}]
    if not clean:
        return None
    passes = clean.count("PASS")
    fails = clean.count("FAIL")
    return "PASS" if passes >= fails else "FAIL"


def stable_result(values):
    result = majority_result(values)
    if result is None:
        return None
    return result


def stable_numeric(values):
    clean = [
        float(v)
        for v in values
        if v is not None and np.isfinite(float(v))
    ]

    if not clean:
        return None

    return float(np.median(clean))


def format_measurement(value, status, suffix=""):
    if value is None:
        return status or "N/A"

    if status in {"PASS", "FAIL"}:
        return f"{value:.3f}{suffix} ({status})"

    return f"{value:.3f}{suffix} ({status or 'N/A'})"


def finalize_measurements(track):
    h_values = [
        abs(float(x))
        for x in track.get("h_value_history", [])
        if x is not None and np.isfinite(float(x))
    ]

    v_values = [
        abs(float(x))
        for x in track.get("v_value_history", [])
        if x is not None and np.isfinite(float(x))
    ]

    if h_values:
        h = (
            "PASS"
            if float(np.median(h_values)) <= H_CENTRICITY_THRESH
            else "FAIL"
        )
    else:
        h = "Pending"

    if v_values:
        v = (
            "PASS"
            if abs(float(np.median(v_values)) - EXPECTED_V)
            <= V_DEVIATION_THRESH
            else "FAIL"
        )
    else:
        v = "Pending"

    if h == "Pending" or v == "Pending":
        fallback_box = track.get("best_complete_box")
        fallback_label = track.get("best_complete_label_box")
        if fallback_box is not None and fallback_label is not None:
            bx_c = (fallback_box[0] + fallback_box[2]) / 2.0
            by_c = (fallback_box[1] + fallback_box[3]) / 2.0
            lx_c = (fallback_label[0] + fallback_label[2]) / 2.0
            ly_c = (fallback_label[1] + fallback_label[3]) / 2.0
            bw_c = max(1.0, float(fallback_box[2] - fallback_box[0]))
            bh_c = max(1.0, float(fallback_box[3] - fallback_box[1]))
            h_off_c = (lx_c - bx_c) / bw_c
            v_off_c = (ly_c - by_c) / bh_c
            if h == "Pending":
                h = "PASS" if abs(h_off_c) <= H_CENTRICITY_THRESH else "FAIL"
                track["final_h_value"] = abs(h_off_c)
                print(
                    f"Bottle #{track['id'] + 1} H fallback → "
                    f"{h_off_c:.3f} ({h})"
                )
            if v == "Pending":
                v = "PASS" if abs(v_off_c - EXPECTED_V) <= V_DEVIATION_THRESH else "FAIL"
                track["final_v_value"] = abs(v_off_c)
                print(
                    f"Bottle #{track['id'] + 1} V fallback → "
                    f"{v_off_c:.3f} ({v})"
                )

    track["h_center"] = h
    track["v_center"] = v

    orientation = majority_result(
        track.get("orientation_history", [])
    )
    h_numeric = stable_numeric([abs(v) for v in track.get("h_value_history", [])])
    v_numeric = stable_numeric([abs(v) for v in track.get("v_value_history", [])])
    if h_numeric is not None:
        track["final_h_value"] = h_numeric
    if v_numeric is not None:
        track["final_v_value"] = v_numeric
    track["final_orientation_angle"] = stable_numeric(
        track.get("orientation_angle_history", [])
    )

    if track["h_center"] is None:
        track["h_center"] = "Pending"
    if track["v_center"] is None:
        track["v_center"] = "Pending"
    if track["orientation"] is None:
        track["orientation"] = "Pending"

    if track.get("capacity") is None:
        track["capacity"] = EXPECTED_CAPACITY

    track["finalized"] = True

    missing_measurement = any(
        track[key] == "Pending"
        for key in ("orientation", "h_center", "v_center")
    )

    if missing_measurement:
        track["final_status"] = "INCOMPLETE"
    elif bool(track["defects"]) or any(
        track[key] == "FAIL"
        for key in ("orientation", "h_center", "v_center")
    ):
        track["final_status"] = "DEFECTIVE"
    else:
        track["final_status"] = "GOOD"
    return track["final_status"]


def analyze_bottle(frame, bottle_box, capacity_boxes, label_boxes, damage_boxes, bump_boxes):
    result = {
        "capacity": None,
        "defects": [],
        "orientation": None,
        "h_center": None,
        "v_center": None,
    }

    _cap_tol = 15
    for cap_box in capacity_boxes:
        if (
            cap_box[0] >= bottle_box[0] - _cap_tol
            and cap_box[2] <= bottle_box[2] + _cap_tol
            and cap_box[1] >= bottle_box[1] - _cap_tol
            and cap_box[3] <= bottle_box[3] + _cap_tol
        ):
            break

    result["defects"] = []

    return result


def draw_bottle_annotation(image, bottle_box, track, label_box=None, damage_boxes=None, bump_boxes=None, orientation_data=None):
    h_img, w_img = image.shape[:2]
    x1, y1, x2, y2 = map(int, bottle_box)

    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    bottle_color = (0, 255, 0)
    label_color = (255, 0, 255)
    axis_color = (255, 255, 0)
    centricity_color = (0, 165, 255)
    orientation_color = (255, 0, 0)
    defect_color = (0, 0, 255)
    text_color = (255, 255, 255)

    cv2.rectangle(image, (x1, y1), (x2, y2), bottle_color, 3)
    cv2.putText(image, "BOTTLE", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bottle_color, 2, cv2.LINE_AA)

    bottle_cx = int(round((x1 + x2) / 2))
    bottle_cy = int(round((y1 + y2) / 2))

    if orientation_data is not None:
        local_orientation = dict(orientation_data)

        draw_mask_contour(
            image,
            local_orientation,
            color=bottle_color,
            thickness=3,
            fill=False
        )

        draw_mask_orientation(
            image,
            local_orientation,
            orientation_color,
            3
        )

    label_cx = None
    label_cy = None
    if label_box is not None:
        lx1, ly1, lx2, ly2 = map(int, label_box)
        lx1 = max(0, min(lx1, w_img - 1))
        ly1 = max(0, min(ly1, h_img - 1))
        lx2 = max(0, min(lx2, w_img - 1))
        ly2 = max(0, min(ly2, h_img - 1))

        if lx2 > lx1 and ly2 > ly1:
            cv2.rectangle(image, (lx1, ly1), (lx2, ly2), label_color, 3)
            cv2.putText(image, "LABEL", (lx1, max(20, ly1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

            label_cx = int(round((lx1 + lx2) / 2))
            label_cy = int(round((ly1 + ly2) / 2))

            cv2.drawMarker(image, (label_cx, label_cy), label_color, cv2.MARKER_CROSS, 20, 2)
            cv2.line(image, (bottle_cx, bottle_cy), (label_cx, bottle_cy), centricity_color, 3)
            cv2.line(image, (label_cx, bottle_cy), (label_cx, label_cy), centricity_color, 3)
            cv2.line(image, (bottle_cx, bottle_cy), (label_cx, label_cy), (255, 255, 255), 1)

    for defect_box in damage_boxes or []:
        dx1, dy1, dx2, dy2 = map(int, defect_box)
        dx1 = max(0, min(dx1, w_img - 1))
        dy1 = max(0, min(dy1, h_img - 1))
        dx2 = max(0, min(dx2, w_img - 1))
        dy2 = max(0, min(dy2, h_img - 1))
        if dx2 > dx1 and dy2 > dy1:
            cv2.rectangle(image, (dx1, dy1), (dx2, dy2), defect_color, 3)
            cv2.putText(image, "DAMAGE", (dx1, max(20, dy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

    for defect_box in bump_boxes or []:
        bx1, by1, bx2, by2 = map(int, defect_box)
        bx1 = max(0, min(bx1, w_img - 1))
        by1 = max(0, min(by1, h_img - 1))
        bx2 = max(0, min(bx2, w_img - 1))
        by2 = max(0, min(by2, h_img - 1))
        if bx2 > bx1 and by2 > by1:
            cv2.rectangle(image, (bx1, by1), (bx2, by2), defect_color, 3)
            cv2.putText(image, "BUMP", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

    is_defective = (
        bool(track["defects"])
        or track["orientation"] == "FAIL"
        or track["h_center"] == "FAIL"
        or track["v_center"] == "FAIL"
    )
    status = "DEFECTIVE" if is_defective else "GOOD"
    status_color = defect_color if is_defective else bottle_color

    # FIX 2: pixel-based centricity values on the saved annotation.
    # Compute px offsets directly from the same centers drawn above,
    # so the saved image and the on-screen value always match.
    h_px_text = ""
    v_px_text = ""
    if label_cx is not None and label_cy is not None:
        h_px_text = f" | {abs(label_cx - bottle_cx)}px"
        v_px_text = f" | {abs(label_cy - bottle_cy)}px"

    lines = [
        f"Bottle #{track['id'] + 1}",
        f"Status: {status}",
        f"Capacity: {track['capacity']} ml" if track["capacity"] else "Capacity: N/A",
        "Orientation: "
        + (
            format_measurement(
                track.get("final_orientation_angle"),
                track.get("orientation"),
                " deg",
            )
            if track.get("final_orientation_angle") is not None
            else (track.get("orientation") or "N/A")
        ),
        "H Center: "
        + (
            format_measurement(
                track.get("final_h_value"),
                track.get("h_center"),
            )
            if track.get("final_h_value") is not None
            else (track.get("h_center") or "Pending")
        )
        + h_px_text,
        "V Center: "
        + (
            format_measurement(
                track.get("final_v_value"),
                track.get("v_center"),
            )
            if track.get("final_v_value") is not None
            else (track.get("v_center") or "Pending")
        )
        + v_px_text,
        f"Defects: {', '.join(track['defects']) if track['defects'] else 'None'}",
        "Blue = orientation axes",
        "Orange = centricity",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.52
    thickness = 2
    line_height = 22
    panel_width = min(w_img - 10, 360)
    panel_height = min(h_img - 10, line_height * len(lines) + 20)

    if panel_width > 20 and panel_height > 20:
        overlay = image.copy()
        cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
        image[:] = cv2.addWeighted(overlay, 0.68, image, 0.32, 0)
        for i, line in enumerate(lines):
            color = status_color if i == 1 else text_color
            cv2.putText(image, line, (12, 25 + i * line_height), font, font_scale, color, thickness, cv2.LINE_AA)


def save_bottle_images(frame, track):
    global completed_count, good_count, defective_count, incomplete_count

    if track["saved"]:
        return False

    if not track.get("finalized", False):
        return False

    saved_orientation_data = None

    if (
        track.get("best_complete_defect_frame") is not None
        and track.get("defects")
    ):
        frame = track["best_complete_defect_frame"]
        box = track["best_complete_defect_box"]
        saved_label_box = track.get(
            "best_complete_defect_label_box"
        )
        saved_damage_boxes = track.get(
            "best_complete_defect_damage_boxes",
            []
        )
        saved_bump_boxes = track.get(
            "best_complete_defect_bump_boxes",
            []
        )
        saved_orientation_data = track.get(
            "best_complete_defect_orientation_data"
        )

    elif track["best_complete_frame"] is not None:
        frame = track["best_complete_frame"]
        box = track["best_complete_box"]
        saved_label_box = track.get("best_complete_label_box")
        saved_damage_boxes = track.get(
            "best_complete_damage_boxes",
            []
        )
        saved_bump_boxes = track.get(
            "best_complete_bump_boxes",
            []
        )
        saved_orientation_data = track.get(
            "best_complete_orientation_data"
        )

    elif track.get("best_defect_frame") is not None and track.get("defects"):
        frame = track["best_defect_frame"]
        box = track["best_defect_box"]
        saved_label_box = track.get("best_defect_label_box")
        saved_damage_boxes = track.get("best_defect_damage_boxes", [])
        saved_bump_boxes = track.get("best_defect_bump_boxes", [])
        saved_orientation_data = track.get("best_defect_orientation_data")

    else:
        print(
            f"WARNING: Bottle #{track['id'] + 1} has no complete "
            f"in-frame detection; skipping save."
        )
        return False

    bx1, by1, bx2, by2 = map(int, box)
    bw = max(1.0, float(bx2 - bx1))
    bh = max(1.0, float(by2 - by1))

    if "bump" in track["defects"] and not saved_bump_boxes:
        saved_bump_boxes = track.get("best_complete_bump_boxes", [])

        if not saved_bump_boxes:
            saved_bump_boxes = [
                (
                    int(round(bx1 + rx1 * bw)),
                    int(round(by1 + ry1 * bh)),
                    int(round(bx1 + rx2 * bw)),
                    int(round(by1 + ry2 * bh)),
                )
                for rx1, ry1, rx2, ry2
                in track.get("best_defect_bump_relative", [])
            ]

    if "damage" in track["defects"] and not saved_damage_boxes:
        saved_damage_boxes = track.get(
            "best_complete_damage_boxes",
            []
        )

        if not saved_damage_boxes:
            saved_damage_boxes = [
                (
                    int(round(bx1 + rx1 * bw)),
                    int(round(by1 + ry1 * bh)),
                    int(round(bx1 + rx2 * bw)),
                    int(round(by1 + ry2 * bh)),
                )
                for rx1, ry1, rx2, ry2
                in track.get("best_defect_damage_relative", [])
            ]

    x1, y1, x2, y2 = map(int, box)
    frame_h, frame_w = frame.shape[:2]

    x1 = max(0, x1 - SAVE_PADDING)
    y1 = max(0, y1 - SAVE_PADDING)
    x2 = min(frame_w, x2 + SAVE_PADDING)
    y2 = min(frame_h, y2 + SAVE_PADDING)

    if x2 <= x1 or y2 <= y1:
        print(f"WARNING: Could not save Bottle #{track['id'] + 1}: invalid crop.")
        return False

    original_crop = frame[y1:y2, x1:x2].copy()
    if original_crop.size == 0:
        print(f"WARNING: Could not save Bottle #{track['id'] + 1}: empty crop.")
        return False

    annotated_crop = original_crop.copy()

    local_box = (int(box[0] - x1), int(box[1] - y1), int(box[2] - x1), int(box[3] - y1))

    def to_local_box(saved_box):
        if saved_box is None:
            return None
        return (int(saved_box[0] - x1), int(saved_box[1] - y1), int(saved_box[2] - x1), int(saved_box[3] - y1))

    local_label_box = to_local_box(saved_label_box)
    local_damage_boxes = [to_local_box(b) for b in saved_damage_boxes]
    local_bump_boxes = [to_local_box(b) for b in saved_bump_boxes]

    local_orientation_data = saved_orientation_data

    if local_orientation_data is not None:
        contour = local_orientation_data.get("mask_contour")
        local_orientation_data = dict(local_orientation_data)
        ocx, ocy = local_orientation_data["center"]
        local_orientation_data["center"] = (ocx - x1, ocy - y1)

        if contour is not None and len(contour) > 0:
            pts = np.asarray(contour, dtype=np.float32)
            local_pts = pts.copy()
            local_pts[:, 0] -= x1
            local_pts[:, 1] -= y1
            local_orientation_data["mask_contour"] = local_pts.tolist()

            print(
                f"DEBUG SAVE Bottle #{track['id'] + 1}: "
                f"local_box={local_box}, "
                f"local_mask_bbox=("
                f"{local_pts[:, 0].min():.0f}, "
                f"{local_pts[:, 1].min():.0f}, "
                f"{local_pts[:, 0].max():.0f}, "
                f"{local_pts[:, 1].max():.0f})"
            )

    draw_bottle_annotation(
        annotated_crop,
        local_box,
        track,
        label_box=local_label_box,
        damage_boxes=local_damage_boxes,
        bump_boxes=local_bump_boxes,
        orientation_data=local_orientation_data,
    )

    final_status = track.get("final_status")
    if final_status is None:
        missing_measurement = any(
            track[key] in {None, "Pending"}
            for key in ("orientation", "h_center", "v_center")
        )
        if missing_measurement:
            final_status = "INCOMPLETE"
        elif bool(track["defects"]) or any(
            track[key] == "FAIL"
            for key in ("orientation", "h_center", "v_center")
        ):
            final_status = "DEFECTIVE"
        else:
            final_status = "GOOD"
        track["final_status"] = final_status

    category = final_status.lower()

    print("=" * 40)
    print(f"Bottle #{track['id'] + 1} FINAL RESULT")
    orientation_text = (
        format_measurement(
            track.get("final_orientation_angle"),
            track.get("orientation"),
            " deg",
        )
        if track.get("final_orientation_angle") is not None
        else (track.get("orientation") or "N/A")
    )
    h_center_text = (
        format_measurement(
            track.get("final_h_value"),
            track.get("h_center"),
        )
        if track.get("final_h_value") is not None
        else (track.get("h_center") or "Pending")
    )
    v_center_text = (
        format_measurement(
            track.get("final_v_value"),
            track.get("v_center"),
        )
        if track.get("final_v_value") is not None
        else (track.get("v_center") or "Pending")
    )

    print(f"Orientation : {orientation_text}")
    print(f"H Center    : {h_center_text}")
    print(f"V Center    : {v_center_text}")
    print(f"Defects     : {', '.join(track['defects']) if track['defects'] else 'None'}")
    print(f"Status      : {final_status}")
    print("=" * 40)

    category_dir = os.path.join(SAVE_DIR, category)
    raw_dir = os.path.join(category_dir, "original")
    annotated_dir = os.path.join(category_dir, "annotated")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)

    bottle_filename = f"bottle_{track['id'] + 1:03d}_{EXPECTED_CAPACITY}ml.jpg"
    original_path = os.path.join(raw_dir, bottle_filename)
    annotated_path = os.path.join(annotated_dir, bottle_filename)

    original_saved = cv2.imwrite(original_path, original_crop)
    annotated_saved = cv2.imwrite(annotated_path, annotated_crop)

    if original_saved and annotated_saved:
        track["saved"] = True
        track["finalized"] = True
        track["final_status"] = final_status

        completed_count += 1
        if final_status == "DEFECTIVE":
            defective_count += 1
        elif final_status == "INCOMPLETE":
            incomplete_count += 1
        else:
            good_count += 1

        print(f"Saved Bottle #{track['id'] + 1}:")
        print(f"  Original  : {original_path}")
        print(f"  Annotated : {annotated_path}")
        print(
            f"Final counts -> Total: {completed_count} | "
            f"Good: {good_count} | Defective: {defective_count} | "
            f"Incomplete: {incomplete_count}"
        )
        return True

    print(f"WARNING: Failed to save both images for Bottle #{track['id'] + 1}.")
    return False


def get_matching_label_box(bottle_box, label_boxes):
    bx, by = box_center(bottle_box)
    candidates = []

    for lbl_box in label_boxes:
        label_overlaps_bottle = (
            lbl_box[2] > bottle_box[0] - LABEL_CONTAINMENT_TOLERANCE
            and lbl_box[0] < bottle_box[2] + LABEL_CONTAINMENT_TOLERANCE
            and lbl_box[3] > bottle_box[1] - LABEL_CONTAINMENT_TOLERANCE
            and lbl_box[1] < bottle_box[3] + LABEL_CONTAINMENT_TOLERANCE
        )

        if not label_overlaps_bottle:
            continue

        clipped_label_box = (
            max(lbl_box[0], bottle_box[0]),
            max(lbl_box[1], bottle_box[1]),
            min(lbl_box[2], bottle_box[2]),
            min(lbl_box[3], bottle_box[3]),
        )

        if (
            clipped_label_box[2] <= clipped_label_box[0]
            or clipped_label_box[3] <= clipped_label_box[1]
        ):
            continue

        lx, ly = box_center(clipped_label_box)
        bw = max(1.0, float(bottle_box[2] - bottle_box[0]))
        bh = max(1.0, float(bottle_box[3] - bottle_box[1]))

        distance = (
            ((lx - bx) / bw) ** 2
            + ((ly - by) / bh) ** 2
        ) ** 0.5

        area = (
            clipped_label_box[2] - clipped_label_box[0]
        ) * (
            clipped_label_box[3] - clipped_label_box[1]
        )

        candidates.append((distance, -area, clipped_label_box))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]

def update_centricity(track, bottle_box, label_boxes, seg_detections=None, bottle_mask=None, frame=None):
    _t0 = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None

    # FIX 1: skip centricity update when the bottle box is not reliably
    # inside the frame (e.g. first detected while still entering at the
    # right edge). Edge-truncated boxes produce a skewed bottle centroid
    # that poisons the offset history and causes spurious centricity FAILs.
    if frame is not None and not is_reliable_bottle_frame(bottle_box, frame.shape):
        return False

    _s = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    clipped_label_box = get_matching_label_box(bottle_box, label_boxes)
    if ACTIVE_PROFILE_THIS_FRAME:
        centricity_profile_times["label_box_match"] += time.perf_counter() - _s
    if clipped_label_box is None:
        return False

    _s = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    label_mask = (
        find_label_mask(bottle_box, seg_detections, frame)
        if seg_detections is not None and frame is not None
        else None
    )
    if ACTIVE_PROFILE_THIS_FRAME:
        centricity_profile_times["label_mask_lookup"] += time.perf_counter() - _s

    _s = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    (
        _h_status,
        _v_status,
        h_offset,
        v_offset,
        (bx, by),
        (lx, ly),
    ) = check_centricity(
        bottle_box,
        clipped_label_box,
        bottle_mask=bottle_mask,
        label_mask=label_mask,
    )
    if ACTIVE_PROFILE_THIS_FRAME:
        centricity_profile_times["centroid_calculation"] += time.perf_counter() - _s

    _s = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else None
    history = track.get("centricity_offset_history", [])
    if len(history) >= 1:
        previous_h, previous_v = history[-1]
        if (abs(h_offset - previous_h) > CENTRICITY_SPATIAL_TOLERANCE or
                abs(v_offset - previous_v) > CENTRICITY_SPATIAL_TOLERANCE):
            print(f"Bottle #{track['id'] + 1} centricity sudden jump ignored: "
                  f"offset=({h_offset:.3f}, {v_offset:.3f}), "
                  f"previous=({previous_h:.3f}, {previous_v:.3f})")
            if ACTIVE_PROFILE_THIS_FRAME:
                centricity_profile_times["history_and_status"] += time.perf_counter() - _s
            return False

    history.append((h_offset, v_offset))
    if len(history) > CENTRICITY_HISTORY_WINDOW:
        del history[:-CENTRICITY_HISTORY_WINDOW]
    track["centricity_offset_history"] = history

    h = "PASS" if abs(h_offset) <= H_CENTRICITY_THRESH else "FAIL"

    v = (
        "PASS"
        if abs(v_offset - EXPECTED_V) <= V_DEVIATION_THRESH
        else "FAIL"
    )
    track["h_center"] = h
    track["v_center"] = v

    # Raw pixel offsets for on-screen / saved-image display (does not affect classification).
    track["h_px_history"].append(abs(lx - bx))
    track["v_px_history"].append(abs(ly - by))

    if ACTIVE_PROFILE_THIS_FRAME:
        centricity_profile_times["history_and_status"] += time.perf_counter() - _s

    print(f"Bottle #{track['id'] + 1} mask centricity updated: H={h}, V={v}, "
          f"mask_offset=({h_offset:.3f}, {v_offset:.3f}), "
          f"bottle_centroid=({bx:.1f}, {by:.1f}), "
          f"label_centroid=({lx:.1f}, {ly:.1f})"
        )

    return True

def defect_on_bottle(defect_box, bottle_mask):
    if bottle_mask is None:
        return False

    dx1, dy1, dx2, dy2 = [int(v) for v in defect_box]

    mh, mw = bottle_mask.shape[:2]

    dx1 = max(0, dx1)
    dy1 = max(0, dy1)
    dx2 = min(mw, dx2)
    dy2 = min(mh, dy2)

    if dx2 <= dx1 or dy2 <= dy1:
        return False

    roi = bottle_mask[dy1:dy2, dx1:dx2]

    overlap_ratio = float(roi.sum()) / max(1, roi.size)

    return overlap_ratio >= DEFECT_OVERLAP_THRESH


def update_defects(
    track,
    bottle_box,
    damage_boxes,
    bump_boxes,
    frame=None,
    label_box=None,
    orientation_data=None,
    bottle_mask=None,
):
    defects = set(track["defects"])

    current_mask = bottle_mask

    if current_mask is None:
        current_mask = track.get("_current_bottle_mask")
    damage_valid = any(
        defect_on_bottle(dmg, current_mask)
        for dmg in damage_boxes
    )

    bump_valid = any(
        defect_on_bottle(bump, current_mask)
        for bump in bump_boxes
    )

    if damage_valid:
        track["defect_streaks"]["damage"] += 1
        track["defect_missing_frames"]["damage"] = 0
    else:
        if track["defect_streaks"]["damage"] > 0:
            track["defect_missing_frames"]["damage"] += 1

            if (
                track["defect_missing_frames"]["damage"]
                > DEFECT_MAX_MISSING_FRAMES
            ):
                track["defect_streaks"]["damage"] = 0
                track["defect_missing_frames"]["damage"] = 0

    if bump_valid:
        track["defect_streaks"]["bump"] += 1
        track["defect_missing_frames"]["bump"] = 0
    else:
        if track["defect_streaks"]["bump"] > 0:
            track["defect_missing_frames"]["bump"] += 1

            if (
                track["defect_missing_frames"]["bump"]
                > DEFECT_MAX_MISSING_FRAMES
            ):
                track["defect_streaks"]["bump"] = 0
                track["defect_missing_frames"]["bump"] = 0

    if (
        track["defect_streaks"]["damage"]
        >= DEFECT_CONFIRMATION_FRAMES
    ):
        defects.add("damage")

    if (
        track["defect_streaks"]["bump"]
        >= DEFECT_CONFIRMATION_FRAMES
    ):
        defects.add("bump")

    track["defects"] = sorted(defects)

    confirmed_defect = (
        damage_valid
        and track["defect_streaks"]["damage"]
        >= DEFECT_CONFIRMATION_FRAMES
    ) or (
        bump_valid
        and track["defect_streaks"]["bump"]
        >= DEFECT_CONFIRMATION_FRAMES
    )

    if confirmed_defect and frame is not None:
        track["best_defect_frame"] = frame.copy()
        track["best_defect_box"] = bottle_box
        track["best_defect_label_box"] = label_box

        track["best_defect_damage_boxes"] = [
            dmg for dmg in damage_boxes
            if defect_on_bottle(dmg, current_mask)
        ]

        track["best_defect_bump_boxes"] = [
            bump for bump in bump_boxes
            if defect_on_bottle(bump, current_mask)
        ]

        bx1, by1, bx2, by2 = map(int, bottle_box)
        bw = max(1.0, float(bx2 - bx1))
        bh = max(1.0, float(by2 - by1))

        track["best_defect_damage_relative"] = [
            (
                (dmg[0] - bx1) / bw,
                (dmg[1] - by1) / bh,
                (dmg[2] - bx1) / bw,
                (dmg[3] - by1) / bh,
            )
            for dmg in track["best_defect_damage_boxes"]
        ]

        track["best_defect_bump_relative"] = [
            (
                (bump[0] - bx1) / bw,
                (bump[1] - by1) / bh,
                (bump[2] - bx1) / bw,
                (bump[3] - by1) / bh,
            )
            for bump in track["best_defect_bump_boxes"]
        ]

        track["best_defect_orientation_data"] = orientation_data

        frame_h, frame_w = frame.shape[:2]
        dx1, dy1, dx2, dy2 = map(int, bottle_box)

        bottle_is_complete = (
            dx1 > 0
            and dy1 > 0
            and dx2 < frame_w
            and dy2 < frame_h
        )

        if bottle_is_complete:
            current_area = max(0, dx2 - dx1) * max(0, dy2 - dy1)

            if track["best_complete_defect_box"] is None:
                should_store_complete_defect = True
            else:
                cx1, cy1, cx2, cy2 = map(
                    int,
                    track["best_complete_defect_box"]
                )
                previous_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)

                should_store_complete_defect = (
                    current_area > previous_area
                )

            if should_store_complete_defect:
                track["best_complete_defect_frame"] = frame.copy()
                track["best_complete_defect_box"] = bottle_box
                track["best_complete_defect_label_box"] = label_box

                track["best_complete_defect_damage_boxes"] = [
                    dmg for dmg in damage_boxes
                    if defect_on_bottle(dmg, current_mask)
                ]

                track["best_complete_defect_bump_boxes"] = [
                    bump for bump in bump_boxes
                    if defect_on_bottle(bump, current_mask)
                ]

                if orientation_data is not None:
                    saved_orientation = dict(orientation_data)

                    if orientation_data.get("mask_contour") is not None:
                        saved_orientation["mask_contour"] = [
                            list(point)
                            for point in orientation_data["mask_contour"]
                        ]

                    track[
                        "best_complete_defect_orientation_data"
                    ] = saved_orientation
                else:
                    track[
                        "best_complete_defect_orientation_data"
                    ] = None

def update_best_complete_detection(
    track,
    box,
    frame,
    label_box=None,
    damage_boxes=None,
    bump_boxes=None,
    force=False,
    orientation_data=None,
):
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    if not is_reliable_bottle_frame(box, frame.shape):
        return

    current_area = max(0, x2 - x1) * max(0, y2 - y1)

    if current_area < MIN_BOTTLE_AREA:
        return

    if orientation_data is None and track.get("best_complete_frame") is not None:
        return

    if track["best_complete_box"] is None:
        should_update = True
    else:
        bx1, by1, bx2, by2 = map(int, track["best_complete_box"])
        best_area = max(0, bx2 - bx1) * max(0, by2 - by1)

        should_update = force or (
            current_area > best_area
            or (
                current_area == best_area
                and track.get("best_complete_label_box") is None
                and label_box is not None
            )
        )

    if not should_update:
        return

    track["best_complete_box"] = box
    track["best_complete_frame"] = frame.copy()
    track["best_complete_label_box"] = label_box
    track["best_complete_damage_boxes"] = list(damage_boxes or [])
    track["best_complete_bump_boxes"] = list(bump_boxes or [])

    if orientation_data is not None:
        saved_orientation = dict(orientation_data)

        if orientation_data.get("mask_contour") is not None:
            saved_orientation["mask_contour"] = [
                list(point)
                for point in orientation_data["mask_contour"]
            ]

        track["best_complete_orientation_data"] = saved_orientation
    else:
        track["best_complete_orientation_data"] = None

def update_best_valid_detection(
    track,
    box,
    frame,
    label_box=None,
    damage_boxes=None,
    bump_boxes=None,
    force=False,
    seg_detections=None,
    bottle_mask=None,
):
    if label_box is None:
        return

    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    if not is_reliable_bottle_frame(box, frame.shape):
        return

    label_mask = (
        find_label_mask(box, seg_detections, frame)
        if seg_detections is not None
        else None
    )

    (
        h,
        v,
        h_offset,
        v_offset,
        _bottle_center,
        _label_center,
    ) = check_centricity(
        box,
        label_box,
        bottle_mask=bottle_mask,
        label_mask=label_mask,
    )

    h_error = abs(h_offset)
    v_error = abs(v_offset - EXPECTED_V)

    centricity_error = h_error + v_error

    current_area = max(0, x2 - x1) * max(0, y2 - y1)
    previous_error = track.get("best_valid_centricity_error")

    if force or track["best_valid_box"] is None:
        should_update = True
    elif previous_error is None:
        should_update = True
    else:
        bx1, by1, bx2, by2 = map(int, track["best_valid_box"])
        best_area = max(0, bx2 - bx1) * max(0, by2 - by1)
        should_update = (
            centricity_error < previous_error - 1e-6
            or (abs(centricity_error - previous_error) <= 1e-6 and current_area > best_area)
        )

    if should_update:
        track["best_valid_box"] = box
        track["best_valid_frame"] = None
        track["annotation_label_box"] = label_box
        track["annotation_damage_boxes"] = list(damage_boxes or [])
        track["annotation_bump_boxes"] = list(bump_boxes or [])
        track["best_valid_h_center"] = h
        track["best_valid_v_center"] = v
        track["best_valid_centricity_error"] = centricity_error


def should_save_bottle(track, frame_shape, force=False):
    if track["saved"] or track.get("finalized", False):
        return False

    if not force and track["missing"] < MAX_MISSING_FRAMES:
        return False

    if track["best_complete_box"] is None or track["best_complete_frame"] is None:
        return False

    finalize_measurements(track)
    return True


# -----------------------------
# Folder Frame Source
# -----------------------------

class FolderFrameBuffer:
    def __init__(self, frame):
        self.frame = frame

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class FolderFrameSource:
    def __init__(self, frame_dir):
        self.frame_dir = frame_dir

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
        }

        self.frames = [
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if os.path.splitext(name)[1].lower() in valid_extensions
        ]

        self.frames.sort(
            key=lambda path: [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", os.path.basename(path))
            ]
        )

        self.index = 0

        print(f"Frame folder: {frame_dir}")
        print(f"Frames found: {len(self.frames)}")

        if not self.frames:
            raise RuntimeError(
                f"No image frames found in folder: {frame_dir}"
            )

    def start(self):
        pass

    def stop(self):
        pass

    def fetch(self):
        if self.index >= len(self.frames):
            raise StopIteration

        frame_path = self.frames[self.index]
        self.index += 1

        frame = cv2.imread(frame_path)

        if frame is None:
            raise RuntimeError(
                f"Unable to read frame: {frame_path}"
            )

        return FolderFrameBuffer(frame)

    def destroy(self):
        pass

# -----------------------------
# Camera
# -----------------------------

h_cam = None

if INPUT_MODE == "folder":

    ia = FolderFrameSource(FRAME_DIR)
    ia.start()

    print("Reading frames from folder...")
    print("Press q to exit.")

else:

    h_cam = Harvester()
    h_cam.add_file(CTI_PATH)
    h_cam.update()

    print(f"Devices found: {len(h_cam.device_info_list)}")
    print(f"Bottle images will be saved to: {SAVE_DIR}")

    ia = h_cam.create(1)
    ia.start()

    print("Streaming... Press q to exit.")


# -----------------------------
# Main Loop
# -----------------------------
run_start_time = time.perf_counter()
frames_processed = 0

profiling_detection_time = 0.0
profiling_segmentation_time = 0.0
profiling_ocr_time = 0.0
profiling_post_processing_time = 0.0

ACTIVE_PROFILE_THIS_FRAME = False
active_profile_frames = 0
active_profile_times = {
    "tracking_match": 0.0,
    "mask_lookup": 0.0,
    "orientation": 0.0,
    "centricity": 0.0,
    "defect_state": 0.0,
    "snapshot_selection": 0.0,
    "trigger": 0.0,
    "ocr": 0.0,
    "saving": 0.0,
    "new_track": 0.0,
    "visualization": 0.0,
}

ocr_variant_times = {
    "variant1": 0.0,
    "variants2_4_batch": 0.0,
    "preprocessing": 0.0,
    "parsing": 0.0,
}
ocr_variant_counts = {
    "variant1": 0,
    "variants2_4_batch": 0,
}

ocr_batch_v13 = {
    "preprocessing": 0.0,
    "batch_prepare": 0.0,
    "inference": 0.0,
    "parsing": 0.0,
    "batch_calls": 0,
    "images_total": 0,
    "width_sum": 0,
    "height_sum": 0,
    "width_min": None,
    "width_max": None,
    "height_min": None,
    "height_max": None,
}

ocr_call_diagnostics = {
    "active_frames": 0,
    "capacity_box_frames": 0,
    "ocr_attempt_frames": 0,
    "ocr_skipped_locked_frames": 0,
    "ocr_no_capacity_box_frames": 0,
    "valid_capacity_observations": 0,
    "lock_activations": 0,
    "tracks_seen": set(),
    "track_ocr_calls": {},
    "track_skipped_locked": {},
}

centricity_profile_times = {
    "label_box_match": 0.0,
    "label_mask_lookup": 0.0,
    "centroid_calculation": 0.0,
    "history_and_status": 0.0,
}

orientation_profile_times = {
    "mask_pixels": 0.0,
    "centering_mean": 0.0,
    "centering_demean": 0.0,
    "centering_covariance_ops": 0.0,
    "centering_covariance": 0.0,
    "eigen_angle": 0.0,
    "projection_center": 0.0,
    "projection_dot": 0.0,
    "projection_max": 0.0,
    "projection": 0.0,
    "contour_prepare": 0.0,
    "contour_find": 0.0,
    "contour_select": 0.0,
    "contour": 0.0,
}


def _profile_centricity_section(bucket, fn, *args, **kwargs):
    if not ACTIVE_PROFILE_THIS_FRAME:
        return fn(*args, **kwargs)
    _t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        centricity_profile_times[bucket] += time.perf_counter() - _t0


def _profile_active_call(bucket, fn, *args, **kwargs):
    global active_profile_times
    if not ACTIVE_PROFILE_THIS_FRAME:
        return fn(*args, **kwargs)
    _t0 = time.perf_counter()
    try:
        return fn(*args, **kwargs)
    finally:
        active_profile_times[bucket] += time.perf_counter() - _t0

try:
    while True:

        if INPUT_MODE == "folder":

            try:
                with ia.fetch() as buffer:
                    frame = buffer.frame

            except StopIteration:
                print("All frames have been processed.")
                break

            except RuntimeError as exc:
                # Corrupted/unreadable frame file (§5.2 recovery: report and
                # resume, nothing silently substituted). FolderFrameSource.fetch()
                # already advanced its internal index past the bad file, so the
                # next loop iteration picks up the following frame.
                skipped_frame_count += 1
                print(f"[RECOVERY] Frame skipped (unreadable): {exc}", flush=True)
                continue

        else:

            try:
                with ia.fetch() as buffer:
                    component = buffer.payload.components[0]
                    width = component.width
                    height = component.height
                    pixel_format = component.data_format
                    data = component.data

                    if pixel_format == "Mono8":
                        frame = data.reshape(height, width)
                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_GRAY2BGR
                        )

                    elif pixel_format == "RGB8":
                        frame = data.reshape(height, width, 3)
                        frame = cv2.cvtColor(
                            frame,
                            cv2.COLOR_RGB2BGR
                        )

                    elif pixel_format == "BGR8":
                        frame = data.reshape(height, width, 3)

                    elif pixel_format in (
                        "BayerRG8",
                        "BayerGB8",
                        "BayerGR8",
                        "BayerBG8",
                    ):
                        bayer_map = {
                            "BayerRG8": cv2.COLOR_BayerRG2BGR,
                            "BayerGB8": cv2.COLOR_BayerGB2BGR,
                            "BayerGR8": cv2.COLOR_BayerGR2BGR,
                            "BayerBG8": cv2.COLOR_BayerBG2BGR,
                        }

                        frame = cv2.cvtColor(
                            data.reshape(height, width),
                            bayer_map[pixel_format],
                        )

                    else:
                        continue

            except Exception as exc:
                # Camera fetch/disconnect failure (§5.2 recovery: report and
                # resume rather than crash the process). Not currently exercised
                # in production since this deployment runs --input folder, but
                # kept symmetric with the folder-mode handling above.
                skipped_frame_count += 1
                print(f"[RECOVERY] Camera frame fetch failed, skipping: {exc}", flush=True)
                continue

        frame_start = time.perf_counter()
        frames_processed += 1
        # FIX: Active FPS timer now starts here (same point as Pipeline FPS),
        # so it covers detection + segmentation + post-processing, not just
        # post-processing. Previously it started after both TRT calls,
        # which excluded the two most expensive ops and inflated Active FPS.
        _active_processing_wall_start = frame_start

        _profile_start = time.perf_counter()
        detections = trt_predict(model, frame, min(PER_CLASS_CONF.values()))
        profiling_detection_time += time.perf_counter() - _profile_start

        _profile_start = time.perf_counter()
        seg_detections = trt_predict(seg_model, frame, min(PER_CLASS_CONF.values()), keep_masks=True)
        profiling_segmentation_time += time.perf_counter() - _profile_start

        _post_processing_start = time.perf_counter()
        _ocr_time_before_frame = profiling_ocr_time

        bottle_boxes = []
        capacity_boxes = []
        label_boxes = []
        damage_boxes = []
        bump_boxes = []

        display = frame.copy()

        for i, cls in enumerate(detections.data["class_name"]):
            if float(detections.confidence[i]) < PER_CLASS_CONF.get(cls, THRESHOLD):
                continue
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            print(f"  [BOTTLE DETECT] cls={cls} conf={float(detections.confidence[i]):.3f}")
            if cls == "bottle":
                bottle_boxes.append((x1, y1, x2, y2))
            elif cls == "capacity":
                capacity_boxes.append((x1, y1, x2, y2))
            elif cls == "label":
                label_boxes.append((x1, y1, x2, y2))
            elif cls == "damage":
                damage_boxes.append((x1, y1, x2, y2))
            elif cls == "bump":
                bump_boxes.append((x1, y1, x2, y2))

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, cls, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ACTIVE_PROFILE_THIS_FRAME = bool(bottle_boxes)
        if ACTIVE_PROFILE_THIS_FRAME:
            active_profile_frames += 1
        ocr_call_diagnostics["active_frames"] += 1

        for track in tracked:
            track["missing"] += 1
            track["frames_seen"] += 1

        recently_saved_boxes = [(b, t - 1) for b, t in recently_saved_boxes if t > 1]

        bottle_boxes = deduplicate_boxes(bottle_boxes, iou_thresh=0.65)

        for bottle in bottle_boxes:
            matched = False

            for track in tracked:
                if track.get("finalized", False) or track.get("saved", False):
                    continue
                if _profile_active_call("tracking_match", bottle_track_match, bottle, track["box"]):
                    track["box"] = bottle
                    track["missing"] = 0
                    matched = True
                    track["matched_frames"] = track.get("matched_frames", 0) + 1

                    current_area = max(0, bottle[2] - bottle[0]) * max(0, bottle[3] - bottle[1])
                    best_area = max(0, track["best_box"][2] - track["best_box"][0]) * max(0, track["best_box"][3] - track["best_box"][1])
                    if current_area > best_area:
                        track["best_box"] = bottle
                        track["best_frame"] = None

                    bottle_mask = _profile_active_call("mask_lookup", find_bottle_mask, bottle, seg_detections)

                    if bottle_mask is not None:
                        bottle_mask = resize_mask_to_frame(bottle_mask, frame)
                        bottle_mask = constrain_mask_to_box(bottle_mask, bottle)

                        track["_current_bottle_mask"] = bottle_mask

                        orientation_data = _profile_active_call("orientation", get_mask_orientation, bottle_mask)

                        if orientation_data is not None:
                            track["orientation_data"] = orientation_data
                            track["orientation"] = orientation_data["status"]

                    else:
                        bottle_mask = track.get("_current_bottle_mask")
                        orientation_data = track.get("orientation_data")

                    bx1, by1, bx2, by2 = bottle
                    fh, fw = frame.shape[:2]
                    centricity_updated = False

                    centricity_updated = _profile_active_call(
                        "centricity",
                        update_centricity,
                        track,
                        bottle,
                        label_boxes,
                        seg_detections=seg_detections,
                        bottle_mask=bottle_mask,
                        frame=frame,
                    )

                    if centricity_updated:
                        if track["h_center"] in {"PASS", "FAIL"}:
                            track["h_history"].append(track["h_center"])

                        if track["v_center"] in {"PASS", "FAIL"}:
                            track["v_history"].append(track["v_center"])

                    if track["orientation"] in {"PASS", "FAIL"}:
                        track["orientation_history"].append(track["orientation"])

                    if centricity_updated and track.get("centricity_offset_history"):
                        latest_h, latest_v = track["centricity_offset_history"][-1]
                        track["h_value_history"].append(float(latest_h))
                        track["v_value_history"].append(float(latest_v))

                    if orientation_data is not None:
                        angle_deg = orientation_data.get("angle_deg")
                        if angle_deg is not None:
                            track["orientation_angle_history"].append(
                                float(angle_deg)
                            )

                    current_label_box = get_matching_label_box(bottle, label_boxes)
                    current_damage_boxes = [dmg for dmg in damage_boxes if dmg[0] >= bottle[0] and dmg[2] <= bottle[2] and dmg[1] >= bottle[1] and dmg[3] <= bottle[3]]
                    current_bump_boxes = [bump for bump in bump_boxes if bump[0] >= bottle[0] and bump[2] <= bottle[2] and bump[1] >= bottle[1] and bump[3] <= bottle[3]]

                    previous_defects = set(track["defects"])
                    _profile_active_call(
                        "defect_state",
                        update_defects,
                        track,
                        bottle,
                        current_damage_boxes,
                        current_bump_boxes,
                        frame=frame,
                        label_box=current_label_box,
                        orientation_data=orientation_data,
                        bottle_mask=bottle_mask,
                    )
                    defect_changed = set(track["defects"]) != previous_defects

                    _profile_active_call(
                        "snapshot_selection",
                        update_best_complete_detection,
                        track, bottle, frame,
                        label_box=current_label_box,
                        damage_boxes=current_damage_boxes,
                        bump_boxes=current_bump_boxes,
                        force=defect_changed,
                        orientation_data=orientation_data,
                    )
                    _profile_active_call(
                        "snapshot_selection",
                        update_best_valid_detection,
                        track, bottle, frame,
                        label_box=current_label_box,
                        damage_boxes=current_damage_boxes,
                        bump_boxes=current_bump_boxes,
                        force=defect_changed,
                        seg_detections=seg_detections,
                        bottle_mask=bottle_mask,
                    )

                    observed_capacity = None
                    _cap_tol = 15
                    track_id = track["id"]
                    ocr_call_diagnostics["tracks_seen"].add(track_id)
                    ocr_call_diagnostics["track_ocr_calls"].setdefault(track_id, 0)
                    ocr_call_diagnostics["track_skipped_locked"].setdefault(track_id, 0)

                    matching_capacity_boxes = [
                        cap_box for cap_box in capacity_boxes
                        if (
                            cap_box[0] >= bottle[0] - _cap_tol
                            and cap_box[2] <= bottle[2] + _cap_tol
                            and cap_box[1] >= bottle[1] - _cap_tol
                            and cap_box[3] <= bottle[3] + _cap_tol
                        )
                    ]

                    if matching_capacity_boxes:
                        ocr_call_diagnostics["capacity_box_frames"] += 1

                    cap_box = matching_capacity_boxes[0] if matching_capacity_boxes else None
                    completed_async = _poll_async_ocr(track, cap_box)

                    if completed_async in {100, 300, 500}:
                        observed_capacity = completed_async
                        ocr_call_diagnostics["ocr_attempt_frames"] += 1
                        ocr_call_diagnostics["track_ocr_calls"][track_id] += 1
                        ocr_call_diagnostics["valid_capacity_observations"] += 1
                        track["ocr_cached_capacity"] = completed_async
                        if cap_box is not None:
                            track["ocr_last_box"] = tuple(cap_box)
                        track["ocr_cache_age"] = 0
                        track["ocr_frames_since_inference"] = 0
                        track["ocr_async_pending"] = False
                    elif _ocr_future is not None and not _ocr_future.done():
                        track["ocr_async_pending"] = True

                    if track.get("ocr_capacity_locked", False):
                        ocr_call_diagnostics["ocr_skipped_locked_frames"] += 1
                        ocr_call_diagnostics["track_skipped_locked"][track_id] += 1
                    elif cap_box is None:
                        ocr_call_diagnostics["ocr_no_capacity_box_frames"] += 1
                        track["ocr_cache_age"] = track.get("ocr_cache_age", 0) + 1
                    else:
                        cached_cap = track.get("ocr_cached_capacity")
                        last_box = track.get("ocr_last_box")
                        frames_since_ocr = track.get("ocr_frames_since_inference", 0)
                        temporal_interval = max(1, int(track.get("ocr_temporal_interval", 3)))
                        spatially_stable = (
                            cached_cap in {100, 300, 500}
                            and _ocr_box_is_still_compatible(last_box, cap_box)
                        )
                        force_temporal_refresh = frames_since_ocr >= (temporal_interval - 1)
                        needs_refresh = (
                            cached_cap not in {100, 300, 500}
                            or not spatially_stable
                            or force_temporal_refresh
                        )

                        if cached_cap in {100, 300, 500} and spatially_stable and not needs_refresh:
                            observed_capacity = cached_cap
                            track["ocr_cache_age"] = track.get("ocr_cache_age", 0) + 1
                            track["ocr_frames_since_inference"] = frames_since_ocr + 1
                        elif not track.get("ocr_async_pending", False):
                            if _submit_async_ocr(track, frame, cap_box):
                                track["ocr_async_pending"] = True
                                track["ocr_last_box"] = tuple(cap_box)
                                track["ocr_frames_since_inference"] = 0

                        if observed_capacity is None and cached_cap in {100, 300, 500} and spatially_stable:
                            observed_capacity = cached_cap
                            track["ocr_cache_age"] = track.get("ocr_cache_age", 0) + 1
                            track["ocr_frames_since_inference"] = frames_since_ocr + 1

                    if observed_capacity is not None:
                        track["capacity_history"].append(observed_capacity)

                        if observed_capacity == track.get("ocr_last_capacity"):
                            track["ocr_same_capacity_count"] = track.get(
                                "ocr_same_capacity_count", 0
                            ) + 1
                        else:
                            track["ocr_last_capacity"] = observed_capacity
                            track["ocr_same_capacity_count"] = 1

                        stable_cap = stable_capacity(track["capacity_history"])

                        if (
                            track["ocr_same_capacity_count"] >= 3
                            and not track.get("ocr_capacity_locked", False)
                        ):
                            track["ocr_capacity_locked"] = True
                            ocr_call_diagnostics["lock_activations"] += 1

                        if stable_cap != track.get("capacity"):
                            track["capacity"] = stable_cap
                            print(
                                f"Bottle #{track['id'] + 1} capacity updated: "
                                f"{stable_cap} ml"
                            )

                            with open(LOG_FILE, 'a', newline='') as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    track['id'] + 1,
                                    stable_cap,
                                    (
                                        format_measurement(
                                            track.get("orientation_angle_history", [])[-1]
                                            if track.get("orientation_angle_history")
                                            else None,
                                            track.get("orientation"),
                                            " deg",
                                        )
                                        if track.get("orientation_angle_history")
                                        else track.get("orientation", "Pending")
                                    ),
                                    (
                                        format_measurement(
                                            abs(track["centricity_offset_history"][-1][0]),
                                            track.get("h_center"),
                                        )
                                        if track.get("centricity_offset_history")
                                        else track.get("h_center", "Pending")
                                    ),
                                    (
                                        format_measurement(
                                            abs(track["centricity_offset_history"][-1][1]),
                                            track.get("v_center"),
                                        )
                                        if track.get("centricity_offset_history")
                                        else track.get("v_center", "Pending")
                                    ),
                                    ', '.join(track['defects']) or 'None',
                                    datetime.now().strftime('%H:%M:%S')
                                ])

                    trigger_crossed = _profile_active_call(
                        "trigger",
                        has_crossed_trigger_line,
                        track, bottle, frame.shape[1]
                    )

                    break

            if not matched:
                _dup_mask = find_bottle_mask(bottle, seg_detections)
                if _dup_mask is not None:
                    _dup_mask = resize_mask_to_frame(_dup_mask, frame)
                    _dup_mask = constrain_mask_to_box(_dup_mask, bottle)
                    for _et in tracked:
                        if _et.get("finalized") or _et.get("saved"):
                            continue
                        if mask_iou(_dup_mask, _et.get("_current_bottle_mask")) > MASK_IOU_DUPLICATE_THRESH:
                            print(
                                f"[DUPLICATE-MASK] Bottle detection skipped — mask IoU "
                                f"{mask_iou(_dup_mask, _et.get('_current_bottle_mask')):.2f} "
                                f"with track #{_et['id'] + 1}"
                            )
                            matched = True
                            break
                else:
                    _nx1, _ny1, _nx2, _ny2 = bottle[:4]
                    _ncx, _ncy = (_nx1 + _nx2) / 2, (_ny1 + _ny2) / 2
                    for _et in tracked:
                        if _et.get("finalized") or _et.get("saved"):
                            continue
                        _ex1, _ey1, _ex2, _ey2 = _et["box"][:4]
                        _ecx, _ecy = (_ex1 + _ex2) / 2, (_ey1 + _ey2) / 2
                        _ew, _eh = _ex2 - _ex1, _ey2 - _ey1
                        if (abs(_ncx - _ecx) < 0.25 * _ew and
                                abs(_ncy - _ecy) < 0.25 * _eh):
                            print(
                                f"[DUPLICATE-BOX] Bottle detection skipped — center "
                                f"({_ncx:.0f},{_ncy:.0f}) close to track #{_et['id']+1} "
                                f"center ({_ecx:.0f},{_ecy:.0f})"
                            )
                            matched = True
                            break

                if matched:
                    continue

                _new_cx = (bottle[0] + bottle[2]) / 2
                _is_ghost = False
                if _new_cx < int(frame.shape[1] * TRIGGER_LINE_X_RATIO):
                    for _sb, _ in recently_saved_boxes:
                        _overlap = max(0, min(bottle[3], _sb[3]) - max(bottle[1], _sb[1]))
                        _height = max(bottle[3] - bottle[1], _sb[3] - _sb[1], 1)
                        if _overlap / _height > 0.70:
                            _is_ghost = True
                            print(
                                f"[GHOST] Skipped re-detection of already-saved bottle "
                                f"at ({int(bottle[0])},{int(bottle[1])},{int(bottle[2])},{int(bottle[3])})"
                            )
                            break
                if _is_ghost:
                    continue

                bottle_count += 1
                track = create_track(bottle)
                track["best_frame"] = frame.copy()

                result = analyze_bottle(frame, bottle, capacity_boxes, label_boxes, damage_boxes, bump_boxes)
                track["capacity"] = result["capacity"]
                track["defects"] = result["defects"]

                _new_matching_capacity_boxes = [
                    cap_box for cap_box in capacity_boxes
                    if (
                        cap_box[0] >= bottle[0] - 15
                        and cap_box[2] <= bottle[2] + 15
                        and cap_box[1] >= bottle[1] - 15
                        and cap_box[3] <= bottle[3] + 15
                    )
                ]
                if _new_matching_capacity_boxes:
                    if _submit_async_ocr(track, frame, _new_matching_capacity_boxes[0]):
                        track["ocr_async_pending"] = True
                        track["ocr_last_box"] = tuple(_new_matching_capacity_boxes[0])

                bottle_mask = _profile_active_call("mask_lookup", find_bottle_mask, bottle, seg_detections)

                if bottle_mask is not None:
                    bottle_mask = resize_mask_to_frame(bottle_mask, frame)
                    bottle_mask = constrain_mask_to_box(
                        bottle_mask,
                        bottle
                    )

                    track["_current_bottle_mask"] = bottle_mask

                    orientation_data = _profile_active_call("orientation", get_mask_orientation, bottle_mask)

                    if orientation_data is not None:
                        track["orientation_data"] = orientation_data
                        track["orientation"] = orientation_data["status"]

                else:
                    track["_current_bottle_mask"] = None
                    orientation_data = None

                initial_label_box = get_matching_label_box(bottle, label_boxes)

                centricity_updated = False
                centricity_updated = _profile_active_call(
                    "centricity", update_centricity,
                    track, bottle, label_boxes,
                    seg_detections=seg_detections,
                    bottle_mask=bottle_mask, frame=frame,
                )

                if centricity_updated:
                    if track["h_center"] in {"PASS", "FAIL"}:
                        track["h_history"].append(track["h_center"])

                    if track["v_center"] in {"PASS", "FAIL"}:
                        track["v_history"].append(track["v_center"])

                    if centricity_updated and track.get("centricity_offset_history"):
                        latest_h, latest_v = track["centricity_offset_history"][-1]
                        track["h_value_history"].append(float(latest_h))
                        track["v_value_history"].append(float(latest_v))

                if track["orientation"] in {"PASS", "FAIL"}:
                    track["orientation_history"].append(track["orientation"])

                if orientation_data is not None:
                    angle_deg = orientation_data.get("angle_deg")
                    if angle_deg is not None:
                        track["orientation_angle_history"].append(
                            float(angle_deg)
                        )

                if track["capacity"] is not None:
                    track["capacity_history"].append(track["capacity"])

                initial_damage_boxes = [
                    dmg for dmg in damage_boxes
                    if dmg[0] >= bottle[0]
                    and dmg[2] <= bottle[2]
                    and dmg[1] >= bottle[1]
                    and dmg[3] <= bottle[3]
                ]

                initial_bump_boxes = [
                    bump for bump in bump_boxes
                    if bump[0] >= bottle[0]
                    and bump[2] <= bottle[2]
                    and bump[1] >= bottle[1]
                    and bump[3] <= bottle[3]
                ]

                _profile_active_call(
                    "defect_state", update_defects,
                    track, bottle, initial_damage_boxes, initial_bump_boxes,
                    frame=frame, label_box=initial_label_box,
                    orientation_data=orientation_data,
                )

                update_best_complete_detection(
                    track,
                    bottle,
                    frame,
                    label_box=initial_label_box,
                    damage_boxes=initial_damage_boxes,
                    bump_boxes=initial_bump_boxes,
                    orientation_data=orientation_data
                )
                update_best_valid_detection(
                    track,
                    bottle,
                    frame,
                    label_box=initial_label_box,
                    damage_boxes=initial_damage_boxes,
                    bump_boxes=initial_bump_boxes,
                    seg_detections=seg_detections,
                    bottle_mask=bottle_mask,
                )

                print("=" * 40)
                print(f"Bottle #{bottle_count}")
                print(f"Capacity    : {track['capacity'] or 'Not detected'} ml")
                print(f"Orientation : {track['orientation']}")
                print(f"H Center    : {track['h_center'] or 'Pending'}")
                print(f"V Center    : {track['v_center'] or 'Pending'}")
                print(f"Defects     : {', '.join(track['defects']) or 'None'}")
                print("=" * 40)

                with open(LOG_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        bottle_count,
                        track['capacity'] or 'Not detected',
                        (
                            format_measurement(
                                track.get("orientation_angle_history", [])[-1]
                                if track.get("orientation_angle_history")
                                else None,
                                track.get("orientation"),
                                " deg",
                            )
                            if track.get("orientation_angle_history")
                            else track.get("orientation", "Pending")
                        ),
                        (
                            format_measurement(
                                abs(track["h_value_history"][-1]),
                                track.get("h_center"),
                            )
                            if track.get("h_value_history")
                            else track.get("h_center", "Pending")
                        ),
                        (
                            format_measurement(
                                abs(track["v_value_history"][-1]),
                                track.get("v_center"),
                            )
                            if track.get("v_value_history")
                            else track.get("v_center", "Pending")
                        ),
                        ", ".join(track["defects"]) or "None",
                        datetime.now().strftime("%H:%M:%S"),
                    ])

                tracked.append(track)

        remaining_tracks = []

        for track in tracked:

            if track.get("trigger_crossed", False) and not track["saved"]:
                if should_save_bottle(track, frame.shape, force=True):
                    print(
                        f"Bottle #{track['id'] + 1} crossed trigger line; "
                        f"finalizing stable result: {track['final_status']}"
                    )
                    _profile_active_call("saving", save_bottle_images, frame, track)
                    recently_saved_boxes.append((track["box"], RECENTLY_SAVED_TTL))

            elif track["missing"] >= MAX_MISSING_FRAMES and not track["saved"]:
                if track.get("matched_frames", 0) < MIN_MATCHED_FRAMES_TO_SAVE:
                    print(
                        f"Bottle #{track['id'] + 1} discarded: only matched "
                        f"{track.get('matched_frames', 0)} frame(s) — "
                        f"likely a false detection, not a real bottle."
                    )
                    track["saved"] = True  # drop silently, not counted
                elif should_save_bottle(track, frame.shape, force=True):
                    print(
                        f"Bottle #{track['id'] + 1} disappeared; "
                        f"finalizing stable result: {track['final_status']}"
                    )
                    _profile_active_call("saving", save_bottle_images, frame, track)

            if not track["saved"]:
                remaining_tracks.append(track)

        tracked = remaining_tracks

        _active_visualization_start = time.perf_counter() if ACTIVE_PROFILE_THIS_FRAME else 0.0
        for track in tracked:
            if track["missing"] > 0:
                continue
            x1, y1, x2, y2 = map(int, track["box"])

            bottle_color = (0, 255, 0)
            label_color = (255, 0, 255)
            axis_color = (255, 0, 0)
            centricity_color = (0, 165, 255)
            defect_color = (0, 0, 255)

            cv2.rectangle(display, (x1, y1), (x2, y2), bottle_color, 3)
            cv2.putText(display, f"Bottle #{track['id'] + 1}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bottle_color, 2, cv2.LINE_AA)

            bottle_cx = int((x1 + x2) / 2)
            bottle_cy = int((y1 + y2) / 2)

            if track.get("orientation_data") is not None and track["missing"] == 0:
                draw_mask_contour(
                    display,
                    track["orientation_data"],
                    (0, 255, 0),
                    3,
                    fill=False,
                )
                draw_mask_orientation(
                    display,
                    track["orientation_data"],
                    axis_color,
                    3,
                )

            live_label_box = get_matching_label_box(track["box"], label_boxes)
            if live_label_box is not None:
                lx1, ly1, lx2, ly2 = map(int, live_label_box)
                cv2.rectangle(display, (lx1, ly1), (lx2, ly2), label_color, 3)
                cv2.putText(display, "LABEL", (lx1, max(20, ly1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, label_color, 2, cv2.LINE_AA)

                label_cx = int((lx1 + lx2) / 2)
                label_cy = int((ly1 + ly2) / 2)
                cv2.drawMarker(display, (label_cx, label_cy), label_color, cv2.MARKER_CROSS, 22, 2)
                cv2.line(display, (bottle_cx, bottle_cy), (label_cx, bottle_cy), centricity_color, 3)
                cv2.line(display, (label_cx, bottle_cy), (label_cx, label_cy), centricity_color, 3)
                cv2.line(display, (bottle_cx, bottle_cy), (label_cx, label_cy), (255, 255, 255), 2)

            live_damage_boxes = [dmg for dmg in damage_boxes if dmg[0] >= x1 and dmg[2] <= x2 and dmg[1] >= y1 and dmg[3] <= y2]
            live_bump_boxes = [bump for bump in bump_boxes if bump[0] >= x1 and bump[2] <= x2 and bump[1] >= y1 and bump[3] <= y2]

            for defect_box in live_damage_boxes:
                dx1, dy1, dx2, dy2 = map(int, defect_box)
                cv2.rectangle(display, (dx1, dy1), (dx2, dy2), defect_color, 3)
                cv2.putText(display, "DAMAGE", (dx1, max(20, dy1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

            for defect_box in live_bump_boxes:
                bx1, by1, bx2, by2 = map(int, defect_box)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), defect_color, 3)
                cv2.putText(display, "BUMP", (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, defect_color, 2, cv2.LINE_AA)

            if track.get("final_status") in {"GOOD", "DEFECTIVE", "INCOMPLETE"}:
                status = track["final_status"]
            elif any(
                track[key] in {None, "Pending"}
                for key in ("orientation", "h_center", "v_center")
            ):
                status = "PENDING"
            elif bool(track["defects"]) or any(
                track[key] == "FAIL"
                for key in ("orientation", "h_center", "v_center")
            ):
                status = "DEFECTIVE"
            else:
                status = "GOOD"

            status_color = (
                (0, 0, 255) if status == "DEFECTIVE"
                else (0, 165, 255) if status in {"PENDING", "INCOMPLETE"}
                else (0, 255, 0)
            )
            text_y = min(display.shape[0] - 20, max(25, y1 + 25))

            live_orientation_angle = (
                track["orientation_angle_history"][-1]
                if track.get("orientation_angle_history")
                else None
            )
            live_h_value = (
                abs(track["centricity_offset_history"][-1][0])
                if track.get("centricity_offset_history")
                else None
            )
            live_v_value = (
                abs(track["centricity_offset_history"][-1][1])
                if track.get("centricity_offset_history")
                else None
            )
            live_h_px = track["h_px_history"][-1] if track.get("h_px_history") else None
            live_v_px = track["v_px_history"][-1] if track.get("v_px_history") else None

            info_lines = [
                f"Status: {status}",
                f"Capacity: {track['capacity']} ml" if track["capacity"] else "Capacity: N/A",
                "Orientation: "
                + (
                    format_measurement(
                        live_orientation_angle,
                        track.get("orientation"),
                        " deg",
                    )
                    if live_orientation_angle is not None
                    else (track["orientation"] or "N/A")
                ),
                "H Center: "
                + (
                    format_measurement(live_h_value, track.get("h_center"))
                    + (f" | {live_h_px:.0f}px" if live_h_px is not None else "")
                    if live_h_value is not None
                    else (track["h_center"] or "Pending")
                ),
                "V Center: "
                + (
                    format_measurement(live_v_value, track.get("v_center"))
                    + (f" | {live_v_px:.0f}px" if live_v_px is not None else "")
                    if live_v_value is not None
                    else (track["v_center"] or "Pending")
                ),
                f"Defects: {', '.join(track['defects']) if track['defects'] else 'None'}",
            ]

            for line_index, line in enumerate(info_lines):
                line_color = status_color if line_index == 0 else (255, 255, 255)
                cv2.putText(display, line, (x1 + 8, text_y + line_index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_color, 2, cv2.LINE_AA)

        if ACTIVE_PROFILE_THIS_FRAME:
            active_profile_times["visualization"] = active_profile_times.get("visualization", 0.0) + (time.perf_counter() - _active_visualization_start)

        cv2.putText(
            display,
            f"Total: {completed_count} | Good: {good_count} | "
            f"Defective: {defective_count} | Incomplete: {incomplete_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        _post_processing_elapsed = time.perf_counter() - _post_processing_start
        _frame_ocr_time = profiling_ocr_time - _ocr_time_before_frame
        profiling_post_processing_time += max(
            0.0,
            _post_processing_elapsed - _frame_ocr_time,
        )

        frame_elapsed = time.perf_counter() - frame_start
        if frame_elapsed > 0:
            instant_fps = 1.0 / frame_elapsed
            fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)
        resource_monitor.update()

        # Print both FPS values to the terminal (not just on-screen),
        # throttled to once every 30 frames to avoid log spam.
        if frames_processed % 30 == 0:
            print(
                f"[FPS] Pipeline: {fps:.1f} | Active(bottle-present): {active_fps:.1f}"
            )

        # FIX 3 (clarity only): explicit labels — "PipelineFPS" is the
        # whole-loop FPS (every frame, bottle or not); "ActiveFPS" is the
        # smoothed FPS measured only across frames containing a bottle.
        # Same two numbers as before, clearer labels so they read as
        # genuinely different metrics rather than duplicates.
        _active_fps_text = f" | ActiveFPS: {active_fps:.1f}" if active_fps > 0 else " | ActiveFPS: N/A"
        cv2.putText(
            display,
            resource_monitor.text(fps).replace("FPS:", "PipelineFPS:") + _active_fps_text,
            (10, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        display_resized = cv2.resize(display, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imshow("Frosch Inference", display_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if ACTIVE_PROFILE_THIS_FRAME:
            _active_total_elapsed = time.perf_counter() - _active_processing_wall_start
            active_profile_times["_total_active_wall"] = active_profile_times.get("_total_active_wall", 0.0) + _active_total_elapsed
            if _active_total_elapsed > 0:
                _inst_active_fps = 1.0 / _active_total_elapsed
                active_fps = _inst_active_fps if active_fps == 0.0 else (0.9 * active_fps + 0.1 * _inst_active_fps)
            ACTIVE_PROFILE_THIS_FRAME = False

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    total_runtime = time.perf_counter() - run_start_time
    average_fps = frames_processed / total_runtime if total_runtime > 0 else 0.0

    discarded_incomplete_state = []  # (§5.2 recovery) tracks lost at shutdown w/ no salvage path

    for track in tracked:
        if track["saved"]:
            continue
        if track.get("matched_frames", 0) < MIN_MATCHED_FRAMES_TO_SAVE:
            print(
                f"Bottle #{track['id'] + 1} discarded at shutdown: only matched "
                f"{track.get('matched_frames', 0)} frame(s) — "
                f"likely a false detection, not a real bottle."
            )
            continue
        if track.get("best_complete_frame") is not None:
            if should_save_bottle(track, (0, 0), force=True):
                save_bottle_images(track["best_complete_frame"], track)
            else:
                discarded_incomplete_state.append(track["id"] + 1)
        else:
            discarded_incomplete_state.append(track["id"] + 1)

    if discarded_incomplete_state:
        print(
            f"[RECOVERY] {len(discarded_incomplete_state)} bottle(s) discarded at "
            f"shutdown, in-flight state lost: "
            f"{', '.join(f'#{n}' for n in discarded_incomplete_state)}"
        )

    if skipped_frame_count:
        print(f"[RECOVERY] {skipped_frame_count} frame(s) skipped during this run "
              f"due to read/fetch failure (see [RECOVERY] lines above for detail).")

    print("=" * 40)
    print("FINAL COUNTS")
    print(
        f"Total: {completed_count} | Good: {good_count} | "
        f"Defective: {defective_count} | Incomplete: {incomplete_count} | "
        f"Discarded-at-shutdown: {len(discarded_incomplete_state)} | "
        f"Frames-skipped: {skipped_frame_count}"
    )
    print("=" * 40)

    if ia is not None:
        ia.stop()

    if h_cam is not None:
        h_cam.reset()

    if _ocr_future is not None and not _ocr_future.done():
        try:
            _ocr_future.result()
        except Exception as exc:
            print(f"[WARNING] Async OCR shutdown: {exc}")
    _ocr_executor.shutdown(wait=True)

    resource_monitor.close()
    cv2.destroyAllWindows()
