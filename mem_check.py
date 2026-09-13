"""
mem_check.py — A11 Soak Test runner for FROSCH Bottle Inspection Pipeline.

Self-contained: does NOT import the live pipeline module (which is fully
procedural — argparse/engine-load/camera-open all run at import time).
Instead this copies the NativeTRTEngine + trt_predict inference path
directly, so only TensorRT engines + a frame source are needed.

Handbook §5.2 / §7.3.3:
  - Duration: D5 = 2 h continuous (use --trial for a short sanity check first)
  - Max memory growth over soak: D4 = 0 (must plateau)
  - Frame mix MUST include detection-failure frames (no bottle / out-of-frame)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import supervision as sv
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

CLASS_NAMES = [
    "Frosch-bottle-UTNY-aUbJ-XBXs",
    "bottle",
    "bump",
    "capacity",
    "damage",
    "label",
    "scratch",
]


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
        self._input_tensor = torch.empty(
            (1, 3, self.input_h, self.input_w), dtype=torch.float32, device="cuda"
        )
        self._output_tensors = {}
        for name in self.output_names:
            out_shape = tuple(self.engine.get_tensor_shape(name))
            if any(d < 0 for d in out_shape):
                raise RuntimeError(f"Dynamic output shape unsupported: {name} {out_shape}")
            np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = {
                np.float32: torch.float32,
                np.float16: torch.float16,
                np.int32: torch.int32,
                np.int64: torch.int64,
            }.get(np_dtype)
            if torch_dtype is None:
                raise RuntimeError(f"Unsupported TensorRT output dtype: {name} {np_dtype}")
            self._output_tensors[name] = torch.empty(out_shape, dtype=torch_dtype, device="cuda")
        self.context.set_tensor_address(self.input_name, self._input_tensor.data_ptr())
        for name, out in self._output_tensors.items():
            self.context.set_tensor_address(name, out.data_ptr())
        print(f"[mem_check] TensorRT loaded: {engine_path} | input={self.input_h}x{self.input_w}")

    def infer(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        normalized -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        normalized /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        chw = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))
        with torch.cuda.stream(self.stream):
            self._input_tensor.copy_(torch.from_numpy(chw).unsqueeze(0), non_blocking=True)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")
        self.stream.synchronize()
        return {name: t.detach().float().cpu().numpy() for name, t in self._output_tensors.items()}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def decode_rfdetr_outputs(raw, frame_shape, score_threshold, keep_masks=False):
    if "dets" not in raw or "labels" not in raw:
        raise RuntimeError(f"Expected dets/labels outputs, got {list(raw)}")
    boxes_cwh = raw["dets"][0]
    logits = raw["labels"][0][:, :-1]
    probs = _sigmoid(logits)
    flat = probs.reshape(-1)
    k = min(boxes_cwh.shape[0], flat.size)
    order = np.argsort(-flat, kind="stable")[:k]
    scores = flat[order]
    num_classes = probs.shape[1]
    query_idx = order // num_classes
    class_ids = order % num_classes
    keep = scores > score_threshold
    scores, query_idx, class_ids = scores[keep], query_idx[keep], class_ids[keep]
    boxes = boxes_cwh[query_idx]
    h, w = frame_shape[:2]
    cx, cy, bw, bh = boxes.T
    xyxy = np.stack(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], axis=1
    )
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, w)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, h)
    detections = sv.Detections(
        xyxy=xyxy.astype(np.float32),
        confidence=scores.astype(np.float32),
        class_id=class_ids.astype(int),
    )
    detections.data["class_name"] = np.array(
        [CLASS_NAMES[int(c)] if int(c) < len(CLASS_NAMES) else f"class_{int(c)}" for c in class_ids],
        dtype=object,
    )
    if keep_masks and "masks" in raw:
        raw_masks = raw["masks"][0][query_idx]
        mask_tensor = torch.from_numpy(raw_masks).float().unsqueeze(1)
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(1)
        detections.mask = (mask_tensor.sigmoid() > 0.5).cpu().numpy()
    return detections


def trt_predict(runtime_model, frame, threshold, keep_masks=False):
    raw = runtime_model.infer(frame)
    return decode_rfdetr_outputs(raw, frame.shape, threshold, keep_masks=keep_masks)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FROSCH A11 soak test — mem_check.py")
    p.add_argument("--source", default="0")
    p.add_argument("--input", choices=("video", "camera", "folder"), default="video")
    p.add_argument("--frame-dir", default=None)
    p.add_argument("--frame-dirs", nargs="+", default=None)
    p.add_argument("--preload", action="store_true")
    p.add_argument("--preload-limit", type=int, default=200)
    p.add_argument("--expected-capacity", type=int, choices=(100, 300, 500), default=None)
    p.add_argument("--expected-capacities", nargs="+", type=int, default=None)
    p.add_argument("--model", default="output/rfdetr-medium.trt")
    p.add_argument("--seg-model", default="output/rfdetr-seg-medium.trt")
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--duration", type=int, default=7200)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--fail-ratio", type=float, default=0.10)
    p.add_argument("--output", default="A11_soak_result.json")
    p.add_argument("--memory-ceiling", type=float, default=2048.0)
    p.add_argument("--latency-budget", type=float, default=100.0)
    p.add_argument("--throughput-min", type=float, default=25.0)
    p.add_argument("--commit", default="")
    p.add_argument("--environment", default="")
    p.add_argument("--trial", action="store_true")
    return p


class FrameSource:
    def __init__(self, source: str, fail_ratio: float, frame_h: int = 480, frame_w: int = 640):
        self._fail_ratio = fail_ratio
        self._blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        try:
            idx = int(source)
            self._cap = cv2.VideoCapture(idx)
        except ValueError:
            self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")
        self._frame_n = 0
        print(f"[mem_check] Video source opened: {source}", flush=True)

    def next_frame(self) -> np.ndarray:
        self._frame_n += 1
        if np.random.random() < self._fail_ratio:
            return self._blank.copy()
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("Video source exhausted and cannot rewind.")
        return frame

    def close(self) -> None:
        self._cap.release()


class FolderFrameSource:
    _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    def __init__(self, frame_dir: str, fail_ratio: float, preload: bool = False,
                 preload_limit: int = 200, frame_h: int = 480, frame_w: int = 640):
        self._fail_ratio = fail_ratio
        self._blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        self._preload = preload
        import re
        self._paths = [
            os.path.join(frame_dir, name)
            for name in os.listdir(frame_dir)
            if os.path.splitext(name)[1].lower() in self._EXTS
        ]
        self._paths.sort(
            key=lambda p: [int(t) if t.isdigit() else t.lower()
                            for t in re.split(r"(\d+)", os.path.basename(p))]
        )
        if not self._paths:
            raise RuntimeError(f"No image frames found in folder: {frame_dir}")
        self._frames = None
        if preload:
            n = len(self._paths)
            if n > preload_limit:
                sample_idx = np.linspace(0, n - 1, preload_limit).astype(int)
                load_paths = [self._paths[i] for i in sample_idx]
            else:
                load_paths = self._paths
            print(f"[mem_check] Preloading {len(load_paths)}/{n} frames into RAM "
                  f"(--preload-limit={preload_limit}) ...", flush=True)
            frames = []
            for p in load_paths:
                frame = cv2.imread(p)
                if frame is None:
                    raise RuntimeError(f"Unable to read frame: {p}")
                frames.append(frame)
            self._frames = frames
            self._paths = load_paths
            print(f"[mem_check] Preload done ({len(self._frames)} frames).", flush=True)
        self._index = 0
        print(f"[mem_check] Frame folder opened: {frame_dir} ({len(self._paths)} frames, "
              f"preload={preload})", flush=True)

    def next_frame(self) -> np.ndarray:
        if np.random.random() < self._fail_ratio:
            return self._blank.copy()
        if self._index >= len(self._paths):
            self._index = 0
        if self._preload:
            frame = self._frames[self._index]
            self._index += 1
            return frame
        path = self._paths[self._index]
        self._index += 1
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Unable to read frame: {path}")
        return frame

    def close(self) -> None:
        pass


class MultiFolderFrameSource:
    def __init__(self, frame_dirs, fail_ratio, preload=False, preload_limit=200):
        self._sources = [
            FolderFrameSource(d, fail_ratio, preload=preload, preload_limit=preload_limit)
            for d in frame_dirs
        ]
        self._i = 0
        print(f"[mem_check] Cycling {len(self._sources)} frame folders round-robin: "
              f"{frame_dirs}", flush=True)

    def next_frame(self) -> np.ndarray:
        src = self._sources[self._i]
        self._i = (self._i + 1) % len(self._sources)
        return src.next_frame()

    def close(self) -> None:
        for s in self._sources:
            s.close()


def make_frame_source(args):
    if args.frame_dirs:
        return MultiFolderFrameSource(args.frame_dirs, args.fail_ratio, preload=args.preload,
                                       preload_limit=args.preload_limit)
    if args.input == "folder":
        if not args.frame_dir:
            raise ValueError("--frame-dir is required when --input folder (or use --frame-dirs)")
        return FolderFrameSource(args.frame_dir, args.fail_ratio, preload=args.preload,
                                  preload_limit=args.preload_limit)
    return FrameSource(args.source, args.fail_ratio)


class MemorySampler:
    def __init__(self) -> None:
        self._proc = psutil.Process(os.getpid())

    def rss_mb(self) -> float:
        return self._proc.memory_info().rss / (1024 ** 2)


def run_soak(args: argparse.Namespace) -> dict:
    if not args.commit and not args.trial:
        raise ValueError("--commit is required for A11 traceability (or pass --trial).")
    commit = args.commit or "trial"

    model = NativeTRTEngine(args.model)
    seg_model = NativeTRTEngine(args.seg_model)

    source = make_frame_source(args)
    sampler = MemorySampler()
    baseline_rss_mb = sampler.rss_mb()
    baseline_vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)

    frame_interval_s = 1.0 / args.fps
    end_time = time.monotonic() + args.duration
    total_frames = 0

    cap_estimate = max(1024, int(args.duration * args.fps * 1.05))
    rss_buf = np.empty(cap_estimate, dtype=np.float32)
    lat_buf = np.empty(cap_estimate, dtype=np.float32)
    vram_buf = np.empty(cap_estimate, dtype=np.float32)

    print(f"[mem_check] {'TRIAL' if args.trial else 'Soak'} started. "
          f"Duration={args.duration}s, FPS target={args.fps}", flush=True)
    report_interval = max(1, int(args.fps * 10 if args.trial else args.fps * 60))
    peak_lat_holder = {"val": 0.0}

    try:
        while time.monotonic() < end_time:
            frame = source.next_frame()
            t0 = time.perf_counter()
            trt_predict(model, frame, args.threshold)
            trt_predict(seg_model, frame, args.threshold, keep_masks=True)
            latency_ms = (time.perf_counter() - t0) * 1000.0

            if total_frames >= cap_estimate:
                rss_buf = np.concatenate([rss_buf, np.empty(cap_estimate, dtype=np.float32)])
                lat_buf = np.concatenate([lat_buf, np.empty(cap_estimate, dtype=np.float32)])
                vram_buf = np.concatenate([vram_buf, np.empty(cap_estimate, dtype=np.float32)])
                cap_estimate = len(rss_buf)

            lat_buf[total_frames] = latency_ms
            rss_buf[total_frames] = sampler.rss_mb()
            vram_buf[total_frames] = torch.cuda.memory_allocated() / (1024 ** 2)

            if latency_ms > peak_lat_holder["val"]:
                peak_lat_holder["val"] = latency_ms
                src_idx = getattr(source, "_i", None)
                src_name = None
                if src_idx is not None and hasattr(source, "_sources"):
                    folder_src = source._sources[src_idx]
                    if hasattr(folder_src, "_index") and hasattr(folder_src, "_paths"):
                        pos = (folder_src._index - 1) % max(1, len(folder_src._paths))
                        src_name = folder_src._paths[pos]
                print(
                    f"[mem_check] NEW PEAK latency={latency_ms:.2f}ms at frame={total_frames} "
                    f"folder_idx={src_idx} frame_file={src_name}",
                    flush=True,
                )

            total_frames += 1

            elapsed = time.perf_counter() - t0
            sleep_s = frame_interval_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

            if total_frames % report_interval == 0:
                remaining = end_time - time.monotonic()
                elapsed_total = args.duration - remaining
                print(
                    f"[mem_check] t={elapsed_total:.0f}s frames={total_frames} "
                    f"rss={rss_buf[total_frames - 1]:.1f}MB lat={latency_ms:.1f}ms",
                    flush=True,
                )
    finally:
        source.close()

    rss_samples = rss_buf[:total_frames]
    latencies_ms = lat_buf[:total_frames]
    vram_samples = vram_buf[:total_frames]

    return _compute_results(args, commit, baseline_rss_mb, rss_samples, latencies_ms,
                             total_frames, baseline_vram_mb, vram_samples)


def _compute_results(args, commit, baseline_rss_mb, rss_samples, latencies_ms, total_frames,
                      baseline_vram_mb=0.0, vram_samples=None) -> dict:
    n = len(rss_samples)
    if n < 10:
        raise RuntimeError(f"Too few RSS samples ({n}) to compute M3.")

    tenth = max(1, n // 10)
    first_peak = float(np.max(rss_samples[:tenth]))
    final_peak = float(np.max(rss_samples[n - tenth:]))
    memory_growth_mb = round(final_peak - first_peak, 3)
    peak_rss_mb = round(float(np.max(rss_samples)), 3)
    peak_growth_above_baseline_mb = round(peak_rss_mb - baseline_rss_mb, 3)

    median_growth_mb = round(
        float(np.median(rss_samples[n - tenth:])) - float(np.median(rss_samples[:tenth])), 3
    )

    vram_growth_mb = 0.0
    peak_vram_mb = 0.0
    if vram_samples is not None and len(vram_samples) >= 10:
        vt = max(1, len(vram_samples) // 10)
        vram_growth_mb = round(
            float(np.max(vram_samples[len(vram_samples) - vt:])) - float(np.max(vram_samples[:vt])), 3
        )
        peak_vram_mb = round(float(np.max(vram_samples)), 3)

    latencies_sorted = np.sort(latencies_ms)
    p95_idx = int(len(latencies_sorted) * 0.95)
    latency_p95_ms = round(float(latencies_sorted[p95_idx]), 3)
    latency_max_ms = round(float(latencies_sorted[-1]), 3)

    throughput_fps = round(total_frames / args.duration, 3)
    growth_gate = True if args.trial else (memory_growth_mb <= 0)

    passes = {
        "D4_memory_growth": growth_gate,
        "memory_ceiling": peak_growth_above_baseline_mb < args.memory_ceiling,
        "latency_p95": latency_p95_ms < args.latency_budget,
        "latency_max": latency_max_ms < args.latency_budget,
        "throughput": throughput_fps >= args.throughput_min,
    }
    overall_pass = all(passes.values())

    return {
        "artifact_id": "A11_trial" if args.trial else "A11",
        "project": "FROSCH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "environment": args.environment,
        "expected_capacity_ml": args.expected_capacities if args.frame_dirs else args.expected_capacity,
        "frame_dirs": args.frame_dirs if args.frame_dirs else args.frame_dir,
        "duration_s": args.duration,
        "total_frames": total_frames,
        "fail_frame_ratio": args.fail_ratio,
        "memory": {
            "baseline_rss_mb": round(baseline_rss_mb, 3),
            "first_tenth_peak_mb": round(first_peak, 3),
            "final_tenth_peak_mb": round(final_peak, 3),
            "growth_mb_M3": memory_growth_mb,
            "median_growth_mb_supplementary": median_growth_mb,
            "peak_rss_mb": peak_rss_mb,
            "peak_growth_above_baseline_mb": peak_growth_above_baseline_mb,
            "ceiling_mb": args.memory_ceiling,
            "ceiling_gate_basis": "growth above post-preload baseline, not absolute RSS",
            "note": "D4 growth gate not enforced in --trial mode (window too short)" if args.trial else "",
        },
        "vram": {
            "baseline_mb": round(baseline_vram_mb, 3),
            "growth_mb": vram_growth_mb,
            "peak_mb": peak_vram_mb,
        },
        "latency": {"p95_ms": latency_p95_ms, "max_ms": latency_max_ms, "budget_ms": args.latency_budget},
        "throughput": {"fps": throughput_fps, "min_required_fps": args.throughput_min},
        "leak_reports": "see_lsan_valgrind_log",
        "recovery": "see_recovery_check_json",
        "gates": passes,
        "overall_pass": overall_pass,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.trial and args.duration == 7200:
        args.duration = 120
    result = run_soak(args)
    out_path = Path(args.output if args.output != "A11_soak_result.json" or not args.trial
                     else "A11_trial_result.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"\n[mem_check] Results written to {out_path}", flush=True)
    print(f"[mem_check] Overall pass: {result['overall_pass']}", flush=True)
    print(f"[mem_check] M3 memory growth: {result['memory']['growth_mb_M3']} MB", flush=True)
    print(f"[mem_check] VRAM growth: {result['vram']['growth_mb']} MB", flush=True)
    print(f"[mem_check] p95 latency: {result['latency']['p95_ms']} ms", flush=True)
    print(f"[mem_check] max latency: {result['latency']['max_ms']} ms", flush=True)
    print(f"[mem_check] Throughput: {result['throughput']['fps']} fps", flush=True)
    if not result["overall_pass"]:
        failed = [k for k, v in result["gates"].items() if not v]
        print(f"[mem_check] FAILED gates: {failed}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
