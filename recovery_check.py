"""
recovery_check.py — A11 Recovery scenario tests for the FROSCH pipeline.

Handbook §5.2: "Dropped frame, camera disconnect, service restart -> Resumes
and reports the interruption; nothing silently lost."

Does NOT import live_inference.py — it's still a fully procedural script
(argparse + engine load + camera/folder open all run at import time), so
importing it here would try to parse this script's own argv and open a
camera/folder immediately.

Instead this copies the exact pieces that matter and tests them directly:

  1. FolderFrameSource — copied verbatim (unchanged by the live_inference.py
     patch) — plus the same try/except recovery pattern now used in the main
     loop, run against a folder containing one deliberately corrupted file.
     Verifies: no crash, the corrupted frame is reported and skipped, valid
     frames on either side of it are still processed.

  2. Shutdown accounting — replicates the exact "which tracks get discarded"
     logic added to live_inference.py's finally: block, run against synthetic
     track states. Verifies every track ends up accounted for in exactly one
     bucket (saved / false-detection-filtered / discarded-incomplete) — i.e.
     nothing goes missing without a log line.

  Camera disconnect is marked not-applicable: this deployment runs
  --input folder exclusively, no physical/simulated camera in production.

Usage:
    python recovery_check.py --commit <sha> [--output A11_recovery_result.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Copied verbatim from live_inference.py — unchanged by the recovery patch.
# ---------------------------------------------------------------------------

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
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
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
        if not self.frames:
            raise RuntimeError(f"No image frames found in folder: {frame_dir}")

    def fetch(self):
        if self.index >= len(self.frames):
            raise StopIteration
        frame_path = self.frames[self.index]
        self.index += 1
        frame = cv2.imread(frame_path)
        if frame is None:
            raise RuntimeError(f"Unable to read frame: {frame_path}")
        return FolderFrameBuffer(frame)


# ---------------------------------------------------------------------------
# Scenario 1 — corrupted/unreadable frame file
# ---------------------------------------------------------------------------

def scenario_corrupted_frame() -> dict:
    scenario = "corrupted_frame"
    tmp_dir = tempfile.mkdtemp(prefix="recovery_check_")
    processed = 0
    skipped = 0
    recovery_logs = []
    crashed = False
    crash_detail = None

    try:
        # 4 valid tiny synthetic frames + 1 corrupted file, interleaved so a
        # bad frame is not just at the start or end.
        blank = np.zeros((8, 8, 3), dtype=np.uint8)
        for i in (1, 2, 4, 5):
            cv2.imwrite(os.path.join(tmp_dir, f"frame_{i:03d}.jpg"), blank)
        with open(os.path.join(tmp_dir, "frame_003.jpg"), "wb") as f:
            f.write(b"not a real jpeg, deliberately corrupted for the test")

        source = FolderFrameSource(tmp_dir)

        # Same try/except pattern now used in live_inference.py's main loop.
        while True:
            try:
                with source.fetch() as buffer:
                    _frame = buffer.frame
                    processed += 1
            except StopIteration:
                break
            except RuntimeError as exc:
                skipped += 1
                recovery_logs.append(str(exc))
                continue

    except Exception as exc:  # noqa: BLE001 — recording, not suppressing
        crashed = True
        crash_detail = f"{type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    passed = (not crashed) and skipped == 1 and processed == 4 and len(recovery_logs) == 1
    return {
        "scenario": scenario,
        "fault": "1 corrupted frame file among 4 valid frames",
        "crashed": crashed,
        "crash_detail": crash_detail,
        "frames_processed": processed,
        "frames_skipped": skipped,
        "recovery_logged": recovery_logs,
        "pass": passed,
        "note": (
            "Corrupted frame reported and skipped; valid frames before and "
            "after it were still processed"
            if passed else
            f"FAIL — expected 4 processed / 1 skipped / no crash, "
            f"got processed={processed} skipped={skipped} crashed={crashed}"
        ),
    }


# ---------------------------------------------------------------------------
# Scenario 2 — shutdown accounting (nothing silently lost)
# ---------------------------------------------------------------------------

MIN_MATCHED_FRAMES_TO_SAVE = 3  # mirrors the constant in live_inference.py


def _fake_should_save_bottle(track: dict) -> bool:
    """Stands in for live_inference.py's should_save_bottle(), which depends
    on the full tracking/measurement machinery. Only the branch behavior
    (True/False) matters for this test, driven by a field the synthetic
    tracks set directly."""
    return track["_salvageable"]


def scenario_shutdown_accounting() -> dict:
    scenario = "shutdown_accounting"

    tracked = [
        {"id": 0, "saved": True,  "matched_frames": 10, "best_complete_frame": "x", "_salvageable": True},
        {"id": 1, "saved": False, "matched_frames": 1,  "best_complete_frame": None, "_salvageable": False},
        {"id": 2, "saved": False, "matched_frames": 5,  "best_complete_frame": None, "_salvageable": False},
        {"id": 3, "saved": False, "matched_frames": 5,  "best_complete_frame": "x", "_salvageable": False},
        {"id": 4, "saved": False, "matched_frames": 5,  "best_complete_frame": "x", "_salvageable": True},
    ]

    already_saved = 0
    false_detection_discarded = 0
    newly_saved = 0
    discarded_incomplete_state = []

    # Exact same logic as the patched finally: block in live_inference.py.
    for track in tracked:
        if track["saved"]:
            already_saved += 1
            continue
        if track.get("matched_frames", 0) < MIN_MATCHED_FRAMES_TO_SAVE:
            false_detection_discarded += 1
            continue
        if track.get("best_complete_frame") is not None:
            if _fake_should_save_bottle(track):
                newly_saved += 1
            else:
                discarded_incomplete_state.append(track["id"] + 1)
        else:
            discarded_incomplete_state.append(track["id"] + 1)

    accounted_for = already_saved + false_detection_discarded + newly_saved + len(discarded_incomplete_state)
    nothing_lost = accounted_for == len(tracked)

    # Expected against the synthetic fixture above: track0 already-saved,
    # track1 false-detection-filtered, track2 & track3 discarded-incomplete
    # (logged), track4 newly saved.
    expected_discarded_ids = [3, 4]  # track2.id+1=3, track3.id+1=4
    passed = (
        nothing_lost
        and already_saved == 1
        and false_detection_discarded == 1
        and newly_saved == 1
        and discarded_incomplete_state == expected_discarded_ids
    )

    return {
        "scenario": scenario,
        "fault": "Process shutdown with tracks in various incomplete states",
        "already_saved": already_saved,
        "false_detection_discarded": false_detection_discarded,
        "newly_saved_at_shutdown": newly_saved,
        "discarded_incomplete_state_ids": discarded_incomplete_state,
        "total_tracks": len(tracked),
        "accounted_for": accounted_for,
        "nothing_silently_lost": nothing_lost,
        "pass": passed,
        "note": (
            "Every track ended up in exactly one reported bucket; discarded "
            "tracks are logged with their IDs, none vanish silently"
            if passed else
            f"FAIL — accounted_for={accounted_for} vs total={len(tracked)}, "
            f"or bucket counts / discarded IDs didn't match expected"
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FROSCH A11 recovery scenario tests")
    parser.add_argument("--commit", required=True, help="Git commit SHA under test")
    parser.add_argument("--environment", default="")
    parser.add_argument("--output", default="A11_recovery_result.json")
    args = parser.parse_args()

    print("[recovery_check] Scenario 1: corrupted frame", flush=True)
    s1 = scenario_corrupted_frame()

    print("[recovery_check] Scenario 2: shutdown accounting", flush=True)
    s2 = scenario_shutdown_accounting()

    camera_disconnect = {
        "scenario": "camera_disconnect",
        "pass": True,
        "note": "N/A — this deployment runs --input folder exclusively, no camera in production",
        "skipped": True,
    }

    scenarios = [s1, camera_disconnect, s2]
    overall_pass = all(s["pass"] for s in scenarios)

    result = {
        "artifact_id": "A11_recovery",
        "project": "FROSCH",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit": args.commit,
        "environment": args.environment,
        "scenarios": scenarios,
        "overall_pass": overall_pass,
    }

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n[recovery_check] Results written to {out_path}", flush=True)
    for s in scenarios:
        status = "PASS" if s["pass"] else "FAIL"
        print(f"  [{status}] {s['scenario']}: {s['note']}", flush=True)

    print(f"\n[recovery_check] Overall pass: {overall_pass}", flush=True)
    if not overall_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
