# modules/occupancy.py
# Occupancy detection using YOLOv8.
#
# What this module does:
#   1. Runs YOLO on every frame to find people
#   2. Applies a height-ratio heuristic to flag likely children
#   3. Tracks how long occupants have been present (unattended timer)
#   4. Returns a structured OccupancyResult every frame
#
# Why YOLO and not a simpler detector?
#   YOLO gives bounding boxes with confidence scores in one pass.
#   The child heuristic (bounding box height / frame height) is an approximation
#   good enough for demo/portfolio; a production system would use a trained
#   child-vs-adult classifier or seat-height calibration.

import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

from utils.config import (
    YOLO_MODEL, CONFIDENCE_THRESH,
    CHILD_HEIGHT_RATIO, UNATTENDED_SECONDS,
    FRAME_WIDTH, FRAME_HEIGHT,
)
from utils.helpers import draw_label
from utils.logger import log_event

# Lazy import so the app doesn't crash if ultralytics isn't installed yet
try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[OCCUPANCY] ultralytics not found — running in simulation mode")


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    is_child: bool


@dataclass
class OccupancyResult:
    count: int                        # total persons detected
    child_count: int                  # persons flagged as children
    detections: List[Detection]       # full list of bounding boxes
    unattended_seconds: float         # how long occupants have been present
    is_unattended: bool               # True if unattended_seconds > threshold
    risk_score: float                 # 0–100 occupancy risk component


class OccupancyDetector:
    """
    Stateful occupancy detector.
    Create once, call process_frame() in the main loop.
    """

    def __init__(self):
        self.model = None
        self._first_detection_time = None   # wall clock when occupants first appeared
        self._last_seen_time       = None

        if _YOLO_AVAILABLE:
            try:
                self.model = YOLO(YOLO_MODEL)
                print(f"[OCCUPANCY] YOLO model loaded: {YOLO_MODEL}")
            except Exception as e:
                print(f"[OCCUPANCY] YOLO load failed ({e}) — simulation mode")
        else:
            print("[OCCUPANCY] Running in simulation mode (no YOLO)")

    # ── Public API ──────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> OccupancyResult:
        """
        Run detection on one frame and return an OccupancyResult.
        Falls back to simulated data if YOLO is unavailable.
        """
        if self.model is not None:
            result = self._yolo_detect(frame)
        else:
            result = self._simulate(frame)

        self._update_timers(result)
        self._draw_overlays(frame, result)
        self._maybe_log(result)
        return result

    # ── Detection logic ─────────────────────────────────────────────────────

    def _yolo_detect(self, frame: np.ndarray) -> OccupancyResult:
        results = self.model(frame, verbose=False, conf=CONFIDENCE_THRESH, classes=[0])
        # class 0 in COCO = "person"

        detections = []
        h, w = frame.shape[:2]

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf     = float(box.conf[0])
            box_h    = y2 - y1
            is_child = (box_h / h) < CHILD_HEIGHT_RATIO

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                is_child=is_child,
            ))

        return self._build_result(detections)

    def _simulate(self, frame: np.ndarray) -> OccupancyResult:
        """
        Produce synthetic detections for demo/testing without a camera.
        Simulates one adult + one child with slightly moving bounding boxes.
        """
        t = time.time()
        jitter = int(5 * np.sin(t * 0.5))   # tiny movement so it looks alive

        h, w = frame.shape[:2]
        detections = [
            Detection(
                bbox=(100 + jitter, 80, 260 + jitter, 380),
                confidence=0.87,
                is_child=False,
            ),
            Detection(
                bbox=(320 + jitter, 160, 430 + jitter, 370),
                confidence=0.79,
                is_child=True,   # shorter box → child heuristic fires
            ),
        ]
        return self._build_result(detections)

    def _build_result(self, detections: List[Detection]) -> OccupancyResult:
        count       = len(detections)
        child_count = sum(1 for d in detections if d.is_child)

        unattended_secs = 0.0
        if count > 0 and self._first_detection_time is not None:
            unattended_secs = time.time() - self._first_detection_time

        is_unattended = unattended_secs > UNATTENDED_SECONDS

        # Risk score: child presence and unattended duration both raise it
        risk = 0.0
        if count > 0:
            risk += 30.0
        if child_count > 0:
            risk += 40.0
        if is_unattended:
            risk += min(30.0, unattended_secs / UNATTENDED_SECONDS * 30.0)

        return OccupancyResult(
            count=count,
            child_count=child_count,
            detections=detections,
            unattended_seconds=round(unattended_secs, 1),
            is_unattended=is_unattended,
            risk_score=min(risk, 100.0),
        )

    # ── Timers ──────────────────────────────────────────────────────────────

    def _update_timers(self, result: OccupancyResult):
        now = time.time()
        if result.count > 0:
            if self._first_detection_time is None:
                self._first_detection_time = now
            self._last_seen_time = now
        else:
            # Reset timer only after cabin is empty for >5 seconds
            if self._last_seen_time and (now - self._last_seen_time > 5):
                self._first_detection_time = None
                self._last_seen_time       = None

    # ── Frame overlays ───────────────────────────────────────────────────────

    def _draw_overlays(self, frame: np.ndarray, result: OccupancyResult):
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            color = (0, 80, 255) if det.is_child else (0, 220, 100)
            label = f"CHILD  {det.confidence:.0%}" if det.is_child else f"ADULT  {det.confidence:.0%}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1 - 6, color=color)

        if result.is_unattended:
            draw_label(frame,
                       f"UNATTENDED  {result.unattended_seconds:.0f}s",
                       10, FRAME_HEIGHT - 60,
                       color=(0, 0, 255))

    # ── Event logging ─────────────────────────────────────────────────────────

    def _maybe_log(self, result: OccupancyResult):
        if result.child_count > 0 and result.is_unattended:
            log_event(
                "CHILD_UNATTENDED", "CRITICAL",
                occupancy=result.count,
                risk_score=result.risk_score,
                notes=f"child_count={result.child_count} duration={result.unattended_seconds}s",
            )
        elif result.child_count > 0:
            log_event(
                "CHILD_DETECTED", "WARNING",
                occupancy=result.count,
                risk_score=result.risk_score,
            )
