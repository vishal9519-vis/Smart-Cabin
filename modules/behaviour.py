# modules/behaviour.py
# Passenger behaviour analysis.
#
# Tracks motion between frames using frame differencing (fast, no extra model).
# If MediaPipe is available it also runs pose estimation for finer-grained
# body language analysis (slumped posture = potential distress signal).
#
# MediaPipe is OPTIONAL — system works fully without it.
# Install: pip install mediapipe  (Python 3.10 / 3.11 only)

import time
import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque

from utils.config import INACTIVITY_SECONDS, FRAME_WIDTH, FRAME_HEIGHT
from utils.helpers import draw_label
from utils.logger import log_event

# ---------- Optional MediaPipe import ----------
_MP_AVAILABLE = False
_mp_pose      = None
_mp_draw      = None

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
    _mp_pose      = mp.solutions.pose
    _mp_draw      = mp.solutions.drawing_utils
    print("[BEHAVIOUR] MediaPipe found — pose estimation enabled")
except ImportError:
    print("[BEHAVIOUR] MediaPipe not installed — pose estimation disabled (system still works)")
except Exception as e:
    print(f"[BEHAVIOUR] MediaPipe import error: {e} — pose estimation disabled")
# -----------------------------------------------


@dataclass
class BehaviourResult:
    motion_level:     float   # 0.0 (no motion) → 1.0 (high motion)
    is_inactive:      bool    # True if no movement for INACTIVITY_SECONDS
    inactive_seconds: float
    distress_flag:    bool    # True if pose + inactivity suggest distress
    risk_score:       float   # 0–100 behaviour risk component


class BehaviourAnalyser:
    """
    Stateful per-frame behaviour analyser.
    Maintains a rolling window of motion levels to detect inactivity.
    """

    def __init__(self):
        self._prev_gray        = None
        self._motion_history   = deque(maxlen=60)   # last 3 seconds at 20fps
        self._last_active_time = time.time()
        self._log_cooldown     = 0.0
        self._pose             = None

        if _MP_AVAILABLE and _mp_pose is not None:
            try:
                self._pose = _mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                print("[BEHAVIOUR] MediaPipe Pose loaded")
            except Exception as e:
                print(f"[BEHAVIOUR] MediaPipe Pose init failed: {e}")
                self._pose = None

    # ── Public API ─────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> BehaviourResult:
        motion_level = self._compute_motion(frame)
        self._motion_history.append(motion_level)
        self._update_inactivity_timer(motion_level)

        inactive_secs = time.time() - self._last_active_time
        is_inactive   = inactive_secs > INACTIVITY_SECONDS

        distress_flag = self._check_distress(frame, is_inactive)
        risk_score    = self._compute_risk(motion_level, is_inactive, inactive_secs)

        self._draw_overlays(frame, motion_level, is_inactive, inactive_secs)
        self._maybe_log(is_inactive, inactive_secs, distress_flag, risk_score)

        return BehaviourResult(
            motion_level=round(motion_level, 3),
            is_inactive=is_inactive,
            inactive_seconds=round(inactive_secs, 1),
            distress_flag=distress_flag,
            risk_score=risk_score,
        )

    # ── Motion detection ────────────────────────────────────────────────────

    def _compute_motion(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0

        diff            = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        nonzero   = cv2.countNonZero(thresh)
        total     = FRAME_WIDTH * FRAME_HEIGHT
        return nonzero / total

    def _update_inactivity_timer(self, motion_level: float):
        if motion_level > 0.003:
            self._last_active_time = time.time()

    # ── Distress check ──────────────────────────────────────────────────────

    def _check_distress(self, frame: np.ndarray, is_inactive: bool) -> bool:
        if not is_inactive:
            return False

        if self._pose is None:
            # No pose model — inactivity alone is a weak distress signal
            return is_inactive

        try:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
        except Exception:
            return is_inactive

        if not results.pose_landmarks:
            return is_inactive

        lm             = results.pose_landmarks.landmark
        nose_y         = lm[_mp_pose.PoseLandmark.NOSE].y
        left_shoulder  = lm[_mp_pose.PoseLandmark.LEFT_SHOULDER].y
        right_shoulder = lm[_mp_pose.PoseLandmark.RIGHT_SHOULDER].y
        shoulder_avg   = (left_shoulder + right_shoulder) / 2
        slumped        = nose_y > shoulder_avg + 0.1

        if slumped and _mp_draw is not None:
            _mp_draw.draw_landmarks(frame, results.pose_landmarks, _mp_pose.POSE_CONNECTIONS)

        return slumped and is_inactive

    # ── Risk scoring ────────────────────────────────────────────────────────

    def _compute_risk(self, motion_level: float, is_inactive: bool, inactive_secs: float) -> float:
        risk = 0.0
        if is_inactive:
            risk += min(50.0, (inactive_secs / INACTIVITY_SECONDS) * 30.0)
        if motion_level > 0.15:
            risk += 25.0
        return min(risk, 100.0)

    # ── Overlays ────────────────────────────────────────────────────────────

    def _draw_overlays(self, frame, motion_level, is_inactive, inactive_secs):
        motion_pct = motion_level * 100
        m_color    = (0, 220, 100) if motion_level < 0.05 else (0, 165, 255)
        draw_label(frame, f"MOTION: {motion_pct:.1f}%", 10, FRAME_HEIGHT - 115, color=m_color)
        if is_inactive:
            draw_label(frame, f"INACTIVITY ALERT  {inactive_secs:.0f}s",
                       10, FRAME_HEIGHT - 140, color=(0, 0, 220))

    def _maybe_log(self, is_inactive, inactive_secs, distress, risk_score):
        now = time.time()
        if distress and (now - self._log_cooldown > 15):
            log_event("PASSENGER_DISTRESS", "CRITICAL",
                      risk_score=risk_score, notes=f"inactive={inactive_secs:.0f}s")
            self._log_cooldown = now
        elif is_inactive and (now - self._log_cooldown > 30):
            log_event("PASSENGER_INACTIVE", "WARNING",
                      risk_score=risk_score, notes=f"inactive={inactive_secs:.0f}s")
            self._log_cooldown = now
