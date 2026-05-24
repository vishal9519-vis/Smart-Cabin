# utils/helpers.py
# Shared utility functions used across multiple modules.
# Keep modules lean — anything reused more than once lives here.

import cv2
import numpy as np
from datetime import datetime


def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height))


def draw_label(frame, text, x, y, color=(0, 255, 100), font_scale=0.52, thickness=1):
    """
    Draw a filled-background text label — always readable regardless of background.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w, h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x - 3, y - h - 4), (x + w + 3, y + baseline), (15, 15, 15), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def draw_alert_banner(frame, text, color=(0, 0, 220)):
    """
    Draw a full-width alert banner at the bottom of the frame.
    Used for CRITICAL alerts so they're impossible to miss.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 38), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"  ⚠  {text}", (8, h - 12), font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)


def timestamp_string():
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def normalize_score(value, min_val, max_val):
    """Map raw value → 0–100 risk score."""
    if max_val == min_val:
        return 0.0
    return clamp((value - min_val) / (max_val - min_val) * 100, 0.0, 100.0)


def generate_thermal_heatmap(frame, temp, zones=None):
    """
    Overlay a temperature-based heatmap on the frame.

    Parameters
    ----------
    frame  : np.ndarray   BGR frame from OpenCV
    temp   : float        Current cabin temperature (drives overall hue)
    zones  : list[dict]   Optional list of {x, y, w, h, intensity} dicts for hot zones

    Returns
    -------
    np.ndarray  Frame with heatmap blended in
    """
    h, w = frame.shape[:2]
    heat_layer = np.zeros((h, w, 3), dtype=np.uint8)

    # Map temperature to a blue→green→red gradient
    # 28°C = cool blue, 36°C = amber, 42°C+ = red
    norm = clamp((temp - 26.0) / (44.0 - 26.0), 0.0, 1.0)

    if norm < 0.5:
        # blue → green
        b = int(255 * (1 - 2 * norm))
        g = int(255 * 2 * norm)
        r = 0
    else:
        # green → red
        b = 0
        g = int(255 * (1 - 2 * (norm - 0.5)))
        r = int(255 * 2 * (norm - 0.5))

    heat_layer[:, :] = (b, g, r)

    # Optionally intensify specific zones (e.g. seat positions)
    if zones:
        for z in zones:
            x1, y1 = z.get("x", 0), z.get("y", 0)
            x2, y2 = x1 + z.get("w", 80), y1 + z.get("h", 80)
            intensity = z.get("intensity", 1.0)
            heat_layer[y1:y2, x1:x2] = np.clip(
                heat_layer[y1:y2, x1:x2] * intensity, 0, 255
            ).astype(np.uint8)

    blended = cv2.addWeighted(frame, 0.6, heat_layer, 0.4, 0)
    return blended
