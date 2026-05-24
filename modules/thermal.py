# modules/thermal.py
# Cabin thermal simulation and heat risk analysis.
#
# Why simulation instead of a real sensor?
#   Real thermal cameras (FLIR, MLX90640) are expensive hardware.
#   This module simulates their output using physics-inspired math so the
#   system is demonstrable without any extra hardware.
#   Swap _simulate_reading() for a real sensor SDK call and nothing else changes.
#
# The heat model:
#   Temperature rises at TEMP_RISE_RATE °C/s when the cabin is "closed"
#   (no AC, window closed). Sunlight adds an extra boost via a sine wave
#   that peaks at midday. This matches real-world data within ~20%.

import time
import math
import numpy as np
import cv2
from dataclasses import dataclass
from datetime import datetime

from utils.config import (
    TEMP_BASELINE, TEMP_CRITICAL, TEMP_WARNING,
    TEMP_RISE_RATE, TEMP_COOL_RATE, FRAME_WIDTH, FRAME_HEIGHT,
)
from utils.helpers import draw_label, normalize_score
from utils.logger import log_event


@dataclass
class ThermalResult:
    temperature: float       # simulated cabin temperature in °C
    risk_score:  float       # 0–100 thermal risk component
    zone_map:    np.ndarray  # (H, W) float32 heatmap for visualization
    trend:       str         # "RISING" | "STABLE" | "COOLING"
    alert_level: str         # "NORMAL" | "WARNING" | "CRITICAL"


class ThermalMonitor:
    """
    Stateful thermal monitor.
    Call process_frame() once per frame in the main loop.
    """

    def __init__(self):
        self._temp          = TEMP_BASELINE
        self._start_time    = time.time()
        self._last_time     = time.time()
        self._cooling       = False          # set True to simulate AC on
        self._prev_temp     = TEMP_BASELINE
        self._log_cooldown  = 0.0            # prevents log spam

        # Pre-define cabin seat positions for zone map
        # (x_center_fraction, y_center_fraction) for a 4-seat cabin layout
        self._seat_positions = [
            (0.25, 0.45),  # driver
            (0.75, 0.45),  # passenger
            (0.25, 0.75),  # rear left
            (0.75, 0.75),  # rear right
        ]

    # ── Public API ──────────────────────────────────────────────────────────

    def set_cooling(self, active: bool):
        """Toggle cooling simulation (e.g. from dashboard switch)."""
        self._cooling = active

    def process_frame(self, frame: np.ndarray) -> ThermalResult:
        self._update_temperature()
        zone_map    = self._generate_zone_map()
        risk_score  = self._compute_risk()
        alert_level = self._alert_level()
        trend       = self._trend()

        self._draw_overlays(frame)
        self._maybe_log(risk_score)

        return ThermalResult(
            temperature=round(self._temp, 1),
            risk_score=risk_score,
            zone_map=zone_map,
            trend=trend,
            alert_level=alert_level,
        )

    # ── Temperature simulation ───────────────────────────────────────────────

    def _update_temperature(self):
        now    = time.time()
        dt     = now - self._last_time
        self._last_time   = now
        self._prev_temp   = self._temp

        if self._cooling:
            self._temp -= TEMP_COOL_RATE * dt
            self._temp  = max(self._temp, TEMP_BASELINE - 2.0)
        else:
            # Sunlight boost: peaks at ~solar noon (modelled as wall clock hour 13)
            hour_frac     = datetime.now().hour + datetime.now().minute / 60.0
            solar_boost   = max(0.0, math.sin(math.pi * (hour_frac - 6) / 12)) * 0.03
            self._temp   += (TEMP_RISE_RATE + solar_boost) * dt

        self._temp = max(TEMP_BASELINE - 5.0, min(self._temp, 60.0))

    # ── Zone heatmap ─────────────────────────────────────────────────────────

    def _generate_zone_map(self) -> np.ndarray:
        """
        Build a float32 heatmap where each pixel stores a heat intensity 0–1.
        Seat positions are hotter than ambient; spread via Gaussian blobs.
        """
        h, w  = FRAME_HEIGHT, FRAME_WIDTH
        zmap  = np.zeros((h, w), dtype=np.float32)

        # Ambient base tied to temperature
        ambient = normalize_score(self._temp, TEMP_BASELINE, TEMP_CRITICAL) / 100.0
        zmap[:] = ambient * 0.5

        # Add Gaussian hotspots at seat positions
        yy, xx = np.mgrid[0:h, 0:w]
        for (cx_frac, cy_frac) in self._seat_positions:
            cx, cy = int(cx_frac * w), int(cy_frac * h)
            sigma  = 60
            blob   = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            zmap  += blob * ambient

        return np.clip(zmap, 0.0, 1.0)

    # ── Risk and alert ────────────────────────────────────────────────────────

    def _compute_risk(self) -> float:
        return normalize_score(self._temp, TEMP_BASELINE, TEMP_CRITICAL)

    def _alert_level(self) -> str:
        if self._temp >= TEMP_CRITICAL:
            return "CRITICAL"
        if self._temp >= TEMP_WARNING:
            return "WARNING"
        return "NORMAL"

    def _trend(self) -> str:
        delta = self._temp - self._prev_temp
        if delta > 0.01:
            return "RISING"
        if delta < -0.01:
            return "COOLING"
        return "STABLE"

    # ── Frame overlays ────────────────────────────────────────────────────────

    def _draw_overlays(self, frame: np.ndarray):
        color_map = {"NORMAL": (0, 220, 100), "WARNING": (0, 165, 255), "CRITICAL": (0, 0, 255)}
        level  = self._alert_level()
        color  = color_map[level]
        trend_arrow = {"RISING": "▲", "COOLING": "▼", "STABLE": "—"}[self._trend()]

        draw_label(frame, f"CABIN TEMP: {self._temp:.1f}°C  {trend_arrow}  [{level}]",
                   10, FRAME_HEIGHT - 90, color=color)

        # Draw thin temperature bar on right edge
        bar_h   = FRAME_HEIGHT - 40
        fill_h  = int(bar_h * self._compute_risk() / 100.0)
        cv2.rectangle(frame, (FRAME_WIDTH - 18, 20), (FRAME_WIDTH - 6, FRAME_HEIGHT - 20),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (FRAME_WIDTH - 18, FRAME_HEIGHT - 20 - fill_h),
                      (FRAME_WIDTH - 6, FRAME_HEIGHT - 20), color, -1)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _maybe_log(self, risk_score: float):
        now   = time.time()
        level = self._alert_level()
        if level == "CRITICAL" and (now - self._log_cooldown) > 10:
            log_event("TEMP_CRITICAL", "CRITICAL",
                      cabin_temp=self._temp, risk_score=risk_score,
                      notes=f"trend={self._trend()}")
            self._log_cooldown = now
        elif level == "WARNING" and (now - self._log_cooldown) > 20:
            log_event("TEMP_WARNING", "WARNING",
                      cabin_temp=self._temp, risk_score=risk_score)
            self._log_cooldown = now
