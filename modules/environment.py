# modules/environment.py
# Cabin environment quality monitoring.
#
# Monitors: CO2 level (ppm), humidity (%), ventilation score, overall AQI.
# All values are simulated using realistic sensor models.
#
# Real integration path:
#   Replace _read_sensors() with actual I2C/UART sensor reads
#   (MH-Z19B for CO2, DHT22 for humidity) — nothing else changes.

import time
import math
import numpy as np
from dataclasses import dataclass

from utils.config import FRAME_HEIGHT
from utils.helpers import draw_label, normalize_score
from utils.logger import log_event


# Thresholds based on WHO / ASHRAE guidelines
CO2_NORMAL   = 800    # ppm — comfortable
CO2_WARNING  = 1500   # ppm — degraded air quality, drowsiness risk
CO2_CRITICAL = 2500   # ppm — unsafe

HUMIDITY_LOW  = 20    # % — too dry
HUMIDITY_HIGH = 70    # % — too humid


@dataclass
class EnvironmentResult:
    co2_ppm:       float
    humidity_pct:  float
    ventilation:   float   # 0–100 ventilation score
    aqi_score:     float   # 0–100 Air Quality Index (higher = better)
    risk_score:    float   # 0–100 environment risk (higher = worse)
    alert_level:   str     # "NORMAL" | "WARNING" | "CRITICAL"


class EnvironmentMonitor:
    """Simulates multi-sensor cabin environment monitoring."""

    def __init__(self):
        self._start_time     = time.time()
        self._co2_base       = 420.0    # outdoor baseline ~420 ppm
        self._log_cooldown   = 0.0
        self._ventilation_on = False

    # ── Public API ────────────────────────────────────────────────────────────

    def set_ventilation(self, active: bool):
        """Toggle ventilation simulation from dashboard."""
        self._ventilation_on = active

    def process_frame(self, frame) -> EnvironmentResult:
        co2, humidity, vent = self._read_sensors()
        aqi_score  = self._compute_aqi(co2, humidity, vent)
        risk_score = 100.0 - aqi_score
        alert      = self._alert_level(co2)

        self._draw_overlays(frame, co2, humidity, aqi_score)
        self._maybe_log(alert, co2, risk_score)

        return EnvironmentResult(
            co2_ppm=round(co2, 0),
            humidity_pct=round(humidity, 1),
            ventilation=round(vent, 1),
            aqi_score=round(aqi_score, 1),
            risk_score=round(risk_score, 1),
            alert_level=alert,
        )

    # ── Sensor simulation ─────────────────────────────────────────────────────

    def _read_sensors(self):
        """
        Simulate CO2 buildup in an enclosed cabin with N occupants.
        Each person exhales ~200 mL/min CO2.
        Ventilation flushes the cabin toward outdoor baseline.
        """
        elapsed = time.time() - self._start_time

        if self._ventilation_on:
            # Drive back toward outdoor baseline
            self._co2_base = max(420.0, self._co2_base - 0.5)
        else:
            # Gradual CO2 rise + small sine noise for realism
            noise = 8.0 * math.sin(elapsed * 0.3)
            self._co2_base = min(self._co2_base + 0.12 + noise * 0.01, 3000.0)

        humidity   = 45.0 + 12.0 * math.sin(elapsed * 0.05) + np.random.normal(0, 0.5)
        humidity   = max(10.0, min(humidity, 90.0))

        vent_score = 85.0 if self._ventilation_on else max(
            10.0, 85.0 - (elapsed / 10.0)
        )

        return self._co2_base, humidity, min(vent_score, 100.0)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_aqi(self, co2, humidity, ventilation) -> float:
        # CO2 score: 100 at baseline, 0 at critical
        co2_score  = normalize_score(co2, CO2_CRITICAL, CO2_NORMAL)

        # Humidity score: best at 40–60%
        if 40 <= humidity <= 60:
            hum_score = 100.0
        elif humidity < 40:
            hum_score = normalize_score(humidity, HUMIDITY_LOW, 40.0)
        else:
            hum_score = normalize_score(humidity, HUMIDITY_HIGH, 60.0)

        # Weighted AQI
        aqi = (co2_score * 0.55) + (hum_score * 0.20) + (ventilation * 0.25)
        return max(0.0, min(aqi, 100.0))

    def _alert_level(self, co2: float) -> str:
        if co2 >= CO2_CRITICAL:
            return "CRITICAL"
        if co2 >= CO2_WARNING:
            return "WARNING"
        return "NORMAL"

    # ── Overlays ──────────────────────────────────────────────────────────────

    def _draw_overlays(self, frame, co2, humidity, aqi):
        from utils.config import FRAME_WIDTH
        color = (0, 220, 100) if aqi > 60 else (0, 165, 255) if aqi > 30 else (0, 0, 200)
        draw_label(frame, f"CO2: {co2:.0f}ppm  HUM: {humidity:.0f}%  AQI: {aqi:.0f}",
                   10, FRAME_HEIGHT - 165, color=color)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _maybe_log(self, alert, co2, risk_score):
        now = time.time()
        if alert in ("WARNING", "CRITICAL") and (now - self._log_cooldown > 20):
            log_event(f"ENV_{alert}", alert,
                      risk_score=risk_score,
                      notes=f"co2={co2:.0f}ppm")
            self._log_cooldown = now
