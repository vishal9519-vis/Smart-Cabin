# modules/risk_engine.py
# AI Cabin Risk Prediction Engine.
#
# Takes outputs from all four monitoring modules and produces:
#   1. A single 0–100 cabin risk score
#   2. A 5-step future risk forecast (next 30 seconds in 5s intervals)
#   3. A safety grade (A–F)
#   4. A prioritised list of active alerts
#
# Fusion strategy:
#   Weighted sum of component risks (weights defined in config).
#   Non-linear amplification: if two or more components are in WARNING state
#   simultaneously, the combined risk gets a multiplicative boost because
#   compound hazards are disproportionately dangerous.
#
# Forecasting:
#   A simple linear extrapolation over the last N frames.
#   Not ML — intentional. For a portfolio demo this is explainable and fast.
#   A real product would run an LSTM over the rolling feature window.

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List
from collections import deque

from utils.config import (
    RISK_WEIGHT_THERMAL, RISK_WEIGHT_OCCUPANCY,
    RISK_WEIGHT_BEHAVIOUR, RISK_WEIGHT_ENV,
)
from utils.logger import log_event


@dataclass
class RiskResult:
    overall_score:  float          # 0–100 fused cabin risk
    safety_grade:   str            # A / B / C / D / F
    alerts:         List[str]      # human-readable alert messages
    forecast:       List[float]    # predicted risk at t+5, +10, +15, +20, +25, +30s
    component_scores: dict         # individual module scores for the dashboard


class RiskEngine:
    """
    Stateful risk fusion engine.
    Call update() each frame with the four module results.
    """

    def __init__(self):
        self._history      = deque(maxlen=100)   # rolling window of risk scores
        self._log_cooldown = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, thermal_result, occupancy_result, behaviour_result, env_result) -> RiskResult:
        components = {
            "thermal":   thermal_result.risk_score,
            "occupancy": occupancy_result.risk_score,
            "behaviour": behaviour_result.risk_score,
            "environment": env_result.risk_score,
        }

        base_risk = (
            components["thermal"]     * RISK_WEIGHT_THERMAL   +
            components["occupancy"]   * RISK_WEIGHT_OCCUPANCY +
            components["behaviour"]   * RISK_WEIGHT_BEHAVIOUR +
            components["environment"] * RISK_WEIGHT_ENV
        )

        # Compound-hazard amplification
        high_count = sum(1 for v in components.values() if v > 60)
        if high_count >= 2:
            base_risk *= (1.0 + 0.12 * high_count)

        overall = min(base_risk, 100.0)
        self._history.append((time.time(), overall))

        alerts   = self._build_alerts(thermal_result, occupancy_result,
                                       behaviour_result, env_result, overall)
        forecast = self._forecast()
        grade    = self._grade(overall)

        self._maybe_log(overall, thermal_result.temperature, occupancy_result.count, alerts)

        return RiskResult(
            overall_score=round(overall, 1),
            safety_grade=grade,
            alerts=alerts,
            forecast=forecast,
            component_scores=components,
        )

    # ── Alert builder ─────────────────────────────────────────────────────────

    def _build_alerts(self, thermal, occupancy, behaviour, env, overall) -> List[str]:
        alerts = []

        if occupancy.child_count > 0 and occupancy.is_unattended:
            alerts.append(f"🚨 CHILD UNATTENDED IN VEHICLE  ({occupancy.unattended_seconds:.0f}s)")

        if thermal.alert_level == "CRITICAL":
            alerts.append(f"🔴 CRITICAL CABIN TEMPERATURE: {thermal.temperature}°C")
        elif thermal.alert_level == "WARNING":
            alerts.append(f"🟠 HIGH CABIN TEMPERATURE: {thermal.temperature}°C")

        if behaviour.distress_flag:
            alerts.append("🚨 POSSIBLE PASSENGER DISTRESS DETECTED")
        elif behaviour.is_inactive:
            alerts.append(f"🟡 PASSENGER INACTIVITY: {behaviour.inactive_seconds:.0f}s")

        if env.alert_level == "CRITICAL":
            alerts.append(f"🔴 UNSAFE AIR QUALITY  CO2: {env.co2_ppm:.0f}ppm")
        elif env.alert_level == "WARNING":
            alerts.append(f"🟠 POOR AIR QUALITY  CO2: {env.co2_ppm:.0f}ppm")

        if occupancy.child_count > 0 and thermal.alert_level != "NORMAL":
            alerts.append("🚨 CHILD + HIGH TEMP — HEATSTROKE RISK")

        if overall >= 80:
            alerts.append("🚨 EMERGENCY: CABIN INTERVENTION REQUIRED")

        return alerts

    # ── Forecasting ───────────────────────────────────────────────────────────

    def _forecast(self) -> List[float]:
        """
        Project risk 30 seconds forward in 5s steps using linear regression
        over the recent history window.
        """
        if len(self._history) < 5:
            last = self._history[-1][1] if self._history else 50.0
            return [round(last, 1)] * 6

        times  = np.array([t for t, _ in self._history])
        scores = np.array([s for _, s in self._history])

        # Normalise time to avoid floating-point instability
        t0    = times[0]
        times = times - t0

        # Least-squares linear fit
        coeffs = np.polyfit(times, scores, 1)
        slope  = coeffs[0]

        last_t  = times[-1]
        last_s  = scores[-1]
        forecast = []
        for step in range(1, 7):   # t+5 through t+30
            projected = last_s + slope * (step * 5)
            forecast.append(round(float(np.clip(projected, 0, 100)), 1))

        return forecast

    # ── Safety grade ──────────────────────────────────────────────────────────

    @staticmethod
    def _grade(score: float) -> str:
        if score < 20:  return "A"
        if score < 40:  return "B"
        if score < 60:  return "C"
        if score < 80:  return "D"
        return "F"

    # ── Logging ───────────────────────────────────────────────────────────────

    def _maybe_log(self, overall, temp, occupancy, alerts):
        now = time.time()
        if overall >= 70 and (now - self._log_cooldown > 15):
            log_event(
                "HIGH_RISK_STATE", "CRITICAL" if overall >= 85 else "WARNING",
                cabin_temp=temp, occupancy=occupancy, risk_score=overall,
                notes=" | ".join(alerts[:2]) if alerts else "",
            )
            self._log_cooldown = now
