# tests/test_modules.py
# Unit tests for Smart Cabin AI modules.
# Run:  python -m pytest tests/ -v

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from utils.helpers   import normalize_score, clamp, generate_thermal_heatmap
from modules.thermal import ThermalMonitor
from modules.environment import EnvironmentMonitor
from modules.risk_engine import RiskEngine

DUMMY_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


# ── helpers ───────────────────────────────────────────────────────────────────

def test_normalize_score_midpoint():
    assert normalize_score(50, 0, 100) == pytest.approx(50.0)

def test_normalize_score_clamp_low():
    assert normalize_score(-10, 0, 100) == 0.0

def test_normalize_score_clamp_high():
    assert normalize_score(110, 0, 100) == 100.0

def test_clamp():
    assert clamp(5, 0, 10)  == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


# ── thermal ───────────────────────────────────────────────────────────────────

def test_thermal_returns_result():
    mon = ThermalMonitor()
    result = mon.process_frame(DUMMY_FRAME.copy())
    assert 0.0 <= result.risk_score <= 100.0
    assert result.alert_level in ("NORMAL", "WARNING", "CRITICAL")
    assert result.trend in ("RISING", "STABLE", "COOLING")
    assert result.zone_map.shape == (480, 640)

def test_thermal_cooling_reduces_risk():
    mon = ThermalMonitor()
    # Let temp rise a little first
    for _ in range(10):
        mon.process_frame(DUMMY_FRAME.copy())
    hot_score = mon.process_frame(DUMMY_FRAME.copy()).risk_score

    mon.set_cooling(True)
    import time; time.sleep(0.5)
    cold_score = mon.process_frame(DUMMY_FRAME.copy()).risk_score
    # Risk should be lower or equal after cooling
    assert cold_score <= hot_score + 2.0   # +2 tolerance for timing


# ── environment ───────────────────────────────────────────────────────────────

def test_environment_returns_result():
    mon = EnvironmentMonitor()
    result = mon.process_frame(DUMMY_FRAME.copy())
    assert 0 <= result.aqi_score <= 100
    assert result.alert_level in ("NORMAL", "WARNING", "CRITICAL")
    assert result.co2_ppm > 0


# ── risk engine ───────────────────────────────────────────────────────────────

def test_risk_engine_grade_mapping():
    engine = RiskEngine()

    class FakeResult:
        pass

    def make_results(risk):
        t = FakeResult(); t.risk_score = risk; t.temperature = 30.0; t.alert_level = "NORMAL"; t.trend = "STABLE"; t.zone_map = np.zeros((480, 640))
        o = FakeResult(); o.risk_score = risk; o.count = 1; o.child_count = 0; o.is_unattended = False; o.unattended_seconds = 0
        b = FakeResult(); b.risk_score = risk; b.motion_level = 0.01; b.is_inactive = False; b.inactive_seconds = 0; b.distress_flag = False
        e = FakeResult(); e.risk_score = risk; e.co2_ppm = 500; e.aqi_score = 80; e.alert_level = "NORMAL"
        return t, o, b, e

    result_a = engine.update(*make_results(0))
    assert result_a.safety_grade == "A"

    result_f = engine.update(*make_results(95))
    assert result_f.safety_grade == "F"

def test_forecast_length():
    engine = RiskEngine()
    mon    = ThermalMonitor()
    from modules.occupancy   import OccupancyDetector
    from modules.behaviour   import BehaviourAnalyser
    from modules.environment import EnvironmentMonitor

    occ  = OccupancyDetector()
    beh  = BehaviourAnalyser()
    env  = EnvironmentMonitor()

    for _ in range(10):
        t = mon.process_frame(DUMMY_FRAME.copy())
        o = occ.process_frame(DUMMY_FRAME.copy())
        b = beh.process_frame(DUMMY_FRAME.copy())
        e = env.process_frame(DUMMY_FRAME.copy())
        r = engine.update(t, o, b, e)

    assert len(r.forecast) == 6   # t+5 through t+30
