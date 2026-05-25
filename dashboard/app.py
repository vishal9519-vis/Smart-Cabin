# dashboard/app.py
# Smart Cabin AI — Streamlit Analytics Dashboard.
#
# Run: streamlit run dashboard/app.py

import sys
import os

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# ── Module imports ────────────────────────────────────────────────────────────
try:
    from modules.occupancy import OccupancyDetector
    from modules.thermal import ThermalMonitor
    from modules.behaviour import BehaviourAnalyser
    from modules.environment import EnvironmentMonitor
    from modules.risk_engine import RiskEngine
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    IMPORT_ERROR = str(e)

try:
    from utils.logger import load_log
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False

try:
    from utils.config import DASHBOARD_REFRESH_MS
except ImportError:
    DASHBOARD_REFRESH_MS = 2000

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Cabin AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0d0f14; color: #e0e4ec; }
.main .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 100%; }

div[data-testid="metric-container"] {
    background: #161a24;
    border: 1px solid #2a2f3e;
    border-radius: 10px;
    padding: 12px 16px;
}
div[data-testid="metric-container"] label { color: #6b7a99 !important; font-size: 0.78rem; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.6rem; font-weight: 600;
}

.alert-critical {
    background: #1f0808; border-left: 3px solid #e53e3e;
    border-radius: 6px; padding: 10px 14px; margin: 4px 0;
    font-size: 0.9rem; color: #feb2b2;
}
.alert-warning {
    background: #1f1408; border-left: 3px solid #d97706;
    border-radius: 6px; padding: 10px 14px; margin: 4px 0;
    font-size: 0.9rem; color: #fde68a;
}
.alert-info {
    background: #081418; border-left: 3px solid #38b2ac;
    border-radius: 6px; padding: 10px 14px; margin: 4px 0;
    font-size: 0.9rem; color: #81e6d9;
}
.alert-error {
    background: #1f0808; border-left: 3px solid #e53e3e;
    border-radius: 6px; padding: 14px 16px; margin: 8px 0;
    font-size: 0.95rem; color: #feb2b2;
}

.grade-badge {
    display: inline-block;
    font-size: 2.2rem; font-weight: 700;
    width: 56px; height: 56px; line-height: 56px;
    text-align: center; border-radius: 8px;
}

h3 { color: #8ab4f8 !important; font-weight: 500; font-size: 0.9rem;
    letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.5rem; }

section[data-testid="stSidebar"] { background: #0a0c12; border-right: 1px solid #1e2230; }
</style>
""", unsafe_allow_html=True)

# ── Show error and stop if modules failed to import ───────────────────────────
if not MODULES_AVAILABLE:
    st.markdown("## 🚗 Smart Cabin AI — Intelligence Platform")
    st.markdown(f"""
<div class="alert-error">
⚠️ <strong>Module import failed:</strong> {IMPORT_ERROR}<br><br>
Make sure you are running from the <strong>project root</strong>:<br>
<code>streamlit run dashboard/app.py</code><br><br>
Also ensure all dependencies are installed:<br>
<code>pip install -r requirements.txt</code>
</div>
""", unsafe_allow_html=True)
    st.stop()

# ── Session state initialisation ─────────────────────────────────────────────
@st.cache_resource
def get_modules():
    return {
        "occupancy":   OccupancyDetector(),
        "thermal":     ThermalMonitor(),
        "behaviour":   BehaviourAnalyser(),
        "environment": EnvironmentMonitor(),
        "risk_engine": RiskEngine(),
    }

try:
    mods = get_modules()
except Exception as e:
    st.error(f"Failed to initialise modules: {e}")
    st.stop()

if "risk_history" not in st.session_state:
    st.session_state.risk_history = []
if "temp_history" not in st.session_state:
    st.session_state.temp_history = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cabin Controls")
    cooling_on      = st.toggle("Air Conditioning", value=False)
    ventilation_on  = st.toggle("Ventilation",      value=False)
    show_heatmap    = st.toggle("Thermal Heatmap",  value=True)
    simulate_mode   = st.toggle("Simulation Mode",  value=True)

    try:
        mods["thermal"].set_cooling(cooling_on)
    except Exception:
        pass
    try:
        mods["environment"].set_ventilation(ventilation_on)
    except Exception:
        pass

    st.markdown("---")
    st.markdown("### 📖 Key Bindings (OpenCV window)")
    st.markdown("`C` — cooling on \n`V` — ventilation on \n`R` — reset \n`Q` — quit")
    st.markdown("---")
    st.markdown("### 🏷️ System Info")
    st.caption(f"Dashboard refresh: {DASHBOARD_REFRESH_MS}ms")
    st.caption("Model: YOLOv8n · MediaPipe Pose")
    st.caption("Data: Simulated sensor feed")

# ── Run one frame of module pipeline ─────────────────────────────────────────
dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

try:
    occ_r   = mods["occupancy"].process_frame(dummy_frame.copy())
    therm_r = mods["thermal"].process_frame(dummy_frame.copy())
    beh_r   = mods["behaviour"].process_frame(dummy_frame.copy())
    env_r   = mods["environment"].process_frame(dummy_frame.copy())
    risk_r  = mods["risk_engine"].update(therm_r, occ_r, beh_r, env_r)
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.exception(e)
    st.stop()

# ── Safe attribute helpers ────────────────────────────────────────────────────
def safe_get(obj, attr, default=0):
    try:
        val = getattr(obj, attr, default)
        return val if val is not None else default
    except Exception:
        return default

def safe_score(obj, key, default=0):
    try:
        scores = getattr(obj, "component_scores", {})
        return scores.get(key, default) if scores else default
    except Exception:
        return default

# Extract values safely
overall_score   = safe_get(risk_r, "overall_score", 0)
safety_grade    = safe_get(risk_r, "safety_grade", "A")
alerts          = safe_get(risk_r, "alerts", []) or []
forecast        = safe_get(risk_r, "forecast", []) or []

temperature     = safe_get(therm_r, "temperature", 25)
trend_raw       = safe_get(therm_r, "trend", "stable")
alert_level     = safe_get(therm_r, "alert_level", "NORMAL")
zone_map        = safe_get(therm_r, "zone_map", None)

occ_count       = safe_get(occ_r, "count", 0)
child_count     = safe_get(occ_r, "child_count", 0)

co2_ppm         = safe_get(env_r, "co2_ppm", 400)
aqi_score       = safe_get(env_r, "aqi_score", 80)

# Convert trend to numeric delta for st.metric (must be number or None)
def parse_trend_delta(trend):
    """Convert trend string like 'rising', '+1.2', 'stable' to float or None."""
    if trend is None:
        return None
    if isinstance(trend, (int, float)):
        return round(float(trend), 1)
    s = str(trend).strip().lower()
    if s in ("stable", "normal", "ok", ""):
        return None
    try:
        return round(float(s), 1)
    except ValueError:
        return None  # Non-numeric strings like "rising" — skip delta

temp_delta = parse_trend_delta(trend_raw)

# Append to rolling history
now_str = datetime.now().strftime("%H:%M:%S")
st.session_state.risk_history.append({"time": now_str, "risk": float(overall_score)})
st.session_state.temp_history.append({"time": now_str, "temp": float(temperature)})
if len(st.session_state.risk_history) > 60:
    st.session_state.risk_history.pop(0)
    st.session_state.temp_history.pop(0)

# ════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ════════════════════════════════════════════════════════════════════════════
st.markdown("## 🚗 Smart Cabin AI — Intelligence Platform")

# ── Row 1: Top-level metrics ──────────────────────────────────────────────────
col1, col2, col3, col4, col5, col6 = st.columns(6)

grade_colors = {"A": "#48bb78", "B": "#68d391", "C": "#ecc94b", "D": "#ed8936", "F": "#e53e3e"}
g_color = grade_colors.get(str(safety_grade), "#718096")

with col1:
    st.metric("Risk Score", f"{overall_score:.0f} / 100")

with col2:
    st.markdown(f"""<div style='text-align:center'>
<div style='color:#6b7a99;font-size:0.78rem;margin-bottom:4px'>SAFETY GRADE</div>
<div class='grade-badge' style='background:{g_color}22;color:{g_color}'>
{safety_grade}
</div></div>""", unsafe_allow_html=True)

with col3:
    st.metric("Cabin Temp", f"{temperature}°C", delta=temp_delta)

with col4:
    child_label = f"{child_count} child" if child_count else None
    st.metric("Occupants", occ_count, delta=child_label)

with col5:
    st.metric("CO₂ Level", f"{co2_ppm:.0f} ppm")

with col6:
    st.metric("AQI Score", f"{aqi_score:.0f} / 100")

st.markdown("---")

# ── Row 2: Active alerts + Component gauges ───────────────────────────────────
alert_col, gauge_col = st.columns([1, 1])

with alert_col:
    st.markdown("### Active alerts")
    if alerts:
        for alert in alerts:
            alert_str = str(alert)
            level = "critical" if any(x in alert_str for x in ["🚨", "🔴", "CRITICAL"]) else \
                    "warning"  if any(x in alert_str for x in ["🟠", "🟡", "WARNING"])  else "info"
            st.markdown(f'<div class="alert-{level}">{alert_str}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-info">✅ All systems normal — cabin environment safe</div>',
                    unsafe_allow_html=True)

with gauge_col:
    st.markdown("### Component risk breakdown")

    def make_gauge(title, value, color):
        value = float(value) if value is not None else 0.0
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": title, "font": {"size": 13, "color": "#8ab4f8"}},
            number={"font": {"size": 18, "color": "#e0e4ec"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#3a4155", "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.22},
                "bgcolor": "#161a24",
                "bordercolor": "#2a2f3e",
                "steps": [
                    {"range": [0,  40], "color": "#0d1520"},
                    {"range": [40, 70], "color": "#1a1400"},
                    {"range": [70,100], "color": "#1f0808"},
                ],
                "threshold": {
                    "line": {"color": "#e53e3e", "width": 2},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
        ))
        fig.update_layout(
            height=140, margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="#0d0f14", font_color="#e0e4ec",
        )
        return fig

    g1, g2 = st.columns(2)
    g3, g4 = st.columns(2)

    with g1:
        st.plotly_chart(make_gauge("Thermal",     safe_score(risk_r, "thermal"),     "#f6ad55"),
                        use_container_width=True, config={"displayModeBar": False})
    with g2:
        st.plotly_chart(make_gauge("Occupancy",   safe_score(risk_r, "occupancy"),   "#68d391"),
                        use_container_width=True, config={"displayModeBar": False})
    with g3:
        st.plotly_chart(make_gauge("Behaviour",   safe_score(risk_r, "behaviour"),   "#76e4f7"),
                        use_container_width=True, config={"displayModeBar": False})
    with g4:
        st.plotly_chart(make_gauge("Environment", safe_score(risk_r, "environment"), "#b794f4"),
                        use_container_width=True, config={"displayModeBar": False})

# ── Row 3: Trend charts ───────────────────────────────────────────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("### Risk score over time")
    if st.session_state.risk_history:
        df_risk = pd.DataFrame(st.session_state.risk_history)
        fig = px.area(df_risk, x="time", y="risk", color_discrete_sequence=["#e53e3e"])
        fig.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=30),
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font_color="#8ab4f8",
            xaxis=dict(showgrid=False, color="#3a4155"),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#1e2230", color="#3a4155"),
        )
        fig.update_traces(fillcolor="rgba(229,62,62,0.15)", line_width=1.5)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with chart_col2:
    st.markdown("### 30-second risk forecast")
    if forecast:
        forecast_times = [f"+{(i+1)*5}s" for i in range(len(forecast))]
        df_fc = pd.DataFrame({"time": forecast_times, "predicted_risk": forecast})
        fig_fc = px.line(df_fc, x="time", y="predicted_risk",
                         markers=True, color_discrete_sequence=["#f6ad55"])
        fig_fc.add_hrect(y0=70, y1=100, fillcolor="rgba(229,62,62,0.1)",
                         line_width=0, annotation_text="critical zone")
        fig_fc.update_layout(
            height=200, margin=dict(l=0, r=0, t=10, b=30),
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font_color="#8ab4f8",
            xaxis=dict(showgrid=False, color="#3a4155"),
            yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#1e2230", color="#3a4155"),
        )
        st.plotly_chart(fig_fc, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("No forecast data available yet.")

# ── Row 4: Thermal heatmap ────────────────────────────────────────────────────
if show_heatmap:
    st.markdown("### Cabin thermal heatmap")
    heat_col1, heat_col2 = st.columns([2, 1])

    with heat_col1:
        # Fallback zone_map if module didn't return one
        if zone_map is None or not hasattr(zone_map, "__len__"):
            zmap = np.random.uniform(0.1, 0.6, (6, 8))
        else:
            try:
                zmap = np.array(zone_map, dtype=float)
                if zmap.ndim != 2:
                    zmap = np.random.uniform(0.1, 0.6, (6, 8))
            except Exception:
                zmap = np.random.uniform(0.1, 0.6, (6, 8))

        fig_heat = go.Figure(go.Heatmap(
            z=zmap,
            colorscale=[
                [0.0,  "#0d1a2e"],
                [0.3,  "#1a4a7a"],
                [0.6,  "#e67e22"],
                [0.85, "#e53e3e"],
                [1.0,  "#fff"],
            ],
            showscale=True,
            colorbar=dict(
                title="Heat intensity",
                titlefont=dict(color="#8ab4f8"),
                tickfont=dict(color="#8ab4f8"),
            ),
            zmin=0, zmax=1,
        ))

        h, w = zmap.shape
        seat_labels = ["Driver", "Passenger", "Rear L", "Rear R"]
        seat_x = [w * 0.25, w * 0.75, w * 0.25, w * 0.75]
        seat_y = [h * 0.45, h * 0.45, h * 0.75, h * 0.75]

        fig_heat.add_trace(go.Scatter(
            x=seat_x, y=seat_y,
            mode="markers+text",
            text=seat_labels,
            textposition="top center",
            textfont=dict(color="white", size=11),
            marker=dict(symbol="circle-open", size=18, color="white",
                        line=dict(width=1.5)),
        ))
        fig_heat.update_layout(
            height=260, margin=dict(l=0, r=0, t=10, b=10),
            paper_bgcolor="#0d0f14", plot_bgcolor="#0d0f14", font_color="#8ab4f8",
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    with heat_col2:
        st.markdown("**Thermal readings**")
        st.metric("Cabin temp",  f"{temperature}°C")
        st.metric("Trend",       str(trend_raw) if trend_raw is not None else "stable")
        st.metric("Alert level", str(alert_level))
        thermal_score = safe_score(risk_r, "thermal")
        st.progress(min(100, max(0, int(thermal_score))) / 100)

# ── Row 5: Event log ──────────────────────────────────────────────────────────
st.markdown("### Event log")

if LOGGER_AVAILABLE:
    try:
        log_entries = load_log()
        if log_entries:
            df_log = pd.DataFrame(log_entries).tail(30)
            df_log = df_log.iloc[::-1]

            def severity_style(val):
                c = {
                    "CRITICAL": "color: #fc8181",
                    "WARNING":  "color: #f6ad55",
                    "INFO":     "color: #68d391",
                }
                return c.get(str(val), "")

            if "severity" in df_log.columns:
                # Use .map() (pandas >= 2.1) with fallback to .applymap()
                try:
                    styled = df_log.style.map(severity_style, subset=["severity"])
                except AttributeError:
                    styled = df_log.style.applymap(severity_style, subset=["severity"])
                st.dataframe(styled, use_container_width=True, height=220)
            else:
                st.dataframe(df_log, use_container_width=True, height=220)
        else:
            st.caption("No events logged yet — run the main.py monitor to generate data.")
    except Exception as ex:
        st.caption(f"Log file not found or unreadable ({ex}) — run main.py first.")
else:
    st.caption("Logger module not available. Run main.py first to generate events.")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=DASHBOARD_REFRESH_MS, key="cabin_refresh")
except ImportError:
    st.caption("💡 Install `streamlit-autorefresh` for live updates: `pip install streamlit-autorefresh`")
