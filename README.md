![Tests](https://github.com/vishal9519-vis/Smart-Cabin/actions/workflows/main.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-1.0.0-orange)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-purple)
![Streamlit](https://img.shields.io/badge/dashboard-streamlit-red)
# Smart Cabin AI

An AI-powered vehicle cabin safety and environmental risk prediction system. Monitors cabin occupancy, thermal conditions, passenger behaviour, and air quality in real time — predicting dangerous situations before they become critical.

Built as a portfolio project targeting automotive AI roles at companies like Bosch, KPIT, Tata Elxsi, and Hyundai Mobis.

---

## What it does

| Module | Capability |
|---|---|
| **Occupancy** | YOLOv8 person detection, child presence heuristic, unattended occupant timer |
| **Thermal** | Physics-based cabin heat simulation, heatstroke risk prediction, seat-zone heatmap |
| **Behaviour** | Frame-differencing motion analysis, inactivity detection, MediaPipe pose distress flag |
| **Environment** | CO₂ buildup simulation, humidity tracking, ventilation scoring, AQI calculation |
| **Risk Engine** | Weighted multi-modal fusion, compound-hazard amplification, 30s forecast, A–F safety grade |
| **Dashboard** | Streamlit dark-theme analytics UI with live gauges, trend charts, and thermal heatmap |

---

## Architecture

```
Camera / Video File
        │
        ▼
┌───────────────────────────────────────┐
│           Core Module Pipeline        │
│  Occupancy ─ Thermal ─ Behaviour ─ Env │
└──────────────────┬────────────────────┘
                   │
                   ▼
        Risk Prediction Engine
        (weighted fusion + forecast)
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Alert System      Streamlit Dashboard
    (CV2 + terminal)  (analytics + heatmaps)
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/vishal9519-vis/smart-cabin-ai.git
cd Smart-Cabin

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run live monitor (webcam)
python main.py

# 4. Run with thermal heatmap overlay
python main.py --heatmap

# 5. Run without camera (simulation mode)
python main.py --simulate

# 6. Launch analytics dashboard
streamlit run dashboard/app.py

# 7. Run tests
python -m pytest tests/ -v
```

---

## Live monitor keyboard controls

| Key | Action |
|-----|--------|
| `C` | Toggle cooling on |
| `V` | Toggle ventilation on |
| `R` | Reset cooling/ventilation |
| `Q` | Quit |

---

## Screenshots & Demo

### Streamlit Analytics Dashboard

| Overview | Risk Score & Forecast |
| --- | --- |
| ![Overview](screenshots/home.png) | ![Risk](screenshots/risk_score.png) |
| Risk gauges + safety grade | Risk over time + 30s forecast + thermal heatmap |

| Event Log | System Info |
| --- | --- |
| ![Events](screenshots/event_log.png) | ![System](screenshots/system_info.png) |
| Critical alerts + event table | Model info + cabin controls |

| No Air Conditioning | No Ventilation |
| --- | --- |
| ![No AC](screenshots/no_air_conditioning.png) | ![No Vent](screenshots/no_ventilation.png) |
| AC toggled off state | Ventilation toggled off state |

---

### Streamlit Analytics Dashboard

| Overview | Thermal Heatmap | Trend Charts |
|---|---|---|
| ![Overview](screenshots/04_dashboard_overview.png) | ![Thermal](screenshots/05_dashboard_thermal_heatmap.png) | ![Trends](screenshots/06_dashboard_trend_charts.png) |
| Risk gauges + safety grade | Seat-zone temperature grid | CO₂ / AQI / temperature over time |

| Behaviour Analysis | Simulation Mode |
|---|---|
| ![Behaviour](screenshots/07_dashboard_behaviour.png) | ![Simulate](screenshots/08_simulation_mode.png) |
| Motion timeline + distress flags | `--simulate` terminal output |

### Demo Video

[![Demo Video](screenshots/04_dashboard_overview.png)](demo/demo_video.mp4)

> Full walkthrough: simulation start → compound hazard triggered → alert fired → dashboard analytics

---

## Project structure

```
smart-cabin-ai/
├── modules/
│   ├── occupancy.py      YOLOv8 detection + child heuristic
│   ├── thermal.py        Heat simulation + risk scoring
│   ├── behaviour.py      Motion analysis + distress detection
│   ├── environment.py    CO2/humidity/AQI simulation
│   ├── risk_engine.py    Multi-modal fusion + forecasting
│   └── alert_system.py  Alert delivery + rate limiting
├── utils/
│   ├── config.py         All tunable parameters
│   ├── logger.py         Structured CSV event logger
│   └── helpers.py        Shared utilities + heatmap renderer
├── dashboard/
│   └── app.py            Streamlit analytics dashboard
├── tests/
│   └── test_modules.py   pytest unit tests
├── data/
│   └── logs/             Event logs (CSV)
├── demo/
│   └── screenshots.png
├── main.py               OpenCV real-time entry point
└── requirements.txt
```

---

## Replacing simulation with real sensors

Every simulation function is isolated and clearly marked. Swapping to real hardware requires changing only one function per module:

- **Thermal camera** (FLIR / MLX90640): replace `ThermalMonitor._update_temperature()` with SDK read
- **CO₂ sensor** (MH-Z19B): replace `EnvironmentMonitor._read_sensors()` with UART read
- **Humidity** (DHT22): add I2C read to `_read_sensors()`
- **Alerts to SMS**: fill in the `AlertSystem._send_sms()` stub with Twilio SDK

---

## Risk scoring logic

The overall cabin risk score fuses four components using configurable weights:

```
risk = thermal(0.35) + occupancy(0.30) + behaviour(0.20) + environment(0.15)
```

When two or more components exceed 60/100 simultaneously, a compound-hazard multiplier is applied — because co-occurring hazards (e.g. high temperature + unattended child) are disproportionately dangerous.

Safety grades map as: A (<20) · B (<40) · C (<60) · D (<80) · F (≥80)

---

## Potential extensions

- Train a lightweight child vs. adult classifier to replace the height-ratio heuristic
- Replace linear forecast with an LSTM trained on real cabin sensor logs
- Add CAN bus integration for vehicle speed / door state as context features
- Deploy on NVIDIA Jetson Nano for in-vehicle edge inference

---

## Tech stack

Python · OpenCV · YOLOv8 (Ultralytics) · MediaPipe · NumPy · Pandas · Streamlit · Plotly

---

## License

MIT
