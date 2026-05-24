# utils/config.py
# Single source of truth for every tunable parameter.
# Nothing gets hardcoded elsewhere — always import from here.

import os

# ── Camera ──────────────────────────────────────────────────────────────────
CAMERA_SOURCE = 0        # 0 = default webcam; replace with "path/to/video.mp4" for testing
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
FPS_TARGET    = 20

# ── YOLO Detection ──────────────────────────────────────────────────────────
YOLO_MODEL         = "yolov8n.pt"   # nano = fastest on CPU; swap to yolov8s.pt for accuracy
CONFIDENCE_THRESH  = 0.45
# Heuristic: if bounding box height < this fraction of frame height, flag as child
CHILD_HEIGHT_RATIO = 0.45

# ── Thermal simulation ──────────────────────────────────────────────────────
TEMP_BASELINE    = 28.0   # °C — normal parked cabin
TEMP_CRITICAL    = 42.0   # °C — CRITICAL alert threshold
TEMP_WARNING     = 36.0   # °C — WARNING alert threshold
TEMP_RISE_RATE   = 0.07   # °C per second (real cabins rise ~0.1–0.15°C/s in sun)
TEMP_COOL_RATE   = 0.04   # °C per second when cooling simulated

# ── Occupancy intelligence ──────────────────────────────────────────────────
UNATTENDED_SECONDS = 30   # seconds before "unattended occupant" flag fires
INACTIVITY_SECONDS = 20   # seconds of no movement before distress flag fires

# ── Risk engine weights (must sum to 1.0) ───────────────────────────────────
RISK_WEIGHT_THERMAL   = 0.35
RISK_WEIGHT_OCCUPANCY = 0.30
RISK_WEIGHT_BEHAVIOUR = 0.20
RISK_WEIGHT_ENV       = 0.15

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_DIR  = "data/logs"
LOG_FILE = os.path.join(LOG_DIR, "cabin_events.csv")

# ── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_MS = 1500
