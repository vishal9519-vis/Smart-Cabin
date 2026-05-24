# utils/logger.py
# Structured CSV event logger.
# Every module calls log_event() — file I/O stays in one place.

import csv
import os
from datetime import datetime
from utils.config import LOG_FILE, LOG_DIR

COLUMNS = [
    "timestamp",
    "event_type",      # e.g. CHILD_DETECTED, TEMP_WARNING
    "severity",        # INFO / WARNING / CRITICAL
    "cabin_temp",
    "occupancy_count",
    "risk_score",
    "notes",
]


def _ensure_log_exists():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()


def log_event(event_type, severity, cabin_temp=0.0, occupancy=0, risk_score=0.0, notes=""):
    """
    Append one structured event row to the CSV log.

    Parameters
    ----------
    event_type : str   Short tag  e.g. "CHILD_DETECTED"
    severity   : str   "INFO" | "WARNING" | "CRITICAL"
    """
    _ensure_log_exists()
    row = {
        "timestamp":       datetime.now().isoformat(timespec="seconds"),
        "event_type":      event_type,
        "severity":        severity,
        "cabin_temp":      round(float(cabin_temp), 2),
        "occupancy_count": int(occupancy),
        "risk_score":      round(float(risk_score), 2),
        "notes":           notes,
    }
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(row)


def load_log():
    """Return the full event log as a list of dicts. Empty list if no log yet."""
    _ensure_log_exists()
    with open(LOG_FILE, "r", newline="") as f:
        return list(csv.DictReader(f))
