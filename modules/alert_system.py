# modules/alert_system.py
# Alert delivery system.
#
# Receives a RiskResult and decides how to surface the alerts:
#   - Terminal print with colour coding
#   - OpenCV on-screen banner
#   - (Optional) system beep for critical alerts
#   - Rate-limited so the same alert doesn't spam every frame
#
# Extension points:
#   - SMS via Twilio: swap _send_sms() stub with real API call
#   - Email via smtplib: already stubbed below
#   - CAN bus message on real vehicle hardware

import time
import cv2
from typing import List

from utils.helpers import draw_alert_banner


# ANSI colour codes for terminal output
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_RESET  = "\033[0m"


class AlertSystem:
    def __init__(self):
        # Track last-printed time per alert text to prevent spam
        self._alert_cooldowns = {}
        self._cooldown_secs   = 5.0

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame, risk_result):
        """
        Display alerts on the frame and print to terminal.
        Call once per frame after risk_engine.update().
        """
        for alert in risk_result.alerts:
            self._display_alert(frame, alert, risk_result.overall_score)
            self._print_alert(alert, risk_result.overall_score)

        # Draw safety grade in top-right corner
        self._draw_grade(frame, risk_result.safety_grade, risk_result.overall_score)

    # ── Frame overlays ────────────────────────────────────────────────────────

    def _display_alert(self, frame, alert: str, score: float):
        if score >= 70:
            color = (0, 0, 200)
        elif score >= 40:
            color = (0, 130, 220)
        else:
            color = (0, 160, 80)
        draw_alert_banner(frame, alert, color=color)

    def _draw_grade(self, frame, grade: str, score: float):
        color_map = {
            "A": (0, 200, 80),
            "B": (80, 200, 0),
            "C": (0, 165, 255),
            "D": (0, 80, 255),
            "F": (0, 0, 220),
        }
        color = color_map.get(grade, (180, 180, 180))
        cv2.putText(
            frame, grade,
            (frame.shape[1] - 36, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1, color, 2, cv2.LINE_AA
        )

    # ── Terminal output ───────────────────────────────────────────────────────

    def _print_alert(self, alert: str, score: float):
        now  = time.time()
        last = self._alert_cooldowns.get(alert, 0.0)
        if (now - last) < self._cooldown_secs:
            return

        self._alert_cooldowns[alert] = now
        ts    = time.strftime("%H:%M:%S")
        color = _RED if score >= 70 else _YELLOW if score >= 40 else _GREEN
        print(f"{color}[{ts}] ALERT  score={score:.0f}  {alert}{_RESET}")

    # ── Extension stubs ───────────────────────────────────────────────────────

    def _send_sms(self, message: str):
        """
        Stub — replace with Twilio SDK call.
        from twilio.rest import Client
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        client.messages.create(body=message, from_=FROM_NUMBER, to=TO_NUMBER)
        """
        pass

    def _send_email(self, subject: str, body: str):
        """
        Stub — replace with smtplib/sendgrid implementation.
        """
        pass
