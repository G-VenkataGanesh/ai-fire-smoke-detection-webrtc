"""
alert_manager.py - Alert and Confirmation Manager for Fire & Smoke Detection.

This module handles false-alarm mitigation via consecutive-frame confirmation,
tracks active alert states, and maintains an audit log of alert events with
timestamps.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


@dataclass
class AlertRecord:
    """Represents an individual confirmed alert event."""
    timestamp: str
    alert_type: str  # "FIRE", "SMOKE", or "FIRE & SMOKE"
    confidence: float
    consecutive_frames: int
    details: str = ""


class AlertManager:
    """
    Manages detection alerts with temporal smoothing (consecutive frame confirmation)
    to minimize false alarms from transient visual artifacts.
    """

    def __init__(
        self,
        consecutive_threshold: int = 3,
        cooldown_seconds: float = 2.0,
    ) -> None:
        """
        Initialize the AlertManager.

        Args:
            consecutive_threshold: Number of consecutive positive frames required
                                   before declaring a confirmed alert.
            cooldown_seconds: Minimum seconds between logging new alert events
                              to prevent log flooding while condition persists.
        """
        self.consecutive_threshold = max(1, consecutive_threshold)
        self.cooldown_seconds = max(0.0, cooldown_seconds)

        # Frame counters for temporal filtering
        self.fire_consecutive: int = 0
        self.smoke_consecutive: int = 0

        # Peak confidences during current confirmation window
        self.fire_peak_conf: float = 0.0
        self.smoke_peak_conf: float = 0.0

        # Active alert states
        self.fire_alert_active: bool = False
        self.smoke_alert_active: bool = False

        # Timestamps of last alert logs for cooldown calculation
        self.last_fire_alert_time: Optional[datetime] = None
        self.last_smoke_alert_time: Optional[datetime] = None

        # History log of confirmed alerts
        self.history: List[AlertRecord] = []

        # Total frames evaluated
        self.total_frames_processed: int = 0

        # Thread-safety lock for concurrent access (e.g. WebRTC worker threads)
        self._lock = threading.RLock()

    def set_consecutive_threshold(self, threshold: int) -> None:
        """Update the consecutive frames threshold."""
        with self._lock:
            self.consecutive_threshold = max(1, int(threshold))

    def update(
        self,
        fire_present: bool,
        smoke_present: bool,
        fire_confidence: float = 0.0,
        smoke_confidence: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> Tuple[bool, bool, List[AlertRecord]]:
        """
        Update the manager with detection results from a new frame.

        Args:
            fire_present: Whether fire was detected in this frame.
            smoke_present: Whether smoke was detected in this frame.
            fire_confidence: Highest fire detection confidence in this frame.
            smoke_confidence: Highest smoke detection confidence in this frame.
            timestamp: Optional datetime; defaults to datetime.now().

        Returns:
            Tuple of:
            - is_fire_alert_confirmed (bool)
            - is_smoke_alert_confirmed (bool)
            - newly_triggered_alerts (List[AlertRecord])
        """
        with self._lock:
            self.total_frames_processed += 1
            now = timestamp or datetime.now()
            new_alerts: List[AlertRecord] = []

            # --- Update Fire Consecutive Counter ---
            if fire_present:
                self.fire_consecutive += 1
                self.fire_peak_conf = max(self.fire_peak_conf, fire_confidence)
            else:
                self.fire_consecutive = 0
                self.fire_peak_conf = 0.0
                self.fire_alert_active = False

            # --- Update Smoke Consecutive Counter ---
            if smoke_present:
                self.smoke_consecutive += 1
                self.smoke_peak_conf = max(self.smoke_peak_conf, smoke_confidence)
            else:
                self.smoke_consecutive = 0
                self.smoke_peak_conf = 0.0
                self.smoke_alert_active = False

            # --- Evaluate Fire Alert Confirmation ---
            if self.fire_consecutive >= self.consecutive_threshold:
                self.fire_alert_active = True
                time_since_last = (
                    (now - self.last_fire_alert_time).total_seconds()
                    if self.last_fire_alert_time
                    else float("inf")
                )
                if time_since_last >= self.cooldown_seconds:
                    record = AlertRecord(
                        timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
                        alert_type="FIRE",
                        confidence=round(self.fire_peak_conf, 3),
                        consecutive_frames=self.fire_consecutive,
                        details=f"Fire confirmed over {self.fire_consecutive} consecutive frames.",
                    )
                    self.history.append(record)
                    new_alerts.append(record)
                    self.last_fire_alert_time = now

            # --- Evaluate Smoke Alert Confirmation ---
            if self.smoke_consecutive >= self.consecutive_threshold:
                self.smoke_alert_active = True
                time_since_last = (
                    (now - self.last_smoke_alert_time).total_seconds()
                    if self.last_smoke_alert_time
                    else float("inf")
                )
                if time_since_last >= self.cooldown_seconds:
                    record = AlertRecord(
                        timestamp=now.strftime("%Y-%m-%d %H:%M:%S"),
                        alert_type="SMOKE",
                        confidence=round(self.smoke_peak_conf, 3),
                        consecutive_frames=self.smoke_consecutive,
                        details=f"Smoke confirmed over {self.smoke_consecutive} consecutive frames.",
                    )
                    self.history.append(record)
                    new_alerts.append(record)
                    self.last_smoke_alert_time = now

            return self.fire_alert_active, self.smoke_alert_active, new_alerts

    def get_status(self) -> Dict[str, object]:
        """Return the current alert status dictionary."""
        with self._lock:
            return {
                "fire_active": self.fire_alert_active,
                "smoke_active": self.smoke_alert_active,
                "fire_consecutive": self.fire_consecutive,
                "smoke_consecutive": self.smoke_consecutive,
                "threshold": self.consecutive_threshold,
                "fire_peak_conf": self.fire_peak_conf,
                "smoke_peak_conf": self.smoke_peak_conf,
                "total_frames": self.total_frames_processed,
                "total_alerts": len(self.history),
            }

    def get_history(self) -> List[Dict[str, object]]:
        """Return history as a list of dicts suitable for display or pandas DataFrame."""
        with self._lock:
            return [
                {
                    "Timestamp": r.timestamp,
                    "Type": r.alert_type,
                    "Confidence": f"{r.confidence * 100:.1f}%",
                    "Consecutive Frames": r.consecutive_frames,
                    "Details": r.details,
                }
                for r in reversed(self.history)
            ]

    def reset_state(self) -> None:
        """Reset consecutive counters and active states without wiping history."""
        with self._lock:
            self.fire_consecutive = 0
            self.smoke_consecutive = 0
            self.fire_peak_conf = 0.0
            self.smoke_peak_conf = 0.0
            self.fire_alert_active = False
            self.smoke_alert_active = False

    def clear_history(self) -> None:
        """Clear all historical alert records and reset state."""
        with self._lock:
            self.history.clear()
            self.reset_state()
            self.total_frames_processed = 0
            self.last_fire_alert_time = None
            self.last_smoke_alert_time = None
