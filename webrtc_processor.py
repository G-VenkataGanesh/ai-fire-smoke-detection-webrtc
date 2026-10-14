"""
webrtc_processor.py - WebRTC Video Stream Processor for AI Fire & Smoke Detection.

This module provides real-time browser-based video stream processing using
streamlit-webrtc and PyAV, enabling continuous webcam detection on remote
cloud platforms (e.g. Streamlit Community Cloud) where cv2.VideoCapture(0)
cannot access visitor hardware.
"""

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

try:
    import av
    from streamlit_webrtc import RTCConfiguration, VideoProcessorBase
    HAS_WEBRTC = True
except ImportError:
    av = None
    HAS_WEBRTC = False
    # Fallback dummy class if streamlit-webrtc or av is not available
    class VideoProcessorBase:  # type: ignore
        pass
    RTCConfiguration = None  # type: ignore

from alert_manager import AlertManager
from detector import FireSmokeDetector

# Standard public STUN configuration for WebRTC NAT traversal across firewalls
DEFAULT_RTC_CONFIGURATION = (
    RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]},
            ]
        }
    )
    if HAS_WEBRTC and RTCConfiguration is not None
    else None
)


class FireSmokeVideoProcessor(VideoProcessorBase):
    """
    Video processor that intercepts incoming browser webcam frames via WebRTC,
    executes YOLO fire/smoke inference, performs temporal false-alarm mitigation,
    overlays bounding boxes and telemetry HUD, and returns the annotated frame.
    """

    def __init__(self) -> None:
        self.detector: Optional[FireSmokeDetector] = None
        self.alert_manager: Optional[AlertManager] = None
        self.conf_threshold: float = 0.25
        self.iou_threshold: float = 0.45

        self.frame_count: int = 0
        self.fps: float = 0.0
        self._fps_start_time: float = time.time()
        self._lock = threading.Lock()

        # Cached telemetry state for frontend display
        self.latest_result: Optional[Dict[str, Any]] = None
        self.latest_fire_alert: bool = False
        self.latest_smoke_alert: bool = False
        self.latest_latency_ms: float = 0.0

    def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
        """
        Process a single video frame received from the client browser.

        Args:
            frame: Incoming av.VideoFrame in client format.

        Returns:
            av.VideoFrame containing annotated detections and telemetry HUD.
        """
        if av is None:
            return frame

        # Convert frame to standard OpenCV BGR format
        bgr_frame: np.ndarray = frame.to_ndarray(format="bgr24")
        if bgr_frame is None or bgr_frame.size == 0:
            return frame

        # If detector is not yet linked, return raw frame unchanged
        if self.detector is None:
            return frame

        # 1. Execute YOLO Fire & Smoke Inference
        res = self.detector.detect(
            bgr_frame,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )

        fire_alert = False
        smoke_alert = False
        fire_consecutive = 0
        smoke_consecutive = 0
        threshold = 3

        # 2. Update Temporal Alert Manager for False Alarm Mitigation
        if self.alert_manager is not None:
            fire_alert, smoke_alert, _ = self.alert_manager.update(
                fire_present=res["fire_detected"],
                smoke_present=res["smoke_detected"],
                fire_confidence=res["max_fire_conf"],
                smoke_confidence=res["max_smoke_conf"],
            )
            status = self.alert_manager.get_status()
            fire_consecutive = status.get("fire_consecutive", 0)
            smoke_consecutive = status.get("smoke_consecutive", 0)
            threshold = status.get("threshold", 3)

        # 3. Calculate Rolling Real-Time FPS
        self.frame_count += 1
        now = time.time()
        if self.frame_count % 5 == 0:
            elapsed = max(0.001, now - self._fps_start_time)
            self.fps = self.frame_count / elapsed

        # 4. Overlay Alert Banner and Telemetry HUD onto the annotated image
        annotated_bgr = self.detector.draw_telemetry_hud(
            image=res["annotated_bgr"],
            fire_alert=fire_alert,
            smoke_alert=smoke_alert,
            fire_consecutive=fire_consecutive,
            smoke_consecutive=smoke_consecutive,
            threshold=threshold,
            fps=self.fps,
            latency_ms=res.get("inference_time_ms", 0.0),
        )

        # 5. Store Latest Telemetry in a Thread-Safe Manner
        with self._lock:
            self.latest_result = res
            self.latest_fire_alert = fire_alert
            self.latest_smoke_alert = smoke_alert
            self.latest_latency_ms = res.get("inference_time_ms", 0.0)

        # 6. Re-encode into av.VideoFrame and return
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    def get_telemetry(self) -> Dict[str, Any]:
        """Thread-safe retrieval of latest streaming telemetry for dashboard display."""
        with self._lock:
            return {
                "frame_count": self.frame_count,
                "fps": round(self.fps, 1),
                "latency_ms": round(self.latest_latency_ms, 1),
                "fire_alert": self.latest_fire_alert,
                "smoke_alert": self.latest_smoke_alert,
                "result": self.latest_result,
            }
