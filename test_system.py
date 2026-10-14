"""
test_system.py - Comprehensive Unit and Integration Test Suite.

Verifies:
- Detector initialization and model class validation
- Image detection and bounding box generation
- Video processing and frame metrics
- AlertManager temporal smoothing and false alarm rejection
- Camera utility tests
- Error handling on missing/incompatible models
"""

import os
import unittest
import numpy as np

from alert_manager import AlertManager, AlertRecord
from detector import (
    FireSmokeDetector,
    IncompatibleModelError,
    ModelNotFoundError,
    get_camera_capture,
    test_camera_access,
)


class TestFireSmokeDetectionSystem(unittest.TestCase):
    """Test suite for the Fire and Smoke detection project."""

    @classmethod
    def setUpClass(cls):
        cls.weights_path = os.path.join("models", "fire_smoke_yolov8n.pt")
        cls.assertTrue(os.path.exists(cls.weights_path), f"Weights file missing at {cls.weights_path}")
        cls.detector = FireSmokeDetector(cls.weights_path)

    def test_01_model_classes_verified(self):
        """Verify the loaded YOLO model supports fire and smoke classes."""
        classes = self.detector.get_supported_classes()
        class_names = [v.lower() for v in classes.values()]
        self.assertTrue(
            any("fire" in name for name in class_names),
            f"Model does not contain 'fire' class: {classes}"
        )
        self.assertTrue(
            any("smoke" in name for name in class_names),
            f"Model does not contain 'smoke' class: {classes}"
        )

    def test_02_image_detection(self):
        """Verify detection pipeline runs on sample images and returns structured output."""
        sample_path = os.path.join("samples", "dfire_examples.png")
        self.assertTrue(os.path.exists(sample_path), f"Sample image not found: {sample_path}")

        res = self.detector.detect(sample_path, conf_threshold=0.25)

        self.assertIn("annotated_rgb", res)
        self.assertIn("detections", res)
        self.assertIn("inference_time_ms", res)
        self.assertIsInstance(res["detections"], list)
        self.assertGreater(len(res["detections"]), 0, "Expected at least 1 detection on dfire_examples.png")

        first_det = res["detections"][0]
        self.assertIn("class_name", first_det)
        self.assertIn("confidence", first_det)
        self.assertIn("box", first_det)
        self.assertEqual(len(first_det["box"]), 4)

    def test_03_temporal_smoothing_false_alarm_rejection(self):
        """Verify that single-frame glitches are rejected and only persistent detections confirm an alert."""
        mgr = AlertManager(consecutive_threshold=3, cooldown_seconds=1.0)

        # Frame 1: Glitch detection (1 frame)
        f_alert, s_alert, new_alerts = mgr.update(fire_present=True, smoke_present=False, fire_confidence=0.8)
        self.assertFalse(f_alert, "Single-frame glitch should not confirm alert.")
        self.assertEqual(len(new_alerts), 0)

        # Frame 2: Glitch disappeared
        f_alert, s_alert, new_alerts = mgr.update(fire_present=False, smoke_present=False)
        self.assertFalse(f_alert)
        self.assertEqual(mgr.fire_consecutive, 0, "Counter should reset to 0 upon drop.")

        # Frames 3, 4, 5: Persistent fire detection
        mgr.update(fire_present=True, smoke_present=False, fire_confidence=0.85)  # count=1
        self.assertFalse(mgr.fire_alert_active)
        mgr.update(fire_present=True, smoke_present=False, fire_confidence=0.87)  # count=2
        self.assertFalse(mgr.fire_alert_active)
        f_alert, _, new_alerts = mgr.update(fire_present=True, smoke_present=False, fire_confidence=0.90)  # count=3

        self.assertTrue(f_alert, "Alert should be confirmed after 3 consecutive frames.")
        self.assertEqual(len(new_alerts), 1)
        self.assertEqual(new_alerts[0].alert_type, "FIRE")

    def test_04_alert_history_logging_and_clearing(self):
        """Verify alert audit logging, dictionary conversion, and clearing."""
        mgr = AlertManager(consecutive_threshold=1, cooldown_seconds=0.0)
        mgr.update(fire_present=True, smoke_present=False, fire_confidence=0.9)
        mgr.update(fire_present=False, smoke_present=True, smoke_confidence=0.85)

        history = mgr.get_history()
        self.assertEqual(len(history), 2)
        self.assertIn("Timestamp", history[0])
        self.assertIn("Type", history[0])

        mgr.clear_history()
        self.assertEqual(len(mgr.get_history()), 0)
        self.assertFalse(mgr.fire_alert_active)

    def test_05_video_processing(self):
        """Verify process_video processes frames and generates output."""
        sample_video = os.path.join("samples", "sample_fire_video.mp4")
        self.assertTrue(os.path.exists(sample_video), f"Sample video not found: {sample_video}")

        res = self.detector.process_video(
            video_path=sample_video,
            conf_threshold=0.25,
            max_frames=10,
        )
        self.assertIn("total_frames_evaluated", res)
        self.assertGreaterEqual(res["total_frames_evaluated"], 1)
        self.assertIn("fps_processed", res)

    def test_06_camera_access_utility(self):
        """Verify camera test utility returns a tuple of (bool, str)."""
        is_ok, msg = test_camera_access(0)
        self.assertIsInstance(is_ok, bool)
        self.assertIsInstance(msg, str)
        print(f"\n[Camera Test] Index 0: {is_ok} - {msg}")

    def test_07_missing_custom_model_error(self):
        """Verify detector raises ModelNotFoundError when custom file path does not exist."""
        with self.assertRaises(ModelNotFoundError):
            FireSmokeDetector("non_existent_weights_xyz.pt")

    def test_08_webrtc_video_processor_frame_handling(self):
        """Verify WebRTC processor intercepts av.VideoFrame, processes it, and returns valid frame."""
        import av
        from webrtc_processor import FireSmokeVideoProcessor

        proc = FireSmokeVideoProcessor()
        proc.detector = self.detector
        proc.alert_manager = AlertManager(consecutive_threshold=2)

        # Create synthetic black video frame
        blank_arr = np.zeros((480, 640, 3), dtype=np.uint8)
        in_frame = av.VideoFrame.from_ndarray(blank_arr, format="bgr24")

        out_frame = proc.recv(in_frame)
        self.assertIsInstance(out_frame, av.VideoFrame)

        out_arr = out_frame.to_ndarray(format="bgr24")
        self.assertEqual(out_arr.shape, (480, 640, 3))
        self.assertEqual(proc.frame_count, 1)

        telemetry = proc.get_telemetry()
        self.assertEqual(telemetry["frame_count"], 1)
        self.assertIn("fps", telemetry)
        self.assertIn("latency_ms", telemetry)
        self.assertFalse(telemetry["fire_alert"])

    def test_09_webrtc_consecutive_alert_generation(self):
        """Verify WebRTC processor triggers confirmed alert upon consecutive fire frames."""
        import av
        import cv2
        from webrtc_processor import FireSmokeVideoProcessor

        sample_fire = os.path.join("samples", "sample_fire_1.jpg")
        if not os.path.exists(sample_fire):
            sample_fire = os.path.join("samples", "dfire_examples.png")
        self.assertTrue(os.path.exists(sample_fire))

        img = cv2.imread(sample_fire)
        img = cv2.resize(img, (640, 480))

        mgr = AlertManager(consecutive_threshold=2, cooldown_seconds=0.0)
        proc = FireSmokeVideoProcessor()
        proc.detector = self.detector
        proc.alert_manager = mgr
        proc.conf_threshold = 0.20

        # Frame 1
        frame1 = av.VideoFrame.from_ndarray(img.copy(), format="bgr24")
        proc.recv(frame1)

        # Frame 2
        frame2 = av.VideoFrame.from_ndarray(img.copy(), format="bgr24")
        proc.recv(frame2)

        telemetry = proc.get_telemetry()
        # Verify alert manager history recorded confirmed alert
        self.assertGreaterEqual(len(mgr.get_history()), 1)
        self.assertTrue(telemetry["fire_alert"] or telemetry["smoke_alert"])

    def test_10_webrtc_processor_safe_fallback(self):
        """Verify WebRTC processor gracefully returns raw frame when detector is not configured."""
        import av
        from webrtc_processor import FireSmokeVideoProcessor

        proc = FireSmokeVideoProcessor()
        raw_arr = np.ones((240, 320, 3), dtype=np.uint8) * 128
        in_frame = av.VideoFrame.from_ndarray(raw_arr, format="bgr24")

        out_frame = proc.recv(in_frame)
        self.assertIsInstance(out_frame, av.VideoFrame)
        np.testing.assert_array_equal(out_frame.to_ndarray(format="bgr24"), raw_arr)


if __name__ == "__main__":
    unittest.main(verbosity=2)

