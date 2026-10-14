"""
detector.py - YOLO-based Fire and Smoke Detection Engine.

This module encapsulates model loading, verification of fire & smoke classes,
inference on images, video frames, and webcam feeds, as well as high-contrast
visual annotation.
"""

import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Base directory for resolving relative paths on Streamlit Cloud & local execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Default Hugging Face repository and model filename
DEFAULT_HF_REPO = "rabahdev/fire-smoke-yolov8n"
DEFAULT_HF_FILENAME = "best.pt"

# Identify default weights path across common working directories
def _resolve_default_weights() -> str:
    candidates = [
        os.path.join(BASE_DIR, "models", "fire_smoke_yolov8n.pt"),
        os.path.join("models", "fire_smoke_yolov8n.pt"),
        os.path.join(BASE_DIR, "fire_smoke_yolov8n.pt"),
        "fire_smoke_yolov8n.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return os.path.join(BASE_DIR, "models", "fire_smoke_yolov8n.pt")

DEFAULT_LOCAL_WEIGHTS = _resolve_default_weights()


class ModelError(Exception):
    """Base exception for model related errors."""
    pass


class ModelNotFoundError(ModelError):
    """Raised when the specified model weights file cannot be found."""
    pass


class IncompatibleModelError(ModelError):
    """Raised when a model does not support fire and smoke classes."""
    pass


class FireSmokeDetector:
    """
    Fire and Smoke Detector wrapping Ultralytics YOLOv8.
    Ensures model validity, performs inference, and generates clean visual annotations.
    """

    # Visual styling constants
    COLOR_FIRE_BGR = (0, 69, 255)       # Vibrant Red-Orange in BGR
    COLOR_SMOKE_BGR = (255, 178, 50)    # Cool Amber / Cyan in BGR
    COLOR_DEFAULT_BGR = (0, 255, 0)     # Green fallback in BGR

    def __init__(
        self,
        model_path: Optional[str] = None,
        auto_download: bool = True,
    ) -> None:
        """
        Initialize the Fire & Smoke Detector.

        Args:
            model_path: Path to .pt weights file. If None, defaults to models/fire_smoke_yolov8n.pt.
            auto_download: If True, attempts to download weights from Hugging Face if missing.
        """
        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Please run: pip install ultralytics"
            )

        self.model_path = model_path or DEFAULT_LOCAL_WEIGHTS
        self.model: Optional[YOLO] = None
        self.classes: Dict[int, str] = {}
        self.fire_class_ids: List[int] = []
        self.smoke_class_ids: List[int] = []
        self._inference_lock = threading.Lock()

        self._load_and_validate_model(auto_download=auto_download)

    def _ensure_weights_exist(self, auto_download: bool) -> str:
        """Verify weights file exists, or download from Hugging Face if using default."""
        if os.path.exists(self.model_path):
            return self.model_path

        # Check relative to BASE_DIR if a relative path was provided
        alt_path = os.path.join(BASE_DIR, self.model_path)
        if os.path.exists(alt_path):
            self.model_path = alt_path
            return self.model_path

        # If a custom path was provided that doesn't exist, don't silently overwrite it
        if os.path.normpath(self.model_path) != os.path.normpath(DEFAULT_LOCAL_WEIGHTS):
            raise ModelNotFoundError(
                f"Custom model file '{self.model_path}' was not found. "
                "Please verify the file path, or leave the path empty to use the default fire & smoke model."
            )

        if not auto_download:
            raise ModelNotFoundError(
                f"Model weights not found at '{self.model_path}'. "
                f"Please download a YOLO fire-and-smoke model (e.g. from Hugging Face: {DEFAULT_HF_REPO}) "
                f"and place it at '{self.model_path}'."
            )

        # Attempt automated download via huggingface_hub for default model
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs(os.path.dirname(os.path.abspath(self.model_path)), exist_ok=True)
            print(f"Downloading fire & smoke model from Hugging Face ({DEFAULT_HF_REPO})...")
            downloaded = hf_hub_download(
                repo_id=DEFAULT_HF_REPO,
                filename=DEFAULT_HF_FILENAME,
                local_dir=os.path.dirname(os.path.abspath(self.model_path)),
            )
            # Standardize filename if needed
            if os.path.exists(downloaded) and downloaded != self.model_path:
                import shutil
                shutil.copy2(downloaded, self.model_path)
            return self.model_path
        except Exception as e:
            raise ModelNotFoundError(
                f"Could not automatically download weights from '{DEFAULT_HF_REPO}': {e}.\n"
                f"Please place your trained fire/smoke .pt file at '{self.model_path}'."
            )

    def _load_and_validate_model(self, auto_download: bool) -> None:
        """Load YOLO model and verify fire and smoke classes."""
        resolved_path = self._ensure_weights_exist(auto_download)
        try:
            self.model = YOLO(resolved_path)
        except Exception as e:
            raise ModelError(f"Failed to load YOLO model from '{resolved_path}': {e}")

        # Extract class mapping
        if hasattr(self.model, "names") and self.model.names:
            self.classes = {int(k): str(v).strip() for k, v in self.model.names.items()}
        else:
            raise IncompatibleModelError("Unable to retrieve class names from the loaded model.")

        # Identify fire and smoke class indices
        self.fire_class_ids = [
            cid for cid, name in self.classes.items() if "fire" in name.lower()
        ]
        self.smoke_class_ids = [
            cid for cid, name in self.classes.items() if "smoke" in name.lower()
        ]

        # Verify that the model actually detects fire and/or smoke
        if not self.fire_class_ids and not self.smoke_class_ids:
            class_list_str = ", ".join([f"'{name}'" for name in self.classes.values()][:10])
            raise IncompatibleModelError(
                f"Incompatible model detected! The loaded model has classes: [{class_list_str}]. "
                "None of these correspond to 'fire' or 'smoke'. Please use a model fine-tuned on "
                "fire and smoke detection, such as 'rabahdev/fire-smoke-yolov8n'."
            )

    def get_supported_classes(self) -> Dict[int, str]:
        """Return the dictionary of model class IDs and names."""
        return self.classes

    def detect(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> Dict[str, Any]:
        """
        Run fire and smoke detection on a single image.

        Args:
            image_input: File path, numpy BGR/RGB array, or PIL Image.
            conf_threshold: Minimum confidence threshold (0.0 to 1.0).
            iou_threshold: NMS IoU threshold (0.0 to 1.0).

        Returns:
            Dictionary containing:
            - 'annotated_bgr': OpenCV BGR image with drawn boxes and labels.
            - 'annotated_rgb': RGB image suitable for Streamlit / PIL display.
            - 'detections': List of detection dictionaries.
            - 'fire_detected': bool.
            - 'smoke_detected': bool.
            - 'fire_count': int.
            - 'smoke_count': int.
            - 'max_fire_conf': float.
            - 'max_smoke_conf': float.
            - 'inference_time_ms': float.
        """
        if self.model is None:
            raise ModelError("Model is not initialized.")

        # Standardize input to numpy BGR image for processing
        bgr_image = self._to_bgr_array(image_input)
        orig_h, orig_w = bgr_image.shape[:2]

        with self._inference_lock:
            t_start = time.perf_counter()
            results = self.model.predict(
                bgr_image,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False,
            )
            t_inference = (time.perf_counter() - t_start) * 1000.0

        detections: List[Dict[str, Any]] = []
        fire_count = 0
        smoke_count = 0
        max_fire_conf = 0.0
        max_smoke_conf = 0.0

        annotated_bgr = bgr_image.copy()

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # Clamp coordinates to image boundaries
                    x1 = max(0, min(orig_w - 1, x1))
                    y1 = max(0, min(orig_h - 1, y1))
                    x2 = max(0, min(orig_w - 1, x2))
                    y2 = max(0, min(orig_h - 1, y2))

                    raw_label = self.classes.get(cls_id, f"Class {cls_id}")
                    is_fire = cls_id in self.fire_class_ids
                    is_smoke = cls_id in self.smoke_class_ids

                    if is_fire:
                        fire_count += 1
                        max_fire_conf = max(max_fire_conf, conf)
                        label_name = "Fire"
                        box_color = self.COLOR_FIRE_BGR
                    elif is_smoke:
                        smoke_count += 1
                        max_smoke_conf = max(max_smoke_conf, conf)
                        label_name = "Smoke"
                        box_color = self.COLOR_SMOKE_BGR
                    else:
                        label_name = raw_label.title()
                        box_color = self.COLOR_DEFAULT_BGR

                    detections.append({
                        "class_id": cls_id,
                        "class_name": label_name,
                        "confidence": round(conf, 4),
                        "box": [x1, y1, x2, y2],
                        "is_fire": is_fire,
                        "is_smoke": is_smoke,
                    })

                    # Draw stylish bounding box and tag
                    self._draw_detection_box(
                        annotated_bgr,
                        box=(x1, y1, x2, y2),
                        label=f"{label_name} {conf * 100:.1f}%",
                        color=box_color,
                    )

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        return {
            "annotated_bgr": annotated_bgr,
            "annotated_rgb": annotated_rgb,
            "detections": detections,
            "fire_detected": fire_count > 0,
            "smoke_detected": smoke_count > 0,
            "fire_count": fire_count,
            "smoke_count": smoke_count,
            "total_detections": len(detections),
            "max_fire_conf": round(max_fire_conf, 4),
            "max_smoke_conf": round(max_smoke_conf, 4),
            "inference_time_ms": round(t_inference, 1),
        }

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        alert_manager: Optional[Any] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Process a video file frame-by-frame with optional alert tracking and export.

        Args:
            video_path: Path to input video file.
            output_path: Path to write annotated output video (optional).
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IoU NMS threshold.
            frame_stride: Process every Nth frame (1 = every frame, 2 = every second frame).
            max_frames: Maximum number of frames to process (None for entire video).
            alert_manager: Optional AlertManager instance to update state.
            progress_callback: Optional callable(frame_idx, total_frames, result_dict, active_alerts).

        Returns:
            Dictionary with processing metrics and summary statistics.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        total_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames_to_process = min(total_input_frames, max_frames) if max_frames else total_input_frames
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            effective_fps = max(1.0, fps / frame_stride)
            writer = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

        processed_count = 0
        fire_frame_count = 0
        smoke_frame_count = 0
        raw_frame_idx = 0

        t_start = time.perf_counter()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                raw_frame_idx += 1
                if max_frames and raw_frame_idx > max_frames:
                    break

                # Apply frame stride
                if (raw_frame_idx - 1) % frame_stride != 0:
                    continue

                processed_count += 1
                result = self.detect(frame, conf_threshold=conf_threshold, iou_threshold=iou_threshold)

                if result["fire_detected"]:
                    fire_frame_count += 1
                if result["smoke_detected"]:
                    smoke_frame_count += 1

                confirmed_alerts = []
                if alert_manager is not None:
                    _, _, confirmed_alerts = alert_manager.update(
                        fire_present=result["fire_detected"],
                        smoke_present=result["smoke_detected"],
                        fire_confidence=result["max_fire_conf"],
                        smoke_confidence=result["max_smoke_conf"],
                    )

                if writer:
                    writer.write(result["annotated_bgr"])

                if progress_callback:
                    # Provide progress update
                    progress_callback(
                        frame_idx=processed_count,
                        total_frames=frames_to_process // frame_stride if frames_to_process else None,
                        result=result,
                        confirmed_alerts=confirmed_alerts,
                    )

        finally:
            cap.release()
            if writer:
                writer.release()

        elapsed = time.perf_counter() - t_start

        return {
            "total_frames_evaluated": processed_count,
            "fire_frames": fire_frame_count,
            "smoke_frames": smoke_frame_count,
            "fire_frame_ratio": round(fire_frame_count / max(1, processed_count), 3),
            "smoke_frame_ratio": round(smoke_frame_count / max(1, processed_count), 3),
            "elapsed_seconds": round(elapsed, 2),
            "fps_processed": round(processed_count / max(0.001, elapsed), 1),
            "output_video_path": output_path if output_path and os.path.exists(output_path) else None,
        }

    def _draw_detection_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        label: str,
        color: Tuple[int, int, int],
        line_thickness: int = 2,
    ) -> None:
        """Draw bounding box with high-contrast label tag."""
        x1, y1, x2, y2 = box

        # Draw main rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)

        # Draw corner accents for high-tech aesthetic
        corner_len = min(20, max(5, (x2 - x1) // 6, (y2 - y1) // 6))
        accent_thickness = line_thickness + 2
        # Top-left corner
        cv2.line(image, (x1, y1), (x1 + corner_len, y1), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y1), (x1, y1 + corner_len), color, accent_thickness, cv2.LINE_AA)
        # Top-right corner
        cv2.line(image, (x2, y1), (x2 - corner_len, y1), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y1), (x2, y1 + corner_len), color, accent_thickness, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(image, (x1, y2), (x1 + corner_len, y2), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x1, y2), (x1, y2 - corner_len), color, accent_thickness, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(image, (x2, y2), (x2 - corner_len, y2), color, accent_thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y2), (x2, y2 - corner_len), color, accent_thickness, cv2.LINE_AA)

        # Label background pill
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        font_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

        tag_y1 = max(0, y1 - text_h - 10)
        tag_y2 = y1
        tag_x1 = x1
        tag_x2 = min(image.shape[1] - 1, x1 + text_w + 10)

        # Draw solid tag background
        cv2.rectangle(image, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)

        # Draw text with high-contrast white font
        text_x = tag_x1 + 5
        text_y = tag_y2 - 5
        cv2.putText(
            image,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    def draw_telemetry_hud(
        self,
        image: np.ndarray,
        fire_alert: bool,
        smoke_alert: bool,
        fire_consecutive: int,
        smoke_consecutive: int,
        threshold: int,
        fps: float = 0.0,
        latency_ms: float = 0.0,
    ) -> np.ndarray:
        """
        Overlay an alert banner and live telemetry HUD directly onto the video frame.
        Ensures instantaneous visual feedback in WebRTC streaming sessions.
        """
        h, w = image.shape[:2]

        # 1. Top Alert Banner
        banner_h = 36
        if fire_alert and smoke_alert:
            banner_color = (0, 0, 180)  # Dark Red BGR
            banner_text = "[CRITICAL ALERT] FIRE & SMOKE DETECTED!"
        elif fire_alert:
            banner_color = (0, 0, 180)  # Red BGR
            banner_text = "[EMERGENCY] CONFIRMED FIRE ALERT!"
        elif smoke_alert:
            banner_color = (0, 140, 255)  # Amber BGR
            banner_text = "[WARNING] CONFIRMED SMOKE ALERT!"
        else:
            banner_color = (30, 30, 30)  # Dark Charcoal BGR
            banner_text = "[MONITORING] Live Feed - Status: Normal"

        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
        cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            image,
            banner_text,
            (12, 25),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 2. Bottom Telemetry Bar
        bar_h = 26
        bar_y1 = max(0, h - bar_h)
        cv2.rectangle(overlay, (0, bar_y1), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)

        telemetry_str = (
            f"FPS: {fps:.1f} | Latency: {latency_ms:.1f}ms | "
            f"Fire Streak: {fire_consecutive}/{threshold} | Smoke Streak: {smoke_consecutive}/{threshold}"
        )
        cv2.putText(
            image,
            telemetry_str,
            (10, h - 7),
            font,
            0.45,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        return image

    def _to_bgr_array(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Convert various input formats to a contiguous BGR uint8 numpy array."""
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file does not exist: {image_input}")
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Could not read image file: {image_input}")
            return img

        if isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if isinstance(image_input, np.ndarray):
            if image_input.ndim != 3 or image_input.shape[2] not in (1, 3, 4):
                raise ValueError(f"Invalid image array shape: {image_input.shape}")
            if image_input.shape[2] == 4:
                return cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            if image_input.shape[2] == 1:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            return image_input.copy()

        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def get_camera_capture(camera_index: int = 0) -> cv2.VideoCapture:
    """
    Open a VideoCapture instance optimized for Windows (using DirectShow)
    with automatic fallback to default backend.
    """
    import platform
    if platform.system() == "Windows":
        # DirectShow is faster and avoids delay on Windows
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            return cap
    return cv2.VideoCapture(camera_index)


def test_camera_access(camera_index: int = 0) -> Tuple[bool, str]:
    """
    Test whether a webcam at camera_index can be successfully opened and read.

    Returns:
        (is_successful, status_message)
    """
    cap = None
    try:
        cap = get_camera_capture(camera_index)
        if not cap.isOpened():
            return False, f"Could not open camera at index {camera_index}. Device may be disconnected or in use by another application."
        ret, frame = cap.read()
        if not ret or frame is None:
            return False, f"Camera at index {camera_index} opened, but failed to capture a valid frame."
        return True, f"Camera {camera_index} active (Resolution: {frame.shape[1]}x{frame.shape[0]})."
    except Exception as e:
        return False, f"Camera error: {e}"
    finally:
        if cap is not None:
            cap.release()


# Expose WebRTC processor from webrtc_processor if available
try:
    from webrtc_processor import FireSmokeVideoProcessor, DEFAULT_RTC_CONFIGURATION
except ImportError:
    FireSmokeVideoProcessor = None  # type: ignore
    DEFAULT_RTC_CONFIGURATION = None  # type: ignore

