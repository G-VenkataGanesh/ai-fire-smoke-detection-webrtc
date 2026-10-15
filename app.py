"""
app.py - Streamlit Dashboard for AI-Based Real-Time Fire and Smoke Detection.

Features:
- Image detection with sample image loader
- Frame-by-frame video detection with export and timeline stats
- Real-time webcam streaming with Start/Stop controls and consecutive-frame confirmation
- Browser camera snapshot fallback
- Interactive detection history, audit log, and CSV export
"""

import os
import tempfile
import time
from datetime import datetime
from typing import Optional

import subprocess
import sys

try:
    import cv2
except ImportError:
    # Remove GUI OpenCV pulled transitively by ultralytics and install headless
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python", "opencv-python-headless"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from alert_manager import AlertManager, AlertRecord
from detector import (
    DEFAULT_LOCAL_WEIGHTS,
    DEFAULT_HF_REPO,
    FireSmokeDetector,
    IncompatibleModelError,
    ModelError,
    ModelNotFoundError,
    get_camera_capture,
    test_camera_access,
)

try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
    from webrtc_processor import DEFAULT_RTC_CONFIGURATION, FireSmokeVideoProcessor, HAS_WEBRTC
except ImportError:
    HAS_WEBRTC = False
    WebRtcMode = None  # type: ignore
    webrtc_streamer = None  # type: ignore
    DEFAULT_RTC_CONFIGURATION = None  # type: ignore
    FireSmokeVideoProcessor = None  # type: ignore


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Fire & Smoke Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    /* Modern Dashboard Styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FF4B4B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E222A;
        border-radius: 8px;
        padding: 16px;
        border-left: 4px solid #FF4B4B;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .status-normal {
        background-color: #0E3A20;
        border: 1px solid #10B981;
        color: #A7F3D0;
        padding: 12px 18px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 12px;
    }
    .status-fire {
        background-color: #4A1010;
        border: 2px solid #EF4444;
        color: #FECACA;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        animation: pulse 1.5s infinite;
        margin-bottom: 12px;
    }
    .status-smoke {
        background-color: #422B08;
        border: 2px solid #F59E0B;
        color: #FDE68A;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 12px;
    }
    @keyframes pulse {
        0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
        70% { transform: scale(1.01); box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
        100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "alert_manager" not in st.session_state:
    st.session_state.alert_manager = AlertManager(consecutive_threshold=3, cooldown_seconds=2.0)

if "detector" not in st.session_state:
    st.session_state.detector = None
    st.session_state.detector_error = None

if "webcam_running" not in st.session_state:
    st.session_state.webcam_running = False


# ---------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------
def load_detector_model(model_path: str):
    """Safely initialize or re-initialize the YOLO detector model."""
    try:
        detector = FireSmokeDetector(model_path=model_path, auto_download=True)
        st.session_state.detector = detector
        st.session_state.detector_error = None
        return detector, None
    except Exception as e:
        st.session_state.detector = None
        st.session_state.detector_error = str(e)
        return None, str(e)


# ---------------------------------------------------------
# Sidebar Controls & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.title("🔥 Control Panel")

    st.subheader("Model Configuration")
    custom_model_path = st.text_input(
        "Model Weights Path (.pt)",
        value=DEFAULT_LOCAL_WEIGHTS,
        help="Path to YOLO fire/smoke weights. Leave as default to use the trained D-Fire model.",
    )

    if st.button("🔄 Reload Model"):
        load_detector_model(custom_model_path)
        st.rerun()

    # Automatic initial load if not loaded yet
    if st.session_state.detector is None and st.session_state.detector_error is None:
        load_detector_model(custom_model_path)

    # Display Model Status
    if st.session_state.detector is not None:
        classes = st.session_state.detector.get_supported_classes()
        st.success(f"✅ Model Ready: {len(classes)} classes")
        class_badges = " ".join([f"`{name}`" for name in classes.values()])
        st.caption(f"Verified Classes: {class_badges}")
    elif st.session_state.detector_error:
        st.error(f"❌ Model Error: {st.session_state.detector_error}")

    st.markdown("---")
    st.subheader("Detection Thresholds")
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence score required to detect an object.",
    )
    iou_thresh = st.slider(
        "IoU (NMS) Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Overlap threshold for Non-Maximum Suppression.",
    )

    st.markdown("---")
    st.subheader("False Alarm Mitigation")
    consecutive_thresh = st.slider(
        "Consecutive Frame Confirmation",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of consecutive positive frames required to confirm an active alert.",
    )
    st.session_state.alert_manager.set_consecutive_threshold(consecutive_thresh)

    cooldown_sec = st.slider(
        "Alert Log Cooldown (seconds)",
        min_value=1.0,
        max_value=10.0,
        value=2.0,
        step=0.5,
        help="Minimum duration between consecutive alert log entries to prevent spam.",
    )
    st.session_state.alert_manager.cooldown_seconds = cooldown_sec

    st.markdown("---")
    st.caption("AI-Based Fire & Smoke Detection System • Ultralytics YOLOv8 & OpenCV")


# ---------------------------------------------------------
# Main Dashboard Header
# ---------------------------------------------------------
st.markdown('<div class="main-header">AI Real-Time Fire & Smoke Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Computer Vision safety monitoring with consecutive-frame confirmation and live telemetry.</div>',
    unsafe_allow_html=True,
)

# Verify detector is active before rendering features
detector: Optional[FireSmokeDetector] = st.session_state.detector

if detector is None:
    st.error(
        f"⚠️ **Model is not loaded.**\n\n"
        f"Reason: `{st.session_state.detector_error}`\n\n"
        f"**How to fix:**\n"
        f"1. Make sure you have an internet connection so the pre-trained weights from Hugging Face `{DEFAULT_HF_REPO}` can download automatically.\n"
        f"2. Or manually place your trained `.pt` weights into `{DEFAULT_LOCAL_WEIGHTS}` and click **Reload Model** in the sidebar."
    )
    st.stop()


# ---------------------------------------------------------
# Tab Navigation
# ---------------------------------------------------------
tabs = st.tabs([
    "📷 Image Detection",
    "🎥 Video Detection",
    "📹 Live Webcam",
    "📊 Alert History & Telemetry",
    "ℹ️ System Overview",
])


# =========================================================
# TAB 1: Image Detection
# =========================================================
with tabs[0]:
    st.subheader("Image Fire & Smoke Analysis")
    st.write("Upload an image or select a sample image to test detection bounding boxes and confidence scores.")

    img_col1, img_col2 = st.columns([1, 1])

    with img_col1:
        img_source = st.radio(
            "Image Input Source:",
            ["Choose from Sample Images", "Upload Local Image File"],
            horizontal=True,
        )

        input_image = None
        if img_source == "Choose from Sample Images":
            sample_options = {
                "Sample Fire & Smoke Combined": "samples/sample_fire_smoke.jpg",
                "Sample Smoke Scene": "samples/sample_smoke_1.jpg",
                "Sample Fire Camp / Blaze": "samples/sample_fire_1.jpg",
                "Multi-Quadrant D-Fire Grid": "samples/dfire_examples.png",
            }
            selected_sample_label = st.selectbox("Select Sample Asset:", list(sample_options.keys()))
            selected_path = sample_options[selected_sample_label]
            if os.path.exists(selected_path):
                input_image = Image.open(selected_path)
            else:
                st.warning(f"Sample asset not found at '{selected_path}'.")
        else:
            uploaded_file = st.file_uploader(
                "Upload Image (JPG, PNG, WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded_file is not None:
                try:
                    input_image = Image.open(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to open uploaded image: {e}")

    if input_image is not None:
        # Run detection
        with st.spinner("Processing image through YOLO network..."):
            result = detector.detect(
                input_image,
                conf_threshold=conf_thresh,
                iou_threshold=iou_thresh,
            )

        # Status Banner
        if result["fire_detected"] and result["smoke_detected"]:
            st.markdown(
                f'<div class="status-fire">🚨 CRITICAL ALERT: FIRE & SMOKE DETECTED! '
                f'(Fire Conf: {result["max_fire_conf"]*100:.1f}%, Smoke Conf: {result["max_smoke_conf"]*100:.1f}%)</div>',
                unsafe_allow_html=True,
            )
        elif result["fire_detected"]:
            st.markdown(
                f'<div class="status-fire">🔥 FIRE DETECTED! Peak Confidence: {result["max_fire_conf"]*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        elif result["smoke_detected"]:
            st.markdown(
                f'<div class="status-smoke">💨 SMOKE DETECTED! Peak Confidence: {result["max_smoke_conf"]*100:.1f}%</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-normal">✅ NORMAL: No Fire or Smoke Detected Above Threshold</div>',
                unsafe_allow_html=True,
            )

        # Visual Side-by-Side Comparison
        col_orig, col_det = st.columns(2)
        with col_orig:
            st.markdown("**Original Image**")
            st.image(input_image, use_container_width=True)

        with col_det:
            st.markdown("**AI Detection Result**")
            st.image(result["annotated_rgb"], use_container_width=True)

        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fire Bounding Boxes", result["fire_count"])
        m2.metric("Smoke Bounding Boxes", result["smoke_count"])
        m3.metric("Total Detections", result["total_detections"])
        m4.metric("Inference Latency", f"{result['inference_time_ms']:.1f} ms")

        # Detailed Detections Table
        if result["detections"]:
            st.markdown("#### Bounding Box Telemetry")
            table_data = [
                {
                    "#": idx + 1,
                    "Detected Class": d["class_name"],
                    "Confidence": f"{d['confidence'] * 100:.2f}%",
                    "Box [X1, Y1, X2, Y2]": str(d["box"]),
                }
                for idx, d in enumerate(result["detections"])
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)


# =========================================================
# TAB 2: Video Detection
# =========================================================
with tabs[1]:
    st.subheader("Video File Analysis & Frame-by-Frame Processing")
    st.write("Upload a video clip or test with the built-in sample fire video.")

    vid_col1, vid_col2 = st.columns([1, 1])

    with vid_col1:
        vid_source = st.radio(
            "Video Input Source:",
            ["Use Built-in Sample Video", "Upload Local Video File"],
            horizontal=True,
        )

        video_path = None
        temp_video_file = None

        if vid_source == "Use Built-in Sample Video":
            sample_vid_path = "samples/sample_fire_video.mp4"
            if os.path.exists(sample_vid_path):
                video_path = sample_vid_path
                st.info(f"Using sample video: `{sample_vid_path}`")
            else:
                st.warning("Sample video not found. Please upload a video file.")
        else:
            uploaded_vid = st.file_uploader(
                "Upload Video (MP4, AVI, MOV, MKV)",
                type=["mp4", "avi", "mov", "mkv"],
            )
            if uploaded_vid is not None:
                temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video_file.write(uploaded_vid.read())
                temp_video_file.close()
                video_path = temp_video_file.name

    with vid_col2:
        frame_stride = st.select_slider(
            "Frame Processing Stride (Speed vs Thoroughness)",
            options=[1, 2, 3, 5],
            value=1,
            help="1 = process every frame. 2 = process every second frame (2x speed).",
        )
        max_frames_to_process = st.number_input(
            "Max Frames to Process (0 for full video)",
            min_value=0,
            max_value=5000,
            value=150,
            step=50,
        )

    if video_path and st.button("🚀 Start Video Analysis", key="btn_start_video"):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        limit_frames = max_frames_to_process if max_frames_to_process > 0 else total_frames
        st.write(f"📊 Video Info: {width}x{height} @ {fps:.1f} FPS, {total_frames} total frames (processing up to {limit_frames}).")

        # Streamlit Live Preview Elements
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        preview_col, stats_col = st.columns([2, 1])

        with preview_col:
            video_frame_placeholder = st.empty()
        with stats_col:
            live_status_placeholder = st.empty()
            live_metrics_placeholder = st.empty()

        # Temporary output video path
        output_temp_path = os.path.join(tempfile.gettempdir(), f"detected_output_{int(time.time())}.mp4")

        # Process frame by frame with live updates
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        effective_fps = max(1.0, fps / frame_stride)
        writer = cv2.VideoWriter(output_temp_path, fourcc, effective_fps, (width, height))

        frame_idx = 0
        evaluated_count = 0
        fire_frames = 0
        smoke_frames = 0
        t0 = time.perf_counter()

        # Reset alert manager state for fresh video run
        st.session_state.alert_manager.reset_state()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if limit_frames and frame_idx > limit_frames:
                break

            if (frame_idx - 1) % frame_stride != 0:
                continue

            evaluated_count += 1
            res = detector.detect(frame, conf_threshold=conf_thresh, iou_threshold=iou_thresh)

            if res["fire_detected"]:
                fire_frames += 1
            if res["smoke_detected"]:
                smoke_frames += 1

            # Update temporal alert manager
            fire_alert, smoke_alert, new_alerts = st.session_state.alert_manager.update(
                fire_present=res["fire_detected"],
                smoke_present=res["smoke_detected"],
                fire_confidence=res["max_fire_conf"],
                smoke_confidence=res["max_smoke_conf"],
            )

            writer.write(res["annotated_bgr"])

            # Update UI at regular intervals
            progress_val = min(1.0, evaluated_count / max(1, limit_frames // frame_stride))
            progress_bar.progress(progress_val)
            status_text.text(f"Processing frame {frame_idx}/{limit_frames}...")

            video_frame_placeholder.image(res["annotated_rgb"], use_container_width=True)

            # Live Alert Status Badge
            if fire_alert and smoke_alert:
                live_status_placeholder.markdown(
                    '<div class="status-fire">🚨 CONFIRMED ALERT: FIRE & SMOKE!</div>',
                    unsafe_allow_html=True,
                )
            elif fire_alert:
                live_status_placeholder.markdown(
                    '<div class="status-fire">🔥 CONFIRMED FIRE ALERT!</div>',
                    unsafe_allow_html=True,
                )
            elif smoke_alert:
                live_status_placeholder.markdown(
                    '<div class="status-smoke">💨 CONFIRMED SMOKE ALERT!</div>',
                    unsafe_allow_html=True,
                )
            else:
                live_status_placeholder.markdown(
                    '<div class="status-normal">✅ Condition: Normal</div>',
                    unsafe_allow_html=True,
                )

            status = st.session_state.alert_manager.get_status()
            live_metrics_placeholder.markdown(
                f"""
                **Live Telemetry:**
                - Fire Consecutive: `{status['fire_consecutive']}/{status['threshold']}`
                - Smoke Consecutive: `{status['smoke_consecutive']}/{status['threshold']}`
                - Current Latency: `{res['inference_time_ms']:.1f} ms`
                - Alerts Triggered: `{len(st.session_state.alert_manager.history)}`
                """
            )

        cap.release()
        writer.release()
        elapsed_time = time.perf_counter() - t0

        st.success(f"🎉 Processing complete! Analyzed {evaluated_count} frames in {elapsed_time:.2f}s ({evaluated_count/max(0.1, elapsed_time):.1f} FPS).")

        # Summary Metrics
        vm1, vm2, vm3, vm4 = st.columns(4)
        vm1.metric("Evaluated Frames", evaluated_count)
        vm2.metric("Fire Frames", f"{fire_frames} ({fire_frames/max(1,evaluated_count)*100:.1f}%)")
        vm3.metric("Smoke Frames", f"{smoke_frames} ({smoke_frames/max(1,evaluated_count)*100:.1f}%)")
        vm4.metric("Logged Alerts", len(st.session_state.alert_manager.history))

        # Download Video Button
        if os.path.exists(output_temp_path):
            with open(output_temp_path, "rb") as vf:
                st.download_button(
                    label="📥 Download Annotated Video",
                    data=vf.read(),
                    file_name="fire_smoke_detected.mp4",
                    mime="video/mp4",
                )


# =========================================================
# TAB 3: Live Webcam Detection
# =========================================================
with tabs[2]:
    st.subheader("Live Real-Time Webcam Stream & Snapshot Detection")
    st.write(
        "Monitor a live video feed from your webcam. The system uses temporal consecutive-frame "
        "confirmation to suppress false alarms before raising an active alert."
    )

    webcam_mode = st.radio(
        "Choose Webcam Mode:",
        [
            "🌐 WebRTC Live Browser Streaming (Cloud & Local)",
            "💻 Local OpenCV Continuous Stream (cv2.VideoCapture)",
            "📸 Browser Snapshot Mode (st.camera_input)",
        ],
        index=0,
        horizontal=True,
        help="WebRTC connects directly to your web browser's camera. Local OpenCV accesses local hardware.",
    )

    if webcam_mode == "🌐 WebRTC Live Browser Streaming (Cloud & Local)":
        st.markdown(
            """
            **Browser WebRTC Streaming:** Requests camera permission directly in your web browser.
            Streamed frames are sent securely and analyzed frame-by-frame with YOLO bounding boxes,
            confidence labels, and consecutive-frame confirmation to prevent false alarms.
            """
        )

        if not HAS_WEBRTC or webrtc_streamer is None:
            st.error(
                "❌ **streamlit-webrtc is not available.** Please ensure `streamlit-webrtc` and `av` are installed in your environment."
            )
        else:
            rtc_config = DEFAULT_RTC_CONFIGURATION or {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

            webrtc_ctx = webrtc_streamer(
                key="fire-smoke-webrtc-stream",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=rtc_config,
                video_processor_factory=FireSmokeVideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            if webrtc_ctx.video_processor:
                # Synchronize slider thresholds and detectors with WebRTC processor
                webrtc_ctx.video_processor.detector = detector
                webrtc_ctx.video_processor.alert_manager = st.session_state.alert_manager
                webrtc_ctx.video_processor.conf_threshold = conf_thresh
                webrtc_ctx.video_processor.iou_threshold = iou_thresh

            # Real-Time Telemetry and Streaming Status Feedback
            if webrtc_ctx.state.playing and webrtc_ctx.video_processor:
                st.success("🟢 **Live WebRTC Stream Active** — Browser webcam frames are being processed in real time.")

                telemetry = webrtc_ctx.video_processor.get_telemetry()
                status = st.session_state.alert_manager.get_status()

                # Dynamic Alert Banner in Streamlit UI
                if telemetry["fire_alert"] and telemetry["smoke_alert"]:
                    st.markdown('<div class="status-fire">🚨 EMERGENCY: FIRE & SMOKE DETECTED!</div>', unsafe_allow_html=True)
                elif telemetry["fire_alert"]:
                    st.markdown('<div class="status-fire">🔥 EMERGENCY: FIRE DETECTED!</div>', unsafe_allow_html=True)
                elif telemetry["smoke_alert"]:
                    st.markdown('<div class="status-smoke">💨 WARNING: SMOKE DETECTED!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-normal">✅ STATUS NORMAL: Monitoring Active</div>', unsafe_allow_html=True)

                webrtc_col1, webrtc_col2, webrtc_col3, webrtc_col4 = st.columns(4)
                webrtc_col1.metric("Processed Frames", telemetry["frame_count"])
                webrtc_col2.metric("Streaming FPS", f"{telemetry['fps']:.1f}")
                webrtc_col3.metric("Inference Latency", f"{telemetry['latency_ms']:.1f} ms")
                webrtc_col4.metric("Confirmed Alerts", status["total_alerts"])
            else:
                st.info("💡 **Ready to stream:** Click the **START** button above to grant browser webcam access and begin detection.")

            with st.expander("ℹ️ WebRTC Diagnostics & Cloud Deployment Notes"):
                st.markdown(
                    """
                    - **Browser Permissions:** Ensure your browser is permitted to access your webcam (click **Allow** on the prompt).
                    - **HTTPS Requirement:** When running on Streamlit Community Cloud, browsers mandate HTTPS for webcam access.
                    - **ICE & STUN Traversal:** Uses Google STUN servers (`stun.l.google.com:19302`) to establish peer connections across firewalls.
                    - **Enterprise Firewalls:** If your network completely blocks WebRTC peer-to-peer UDP media streams, use **Browser Snapshot Mode** above.
                    """
                )

    elif webcam_mode == "💻 Local OpenCV Continuous Stream (cv2.VideoCapture)":
        w_col1, w_col2, w_col3 = st.columns([1, 1, 1])

        with w_col1:
            cam_idx = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1)
        with w_col2:
            test_cam = st.button("🔍 Test Camera Access")
        with w_col3:
            st.write("")  # alignment spacer
            if not st.session_state.webcam_running:
                start_btn = st.button("▶️ Start Webcam Stream", type="primary")
            else:
                start_btn = False
            stop_btn = st.button("⏹️ Stop Stream")

        if test_cam:
            is_ok, msg = test_camera_access(cam_idx)
            if is_ok:
                st.success(f"✅ {msg}")
            else:
                st.error(f"❌ {msg}\n\n*Troubleshooting tip:* If you are using a browser without direct USB access, switch to 'Browser Snapshot Mode' above.")

        if start_btn:
            st.session_state.webcam_running = True
            st.rerun()

        if stop_btn:
            st.session_state.webcam_running = False
            st.session_state.alert_manager.reset_state()
            st.rerun()

        # Stream Display Area
        stream_placeholder = st.empty()
        banner_placeholder = st.empty()
        telemetry_placeholder = st.empty()

        if st.session_state.webcam_running:
            cap = get_camera_capture(cam_idx)
            if not cap.isOpened():
                st.error(
                    f"❌ Could not open camera {cam_idx}. Make sure your webcam is plugged in, "
                    f"permissions are granted, and no other app (like Zoom or Teams) is locking it."
                )
                st.session_state.webcam_running = False
            else:
                frame_count = 0
                fps_calc_start = time.time()
                current_fps = 0.0

                try:
                    while st.session_state.webcam_running:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("⚠️ Failed to capture frame from webcam.")
                            break

                        frame_count += 1
                        res = detector.detect(
                            frame,
                            conf_threshold=conf_thresh,
                            iou_threshold=iou_thresh,
                        )

                        # Evaluate Consecutive-Frame Confirmation
                        fire_alert, smoke_alert, new_alerts = st.session_state.alert_manager.update(
                            fire_present=res["fire_detected"],
                            smoke_present=res["smoke_detected"],
                            fire_confidence=res["max_fire_conf"],
                            smoke_confidence=res["max_smoke_conf"],
                        )

                        # Render status banner
                        if fire_alert and smoke_alert:
                            banner_placeholder.markdown(
                                '<div class="status-fire">🚨 EMERGENCY: FIRE & SMOKE DETECTED!</div>',
                                unsafe_allow_html=True,
                            )
                        elif fire_alert:
                            banner_placeholder.markdown(
                                '<div class="status-fire">🔥 EMERGENCY: FIRE DETECTED!</div>',
                                unsafe_allow_html=True,
                            )
                        elif smoke_alert:
                            banner_placeholder.markdown(
                                '<div class="status-smoke">💨 WARNING: SMOKE DETECTED!</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            banner_placeholder.markdown(
                                '<div class="status-normal">✅ STATUS NORMAL: Monitoring Active</div>',
                                unsafe_allow_html=True,
                            )

                        # Draw telemetry overlay onto frame
                        display_img = res["annotated_rgb"]

                        # Calculate live FPS
                        if frame_count % 5 == 0:
                            current_fps = frame_count / max(0.001, time.time() - fps_calc_start)

                        stream_placeholder.image(display_img, use_container_width=True)

                        # Telemetry status row
                        status = st.session_state.alert_manager.get_status()
                        telemetry_placeholder.markdown(
                            f"**Frame:** `{frame_count}` | **FPS:** `{current_fps:.1f}` | "
                            f"**Latency:** `{res['inference_time_ms']:.1f}ms` | "
                            f"**Fire Count:** `{status['fire_consecutive']}/{status['threshold']}` | "
                            f"**Smoke Count:** `{status['smoke_consecutive']}/{status['threshold']}` | "
                            f"**Confirmed Alerts:** `{status['total_alerts']}`"
                        )

                        # Small sleep to prevent CPU thread starvation
                        time.sleep(0.02)

                finally:
                    cap.release()

    else:
        # Browser Snapshot Mode (st.camera_input)
        st.info("Browser Snapshot Mode uses your web browser's direct camera API. Take a photo to detect fire or smoke immediately.")
        cam_photo = st.camera_input("Capture Camera Snapshot")
        if cam_photo is not None:
            photo_img = Image.open(cam_photo)
            res = detector.detect(photo_img, conf_threshold=conf_thresh, iou_threshold=iou_thresh)

            if res["fire_detected"] or res["smoke_detected"]:
                # Log to alert manager as confirmed snapshot event
                st.session_state.alert_manager.update(
                    fire_present=res["fire_detected"],
                    smoke_present=res["smoke_detected"],
                    fire_confidence=res["max_fire_conf"],
                    smoke_confidence=res["max_smoke_conf"],
                )
                alert_text = "FIRE & SMOKE" if (res["fire_detected"] and res["smoke_detected"]) else ("FIRE" if res["fire_detected"] else "SMOKE")
                st.markdown(f'<div class="status-fire">🚨 {alert_text} DETECTED IN SNAPSHOT!</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-normal">✅ NORMAL: No hazards detected in snapshot.</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            with col1:
                st.image(photo_img, caption="Original Snapshot", use_container_width=True)
            with col2:
                st.image(res["annotated_rgb"], caption="AI Detection Result", use_container_width=True)


# =========================================================
# TAB 4: Alert History & Telemetry
# =========================================================
with tabs[3]:
    st.subheader("📋 Detection History & Alert Audit Log")
    st.write("All confirmed fire and smoke hazard detections verified through consecutive-frame temporal filtering.")

    history_records = st.session_state.alert_manager.get_history()

    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    total_alerts = len(history_records)
    fire_alerts_count = sum(1 for r in history_records if "FIRE" in str(r.get("Type", "")))
    smoke_alerts_count = sum(1 for r in history_records if "SMOKE" in str(r.get("Type", "")))
    last_timestamp = history_records[0]["Timestamp"] if history_records else "None"

    h_col1.metric("Total Confirmed Alerts", total_alerts)
    h_col2.metric("Fire Incidents", fire_alerts_count)
    h_col3.metric("Smoke Incidents", smoke_alerts_count)
    h_col4.metric("Last Alert At", last_timestamp)

    st.markdown("---")

    action_col1, action_col2 = st.columns([1, 4])
    with action_col1:
        if st.button("🗑️ Clear History"):
            st.session_state.alert_manager.clear_history()
            st.success("Alert history cleared.")
            st.rerun()

    if history_records:
        df_history = pd.DataFrame(history_records)
        st.dataframe(df_history, use_container_width=True)

        # CSV Export
        csv_bytes = df_history.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export Alert Log as CSV",
            data=csv_bytes,
            file_name=f"fire_smoke_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
    else:
        st.info("No confirmed alerts logged yet. Alerts will appear here when fire or smoke is detected over consecutive frames.")


# =========================================================
# TAB 5: System Overview & Architecture
# =========================================================
with tabs[4]:
    st.subheader("System Architecture & Implementation Details")
    st.markdown(
        f"""
        ### 🧠 AI Engine
        - **Model Architecture:** Ultralytics YOLOv8 Nano (optimized for low-latency edge inference)
        - **Pretrained Weights:** Fine-tuned on the D-Fire / Fire-Smoke benchmark dataset (`{DEFAULT_HF_REPO}`)
        - **Supported Classes:** `{detector.get_supported_classes()}`
        - **Inference Runtime:** PyTorch on `{detector.model.device}`

        ### 🛡️ False-Alarm Mitigation via Temporal Filtering
        Single-frame detectors frequently suffer from transient false alarms caused by:
        - Sun glints and headlights
        - Steam, dust, or aerosols
        - Yellow/orange clothing or reflectors

        This system applies **Consecutive-Frame Confirmation**:
        - Requires an incident to persist for **$K$ consecutive frames** (default: 3 frames).
        - If a detection drops for even a single frame during confirmation, the counter resets.
        - Once confirmed, a cooldown window prevents log flooding while continuing visual alerts.

        ### 📁 Project Structure
        - `app.py`: Streamlit multi-tab user interface with telemetry and streaming
        - `detector.py`: Modular YOLO inference engine with class validation and OpenCV drawing
        - `webrtc_processor.py`: Real-time WebRTC browser video processor for cloud deployments
        - `alert_manager.py`: Temporal smoothing state machine and incident logger
        - `requirements.txt`: Python package specifications
        - `README.md`: Installation instructions and architecture documentation
        """
    )
