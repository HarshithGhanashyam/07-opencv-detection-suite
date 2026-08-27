from __future__ import annotations

import tempfile

import cv2
import numpy as np
import streamlit as st

from detection.detectors import (
    annotate, canny_edges, detect_color_blobs, run_all_detections,
)

st.set_page_config(page_title="OpenCV Detection Suite", page_icon="🎥", layout="wide")
st.title("🎥 Real-Time Face / Eye / Object Detection Suite")
st.caption("Haar-cascade face/eye/smile/body detection, Canny edges, and HSV color-blob tracking — image, video, or live webcam.")

mode = st.sidebar.radio("Mode", ["Image upload", "Video upload", "Live webcam"])
task = st.sidebar.selectbox(
    "Detection task",
    ["Face / Eye / Smile / Body", "Canny edges", "Color blob tracking"],
)
color_choice = None
if task == "Color blob tracking":
    color_choice = st.sidebar.selectbox("Track color", ["red", "green", "blue", "yellow"])
body_mode = "full"
if task == "Face / Eye / Smile / Body":
    body_mode = st.sidebar.radio("Body detector", ["full", "upper"], horizontal=True)


def process_frame(frame_bgr: np.ndarray) -> np.ndarray:
    if task == "Face / Eye / Smile / Body":
        result = run_all_detections(frame_bgr, body_mode=body_mode)
        return annotate(frame_bgr, result)
    elif task == "Canny edges":
        edges = canny_edges(frame_bgr)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        boxes = detect_color_blobs(frame_bgr, color_choice)
        out = frame_bgr.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return out


if mode == "Image upload":
    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Could not decode this image.")
        else:
            out = process_frame(frame)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with col2:
                st.subheader("Detected")
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    else:
        st.info("Upload an image, or try the bundled synthetic sample below.")
        if st.button("Run on bundled sample (sample_data/shapes.png)"):
            frame = cv2.imread("sample_data/shapes.png")
            out = process_frame(frame)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with col2:
                st.subheader("Detected")
                st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

elif mode == "Video upload":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    frame_skip = st.slider("Process every Nth frame (speed vs. smoothness)", 1, 10, 3)
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.getvalue())
            video_path = tmp.name

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Could not open this video file.")
        else:
            stframe = st.empty()
            progress = st.progress(0)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idx = 0
            run = st.button("Process video")
            if run:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if idx % frame_skip == 0:
                        out = process_frame(frame)
                        stframe.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
                    idx += 1
                    progress.progress(min(idx / total, 1.0))
                cap.release()
                st.success(f"Done — processed {idx} frames.")

else:  # Live webcam
    st.write("Live webcam mode uses `streamlit-webrtc` for in-browser camera access.")
    try:
        from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
        import av

        class Processor(VideoProcessorBase):
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                out = process_frame(img)
                return av.VideoFrame.from_ndarray(out, format="bgr24")

        webrtc_streamer(key="detect", video_processor_factory=Processor)
    except ImportError:
        st.warning(
            "streamlit-webrtc is not available in this environment. "
            "Install it with `pip install streamlit-webrtc av` to enable live webcam mode. "
            "Image and video upload modes work without it."
        )
