# 🎥 OpenCV Real-Time Face / Eye / Object Detection Suite

A Streamlit app for three classic computer-vision tasks, over three input modes.

## Detection tasks
- **Face / Eye / Smile / Body** — Haar cascades (bundled with `opencv-python`, no
  downloads needed): `haarcascade_frontalface_default`, `haarcascade_eye`,
  `haarcascade_smile`, and `haarcascade_fullbody` / `haarcascade_upperbody`. Eyes and
  smiles are searched only inside each detected face's bounding box for speed and
  fewer false positives.
- **Canny edge detection** — Gaussian blur + `cv2.Canny`.
- **HSV color-blob tracking** — thresholds an HSV mask per color (red/green/blue/yellow),
  cleans it with morphological opening, then finds and boxes contours above a minimum
  area.

## Input modes
- **Image upload** — single image, side-by-side original vs. annotated.
- **Video upload** — processes an uploaded video frame-by-frame (with an adjustable
  "process every Nth frame" control for speed), streaming annotated frames live.
- **Live webcam** — in-browser webcam via `streamlit-webrtc`, applying the selected
  detector to each incoming frame in real time. Falls back to a clear message if
  `streamlit-webrtc` isn't installed — image/video modes work regardless.

## Run it
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tested
- All four detector functions verified against a synthetic shapes image
  (`sample_data/shapes.png`): correctly found **0** faces (none present), found edge
  pixels via Canny, and correctly boxed each colored shape (red/green/blue) by HSV range.
- Video pipeline verified end-to-end against a synthetic 30-frame test clip
  (`sample_data/test_video.mp4`): `cv2.VideoCapture` opens it, reports the correct
  frame count, and reads frames successfully.
- Streamlit app confirmed booting and serving (`HTTP 200`).
- `streamlit-webrtc` import confirmed available in this environment.

> Note: real face/eye/smile detection needs an actual photographed face — the bundled
> sample is a synthetic shapes image (used to verify the *pipeline*, and correctly
> returns zero face detections). Upload your own photo to see face/eye/smile boxes.

## Project layout
```
app.py                    Streamlit UI (3 modes x 3 tasks)
detection/detectors.py    Cascade loading, detection, edge, and color-blob functions
sample_data/               Synthetic test image + video, and the video generator
```
