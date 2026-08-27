"""
OpenCV detection primitives: Haar-cascade face/eye/smile/body detection,
Canny edge detection, and HSV color-blob tracking.
All cascades ship inside opencv-python's `cv2.data.haarcascades` directory,
so no extra downloads are needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


@dataclass
class DetectionResult:
    faces: List[Box]
    eyes: List[Box]
    smiles: List[Box]
    bodies: List[Box]


class CascadeBank:
    """Loads and caches all Haar cascades used by the app."""

    def __init__(self) -> None:
        base = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self.eye = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        self.smile = cv2.CascadeClassifier(base + "haarcascade_smile.xml")
        self.fullbody = cv2.CascadeClassifier(base + "haarcascade_fullbody.xml")
        self.upperbody = cv2.CascadeClassifier(base + "haarcascade_upperbody.xml")

    def ready(self) -> bool:
        return not any(
            c.empty() for c in
            (self.face, self.eye, self.smile, self.fullbody, self.upperbody)
        )


_bank: CascadeBank | None = None


def get_bank() -> CascadeBank:
    global _bank
    if _bank is None:
        _bank = CascadeBank()
    return _bank


def detect_faces_eyes_smiles(gray: np.ndarray) -> Tuple[List[Box], List[Box], List[Box]]:
    bank = get_bank()
    faces = bank.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    all_eyes: List[Box] = []
    all_smiles: List[Box] = []
    for (fx, fy, fw, fh) in faces:
        roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = bank.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=8, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            all_eyes.append((fx + ex, fy + ey, ew, eh))
        smiles = bank.smile.detectMultiScale(roi, scaleFactor=1.7, minNeighbors=20, minSize=(20, 20))
        for (sx, sy, sw, sh) in smiles:
            all_smiles.append((fx + sx, fy + sy, sw, sh))
    return list(faces), all_eyes, all_smiles


def detect_bodies(gray: np.ndarray, mode: str = "full") -> List[Box]:
    bank = get_bank()
    cascade = bank.fullbody if mode == "full" else bank.upperbody
    bodies = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(40, 80))
    return list(bodies)


def run_all_detections(bgr_frame: np.ndarray, body_mode: str = "full") -> DetectionResult:
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces, eyes, smiles = detect_faces_eyes_smiles(gray)
    bodies = detect_bodies(gray, mode=body_mode)
    return DetectionResult(faces=faces, eyes=eyes, smiles=smiles, bodies=bodies)


def draw_boxes(frame: np.ndarray, boxes: List[Box], color: Tuple[int, int, int], label: str) -> None:
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    out = frame.copy()
    draw_boxes(out, result.faces, (0, 255, 0), "face")
    draw_boxes(out, result.eyes, (255, 255, 0), "eye")
    draw_boxes(out, result.smiles, (0, 165, 255), "smile")
    draw_boxes(out, result.bodies, (255, 0, 255), "body")
    return out


# --------------------------------------------------------------------------- #
# Edge detection
# --------------------------------------------------------------------------- #
def canny_edges(bgr_frame: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    return edges


# --------------------------------------------------------------------------- #
# Color-blob tracking (HSV thresholding + contours)
# --------------------------------------------------------------------------- #
HSV_RANGES = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((36, 60, 60), (89, 255, 255))],
    "blue": [((90, 60, 60), (128, 255, 255))],
    "yellow": [((20, 100, 100), (35, 255, 255))],
}


def detect_color_blobs(bgr_frame: np.ndarray, color: str, min_area: int = 300) -> List[Box]:
    if color not in HSV_RANGES:
        raise ValueError(f"Unsupported color '{color}'. Choose from {list(HSV_RANGES)}")
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in HSV_RANGES[color]:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area >= min_area:
            boxes.append(cv2.boundingRect(c))
    return boxes
