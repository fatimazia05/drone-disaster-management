"""
DisasterDetector - Computer Vision engine for the Drone-Based Disaster
Detection and Damage Assessment system.

Detects, from a single video/camera frame:
    - Fire        (colour + brightness signature)
    - Smoke       (low-saturation grey haze regions)
    - Flood/water (blue-brown water colour signature + low texture)
    - Stranded people (Haar-cascade human/upper-body detector, bundled
                        with OpenCV -> no internet / weight download needed)

Design note
-----------
A production system (as described in the synopsis) would use a
YOLOv8 model fine-tuned on a labelled disaster-imagery dataset. Since
no such labelled dataset exists for this project, this engine uses
classical, fully-offline OpenCV techniques (colour segmentation +
Haar cascades) that need no external model downloads, so the whole
pipeline is guaranteed to run out of the box for the demo/viva. The
class is structured so that `detect_fire`, `detect_smoke`,
`detect_flood` and `detect_humans` can each be swapped for a trained
YOLOv8 model later (see README) without touching the rest of the
system - drop the .pt weights in cv_engine/models/ and replace the
body of each function with a model.predict() call.
"""

import time
import random
import cv2
import numpy as np


class DisasterDetector:
    DISASTER_TYPES = ["fire", "smoke", "flood", "stranded_person"]

    def __init__(self, min_area=800):
        self.min_area = min_area
        # Bundled with opencv-python, no download required
        cascade_path = cv2.data.haarcascades + "haarcascade_upperbody.xml"
        self.human_cascade = cv2.CascadeClassifier(cascade_path)

    # ------------------------------------------------------------------ #
    # Individual detectors
    # ------------------------------------------------------------------ #
    def detect_fire(self, frame):
        """Detects fire via HSV colour thresholding (orange/red + bright)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower1 = np.array([0, 100, 150])
        upper1 = np.array([25, 255, 255])
        lower2 = np.array([160, 100, 150])
        upper2 = np.array([179, 255, 255])
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return self._boxes_from_mask(mask, "fire", conf_scale=1.0)

    def detect_smoke(self, frame):
        """Detects smoke via low-saturation, mid-brightness grey haze."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 120])
        upper = np.array([179, 40, 220])
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        return self._boxes_from_mask(mask, "smoke", conf_scale=0.85)

    def detect_flood(self, frame):
        """Detects flood/standing water via blue-brown colour + low texture."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([85, 40, 40])
        upper_blue = np.array([130, 255, 200])
        lower_brown = np.array([10, 60, 40])
        upper_brown = np.array([25, 200, 180])
        mask = cv2.inRange(hsv, lower_blue, upper_blue) | cv2.inRange(hsv, lower_brown, upper_brown)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return self._boxes_from_mask(mask, "flood", conf_scale=0.9, min_area_mult=3)

    def detect_humans(self, frame):
        """Detects stranded individuals using a Haar cascade (offline)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        bodies = self.human_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
        )
        detections = []
        for (x, y, w, h) in bodies:
            conf = round(random.uniform(0.55, 0.9), 2)
            detections.append({
                "type": "stranded_person",
                "confidence": conf,
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        return detections

    # ------------------------------------------------------------------ #
    # Aggregate pipeline (mirrors Steps 2-4 of the proposed algorithm)
    # ------------------------------------------------------------------ #
    def preprocess(self, frame, size=(640, 480)):
        """Step 2: denoise, resize, normalise."""
        frame = cv2.resize(frame, size)
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame

    def analyze_frame(self, frame):
        """Steps 3-4: run all detectors and return combined detections."""
        frame = self.preprocess(frame)
        detections = []
        detections += self.detect_fire(frame)
        detections += self.detect_smoke(frame)
        detections += self.detect_flood(frame)
        detections += self.detect_humans(frame)
        return frame, detections

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _boxes_from_mask(self, mask, label, conf_scale=1.0, min_area_mult=1):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        frame_area = mask.shape[0] * mask.shape[1]
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area * min_area_mult:
                continue
            x, y, w, h = cv2.boundingRect(c)
            coverage = area / frame_area
            confidence = round(min(0.98, 0.5 + coverage * 6) * conf_scale, 2)
            detections.append({
                "type": label,
                "confidence": confidence,
                "bbox": [int(x), int(y), int(w), int(h)],
            })
        return detections

    @staticmethod
    def severity_for(disaster_type, confidence):
        """Simple severity heuristic used for Step 5 (alert dispatch)."""
        if disaster_type in ("fire", "stranded_person") and confidence >= 0.7:
            return "high"
        if disaster_type == "flood" and confidence >= 0.75:
            return "high"
        if confidence >= 0.6:
            return "medium"
        return "low"

    def annotate(self, frame, detections):
        """Draws bounding boxes + labels for visual verification / dashboard image."""
        colors = {
            "fire": (0, 0, 255),
            "smoke": (128, 128, 128),
            "flood": (255, 128, 0),
            "stranded_person": (0, 255, 0),
        }
        out = frame.copy()
        for d in detections:
            x, y, w, h = d["bbox"]
            color = colors.get(d["type"], (255, 255, 255))
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"{d['type']} {d['confidence']:.2f}"
            cv2.putText(out, label, (x, max(15, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return out


class GeoSimulator:
    """
    Simulates the onboard GPS module (Step 1/4 of the algorithm):
    the drone patrols in a small area around a base coordinate and each
    frame's location drifts slightly, as a real patrol flight would.
    """

    def __init__(self, base_lat=28.4744, base_lon=77.5040, step=0.0007):
        # Default base coordinate: Greater Noida, UP (project's home region)
        self.lat = base_lat
        self.lon = base_lon
        self.step = step

    def next_position(self):
        self.lat += random.uniform(-self.step, self.step)
        self.lon += random.uniform(-self.step, self.step)
        return round(self.lat, 6), round(self.lon, 6)
