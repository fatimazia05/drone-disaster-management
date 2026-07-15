"""
run_pipeline.py - runs the full algorithm from Chapter 3.2 of the synopsis:

  Step 1: Drone Initialization and Aerial Patrol   -> frame source (video/webcam/folder)
  Step 2: Image Pre-processing                     -> DisasterDetector.preprocess
  Step 3: Object and Anomaly Detection              -> DisasterDetector.analyze_frame
  Step 4: Disaster Classification and Geo-tagging   -> GeoSimulator
  Step 5: Alert Generation and Dispatch             -> POST /api/detections (backend decides dispatch)
  Step 6: Dashboard Update and User Notification    -> dashboard polls the backend automatically

Usage:
    # Demo mode - runs on the synthetic sample frames (no drone/dataset needed)
    python run_pipeline.py --source ../sample_data/frames --backend http://localhost:5000

    # Real usage - point at a video file or webcam index
    python run_pipeline.py --source path/to/video.mp4
    python run_pipeline.py --source 0
"""

import argparse
import glob
import os
import time

import cv2
import requests

from detector import DisasterDetector, GeoSimulator

CONF_THRESHOLD = 0.55


def iter_frames(source):
    """Yields BGR frames from a folder of images, a video file, or a webcam index."""
    if os.path.isdir(source):
        for path in sorted(glob.glob(os.path.join(source, "*.jpg"))) + \
                    sorted(glob.glob(os.path.join(source, "*.png"))):
            frame = cv2.imread(path)
            if frame is not None:
                yield os.path.basename(path), frame
        return

    try:
        cam_index = int(source)
        cap = cv2.VideoCapture(cam_index)
    except ValueError:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield f"frame_{frame_id:05d}", frame
        frame_id += 1
    cap.release()


def send_to_backend(backend_url, detection, image_path=None):
    payload = dict(detection)
    if image_path:
        payload["image_url"] = f"/images/{os.path.basename(image_path)}"
    try:
        resp = requests.post(f"{backend_url}/api/detections", json=payload, timeout=3)
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  [!] Could not reach backend ({e}). Is app.py running?")
        return None


def main():
    parser = argparse.ArgumentParser(description="Drone disaster detection pipeline")
    parser.add_argument("--source", default="../sample_data/frames",
                         help="Folder of images, path to video file, or webcam index (default: sample frames)")
    parser.add_argument("--backend", default="http://localhost:5000",
                         help="Backend base URL to POST detections to")
    parser.add_argument("--save-annotated", default="../backend/images",
                         help="Directory to save annotated alert images (defaults into the "
                              "backend's images/ folder so the dashboard can display them)")
    parser.add_argument("--no-backend", action="store_true",
                         help="Run detection only, skip sending to backend (prints to console)")
    args = parser.parse_args()

    os.makedirs(args.save_annotated, exist_ok=True)

    detector = DisasterDetector()
    geo = GeoSimulator()

    print(f"[Step 1] Drone patrol started. Source: {args.source}")
    frame_count = 0
    alert_count = 0

    for name, frame in iter_frames(args.source):
        frame_count += 1
        processed, detections = detector.analyze_frame(frame)  # Steps 2 & 3

        if not detections:
            print(f"[{name}] no anomalies detected.")
            continue

        annotated = detector.annotate(processed, detections)

        for d in detections:
            if d["confidence"] < CONF_THRESHOLD:
                continue

            lat, lon = geo.next_position()  # Step 4: geo-tag
            severity = detector.severity_for(d["type"], d["confidence"])

            alert_count += 1
            image_name = f"alert_{alert_count:04d}_{d['type']}.jpg"
            image_path = os.path.join(args.save_annotated, image_name)
            cv2.imwrite(image_path, annotated)

            detection_payload = {
                "disaster_type": d["type"],
                "confidence": d["confidence"],
                "severity": severity,
                "latitude": lat,
                "longitude": lon,
                "bbox": d["bbox"],
                "drone_id": "DRONE-01",
                "mission_id": "MISSION-DEMO",
            }

            print(f"[{name}] DETECTED {d['type']} (conf={d['confidence']}, "
                  f"severity={severity}) at ({lat}, {lon})")

            if not args.no_backend:
                result = send_to_backend(args.backend, detection_payload, image_path)  # Step 5
                if result and result.get("dispatched_to"):
                    print(f"  -> Alert dispatched to: {', '.join(result['dispatched_to'])}")
                elif result:
                    print("  -> Logged for review (below high-severity threshold)")

        time.sleep(0.1)  # simulate real-time frame interval

    print(f"\n[Step 6] Patrol complete. {frame_count} frames processed, "
          f"{alert_count} alert(s) generated.")
    print("Open the dashboard (http://localhost:5000) to view live results.")


if __name__ == "__main__":
    main()
