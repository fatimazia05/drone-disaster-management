# Drone-Based Computer Vision System for Disaster Detection and Damage Assessment

A working, offline-runnable implementation of the system described in the
project synopsis. It covers the full pipeline from Chapter 3.2:

1. **Drone Initialization & Aerial Patrol** — frame source (video / webcam / image folder)
2. **Image Pre-processing** — denoise, resize, normalize
3. **Object & Anomaly Detection** — fire, smoke, flood, and stranded-person detection
4. **Disaster Classification & Geo-tagging** — simulated onboard GPS
5. **Alert Generation & Dispatch** — REST API decides severity and simulates NDRF/SDRF dispatch
6. **Dashboard Update & Notification** — live web dashboard (map + alert feed + log)

## Why this design

The synopsis specifies YOLOv8 + a labelled disaster-imagery dataset. Since no
such dataset exists for this project, the **detection engine uses classical,
fully-offline OpenCV techniques** (HSV colour segmentation for fire/smoke/flood,
a bundled Haar cascade for people) instead of a downloaded pretrained model.
This means the whole project runs with **zero internet dependency and zero
GPU requirement** — important for a reliable viva/demo. `cv_engine/detector.py`
is structured so each detector can be swapped for a trained YOLOv8 model later
(drop weights into `cv_engine/models/`, replace the body of the `detect_*`
methods with `model.predict()` calls) without touching the rest of the system.

Similarly, **SQLite stands in for MongoDB/DynamoDB** — no database server to
install, the whole backend is one Python file.

## Project structure

```
drone_disaster_project/
├── cv_engine/
│   ├── detector.py          # DisasterDetector + GeoSimulator (Steps 2-4)
│   ├── run_pipeline.py      # End-to-end pipeline runner (Steps 1-6)
│   └── requirements.txt
├── backend/
│   ├── app.py               # Flask + SQLite REST API, alert engine (Step 5), serves dashboard
│   └── requirements.txt
├── dashboard/
│   └── index.html           # Live dashboard: map, alert feed, detection log (Step 6)
├── sample_data/
│   └── create_sample_data.py  # Generates synthetic demo frames (fire/flood/smoke/person)
└── README.md
```

## Setup

```bash
cd drone_disaster_project
pip install -r backend/requirements.txt
pip install -r cv_engine/requirements.txt
```

(Both `requirements.txt` files only need Flask, OpenCV, NumPy, and Requests —
no heavy ML frameworks, so install is quick.)

## Run the demo (3 terminals / steps)

**1. Generate synthetic demo frames** (skip this if you have your own drone
footage or images — see "Using real footage" below):

```bash
cd sample_data
python create_sample_data.py
```

**2. Start the backend** (serves the API *and* the dashboard on the same port):

```bash
cd backend
python app.py
```

Leave this running. Open **http://localhost:5000** in a browser — you'll see
the live dashboard (empty until Step 3 runs).

**3. Run the detection pipeline** (in a new terminal):

```bash
cd cv_engine
python run_pipeline.py --source ../sample_data/frames
```

Watch the terminal print each detection, its confidence, severity, and
simulated GPS coordinates, and whether it was dispatched to NDRF/SDRF. Refresh
the dashboard — the map, stat cards, alert feed, and detection log update
automatically (it polls every 4 seconds).

## Using real footage

```bash
# On a video file
python run_pipeline.py --source path/to/drone_video.mp4

# On a live webcam (acting as the "drone camera" for a demo)
python run_pipeline.py --source 0
```

## API reference (backend/app.py)

| Method | Endpoint                | Description                                   |
|--------|--------------------------|------------------------------------------------|
| POST   | `/api/detections`        | Submit a detection event (used by the pipeline) |
| GET    | `/api/detections`        | List detections (`?type=`, `?severity=`, `?limit=`) |
| GET    | `/api/detections/<id>`   | Get one detection                              |
| GET    | `/api/alerts`            | High-severity detections dispatched to authorities |
| GET    | `/api/stats`             | Totals by type/severity                        |
| GET    | `/images/<filename>`     | Annotated alert image                          |
| GET    | `/api/health`            | Health check                                   |

## Extending toward the full synopsis scope

- **Real YOLOv8 model**: replace `detect_fire`/`detect_smoke`/`detect_flood`/
  `detect_humans` in `detector.py` with `ultralytics` YOLOv8 inference once a
  labelled dataset (e.g. from Roboflow's public wildfire/flood datasets) is
  available and trained.
- **Real GPS**: replace `GeoSimulator` with a DroneKit/MAVLink telemetry read.
- **Real notifications**: replace `dispatch_alert()` in `app.py` with
  Firebase Cloud Messaging (push), Twilio (SMS), and SMTP (email).
- **Production DB**: swap SQLite for MongoDB by replacing the `sqlite3` calls
  in `app.py` with `pymongo` — the API surface stays identical.
- **Mobile app**: the same REST API can be consumed by a React Native app
  (Chapter 5 of the synopsis) — the endpoints above are already
  mobile-friendly JSON.
