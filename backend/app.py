"""
Backend - Application/Backend Layer of the system architecture.

Responsibilities (Chapter 4 of the synopsis):
    - REST API for detection events and alerts
    - Geo-tagging + alert-engine bookkeeping (severity, dispatch status)
    - Database (SQLite standing in for MongoDB/DynamoDB - zero setup,
      file-based, so the whole project runs with no external DB server)
    - Serves captured/annotated images for the dashboard

Run:  python app.py            (starts on http://localhost:5000)
"""

import os
import sqlite3
import time
import uuid
from flask import Flask, request, jsonify, send_from_directory, g

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "disaster_system.db")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

app = Flask(__name__)

# Simulated list of authorities notified for high-severity events
AUTHORITIES = ["NDRF", "SDRF", "Local Administration"]


# ---------------------------------------------------------------------- #
# Database helpers
# ---------------------------------------------------------------------- #
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            disaster_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            severity TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            bbox TEXT,
            drone_id TEXT,
            mission_id TEXT,
            image_url TEXT,
            timestamp REAL NOT NULL,
            dispatched_to TEXT
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------- #
# Alert engine (Step 5 of the proposed algorithm)
# ---------------------------------------------------------------------- #
def dispatch_alert(disaster_type, severity):
    """Simulated dispatch: high severity -> notify authorities immediately,
    lower severity -> logged for review only. Replace this with real
    SMS/email/push integration (e.g. Firebase Cloud Messaging, Twilio)."""
    if severity == "high":
        return list(AUTHORITIES)
    return []


# ---------------------------------------------------------------------- #
# Routes
# ---------------------------------------------------------------------- #
@app.route("/api/detections", methods=["POST"])
def create_detection():
    data = request.get_json(force=True)
    required = ["disaster_type", "confidence", "latitude", "longitude"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"missing fields: {missing}"}), 400

    severity = data.get("severity", "medium")
    dispatched_to = dispatch_alert(data["disaster_type"], severity)

    record = {
        "id": str(uuid.uuid4()),
        "disaster_type": data["disaster_type"],
        "confidence": float(data["confidence"]),
        "severity": severity,
        "latitude": float(data["latitude"]),
        "longitude": float(data["longitude"]),
        "bbox": str(data.get("bbox", [])),
        "drone_id": data.get("drone_id", "DRONE-01"),
        "mission_id": data.get("mission_id", "MISSION-01"),
        "image_url": data.get("image_url", ""),
        "timestamp": time.time(),
        "dispatched_to": ",".join(dispatched_to),
    }

    db = get_db()
    db.execute("""
        INSERT INTO detections
        (id, disaster_type, confidence, severity, latitude, longitude,
         bbox, drone_id, mission_id, image_url, timestamp, dispatched_to)
        VALUES (:id, :disaster_type, :confidence, :severity, :latitude, :longitude,
                :bbox, :drone_id, :mission_id, :image_url, :timestamp, :dispatched_to)
    """, record)
    db.commit()

    return jsonify({"status": "ok", "detection": record, "dispatched_to": dispatched_to}), 201


@app.route("/api/detections", methods=["GET"])
def list_detections():
    disaster_type = request.args.get("type")
    severity = request.args.get("severity")
    limit = int(request.args.get("limit", 200))

    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if disaster_type:
        query += " AND disaster_type = ?"
        params.append(disaster_type)
    if severity:
        query += " AND severity = ?"
        params.append(severity)
    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    rows = get_db().execute(query, params).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/detections/<detection_id>", methods=["GET"])
def get_detection(detection_id):
    row = get_db().execute("SELECT * FROM detections WHERE id = ?", (detection_id,)).fetchone()
    if row is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(dict(row))


@app.route("/api/alerts", methods=["GET"])
def list_alerts():
    """High-severity detections that were dispatched to authorities."""
    rows = get_db().execute(
        "SELECT * FROM detections WHERE severity = 'high' ORDER BY timestamp DESC LIMIT 100"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/stats", methods=["GET"])
def stats():
    db = get_db()
    total = db.execute("SELECT COUNT(*) c FROM detections").fetchone()["c"]
    by_type = db.execute(
        "SELECT disaster_type, COUNT(*) c FROM detections GROUP BY disaster_type"
    ).fetchall()
    by_severity = db.execute(
        "SELECT severity, COUNT(*) c FROM detections GROUP BY severity"
    ).fetchall()
    return jsonify({
        "total_detections": total,
        "by_type": {r["disaster_type"]: r["c"] for r in by_type},
        "by_severity": {r["severity"]: r["c"] for r in by_severity},
    })


@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "online", "service": "disaster-detection-backend"})


# Serve the dashboard directly from Flask so the whole demo is one process
DASHBOARD_DIR = os.path.join(os.path.dirname(BASE_DIR), "dashboard")


@app.route("/")
def dashboard():
    return send_from_directory(DASHBOARD_DIR, "index.html")


if __name__ == "__main__":
    init_db()
    print("Disaster Detection backend running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
