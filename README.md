<!--
README structure intentionally mirrors a production portfolio.
Keep screenshots in /assets and update image paths as needed.
-->

## [Hero Section]

# RoadGuard AI – Road Safety Intelligence Platform

**AI-powered road damage + hazard intelligence** built with **Flask** and **YOLOv8** to detect infrastructure issues (potholes, cracks, rutting, patches), evaluate multi-threat risk (traffic, animals, accidents), and trigger **GPS-based SOS dispatch** with **real-time alerts**.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-API%20Server-000000?logo=flask&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111827)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Geospatial%20Data-47A248?logo=mongodb&logoColor=white)
![SSE](https://img.shields.io/badge/Realtime-SSE%20Alerts-0EA5E9)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-success)

---

## [Why This Project Matters]

Road defects and sudden hazards are not just “bad infrastructure” problems—they’re **preventable safety failures**.

- **Infrastructure risk**: undetected road damage increases accidents, vehicle damage, and maintenance costs.
- **Operational visibility**: cities and fleets need **where/when** damage occurs, not just raw detections.
- **Emergency response**: in high-risk events, seconds matter—**GPS + SOS dispatch** and **real-time alert streaming** improves response coordination.

RoadGuard AI is designed to look and feel like a production-grade platform: detection → analytics → geospatial visualization → live monitoring → emergency workflow.

---

## [Core Features]

### Road Damage Detection (YOLOv8)

- Detects **potholes, cracks, rutting, patches**
- Supports:
  - **Image inference** (REST upload)
  - **Video inference** (uploaded file)
  - **Live webcam monitoring** with annotated frames
- Stores detections with metadata (time, severity, damage breakdown)

### Multi‑Threat Road Intelligence

- Multi-factor report per frame:
  - **Road damage severity**
  - **Traffic density**
  - **Animal presence**
  - **Accident likelihood**
- Optional enrichment (when available):
  - **Road score**
  - **Repair cost estimate**
  - **Pothole depth approximation**
  - **Weather / low-light risk signals**

### Real‑Time Monitoring & Alerts

- **Server-Sent Events (SSE)** stream for alert delivery
- Heartbeat-enabled stream for stable connections
- Integrates alert sending logic via `alerts.py`

### SOS Emergency System (GPS + Twilio)

- **Live GPS updates** per user
- **SOS trigger** endpoint with configurable notify mode:
  - voice call
  - SMS
  - both
  - none
- MongoDB-backed geospatial targeting using **2dsphere** queries for nearby users (when enabled)

### Analytics & Visualization

- Dashboard endpoints for:
  - aggregate stats
  - damage type breakdown
  - daily trend chart
- Heatmap view powered by stored detection geolocations

---

## [Tech Stack]

- **Backend**: Flask, Werkzeug
- **AI / CV**: Ultralytics YOLOv8, OpenCV, Pillow
- **Database**: MongoDB (local or Atlas), PyMongo, geospatial indexing (`2dsphere`)
- **Realtime**: Server-Sent Events (SSE)
- **Alerts / Comms**: Twilio (Voice + SMS), Slack webhook (optional), Requests
- **DevOps**: Docker, Docker Compose

---

## [Project Structure]

```text
road-safety/
  pothole_detection/
    app.py                  # Flask app + API routes + SSE + webcam feed
    detect.py                # YOLOv8 road damage inference + persistence
    multidetect.py           # Multi-threat analysis (traffic/animals/accident)
    road_intelligence.py     # Scoring + cost + depth + weather risk helpers
    database.py              # MongoDB + geospatial indexes + emergency persistence
    alerts.py                # Alert publish/subscribe + sending integrations
    gps_service.py           # GPS update helpers + latest-location logic
    emergency_system.py      # SOS dispatch + routing suggestions
    voice_service.py         # Twilio TwiML + voice status callbacks
    seed_data.py             # Optional seed utilities
    templates/
      dashboard.html
      heatmap.html
      live.html
      sos.html
      index.html
  requirements.txt
  Dockerfile
  docker-compose.yml
  LICENSE
  README.md
  assets/
    dashboard.png
    live-detection.gif
    heatmap.png
    sos.png
```

---

## [Getting Started]

### Prerequisites

- **Python 3.10+**
- **MongoDB** (Atlas or local) and a valid `MONGO_URI`
- Model files present:
  - `pothole_detection/road_damage.pt`
  - `yolov8n.pt` (repo root)

### Installation

```bash
python -m venv venv
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
```

Install deps:

```bash
pip install -r requirements.txt
```

### Environment variables

Create `.env` in the repository root:

```env
# Database
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority

# Optional: fixed emergency destinations
POLICE_TO=+91XXXXXXXXXX
HOSPITAL_TO=+91XXXXXXXXXX
EMERGENCY_RADIUS_M=1000

# Twilio (voice + SMS)
TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM=+1XXXXXXXXXX
PUBLIC_BASE_URL=https://<public-url-for-callbacks>
VOICE_PROVIDER=twilio
VOICE_LANGUAGE=en-IN
MAX_CALL_RETRIES=3
RETRY_CALL_STATUSES=failed,busy,no-answer,canceled,timeout

# Optional generic alerts
ALERT_TO=+91XXXXXXXXXX
SLACK_WEBHOOK=https://hooks.slack.com/services/xxx/yyy/zzz

# Optional webcam emergency cooldown
ACCIDENT_EMERGENCY_COOLDOWN_SEC=300
```

---

## [Usage]

### Run locally

```bash
python pothole_detection/app.py
```

Open the UI:

- **Dashboard**: `http://127.0.0.1:5000/`
- **Live camera**: `http://127.0.0.1:5000/live`
- **Heatmap**: `http://127.0.0.1:5000/heatmap`
- **SOS**: `http://127.0.0.1:5000/sos`

### Run with Docker

```bash
docker compose up --build
```

App: `http://127.0.0.1:5000/`

---

## [API Reference]

Base URL (local): `http://127.0.0.1:5000`

### Detection

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/detect/image` | Image road-damage detection (supports optional `lat`,`lng`) |
| POST | `/api/detect/video` | Video detection pipeline |
| POST | `/api/detect/multithreat` | Damage + traffic + animals + accident risk report |
| POST | `/api/enhance` | Low-light enhancement (only when enhancement module is available) |

### Analytics & Data

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/stats` | Global severity + damage totals |
| GET | `/api/damage/breakdown` | Totals per damage type |
| GET | `/api/detections?severity=&limit=` | Recent detections (filterable) |
| GET | `/api/chart/daily?days=7` | Daily detection trend |

### Webcam

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/webcam/start` | Start webcam worker thread |
| POST | `/api/webcam/stop` | Stop webcam |
| GET | `/api/webcam/feed` | MJPEG annotated frame stream |

### Alerts, GPS, SOS, Voice

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/alerts/stream?user_id=` | SSE: real-time alert stream |
| POST | `/api/gps/update` | Upsert live user GPS (and optional contacts) |
| GET | `/api/gps/latest/<user_id>` | Latest location for user |
| GET | `/api/gps/latest?user_id=&max_age_seconds=180` | Latest location for user (query form) |
| POST | `/api/sos/trigger` | Trigger SOS (`notify_mode=call|sms|both|none`) |
| GET | `/api/route/suggest` | Best-effort route suggestion |
| GET | `/api/voice/twiml` | Twilio TwiML (`<Say>`) endpoint |
| POST/GET | `/api/voice/status` | Twilio status callback for retry/fail-safe |

### Quick examples

Update live GPS:

```bash
curl -X POST http://127.0.0.1:5000/api/gps/update \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"u1\",\"lat\":23.5204,\"lng\":87.3119,\"phone_number\":\"+91XXXXXXXXXX\"}"
```

Trigger SOS (call + SMS):

```bash
curl -X POST http://127.0.0.1:5000/api/sos/trigger \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"u1\",\"lat\":23.5204,\"lng\":87.3119,\"emergency_type\":\"accident\",\"severity\":\"high\",\"notify_mode\":\"both\"}"
```

---

## [Screenshots / Demo]


**Dashboard**
<img width="1919" height="866" alt="Screenshot 2026-03-26 111125" src="https://github.com/user-attachments/assets/2de51d2b-2b03-4564-ac21-5d512035b0d0" />
![Road threat scan](https://github.com/user-attachments/assets/ec55065a-5dc7-46bd-ac84-008837a0fb94)

**Live detection (webcam)**

![Live Detection Demo](assets/live-detection.gif)
<img width="1918" height="864" alt="Screenshot 2026-03-26 111301" src="https://github.com/user-attachments/assets/2f090d0c-ebb7-4d39-8603-1a0e83182515" />

**Heatmap visualization**

<img width="1919" height="861" alt="Screenshot 2026-03-26 120240" src="https://github.com/user-attachments/assets/2d20122f-9e47-44de-be8a-a7a9884aa1db" />

**SOS & emergency dispatch**

<img width="1919" height="858" alt="Screenshot 2026-03-26 111603" src="https://github.com/user-attachments/assets/1e42c667-80c7-44a5-9278-db385f8794aa" />

### 🚨 SOS Incoming Call
<p align="center">
  <img src="https://github.com/user-attachments/assets/0e9cc823-346e-48ad-9b90-434c827594cb" width="250"/>
</p>

### 📩 SOS Message


## [Performance & Architecture]

- **Single-load inference**: YOLO models are loaded once at startup to reduce per-request latency.
- **Fail-safe emergency workflow**: emergency dispatch is best-effort and designed not to break detection endpoints when providers are not configured.
- **Real-time streaming**:
  - MJPEG webcam feed for annotated frames
  - SSE for alert streaming with heartbeat keep-alive
- **Geospatial readiness**: MongoDB `2dsphere` index on `user_locations.loc` enables radius-based targeting and queries.
- **Operational knobs**:
  - `EMERGENCY_RADIUS_M` for SOS targeting radius
  - `ACCIDENT_EMERGENCY_COOLDOWN_SEC` to prevent repeat triggers from continuous webcam loops

---

## [Troubleshooting]

- **`No recent location for this user`**
  - Call `POST /api/gps/update` first.
  - Increase `max_age_seconds` in `GET /api/gps/latest`.

- **Twilio voice call cannot fetch TwiML / callbacks fail**
  - Ensure `PUBLIC_BASE_URL` is publicly reachable (use a tunnel such as ngrok for local dev).

- **Mongo connection errors**
  - Verify `MONGO_URI` is set and valid.
  - Confirm the DB network access rules (Atlas) allow your IP.

- **Model load failures**
  - Confirm these files exist:
    - `pothole_detection/road_damage.pt`
    - `yolov8n.pt` (repo root)

- **`/upload` route errors**
  - The Flask app exposes `/upload`, but the template `pothole_detection/templates/upload.html` may be missing. Add it (or remove the route) if you want an upload page UI.

- **Screenshots not rendering**
  - Ensure `assets/` exists and filenames match:
    - `assets/dashboard.png`
    - `assets/live-detection.gif`
    - `assets/heatmap.png`
    - `assets/sos.png`

---

## [Roadmap]

- Add authentication + role-based dashboards (city admin / fleet operator)
- Map-based “nearest hospital/police” lookup via routing APIs (instead of fixed numbers)
- Async job queue for heavy video workloads (Celery/RQ)
- Model monitoring + drift checks and structured observability (logs/metrics/traces)
- Exportable reports (CSV/PDF) for municipal maintenance workflows

---

## [Contributing]

Contributions are welcome.

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-change`
3. Commit with a clear message
4. Open a pull request with:
   - what changed
   - why it matters
   - how to test

If you’re adding new endpoints, please update the **[API Reference]** section above.

---

## [License]

This project is licensed under the terms of the `LICENSE` file in the repository root.

