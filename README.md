# RoadGuard AI - Road Safety Intelligence Platform

RoadGuard AI is an end-to-end Flask + YOLOv8 platform for:

- Road damage detection (`pothole`, `crack`, `patch`, `rutting`)
- Multi-threat analysis (damage, traffic density, animals, accident risk)
- Live webcam monitoring with alert streaming
- GPS-based SOS dispatch (nearby users, police/hospital contacts)
- Emergency notifications via voice call and SMS (Twilio)
- Dashboard + heatmap + live feed UI

---

## Features

- Real-time and batch inference
  - Image upload detection
  - Video frame-sampled detection
  - Webcam live feed with annotated frames
- Geo-aware workflows
  - Save user live GPS (`/api/gps/update`)
  - Fetch latest location by user (`/api/gps/latest`)
  - SOS trigger with radius-based nearby targeting
- Emergency dispatch pipeline
  - Browser SSE emergency alerts
  - Voice calls with retry logic
  - SMS dispatch (best-effort) via Twilio
- Analytics
  - Severity stats
  - Damage breakdown
  - Daily chart API
  - Heatmap view from stored geolocations

---

## Tech Stack

- Backend: Flask, PyMongo
- AI/CV: Ultralytics YOLOv8, OpenCV, Pillow
- Data: MongoDB (Atlas/local)
- Alerts: Twilio (voice + SMS), Slack webhook, SSE
- Frontend: HTML templates (dashboard/live/heatmap/sos)

---

## Project Structure

```text
road-safety/
  pothole_detection/
    app.py
    detect.py
    multidetect.py
    gps_service.py
    emergency_system.py
    voice_service.py
    database.py
    alerts.py
    templates/
  requirements.txt
  Dockerfile
  docker-compose.yml
```

---

## Prerequisites

- Python 3.10+
- MongoDB URI (Atlas/local)
- Model files present:
  - `pothole_detection/road_damage.pt`
  - `yolov8n.pt` (repo root)

---

## Local Setup

### 1) Clone and create virtual environment

```bash
git clone https://github.com/<your-username>/road-safety.git
cd road-safety
python -m venv venv
```

Windows:

```bat
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create `.env`

Create a `.env` in project root:

```env
# Database
MONGO_URI=mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority

# Optional: emergency destination numbers
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

### 4) Run the app

```bash
python pothole_detection/app.py
```

Open:

- Dashboard: `http://127.0.0.1:5000`
- Upload: `http://127.0.0.1:5000/upload`
- Live Feed: `http://127.0.0.1:5000/live`
- Heatmap: `http://127.0.0.1:5000/heatmap`
- SOS: `http://127.0.0.1:5000/sos`

---

## Docker

### Build and run

```bash
docker compose up --build
```

App: `http://127.0.0.1:5000`

---

## Key API Endpoints

### Detection

- `POST /api/detect/image` - image road-damage detection
- `POST /api/detect/multithreat` - multi-threat analysis + optional emergency dispatch
- `POST /api/detect/video` - video processing
- `POST /api/enhance` - night/low-light enhancement (if module available)

### Dashboard/Data

- `GET /api/stats`
- `GET /api/damage/breakdown`
- `GET /api/detections?severity=<none|low|medium|high>&limit=50`
- `GET /api/chart/daily?days=7`

### Webcam

- `POST /api/webcam/start`
- `POST /api/webcam/stop`
- `GET /api/webcam/feed`

### Alerts + GPS + SOS

- `GET /api/alerts/stream?user_id=<id>` (SSE)
- `POST /api/gps/update`
- `GET /api/gps/latest/<user_id>`
- `GET /api/gps/latest?user_id=<user_id>&max_age_seconds=180`
- `POST /api/sos/trigger` (`notify_mode=call|sms|both|none`)
- `GET /api/route/suggest`

### Voice callbacks

- `GET /api/voice/twiml`
- `POST|GET /api/voice/status`

---

## Quick API Examples

### Update live GPS

```bash
curl -X POST http://127.0.0.1:5000/api/gps/update \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"u1\",\"lat\":23.5204,\"lng\":87.3119,\"phone_number\":\"+91XXXXXXXXXX\"}"
```

### Fetch latest GPS for user

```bash
curl "http://127.0.0.1:5000/api/gps/latest/u1"
```

### Trigger SOS (call + SMS)

```bash
curl -X POST http://127.0.0.1:5000/api/sos/trigger \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"u1\",\"lat\":23.5204,\"lng\":87.3119,\"emergency_type\":\"accident\",\"severity\":\"high\",\"notify_mode\":\"both\"}"
```

---

## Performance and Reliability Notes

- YOLO models are loaded once at startup to reduce per-request latency.
- Emergency dispatch runs calls/SMS in background threads.
- Voice callback status supports retry logic (`MAX_CALL_RETRIES`).
- Geospatial queries use MongoDB `2dsphere` index on `user_locations.loc`.
- SOS radius targeting defaults to `1000m` (`EMERGENCY_RADIUS_M`).

---

## Troubleshooting

- `No recent location for this user`
  - Call `/api/gps/update` first.
  - Check `max_age_seconds` in `/api/gps/latest`.
- Voice call says it cannot fetch TwiML
  - `PUBLIC_BASE_URL` must be publicly reachable (use tunnel/ngrok in local dev).
- `MONGO_URI` errors
  - Verify `.env` is loaded and URI is valid.
- Model load failure
  - Confirm `pothole_detection/road_damage.pt` and `yolov8n.pt` exist.

---

## Roadmap Ideas

- Nearest police/hospital lookup by map APIs (instead of fixed numbers)
- Auth + role-based dashboards
- Structured logs + monitoring
- Async job queue for heavy video workloads

---

## License

This project includes a `LICENSE` file in the repository root.

