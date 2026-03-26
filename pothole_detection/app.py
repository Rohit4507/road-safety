# ================================================================
# app.py  —  RoadGuard AI  —  Flask server
# ================================================================

from flask import Flask, request, jsonify, render_template, Response, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
import os, threading, cv2

from detect         import detect_image, detect_video, model, parse_results, RESULTS_DIR
from database       import (save_detection, get_all_detections, get_by_severity,
                             get_summary_stats, get_damage_breakdown, detections_col)
from multidetect    import analyse_frame
from alerts         import send_alert, subscribe_alerts, unsubscribe_alerts

# Optional — load if available
try:
    from modules.vision_enhance   import enhance_image, detect_conditions
    ENHANCE_AVAILABLE = True
except ImportError:
    try:
        from vision_enhance import enhance_image, detect_conditions
        ENHANCE_AVAILABLE = True
    except ImportError:
        ENHANCE_AVAILABLE = False

try:
    from modules.road_intelligence import road_score, repair_cost, pothole_depth, weather_risk
    INTELLIGENCE_AVAILABLE = True
except ImportError:
    try:
        from road_intelligence import road_score, repair_cost, pothole_depth, weather_risk
        INTELLIGENCE_AVAILABLE = True
    except ImportError:
        INTELLIGENCE_AVAILABLE = False

# ── App config ────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024   # 200 MB

ALLOWED_IMAGE = {"jpg", "jpeg", "png", "bmp", "webp"}
ALLOWED_VIDEO = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs("uploads",         exist_ok=True)
os.makedirs(RESULTS_DIR,       exist_ok=True)
os.makedirs("results/enhanced", exist_ok=True)

webcam_active = False
latest_frame  = None
frame_lock    = threading.Lock()

# Prevent repeated accident SOS triggers from the webcam loop.
_last_accident_emergency_ts = 0.0
_ACCIDENT_EMERGENCY_COOLDOWN_SEC = int(os.getenv("ACCIDENT_EMERGENCY_COOLDOWN_SEC", "300"))

def allowed_img(f): return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_IMAGE
def allowed_vid(f): return "." in f and f.rsplit(".",1)[1].lower() in ALLOWED_VIDEO


# ── Pages ─────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/upload")
def upload_page():
    return render_template("upload.html")

@app.route("/live")
def live_page():
    return render_template("live.html")

@app.route("/heatmap")
def heatmap_page():
    return render_template("heatmap.html")

@app.route("/sos")
def sos_page():
    return render_template("sos.html")


# ── Image detection ───────────────────────────────────────────────
@app.route("/api/detect/image", methods=["POST"])
def api_detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not allowed_img(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Auto night-vision enhance if available
    enhanced_info = {"enhanced": False, "output_path": path, "conditions": []}
    if ENHANCE_AVAILABLE:
        enhanced_info = enhance_image(path)
    detect_path = enhanced_info["output_path"] if enhanced_info["enhanced"] else path

    lat = request.form.get("lat")
    lng = request.form.get("lng")
    location = {"lat": float(lat), "lng": float(lng)} if lat and lng else {}

    try:
        summary = detect_image(detect_path, location=location)

        # Road intelligence enrichment
        response = {
            "success":        True,
            "pothole_count":  summary["pothole_count"],
            "total_damage":   summary["total_damage"],
            "severity":       summary["severity"],
            "damage_counts":  summary["damage_counts"],
            "damage_summary": summary["damage_summary"],
            "inference_time": summary["inference_time"],
            "result_image":   f"/results/detected_{filename}",
            "enhanced":       enhanced_info["enhanced"],
            "conditions":     enhanced_info.get("conditions", []),
            "doc_id":         summary["doc_id"],
        }

        if INTELLIGENCE_AVAILABLE:
            potholes_list = [{"box": b} for b in summary["boxes"]]
            rs  = road_score(
                potholes = summary["pothole_count"],
                severity = summary["severity"],
            )
            rc  = repair_cost(potholes_list, summary["severity"])
            depths = [
                pothole_depth(b, c, cv2.imread(detect_path).shape)
                for b, c in zip(summary["boxes"], summary["confidences"])
            ]
            # Weather risk from image
            cond = detect_conditions(path) if ENHANCE_AVAILABLE else {}
            wx = weather_risk(
                brightness = cond.get("brightness", 128),
                saturation = cond.get("saturation", 100),
                contrast   = cond.get("contrast",   50),
                blur       = cond.get("blur_score", 100),
            )
            response.update({
                "road_score":   rs,
                "repair_cost":  rc,
                "depths":       depths,
                "weather_risk": wx,
            })

        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Multi-threat detection ────────────────────────────────────────
@app.route("/api/detect/multithreat", methods=["POST"])
def api_multithreat():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if not allowed_img(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Auto enhance
    enhanced_info = {"enhanced": False, "output_path": path, "conditions": []}
    if ENHANCE_AVAILABLE:
        enhanced_info = enhance_image(path)
    detect_path = enhanced_info["output_path"] if enhanced_info["enhanced"] else path

    lat = request.form.get("lat")
    lng = request.form.get("lng")
    location = {"lat": float(lat), "lng": float(lng)} if lat and lng else {}

    try:
        report = analyse_frame(detect_path, location=location)
        report["enhanced"]   = enhanced_info["enhanced"]
        report["conditions"] = enhanced_info.get("conditions", [])

        # Enrich with road intelligence
        if INTELLIGENCE_AVAILABLE:
            pot_count = report["potholes"]["count"]
            severity  = report["potholes"]["severity"]
            vehicles  = report["traffic"]["vehicle_count"]
            animals   = report["animals"]["total_count"]
            acc_risk  = report["accident"]["accident_likelihood"]

            rs = road_score(
                potholes      = pot_count,
                severity      = severity,
                vehicles      = vehicles,
                animal        = animals > 0,
                accident_risk = acc_risk,
            )
            rc = repair_cost(
                [{"box": b} for b in report["potholes"]["boxes"]],
                severity
            )
            depths = [
                pothole_depth(b, s, cv2.imread(detect_path).shape)
                for b, s in zip(
                    report["potholes"]["boxes"],
                    report["potholes"]["scores"]
                )
            ]
            cond = detect_conditions(path) if ENHANCE_AVAILABLE else {}
            wx   = weather_risk(
                brightness = cond.get("brightness", 128),
                saturation = cond.get("saturation", 100),
                contrast   = cond.get("contrast",   50),
                blur       = cond.get("blur_score", 100),
            )
            report.update({
                "road_score":   rs,
                "repair_cost":  rc,
                "depths":       depths,
                "weather_risk": wx,
            })

        # Accident likelihood -> emergency dispatch (police/hospital/family/nearby + voice)
        try:
            acc = report.get("accident") or {}
            if acc.get("alert_needed") and location and "lat" in location and "lng" in location:
                from emergency_system import trigger_emergency
                trigger_emergency(
                    emergency_type="accident",
                    severity=acc.get("accident_likelihood", "high"),
                    location=location,
                    triggered_by_user_id=None,
                    metadata={"source": "camera_based_multithreat_scan"},
                    family_numbers=[],
                    voice_call=True,
                )
        except Exception:
            # Never fail the scan endpoint because emergency provider isn't configured.
            pass

        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Night vision only ─────────────────────────────────────────────
@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    if not ENHANCE_AVAILABLE:
        return jsonify({"error": "vision_enhance.py not found"}), 400
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400

    file = request.files["image"]
    if not allowed_img(file.filename):
        return jsonify({"error": "Invalid type"}), 400

    filename = secure_filename(file.filename)
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    cond   = detect_conditions(path)
    force  = request.form.get("force", "false").lower() == "true"
    result = enhance_image(path, force=force)

    return jsonify({
        "success":       True,
        "enhanced":      result["enhanced"],
        "conditions":    result.get("conditions", []),
        "fixes_applied": result.get("fixes_applied", []),
        "processing_ms": result.get("ms", 0),
        "original_url":  f"/uploads/{filename}",
        "enhanced_url":  f"/results/enhanced/enhanced_{filename}" if result["enhanced"] else f"/uploads/{filename}",
        "compare_url":   f"/results/enhanced/compare_{filename}"  if result["enhanced"] else None,
        "quality": {
            "brightness": cond["brightness"],
            "contrast":   cond["contrast"],
            "blur_score": cond["blur_score"],
            "saturation": cond["saturation"],
        }
    })


# ── Video ─────────────────────────────────────────────────────────
@app.route("/api/detect/video", methods=["POST"])
def api_detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video"}), 400
    file = request.files["video"]
    if not allowed_vid(file.filename):
        return jsonify({"error": "Invalid type"}), 400
    filename = secure_filename(file.filename)
    path     = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)
    try:
        summary = detect_video(path)
        return jsonify({"success": True, **summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Data APIs ─────────────────────────────────────────────────────
@app.route("/api/stats")
def api_stats():
    return jsonify(get_summary_stats())

@app.route("/api/damage/breakdown")
def api_damage_breakdown():
    return jsonify(get_damage_breakdown())

@app.route("/api/detections")
def api_detections():
    sev   = request.args.get("severity")
    limit = int(request.args.get("limit", 50))
    data  = get_by_severity(sev) if sev else get_all_detections(limit)
    return jsonify(data)

@app.route("/api/chart/daily")
def api_chart_daily():
    days   = int(request.args.get("days", 7))
    result = []
    for i in range(days - 1, -1, -1):
        day_start = datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0) - timedelta(days=i)
        day_end   = day_start + timedelta(days=1)
        count     = detections_col.count_documents({"timestamp": {"$gte": day_start, "$lt": day_end}})
        result.append({"date": day_start.strftime("%b %d"), "count": count})
    return jsonify(result)


# ── Webcam ────────────────────────────────────────────────────────
def webcam_worker(camera_id=0, save_interval=30):
    global latest_frame, webcam_active
    cap       = cv2.VideoCapture(camera_id)
    frame_num = 0
    while webcam_active:
        ret, frame = cap.read()
        if not ret: break
        frame_num += 1
        results   = model(frame, verbose=False)[0]
        annotated = results.plot()
        boxes, confs, class_names = parse_results(results)
        from detect import get_damage_severity, get_damage_summary
        severity = get_damage_severity(class_names)
        summary  = get_damage_summary(class_names)
        color    = (0,255,0) if severity=="none" else (0,200,255) if severity=="low" else (0,0,255)
        cv2.putText(annotated, f"{summary} | {severity.upper()}",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with frame_lock: latest_frame = buf.tobytes()
        if frame_num % save_interval == 0 and len(boxes) > 0:
            temp = f"uploads/webcam_{frame_num}.jpg"
            cv2.imwrite(temp, frame)
            detect_image(temp)

            # Best-effort camera-based accident checks using the existing
            # multi-threat traffic model (separate from pothole annotations).
            global _last_accident_emergency_ts
            if (datetime.utcnow().timestamp() - _last_accident_emergency_ts) > _ACCIDENT_EMERGENCY_COOLDOWN_SEC:
                try:
                    from gps_service import get_latest_location
                    from emergency_system import trigger_emergency
                    from multidetect import (
                        traffic_model,
                        parse_boxes,
                        detect_accident,
                        VEHICLE_CLASSES,
                        PERSON_CLASS,
                    )

                    # Run traffic model on the saved frame.
                    traffic_res = traffic_model(temp, verbose=False)[0]
                    vehicles = parse_boxes(traffic_res, VEHICLE_CLASSES)
                    persons  = parse_boxes(traffic_res, PERSON_CLASS)
                    acc = detect_accident(vehicles, persons, cv2.imread(temp).shape)

                    if acc.get("alert_needed"):
                        latest = get_latest_location(max_age_seconds=180)
                        if latest:
                            _last_accident_emergency_ts = datetime.utcnow().timestamp()
                            trigger_emergency(
                                emergency_type="accident",
                                severity=acc.get("accident_likelihood", "high"),
                                location={"lat": latest["lat"], "lng": latest["lng"]},
                                triggered_by_user_id=None,
                                metadata={"source": "webcam_based_accident_check"},
                                family_numbers=[],
                                voice_call=True,
                            )
                except Exception:
                    pass   # fail-safe: never break webcam loop
    cap.release()
    webcam_active = False

def gen_frames():
    while webcam_active:
        with frame_lock: frame = latest_frame
        if frame:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

@app.route("/api/webcam/start", methods=["POST"])
def start_webcam():
    global webcam_active
    webcam_active = True
    threading.Thread(target=webcam_worker, daemon=True).start()
    return jsonify({"success": True})

@app.route("/api/webcam/stop", methods=["POST"])
def stop_webcam():
    global webcam_active
    webcam_active = False
    return jsonify({"success": True})

@app.route("/api/webcam/feed")
def webcam_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ── SSE: Real-time alert stream ────────────────────────────────
def _sse_stream(user_id=None):
    q = subscribe_alerts(user_id=user_id)
    try:
        # send a heartbeat comment every 15 s to keep connection alive
        import queue as _q
        while True:
            try:
                data = q.get(timeout=15)
                yield f"data: {data}\n\n"
            except _q.Empty:
                yield ": heartbeat\n\n"
    except GeneratorExit:
        pass
    finally:
        unsubscribe_alerts(q)

@app.route("/api/alerts/stream")
def alert_stream():
    user_id = request.args.get("user_id")
    return Response(_sse_stream(user_id=user_id), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── GPS + SOS + Voice APIs ────────────────────────────────────

@app.route("/api/gps/update", methods=["POST"])
def api_gps_update():
    """
    Client sends live GPS + optional contacts:
      - user_id, lat, lng (required)
      - phone_number (optional)
      - family_numbers (optional; list or comma-separated string)
    """
    from gps_service import upsert_from_payload

    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True)

    fam = data.get("family_numbers")
    if isinstance(fam, str):
        data["family_numbers"] = [x.strip() for x in fam.split(",") if x.strip()]

    try:
        return jsonify(upsert_from_payload(data))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/gps/latest", methods=["GET"])
@app.route("/api/gps/latest/<user_id>", methods=["GET"])
def api_gps_latest(user_id=None):
    """
    Fetch latest live location for a user.
    Accepts `user_id` as path param or query param.
    Optional: `max_age_seconds` (default 180).
    """
    from database import get_latest_user_location_for_user

    uid = (user_id or request.args.get("user_id") or "").strip()
    if not uid:
        return jsonify({"success": False, "error": "user_id is required"}), 400

    max_age_raw = (request.args.get("max_age_seconds") or "180").strip()
    try:
        max_age = int(max_age_raw)
    except ValueError:
        return jsonify({"success": False, "error": "max_age_seconds must be an integer"}), 400

    loc = get_latest_user_location_for_user(uid, max_age_seconds=max_age)
    if not loc:
        return jsonify({"success": False, "error": "No recent location for this user"}), 404

    return jsonify(
        {
            "success": True,
            "user_id": loc.get("user_id"),
            "lat": loc.get("lat"),
            "lng": loc.get("lng"),
            "updated_at": loc.get("updated_at"),
        }
    )


@app.route("/api/sos/trigger", methods=["POST"])
def api_sos_trigger():
    """
    SOS emergency trigger endpoint.
    Required: lat, lng
    Optional: user_id, emergency_type, severity, family_numbers, notify_mode
    notify_mode: call | sms | both | none
    """
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True)

    fam = data.get("family_numbers")
    if isinstance(fam, str):
        data["family_numbers"] = [x.strip() for x in fam.split(",") if x.strip()]

    lat = data.get("lat")
    lng = data.get("lng")
    if lat is None or lng is None:
        return jsonify({"error": "lat and lng are required"}), 400

    try:
        from emergency_system import trigger_emergency
        notify_mode = (data.get("notify_mode") or "both").strip().lower()
        voice_call = notify_mode in ("call", "both")
        sms_alert = notify_mode in ("sms", "both")
        return jsonify(
            trigger_emergency(
                emergency_type=(data.get("emergency_type") or "accident").strip().lower(),
                severity=(data.get("severity") or "high").strip().lower(),
                location={"lat": float(lat), "lng": float(lng)},
                triggered_by_user_id=(data.get("user_id") or None),
                metadata={"source": "sos_endpoint"},
                family_numbers=data.get("family_numbers") or [],
                voice_call=voice_call,
                sms_alert=sms_alert,
            )
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/voice/twiml", methods=["GET"])
def api_voice_twiml():
    """TwiML endpoint for Twilio `<Say>` messages."""
    from voice_service import get_twiml_xml

    msg = request.args.get("msg") or ""
    xml = get_twiml_xml(msg)
    return Response(xml, mimetype="text/xml")


@app.route("/api/voice/status", methods=["POST", "GET"])
def api_voice_status():
    """Twilio status callback for retry/fail-safe logic."""
    from voice_service import handle_voice_status_callback

    vcid = request.args.get("vcid") or request.form.get("vcid") or request.values.get("vcid")
    call_status = (
        request.form.get("CallStatus")
        or request.args.get("CallStatus")
        or request.values.get("CallStatus")
    )
    provider_call_sid = request.form.get("CallSid") or request.args.get("CallSid") or request.values.get("CallSid")

    raw = request.form.to_dict(flat=True) if request.form else {}
    handle_voice_status_callback(
        vcid=vcid,
        call_status=call_status,
        provider_call_sid=provider_call_sid,
        raw=raw,
    )
    return jsonify({"success": True})


@app.route("/api/route/suggest", methods=["GET"])
def api_route_suggest():
    """Best-effort route suggestion based on emergency context."""
    from emergency_system import _route_suggestion
    from gps_service import get_latest_location
    from database import get_recent_emergencies

    lat = request.args.get("lat")
    lng = request.args.get("lng")
    location = {}
    if lat is not None and lng is not None:
        location = {"lat": float(lat), "lng": float(lng)}
    else:
        location = get_latest_location(max_age_seconds=180)

    if not location:
        return jsonify({"success": False, "error": "No location available"}), 400

    def haversine_km(lat1, lon1, lat2, lon2) -> float:
        # Small helper for approximate distance (no external routing APIs).
        import math
        r = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    nearest = None
    nearest_km = None
    for e in get_recent_emergencies(max_age_seconds=900, limit=40):
        loc = e.get("location") or {}
        elat = loc.get("lat")
        elng = loc.get("lng")
        if elat is None or elng is None:
            continue
        km = haversine_km(float(location["lat"]), float(location["lng"]), float(elat), float(elng))
        if km > 1.0:
            continue
        if nearest_km is None or km < nearest_km:
            nearest_km = km
            nearest = e

    if nearest:
        return jsonify(
            {
                "success": True,
                "route_suggestion": _route_suggestion(nearest.get("type", "accident"), nearest.get("severity", "high")),
            }
        )

    return jsonify({"success": True, "route_suggestion": _route_suggestion("accident", "high")})


# ── Static files ──────────────────────────────────────────────────
@app.route("/results/<path:filename>")
def result_file(filename):
    return send_from_directory(RESULTS_DIR, filename)

@app.route("/uploads/<filename>")
def upload_file(filename):
    return send_from_directory("uploads", filename)


# ── Run ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🛡️  RoadGuard AI — Road Safety Platform")
    print("="*55)
    print(f"  Night Vision : {'✅ ON' if ENHANCE_AVAILABLE else '❌ OFF (vision_enhance.py missing)'}")
    print(f"  Intelligence : {'✅ ON' if INTELLIGENCE_AVAILABLE else '❌ OFF (road_intelligence.py missing)'}")
    print("="*55)
    print("  Dashboard    →  http://127.0.0.1:5000")
    print("  Upload       →  http://127.0.0.1:5000/upload")
    print("  Live Camera  →  http://127.0.0.1:5000/live")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
