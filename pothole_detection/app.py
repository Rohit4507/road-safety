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
from alerts         import send_alert

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