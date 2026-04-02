# ================================================================
# multi_detect.py  —  Multi-threat road safety detection
# Detects: Road Damage + Accidents + Animals + Traffic congestion
# ================================================================

import cv2
import os
import time
from ultralytics import YOLO
from database import save_detection, get_severity, detections_col
from alerts import send_alert

# ── Models ───────────────────────────────────────────────────────

DAMAGE_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "road_damage.pt")   # NEW — multi-class damage model
TRAFFIC_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "yolov8n.pt")       # pretrained — 80 COCO classes

os.makedirs("results", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

print("🔄 Loading models...")
damage_model  = YOLO(DAMAGE_MODEL_PATH)
traffic_model = YOLO(TRAFFIC_MODEL_PATH)
print("✅ Both models loaded!")

# ── Damage class names ────────────────────────────────────────────
# These come from your road_damage.pt training
# Check your data.yaml to confirm exact names — update if different
DAMAGE_CLASSES = ["pothole", "crack", "patch", "rutting"]

# ── COCO class IDs we care about ──────────────────────────────────
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
ANIMAL_CLASSES  = {
    15: "bird", 16: "cat", 17: "dog",  18: "horse",
    19: "sheep", 20: "cow", 21: "elephant", 22: "bear",
    23: "zebra", 24: "giraffe"
}
PERSON_CLASS    = {0: "person"}
CONF_THRESHOLD  = 0.40


# ── Parse YOLO boxes ──────────────────────────────────────────────

def parse_boxes(results, filter_classes=None):
    """Extract boxes for specific class IDs (or all if None)."""
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf   = box.conf[0].item()
        if conf < CONF_THRESHOLD:
            continue
        if filter_classes and cls_id not in filter_classes:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id":   cls_id,
            "class_name": filter_classes.get(cls_id, f"class_{cls_id}") if filter_classes else str(cls_id),
            "confidence": round(conf, 3),
            "box":        [round(x1), round(y1), round(x2), round(y2)],
        })
    return detections


def parse_damage_boxes(results) -> list:
    """
    Parse damage model results.
    Returns detections with class_name from damage model's own names.
    Works regardless of how many damage classes you trained.
    """
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf   = float(box.conf[0].item())
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Get class name from model itself — works with any dataset
        class_name = damage_model.names.get(cls_id, f"damage_{cls_id}")

        detections.append({
            "class_id":   cls_id,
            "class_name": class_name,
            "confidence": round(conf, 3),
            "box":        [round(x1), round(y1), round(x2), round(y2)],
        })
    return detections


# ── Damage analysis ───────────────────────────────────────────────

def analyze_damage(detections: list) -> dict:
    """
    Break down damage detections by type.
    Handles any class names your model was trained on.
    """
    # Count each damage type
    counts = {}
    for det in detections:
        name = det["class_name"].lower()
        counts[name] = counts.get(name, 0) + 1

    total     = len(detections)
    potholes  = counts.get("pothole", 0)
    cracks    = counts.get("crack", 0)
    patches   = counts.get("patch", 0)
    rutting   = counts.get("rutting", 0)

    # Severity logic — potholes and rutting are worse than cracks
    if potholes >= 3 or rutting >= 2:
        severity = "high"
    elif potholes >= 1 or cracks >= 4:
        severity = "medium"
    elif total > 0:
        severity = "low"
    else:
        severity = "none"

    # Human readable summary
    parts = []
    if potholes: parts.append(f"{potholes} pothole{'s' if potholes>1 else ''}")
    if cracks:   parts.append(f"{cracks} crack{'s' if cracks>1 else ''}")
    if patches:  parts.append(f"{patches} patch{'es' if patches>1 else ''}")
    if rutting:  parts.append(f"{rutting} rutting")
    summary = ", ".join(parts) if parts else "No damage detected"

    return {
        "total_count": total,
        "severity":    severity,
        "counts":      counts,
        "potholes":    potholes,
        "cracks":      cracks,
        "patches":     patches,
        "rutting":     rutting,
        "summary":     summary,
        "boxes":       [d["box"] for d in detections],
        "scores":      [d["confidence"] for d in detections],
    }


# ── Traffic density analysis ──────────────────────────────────────

def analyze_traffic(vehicle_count: int) -> dict:
    if vehicle_count == 0:
        level, risk = "clear", "none"
    elif vehicle_count <= 3:
        level, risk = "light", "low"
    elif vehicle_count <= 8:
        level, risk = "moderate", "medium"
    elif vehicle_count <= 15:
        level, risk = "heavy", "high"
    else:
        level, risk = "gridlock", "critical"

    return {
        "vehicle_count":   vehicle_count,
        "density_level":   level,
        "congestion_risk": risk,
        "recommendation": {
            "none":     "Road is clear",
            "low":      "Normal flow — no action needed",
            "medium":   "Monitor — may slow down",
            "high":     "Alert traffic control",
            "critical": "GRIDLOCK — deploy traffic police immediately",
        }.get(risk, "Unknown"),
    }


# ── Accident detection heuristic ─────────────────────────────────

def detect_accident(vehicles: list, persons: list, frame_shape: tuple) -> dict:
    accident_score = 0
    reasons = []

    def point_in_box(px: float, py: float, box: list) -> bool:
        return box[0] <= px <= box[2] and box[1] <= py <= box[3]

    for i, v1 in enumerate(vehicles):
        for v2 in vehicles[i+1:]:
            iou = compute_iou(v1["box"], v2["box"])
            if iou > 0.15:
                accident_score += 40
                reasons.append(f"Vehicles overlapping (IoU={iou:.2f})")

    for person in persons:
        px1, py1, px2, py2 = person["box"]
        p_cx = (px1 + px2) / 2
        p_cy = (py1 + py2) / 2
        for vehicle in vehicles:
            vx1, vy1, vx2, vy2 = vehicle["box"]
            v_cx = (vx1 + vx2) / 2
            v_cy = (vy1 + vy2) / 2
            dist = ((p_cx - v_cx)**2 + (p_cy - v_cy)**2) ** 0.5
            if point_in_box(p_cx, p_cy, vehicle["box"]):
                accident_score += 50
                reasons.append("Person appears within vehicle bounding box")
            if dist < 80:
                accident_score += 30
                reasons.append("Person very close to vehicle")

    # Additional boost when we see both multiple vehicles and persons.
    if len(vehicles) >= 2 and len(persons) >= 1:
        accident_score += 20
        reasons.append("Multi-vehicle + person presence pattern")

    if accident_score >= 70:   likelihood = "high"
    elif accident_score >= 40: likelihood = "medium"
    elif accident_score >= 15: likelihood = "low"
    else:                      likelihood = "none"

    return {
        "accident_likelihood": likelihood,
        "accident_score":      accident_score,
        "reasons":             reasons,
        "alert_needed":        likelihood in ("medium", "high"),
    }


def compute_iou(box1, box2) -> float:
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = a1 + a2 - inter
    return round(inter/union, 3) if union > 0 else 0.0


# ── Animal hazard analysis ────────────────────────────────────────

def analyze_animal_hazard(animals: list) -> dict:
    if not animals:
        return {"animals_detected": False, "risk": "none", "animal_types": [], "total_count": 0}

    HIGH_RISK    = {"cow", "horse", "elephant", "bear"}
    animal_names = [a["class_name"] for a in animals]
    has_large    = any(n in HIGH_RISK for n in animal_names)

    return {
        "animals_detected": True,
        "animal_types":     list(set(animal_names)),
        "total_count":      len(animals),
        "risk":             "high" if has_large else "medium",
        "recommendation":   "SLOW DOWN — Large animals on road!" if has_large
                            else "Caution — Animals on road",
    }


# ── Master: analyse full frame ────────────────────────────────────

def analyse_frame(image_path: str, location: dict = {}) -> dict:
    """
    Run ALL detections on one image/frame.
    Returns a complete threat analysis report.
    """
    start = time.perf_counter()

    # Run both models
    damage_res  = damage_model(image_path,  verbose=False)[0]   # road damage
    traffic_res = traffic_model(image_path, verbose=False)[0]   # vehicles/animals

    # Parse results
    damage_dets = parse_damage_boxes(damage_res)
    vehicles    = parse_boxes(traffic_res, VEHICLE_CLASSES)
    animals     = parse_boxes(traffic_res, ANIMAL_CLASSES)
    persons     = parse_boxes(traffic_res, PERSON_CLASS)

    # Analysis modules
    damage_info   = analyze_damage(damage_dets)
    traffic_info  = analyze_traffic(len(vehicles))
    accident_info = detect_accident(vehicles, persons, cv2.imread(image_path).shape)
    animal_info   = analyze_animal_hazard(animals)

    elapsed = round((time.perf_counter() - start) * 1000, 1)

    # Overall threat level
    threats = []
    if damage_info["severity"] in ("medium", "high"):
        threats.append("pothole")
    if traffic_info["congestion_risk"] in ("high", "critical"):
        threats.append("congestion")
    if accident_info["alert_needed"]:
        threats.append("accident")
    if animal_info["risk"] in ("medium", "high"):
        threats.append("animal")

    overall_risk = "critical" if len(threats) >= 3 else \
                   "high"     if len(threats) == 2 else \
                   "medium"   if len(threats) == 1 else "safe"

    report = {
        "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "image_path":     image_path,
        "location":       location,
        "inference_ms":   elapsed,
        "overall_risk":   overall_risk,
        "active_threats": threats,

        # NEW — replaces old "potholes" key with full damage breakdown
        "potholes": {
            "count":    damage_info["total_count"],
            "severity": damage_info["severity"],
            "summary":  damage_info["summary"],
            "boxes":    damage_info["boxes"],
            "scores":   damage_info["scores"],
            # NEW fields
            "by_type":  damage_info["counts"],
            "potholes": damage_info["potholes"],
            "cracks":   damage_info["cracks"],
            "patches":  damage_info["patches"],
            "rutting":  damage_info["rutting"],
        },
        "traffic":  traffic_info,
        "accident": accident_info,
        "animals":  animal_info,
        "raw": {
            "damage":   damage_dets,
            "vehicles": vehicles,
            "persons":  persons,
            "animals":  animals,
        }
    }

    # Save to MongoDB — use save_detection so damage_counts are stored properly
    from datetime import datetime
    save_detection(
        image_path        = image_path,
        potholes_found    = damage_info["total_count"] > 0,
        boxes             = damage_info["boxes"],
        confidence_scores = damage_info["scores"],
        source            = "multi_threat",
        location          = location,
        damage_counts     = damage_info["counts"],
        damage_summary    = damage_info["summary"],
    )
    # Also store full multi-threat report separately for the log tab
    detections_col.insert_one({**report, "timestamp": datetime.utcnow(), "type": "multi_threat"})

    # Fire legacy "road damage" alerts only.
    # Accident / congestion notifications are handled by the dedicated emergency pipeline.
    for threat in threats:
        if threat != "pothole":
            continue
        send_alert(
            severity="high",
            count=damage_info["total_count"],
            location=location,
        )

    # Console summary
    print(f"\n{'='*50}")
    print(f"  🔍 THREAT ANALYSIS — {elapsed}ms")
    print(f"{'='*50}")
    print(f"  Overall Risk  : {overall_risk.upper()}")
    print(f"  Road Damage   : {damage_info['summary']}")
    print(f"  Severity      : {damage_info['severity']}")
    print(f"  Vehicles      : {len(vehicles)} ({traffic_info['density_level']})")
    print(f"  Animals       : {animal_info['total_count']} detected")
    print(f"  Accident Risk : {accident_info['accident_likelihood']}")
    print(f"  Threats       : {threats or ['none']}")
    print(f"{'='*50}")

    return report


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, json
    path = sys.argv[1] if len(sys.argv) > 1 else "test_road.jpg"
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        sys.exit(1)
    result = analyse_frame(path)
    print("\nFull Report:")
    print(json.dumps(
        {k: v for k, v in result.items() if k != "raw"},
        indent=2, default=str
    ))
