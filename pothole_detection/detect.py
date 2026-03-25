# ============================================================
# detect.py — YOLOv8 Road Damage Detection (image / video / webcam)
# ============================================================
from alerts import send_alert
import cv2
import os
import time
from ultralytics import YOLO
from database import save_detection

# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------
DAMAGE_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "road_damage.pt")  # multi-class damage model
CONFIDENCE_THRESH  = 0.40
RESULTS_DIR        = "results"
UPLOADS_DIR        = "uploads"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Load model once at startup 
print(f"🔄 Loading model: {DAMAGE_MODEL_PATH}")
model = YOLO(DAMAGE_MODEL_PATH)
print(f"✅ Model loaded! Classes: {list(model.names.values())}")


# ---------------------------------------------------------------
# Core: parse YOLO output
# ---------------------------------------------------------------

def parse_results(yolo_results):
    """
    Extract boxes, confidences, and class names from YOLO output.
    Works with any number of damage classes.
    """
    boxes       = []
    confidences = []
    class_names = []

    for box in yolo_results.boxes:
        conf = box.conf[0].item()
        if conf < CONFIDENCE_THRESH:
            continue
        cls_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append([round(x1), round(y1), round(x2), round(y2)])
        confidences.append(round(conf, 4))
        class_names.append(model.names.get(cls_id, f"damage_{cls_id}"))

    return boxes, confidences, class_names


def get_damage_counts(class_names: list) -> dict:
    """Count detections per damage type."""
    counts = {}
    for name in class_names:
        counts[name] = counts.get(name, 0) + 1
    return counts


def get_damage_severity(class_names: list) -> str:
    """
    Severity based on damage types and counts.
    Potholes and rutting are worse than cracks.
    """
    counts   = get_damage_counts(class_names)
    potholes = counts.get("pothole", 0)
    rutting  = counts.get("rutting", 0)
    cracks   = counts.get("crack",   0)
    total    = len(class_names)

    if potholes >= 3 or rutting >= 2:
        return "high"
    elif potholes >= 1 or cracks >= 4:
        return "medium"
    elif total > 0:
        return "low"
    return "none"


def get_damage_summary(class_names: list) -> str:
    """Human readable damage summary e.g. '2 potholes, 1 crack'"""
    counts = get_damage_counts(class_names)
    parts  = []
    for name, count in counts.items():
        label = name + ("s" if count > 1 and not name.endswith("s") else "")
        parts.append(f"{count} {label}")
    return ", ".join(parts) if parts else "No damage detected"


# ---------------------------------------------------------------
# 1. Detect from image file
# ---------------------------------------------------------------

def detect_image(image_path: str, location: dict | None = None) -> dict:
    """
    Run detection on a single image.
    Saves annotated result to /results and stores in MongoDB.
    Returns a summary dict.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"\n📸 Scanning road: {image_path}")
    start = time.time()

    results                    = model(image_path)[0]
    boxes, confs, class_names  = parse_results(results)

    # Save annotated image
    filename    = os.path.basename(image_path)
    output_path = os.path.join(RESULTS_DIR, f"detected_{filename}")
    results.save(filename=output_path)

    elapsed  = round(time.time() - start, 3)
    severity = get_damage_severity(class_names)
    counts   = get_damage_counts(class_names)
    summary_text = get_damage_summary(class_names)

    # Save to MongoDB
    doc_id = save_detection(
        image_path        = output_path,
        potholes_found    = len(boxes) > 0,
        boxes             = boxes,
        confidence_scores = confs,
        source            = "image",
        location          = location,
        # extra fields for new damage model
        damage_counts     = counts,
        damage_summary    = summary_text,
    )

    send_alert(severity, len(boxes), location)

    summary = {
        "doc_id":         doc_id,
        "image":          output_path,
        "pothole_count":  len(boxes),           # kept for backward compat
        "total_damage":   len(boxes),
        "severity":       severity,
        "inference_time": f"{elapsed}s",
        "boxes":          boxes,
        "confidences":    confs,
        "class_names":    class_names,
        "damage_counts":  counts,
        "damage_summary": summary_text,
    }

    print(f"   Damage found   : {summary_text}")
    print(f"   Severity       : {severity.upper()}")
    print(f"   Inference time : {elapsed}s")
    print(f"   Saved image    : {output_path}")
    return summary


# ---------------------------------------------------------------
# 2. Detect from video file
# ---------------------------------------------------------------

def detect_video(video_path: str, frame_interval: int = 10) -> dict:
    """
    Process every Nth frame of a video.
    Returns summary with total damage counts and frames processed.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"\n🎥 Processing video: {video_path}")
    cap          = cv2.VideoCapture(video_path)
    frame_num    = 0
    saved        = 0
    total_damage = 0
    all_counts   = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        if frame_num % frame_interval != 0:
            continue

        temp_path = os.path.join(UPLOADS_DIR, f"frame_{frame_num}.jpg")
        cv2.imwrite(temp_path, frame)

        result        = detect_image(temp_path)
        total_damage += result["total_damage"]
        saved        += 1

        # Accumulate damage type counts
        for dtype, count in result["damage_counts"].items():
            all_counts[dtype] = all_counts.get(dtype, 0) + count

        os.remove(temp_path)

    cap.release()

    print(f"\n✅ Video done!")
    print(f"   Frames processed : {saved}")
    print(f"   Total damage     : {total_damage}")
    print(f"   By type          : {all_counts}")

    return {
        "frames_processed": saved,
        "total_damage":     total_damage,
        "total_potholes":   total_damage,    # backward compat
        "damage_by_type":   all_counts,
    }


# ---------------------------------------------------------------
# 3. Detect from webcam (live)
# ---------------------------------------------------------------

def detect_webcam(camera_id: int = 0, save_interval: int = 30):
    """
    Live webcam detection.
    Press Q to quit.
    Saves every Nth frame to MongoDB.
    """
    print("\n📷 Starting live road scan... Press Q to quit")
    cap       = cv2.VideoCapture(camera_id)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read from webcam")
            break

        frame_num += 1

        results                   = model(frame, verbose=False)[0]
        annotated                 = results.plot()
        boxes, confs, class_names = parse_results(results)
        severity                  = get_damage_severity(class_names)
        summary_text              = get_damage_summary(class_names)

        # Colour by severity
        color = (0, 255,   0) if severity == "none"   else \
                (0, 200, 255) if severity == "low"    else \
                (0, 140, 255) if severity == "medium" else \
                (0,   0, 255)

        # Show damage summary on frame
        cv2.putText(annotated,
                    f"Damage: {summary_text} | {severity.upper()}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("RoadGuard AI — Press Q to quit", annotated)

        # Save to MongoDB every N frames if damage found
        if frame_num % save_interval == 0 and len(boxes) > 0:
            temp = os.path.join(UPLOADS_DIR, f"webcam_{frame_num}.jpg")
            cv2.imwrite(temp, frame)
            detect_image(temp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Live scan stopped")


# ---------------------------------------------------------------
# Quick demo — run this file directly
# ---------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\n📌 Usage:")
        print("  python detect.py image  path/to/road.jpg")
        print("  python detect.py video  path/to/road.mp4")
        print("  python detect.py webcam")
        sys.exit(0)

    mode = sys.argv[1].lower()

    if mode == "image" and len(sys.argv) == 3:
        detect_image(sys.argv[2])
    elif mode == "video" and len(sys.argv) == 3:
        detect_video(sys.argv[2])
    elif mode == "webcam":
        detect_webcam()
    else:
        print("❌ Invalid arguments. See usage above.")