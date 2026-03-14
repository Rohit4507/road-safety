# ============================================================
# database.py — MongoDB connection & all DB operations
# ============================================================

from pymongo import MongoClient
from datetime import datetime
import base64
import cv2
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# 🔧 CHANGE THIS: Paste your MongoDB Atlas connection string here
# ---------------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME   = "pothole_db"

# Connect
client         = MongoClient(MONGO_URI)
db             = client[DB_NAME]
detections_col = db["detections"]
stats_col      = db["stats"]


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def get_severity(count: int) -> str:
    if count == 0: return "none"
    if count <= 2: return "low"
    if count <= 5: return "medium"
    return "high"


def image_to_base64(image_path: str) -> str | None:
    """Convert image file to base64 string for DB storage."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        _, buffer = cv2.imencode(".jpg", img)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"⚠️  Could not encode image: {e}")
        return None


# ---------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------

def save_detection(image_path: str,
                   potholes_found: bool,
                   boxes: list,
                   confidence_scores: list,
                   source: str            = "image",
                   location: dict | None  = None,
                   damage_counts: dict    = None,   # NEW — {"pothole":2,"crack":1}
                   damage_summary: str    = None,   # NEW — "2 potholes, 1 crack"
                   class_names: list      = None,   # NEW — ["pothole","crack",...]
                   ) -> str:
    """
    Save one detection result to MongoDB.
    Returns the inserted document _id as string.
    """
    # If damage_counts not passed (old callers), build from boxes count
    _damage_counts  = damage_counts  or {"pothole": len(boxes)}
    _damage_summary = damage_summary or (f"{len(boxes)} pothole(s)" if boxes else "No damage")
    _class_names    = class_names    or ["pothole"] * len(boxes)

    # Severity — prefer damage-aware if class_names given, else fallback
    if class_names:
        severity = _get_damage_severity(class_names)
    else:
        severity = get_severity(len(boxes))

    record = {
        "timestamp":       datetime.utcnow(),
        "source":          source,
        "image_path":      image_path,
        "image_base64":    image_to_base64(image_path),
        "potholes_found":  potholes_found,
        "total_count":     len(boxes),
        "severity":        severity,
        "location":        location or {},

        # NEW — damage breakdown
        "damage_counts":   _damage_counts,
        "damage_summary":  _damage_summary,

        "detections": [
            {
                "box":        box,
                "confidence": float(conf),
                "class_name": cls,
            }
            for box, conf, cls in zip(boxes, confidence_scores, _class_names)
        ],
    }

    result = detections_col.insert_one(record)
    _id    = str(result.inserted_id)
    print(f"✅ Saved to MongoDB  →  ID: {_id}  |  {_damage_summary}  |  Severity: {severity}")
    return _id


def _get_damage_severity(class_names: list) -> str:
    """Severity based on damage types — mirrors detect.py logic."""
    counts   = {}
    for n in class_names:
        counts[n] = counts.get(n, 0) + 1
    potholes = counts.get("pothole", 0)
    rutting  = counts.get("rutting", 0)
    cracks   = counts.get("crack",   0)
    total    = len(class_names)

    if potholes >= 3 or rutting >= 2: return "high"
    if potholes >= 1 or cracks  >= 4: return "medium"
    if total > 0:                      return "low"
    return "none"


def get_all_detections(limit: int = 100) -> list:
    """Return latest detections (without heavy base64 image data)."""
    cursor = detections_col.find({}, {"image_base64": 0}).sort("timestamp", -1).limit(limit)
    docs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return docs


def get_by_severity(severity: str) -> list:
    cursor = detections_col.find({"severity": severity}, {"image_base64": 0}).sort("timestamp", -1)
    docs = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        docs.append(doc)
    return docs


def get_summary_stats() -> dict:
    """Return aggregate counts for the dashboard."""
    total    = detections_col.count_documents({})
    none_c   = detections_col.count_documents({"severity": "none"})
    low_c    = detections_col.count_documents({"severity": "low"})
    medium_c = detections_col.count_documents({"severity": "medium"})
    high_c   = detections_col.count_documents({"severity": "high"})

    # NEW — damage type totals across all records
    pipeline = [
        {"$project": {"damage_counts": 1}},
        {"$group": {
            "_id":     None,
            "potholes": {"$sum": "$damage_counts.pothole"},
            "cracks":   {"$sum": "$damage_counts.crack"},
            "patches":  {"$sum": "$damage_counts.patch"},
            "rutting":  {"$sum": "$damage_counts.rutting"},
        }}
    ]
    damage_totals = {}
    agg = list(detections_col.aggregate(pipeline))
    if agg:
        damage_totals = {
            "total_potholes": agg[0].get("potholes", 0),
            "total_cracks":   agg[0].get("cracks",   0),
            "total_patches":  agg[0].get("patches",  0),
            "total_rutting":  agg[0].get("rutting",  0),
        }

    return {
        "total":        total,
        "clean_roads":  none_c,
        "low":          low_c,
        "medium":       medium_c,
        "high":         high_c,
        **damage_totals,         # NEW — merged in
    }


def get_damage_breakdown() -> dict:
    """
    NEW — Returns per-type totals for dashboard charts.
    Use this to power the damage distribution chart.
    """
    pipeline = [
        {"$group": {
            "_id":     None,
            "potholes": {"$sum": "$damage_counts.pothole"},
            "cracks":   {"$sum": "$damage_counts.crack"},
            "patches":  {"$sum": "$damage_counts.patch"},
            "rutting":  {"$sum": "$damage_counts.rutting"},
        }}
    ]
    agg = list(detections_col.aggregate(pipeline))
    if not agg:
        return {"potholes": 0, "cracks": 0, "patches": 0, "rutting": 0}
    return {
        "potholes": agg[0].get("potholes", 0),
        "cracks":   agg[0].get("cracks",   0),
        "patches":  agg[0].get("patches",  0),
        "rutting":  agg[0].get("rutting",  0),
    }


def delete_all() -> int:
    """Wipe the collection — use carefully!"""
    result = detections_col.delete_many({})
    print(f"🗑️  Deleted {result.deleted_count} records")
    return result.deleted_count


# ---------------------------------------------------------------
# Quick connection test
# ---------------------------------------------------------------
if __name__ == "__main__":
    try:
        client.admin.command("ping")
        print("✅ MongoDB connection successful!")
        print(f"   Database    : {DB_NAME}")
        print(f"   Collections : {db.list_collection_names()}")
        print(f"\n📊 Current stats:")
        stats = get_summary_stats()
        for k, v in stats.items():
            print(f"   {k:20s}: {v}")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   → Check your MONGO_URI in .env file")