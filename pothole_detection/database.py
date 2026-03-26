# ============================================================
# database.py — MongoDB connection & all DB operations
# ============================================================

from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Any
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

# ── Emergency / GPS collections ────────────────────────────────
user_profiles_col  = db["user_profiles"]
user_locations_col = db["user_locations"]
emergencies_col    = db["emergencies"]
voice_calls_col    = db["voice_calls"]

# Geospatial index for 2dsphere queries (radius in meters).
try:
    user_locations_col.create_index([("loc", "2dsphere")])
    user_locations_col.create_index([("updated_at", -1)])
    user_profiles_col.create_index([("user_id", 1)], unique=True)
except Exception:
    # Index creation can fail on misconfigured Mongo or offline environments;
    # queries will still work if indexes already exist.
    pass


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
# 🚨 Emergency & GPS persistence
# ---------------------------------------------------------------

def upsert_user_profile(
    user_id: str,
    phone_number: str | None = None,
    family_numbers: list[str] | None = None,
):
    """Store/refresh user contact details for voice/SMS dispatch."""
    if not user_id:
        return
    update: dict = {"updated_at": datetime.utcnow()}
    if phone_number:
        update["phone_number"] = phone_number
    if family_numbers is not None:
        update["family_numbers"] = family_numbers

    user_profiles_col.update_one({"user_id": user_id}, {"$set": update}, upsert=True)


def upsert_user_location(
    user_id: str,
    lat: float,
    lng: float,
    metadata: dict | None = None,
):
    if not user_id:
        return
    user_locations_col.update_one(
        {"user_id": user_id},
        {
            "$set": {
                "user_id": user_id,
                "loc": {"type": "Point", "coordinates": [float(lng), float(lat)]},
                "lat": float(lat),
                "lng": float(lng),
                "updated_at": datetime.utcnow(),
                "metadata": metadata or {},
            }
        },
        upsert=True,
    )


def get_latest_user_location(max_age_seconds: int = 180) -> dict | None:
    """Return the most recently updated location for any user."""
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
    doc = user_locations_col.find_one(
        {"updated_at": {"$gte": cutoff}},
        sort=[("updated_at", -1)],
        projection={"_id": 0, "user_id": 1, "lat": 1, "lng": 1, "updated_at": 1},
    )
    return doc


def get_latest_user_location_for_user(user_id: str, max_age_seconds: int = 180) -> dict | None:
    """Return latest location for a specific user within freshness window."""
    if not user_id:
        return None
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
    doc = user_locations_col.find_one(
        {"user_id": user_id, "updated_at": {"$gte": cutoff}},
        sort=[("updated_at", -1)],
        projection={"_id": 0, "user_id": 1, "lat": 1, "lng": 1, "updated_at": 1},
    )
    return doc


def get_nearby_users(
    *,
    lat: float,
    lng: float,
    radius_m: int = 1000,
    within_seconds: int = 900,
    limit: int = 50,
) -> list[dict]:
    """
    Returns user targets (phone numbers) within `radius_m` of the point.
    Uses 2dsphere index on `user_locations.loc` when available.
    """
    cutoff = datetime.utcnow() - timedelta(seconds=within_seconds)

    # Join with profiles in-app for minimal DB work.
    loc_query = {
        "updated_at": {"$gte": cutoff},
        "loc": {
            "$nearSphere": {
                "$geometry": {"type": "Point", "coordinates": [float(lng), float(lat)]},
                "$maxDistance": float(radius_m),
            }
        },
    }

    cursor = (
        user_locations_col.find(
            loc_query,
            projection={"_id": 0, "user_id": 1, "lat": 1, "lng": 1, "updated_at": 1},
        )
        .limit(limit)
    )

    users = []
    user_ids = []
    locs = list(cursor)
    user_ids = [d.get("user_id") for d in locs if d.get("user_id")]

    if not user_ids:
        return []

    profiles = {
        p["user_id"]: p
        for p in user_profiles_col.find(
            {"user_id": {"$in": user_ids}},
            projection={"_id": 0, "user_id": 1, "phone_number": 1, "family_numbers": 1},
        )
    }

    for loc in locs:
        uid = loc.get("user_id")
        profile = profiles.get(uid, {})
        users.append(
            {
                "user_id": uid,
                "lat": loc.get("lat"),
                "lng": loc.get("lng"),
                "updated_at": loc.get("updated_at"),
                "phone_number": profile.get("phone_number"),
                "family_numbers": profile.get("family_numbers") or [],
            }
        )

    return users


def create_emergency_event(
    *,
    emergency_type: str,
    severity: str,
    location: dict[str, Any],
    triggered_by_user_id: str | None,
    metadata: dict[str, Any] | None = None,
    route_suggestion: dict[str, Any] | None = None,
) -> str:
    doc = {
        "timestamp": datetime.utcnow(),
        "type": emergency_type,
        "severity": severity,
        "location": location or {},
        "triggered_by_user_id": triggered_by_user_id,
        "metadata": metadata or {},
        "route_suggestion": route_suggestion or {},
        "status": "active",
    }
    res = emergencies_col.insert_one(doc)
    return str(res.inserted_id)


def get_recent_emergencies(max_age_seconds: int = 1800, limit: int = 50) -> list[dict]:
    """Returns recent active emergency events for route suggestion."""
    cutoff = datetime.utcnow() - timedelta(seconds=max_age_seconds)
    cursor = (
        emergencies_col.find(
            {"timestamp": {"$gte": cutoff}, "status": "active"},
            projection={"_id": 1, "type": 1, "severity": 1, "location": 1, "timestamp": 1},
        )
        .sort("timestamp", -1)
        .limit(limit)
    )
    out = []
    for d in cursor:
        d["_id"] = str(d.get("_id"))
        out.append(d)
    return out


def create_voice_call_attempt(
    *,
    emergency_id: str | None,
    to_number: str,
    attempt_number: int,
    status: str,
    error: str | None,
    provider_call_sid: str | None,
    metadata: dict | None = None,
) -> str:
    doc = {
        "created_at": datetime.utcnow(),
        "emergency_id": emergency_id,
        "to_number": to_number,
        "attempt_number": int(attempt_number),
        "status": status,
        "error": error,
        "provider_call_sid": provider_call_sid,
        "message": (metadata or {}).get("message"),
        "metadata": metadata or {},
    }
    res = voice_calls_col.insert_one(doc)
    return str(res.inserted_id)


def get_voice_call_attempt(vcid: str) -> dict | None:
    if not vcid:
        return None
    try:
        from bson import ObjectId

        doc = voice_calls_col.find_one(
            {"_id": ObjectId(vcid)},
            projection={"_id": 0},
        )
        return doc
    except Exception:
        return None


def set_voice_call_status(
    vcid: str,
    *,
    status: str,
    provider_call_sid: str | None,
    error: str | None,
    raw: dict | None = None,
) -> None:
    if not vcid:
        return
    try:
        from bson import ObjectId

        update: dict[str, Any] = {
            "$set": {
                "status": status,
                "provider_call_sid": provider_call_sid,
                "error": error,
            }
        }
        if raw is not None:
            update["$set"]["raw"] = raw
        user = voice_calls_col.update_one({"_id": ObjectId(vcid)}, update)
        return
    except Exception:
        return


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
