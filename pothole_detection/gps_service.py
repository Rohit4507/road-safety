from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from database import (
    upsert_user_profile,
    upsert_user_location,
    get_latest_user_location,
    get_nearby_users,
)


@dataclass(frozen=True)
class GPSUpdate:
    user_id: str
    lat: float
    lng: float
    phone_number: str | None = None
    family_numbers: list[str] | None = None
    metadata: dict[str, Any] | None = None


def upsert_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Stores/refreshes:
      - user profile (phone + family contacts)
      - user live location (for 1km radius queries)
    """
    user_id = (payload.get("user_id") or "").strip()
    if not user_id:
        raise ValueError("user_id is required")

    lat = payload.get("lat")
    lng = payload.get("lng")
    if lat is None or lng is None:
        raise ValueError("lat and lng are required")

    phone_number = payload.get("phone_number")
    family_numbers = payload.get("family_numbers")
    metadata = payload.get("metadata") or {}

    update = GPSUpdate(
        user_id=user_id,
        lat=float(lat),
        lng=float(lng),
        phone_number=str(phone_number).strip() if phone_number else None,
        family_numbers=[str(n).strip() for n in family_numbers or [] if str(n).strip()],
        metadata=metadata,
    )

    if update.phone_number or update.family_numbers:
        upsert_user_profile(
            user_id=update.user_id,
            phone_number=update.phone_number,
            family_numbers=update.family_numbers,
        )

    upsert_user_location(
        user_id=update.user_id,
        lat=update.lat,
        lng=update.lng,
        metadata=update.metadata,
    )

    return {"success": True, "user_id": update.user_id}


def get_latest_location(max_age_seconds: int = 180) -> dict[str, Any]:
    """Best-effort fallback location for camera-based emergencies."""
    latest = get_latest_user_location(max_age_seconds=max_age_seconds)
    return latest or {}


def get_targets_nearby(lat: float, lng: float, radius_m: int = 1000) -> list[dict[str, Any]]:
    return get_nearby_users(lat=lat, lng=lng, radius_m=radius_m)

