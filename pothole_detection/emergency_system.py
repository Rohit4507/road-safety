from __future__ import annotations

import os
import threading
from datetime import datetime
from typing import Any

from alerts import browser_alert
from database import create_emergency_event
from gps_service import get_targets_nearby
from voice_service import initiate_voice_call


POLICE_TO_NUMBER = os.getenv("POLICE_TO", "").strip()
HOSPITAL_TO_NUMBER = os.getenv("HOSPITAL_TO", "").strip()
EMERGENCY_RADIUS_M = int(os.getenv("EMERGENCY_RADIUS_M", "1000"))


def _severity_to_voice_text(severity: str, emergency_type: str) -> str:
    sev = (severity or "high").upper()
    et  = (emergency_type or "EMERGENCY").upper()
    # Keep message short to reduce call time and truncation.
    return f"{et} detected with {sev} severity near your location. Please move to safety, slow down, and call emergency services if needed."


def _route_suggestion(emergency_type: str, severity: str) -> dict[str, Any]:
    sev = (severity or "high").lower()
    if emergency_type == "accident":
        if sev in ("high", "critical"):
            return {
                "headline": "Avoid the affected road segment",
                "details": "Use an alternate route if possible. Expect delays and follow diversion guidance.",
                "eta_minutes_range": [10, 20],
            }
        if sev == "medium":
            return {
                "headline": "Drive with caution",
                "details": "Slow down and maintain safe distance until you pass the risk area.",
                "eta_minutes_range": [5, 12],
            }
    # Default fallback
    return {
        "headline": "Proceed with caution",
        "details": "Stay alert and follow local traffic guidance.",
        "eta_minutes_range": [3, 8],
    }


def trigger_emergency(
    *,
    emergency_type: str,
    severity: str,
    location: dict[str, Any],
    triggered_by_user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    family_numbers: list[str] | None = None,
    voice_call: bool = True,
) -> dict[str, Any]:
    """
    Emergency dispatch pipeline:
      - store emergency record
      - compute nearby targets within 1km
      - dispatch alerts (SSE) + voice calls with retry
      - include route suggestion in the response
    """
    lat = location.get("lat")
    lng = location.get("lng")
    if lat is None or lng is None:
        raise ValueError("location.lat and location.lng are required for emergency radius logic")

    emergency_id = create_emergency_event(
        emergency_type=emergency_type,
        severity=severity,
        location={"lat": float(lat), "lng": float(lng)},
        triggered_by_user_id=triggered_by_user_id,
        metadata=metadata or {},
        route_suggestion=_route_suggestion(emergency_type, severity),
    )

    # Determine nearby user targets (phone numbers + user_id).
    nearby_targets = get_targets_nearby(float(lat), float(lng), radius_m=EMERGENCY_RADIUS_M)
    nearby_user_ids = [t.get("user_id") for t in nearby_targets if t.get("user_id")]
    nearby_numbers = [t.get("phone_number") for t in nearby_targets if t.get("phone_number")]
    nearby_family_numbers = []
    for t in nearby_targets:
        for n in (t.get("family_numbers") or []):
            if n and n not in nearby_family_numbers:
                nearby_family_numbers.append(n)

    police_numbers = [POLICE_TO_NUMBER] if POLICE_TO_NUMBER else []
    hospital_numbers = [HOSPITAL_TO_NUMBER] if HOSPITAL_TO_NUMBER else []
    family_numbers = (family_numbers or []) + nearby_family_numbers

    # De-duplicate
    voice_numbers: list[str] = []
    for n in police_numbers + hospital_numbers + family_numbers + nearby_numbers:
        if n and n not in voice_numbers:
            voice_numbers.append(n)

    voice_text = _severity_to_voice_text(severity=severity, emergency_type=emergency_type)

    # 1) Push emergency alert to UI (SSE). We still send it to all, but include targets for filtering if available.
    browser_alert(
        severity="high",
        count=1,
        location={"lat": float(lat), "lng": float(lng)},
        message=f"🚨 EMERGENCY: {emergency_type} ({severity}) near you",
        category="emergency",
        target_user_ids=nearby_user_ids,
        extra={"emergency_id": emergency_id},
    )

    # 2) Place voice calls in background
    if voice_call and voice_numbers:
        def _call_all():
            for number in voice_numbers:
                initiate_voice_call(
                    emergency_id=emergency_id,
                    to_number=number,
                    message=voice_text,
                    attempt_number=1,
                    metadata={"emergency_type": emergency_type, "severity": severity},
                )

        threading.Thread(target=_call_all, daemon=True).start()

    return {
        "success": True,
        "emergency_id": emergency_id,
        "nearby_count": len(nearby_targets),
        "targets": {
            "police": police_numbers,
            "hospitals": hospital_numbers,
            "family": family_numbers,
            "nearby_numbers": nearby_numbers,
        },
        "route_suggestion": _route_suggestion(emergency_type, severity),
    }

