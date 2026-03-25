from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any
from urllib.parse import quote

from database import (
    create_voice_call_attempt,
    set_voice_call_status,
    get_voice_call_attempt,
)


VOICE_PROVIDER = os.getenv("VOICE_PROVIDER", "twilio").strip().lower()
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_SID", "").strip()
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_TOKEN", "").strip()
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://127.0.0.1:5000").strip()

DEFAULT_LANGUAGE = os.getenv("VOICE_LANGUAGE", "en-IN").strip()

# Retry policy for call failures
MAX_CALL_RETRIES = int(os.getenv("MAX_CALL_RETRIES", "3"))
RETRY_STATUSES = set(
    s.strip().lower()
    for s in os.getenv(
        "RETRY_CALL_STATUSES",
        "failed,busy,no-answer,canceled,timeout",
    ).split(",")
    if s.strip()
)


@dataclass(frozen=True)
class CallRequest:
    emergency_id: str | None
    to_number: str
    message: str
    attempt_number: int = 1
    metadata: dict[str, Any] | None = None


def _configured() -> bool:
    if VOICE_PROVIDER != "twilio":
        return False
    return bool(TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER)


def _build_twiml_message(message: str) -> str:
    # TwiML is XML; escape message content.
    msg = escape(message)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        f"<Say language=\"{DEFAULT_LANGUAGE}\" voice=\"alice\">{msg}</Say>"
        "</Response>"
    )


def get_twiml_xml(message: str) -> str:
    """Used by the `/api/voice/twiml` endpoint."""
    return _build_twiml_message(message)


def initiate_voice_call(emergency_id: str | None, to_number: str, message: str, attempt_number: int = 1, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Places a voice call asynchronously (caller should spawn thread if needed).
    Returns a record id for tracking.
    """
    if not _configured():
        # Fail-safe: still create a record so retries/visibility can work later.
        vcid = create_voice_call_attempt(
            emergency_id=emergency_id,
            to_number=to_number,
            attempt_number=attempt_number,
            status="skipped_unconfigured",
            error="voice provider not configured",
            provider_call_sid=None,
            metadata={"message": message, **(metadata or {})},
        )
        return {"success": False, "voice_call_id": vcid, "reason": "unconfigured"}

    # Insert attempt record first so callback has an id to update.
    vcid = create_voice_call_attempt(
        emergency_id=emergency_id,
        to_number=to_number,
        attempt_number=attempt_number,
        status="initiated",
        error=None,
        provider_call_sid=None,
        metadata={"message": message, **(metadata or {})},
    )

    def _place_call():
        try:
            from twilio.rest import Client  # optional; required only when enabled

            twilio = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            twiml_url = f"{PUBLIC_BASE_URL}/api/voice/twiml?vcid={quote(vcid)}&msg={quote(message)}"
            status_cb = f"{PUBLIC_BASE_URL}/api/voice/status?vcid={quote(vcid)}"

            call = twilio.calls.create(
                to=to_number,
                from_=TWILIO_FROM_NUMBER,
                url=twiml_url,
                status_callback=status_cb,
                status_callback_event=["initiated", "ringing", "answered", "completed", "busy", "no-answer", "failed"],
                method="GET",
            )

            set_voice_call_status(vcid, status="initiated", provider_call_sid=call.sid, error=None)
        except Exception as e:
            set_voice_call_status(vcid, status="failed", provider_call_sid=None, error=str(e))

    # Spawn background thread so HTTP request isn't blocked.
    threading.Thread(target=_place_call, daemon=True).start()
    return {"success": True, "voice_call_id": vcid}


def handle_voice_status_callback(vcid: str, call_status: str, provider_call_sid: str | None = None, raw: dict[str, Any] | None = None) -> None:
    """
    Called by `/api/voice/status`.
    Schedules a retry if this attempt didn't complete successfully.
    """
    if not vcid:
        return

    normalized = (call_status or "").strip().lower()
    set_voice_call_status(vcid, status=normalized or "unknown", provider_call_sid=provider_call_sid, error=None, raw=raw or {})

    # Decide whether to retry.
    attempt = get_voice_call_attempt(vcid)
    if not attempt:
        return

    next_attempt = int(attempt.get("attempt_number", 1)) + 1
    emergency_id = attempt.get("emergency_id")
    to_number = attempt.get("to_number")
    message = attempt.get("message")

    if not to_number or not message:
        return

    if next_attempt > MAX_CALL_RETRIES:
        return

    if normalized in ("completed", "canceled"):
        return

    if normalized not in RETRY_STATUSES:
        # Unknown status -> no retry by default.
        return

    # Spawn retry in background.
    def _retry():
        initiate_voice_call(
            emergency_id=emergency_id,
            to_number=to_number,
            message=message,
            attempt_number=next_attempt,
            metadata=attempt.get("metadata") or {},
        )

    threading.Thread(target=_retry, daemon=True).start()

