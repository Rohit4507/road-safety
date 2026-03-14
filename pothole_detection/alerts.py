# ================================================================
# alerts.py  —  Notification system for high-severity detections
# ================================================================
# Supports:
#   - Console alerts (always on)
#   - Sound alert (beep on high severity)
#   - SMS via Twilio (optional, needs API key)
#   - Webhook / Slack notification (optional)
# ================================================================

import os, json, requests
from datetime import datetime

# ── Config — fill in to enable optional alerts ────────────────
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_SID",   "")   # optional
TWILIO_AUTH_TOKEN  = os.getenv("TWILIO_TOKEN",  "")   # optional
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM",   "")   # e.g. +14155551234
ALERT_TO_NUMBER    = os.getenv("ALERT_TO",      "")   # your phone number

SLACK_WEBHOOK_URL  = os.getenv("SLACK_WEBHOOK", "")   # optional Slack webhook
ALERT_THRESHOLD    = "high"                            # alert only on high severity


# ── Severity check ────────────────────────────────────────────

SEVERITY_ORDER = ["none", "low", "medium", "high"]

def is_alert_worthy(severity: str) -> bool:
    return SEVERITY_ORDER.index(severity) >= SEVERITY_ORDER.index(ALERT_THRESHOLD)


# ── Console alert ─────────────────────────────────────────────

def console_alert(severity: str, count: int, location: dict = {}):
    now = datetime.now().strftime("%H:%M:%S")
    loc = f"  📍 Location: {location}" if location else ""
    print(f"""
╔══════════════════════════════════════╗
║  🚨 POTHOLE ALERT — {severity.upper():<14} ║
║  {count} pothole(s) detected at {now}  ║{loc}
╚══════════════════════════════════════╝""")


# ── Sound alert (beep) ────────────────────────────────────────

def sound_alert(severity: str):
    try:
        if severity == "high":
            # Three short beeps for high
            for _ in range(3):
                print("\a", end="", flush=True)
        elif severity == "medium":
            print("\a", end="", flush=True)
    except Exception:
        pass   # silent fail on systems without audio


# ── SMS alert via Twilio ──────────────────────────────────────

def sms_alert(severity: str, count: int, location: dict = {}):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER, ALERT_TO_NUMBER]):
        return   # Twilio not configured

    try:
        from twilio.rest import Client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        loc_str = f"\nLocation: {location}" if location else ""
        msg     = (
            f"🚨 ROAD ALERT — {severity.upper()}\n"
            f"{count} pothole(s) detected\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            f"{loc_str}"
        )

        message = client.messages.create(
            body = msg,
            from_= TWILIO_FROM_NUMBER,
            to   = ALERT_TO_NUMBER
        )
        print(f"📱 SMS sent! SID: {message.sid}")

    except ImportError:
        print("⚠️  Twilio not installed. Run: pip install twilio")
    except Exception as e:
        print(f"⚠️  SMS failed: {e}")


# ── Slack alert ───────────────────────────────────────────────

def slack_alert(severity: str, count: int, location: dict = {}):
    if not SLACK_WEBHOOK_URL:
        return

    color_map = {"none":"#3dd68c","low":"#60a5fa","medium":"#f59e0b","high":"#ef4444"}
    color     = color_map.get(severity, "#888")
    loc_str   = f" | 📍 {location}" if location else ""

    payload = {
        "attachments": [{
            "color": color,
            "title": f"🚨 Road Pothole Alert — {severity.upper()}",
            "text":  f"{count} pothole(s) detected{loc_str}",
            "footer":f"PotholeAI | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        }]
    }

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json=payload, timeout=5)
        if resp.status_code == 200:
            print("📢 Slack alert sent!")
    except Exception as e:
        print(f"⚠️  Slack alert failed: {e}")


# ── Main dispatch ─────────────────────────────────────────────

def send_alert(severity: str, count: int, location: dict = {}):
    """
    Call this after every detection.
    Only fires alerts for severities >= ALERT_THRESHOLD.
    """
    if not is_alert_worthy(severity):
        return

    console_alert(severity, count, location)
    sound_alert(severity)
    sms_alert(severity, count, location)
    slack_alert(severity, count, location)


# ── Test ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing alert system...\n")
    send_alert("high", 7, {"lat": 23.52, "lng": 87.31})
    print("\n✅ Alert test complete!")
    print("Configure Twilio or Slack env vars to enable remote alerts.")