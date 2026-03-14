# ================================================================
# road_intelligence.py  —  Road Quality Score | Repair Cost
#                           Depth Estimator | Speed Tracker
#                           Wrong-Way Detector | Weather Risk
# ================================================================

import cv2, numpy as np
from datetime import datetime

# ── 1. ROAD QUALITY SCORE ────────────────────────────────────────

def road_score(potholes=0, severity="none", vehicles=0,
               animal=False, accident_risk="none", history=0) -> dict:
    s = 100
    s -= min(potholes * 7, 40)
    s -= {"none":0,"low":5,"medium":18,"high":35}.get(severity, 0)
    s -= 10 if vehicles > 15 else 5 if vehicles > 8 else 0
    s -= 12 if animal else 0
    s -= {"none":0,"low":5,"medium":15,"high":25}.get(accident_risk, 0)
    s -= min(history * 3, 15)
    s = max(0, min(100, s))

    if s >= 85: g, c, col = "A", "Excellent",  "00e676"
    elif s >= 70: g, c, col = "B", "Good",     "7ecb20"
    elif s >= 55: g, c, col = "C", "Fair",     "ffb400"
    elif s >= 35: g, c, col = "D", "Poor",     "ff6b00"
    else:         g, c, col = "F", "Critical", "ff2d55"

    actions = {"A":"No action","B":"Routine check in 30d","C":"Plan repair < 30d",
               "D":"Repair within 7 days","F":"⚠️ CLOSE ROAD NOW"}

    return {"score":s,"grade":g,"condition":c,"color":col,"action":actions[g]}


# ── 2. REPAIR COST ESTIMATOR ─────────────────────────────────────

_COSTS = {
    "patching":    {"sqm":450,  "desc":"Cold-mix patching"},
    "hotmix":      {"sqm":1200, "desc":"Hot-mix asphalt"},
    "base_repair": {"sqm":3500, "desc":"Base layer + surface"},
    "full_recon":  {"sqm":8500, "desc":"Full reconstruction"},
}
_SEV_REPAIR = {"low":"patching","medium":"hotmix","high":"base_repair"}
_ROAD_MULT  = {"urban":1.0,"highway":1.8,"rural":0.7}

def repair_cost(potholes: list, severity: str, road_type="urban") -> dict:
    if not potholes or severity == "none":
        return {"total_inr":0,"repair_type":"None","deadline":"—"}

    rk  = _SEV_REPAIR.get(severity,"hotmix")
    ri  = _COSTS[rk]
    mul = _ROAD_MULT.get(road_type, 1.0)

    total = 0
    for ph in potholes:
        b = ph.get("box",[0,0,100,100])
        w_m = (b[2]-b[0]) / (640/5.0)
        h_m = (b[3]-b[1]) / (480/3.5)
        total += max(0.1, w_m*h_m) * ri["sqm"] * mul

    total *= 1.25   # overhead + materials

    deadlines = {"low":"7 days","medium":"48 hours","high":"24 hours"}
    return {
        "total_inr":   round(total),
        "total_usd":   round(total/83.5, 1),
        "repair_type": ri["desc"],
        "repair_key":  rk,
        "deadline":    deadlines.get(severity,"7 days"),
        "urgency":     severity,
        "count":       len(potholes),
    }


# ── 3. POTHOLE DEPTH ESTIMATOR ───────────────────────────────────

def pothole_depth(box: list, conf: float, img_shape: tuple) -> dict:
    area  = (box[2]-box[0]) * (box[3]-box[1])
    ratio = area / (img_shape[0] * img_shape[1])
    score = ratio * 100 + conf * 15

    if score < 3:    d, cm, r = "Shallow",  "2–5 cm",  "Cold patching"
    elif score < 8:  d, cm, r = "Moderate", "5–12 cm", "Hot-mix"
    elif score < 18: d, cm, r = "Deep",     "12–25 cm","Base repair"
    else:            d, cm, r = "Critical", "25+ cm",  "Full reconstruction"

    return {"depth":d,"cm_range":cm,"repair":r,"score":round(score,2),"dangerous":score>=18}


# ── 4. SPEED TRACKER ─────────────────────────────────────────────

class SpeedTracker:
    def __init__(self, fps=30, road_w=7.0, frame_w=640, limit=50):
        self.fps = fps
        self.ppm = (frame_w * 0.65) / road_w
        self.limit = limit
        self.prev = {}
        self.violations = []
        self.frame = 0

    def update(self, detections: list) -> list:
        self.frame += 1
        out = []
        for i, d in enumerate(detections):
            b  = d["box"]
            cx = (b[0]+b[2])/2; cy = (b[1]+b[3])/2
            vid = f"{d['class_name']}_{i}"
            spd = None
            if vid in self.prev:
                px, py, pf = self.prev[vid]
                fe = self.frame - pf
                if fe > 0:
                    dist = ((cx-px)**2+(cy-py)**2)**0.5 / self.ppm
                    spd  = round(dist/(fe/self.fps)*3.6, 1)
                    if spd > self.limit:
                        self.violations.append({
                            "vehicle":d["class_name"],"speed":spd,
                            "excess":round(spd-self.limit,1),
                            "ts":datetime.utcnow().isoformat()
                        })
            self.prev[vid] = (cx, cy, self.frame)
            out.append({**d, "speed_kmh":spd, "speeding": spd and spd>self.limit})
        return out


# ── 5. WRONG-WAY DETECTOR ────────────────────────────────────────

class WrongWayDetector:
    def __init__(self, expected="left_to_right"):
        self.expected = expected
        self.hist = {}
        self.alerts = []

    def update(self, detections: list) -> list:
        out = []
        for i, d in enumerate(detections):
            cx = (d["box"][0]+d["box"][2])/2
            vid = f"{d['class_name']}_{i}"
            self.hist.setdefault(vid, []).append(cx)
            wrong = False
            if len(self.hist[vid]) >= 4:
                mv = self.hist[vid][-1] - self.hist[vid][-4]
                wrong = (mv < -30) if self.expected=="left_to_right" else (mv > 30)
                if wrong:
                    self.alerts.append({"vehicle":d["class_name"],"ts":datetime.utcnow().isoformat()})
            out.append({**d, "wrong_way": wrong})
        return out


# ── 6. WEATHER RISK SCORER ───────────────────────────────────────

def weather_risk(brightness: float, saturation: float,
                 contrast: float, blur: float) -> dict:
    """
    Derive a weather risk level from image stats alone.
    No weather API needed — fully offline.
    """
    risk = 0
    tags = []

    if brightness < 60:  risk += 35; tags.append("night")
    elif brightness < 90: risk += 15; tags.append("dusk/dawn")

    if saturation < 30:  risk += 20; tags.append("fog/rain")
    if contrast < 25:    risk += 20; tags.append("low_visibility")
    if blur < 60:        risk += 10; tags.append("spray/blur")

    risk = min(100, risk)
    if risk >= 70: level, col = "High",   "ff2d55"
    elif risk >= 40: level, col = "Medium","ffb400"
    elif risk >= 15: level, col = "Low",  "7ecb20"
    else:            level, col = "Clear","00e676"

    recs = {
        "night":          "Headlights required — reduced visibility",
        "fog/rain":        "Reduce speed — fog/rain detected",
        "low_visibility":  "Caution — poor visibility conditions",
        "dusk/dawn":       "Watch for glare and sudden darkness",
        "spray/blur":      "Heavy rain spray — increase following distance",
    }
    recommendation = recs.get(tags[0], "Drive safely") if tags else "Conditions normal"

    return {
        "risk_score":     risk,
        "risk_level":     level,
        "color":          col,
        "weather_tags":   tags,
        "recommendation": recommendation,
    }