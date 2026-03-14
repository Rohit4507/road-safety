# ================================================================
# vision_enhance.py  —  Night Vision + Full Image Enhancement
# ================================================================
# Fixes: Dark/Night  |  Fog/Haze  |  Rain  |  Blur  |  Overcast
# ================================================================

import cv2
import numpy as np
import os, time

OUT_DIR = "results/enhanced"
os.makedirs(OUT_DIR, exist_ok=True)


# ── Condition detector ────────────────────────────────────────────

def detect_conditions(image_path: str) -> dict:
    img  = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = float(gray.mean())
    contrast   = float(gray.std())
    blur_var   = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturation = float(hsv[:, :, 1].mean())

    conditions, fixes = [], []

    if brightness < 60:
        conditions.append("🌙 Night / Dark");  fixes.append("night")
    elif brightness < 100:
        conditions.append("🌅 Underexposed");  fixes.append("exposure")

    if contrast < 30:
        conditions.append("🌫️ Foggy / Hazy");  fixes.append("dehaze")

    if blur_var < 80:
        conditions.append("🔘 Blurry");         fixes.append("sharpen")

    if saturation < 40:
        conditions.append("🌧️ Overcast / Rain"); fixes.append("saturation")

    return {
        "brightness":  round(brightness, 1),
        "contrast":    round(contrast, 1),
        "blur_score":  round(blur_var, 1),
        "saturation":  round(saturation, 1),
        "conditions":  conditions or ["✅ Clear"],
        "fixes":       fixes,
        "needs_fix":   len(fixes) > 0,
    }


# ── Enhancement functions ─────────────────────────────────────────

def night_vision(img):
    """CLAHE on LAB L-channel + gamma lift."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    table = np.array([(i/255)**0.6 * 255 for i in range(256)], np.uint8)
    return cv2.LUT(out, table)


def dehaze(img):
    """Dark-channel prior fog removal."""
    def dark_ch(x, sz=15):
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
        return cv2.erode(np.min(x, axis=2), k)
    f = img.astype(np.float64) / 255.0
    dc = dark_ch(f)
    idx = np.argsort(dc.flatten())[::-1][:max(1, dc.size//1000)]
    atm = np.max(f.reshape(-1,3)[idx], axis=0)
    t = np.clip(1 - 0.95 * dark_ch(f / atm), 0.1, 1)[:,:,None]
    out = np.clip((f - atm) / t + atm, 0, 1)
    return (out * 255).astype(np.uint8)


def sharpen(img):
    """Unsharp mask."""
    blur = cv2.GaussianBlur(img, (0,0), 3)
    return cv2.addWeighted(img, 2.2, blur, -1.2, 0)


def boost_saturation(img, f=1.4):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,1] = np.clip(hsv[:,:,1] * f, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def exposure_correct(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.5 + 25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def white_balance(img):
    r = img.astype(np.float32)
    for i in range(3):
        avg = r[:,:,i].mean()
        grey = sum(r[:,:,c].mean() for c in range(3)) / 3
        r[:,:,i] = np.clip(r[:,:,i] * (grey / avg), 0, 255)
    return r.astype(np.uint8)


FIX_FN = {
    "night":      night_vision,
    "dehaze":     dehaze,
    "sharpen":    sharpen,
    "saturation": boost_saturation,
    "exposure":   exposure_correct,
}


# ── Master pipeline ───────────────────────────────────────────────

def enhance_image(image_path: str, force=False) -> dict:
    t0   = time.perf_counter()
    cond = detect_conditions(image_path)

    if not cond["needs_fix"] and not force:
        return {"enhanced": False, "output_path": image_path,
                "original_path": image_path, "conditions": cond["conditions"]}

    img  = cv2.imread(image_path)
    orig = img.copy()
    applied = []

    for fix in cond["fixes"]:
        img = FIX_FN[fix](img)
        applied.append(fix)

    img = white_balance(img)
    applied.append("white_balance")

    fname        = os.path.basename(image_path)
    out_path     = os.path.join(OUT_DIR, f"enhanced_{fname}")
    compare_path = os.path.join(OUT_DIR, f"compare_{fname}")
    cv2.imwrite(out_path, img)
    _save_compare(orig, img, compare_path)

    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {
        "enhanced":      True,
        "conditions":    cond["conditions"],
        "fixes_applied": applied,
        "output_path":   out_path,
        "original_path": image_path,
        "compare_path":  compare_path,
        "ms":            ms,
        "quality_before": {k: cond[k] for k in ("brightness","contrast","blur_score","saturation")},
    }


def _save_compare(orig, enh, out_path):
    h = max(orig.shape[0], enh.shape[0])
    w = orig.shape[1]
    o = cv2.resize(orig, (w, h))
    e = cv2.resize(enh,  (w, h))
    f = cv2.FONT_HERSHEY_SIMPLEX
    for img, lbl, col in [(o,"BEFORE",(80,80,80)),(e,"AFTER",(0,230,118))]:
        cv2.rectangle(img, (0,0),(160,38),(0,0,0),-1)
        cv2.putText(img, lbl, (10,26), f, 0.9, col, 2)
    sep = np.full((h, 3, 3), [0,180,255], dtype=np.uint8)
    cv2.imwrite(out_path, np.hstack([o, sep, e]))