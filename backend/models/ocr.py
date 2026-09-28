"""
PLATE OCR
==========
EasyOCR on CPU — GPU benefit minimal for text recognition;
temporal fusion (OCRStabilizer) compensates for per-frame noise.
"""
import easyocr
import numpy as np
import cv2
import re


class PlateOCR:
    _PLATE_RE = re.compile(r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{3,4}")

    def __init__(self, lang="en"):
        self.reader = easyocr.Reader([lang], gpu=False, verbose=False)
        print("[PlateOCR] EasyOCR ready (CPU)")

    def predict(self, plate_image: np.ndarray) -> dict:
        if plate_image is None or plate_image.size == 0:
            return {"text": "", "confidence": 0.0}

        gray  = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        gray  = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        gray  = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
        )
        h, w = gray.shape[:2]
        if max(w, h) < 200:
            scale = 200 / max(w, h)
            gray  = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        h2, _ = gray.shape[:2]
        if h2 < 60:
            gray = cv2.resize(gray, None, fx=1.0, fy=60 / h2, interpolation=cv2.INTER_CUBIC)

        try:
            results = self.reader.readtext(
                gray, detail=1, paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            )
        except Exception:
            return {"text": "", "confidence": 0.0}

        valid = []
        for bbox, text, conf in results:
            clean = text.replace(" ", "").upper()
            if len(clean) >= 2 and conf >= 0.25:
                top_y = min(pt[1] for pt in bbox)
                valid.append((top_y, clean, float(conf)))

        if not valid:
            return {"text": "", "confidence": 0.0}

        valid.sort(key=lambda x: x[0])
        merged      = "".join(f for _, f, _ in valid)
        merged_conf = float(np.mean([c for _, _, c in valid]))

        m = self._PLATE_RE.search(merged)
        if m:
            return {"text": m.group(0), "confidence": round(merged_conf, 4)}

        best_text, best_conf = "", 0.0
        for _, clean, conf in valid:
            if len(clean) >= 5 and conf > best_conf:
                best_text, best_conf = clean, conf
        if not best_text:
            for _, clean, conf in valid:
                if conf > best_conf:
                    best_text, best_conf = clean, conf
        if not best_text:
            return {"text": "", "confidence": 0.0}
        if not self._PLATE_RE.search(best_text):
            best_conf *= 0.75
        return {"text": best_text, "confidence": round(best_conf, 4)}
