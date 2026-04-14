import easyocr
import numpy as np
import cv2
import re


class PlateOCR:
    def __init__(self, lang="en"):
        self.reader = easyocr.Reader(
            [lang],
            gpu=False,
            verbose=False
        )

        # Indian plate regex (tolerant) — matches single-line or combined two-line
        self.plate_regex = re.compile(
            r"[A-Z]{2}[0-9]{1,2}[A-Z]{0,2}[0-9]{3,4}"
        )

    def predict(self, plate_image: np.ndarray):
        if plate_image is None or plate_image.size == 0:
            return {"text": "", "confidence": 0.0}

        # -------------------------------
        # Plate-friendly preprocessing
        # -------------------------------
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        # Adaptive binarization works well for HSRP
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        # Upscale small plates — use a taller minimum for two-line plates
        h, w = gray.shape[:2]
        min_dim = max(w, h)
        if min_dim < 200:
            scale = 200 / min_dim
            gray = cv2.resize(
                gray,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )
        # If the plate is very short (two-line plates are roughly square),
        # also upscale vertically so EasyOCR can see both lines clearly.
        h2, w2 = gray.shape[:2]
        if h2 < 60:
            scale_h = 60 / h2
            gray = cv2.resize(
                gray,
                None,
                fx=1.0,
                fy=scale_h,
                interpolation=cv2.INTER_CUBIC
            )

        try:
            results = self.reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )
        except Exception:
            return {"text": "", "confidence": 0.0}

        if not results:
            return {"text": "", "confidence": 0.0}

        # ── Filter weak / very short fragments ──────────────────────────
        valid = []
        for bbox, text, conf in results:
            clean = text.replace(" ", "").upper()
            if len(clean) >= 2 and conf >= 0.25:
                # Store with top-left y coord for line sorting
                top_y = min(pt[1] for pt in bbox)
                valid.append((top_y, clean, float(conf)))

        if not valid:
            return {"text": "", "confidence": 0.0}

        # ── Sort fragments top-to-bottom (handles two-line plates) ──────
        valid.sort(key=lambda x: x[0])

        # ── Try merging all fragments into one plate string ──────────────
        merged_text = "".join(frag for _, frag, _ in valid)
        merged_conf = float(np.mean([c for _, _, c in valid]))

        # Try to find a valid plate pattern in the merged string first
        match = self.plate_regex.search(merged_text)
        if match:
            return {
                "text": match.group(0),
                "confidence": round(merged_conf, 4),
            }

        # ── Fallback: pick the single best fragment ──────────────────────
        # (original behaviour — highest confidence fragment of length >= 5)
        best_text = ""
        best_conf = 0.0
        for _, clean, conf in valid:
            if len(clean) >= 5 and conf > best_conf:
                best_text = clean
                best_conf = conf

        if not best_text:
            # Accept shorter fragments if nothing else worked
            for _, clean, conf in valid:
                if conf > best_conf:
                    best_text = clean
                    best_conf = conf

        if not best_text:
            return {"text": "", "confidence": 0.0}

        # Soft regex validation on fallback
        if not self.plate_regex.search(best_text):
            best_conf *= 0.75

        return {
            "text": best_text,
            "confidence": round(best_conf, 4),
        }
