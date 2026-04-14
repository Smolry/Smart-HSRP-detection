from collections import defaultdict, Counter
import re


PLATE_REGEX = re.compile(r"[A-Z0-9]{6,10}")


class OCRStabilizer:
    """
    Track-ID based OCR stabilizer.

    - One history per vehicle track
    - Confidence-weighted voting
    - Regex filtering
    """

    def __init__(self, max_history: int = 7):
        self.max_history = max_history
        self.history = defaultdict(list)

    def update(self, *, vehicle_id: str, text: str, confidence: float):
        """
        Update OCR history for a given vehicle track.

        Args:
            vehicle_id: persistent track id
            text: raw OCR text
            confidence: OCR confidence
        """

        if not text or not vehicle_id:
            return {"text": None, "confidence": 0.0}

        text = self._clean_text(text)

        match = PLATE_REGEX.search(text)
        if not match:
             return {"text": None, "confidence": 0.0}
        
        text = match.group(0)

        hist = self.history[vehicle_id]
        hist.append((text, confidence))

        if len(hist) > self.max_history:
            hist.pop(0)

        return self._vote(hist)

    # -------------------------------------------------
    # INTERNALS
    # -------------------------------------------------

    def _vote(self, history):
        scores = defaultdict(float)

        for text, conf in history:
            scores[text] += conf

        best_text, best_score = max(scores.items(), key=lambda x: x[1])

        count = sum(1 for t, _ in history if t == best_text)
        avg_conf = best_score / max(1, count)


        return {
            "text": best_text,
            "confidence": round(avg_conf, 3)
        }

    def _clean_text(self, text: str) -> str:
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text
