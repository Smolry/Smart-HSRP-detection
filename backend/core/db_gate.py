"""
DATABASE GATING LOGIC
======================
Smart storage with:
- Quality scoring (violation confidence avg, stability, temporal consistency, consecutive detections)
- Prediction-aware storage (marks for manual review if no prediction preceded detection)
- Duplicate debouncing (same plate + violation type on same day)
- Final track-level decision using complete metadata
"""

import hashlib
import time
from datetime import datetime, date
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TrackAccumulator:
    """
    Collects all data for a track until it disappears.
    Averages are computed when the track is finalised.
    """
    track_id: str
    violation_type: Optional[str] = None
    vehicle_class: str = "unknown"

    # Quality signals
    violation_confidences:   List[float] = field(default_factory=list)
    stability_scores:        List[float] = field(default_factory=list)
    temporal_consistencies:  List[float] = field(default_factory=list)
    consecutive_counts:      List[int]   = field(default_factory=list)

    # OCR / plate info
    plate_texts:             List[str]   = field(default_factory=list)
    plate_confidences:       List[float] = field(default_factory=list)

    # Prediction tracking
    prediction_warned:   bool = False  # Did prediction engine warn before detection?
    prediction_risk_max: float = 0.0

    # Frame range
    first_frame: int = 0
    last_frame:  int = 0

    # Raw violation events
    violation_events: int = 0


@dataclass
class GatingConfig:
    """Configuration for the gating logic."""
    # Quality thresholds
    min_violation_conf_avg:  float = 0.65
    min_stability_avg:       float = 0.60
    min_temporal_consistency: float = 0.55
    min_consecutive_detections: int = 3

    # Prediction requirement
    require_prediction_for_auto_store: bool = True

    # Deduplication
    dedup_same_day: bool = True  # Same plate + type on same day → skip


class DatabaseGate:
    """
    Evaluates whether a completed track should be stored in the database.

    Usage:
        gate = DatabaseGate()

        # Feed data frame-by-frame
        gate.feed(track_id, frame_id, violation_conf=0.82, stability=0.75, ...)

        # When track disappears, evaluate
        decision = gate.evaluate_track(track_id)
        if decision["should_store"]:
            db.insert(decision["record"])
    """

    def __init__(self, config: Optional[GatingConfig] = None):
        self.config = config or GatingConfig()
        self._tracks: Dict[str, TrackAccumulator] = {}

        # Dedup cache: (plate_text, violation_type, date_str) → stored_at
        self._stored_today: Dict[str, float] = {}

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def feed(
        self,
        track_id: str,
        frame_id: int,
        *,
        violation_type: Optional[str] = None,
        vehicle_class: str = "unknown",
        violation_conf: float = 0.0,
        stability: float = 0.0,
        temporal_consistency: float = 0.0,
        consecutive_count: int = 0,
        plate_text: Optional[str] = None,
        plate_conf: float = 0.0,
        prediction_warned: bool = False,
        prediction_risk: float = 0.0,
    ):
        """Feed one frame's worth of data for a track."""
        acc = self._get_or_create(track_id, frame_id)
        acc.last_frame = frame_id

        if violation_type and not acc.violation_type:
            acc.violation_type = violation_type
        if vehicle_class != "unknown":
            acc.vehicle_class = vehicle_class

        if violation_conf > 0:
            acc.violation_confidences.append(violation_conf)
        if stability > 0:
            acc.stability_scores.append(stability)
        if temporal_consistency > 0:
            acc.temporal_consistencies.append(temporal_consistency)
        if consecutive_count > 0:
            acc.consecutive_counts.append(consecutive_count)
        if plate_text:
            acc.plate_texts.append(plate_text)
            acc.plate_confidences.append(plate_conf)

        if prediction_warned:
            acc.prediction_warned = True
        acc.prediction_risk_max = max(acc.prediction_risk_max, prediction_risk)
        acc.violation_events += 1

    def evaluate_track(self, track_id: str) -> Dict[str, Any]:
        """
        Evaluate whether a track's violations should be stored.

        Returns a dict with:
        - should_store (bool)
        - needs_manual_review (bool)
        - quality_score (float 0–1)
        - record (dict, the DB-ready payload)
        - reason (str, why this decision was made)
        """
        acc = self._tracks.get(track_id)
        if acc is None:
            return {"should_store": False, "reason": "Track not found"}

        # ── Compute quality score ────────────────────
        quality = self._compute_quality(acc)

        # ── Check prediction requirement ─────────────
        needs_manual_review = False
        if self.config.require_prediction_for_auto_store:
            if not acc.prediction_warned:
                # Violation appeared without any prediction → suspicious
                needs_manual_review = True

        # ── Threshold gate ───────────────────────────
        should_store = quality["overall"] >= 0.55 and acc.violation_type is not None

        # ── Best plate number ────────────────────────
        plate_number = self._best_plate(acc)

        # ── Dedup check ──────────────────────────────
        if should_store and plate_number and self.config.dedup_same_day:
            if self._is_duplicate(plate_number, acc.violation_type):
                return {
                    "should_store": False,
                    "reason": "Duplicate: same plate + violation type already stored today",
                    "quality_score": quality["overall"],
                }
            self._mark_stored(plate_number, acc.violation_type)

        reason = (
            "High quality detection with prediction"
            if (should_store and not needs_manual_review)
            else "Stored with manual review flag (no prediction preceded detection)"
            if (should_store and needs_manual_review)
            else f"Below quality threshold ({quality['overall']:.2f} < 0.55)"
        )

        record = self._build_record(acc, quality, plate_number, needs_manual_review)

        return {
            "should_store":       should_store,
            "needs_manual_review": needs_manual_review,
            "quality_score":      quality["overall"],
            "record":             record,
            "reason":             reason,
        }

    def remove_track(self, track_id: str):
        """Remove track data after it has been processed."""
        self._tracks.pop(track_id, None)

    def cleanup_old_tracks(self, current_frame: int, max_age: int = 200):
        """Auto-remove very old tracks."""
        stale = [
            tid for tid, acc in self._tracks.items()
            if current_frame - acc.last_frame > max_age
        ]
        for tid in stale:
            del self._tracks[tid]
        return len(stale)

    # ─────────────────────────────────────────────
    # QUALITY SCORING
    # ─────────────────────────────────────────────

    def _compute_quality(self, acc: TrackAccumulator) -> Dict[str, float]:
        """
        Compute a composite quality score from:
        - violation confidence average
        - stability score
        - temporal consistency
        - minimum consecutive detection count (normalized)
        """
        def _safe_avg(lst: List[float]) -> float:
            return float(np.mean(lst)) if lst else 0.0

        conf_avg  = _safe_avg(acc.violation_confidences[-10:])
        stab_avg  = _safe_avg(acc.stability_scores[-10:])
        temp_avg  = _safe_avg(acc.temporal_consistencies[-10:])

        consec = max(acc.consecutive_counts) if acc.consecutive_counts else 0
        consec_norm = min(1.0, consec / max(1, self.config.min_consecutive_detections * 2))

        # Weighted composite
        overall = (
            0.35 * conf_avg
            + 0.25 * stab_avg
            + 0.25 * temp_avg
            + 0.15 * consec_norm
        )

        return {
            "overall":             round(float(overall), 4),
            "conf_avg":            round(conf_avg, 4),
            "stability_avg":       round(stab_avg, 4),
            "temporal_avg":        round(temp_avg, 4),
            "consecutive_norm":    round(consec_norm, 4),
            "max_consecutive":     consec,
            "total_events":        acc.violation_events,
        }

    # ─────────────────────────────────────────────
    # RECORD BUILDING
    # ─────────────────────────────────────────────

    def _build_record(
        self,
        acc: TrackAccumulator,
        quality: Dict[str, float],
        plate_number: Optional[str],
        needs_manual_review: bool,
    ) -> Dict[str, Any]:
        """Build the DB-ready record."""
        return {
            # Core identifiers
            "track_id":             acc.track_id,
            "vehicle_number":       plate_number,
            "vehicle_class":        acc.vehicle_class,
            # Timestamps (DB will add exact timestamp on insert)
            "first_frame":          acc.first_frame,
            "last_frame":           acc.last_frame,
            "track_duration_frames": acc.last_frame - acc.first_frame,
            # Violation details
            "violation_type":       acc.violation_type,
            "violation_confidence": quality["conf_avg"],
            # Quality breakdown
            "stability_score":      quality["stability_avg"],
            "temporal_consistency": quality["temporal_avg"],
            "consecutive_detections": quality["max_consecutive"],
            "quality_score":        quality["overall"],
            # Flags
            "needs_manual_review":  needs_manual_review,
            "prediction_preceded":  acc.prediction_warned,
            "prediction_risk_max":  round(acc.prediction_risk_max, 4),
            # Screenshot / metadata links (to be filled later)
            "screenshot_path":      None,
            "metadata_path":        None,
        }

    # ─────────────────────────────────────────────
    # DEDUP
    # ─────────────────────────────────────────────

    def _dedup_key(self, plate: str, violation_type: str) -> str:
        today = date.today().isoformat()
        raw = f"{plate}|{violation_type}|{today}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_duplicate(self, plate: str, violation_type: str) -> bool:
        key = self._dedup_key(plate, violation_type)
        return key in self._stored_today

    def _mark_stored(self, plate: str, violation_type: str):
        key = self._dedup_key(plate, violation_type)
        self._stored_today[key] = time.time()

        # Prune old entries (keep only today's)
        today = date.today().isoformat()
        to_del = []
        for k in self._stored_today:
            try:
                stored_date = datetime.fromtimestamp(self._stored_today[k]).date().isoformat()
                if stored_date != today:
                    to_del.append(k)
            except Exception:
                pass
        for k in to_del:
            del self._stored_today[k]

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _best_plate(self, acc: TrackAccumulator) -> Optional[str]:
        """Return the most-voted plate text with highest confidence."""
        if not acc.plate_texts:
            return None
        from collections import Counter
        cnt = Counter(acc.plate_texts)
        return cnt.most_common(1)[0][0]

    def _get_or_create(self, track_id: str, frame_id: int) -> TrackAccumulator:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackAccumulator(
                track_id=track_id,
                first_frame=frame_id,
                last_frame=frame_id,
            )
        return self._tracks[track_id]


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_global_gate: Optional[DatabaseGate] = None


def get_database_gate() -> DatabaseGate:
    global _global_gate
    if _global_gate is None:
        _global_gate = DatabaseGate()
    return _global_gate
