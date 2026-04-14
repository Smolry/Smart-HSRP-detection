"""
VIOLATION PREDICTION ENGINE
=============================
Predicts violations BEFORE they clearly occur.
Early warning system (1–20 frames ahead).
Pattern recognition + trend analysis.
Risk scoring.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class ViolationState(str, Enum):
    NORMAL     = "NORMAL"
    WARNING    = "WARNING"    # Prediction firing
    CONFIRMED  = "CONFIRMED"  # Actual violation
    CLEARED    = "CLEARED"    # Previously warned but did not confirm


@dataclass
class TrackPrediction:
    """Prediction state for a single track."""
    track_id: str
    # Per-type score history (deque of floats)
    hsrp_scores:   deque = field(default_factory=lambda: deque(maxlen=25))
    helmet_scores: deque = field(default_factory=lambda: deque(maxlen=25))
    # Trend accumulators
    hsrp_slope:    float = 0.0
    helmet_slope:  float = 0.0
    # Current state per type
    hsrp_state:    ViolationState = ViolationState.NORMAL
    helmet_state:  ViolationState = ViolationState.NORMAL
    # Risk scores  (0–1)
    hsrp_risk:     float = 0.0
    helmet_risk:   float = 0.0
    # Frame counters
    last_frame:    int = 0
    first_frame:   int = 0
    # Prediction tracking
    hsrp_warned_at:   Optional[int] = None
    helmet_warned_at: Optional[int] = None


@dataclass
class PredictionConfig:
    """Configuration for the prediction engine."""
    warning_threshold:   float = 0.55   # Risk score to trigger WARNING
    confirm_threshold:   float = 0.70   # Risk score to trigger CONFIRMED
    trend_weight:        float = 0.40   # Weight of trend vs raw score
    min_history:         int   = 4      # Min frames before predicting
    lookahead_frames:    int   = 10     # How many frames ahead to warn
    ema_alpha:           float = 0.35   # EMA smoothing
    slope_window:        int   = 5      # Frames for slope estimation
    pattern_bonus:       float = 0.10   # Bonus for consistent patterns


class ViolationPredictor:
    """
    Predicts violations before they are confirmed.

    Algorithm:
    1. For each track, maintain a rolling score history.
    2. Compute EMA-smoothed score.
    3. Estimate linear trend (slope) over last N frames.
    4. Combine smoothed score + trend extrapolation → risk score.
    5. Apply pattern recognition bonus for oscillating / rising patterns.
    6. Emit WARNING when risk ≥ warning_threshold.
    7. Emit CONFIRMED when risk ≥ confirm_threshold.

    Usage:
        predictor = ViolationPredictor()
        result = predictor.update(
            track_id="veh_12",
            frame_id=150,
            hsrp_score=0.62,
            helmet_score=0.71,
        )
        if result["hsrp"]["state"] == "WARNING":
            # Start preparing evidence capture
    """

    def __init__(self, config: Optional[PredictionConfig] = None):
        self.config = config or PredictionConfig()
        self.tracks: Dict[str, TrackPrediction] = {}

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def update(
        self,
        track_id: str,
        frame_id: int,
        hsrp_score: Optional[float] = None,
        helmet_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Update prediction state for a track.

        Args:
            track_id:     Unique vehicle track id
            frame_id:     Current frame number
            hsrp_score:   Raw HSRP violation probability (0–1)
            helmet_score: Raw no-helmet probability (0–1)

        Returns:
            Dict with per-type predictions:
            {
              "hsrp": {
                "state":        "NORMAL" | "WARNING" | "CONFIRMED",
                "risk_score":   0.0–1.0,
                "frames_ahead": int,    # How many frames ahead the warning is
              },
              "helmet": { ... },
              "any_warning":   bool,
              "any_confirmed": bool,
            }
        """
        track = self._get_or_create_track(track_id, frame_id)
        track.last_frame = frame_id

        hsrp_result   = self._update_type(track, "hsrp",   hsrp_score,   frame_id)
        helmet_result = self._update_type(track, "helmet", helmet_score, frame_id)

        any_warning   = (
            hsrp_result["state"]   == ViolationState.WARNING
            or helmet_result["state"] == ViolationState.WARNING
        )
        any_confirmed = (
            hsrp_result["state"]   == ViolationState.CONFIRMED
            or helmet_result["state"] == ViolationState.CONFIRMED
        )

        return {
            "track_id":     track_id,
            "frame_id":     frame_id,
            "hsrp":         hsrp_result,
            "helmet":       helmet_result,
            "any_warning":  any_warning,
            "any_confirmed": any_confirmed,
        }

    def get_risk(self, track_id: str, violation_type: str) -> float:
        """Return the latest risk score for a track and type."""
        track = self.tracks.get(track_id)
        if track is None:
            return 0.0
        return track.hsrp_risk if violation_type == "hsrp" else track.helmet_risk

    def get_state(self, track_id: str, violation_type: str) -> ViolationState:
        """Return the current violation state for a track and type."""
        track = self.tracks.get(track_id)
        if track is None:
            return ViolationState.NORMAL
        return track.hsrp_state if violation_type == "hsrp" else track.helmet_state

    def cleanup_old_tracks(self, current_frame: int, max_age: int = 150):
        """Remove stale tracks."""
        stale = [
            tid for tid, t in self.tracks.items()
            if current_frame - t.last_frame > max_age
        ]
        for tid in stale:
            del self.tracks[tid]
        return len(stale)

    # ─────────────────────────────────────────────
    # INTERNAL
    # ─────────────────────────────────────────────

    def _update_type(
        self,
        track: TrackPrediction,
        vtype: str,
        raw_score: Optional[float],
        frame_id: int,
    ) -> Dict[str, Any]:
        """Process one score update for a single violation type."""
        if raw_score is None:
            scores_deq = track.hsrp_scores if vtype == "hsrp" else track.helmet_scores
            current_state = track.hsrp_state if vtype == "hsrp" else track.helmet_state
            risk = track.hsrp_risk if vtype == "hsrp" else track.helmet_risk
            return {
                "state": current_state,
                "risk_score": risk,
                "frames_ahead": 0,
                "trend": 0.0,
            }

        # Append to history
        scores_deq = track.hsrp_scores if vtype == "hsrp" else track.helmet_scores
        scores_deq.append(raw_score)

        if len(scores_deq) < self.config.min_history:
            risk = float(np.mean(list(scores_deq)))
            if vtype == "hsrp":
                track.hsrp_risk = risk
            else:
                track.helmet_risk = risk
            return {
                "state": ViolationState.NORMAL,
                "risk_score": round(risk, 4),
                "frames_ahead": 0,
                "trend": 0.0,
            }

        arr = np.array(list(scores_deq))

        # ── EMA smoothed score ──────────────────────
        ema = arr[-1]
        for v in reversed(arr[:-1]):
            ema = self.config.ema_alpha * v + (1 - self.config.ema_alpha) * ema

        # ── Trend (slope) ───────────────────────────
        window = min(self.config.slope_window, len(arr))
        recent = arr[-window:]
        x = np.arange(window)
        slope = float(np.polyfit(x, recent, 1)[0]) if window >= 2 else 0.0

        # ── Extrapolated future score ────────────────
        future_score = ema + slope * self.config.lookahead_frames
        future_score = float(np.clip(future_score, 0.0, 1.0))

        # ── Pattern bonus ────────────────────────────
        pattern_bonus = self._detect_pattern(arr)

        # ── Composite risk ───────────────────────────
        risk = (
            (1 - self.config.trend_weight) * ema
            + self.config.trend_weight * future_score
            + pattern_bonus
        )
        risk = float(np.clip(risk, 0.0, 1.0))

        # ── State machine ────────────────────────────
        if vtype == "hsrp":
            old_state  = track.hsrp_state
            track.hsrp_risk = risk
        else:
            old_state  = track.helmet_state
            track.helmet_risk = risk

        new_state  = self._transition(old_state, risk)

        if vtype == "hsrp":
            track.hsrp_state = new_state
            if new_state == ViolationState.WARNING and track.hsrp_warned_at is None:
                track.hsrp_warned_at = frame_id
        else:
            track.helmet_state = new_state
            if new_state == ViolationState.WARNING and track.helmet_warned_at is None:
                track.helmet_warned_at = frame_id

        # Frames ahead: how long since warning fired
        warned_at = track.hsrp_warned_at if vtype == "hsrp" else track.helmet_warned_at
        frames_ahead = (frame_id - warned_at) if warned_at is not None else 0

        return {
            "state":       new_state,
            "risk_score":  round(risk, 4),
            "frames_ahead": frames_ahead,
            "trend":       round(slope, 4),
            "ema_score":   round(ema, 4),
        }

    def _transition(
        self,
        current_state: ViolationState,
        risk: float,
    ) -> ViolationState:
        """State machine transitions."""
        cfg = self.config
        if risk >= cfg.confirm_threshold:
            return ViolationState.CONFIRMED
        if risk >= cfg.warning_threshold:
            return ViolationState.WARNING
        if current_state == ViolationState.WARNING and risk < cfg.warning_threshold * 0.8:
            return ViolationState.CLEARED
        if current_state == ViolationState.CLEARED:
            return ViolationState.NORMAL
        return current_state if current_state != ViolationState.CONFIRMED else ViolationState.CONFIRMED

    def _detect_pattern(self, arr: np.ndarray) -> float:
        """
        Detect risk-increasing patterns and return a bonus (0–pattern_bonus).
        Patterns: monotonic rise, oscillation around threshold, sustained high.
        """
        if len(arr) < 4:
            return 0.0

        recent = arr[-6:]

        # Rising pattern: last 3 > first 3
        if len(recent) >= 6:
            first_half = recent[:3].mean()
            second_half = recent[3:].mean()
            if second_half > first_half + 0.08:
                return self.config.pattern_bonus

        # Sustained high: all recent above 0.5
        if (recent > 0.50).all():
            return self.config.pattern_bonus * 0.7

        return 0.0

    def _get_or_create_track(self, track_id: str, frame_id: int) -> TrackPrediction:
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackPrediction(
                track_id=track_id,
                first_frame=frame_id,
                last_frame=frame_id,
            )
        return self.tracks[track_id]


# ─────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────

_global_predictor: Optional[ViolationPredictor] = None


def get_violation_predictor() -> ViolationPredictor:
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = ViolationPredictor()
    return _global_predictor
