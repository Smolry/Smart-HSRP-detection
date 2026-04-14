"""
TRACK-LEVEL ADAPTIVE TEMPORAL DECISION FUSION
==============================================
Advanced temporal decision fusion with:
1. Adaptive bias capping (prevents over-correction)
2. Exponential decay (recent frames matter more)
3. EMA smoothing (reduces noise)
4. Drift control (prevents bias accumulation)
5. Frame skip awareness (handles sparse processing)

This system maintains per-track decision states and adaptively
adjusts model thresholds to reduce classification fluctuations.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class DecisionState:
    """
    Per-track decision state for a single classification type.
    
    Attributes:
        history: Recent decision history (score, frame_id, timestamp)
        ema_score: Exponentially weighted moving average score
        bias: Current adaptive bias applied to threshold
        consecutive_positives: Count of consecutive positive detections
        consecutive_negatives: Count of consecutive negative detections
        last_update_frame: Last frame this was updated
        total_observations: Total number of observations
        confidence_sum: Sum of confidences for averaging
    """
    history: deque = field(default_factory=lambda: deque(maxlen=15))
    ema_score: float = 0.5
    bias: float = 0.0
    consecutive_positives: int = 0
    consecutive_negatives: int = 0
    last_update_frame: int = 0
    total_observations: int = 0
    confidence_sum: float = 0.0
    last_decision: Optional[bool] = None
    decision_streak: int = 0


class TemporalFusionConfig:
    """
    Configuration for temporal fusion parameters.
    
    Tuning guide:
    - ema_alpha: Higher = more reactive (0.2-0.4 for smooth, 0.5-0.7 for responsive)
    - bias_cap: Higher = stronger correction (0.05-0.10 typical)
    - decay_rate: Higher = faster forgetting (0.90-0.98 typical)
    - drift_threshold: Lower = stricter drift control (0.15-0.30 typical)
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.2,           # EMA smoothing factor (lowered from 0.3 → less reactive, less bleed)
        bias_cap: float = 0.05,           # Maximum bias adjustment (lowered from 0.08 → softer threshold shift)
        decay_rate: float = 0.92,         # Exponential decay per frame (lowered from 0.95 → faster forgetting)
        drift_threshold: float = 0.15,    # Max allowed bias drift (tightened from 0.20 → stricter control)
        min_history: int = 3,             # Minimum observations before fusion
        max_history: int = 15,            # Maximum history length
        confidence_weight: float = 0.6,   # Weight for confidence in decisions (lowered from 0.7)
        stability_bonus: float = 0.01,    # Bonus for consistent decisions (lowered from 0.02)
        frame_skip_decay: float = 0.96,   # Decay when frames are skipped (lowered from 0.98 → stronger gap decay)
    ):
        self.ema_alpha = ema_alpha
        self.bias_cap = bias_cap
        self.decay_rate = decay_rate
        self.drift_threshold = drift_threshold
        self.min_history = min_history
        self.max_history = max_history
        self.confidence_weight = confidence_weight
        self.stability_bonus = stability_bonus
        self.frame_skip_decay = frame_skip_decay
        
        # Derived parameters
        self.min_confidence = 0.3  # Minimum confidence to trust
        self.strong_confidence = 0.75  # Threshold for strong decisions


class TemporalDecisionFusion:
    """
    Adaptive temporal decision fusion engine.
    
    This class implements sophisticated temporal logic to:
    1. Reduce false positives/negatives through temporal voting
    2. Smooth classification fluctuations across frames
    3. Adapt decision thresholds based on track history
    4. Control bias drift to prevent over-correction
    
    Usage:
        fusion = TemporalDecisionFusion(config)
        decision = fusion.update(
            track_id="veh_123",
            decision_type="hsrp",
            raw_score=0.65,
            confidence=0.82,
            frame_id=150
        )
    """
    
    def __init__(self, config: Optional[TemporalFusionConfig] = None):
        self.config = config or TemporalFusionConfig()
        
        # Track states: {track_id: {decision_type: DecisionState}}
        self.track_states: Dict[str, Dict[str, DecisionState]] = {}
        
        # Global statistics for calibration
        self.global_stats = {
            "total_decisions": 0,
            "positive_decisions": 0,
            "negative_decisions": 0,
            "avg_confidence": 0.0,
        }

        self.debug = True
        self.debug_track = "veh_0"       # Set to "veh_0" to filter
        self.debug_type = "helmet"    # Only print for helmet
    
    def update(
        self,
        track_id: str,
        decision_type: str,  # e.g., "hsrp", "helmet", "ocr_valid"
        raw_score: float,    # Raw model output (0-1, higher = positive)
        confidence: float,   # Model confidence (0-1)
        frame_id: int,
        base_threshold: float = 0.5,  # Default decision threshold
    ) -> Dict[str, Any]:
        """
        Update temporal state and return fused decision.
        
        Args:
            track_id: Unique track identifier
            decision_type: Type of decision (hsrp, helmet, etc.)
            raw_score: Raw model prediction score
            confidence: Model confidence in prediction
            frame_id: Current frame number
            base_threshold: Base threshold for binary decision
            
        Returns:
            Dict containing:
                - fused_score: Temporally smoothed score
                - adjusted_threshold: Adaptively adjusted threshold
                - decision: Binary decision (True/False)
                - confidence: Final confidence
                - bias: Current bias value
                - stability: Decision stability metric
        """
        
        # Initialize track state if needed
        if track_id not in self.track_states:
            self.track_states[track_id] = {}
        
        if decision_type not in self.track_states[track_id]:
            self.track_states[track_id][decision_type] = DecisionState()
        
        state = self.track_states[track_id][decision_type]
        
        # Calculate frame skip decay
        frames_skipped = max(0, frame_id - state.last_update_frame - 1)
        skip_decay = self.config.frame_skip_decay ** frames_skipped
        
        # Apply decay to existing state
        state.ema_score *= skip_decay
        state.bias *= (skip_decay * self.config.decay_rate)
        
        # Update history with weighted observation
        timestamp = time.time()
        state.history.append({
            'score': raw_score,
            'confidence': confidence,
            'frame_id': frame_id,
            'timestamp': timestamp,
        })
        
        state.total_observations += 1
        state.confidence_sum += confidence
        state.last_update_frame = frame_id
        
        # EMA smoothing
        state.ema_score = (
            self.config.ema_alpha * raw_score +
            (1 - self.config.ema_alpha) * state.ema_score
        )
        
        # Calculate fused score with confidence weighting
        fused_score = self._calculate_fused_score(state, raw_score, confidence)
        
        # Adaptive bias calculation
        bias = self._calculate_adaptive_bias(state, fused_score, base_threshold)
        
        # Drift control
        bias = self._apply_drift_control(bias, state)
        
        state.bias = bias
        
        # Adjusted threshold
        adjusted_threshold = base_threshold + bias
        adjusted_threshold = np.clip(adjusted_threshold, 0.1, 0.9)
        
        # Make decision
        decision = fused_score >= adjusted_threshold

        # ---------------- DEBUG BLOCK ----------------
        if (
            self.debug
            and decision_type == self.debug_type
            and (self.debug_track is None or track_id == self.debug_track)
        ):
            print(
                f"[TEMPORAL DEBUG] "
                f"Frame:{frame_id} | "
                f"Track:{track_id} | "
                f"Type:{decision_type} | "
                f"Raw:{raw_score:.3f} | "
                f"EMA:{state.ema_score:.3f} | "
                f"Fused:{fused_score:.3f} | "
                f"Bias:{bias:.3f} | "
                f"AdjThr:{adjusted_threshold:.3f} | "
                f"Decision:{decision} | "
                f"Streak:{state.decision_streak}"
            )
# ------------------------------------------------
        
        # Update streak tracking
        if state.last_decision == decision:
            state.decision_streak += 1
        else:
            state.decision_streak = 1
        state.last_decision = decision
        
        # Update consecutive counters
        if decision:
            state.consecutive_positives += 1
            state.consecutive_negatives = 0
        else:
            state.consecutive_negatives += 1
            state.consecutive_positives = 0
        
        # Calculate stability metric
        stability = self._calculate_stability(state)
        
        # Final confidence with stability bonus
        final_confidence = confidence
        if state.decision_streak >= 3:
            final_confidence = min(1.0, confidence + self.config.stability_bonus)
        
        # Update global stats
        self._update_global_stats(decision, confidence)
        
        return {
            'decision': decision,
            'fused_score': round(fused_score, 4),
            'raw_score': round(raw_score, 4),
            'adjusted_threshold': round(adjusted_threshold, 4),
            'base_threshold': base_threshold,
            'bias': round(bias, 4),
            'confidence': round(final_confidence, 4),
            'stability': round(stability, 4),
            'consecutive_streak': state.decision_streak,
            'history_size': len(state.history),
        }
    
    def _calculate_fused_score(
        self,
        state: DecisionState,
        raw_score: float,
        confidence: float
    ) -> float:
        """
        Calculate temporally fused score with confidence weighting.
        """
        
        if len(state.history) < self.config.min_history:
            # Not enough history, use EMA only
            return state.ema_score
        
        # Confidence-weighted temporal aggregation
        recent_scores = []
        recent_confidences = []
        
        for obs in list(state.history)[-7:]:  # Last 7 observations
            recent_scores.append(obs['score'])
            recent_confidences.append(obs['confidence'])
        
        # Weighted average
        weights = np.array(recent_confidences)
        weights = weights / (weights.sum() + 1e-8)
        
        weighted_avg = np.sum(np.array(recent_scores) * weights)
        
        # Combine EMA and weighted average
        fused = (
            self.config.confidence_weight * weighted_avg +
            (1 - self.config.confidence_weight) * state.ema_score
        )
        
        return fused
    
    def _calculate_adaptive_bias(
        self,
        state: DecisionState,
        fused_score: float,
        base_threshold: float
    ) -> float:
        """
        Calculate adaptive bias to adjust threshold.
        
        Logic:
        - If consistently above threshold → negative bias (make it harder)
        - If consistently below threshold → positive bias (make it easier)
        - Cap bias to prevent over-correction
        """
        
        if len(state.history) < self.config.min_history:
            return 0.0
        
        # Calculate how far from threshold
        deviation = fused_score - base_threshold
        
        # Direction-based bias
        if state.consecutive_positives >= 7:
            # Too many positives, make it harder
            bias = -self.config.bias_cap * 0.5
        elif state.consecutive_negatives >= 7:
            # Too many negatives, make it easier
            bias = self.config.bias_cap * 0.5
        else:
            # Proportional bias based on deviation
            bias = -deviation * 0.3
        
        # Apply cap
        bias = np.clip(bias, -self.config.bias_cap, self.config.bias_cap)
        
        # Smooth bias changes
        smoothed_bias = 0.7 * bias + 0.3 * state.bias
        
        return smoothed_bias
    
    def _apply_drift_control(self, bias: float, state: DecisionState) -> float:
        """
        Prevent bias from drifting too far from zero.
        """
        
        # If bias is drifting too much, pull it back
        if abs(bias) > self.config.drift_threshold:
            # Exponential pullback
            pullback_factor = 0.8
            bias = bias * pullback_factor
        
        # Long-term drift correction
        if len(state.history) >= 10:
            avg_bias = np.mean([abs(state.bias)] * 5) if state.bias != 0 else 0
            if avg_bias > self.config.drift_threshold:
                bias *= 0.9  # Gradual correction
        
        return bias
    
    def _calculate_stability(self, state: DecisionState) -> float:
        """
        Calculate decision stability metric (0-1).
        
        Higher = more stable decisions over time
        """
        
        if len(state.history) < 3:
            return 0.5
        
        # Check variance in recent decisions
        recent = list(state.history)[-7:]
        scores = [obs['score'] for obs in recent]
        variance = np.var(scores)
        
        # Lower variance = higher stability
        stability = 1.0 / (1.0 + variance * 10)
        
        # Boost stability for long decision streaks
        if state.decision_streak >= 5:
            stability = min(1.0, stability + 0.15)
        
        return stability
    
    def _update_global_stats(self, decision: bool, confidence: float):
        """Update global statistics for calibration."""
        
        self.global_stats['total_decisions'] += 1
        if decision:
            self.global_stats['positive_decisions'] += 1
        else:
            self.global_stats['negative_decisions'] += 1
        
        # Running average confidence
        n = self.global_stats['total_decisions']
        old_avg = self.global_stats['avg_confidence']
        self.global_stats['avg_confidence'] = (
            (old_avg * (n - 1) + confidence) / n
        )
    
    def get_track_state(self, track_id: str, decision_type: str) -> Optional[DecisionState]:
        """Get current state for a track and decision type."""
        
        if track_id not in self.track_states:
            return None
        return self.track_states[track_id].get(decision_type)
    
    def reset_track(self, track_id: str, decision_type: Optional[str] = None):
        """Reset state for a track (useful when track ends)."""
        
        if track_id not in self.track_states:
            return
        
        if decision_type:
            if decision_type in self.track_states[track_id]:
                del self.track_states[track_id][decision_type]
        else:
            del self.track_states[track_id]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fusion engine statistics."""
        
        return {
            'global_stats': self.global_stats,
            'active_tracks': len(self.track_states),
            'total_states': sum(
                len(types) for types in self.track_states.values()
            ),
        }
    
    def cleanup_old_tracks(self, current_frame: int, max_age: int = 100):
        """
        Clean up tracks that haven't been updated recently.
        
        Args:
            current_frame: Current frame number
            max_age: Maximum frames since last update before cleanup
        """
        
        tracks_to_remove = []
        
        for track_id, decision_types in self.track_states.items():
            # Check if any decision type is still active
            all_old = True
            for decision_type, state in decision_types.items():
                if current_frame - state.last_update_frame <= max_age:
                    all_old = False
                    break
            
            if all_old:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_states[track_id]
        
        return len(tracks_to_remove)
