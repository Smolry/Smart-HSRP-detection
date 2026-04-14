"""
ADAPTIVE THRESHOLD LEARNING
============================
Automatically learns optimal decision thresholds.
Adapts to camera quality and lighting conditions.
Optimizes precision/recall balance.
Supports supervised, semi-supervised, and unsupervised modes.
Saves learning progress to disk. Continuous learning.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from collections import deque


# ─────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class ThresholdState:
    """Persistent state for one decision type."""
    decision_type: str
    current_threshold: float
    base_threshold: float
    # Online performance estimates
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    # Temporal tracking
    threshold_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    # Camera quality estimate (0-1)
    estimated_quality: float = 0.5
    # Adaptation momentum
    momentum: float = 0.0
    last_adapted_at: float = field(default_factory=time.time)
    total_samples: int = 0


@dataclass
class LearningConfig:
    """Configuration for threshold learning."""
    learning_rate: float = 0.003         # How fast thresholds adapt (lowered from 0.01 → slower adaptation)
    momentum_factor: float = 0.6         # Momentum for smoother adaptation (lowered from 0.8 → less momentum carry)
    target_precision: float = 0.85       # Desired precision
    target_recall: float = 0.80          # Desired recall
    adaptation_window: int = 200         # Samples before adapting (raised from 100 → adapt less frequently)
    min_threshold: float = 0.20          # Hard lower bound
    max_threshold: float = 0.95          # Hard upper bound
    quality_smoothing: float = 0.05      # EMA for quality estimate (lowered from 0.1 → smoother quality tracking)
    save_interval: int = 500             # Save every N samples
    persistence_path: str = "state/thresholds.json"


# ─────────────────────────────────────────────
# CORE ENGINE
# ─────────────────────────────────────────────

class AdaptiveThresholdLearner:
    """
    Adaptive threshold learning engine.

    Modes:
    - supervised:       uses labelled feedback (ground truth labels)
    - semi_supervised:  uses high-confidence predictions as pseudo-labels
    - unsupervised:     adapts based on confidence distribution statistics

    Usage:
        learner = AdaptiveThresholdLearner()
        learner.observe(
            decision_type="hsrp",
            score=0.72,
            confidence=0.80,
            label=True,       # None for unsupervised
            mode="supervised"
        )
        threshold = learner.get_threshold("hsrp")
    """

    DEFAULT_BASE_THRESHOLDS = {
        "hsrp": 0.50,
        "helmet": 0.40,
        "ocr_confidence": 0.60,
    }

    def __init__(
        self,
        config: Optional[LearningConfig] = None,
        persistence_path: Optional[str] = None,
    ):
        self.config = config or LearningConfig()
        if persistence_path:
            self.config.persistence_path = persistence_path

        self.states: Dict[str, ThresholdState] = {}
        self._observation_buffer: Dict[str, deque] = {}
        self._sample_counter = 0

        self._load_progress()

    # ─────────────────────────────────────────────
    # PUBLIC API
    # ─────────────────────────────────────────────

    def observe(
        self,
        decision_type: str,
        score: float,
        confidence: float,
        label: Optional[bool] = None,
        mode: str = "semi_supervised",
    ) -> Dict[str, Any]:
        """
        Feed one observation into the learner.

        Args:
            decision_type: "hsrp" | "helmet" | "ocr_confidence"
            score:         Raw model score (0–1)
            confidence:    Model confidence (0–1)
            label:         True ground-truth label (or None for unsupervised)
            mode:          "supervised" | "semi_supervised" | "unsupervised"

        Returns:
            Dict with adapted threshold and learning metrics.
        """
        state = self._get_or_create_state(decision_type)
        buf   = self._get_buffer(decision_type)

        buf.append({"score": score, "confidence": confidence, "label": label})
        state.total_samples += 1
        state.confidence_history.append(confidence)
        if len(state.confidence_history) > 500:
            state.confidence_history = state.confidence_history[-500:]

        # Update quality estimate
        state.estimated_quality = (
            self.config.quality_smoothing * confidence
            + (1 - self.config.quality_smoothing) * state.estimated_quality
        )

        # Apply label feedback if available
        if label is not None and mode == "supervised":
            self._update_confusion_matrix(state, score, label)

        # Adapt when enough samples
        adapted = False
        if len(buf) >= self.config.adaptation_window:
            self._adapt_threshold(state, buf, mode)
            buf.clear()
            adapted = True

        # Periodic save
        self._sample_counter += 1
        if self._sample_counter % self.config.save_interval == 0:
            self.save_progress()

        return {
            "decision_type": decision_type,
            "current_threshold": round(state.current_threshold, 4),
            "estimated_quality": round(state.estimated_quality, 4),
            "total_samples": state.total_samples,
            "adapted_this_step": adapted,
            "metrics": self._compute_metrics(state),
        }

    def get_threshold(self, decision_type: str) -> float:
        """Return the current learned threshold for a decision type."""
        state = self._get_or_create_state(decision_type)
        return state.current_threshold

    def get_all_thresholds(self) -> Dict[str, float]:
        """Return all current thresholds."""
        return {dt: s.current_threshold for dt, s in self.states.items()}

    def get_quality_estimate(self, decision_type: str) -> float:
        """Return camera/condition quality estimate (0–1)."""
        state = self._get_or_create_state(decision_type)
        return state.estimated_quality

    def save_progress(self):
        """Persist learned thresholds and state to disk."""
        path = Path(self.config.persistence_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_data = {
            "saved_at": time.time(),
            "states": {},
        }

        for dt, state in self.states.items():
            save_data["states"][dt] = {
                "current_threshold": state.current_threshold,
                "base_threshold": state.base_threshold,
                "estimated_quality": state.estimated_quality,
                "total_samples": state.total_samples,
                "true_positives": state.true_positives,
                "false_positives": state.false_positives,
                "true_negatives": state.true_negatives,
                "false_negatives": state.false_negatives,
                "threshold_history": state.threshold_history[-50:],
            }

        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

    def reset_decision_type(self, decision_type: str):
        """Reset a specific decision type to its base threshold."""
        base = self.DEFAULT_BASE_THRESHOLDS.get(decision_type, 0.5)
        self.states[decision_type] = ThresholdState(
            decision_type=decision_type,
            current_threshold=base,
            base_threshold=base,
        )

    # ─────────────────────────────────────────────
    # INTERNAL ADAPTATION LOGIC
    # ─────────────────────────────────────────────

    def _adapt_threshold(
        self,
        state: ThresholdState,
        buffer: deque,
        mode: str,
    ):
        observations = list(buffer)
        scores = [o["score"] for o in observations]
        confidences = [o["confidence"] for o in observations]
        labels = [o["label"] for o in observations]

        if mode == "supervised":
            gradient = self._supervised_gradient(state, scores, labels)
        elif mode == "semi_supervised":
            gradient = self._semi_supervised_gradient(state, scores, confidences)
        else:  # unsupervised
            gradient = self._unsupervised_gradient(scores, confidences)

        # Apply momentum
        state.momentum = (
            self.config.momentum_factor * state.momentum
            + (1 - self.config.momentum_factor) * gradient
        )

        # Quality-adjusted learning rate: lower quality → slower adaptation
        effective_lr = self.config.learning_rate * (0.5 + 0.5 * state.estimated_quality)

        new_threshold = state.current_threshold + effective_lr * state.momentum
        new_threshold = float(np.clip(
            new_threshold,
            self.config.min_threshold,
            self.config.max_threshold,
        ))

        state.threshold_history.append(new_threshold)
        if len(state.threshold_history) > 200:
            state.threshold_history = state.threshold_history[-200:]

        state.current_threshold = new_threshold
        state.last_adapted_at = time.time()

    def _supervised_gradient(
        self,
        state: ThresholdState,
        scores: List[float],
        labels: List[Optional[bool]],
    ) -> float:
        """
        Gradient based on precision–recall trade-off.
        Positive gradient → raise threshold (fewer positives, higher precision).
        Negative gradient → lower threshold (more positives, higher recall).
        """
        labelled = [(s, l) for s, l in zip(scores, labels) if l is not None]
        if not labelled:
            return 0.0

        tp = sum(1 for s, l in labelled if s >= state.current_threshold and l)
        fp = sum(1 for s, l in labelled if s >= state.current_threshold and not l)
        fn = sum(1 for s, l in labelled if s <  state.current_threshold and l)

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        # Push gradient toward target precision/recall
        precision_error = precision - self.config.target_precision
        recall_error    = recall    - self.config.target_recall

        # If precision is low (too many FP) → raise threshold
        # If recall is low  (too many FN) → lower threshold
        gradient = precision_error - recall_error
        return float(np.clip(gradient, -0.5, 0.5))

    def _semi_supervised_gradient(
        self,
        state: ThresholdState,
        scores: List[float],
        confidences: List[float],
    ) -> float:
        """
        Use high-confidence predictions as pseudo-labels.
        """
        HIGH_CONF = 0.75
        pseudo = [
            (s, s >= state.current_threshold)
            for s, c in zip(scores, confidences)
            if c >= HIGH_CONF
        ]
        if not pseudo:
            return self._unsupervised_gradient(scores, confidences)

        pseudo_labels = [l for _, l in pseudo]
        pseudo_scores = [s for s, _ in pseudo]

        positive_rate = sum(pseudo_labels) / len(pseudo_labels)
        target_rate   = 1.0 - self.config.target_precision  # approximate

        # If too many positives → push threshold up
        # Tighter clip: prevents aggressive threshold swings
        gradient = positive_rate - target_rate
        return float(np.clip(gradient * 0.3, -0.15, 0.15))

    def _unsupervised_gradient(
        self,
        scores: List[float],
        confidences: List[float],
    ) -> float:
        """
        Adapt based on score distribution statistics.
        Keeps threshold at the natural separation point.
        """
        arr = np.array(scores)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))

        # Natural separation heuristic: mean + 0.5 * std
        natural_threshold = mean + 0.5 * std
        quality_avg = float(np.mean(confidences))

        # Only nudge if quality is decent
        if quality_avg < 0.3:
            return 0.0

        # Very gentle gradient toward natural separation
        return float(np.clip(natural_threshold - 0.5, -0.2, 0.2)) * 0.1

    def _update_confusion_matrix(
        self,
        state: ThresholdState,
        score: float,
        label: bool,
    ):
        predicted = score >= state.current_threshold
        if predicted and label:
            state.true_positives += 1
        elif predicted and not label:
            state.false_positives += 1
        elif not predicted and label:
            state.false_negatives += 1
        else:
            state.true_negatives += 1

    def _compute_metrics(self, state: ThresholdState) -> Dict[str, float]:
        tp = state.true_positives
        fp = state.false_positives
        tn = state.true_negatives
        fn = state.false_negatives

        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy  = (tp + tn) / (tp + fp + tn + fn + 1e-8)

        return {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "accuracy":  round(accuracy, 4),
        }

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _get_or_create_state(self, decision_type: str) -> ThresholdState:
        if decision_type not in self.states:
            base = self.DEFAULT_BASE_THRESHOLDS.get(decision_type, 0.5)
            self.states[decision_type] = ThresholdState(
                decision_type=decision_type,
                current_threshold=base,
                base_threshold=base,
            )
        return self.states[decision_type]

    def _get_buffer(self, decision_type: str) -> deque:
        if decision_type not in self._observation_buffer:
            self._observation_buffer[decision_type] = deque(
                maxlen=self.config.adaptation_window * 3
            )
        return self._observation_buffer[decision_type]

    def _load_progress(self):
        """Load previously saved progress if it exists."""
        path = Path(self.config.persistence_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)

            for dt, saved in data.get("states", {}).items():
                base = self.DEFAULT_BASE_THRESHOLDS.get(dt, 0.5)
                state = ThresholdState(
                    decision_type=dt,
                    current_threshold=saved.get("current_threshold", base),
                    base_threshold=saved.get("base_threshold", base),
                    estimated_quality=saved.get("estimated_quality", 0.5),
                    total_samples=saved.get("total_samples", 0),
                    true_positives=saved.get("true_positives", 0),
                    false_positives=saved.get("false_positives", 0),
                    true_negatives=saved.get("true_negatives", 0),
                    false_negatives=saved.get("false_negatives", 0),
                    threshold_history=saved.get("threshold_history", []),
                )
                self.states[dt] = state

        except Exception as e:
            print(f"[AdaptiveThreshold] Could not load saved state: {e}")


# ─────────────────────────────────────────────
# SINGLETON (shared across pipeline)
# ─────────────────────────────────────────────

_global_learner: Optional[AdaptiveThresholdLearner] = None


def get_threshold_learner() -> AdaptiveThresholdLearner:
    global _global_learner
    if _global_learner is None:
        _global_learner = AdaptiveThresholdLearner()
    return _global_learner
