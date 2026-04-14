"""
INTEGRATED DECISION MANAGERS
=============================
High-level decision managers that integrate temporal fusion
with existing model outputs.
"""

from typing import Dict, Any, Optional
from backend.core.temporal_fusion import TemporalDecisionFusion, TemporalFusionConfig
import numpy as np


# ============================================================
# HSRP DECISION MANAGER
# ============================================================

class HSRPDecisionManager:

    def __init__(self, config: Optional[TemporalFusionConfig] = None):

        if config is None:
            config = TemporalFusionConfig(
                ema_alpha=0.20,
                bias_cap=0.05,
                decay_rate=0.92,
                drift_threshold=0.15,
                min_history=3,
                stability_bonus=0.01,
            )

        self.fusion = TemporalDecisionFusion(config)
        self.base_threshold = 0.5

    def process(
        self,
        track_id: str,
        hsrp_result: Dict[str, Any],
        frame_id: int,
    ) -> Dict[str, Any]:

        if not hsrp_result or hsrp_result.get("label") is None:
            return {
                "label": None,
                "confidence": 0.0,
                "is_hsrp": None,
                "raw_label": None,
                "raw_confidence": 0.0,
                "temporal_info": None,
            }

        raw_score = hsrp_result["prob_non_hsrp"]
        raw_confidence = hsrp_result["confidence"]

        fusion_result = self.fusion.update(
            track_id=track_id,
            decision_type="hsrp",
            raw_score=raw_score,
            confidence=raw_confidence,
            frame_id=frame_id,
            base_threshold=self.base_threshold,
        )

        is_non_hsrp = fusion_result["decision"]
        is_hsrp = not is_non_hsrp

        return {
            "label": "non_hsrp" if is_non_hsrp else "hsrp",
            "confidence": fusion_result["confidence"],
            "is_hsrp": is_hsrp,
            "is_violation": is_non_hsrp,
            "raw_label": hsrp_result["label"],
            "raw_confidence": raw_confidence,
            "raw_prob_non_hsrp": raw_score,
            "temporal_info": {
                "fused_score": fusion_result["fused_score"],
                "adjusted_threshold": fusion_result["adjusted_threshold"],
                "bias": fusion_result["bias"],
                "stability": fusion_result["stability"],
                "consecutive_streak": fusion_result["consecutive_streak"],
            },
        }


# ============================================================
# HELMET DECISION MANAGER (Safer Violation Logic Added)
# ============================================================

class HelmetDecisionManager:

    def __init__(self, config: Optional[TemporalFusionConfig] = None):

        if config is None:
            config = TemporalFusionConfig(
                ema_alpha=0.20,
                bias_cap=0.04,
                decay_rate=0.92,
                drift_threshold=0.15,
                min_history=2,
                stability_bonus=0.01,
            )

        self.fusion = TemporalDecisionFusion(config)
        self.base_threshold = 0.4

    def process(
        self,
        track_id: str,
        helmet_result: Dict[str, Any],
        frame_id: int,
    ) -> Dict[str, Any]:

        status = helmet_result.get("status", "UNCERTAIN")

        if status == "UNCERTAIN":
            return {
                "status": "UNCERTAIN",
                "confidence": helmet_result.get("confidence", 0.0),
                "has_helmet": None,
                "is_violation": False,
                "raw_status": status,
                "temporal_info": None,
            }

        if status == "NO_HELMET":
            raw_score = helmet_result["confidence"]
        elif status == "HELMET":
            raw_score = 1.0 - helmet_result["confidence"]
        else:
            raw_score = 0.5

        raw_confidence = helmet_result["confidence"]

        fusion_result = self.fusion.update(
            track_id=track_id,
            decision_type="helmet",
            raw_score=raw_score,
            confidence=raw_confidence,
            frame_id=frame_id,
            base_threshold=self.base_threshold,
        )

        is_no_helmet = fusion_result["decision"]
        has_helmet = not is_no_helmet

        stability = fusion_result["stability"]

        # Final fused status
        if stability >= 0.4 and fusion_result["consecutive_streak"] >= 2:
            fused_status = "NO_HELMET" if is_no_helmet else "HELMET"
        elif stability < 0.4:
            fused_status = "UNCERTAIN"
        else:
            fused_status = "NO_HELMET" if is_no_helmet else "HELMET"

        # 🔒 SAFER violation condition (added stability requirement)
        is_violation = (
            is_no_helmet
            and fused_status == "NO_HELMET"
            and stability >= 0.55
        )

        return {
            "status": fused_status,
            "confidence": fusion_result["confidence"],
            "has_helmet": has_helmet,
            "is_violation": is_violation,
            "raw_status": status,
            "raw_confidence": raw_confidence,
            "temporal_info": {
                "fused_score": fusion_result["fused_score"],
                "adjusted_threshold": fusion_result["adjusted_threshold"],
                "bias": fusion_result["bias"],
                "stability": stability,
                "consecutive_streak": fusion_result["consecutive_streak"],
            },
        }


# ============================================================
# OCR DECISION MANAGER (unchanged logic)
# ============================================================

class OCRDecisionManager:

    def __init__(self, config: Optional[TemporalFusionConfig] = None):

        if config is None:
            config = TemporalFusionConfig(
                ema_alpha=0.20,
                bias_cap=0.04,
                decay_rate=0.93,
                drift_threshold=0.12,
                min_history=3,
                stability_bonus=0.01,
            )

        self.fusion = TemporalDecisionFusion(config)
        self.base_threshold = 0.45

    def process(
        self,
        track_id: str,
        ocr_text: Optional[str],
        ocr_confidence: float,
        frame_id: int,
        is_valid_format: bool = True,
    ) -> Dict[str, Any]:

        if not ocr_text or not is_valid_format:
            return {
                "text": None,
                "confidence": 0.0,
                "is_reliable": False,
                "temporal_info": None,
            }

        fusion_result = self.fusion.update(
            track_id=track_id,
            decision_type="ocr_confidence",
            raw_score=ocr_confidence,
            confidence=ocr_confidence,
            frame_id=frame_id,
            base_threshold=self.base_threshold,
        )

        is_reliable = fusion_result["decision"]

        return {
            "text": ocr_text if is_reliable else None,
            "confidence": fusion_result["confidence"],
            "is_reliable": is_reliable,
            "raw_confidence": ocr_confidence,
            "temporal_info": {
                "fused_score": fusion_result["fused_score"],
                "adjusted_threshold": fusion_result["adjusted_threshold"],
                "bias": fusion_result["bias"],
                "stability": fusion_result["stability"],
                "consecutive_streak": fusion_result["consecutive_streak"],
            },
        }


# ============================================================
# VIOLATION AGGREGATOR (BUG FIXED)
# ============================================================

class ViolationDecisionAggregator:

    def __init__(self):
        self.track_violations = {}

    def evaluate_violation(
        self,
        track_id: str,
        vehicle_class: str,
        hsrp_decision: Optional[Dict[str, Any]],
        helmet_decision: Optional[Dict[str, Any]],
        ocr_decision: Optional[Dict[str, Any]],
        frame_id: int,
    ) -> Dict[str, Any]:

        if track_id not in self.track_violations:
            self.track_violations[track_id] = {
                "hsrp_violations": [],
                "helmet_violations": [],
                "first_seen_frame": frame_id,
                "last_seen_frame": frame_id,
            }

        track_data = self.track_violations[track_id]
        track_data["last_seen_frame"] = frame_id

        violation_type = None
        violation_confidence = 0.0
        should_store = False

        # ---------------- HSRP ----------------
        if hsrp_decision and hsrp_decision.get("is_violation"):
            temporal_info = hsrp_decision.get("temporal_info", {})
            if temporal_info.get("stability", 0) >= 0.6:
                track_data["hsrp_violations"].append({
                    "frame": frame_id,
                    "confidence": hsrp_decision["confidence"],
                    "stability": temporal_info.get("stability", 0),
                })

        # ---------------- HELMET ----------------
        if (
            vehicle_class == "motorcycle"
            and helmet_decision
            and helmet_decision.get("is_violation")
        ):
            temporal_info = helmet_decision.get("temporal_info", {})
            if temporal_info.get("stability", 0) >= 0.65:
                track_data["helmet_violations"].append({
                    "frame": frame_id,
                    "confidence": helmet_decision["confidence"],
                    "stability": temporal_info.get("stability", 0),
                })

        # ---------------- Final Decision ----------------

        if len(track_data["hsrp_violations"]) >= 3:
            avg_conf = np.mean(
                [v["confidence"] for v in track_data["hsrp_violations"][-5:]]
            )
            avg_stability = np.mean(
                [v["stability"] for v in track_data["hsrp_violations"][-5:]]
            )

            if avg_conf >= 0.7 and avg_stability >= 0.65:
                violation_type = "non_hsrp_plate"
                violation_confidence = avg_conf
                should_store = True

        if len(track_data["helmet_violations"]) >= 3:
            avg_conf = np.mean(
                [v["confidence"] for v in track_data["helmet_violations"][-5:]]
            )
            avg_stability = np.mean(
                [v["stability"] for v in track_data["helmet_violations"][-5:]]
            )

            if avg_conf >= 0.52 and avg_stability >= 0.55:
                violation_type = "no_helmet"
                violation_confidence = avg_conf
                should_store = True

        # ✅ FIXED evidence_frames
        if violation_type == "non_hsrp_plate":
            evidence_count = len(track_data["hsrp_violations"])
        elif violation_type == "no_helmet":
            evidence_count = len(track_data["helmet_violations"])
        else:
            evidence_count = 0

        return {
            "has_violation": should_store,
            "violation_type": violation_type,
            "confidence": round(violation_confidence, 4),
            "evidence_frames": evidence_count,
            "should_store": should_store,
            "track_duration_frames":
                track_data["last_seen_frame"] - track_data["first_seen_frame"],
            "metadata": {
                "hsrp_violations": len(track_data["hsrp_violations"]),
                "helmet_violations": len(track_data["helmet_violations"]),
            },
        }
    
    def cleanup_old_tracks(self, current_frame: int, max_age: int = 150):
        """
        Clean up old violation tracking data.

        Args:
            current_frame: Current frame number
            max_age: Maximum allowed frame gap before removing track
        """

        tracks_to_remove = []

        for track_id, data in self.track_violations.items():
            if current_frame - data["last_seen_frame"] > max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_violations[track_id]

        return len(tracks_to_remove)
