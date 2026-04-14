"""
ENHANCED VIDEO PIPELINE v2 - FIXED
===================================
Integrates:
- FramePipeline (temporal fusion)
- ViolationPredictor (early warning)
- AdaptiveThresholdLearner (continuous learning)
- DatabaseGate (smart storage decisions)
- VideoAnnotator (3-state visualization)
- AsyncVideoWriter (FPS-preserving output)

FIXES:
- Better error handling and logging
- Proper video writer cleanup
- Fixed frame copy issue
- Enhanced exception reporting
"""

import time
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from collections import deque
import traceback
import logging

from backend.services.video_reader import read_video
from backend.core.model_registry import (
    vehicle_detector, plate_detector,
    helmet_detector, hsrp_classifier, ocr_model,
)
from backend.services.ocr_stabilizer import OCRStabilizer
from backend.services.vehicle_tracker import SimpleTracker
from backend.core.frame_pipeline import FramePipeline
from backend.core.violation_predictor import ViolationPredictor, ViolationState
from backend.core.adaptive_threshold import AdaptiveThresholdLearner, LearningConfig
from backend.core.db_gate import DatabaseGate, GatingConfig
from backend.core.video_annotator import (
    annotate_frame,
    enhance_frame_for_capture,
    AsyncVideoWriter,
)

# Setup logging
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def process_video(
    video_path: str,
    output_video_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    target_fps: int = 12,
    enable_tracking: bool = True,
    enable_ocr_stabilization: bool = True,
    enable_temporal_fusion: bool = True,
    enable_prediction: bool = True,
    enable_adaptive_thresholds: bool = True,
    enable_db_gating: bool = True,
    progress_callback: Optional[Callable] = None,
    cleanup_interval: int = 50,
    # ── NEW: annotation + OCR mode params ────
    annotate_violations:    bool = True,
    annotate_no_violations: bool = False,
    ocr_mode:               str  = "always",   # "always"|"on_violation"|"on_clean"|"off"
) -> Dict[str, Any]:
    """
    Full processing pipeline for a video file.

    Returns output dict with:
    - frames: per-frame results
    - violations: aggregated violations (from DB gate)
    - track_summaries: per track_id summary for ALL detected vehicles (not just violators)
    - all_tracks: alias for track_summaries (full set)
    - temporal_stats: fusion engine stats
    - metadata: processing metrics
    """

    # ── Components ───────────────────────────────
    ocr_stabilizer = OCRStabilizer() if enable_ocr_stabilization else None
    tracker        = SimpleTracker() if enable_tracking else None
    pipeline       = FramePipeline() if enable_temporal_fusion else None
    predictor      = ViolationPredictor() if enable_prediction else None
    learner        = AdaptiveThresholdLearner() if enable_adaptive_thresholds else None
    db_gate        = DatabaseGate() if enable_db_gating else None

    # ── Per-track live state (ALL vehicles, not just violators) ──────────
    # Keyed by track_id. We collect the latest HSRP and helmet decision
    # for every track so we can build a complete summary at the end.
    live_track_data: Dict[str, Dict[str, Any]] = {}

    # ── Output containers ─────────────────────────
    output = {
        "video_path":      video_path,
        "frames":          [],
        "violations":      [],
        "track_summaries": {},
        "temporal_stats":  {},
        "metadata": {
            "frame_skip":       frame_skip,
            "processing_start": time.time(),
        },
    }

    frame_count     = 0
    processed_count = 0
    processing_times: deque = deque(maxlen=100)

    # ── Video properties for output writer ───────
    video_writer: Optional[AsyncVideoWriter] = None
    source_fps = _get_video_fps(video_path)

    # Map for quick frame lookup when writing annotated video
    prediction_cache: Dict[str, Dict] = {}   # track_id → latest predictor output

    try:
        logger.info(f"Starting video processing: {video_path}")
        logger.info(f"Output video path: {output_video_path}")
        
        for frame_id, frame in read_video(video_path):
            frame_count += 1

            # Frame skipping
            if frame_skip > 1 and frame_id % frame_skip != 0:
                continue
            if max_frames and processed_count >= max_frames:
                break

            t0 = time.time()

            try:
                # ── Run frame pipeline ────────────────
                if pipeline:
                    frame_result = pipeline.process_frame(
                        frame=frame,
                        frame_id=frame_id,
                        vehicle_detector=vehicle_detector,
                        helmet_detector=helmet_detector,
                        plate_detector=plate_detector,
                        hsrp_classifier=hsrp_classifier,
                        ocr_model=ocr_model,
                        ocr_stabilizer=ocr_stabilizer,
                        tracker=tracker,
                        skip_empty_frames=True,
                        frame_skip=frame_skip,
                        annotate_violations=annotate_violations,
                        annotate_no_violations=annotate_no_violations,
                        ocr_mode=ocr_mode,
                    )
                else:
                    frame_result = {
                        "frame_id": frame_id,
                        "entities": {},
                        "enforcements": {},
                        "associations": {},
                        "violations": [],
                    }

                # ── Update live_track_data for ALL vehicles ───────────────
                # This is what makes clean vehicles show up in the final table.
                vehicles = frame_result.get("entities", {}).get("vehicles", [])
                plates   = frame_result.get("entities", {}).get("plates",   [])
                assoc_vp = frame_result.get("associations", {}).get("vehicle_plate", {})
                enforcements_list = frame_result.get("enforcements", {}).get("two_wheelers", [])
                enf_by_vid = {e["vehicle_id"]: e for e in enforcements_list}

                for v in vehicles:
                    vid      = v.get("id", "")
                    track_id = v.get("track_id", vid)
                    vclass   = v.get("class") or v.get("vehicle_class") or "unknown"

                    pid   = assoc_vp.get(vid)
                    plate = next((p for p in plates if p["id"] == pid), None) if pid else None
                    enf   = enf_by_vid.get(vid)

                    # Initialise entry on first sight
                    if track_id not in live_track_data:
                        live_track_data[track_id] = {
                            "track_id":       track_id,
                            "vehicle_class":  vclass,
                            "first_frame":    frame_id,
                            "last_frame":     frame_id,
                            # HSRP
                            "hsrp_label":     None,
                            "hsrp_confidence": 0.0,
                            # Helmet
                            "helmet_status":  None,
                            "helmet_confidence": 0.0,
                            # OCR
                            "plate_number":   None,
                            "ocr_confidence": 0.0,
                            # Violation (filled later from db_gate)
                            "violation_type": None,
                            "violation_confidence": 0.0,
                            "quality_score":  0.0,
                            "should_store":   False,
                            "needs_review":   False,
                            "prediction_preceded": False,
                        }

                    td = live_track_data[track_id]
                    td["last_frame"]   = frame_id
                    td["vehicle_class"] = vclass  # keep updated

                    # Update HSRP
                    if plate:
                        hsrp_label = plate.get("hsrp")
                        hsrp_conf  = plate.get("hsrp_confidence", 0.0)
                        if hsrp_label and hsrp_conf > td.get("hsrp_confidence", 0.0):
                            td["hsrp_label"]      = hsrp_label
                            td["hsrp_confidence"] = hsrp_conf

                        ocr_text = plate.get("ocr_text")
                        ocr_conf = plate.get("ocr_confidence", 0.0)
                        if ocr_text and ocr_conf > td.get("ocr_confidence", 0.0):
                            td["plate_number"]   = ocr_text
                            td["ocr_confidence"] = ocr_conf

                    # Update Helmet (only for two-wheelers)
                    if enf and enf.get("helmet"):
                        h = enf["helmet"]
                        h_status = h.get("status")
                        h_conf   = h.get("confidence", 0.0)
                        if h_status and h_conf > td.get("helmet_confidence", 0.0):
                            td["helmet_status"]     = h_status
                            td["helmet_confidence"] = h_conf

                # ── Violation predictor ────────────────
                current_predictions = {}
                if predictor:
                    for v in vehicles:
                        track_id = v.get("track_id", v.get("id"))
                        pid      = assoc_vp.get(v["id"])
                        plate    = next((p for p in plates if p["id"] == pid), None) if pid else None
                        enf      = enf_by_vid.get(v["id"])

                        # Compute violation scores (0-1 probability)
                        hsrp_score  = None
                        if plate:
                            # If non_hsrp detected, score = confidence, else 0
                            if plate.get("hsrp") == "non_hsrp":
                                hsrp_score = plate.get("hsrp_confidence", 0.0)
                            elif plate.get("hsrp") == "hsrp":
                                hsrp_score = 0.0  # Compliant = no violation

                        helmet_score = None
                        if enf and enf.get("helmet"):
                            # If violation detected, score = confidence, else 0
                            if enf["helmet"].get("is_violation"):
                                helmet_score = enf["helmet"].get("confidence", 0.0)
                            elif enf["helmet"].get("status") == "HELMET":
                                helmet_score = 0.0  # Has helmet = no violation

                        pred_result = predictor.update(
                            track_id=track_id,
                            frame_id=frame_id,
                            hsrp_score=hsrp_score,
                            helmet_score=helmet_score,
                        )
                        current_predictions[track_id] = pred_result

                        # Update prediction_preceded in live_track_data
                        if track_id in live_track_data:
                            if pred_result.get("any_warning") or pred_result.get("any_confirmed"):
                                live_track_data[track_id]["prediction_preceded"] = True

                        # Adaptive threshold feedback
                        if learner and plate and hsrp_score is not None:
                            learner.observe(
                                decision_type="hsrp",
                                score=hsrp_score,
                                confidence=plate.get("hsrp_confidence", 0.5),
                                mode="semi_supervised",
                            )

                # ── DB Gate feed (violation tracks only) ──────────────────
                if db_gate:
                    for v in vehicles:
                        track_id = v.get("track_id", v.get("id"))
                        pid      = assoc_vp.get(v["id"])
                        plate    = next((p for p in plates if p["id"] == pid), None) if pid else None

                        pred_info = current_predictions.get(track_id, {})
                        any_warn  = pred_info.get("any_warning", False)
                        any_conf  = pred_info.get("any_confirmed", False)

                        vtype = None
                        vconf = 0.0
                        stab  = 0.0

                        if plate and plate.get("hsrp") == "non_hsrp":
                            vtype = "non_hsrp_plate"
                            vconf = plate.get("hsrp_confidence", 0.0)
                            stab  = (plate.get("hsrp_temporal") or {}).get("stability", 0.0)

                        for tw in enforcements_list:
                            if tw.get("vehicle_id") == v["id"]:
                                h = tw.get("helmet") or {}
                                if h.get("is_violation"):
                                    vtype = "no_helmet"
                                    vconf = h.get("confidence", 0.0)
                                    stab  = (tw.get("helmet_temporal") or {}).get("stability", 0.0)
                                break

                        if vtype:
                            db_gate.feed(
                                track_id=track_id,
                                frame_id=frame_id,
                                violation_type=vtype,
                                vehicle_class=v.get("class", "unknown"),
                                violation_conf=vconf,
                                stability=stab,
                                temporal_consistency=stab,
                                consecutive_count=int(stab * 10),
                                plate_text=plate.get("ocr_text") if plate else None,
                                plate_conf=plate.get("ocr_confidence", 0.0) if plate else 0.0,
                                prediction_warned=(any_warn or any_conf),
                                prediction_risk=max(
                                    (pred_info.get("hsrp",   {}).get("risk_score", 0.0)),
                                    (pred_info.get("helmet", {}).get("risk_score", 0.0)),
                                ),
                            )

                # ── Annotate frame + write to video ───
                any_prediction_active = False
                if output_video_path:
                    try:
                        # CRITICAL FIX: Make a copy of the frame BEFORE annotation
                        frame_copy = frame.copy()
                        annotated, any_prediction_active = annotate_frame(
                            frame_copy, frame_result, current_predictions
                        )

                        if any_prediction_active:
                            annotated = enhance_frame_for_capture(annotated)

                        if video_writer is None:
                            h, w = annotated.shape[:2]
                            logger.info(f"Initializing video writer: {output_video_path}, {w}x{h}, {source_fps}fps")
                            video_writer = AsyncVideoWriter(
                                output_video_path,
                                fps=source_fps,
                                width=w,
                                height=h,
                            )
                        
                        video_writer.write(annotated)
                    except Exception as e:
                        logger.error(f"Error annotating/writing frame {frame_id}: {e}")
                        logger.error(traceback.format_exc())
                        # Continue processing even if annotation fails

                processing_times.append(time.time() - t0)
                frame_result["processing_time_ms"] = round(processing_times[-1] * 1000, 2)
                frame_result["predictions"] = current_predictions
                output["frames"].append(frame_result)
                processed_count += 1

                # Periodic cleanup
                if processed_count % cleanup_interval == 0:
                    if pipeline:
                        pipeline.cleanup(frame_id)
                    if predictor:
                        predictor.cleanup_old_tracks(frame_id)
                    if db_gate:
                        db_gate.cleanup_old_tracks(frame_id)

                if progress_callback and processed_count % 5 == 0:
                    safe_total = max_frames if isinstance(max_frames, int) else 0
                    progress_callback(processed_count, safe_total)
                    
            except Exception as frame_error:
                logger.error(f"Error processing frame {frame_id}: {frame_error}")
                logger.error(traceback.format_exc())
                # Continue with next frame instead of crashing
                continue

    except Exception as e:
        logger.error(f"Critical error in video processing: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # CRITICAL: Properly release video writer
        if video_writer:
            try:
                logger.info("Releasing video writer...")
                video_writer.release()
                logger.info("Video writer released successfully")
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")

        # ── Evaluate all tracks for storage ───────────────────────────
        violations      = []
        track_summaries = {}

        if db_gate:
            for tid in list(db_gate._tracks.keys()):
                decision = db_gate.evaluate_track(tid)
                if decision["should_store"]:
                    violations.append(decision["record"])

                # Overlay db_gate quality info onto live_track_data
                if tid in live_track_data:
                    live_track_data[tid].update({
                        "violation_type":       db_gate._tracks[tid].violation_type,
                        "violation_confidence": decision.get("quality_score", 0.0),
                        "quality_score":        decision.get("quality_score", 0.0),
                        "should_store":         decision["should_store"],
                        "needs_review":         decision.get("needs_manual_review", False),
                        "plate_number":         (
                            live_track_data[tid].get("plate_number")
                            or _best_plate(db_gate._tracks[tid])
                        ),
                    })

        # ── Build final track_summaries from live_track_data ──────────
        # This includes ALL detected vehicles — violating AND clean.
        for tid, td in live_track_data.items():
            track_summaries[tid] = td

        output["violations"]      = violations
        output["track_summaries"] = track_summaries
        output["all_tracks"]      = track_summaries   # alias for routes.py merge

        # ── Final metadata ────────────────────────
        output["metadata"]["total_frames_read"]      = frame_count
        output["metadata"]["total_frames_processed"] = processed_count
        output["metadata"]["processing_end"]         = time.time()
        output["metadata"]["total_time_seconds"]     = round(
            output["metadata"]["processing_end"] - output["metadata"]["processing_start"], 2
        )
        if processing_times:
            avg = sum(processing_times) / len(processing_times)
            output["metadata"]["avg_frame_time_ms"] = round(avg * 1000, 2)
            output["metadata"]["avg_fps"]           = round(1.0 / avg if avg > 0 else 0, 2)

        if pipeline:
            output["temporal_stats"] = pipeline.get_statistics()

        if learner:
            output["adaptive_thresholds"] = learner.get_all_thresholds()
            learner.save_progress()
            
        logger.info(f"Processing complete: {processed_count} frames, {len(violations)} violations")

    return output


def generate_violation_summary(video_output: Dict[str, Any]) -> Dict[str, Any]:
    """Summary statistics for frontend display."""
    violations = video_output.get("violations", [])
    by_type = {}
    for v in violations:
        vt = v.get("violation_type", "unknown")
        by_type.setdefault(vt, []).append(v.get("violation_confidence", 0.0))

    return {
        "total":        len(violations),
        "by_type":      {
            vt: {"count": len(confs), "avg_conf": round(float(np.mean(confs)), 4)}
            for vt, confs in by_type.items()
        },
        "needs_review": sum(1 for v in violations if v.get("needs_manual_review")),
        "auto_store":   sum(1 for v in violations if not v.get("needs_manual_review")),
    }


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _get_video_fps(path: str) -> float:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 25.0


def _best_plate(acc) -> Optional[str]:
    if not acc.plate_texts:
        return None
    from collections import Counter
    return Counter(acc.plate_texts).most_common(1)[0][0]
