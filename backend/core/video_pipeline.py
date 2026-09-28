"""
GPU-OPTIMIZED VIDEO PIPELINE
==============================
Main entry point for processing a video file end-to-end.

GPU optimisations over original:
  - Frame decode on GPU via cv2.cudacodec (falls back to CPU)
  - BATCHED YOLO inference: vehicle + plate detectors run once per
    batch of N frames instead of once per frame, saturating tensor
    cores and maximising GPU utilisation.
  - Tail-padding in video_reader ensures the final batch is always
    full; padded frame results are discarded before temporal fusion.
  - CUDA streams parallelise HSRP + Helmet inference
  - Async video writer uses bounded queue (no disk I/O stall)
  - Periodic CUDA cache flush prevents memory fragmentation

Temporal fusion, DB gating, violation prediction — preserved exactly
as specified in the paper. Batch results are unpacked and fed to
temporal fusion in strict frame_id order so EMA/bias/decay are
unaffected by batching.

Batch size is controlled by INFERENCE_BATCH_SIZE (default 8).
Set to 1 to fall back to the original frame-by-frame behaviour.
"""

import os
import time
import cv2
import numpy as np
import torch
import subprocess
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
from collections import deque
import traceback
import logging

from backend.utils.local_storage import get_video_url
from backend.services.video_reader import read_video_batched
from backend.core.model_registry import (
    vehicle_detector, plate_detector,
    helmet_detector, hsrp_classifier, ocr_model,
)
from backend.services.ocr_stabilizer import OCRStabilizer
from backend.services.vehicle_tracker import SimpleTracker, VehicleTracker
from backend.core.frame_pipeline import FramePipeline
from backend.core.violation_predictor import ViolationPredictor
from backend.core.adaptive_threshold import AdaptiveThresholdLearner
from backend.core.db_gate import DatabaseGate
from backend.core.video_annotator import annotate_frame, enhance_frame_for_capture, AsyncVideoWriter

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()

# How many frames to batch for YOLO inference at once.
# 8 is optimal for most YOLO configs on a single GPU.
# Lower if you hit OOM; set to 1 to disable batching.
INFERENCE_BATCH_SIZE = 8


# ─────────────────────────────────────────────────────────────────────────────
# BATCH YOLO HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _batch_vehicle_detect(frames: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
    """
    Run vehicle detector on a batch of frames.

    Delegates to VehicleDetector.detect_batch() which handles both PyTorch
    and TensorRT backends transparently (TRT engines need exactly trt_batch
    frames; detect_batch pads/trims as needed).

    Returns a list (one entry per frame) of detection lists.
    """
    return vehicle_detector.detect_batch(frames)


def _batch_plate_detect(frames: List[np.ndarray]) -> List[List[List[int]]]:
    """
    Run plate detector on a batch of frames.

    Delegates to PlateDetector.predict_batch() which handles TRT batch
    size requirements transparently.

    Returns a list (one entry per frame) of plate box lists.
    """
    return plate_detector.predict_batch(frames)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def process_video(
    video_path: str,
    output_video_path: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_skip: int = 1,
    annotate_violations:    bool = True,
    annotate_no_violations: bool = False,
    ocr_mode:               str  = "on_violation",
    progress_callback: Optional[Callable] = None,
    cleanup_interval:  int = 50,
    job_id: Optional[str] = None,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> Dict[str, Any]:
    """
    Full GPU-accelerated processing pipeline for a video file.

    Frames are read and decoded in batches of `batch_size`.  Vehicle and
    plate YOLO models run once per batch (saturating tensor cores), then
    per-frame results are unpacked and fed to the temporal fusion engine
    in strict frame_id order — preserving EMA/bias/decay correctness.

    Returns:
        output dict with violations, track_summaries, metadata, video_url
    """

    # ── Components ────────────────────────────────────────────────────────
    ocr_stabilizer = OCRStabilizer()
    try:
        tracker = VehicleTracker() if USE_CUDA else SimpleTracker()
    except Exception:
        tracker = SimpleTracker()

    pipeline  = FramePipeline()
    predictor = ViolationPredictor()
    learner   = AdaptiveThresholdLearner()
    db_gate   = DatabaseGate()

    live_track_data: Dict[str, Dict[str, Any]] = {}

    output = {
        "video_path":      video_path,
        "frames":          [],
        "violations":      [],
        "track_summaries": {},
        "temporal_stats":  {},
        "metadata": {
            "frame_skip":       frame_skip,
            "batch_size":       batch_size,
            "processing_start": time.time(),
            "gpu":              USE_CUDA,
        },
    }

    frame_count      = 0
    processed_count  = 0
    processing_times: deque = deque(maxlen=100)
    video_writer: Optional[AsyncVideoWriter] = None
    local_url:    Optional[str]              = None
    prediction_cache: Dict[str, Dict]        = {}

    source_fps         = _get_video_fps(video_path)
    total_video_frames = _get_video_frame_count(video_path, frame_skip)
    _cuda_warmed       = False

    try:
        logger.info(
            f"[{job_id}] Starting | gpu={USE_CUDA} | fps={source_fps} | batch={batch_size}"
        )

        # ── Batch loop ────────────────────────────────────────────────────
        # read_video_batched handles frame_skip and tail-padding internally.
        for batch in read_video_batched(video_path, batch_size=batch_size, frame_skip=frame_skip):
            # batch = list of (frame_id, frame, is_pad)

            # GPU warm-up on very first real frame
            if USE_CUDA and not _cuda_warmed:
                _warm_cuda()
                _cuda_warmed = True

            # Separate real vs padded slots
            real_items = [(fid, fr, pad) for fid, fr, pad in batch if not pad]
            frame_count += len(real_items)

            if max_frames and processed_count >= max_frames:
                break

            if not real_items:
                continue

            # ── Batched YOLO inference (L1 + L4) ─────────────────────────
            # Extract just the numpy frames for the YOLO call
            all_frames  = [fr for _, fr, _ in batch]   # includes pads (needed for fixed batch)
            real_frames = [fr for _, fr, _ in real_items]

            t_batch_start = time.time()

            try:
                # Run vehicle + plate detection on the full batch at once
                batch_vehicle_results = _batch_vehicle_detect(all_frames)
                batch_plate_results   = _batch_plate_detect(all_frames)
            except Exception as batch_err:
                logger.error(f"[{job_id}] Batch YOLO error: {batch_err} — retrying via detect_batch")
                # detect_batch handles TRT padding internally; safe for both backends
                try:
                    batch_vehicle_results = vehicle_detector.detect_batch(all_frames)
                    batch_plate_results   = plate_detector.predict_batch(all_frames)
                except Exception as fallback_err:
                    logger.error(f"[{job_id}] detect_batch also failed: {fallback_err} — skipping batch")
                    continue

            # ── Per-frame temporal processing ─────────────────────────────
            # Unpack in strict frame_id order; skip padded slots.
            for slot_idx, (frame_id, frame, is_pad) in enumerate(batch):
                if is_pad:
                    continue  # Discard padding — never feed to temporal fusion

                if max_frames and processed_count >= max_frames:
                    break

                t0 = time.time()

                try:
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
                        # Pass pre-computed YOLO results — L1 and L4 are skipped
                        precomputed_vehicle_detections=batch_vehicle_results[slot_idx],
                        precomputed_plate_boxes=batch_plate_results[slot_idx],
                    )

                    # ── Update live track data ────────────────────────────
                    vehicles   = frame_result.get("entities", {}).get("vehicles", [])
                    plates     = frame_result.get("entities", {}).get("plates",   [])
                    assoc_vp   = frame_result.get("associations", {}).get("vehicle_plate", {})
                    enf_list   = frame_result.get("enforcements", {}).get("two_wheelers",  [])
                    enf_by_vid = {e["vehicle_id"]: e for e in enf_list}

                    for v in vehicles:
                        vid      = v.get("id", "")
                        track_id = v.get("track_id", vid)
                        vclass   = v.get("class", "unknown")
                        pid      = assoc_vp.get(vid)
                        plate    = next((p for p in plates if p["id"] == pid), None) if pid else None
                        enf      = enf_by_vid.get(vid)

                        if track_id not in live_track_data:
                            live_track_data[track_id] = {
                                "track_id":             track_id,
                                "vehicle_class":        vclass,
                                "first_frame":          frame_id,
                                "last_frame":           frame_id,
                                "hsrp_label":           None,
                                "hsrp_confidence":      0.0,
                                "helmet_status":        None,
                                "helmet_confidence":    0.0,
                                "plate_number":         None,
                                "ocr_confidence":       0.0,
                                "violation_type":       None,
                                "violation_confidence": 0.0,
                                "quality_score":        0.0,
                                "should_store":         False,
                                "needs_review":         False,
                                "prediction_preceded":  False,
                            }

                        td = live_track_data[track_id]
                        td["last_frame"]    = frame_id
                        td["vehicle_class"] = vclass

                        if plate:
                            hl, hc = plate.get("hsrp"), plate.get("hsrp_confidence", 0.0)
                            if hl and hc > td["hsrp_confidence"]:
                                td["hsrp_label"], td["hsrp_confidence"] = hl, hc
                            ot, oc = plate.get("ocr_text"), plate.get("ocr_confidence", 0.0)
                            if ot and oc > td["ocr_confidence"]:
                                td["plate_number"], td["ocr_confidence"] = ot, oc

                        if enf and enf.get("helmet"):
                            hs, hconf = enf["helmet"].get("status"), enf["helmet"].get("confidence", 0.0)
                            if hs and hconf > td["helmet_confidence"]:
                                td["helmet_status"], td["helmet_confidence"] = hs, hconf

                    # ── Violation Predictor ───────────────────────────────
                    current_predictions = {}
                    for v in vehicles:
                        track_id  = v.get("track_id", v.get("id"))
                        pid       = assoc_vp.get(v["id"])
                        plate     = next((p for p in plates if p["id"] == pid), None) if pid else None
                        enf       = enf_by_vid.get(v["id"])

                        hsrp_score   = None
                        helmet_score = None

                        if plate:
                            if plate.get("hsrp") == "non_hsrp":
                                hsrp_score = plate.get("hsrp_confidence", 0.0)
                            elif plate.get("hsrp") == "hsrp":
                                hsrp_score = 0.0

                        if enf and enf.get("helmet"):
                            if enf["helmet"].get("is_violation"):
                                helmet_score = enf["helmet"].get("confidence", 0.0)
                            elif enf["helmet"].get("status") == "HELMET":
                                helmet_score = 0.0

                        pred = predictor.update(
                            track_id=track_id, frame_id=frame_id,
                            hsrp_score=hsrp_score, helmet_score=helmet_score,
                        )
                        current_predictions[track_id] = pred

                        if track_id in live_track_data:
                            if pred.get("any_warning") or pred.get("any_confirmed"):
                                live_track_data[track_id]["prediction_preceded"] = True

                        if plate and hsrp_score is not None:
                            learner.observe(
                                "hsrp", hsrp_score,
                                plate.get("hsrp_confidence", 0.5),
                                mode="semi_supervised",
                            )

                    # ── DB Gate feed ──────────────────────────────────────
                    for v in vehicles:
                        track_id  = v.get("track_id", v.get("id"))
                        pid       = assoc_vp.get(v["id"])
                        plate     = next((p for p in plates if p["id"] == pid), None) if pid else None
                        pred_info = current_predictions.get(track_id, {})

                        vtype, vconf, stab = None, 0.0, 0.0

                        if plate and plate.get("hsrp") == "non_hsrp":
                            vtype = "non_hsrp_plate"
                            vconf = plate.get("hsrp_confidence", 0.0)
                            stab  = (plate.get("hsrp_temporal") or {}).get("stability", 0.0)

                        for tw in enf_list:
                            if tw.get("vehicle_id") == v["id"] and tw.get("helmet", {}).get("is_violation"):
                                vtype = "no_helmet"
                                vconf = tw["helmet"].get("confidence", 0.0)
                                stab  = (tw.get("helmet_temporal") or {}).get("stability", 0.0)
                                break

                        if vtype:
                            db_gate.feed(
                                track_id=track_id, frame_id=frame_id,
                                violation_type=vtype, vehicle_class=v.get("class", "unknown"),
                                violation_conf=vconf, stability=stab,
                                temporal_consistency=stab,
                                consecutive_count=int(stab * 10),
                                plate_text=plate.get("ocr_text") if plate else None,
                                plate_conf=plate.get("ocr_confidence", 0.0) if plate else 0.0,
                                prediction_warned=(
                                    pred_info.get("any_warning") or pred_info.get("any_confirmed")
                                ),
                                prediction_risk=max(
                                    pred_info.get("hsrp",   {}).get("risk_score", 0.0),
                                    pred_info.get("helmet", {}).get("risk_score", 0.0),
                                ),
                            )

                    # ── Annotate + Write Frame ────────────────────────────
                    if output_video_path:
                        try:
                            annotated, any_pred = annotate_frame(
                                frame.copy(), frame_result, current_predictions
                            )
                            if any_pred:
                                annotated = enhance_frame_for_capture(annotated)

                            if video_writer is None:
                                h, w = annotated.shape[:2]
                                logger.info(
                                    f"[{job_id}] Writer: {output_video_path} {w}x{h}@{source_fps}fps"
                                )
                                video_writer = AsyncVideoWriter(
                                    output_video_path, fps=source_fps, width=w, height=h
                                )

                            video_writer.write(annotated)
                        except Exception as e:
                            logger.error(f"[{job_id}] Annotation error frame {frame_id}: {e}")

                    processing_times.append(time.time() - t0)
                    frame_result["processing_time_ms"] = round(processing_times[-1] * 1000, 2)
                    frame_result["predictions"]        = current_predictions
                    output["frames"].append(frame_result)
                    processed_count += 1

                    # Periodic cleanup + CUDA cache flush
                    if processed_count % cleanup_interval == 0:
                        pipeline.cleanup(frame_id)
                        predictor.cleanup_old_tracks(frame_id)
                        db_gate.cleanup_old_tracks(frame_id)
                        if USE_CUDA and processed_count % (cleanup_interval * 4) == 0:
                            torch.cuda.empty_cache()

                    if progress_callback and processed_count % 5 == 0:
                        progress_callback(processed_count, max_frames or total_video_frames)

                except Exception as frame_err:
                    logger.error(f"[{job_id}] Frame {frame_id} error: {frame_err}")
                    continue

    except Exception as e:
        logger.error(f"[{job_id}] Critical error: {e}\n{traceback.format_exc()}")
        raise

    finally:
        if video_writer:
            try:
                logger.info(f"[{job_id}] Releasing video writer...")
                video_writer.release()
                logger.info(f"[{job_id}] Video writer released")
            except Exception as e:
                logger.error(f"[{job_id}] Writer release error: {e}")

        # Re-encode to H.264 for browser playback
        if output_video_path and os.path.exists(output_video_path):
            _reencode_for_web(output_video_path)
            local_url = get_video_url(job_id)
            logger.info(f"[{job_id}] Output ready: {output_video_path} → {local_url}")

        # Evaluate all tracks via DB gate
        violations      = []
        track_summaries = {}

        for tid in list(db_gate._tracks.keys()):
            decision = db_gate.evaluate_track(tid)
            if decision["should_store"]:
                violations.append(decision["record"])
            if tid in live_track_data:
                live_track_data[tid].update({
                    "violation_type":       db_gate._tracks[tid].violation_type,
                    "violation_confidence": decision.get("quality_score", 0.0),
                    "quality_score":        decision.get("quality_score", 0.0),
                    "should_store":         decision["should_store"],
                    "needs_review":         decision.get("needs_manual_review", False),
                    "plate_number": (
                        live_track_data[tid].get("plate_number")
                        or _best_plate(db_gate._tracks[tid])
                    ),
                })

        for tid, td in live_track_data.items():
            track_summaries[tid] = td

        output["violations"]      = violations
        output["track_summaries"] = track_summaries
        output["all_tracks"]      = track_summaries
        output["video_url"]       = local_url

        output["metadata"].update({
            "total_frames_read":      frame_count,
            "total_frames_processed": processed_count,
            "processing_end":         time.time(),
            "total_time_seconds":     round(
                time.time() - output["metadata"]["processing_start"], 2
            ),
        })

        if processing_times:
            avg = sum(processing_times) / len(processing_times)
            output["metadata"]["avg_frame_time_ms"] = round(avg * 1000, 2)
            output["metadata"]["avg_fps"]           = round(1.0 / avg if avg > 0 else 0, 2)

        output["temporal_stats"] = pipeline.get_statistics()

        if learner:
            output["adaptive_thresholds"] = learner.get_all_thresholds()
            learner.save_progress()

        if USE_CUDA:
            torch.cuda.empty_cache()

        logger.info(f"[{job_id}] Done: {processed_count} frames | {len(violations)} violations")

    return output


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def generate_violation_summary(video_output: Dict[str, Any]) -> Dict[str, Any]:
    violations = video_output.get("violations", [])
    by_type: Dict[str, list] = {}
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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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


def _warm_cuda():
    """Run a tiny GPU operation to initialise CUDA kernels."""
    try:
        dummy = torch.zeros(1, 3, 64, 64, device="cuda", dtype=torch.float16)
        _ = dummy + 1
        del dummy
        torch.cuda.synchronize()
        logger.info("[GPU] CUDA warmed up")
    except Exception as e:
        logger.warning(f"[GPU] Warm-up failed (non-fatal): {e}")


def _reencode_for_web(path: str):
    """
    Re-encode OpenCV mp4v → H.264 + faststart for browser inline playback.
    Falls back silently if ffmpeg not installed.
    """
    tmp = path.replace(".mp4", "_tmp.mp4")
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-movflags", "+faststart", "-an", tmp],
            capture_output=True, timeout=600,
        )
        if result.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, path)
            logger.info(f"Re-encoded to H.264: {path}")
        else:
            logger.warning(f"ffmpeg failed (rc={result.returncode})")
            if os.path.exists(tmp):
                os.remove(tmp)
    except FileNotFoundError:
        logger.warning("ffmpeg not found — install it for browser playback")
    except Exception as e:
        logger.warning(f"Re-encode error: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)


def _get_video_frame_count(path: str, frame_skip: int = 1) -> int:
    """
    Return the number of frames that will actually be processed after
    applying frame_skip.  Used so progress_callback reports a real total
    instead of always falling back to '?'.

    CAP_PROP_FRAME_COUNT is a fast metadata read — no frame decode.
    Returns 0 if the container does not expose this property.
    """
    cap   = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total <= 0:
        return 0
    skip = max(1, frame_skip)
    return max(1, (total + skip - 1) // skip)
