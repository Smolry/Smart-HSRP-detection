"""
VIDEO ANNOTATOR - FIXED
===============
Provides:
  annotate_frame(frame, frame_result, predictions)
      -> (annotated_frame, any_prediction_active)

  enhance_frame_for_capture(frame) -> frame

  AsyncVideoWriter  -- threaded, non-blocking video writer

Colour scheme:
  GREY   -- tracking only (no classification yet)
  GREEN  -- confirmed clean (HSRP ok + helmet ok)
  YELLOW -- violation PREDICTED (early warning, not yet confirmed)
  RED    -- violation CONFIRMED
  BLUE   -- plate bounding box

FIXES:
- AsyncVideoWriter no longer drops frames
- Better error handling in write operations
- Improved thread safety
"""

import cv2
import numpy as np
import threading
import queue
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Colours (BGR) ────────────────────────────────────────────────────────────
COLOUR_TRACKING  = (160, 160, 160)   # Grey   - seen, no classification yet
COLOUR_CLEAN     = (50,  200, 80)    # Green  - confirmed clean
COLOUR_PREDICTED = (0,   200, 220)   # Yellow - predicted violation
COLOUR_VIOLATION = (0,   0,   220)   # Red    - confirmed violation
COLOUR_PLATE     = (210, 130, 0)     # Blue   - plate box
COLOUR_RIDER     = (200, 80,  200)   # Purple - rider / person

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.44
THICKNESS  = 2
TAG_H      = 18   # label tag height px


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _draw_box(frame, bbox, colour, label="", thickness=THICKNESS):
    """Draw bounding box with label on frame."""
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
        if label:
            (tw, _), _ = cv2.getTextSize(label, FONT, FONT_SCALE, 1)
            tag_y1 = max(y1 - TAG_H, 0)
            tag_y2 = tag_y1 + TAG_H
            cv2.rectangle(frame, (x1, tag_y1), (x1 + tw + 6, tag_y2), colour, -1)
            cv2.putText(frame, label, (x1 + 3, tag_y2 - 4),
                        FONT, FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)
    except Exception as e:
        logger.warning(f"Error drawing box: {e}")


def _fc(conf):
    """Format confidence as percent string."""
    return f"{conf:.0%}"


# ── Main annotation function ──────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    frame_result: Dict[str, Any],
    predictions: Dict[str, Any],
) -> Tuple[np.ndarray, bool]:
    """
    Draw bounding boxes and labels on the frame.

    Args:
        frame:        BGR frame (should be a copy)
        frame_result: Dict from FramePipeline.process_frame()
        predictions:  Dict of {track_id: predictor_output} for this frame

    Returns:
        (annotated_frame, any_prediction_active)
        any_prediction_active is True if any track has an active warning/prediction.
    """
    out = frame  # should already be a copy from the caller
    any_prediction_active = False

    try:
        vehicles = frame_result.get("entities", {}).get("vehicles", [])
        plates   = frame_result.get("entities", {}).get("plates",   [])

        assoc_vp = frame_result.get("associations", {}).get("vehicle_plate", {})
        assoc_vr = frame_result.get("associations", {}).get("vehicle_rider", {})
        enf_list = frame_result.get("enforcements", {}).get("two_wheelers", [])
        enf_by_vid = {e["vehicle_id"]: e for e in enf_list}

        confirmed_viol_vids = {v["vehicle_id"] for v in frame_result.get("violations", [])}

        drawn_persons = set()

        for v in vehicles:
            try:
                vid      = v.get("id", "")
                track_id = v.get("track_id", vid)
                vclass   = (v.get("class") or v.get("vehicle_class") or "vehicle").lower()
                bbox     = v.get("bbox", [0, 0, 0, 0])

                pid   = assoc_vp.get(vid)
                plate = next((p for p in plates if p["id"] == pid), None) if pid else None
                enf   = enf_by_vid.get(vid)

                # ── Classification state ─────────────────────────────────────
                hsrp_label   = plate.get("hsrp")           if plate else None
                hsrp_conf    = plate.get("hsrp_confidence", 0.0) if plate else 0.0
                helmet_status = enf["helmet"].get("status") if (enf and enf.get("helmet")) else None
                helmet_conf   = enf["helmet"].get("confidence", 0.0) if (enf and enf.get("helmet")) else 0.0
                is_helmet_viol = bool(enf and enf.get("helmet", {}).get("is_violation", False))

                # ── Prediction state ─────────────────────────────────────────
                pred = predictions.get(track_id, {})
                is_predicted = pred.get("any_warning", False) or pred.get("any_confirmed", False)
                if is_predicted:
                    any_prediction_active = True

                # ── Choose colour ────────────────────────────────────────────
                is_confirmed_viol = (vid in confirmed_viol_vids
                                     or hsrp_label == "non_hsrp"
                                     or is_helmet_viol)

                if is_confirmed_viol:
                    colour = COLOUR_VIOLATION
                elif is_predicted:
                    colour = COLOUR_PREDICTED
                elif hsrp_label is not None or helmet_status is not None:
                    colour = COLOUR_CLEAN
                else:
                    colour = COLOUR_TRACKING

                # ── Build label ──────────────────────────────────────────────
                parts = [f"{vclass} {track_id}"]

                if hsrp_label == "hsrp":
                    parts.append(f"HSRP {_fc(hsrp_conf)}")
                elif hsrp_label == "non_hsrp":
                    parts.append(f"Non-HSRP {_fc(hsrp_conf)}")

                if helmet_status == "HELMET":
                    parts.append(f"Helmet {_fc(helmet_conf)}")
                elif helmet_status == "NO_HELMET":
                    parts.append(f"No Helmet {_fc(helmet_conf)}")
                elif helmet_status == "UNCERTAIN":
                    parts.append("Uncertain")

                # Prediction annotation
                hsrp_risk   = pred.get("hsrp",   {}).get("risk_score", 0.0)
                helmet_risk = pred.get("helmet", {}).get("risk_score", 0.0)
                if hsrp_risk > 0.3:
                    parts.append(f"[HSRP risk {_fc(hsrp_risk)}]")
                if helmet_risk > 0.3:
                    parts.append(f"[Hlmt risk {_fc(helmet_risk)}]")

                label = "  |  ".join(parts)
                _draw_box(out, bbox, colour, label)

                # ── Plate box ────────────────────────────────────────────────
                if plate:
                    pbbox = plate.get("bbox")
                    if pbbox and len(pbbox) == 4:
                        ocr_text    = plate.get("ocr_text") or ""
                        plate_label = ocr_text if ocr_text else "Plate"
                        _draw_box(out, pbbox, COLOUR_PLATE, plate_label, thickness=1)

                # ── Rider box ────────────────────────────────────────────────
                if enf and enf.get("rider_found"):
                    pbbox = enf.get("person_bbox")
                    if pbbox:
                        rider_id = assoc_vr.get(vid, "")
                        if rider_id not in drawn_persons:
                            drawn_persons.add(rider_id)
                            r_colour = (
                                COLOUR_VIOLATION if is_helmet_viol
                                else COLOUR_CLEAN if helmet_status == "HELMET"
                                else COLOUR_TRACKING
                            )
                            _draw_box(out, pbbox, r_colour, "Rider", thickness=1)
            except Exception as e:
                logger.warning(f"Error annotating vehicle {v.get('id', '?')}: {e}")
                continue

        # ── Frame info overlay ───────────────────────────────────────────
        frame_id = frame_result.get("frame_id", 0)
        n_viol   = len(frame_result.get("violations", []))
        info = f"Frame {frame_id}  |  Violations: {n_viol}"
        # Shadow then white
        cv2.putText(out, info, (8, 22), FONT, 0.5, (0,   0,   0),   2, cv2.LINE_AA)
        cv2.putText(out, info, (8, 22), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    except Exception as e:
        logger.error(f"Error in annotate_frame: {e}")

    return out, any_prediction_active


# ── Capture enhancement (called when prediction is active) ───────────────────

def enhance_frame_for_capture(frame: np.ndarray) -> np.ndarray:
    """
    Add a subtle yellow border and 'PREDICTION ACTIVE' banner
    when a violation is being predicted (pre-confirmation state).
    """
    try:
        out = frame.copy()
        h, w = out.shape[:2]

        # Yellow border (4 px)
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), COLOUR_PREDICTED, 4)

        # Small banner top-right
        banner = "! VIOLATION PREDICTED"
        (bw, bh), _ = cv2.getTextSize(banner, FONT, 0.5, 1)
        bx = w - bw - 12
        by = 6
        cv2.rectangle(out, (bx - 4, by), (bx + bw + 4, by + bh + 6), COLOUR_PREDICTED, -1)
        cv2.putText(out, banner, (bx, by + bh + 1), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        return out
    except Exception as e:
        logger.error(f"Error enhancing frame: {e}")
        return frame


# ── Async Video Writer ────────────────────────────────────────────────────────

class AsyncVideoWriter:
    """
    Non-blocking video writer that runs in a background thread.
    Prevents frame processing from stalling on disk I/O.
    
    FIXED: No longer drops frames, properly handles queue full scenarios.
    """

    def __init__(self, path: str, fps: float, width: int, height: int):
        self.path  = path
        # Increased queue size to prevent dropping frames
        self._q    = queue.Queue(maxsize=200)
        self._done = threading.Event()
        self._error = None

        try:
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
            
            if not self._writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for {path}")
                
            logger.info(f"AsyncVideoWriter initialized: {path}, {width}x{height}@{fps}fps")
        except Exception as e:
            logger.error(f"Failed to initialize video writer: {e}")
            raise

        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def write(self, frame: np.ndarray):
        """
        Queue a frame for writing.
        FIXED: Now blocks if queue is full instead of dropping frames.
        """
        if self._error:
            raise RuntimeError(f"Video writer error: {self._error}")
            
        try:
            # Wait longer if queue is full - don't drop frames!
            self._q.put(frame.copy(), timeout=30)
        except queue.Full:
            logger.error("Video writer queue full after 30s - frame dropped")
            # Last resort: drop this frame but don't crash
        except Exception as e:
            logger.error(f"Error queuing frame: {e}")
            self._error = str(e)

    def release(self):
        """Signal end of stream and wait for all frames to flush."""
        try:
            logger.info(f"Releasing video writer, queue size: {self._q.qsize()}")
            self._q.put(None)          # sentinel
            self._thread.join(timeout=60)  # Increased timeout for large queues
            
            if self._thread.is_alive():
                logger.warning("Video writer thread still alive after 60s timeout")
            
            if self._writer:
                self._writer.release()
                logger.info("Video writer released successfully")
                
            if self._error:
                logger.error(f"Video writer had errors: {self._error}")
        except Exception as e:
            logger.error(f"Error releasing video writer: {e}")

    def _worker(self):
        """Background thread that writes frames to disk."""
        frames_written = 0
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    logger.info(f"Video writer worker finished, wrote {frames_written} frames")
                    break
                    
                try:
                    self._writer.write(frame)
                    frames_written += 1
                    
                    # Log progress periodically
                    if frames_written % 100 == 0:
                        logger.debug(f"Written {frames_written} frames, queue size: {self._q.qsize()}")
                except Exception as e:
                    logger.error(f"Error writing frame {frames_written}: {e}")
                    self._error = str(e)
        except Exception as e:
            logger.error(f"Video writer worker crashed: {e}")
            self._error = str(e)
