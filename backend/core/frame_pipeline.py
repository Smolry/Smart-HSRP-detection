"""
GPU-OPTIMIZED FRAME PIPELINE
==============================
Seven-layer pipeline as described in the paper:
  L1: Vehicle Detection (YOLOv11m GPU)
  L2: Entity Splitting (persons / vehicles)
  L3: Object Tracking (DeepSORT GPU embedder)
  L4: Plate Detection (YOLOv10s GPU)
  L5: HSRP Classification (EfficientNet-B0 GPU FP16)
  L6: OCR Extraction (EasyOCR CPU — conditional on non-HSRP)
  L7: Helmet Detection (YOLOv8s GPU)

GPU optimisations:
  - torch.cuda.Stream overlap: HSRP + Helmet run in parallel CUDA streams
  - All YOLO/TRT models use predict_batch() — never single-image calls,
    so TRT engines with fixed max_batch_size never receive batch=1
  - Helmet crops from all two-wheelers in a frame are batched into one
    predict_batch() call per frame
  - Normalisation tensors pinned on GPU

Temporal fusion engine is preserved EXACTLY as in the paper.
"""

import torch
import numpy as np
from typing import Optional, List, Dict, Any

from backend.core import rules
from backend.services import cropper
from backend.services.decision_managers import (
    HSRPDecisionManager,
    HelmetDecisionManager,
    OCRDecisionManager,
    ViolationDecisionAggregator,
)

TWO_WHEELERS = {"motorcycle"}
USE_CUDA     = torch.cuda.is_available()


class FramePipeline:
    def __init__(self):
        self.hsrp_manager         = HSRPDecisionManager()
        self.helmet_manager       = HelmetDecisionManager()
        self.ocr_manager          = OCRDecisionManager()
        self.violation_aggregator = ViolationDecisionAggregator()
        self.frames_processed     = 0
        self.violations_detected  = 0

        if USE_CUDA:
            self._stream_hsrp   = torch.cuda.Stream()
            self._stream_helmet = torch.cuda.Stream()

    def process_frame(
        self,
        frame,
        frame_id: int,
        vehicle_detector,
        helmet_detector,
        plate_detector,
        hsrp_classifier,
        ocr_model,
        ocr_stabilizer,
        tracker,
        skip_empty_frames:       bool = True,
        frame_skip:              int  = 1,
        annotate_violations:     bool = True,
        annotate_no_violations:  bool = False,
        ocr_mode:                str  = "on_violation",
        precomputed_vehicle_detections: Optional[List[Dict[str, Any]]] = None,
        precomputed_plate_boxes:        Optional[List[List[int]]]      = None,
    ):
        self.frames_processed += 1

        result = {
            "frame_id":     frame_id,
            "entities":     {"vehicles": [], "persons": [], "plates": []},
            "associations": {"vehicle_plate": {}, "vehicle_rider": {}},
            "enforcements": {"two_wheelers": []},
            "violations":   [],
            "temporal_stats": {},
        }

        # ── L1: Vehicle Detection ────────────────────────────────────────
        if precomputed_vehicle_detections is not None:
            detections = precomputed_vehicle_detections
        else:
            # Use detect_batch([frame]) — never single-frame detect()
            # so TRT engines always get the right batch size
            detections = vehicle_detector.detect_batch([frame])[0]

        if not detections:
            return result

        # ── L2: Entity Splitting ─────────────────────────────────────────
        persons, vehicles = [], []
        for idx, det in enumerate(detections):
            if det["class"] == "person":
                det["id"] = f"p{idx}"
                persons.append(det)
            else:
                det["id"] = f"v{idx}"
                vehicles.append(det)

        if not vehicles:
            result["entities"]["persons"] = persons
            return result

        # ── L3: Object Tracking ──────────────────────────────────────────
        if hasattr(tracker, "tracker"):
            vehicles = tracker.update(vehicles, frame)
        else:
            vehicles = tracker.update(vehicles)

        # ── L4: Plate Detection ──────────────────────────────────────────
        if precomputed_plate_boxes is not None:
            raw_plate_boxes = precomputed_plate_boxes
        else:
            raw_plate_boxes = plate_detector.predict_batch([frame])[0]

        if not raw_plate_boxes:
            result["entities"]["vehicles"] = vehicles
            result["entities"]["persons"]  = persons
            return result

        plate_crops = cropper.crop_plates(frame, raw_plate_boxes)
        plates = []
        for idx, (box, crop) in enumerate(zip(raw_plate_boxes, plate_crops)):
            if not box or len(box) != 4:
                continue
            plates.append({
                "id":              f"pl{idx}",
                "bbox":            [int(x) for x in box],
                "confidence":      1.0,
                "hsrp":            None,
                "hsrp_confidence": 0.0,
                "hsrp_temporal":   None,
                "ocr_text":        None,
                "ocr_confidence":  0.0,
                "ocr_temporal":    None,
                "_crop":           crop,
            })

        result["entities"]["vehicles"] = vehicles
        result["entities"]["persons"]  = persons
        result["entities"]["plates"]   = plates

        # ── Association: vehicle ↔ plate ─────────────────────────────────
        available_plates = plates.copy()
        for v in vehicles:
            p = rules.associate_plate(v["bbox"], available_plates)
            if p:
                result["associations"]["vehicle_plate"][v["id"]] = p["id"]
                available_plates.remove(p)

        # ── L5 + L6: HSRP Classification + OCR (per vehicle) ────────────
        for v in vehicles:
            vid      = v["id"]
            track_id = v.get("track_id", vid)
            pid      = result["associations"]["vehicle_plate"].get(vid)
            if not pid:
                continue
            plate = next((p for p in plates if p["id"] == pid), None)
            if not plate:
                continue
            crop = plate["_crop"]
            if crop is None or crop.size == 0:
                continue

            # L5: HSRP — TorchScript model, always batch=1 tensor (fine)
            if crop.shape[0] >= 20 and crop.shape[1] >= 40:
                if USE_CUDA:
                    with torch.cuda.stream(self._stream_hsrp):
                        raw_hsrp = hsrp_classifier.predict(crop)
                else:
                    raw_hsrp = hsrp_classifier.predict(crop)

                hsrp_decision            = self.hsrp_manager.process(track_id, raw_hsrp, frame_id)
                plate["hsrp"]            = hsrp_decision["label"]
                plate["hsrp_confidence"] = hsrp_decision["confidence"]
                plate["hsrp_temporal"]   = hsrp_decision["temporal_info"]

            # L6: OCR (conditional)
            _run_ocr = (
                ocr_mode == "always"
                or (ocr_mode == "on_violation" and plate.get("hsrp") in (None, "non_hsrp"))
                or (ocr_mode == "on_clean"     and plate.get("hsrp") not in (None, "non_hsrp"))
            )
            if _run_ocr and ocr_mode != "off":
                raw_ocr = ocr_model.predict(crop)
                if raw_ocr.get("text"):
                    stable_ocr   = ocr_stabilizer.update(
                        vehicle_id=track_id,
                        text=raw_ocr["text"],
                        confidence=raw_ocr.get("confidence", 0.0),
                    )
                    ocr_decision = self.ocr_manager.process(
                        track_id,
                        stable_ocr["text"],
                        stable_ocr["confidence"],
                        frame_id,
                        is_valid_format=bool(stable_ocr.get("text")),
                    )
                    plate["ocr_text"]       = ocr_decision["text"]
                    plate["ocr_confidence"] = ocr_decision["confidence"]
                    plate["ocr_temporal"]   = ocr_decision["temporal_info"]

        if USE_CUDA:
            torch.cuda.current_stream().wait_stream(self._stream_hsrp)

        # ── L3 (rider): vehicle ↔ person association ─────────────────────
        two_wheelers = [v for v in vehicles if v["class"] in TWO_WHEELERS]
        for v in two_wheelers:
            rider = rules.associate_rider(v["bbox"], persons)
            if rider:
                result["associations"]["vehicle_rider"][v["id"]] = rider["id"]

        # ── L7: Helmet Detection — batched across all two-wheelers ────────
        # Collect all head crops first, run one predict_batch() call,
        # then distribute results back. This ensures TRT engines never
        # receive batch=1 (the padding happens inside predict_batch).
        helmet_jobs = []   # list of (enf_dict, head_crop)

        for v in two_wheelers:
            track_id = v.get("track_id", v["id"])
            enf = {
                "vehicle_id":         v["id"],
                "track_id":           track_id,
                "vehicle_bbox":       v["bbox"],
                "vehicle_confidence": v["confidence"],
                "rider_found":        False,
                "person_bbox":        None,
                "helmet":             None,
                "helmet_temporal":    None,
                "plate_id":           result["associations"]["vehicle_plate"].get(v["id"]),
            }

            rider_id = result["associations"]["vehicle_rider"].get(v["id"])
            if rider_id:
                rider = next((p for p in persons if p["id"] == rider_id), None)
                if rider:
                    enf["rider_found"] = True
                    enf["person_bbox"] = rider["bbox"]
                    head, _ = rules.crop_head(frame, rider["bbox"])
                    if head is not None:
                        helmet_jobs.append((enf, head))

            result["enforcements"]["two_wheelers"].append(enf)

        # Run all head crops through helmet detector in one batched call
        if helmet_jobs:
            heads      = [head for _, head in helmet_jobs]
            enf_refs   = [enf  for enf, _ in helmet_jobs]

            if USE_CUDA:
                with torch.cuda.stream(self._stream_helmet):
                    helmet_results = helmet_detector.predict_batch(heads)
            else:
                helmet_results = helmet_detector.predict_batch(heads)

            for enf, raw_helmet in zip(enf_refs, helmet_results):
                helmet_decision      = self.helmet_manager.process(
                    enf["track_id"], raw_helmet, frame_id
                )
                enf["helmet"]        = {
                    "status":       helmet_decision["status"],
                    "confidence":   helmet_decision["confidence"],
                    "has_helmet":   helmet_decision["has_helmet"],
                    "is_violation": helmet_decision["is_violation"],
                    "raw_status":   raw_helmet.get("status"),
                }
                enf["helmet_temporal"] = helmet_decision["temporal_info"]

        if USE_CUDA:
            torch.cuda.current_stream().wait_stream(self._stream_helmet)

        # ── Violation Aggregation ────────────────────────────────────────
        enf_by_vid = {e["vehicle_id"]: e for e in result["enforcements"]["two_wheelers"]}
        assoc_vp   = result["associations"]["vehicle_plate"]

        for v in vehicles:
            track_id      = v.get("track_id", v["id"])
            vehicle_class = v.get("class", "unknown")
            pid           = assoc_vp.get(v["id"])
            plate         = next((p for p in plates if p["id"] == pid), None) if pid else None
            enf           = enf_by_vid.get(v["id"])

            hsrp_decision = None
            if plate and plate.get("hsrp_temporal"):
                hsrp_decision = {
                    "is_violation": plate.get("hsrp") == "non_hsrp",
                    "confidence":   plate.get("hsrp_confidence", 0.0),
                    "temporal_info": plate.get("hsrp_temporal"),
                }

            helmet_decision = None
            if vehicle_class in TWO_WHEELERS and enf and enf.get("helmet"):
                helmet_decision = {
                    "is_violation": enf["helmet"].get("is_violation", False),
                    "confidence":   enf["helmet"].get("confidence", 0.0),
                    "temporal_info": enf.get("helmet_temporal"),
                }

            ocr_decision = None
            if plate and plate.get("ocr_temporal"):
                ocr_decision = {
                    "text":          plate.get("ocr_text"),
                    "confidence":    plate.get("ocr_confidence", 0.0),
                    "temporal_info": plate.get("ocr_temporal"),
                }

            vio = self.violation_aggregator.evaluate_violation(
                track_id=track_id,
                vehicle_class=vehicle_class,
                hsrp_decision=hsrp_decision,
                helmet_decision=helmet_decision,
                ocr_decision=ocr_decision,
                frame_id=frame_id,
            )
            if vio["should_store"]:
                result["violations"].append({
                    "track_id":        track_id,
                    "vehicle_id":      v["id"],
                    "violation_type":  vio["violation_type"],
                    "confidence":      vio["confidence"],
                    "evidence_frames": vio["evidence_frames"],
                    "frame_id":        frame_id,
                    "metadata":        vio["metadata"],
                })
                self.violations_detected += 1

        for plate in result["entities"]["plates"]:
            plate.pop("_crop", None)

        result["temporal_stats"] = {
            "frames_processed":    self.frames_processed,
            "violations_detected": self.violations_detected,
        }
        return result

    def cleanup(self, current_frame: int):
        self.hsrp_manager.fusion.cleanup_old_tracks(current_frame)
        self.helmet_manager.fusion.cleanup_old_tracks(current_frame)
        self.ocr_manager.fusion.cleanup_old_tracks(current_frame)
        self.violation_aggregator.cleanup_old_tracks(current_frame)

    def get_statistics(self):
        return {
            "frames_processed":    self.frames_processed,
            "violations_detected": self.violations_detected,
            "hsrp_fusion":         self.hsrp_manager.fusion.get_statistics(),
            "helmet_fusion":       self.helmet_manager.fusion.get_statistics(),
        }
