"""
ENHANCED FRAME PIPELINE WITH TEMPORAL FUSION
=============================================
Integrates temporal decision fusion into the frame processing pipeline.

Key improvements:
1. Track-level decision fusion for HSRP, Helmet, OCR
2. Reduced classification fluctuations
3. Advanced violation decision logic
4. Frame-skip aware processing
"""

from backend.core import rules
from backend.services import cropper
from backend.services.decision_managers import (
    HSRPDecisionManager,
    HelmetDecisionManager,
    OCRDecisionManager,
    ViolationDecisionAggregator,
)
from typing import Optional

TWO_WHEELERS = {"motorcycle"}


class FramePipeline:
    """
    Frame processing pipeline with temporal decision fusion.
    
    Usage:
        pipeline = EnhancedFramePipeline()
        result = pipeline.process_frame(frame, frame_id, models, tracker, ocr_stabilizer)
    """
    
    def __init__(self):
        # Initialize decision managers
        self.hsrp_manager = HSRPDecisionManager()
        self.helmet_manager = HelmetDecisionManager()
        self.ocr_manager = OCRDecisionManager()
        self.violation_aggregator = ViolationDecisionAggregator()
        
        # Statistics
        self.frames_processed = 0
        self.violations_detected = 0
    
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
        skip_empty_frames: bool = True,
        frame_skip: int = 1,          # Important for temporal fusion
        annotate_violations: bool = True,   # Annotate when violation detected
        annotate_no_violations: bool = False, # Annotate when no violation
        ocr_mode: str = "on_violation",     # "always" | "on_violation" | "on_clean" | "off"
    ):
        """
        Enhanced per-frame inference pipeline with temporal fusion.
        
        Args:
            frame: Input frame
            frame_id: Frame number
            vehicle_detector: Vehicle detector instance
            helmet_detector: Helmet detector instance
            plate_detector: Plate detector instance
            hsrp_classifier: HSRP classifier instance
            ocr_model: OCR model instance
            ocr_stabilizer: OCR stabilizer instance
            tracker: Vehicle tracker instance
            skip_empty_frames: Whether to skip empty frames
            frame_skip: Number of frames being skipped (for temporal fusion)
        """
        
        self.frames_processed += 1
        
        result = {
            "frame_id": frame_id,
            "entities": {"vehicles": [], "persons": [], "plates": []},
            "associations": {"vehicle_plate": {}, "vehicle_rider": {}},
            "enforcements": {"two_wheelers": []},
            "violations": [],  # New: aggregated violations
            "temporal_stats": {},  # New: fusion statistics
        }
        
        # OPTIMIZATION 1: Early exit for empty frames
        detections = vehicle_detector.detect(frame)
        if not detections:
            if skip_empty_frames:
                return result
            return result
        
        # OPTIMIZATION 2: Split persons/vehicles in single pass
        persons = []
        vehicles = []
        
        for idx, det in enumerate(detections):
            if det["class"] == "person":
                det["id"] = f"p{idx}"
                persons.append(det)
            else:
                det["id"] = f"v{idx}"
                vehicles.append(det)
        
        # OPTIMIZATION 3: Skip plate detection if no vehicles
        if not vehicles:
            result["entities"]["persons"] = persons
            return result
        
        # Update tracking (with persistent IDs)
        vehicles = tracker.update(vehicles)
        
        # OPTIMIZATION 4: Only detect plates if we have vehicles
        raw_plate_boxes = plate_detector.predict(frame) or []
        
        # Early exit if no plates detected
        if not raw_plate_boxes:
            result["entities"]["vehicles"] = vehicles
            result["entities"]["persons"] = persons
            return result
        
        # Batch crop all plates at once
        plate_crops = cropper.crop_plates(frame, raw_plate_boxes)
        
        # OPTIMIZATION 5: Process plates with temporal fusion
        plates = []
        
        for idx, (box, crop) in enumerate(zip(raw_plate_boxes, plate_crops)):
            if not box or len(box) != 4:
                continue
            
            plate = {
                "id": f"pl{idx}",
                "bbox": [int(x) for x in box],
                "confidence": 1.0,
                "hsrp": None,
                "hsrp_confidence": 0.0,
                "hsrp_temporal": None,  # NEW: temporal fusion info
                "ocr_text": None,
                "ocr_confidence": 0.0,
                "ocr_temporal": None,  # NEW: temporal fusion info
                "_crop": crop
            }
            
            plates.append(plate)
        
        result["entities"]["vehicles"] = vehicles
        result["entities"]["persons"] = persons
        result["entities"]["plates"] = plates
        
        # OPTIMIZATION 6: Association
        available_plates = plates.copy()
        
        for v in vehicles:
            p = rules.associate_plate(v["bbox"], available_plates)
            if p:
                result["associations"]["vehicle_plate"][v["id"]] = p["id"]
                available_plates.remove(p)
        
        # OPTIMIZATION 7: Enhanced plate processing with temporal fusion
        for v in vehicles:
            vid = v["id"]
            track_id = v.get("track_id", vid)
            
            pid = result["associations"]["vehicle_plate"].get(vid)
            if not pid:
                continue
            
            plate = next((p for p in plates if p["id"] == pid), None)
            if not plate:
                continue
            
            crop = plate["_crop"]
            if crop is None or crop.size == 0:
                continue
            
            # --- HSRP Classification with Temporal Fusion ---
            if crop.shape[0] >= 20 and crop.shape[1] >= 40:
                raw_hsrp = hsrp_classifier.predict(crop)
                
                # Apply temporal fusion
                hsrp_decision = self.hsrp_manager.process(
                    track_id=track_id,
                    hsrp_result=raw_hsrp,
                    frame_id=frame_id,
                )
                
                plate["hsrp"] = hsrp_decision["label"]
                plate["hsrp_confidence"] = hsrp_decision["confidence"]
                plate["hsrp_temporal"] = hsrp_decision["temporal_info"]
                plate["hsrp_raw"] = {
                    "label": raw_hsrp.get("label"),
                    "confidence": raw_hsrp.get("confidence", 0.0)
                }
            
            # ==============================
            # OCR (Always Run)
            # ==============================
            _run_ocr = False
            
            if ocr_mode == "always":
                _run_ocr = True
                
            elif ocr_mode == "on_violation":
                hsrp_label = plate.get("hsrp")
                _run_ocr = (hsrp_label is None) or (hsrp_label == "non_hsrp")
                
            elif ocr_mode == "on_clean":
                hsrp_label = plate.get("hsrp")
                _run_ocr = (hsrp_label is None) or (hsrp_label != "non_hsrp")
                
            elif ocr_mode == "off":
                _run_ocr = False
                
            
            if _run_ocr:
                raw_ocr = ocr_model.predict(crop)
                
                if raw_ocr.get("text"):
                    stable_ocr = ocr_stabilizer.update(
                        vehicle_id=track_id,
                        text=raw_ocr["text"],
                        confidence=raw_ocr.get("confidence", 0.0)
                    )
                    
                    ocr_decision = self.ocr_manager.process(
                        track_id=track_id,
                        ocr_text=stable_ocr["text"],
                        ocr_confidence=stable_ocr["confidence"],
                        frame_id=frame_id,
                        is_valid_format=stable_ocr["text"] is not None,
                    )
                    
                    plate["ocr_text"] = ocr_decision["text"]
                    plate["ocr_confidence"] = ocr_decision["confidence"]
                    plate["ocr_temporal"] = ocr_decision["temporal_info"]

        
        # OPTIMIZATION 8: Rider association (only for two-wheelers)
        two_wheeler_vehicles = [v for v in vehicles if v["class"] in TWO_WHEELERS]
        
        for v in two_wheeler_vehicles:
            rider = rules.associate_rider(v["bbox"], persons)
            if rider:
                result["associations"]["vehicle_rider"][v["id"]] = rider["id"]
        
        # OPTIMIZATION 9: Enhanced helmet detection with temporal fusion
        for v in two_wheeler_vehicles:
            track_id = v.get("track_id", v["id"])
            
            enforcement = {
                "vehicle_id": v["id"],
                "track_id": track_id,
                "vehicle_bbox": v["bbox"],
                "vehicle_confidence": v["confidence"],
                "rider_found": False,
                "person_bbox": None,
                "helmet": None,
                "helmet_temporal": None,  # NEW
                "plate_id": result["associations"]["vehicle_plate"].get(v["id"])
            }
            
            rider_id = result["associations"]["vehicle_rider"].get(v["id"])
            if not rider_id:
                result["enforcements"]["two_wheelers"].append(enforcement)
                continue
            
            rider = next((p for p in persons if p["id"] == rider_id), None)
            if not rider:
                result["enforcements"]["two_wheelers"].append(enforcement)
                continue
            
            enforcement["rider_found"] = True
            enforcement["person_bbox"] = rider["bbox"]
            
            # Helmet detection on head crop
            head, origin = rules.crop_head(frame, rider["bbox"])
            if head is not None:
                raw_helmet = helmet_detector.predict(head)
                
                helmet_decision = self.helmet_manager.process(
                    track_id=track_id,
                    helmet_result=raw_helmet,
                    frame_id=frame_id,
                )

                
                enforcement["helmet"] = {
                    "status": helmet_decision["status"],
                    "confidence": helmet_decision["confidence"],
                    "has_helmet": helmet_decision["has_helmet"],
                    "is_violation": helmet_decision["is_violation"],
                    "raw_status": raw_helmet.get("status"),
                    "raw_confidence": raw_helmet.get("confidence", 0.0),
                }
                enforcement["helmet_temporal"] = helmet_decision["temporal_info"]
            
            result["enforcements"]["two_wheelers"].append(enforcement)
        
        # OPTIMIZATION 10: Advanced violation aggregation
        for v in vehicles:
            track_id = v.get("track_id", v["id"])
            vehicle_class = v.get("class", "unknown")
            
            # Get decisions for this vehicle
            pid = result["associations"]["vehicle_plate"].get(v["id"])
            plate = next((p for p in plates if p["id"] == pid), None) if pid else None
            
            hsrp_decision = None
            if plate and plate.get("hsrp_temporal"):
                hsrp_decision = {
                    "is_violation": plate.get("hsrp") == "non_hsrp",
                    "confidence": plate.get("hsrp_confidence", 0.0),
                    "temporal_info": plate.get("hsrp_temporal"),
                }
            
            helmet_decision = None
            if vehicle_class in TWO_WHEELERS:
                enforcement = next(
                    (e for e in result["enforcements"]["two_wheelers"] 
                     if e["vehicle_id"] == v["id"]),
                    None
                )
                if enforcement and enforcement.get("helmet"):
                    helmet_decision = {
                        "is_violation": enforcement["helmet"].get("is_violation", False),
                        "confidence": enforcement["helmet"].get("confidence", 0.0),
                        "temporal_info": enforcement.get("helmet_temporal"),
                    }
            
            ocr_decision = None
            if plate and plate.get("ocr_temporal"):
                ocr_decision = {
                    "text": plate.get("ocr_text"),
                    "confidence": plate.get("ocr_confidence", 0.0),
                    "temporal_info": plate.get("ocr_temporal"),
                }
            
            # Evaluate violation
            violation_result = self.violation_aggregator.evaluate_violation(
                track_id=track_id,
                vehicle_class=vehicle_class,
                hsrp_decision=hsrp_decision,
                helmet_decision=helmet_decision,
                ocr_decision=ocr_decision,
                frame_id=frame_id,
            )
            
            if violation_result["should_store"]:
                result["violations"].append({
                    "track_id": track_id,
                    "vehicle_id": v["id"],
                    "violation_type": violation_result["violation_type"],
                    "confidence": violation_result["confidence"],
                    "evidence_frames": violation_result["evidence_frames"],
                    "frame_id": frame_id,
                    "metadata": violation_result["metadata"],
                })
                self.violations_detected += 1
        
        # Cleanup temporary data
        for plate in result["entities"]["plates"]:
            plate.pop("_crop", None)
        
        # Add temporal statistics
        result["temporal_stats"] = {
            "frames_processed": self.frames_processed,
            "violations_detected": self.violations_detected,
        }
        
        return result
    
    def cleanup(self, current_frame: int):
        """Cleanup old track data from decision managers."""
        
        # Get fusion engines from managers
        self.hsrp_manager.fusion.cleanup_old_tracks(current_frame)
        self.helmet_manager.fusion.cleanup_old_tracks(current_frame)
        self.ocr_manager.fusion.cleanup_old_tracks(current_frame)
        self.violation_aggregator.cleanup_old_tracks(current_frame)
    
    def get_statistics(self):
        """Get comprehensive statistics from all decision managers."""
        
        return {
            "hsrp_fusion": self.hsrp_manager.fusion.get_statistics(),
            "helmet_fusion": self.helmet_manager.fusion.get_statistics(),
            "ocr_fusion": self.ocr_manager.fusion.get_statistics(),
            "frames_processed": self.frames_processed,
            "violations_detected": self.violations_detected,
        }


# Convenience function matching original API
def process_frame_enhanced(
    frame,
    frame_id,
    vehicle_detector,
    helmet_detector,
    plate_detector,
    hsrp_classifier,
    ocr_model,
    ocr_stabilizer,
    tracker,
    pipeline: Optional[FramePipeline] = None,
    skip_empty_frames=True,
    frame_skip=1,
    annotate_violations: bool = True,
    annotate_no_violations: bool = False,
    ocr_mode: str = "always",
):
    """
    Drop-in replacement for original process_frame with temporal fusion.
    
    If pipeline is provided, use it (maintains state).
    Otherwise, create a new one (stateless mode).
    """
    
    if pipeline is None:
        pipeline = FramePipeline()
    
    return pipeline.process_frame(
        frame=frame,
        frame_id=frame_id,
        vehicle_detector=vehicle_detector,
        helmet_detector=helmet_detector,
        plate_detector=plate_detector,
        hsrp_classifier=hsrp_classifier,
        ocr_model=ocr_model,
        ocr_stabilizer=ocr_stabilizer,
        tracker=tracker,
        skip_empty_frames=skip_empty_frames,
        frame_skip=frame_skip,
        annotate_violations=annotate_violations,
        annotate_no_violations=annotate_no_violations,
        ocr_mode=ocr_mode,
    )
