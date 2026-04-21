"""
MODEL REGISTRY
===============
Loads all models ONCE at process start.
All downstream code imports singletons from here — no repeated loading.
"""
from backend.models.vehicle_detector import VehicleDetector
from backend.models.plate_detector   import PlateDetector
from backend.models.helmet_detector  import HelmetDetector
from backend.models.hsrp_classifier  import HSRPClassifier
from backend.models.ocr              import PlateOCR

print("[ModelRegistry] Loading models...")
vehicle_detector = VehicleDetector()
plate_detector   = PlateDetector()
helmet_detector  = HelmetDetector()
hsrp_classifier  = HSRPClassifier()
ocr_model        = PlateOCR()
print("[ModelRegistry] All models ready.")
