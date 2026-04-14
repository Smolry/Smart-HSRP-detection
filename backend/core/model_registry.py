from config.settings import settings
from backend.models.helmet_detector import HelmetDetector
from backend.models.plate_detector import PlateDetector
from backend.models.hsrp_classifier import HSRPClassifier
from backend.models.ocr import PlateOCR
from backend.models.vehicle_detector import VehicleDetector

vehicle_detector = VehicleDetector(settings.VEHICLE_MODEL_PATH)
helmet_detector = HelmetDetector(settings.HELMET_MODEL_PATH)
plate_detector = PlateDetector(settings.PLATE_MODEL_PATH)
hsrp_classifier = HSRPClassifier(settings.HSRP_MODEL_PATH)
ocr_model = PlateOCR()
