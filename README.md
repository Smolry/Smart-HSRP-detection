# Smart-HSRP-detection
This is our final year project for BE-2026

### Camera / Video Feed
###     │
###     ▼
### [YOLOv8 Detection Model]  → detects license plate bounding box
        │
        ▼
### Crop License Plate Region
        │
        ▼
### [YOLOv8 Classification Model] → Classify: HSRP / Non-HSRP
        │
   ┌─────┴───────┐
   │             │
   ▼             ▼
### HSRP        Non-HSRP
                │
                ▼
###     [Feature Detection Module]
        ├─ Check hologram
        ├─ Check font type
        ├─ Check embossing
        └─ Check placement/alignment
                │
                ▼
###     Output: "Non-HSRP because of <feature>"

Detect the license plate in a car image (bounding box).

Classify whether it’s HSRP or Non-HSRP.

Explain why (detect specific features of a fake/non-HSRP plate).

That means you’ll eventually need both detection + classification:

YOLO Detection model (yolov8s.pt)
→ Find license plates in real-time video (bounding box).

YOLO Classification model (yolov8s-cls.pt)
→ Classify cropped plate into HSRP vs Non-HSRP.

(Optional) Feature detection model
→ Train on annotations of “features” (e.g., missing hologram, wrong font, incorrect embossing) so the model can explain why a plate is non-HSRP.
