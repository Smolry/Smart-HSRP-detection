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
