# Helmet Detection Using YOLOv8

## 📋 Project Overview
This project implements a **real-time helmet and rider detection system** using **YOLOv8**. The system is designed for traffic safety monitoring and enforcement.

### Key Features
It can detect:
    - Riders  
    - Helmet status (proper, improper, or no helmet)  
    - Vehicle number plates 

## 🛠️ Setup and Installation

### Google Colab Setup
1. Open the [Colab Notebook](https://colab.research.google.com/drive/)

2. Upload the dataset which is inside the zip file
    (dataset link :- https://www.kaggle.com/datasets/aryanvaid13/indian-helmet-detection-dataset?select=data.yaml)


3. Install YOLOv8:
```bash
!pip install ultralytics
```

4. Download and prepare the dataset:
```bash
!unzip helmet_detection.zip -d /content/helmet_detection
```

## 📊 Dataset Information
- **Source:** Indian Helmet Detection Dataset (Kaggle)
- **Size:** 942 images
- **Format:** YOLOv8 annotation (.txt with normalized coordinates)
- **Resolution:** 640×640 pixels

### Data Preprocessing
- Image resizing to 640×640
- Data augmentation techniques:
  - Mosaic
  - Flip
  - HSV adjustment
  - Mixup
  - Copy-paste

### Project Structure
```
helmet_detection/
├── data.yaml          # Dataset configuration
├── train/
│   ├── images/        # Training images
│   └── labels/        # Training annotations
└── valid/
    ├── images/        # Validation images
    └── labels/        # Validation annotations
```

### Class Labels
| ID | Class Name | Description |
|----|------------|-------------|
| 0 | number_plate | Vehicle registration plate |
| 1 | face_no_helmet | Rider without helmet |
| 2 | face_helmet_good | Proper helmet usage |
| 3 | face_helmet_bad | Improper helmet usage |
| 4 | rider | Person riding the vehicle |

## 💻 Usage Guide

### Google Colab Training

1. **Save the trained model:**
```python
from google.colab import files

# Download trained weights
files.download('/content/runs/detect/train/weights/best.pt') 
# replace the train with the best weight like train3 or train4 
```
### Model Configuration
The `data.yaml` file contains class configurations:
```yaml
nc: 5  # number of classes
names: ['number_plate', 'face_no_helmet', 'face_helmet_good', 'face_helmet_bad', 'rider']
```

## ✨ Capabilities
- Real-time detection in images, videos, and camera feeds
- Multi-object detection in single frames
- Helmet status classification
- Number plate detection
- Integration support for OCR and violation logging

## ⚠️ Limitations
1. Reduced accuracy for improper helmet detection (limited training samples)
2. HSRP security marks detection not implemented



