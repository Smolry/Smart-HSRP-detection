from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')   # or yolov8n.pt (faster), yolov8m.pt (more accurate)
# Initial Training
model.train(
    data='/content/helmet_detection/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16
)
# Load the Best Model from Previous Training
model = YOLO('/content/runs/detect/train3/weights/best.pt')

#Training with Data Augmentation ! use for colab notebook 
!yolo train \
  model=/content/runs/detect/train3/weights/best.pt \
  data=/content/helmet_detection/data.yaml \
  epochs=150 \
  imgsz=640 \
  lr0=0.002 \
  degrees=10 \
  translate=0.1 \
  scale=0.8 \
  shear=2 \
  perspective=0.0005 \
  fliplr=0.5 \
  flipud=0.1 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  mosaic=1.0 \
  mixup=0.2 \
  copy_paste=0.2

# Load the Best Model (train4) and Predict
model = YOLO('/content/runs/detect/train4/weights/best.pt')
model.predict(source='/content/helmet_detection/valid/images', save=True)

# Validate Model
results = model.val()

# Visualize one of the predictions
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('/content/runs/detect/predict/Screenshot-2023-08-02-at-12-27-33-PM_png.rf.1a950cbbe661db30fac5547969a4908a.jpg')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')