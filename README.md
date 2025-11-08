# Smart-HSRP-detection
This is our final year project for BE-2026

## Collaborators
<p>
  <a href="https://github.com/Guardian-22"><img alt="@Guardian-22" src="https://github.com/Guardian-22.png" width="80" /></a>
</p>


<img width="1536" height="1024" alt="ChatGPT Image Oct 2, 2025, 02_13_58 AM" src="https://github.com/user-attachments/assets/e06d0977-7252-4899-bed7-a9cd6879f62f" />

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

### steps to test the model:
1) load the notebook on google colab.
2) include the model in your working directory.
3) change path if necessary model=YOLO("path/model.pt")
4) include the test image in the working directory
5) change the results path as per working directory results = model.predict/classify("path/image", save=True,project="path/yolo_predictions",name="exp")

### caution:
1) never leave runtime running while not in use.
2) save newly trained model/best weights/results onto drive before changing or disconnecting runtime.
3) use gpu based runtime for testing/training.
4) try to keep the dataset and directory organized.
5) in order to ensure cross validation try to test on images outside of the dataset.
