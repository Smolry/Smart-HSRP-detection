import numpy as np

def crop_rois(image, bboxes):
    crops = []
    h, w = image.shape[:2]      # Get image height and width

    for x1, y1, x2, y2 in bboxes:
        x1, y1 = max(0, x1), max(0, y1) 
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            crops.append(image[y1:y2, x1:x2])

    return crops

def crop_plates(image, bboxes):
    crops = []
    h, w = image.shape[:2]      # Get image height and width

    for x1, y1, x2, y2 in bboxes:
        x1, y1 = max(0, x1), max(0, y1) 
        x2, y2 = min(w, x2), min(h, y2)

        if x2 > x1 and y2 > y1:
            crops.append(image[y1:y2, x1:x2])

    return crops
