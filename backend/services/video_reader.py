import cv2

def read_video(video_path):
    """
    Lazily read video frame-by-frame.
    Yields: (frame_id, frame)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    frame_id = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            yield frame_id, frame
            frame_id += 1
    finally:
        cap.release()
        print(f"Video reading finished. Total frames: {frame_id}")