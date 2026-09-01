import cv2
import numpy as np
from typing import Iterator, Tuple, List, Optional


def read_video(video_path: str) -> Iterator[Tuple[int, "np.ndarray"]]:
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


def read_video_batched(
    video_path: str,
    batch_size: int = 8,
    frame_skip: int = 1,
) -> Iterator[List[Tuple[int, "np.ndarray"]]]:
    """
    Read video and yield batches of (frame_id, frame) tuples.

    Handles two concerns:
      1. frame_skip — only sampled frames are included in each batch
      2. tail padding — the final batch is padded with repeats of the last
         real frame so downstream YOLO calls always receive a full batch.
         Callers MUST check the `is_pad` flag on each item (added by this
         generator) to discard padded results before feeding temporal fusion.

    Yields:
        List of (frame_id, frame, is_pad) where is_pad=True means the frame
        is padding and its inference result should be discarded.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_id = 0
    buffer: List[Tuple[int, "np.ndarray", bool]] = []
    last_real_frame: Optional["np.ndarray"] = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_skip <= 1 or frame_id % frame_skip == 0:
                buffer.append((frame_id, frame, False))
                last_real_frame = frame

            frame_id += 1

            if len(buffer) == batch_size:
                yield buffer
                buffer = []

        # Flush remaining frames — pad to full batch_size with last real frame
        if buffer:
            if last_real_frame is not None:
                while len(buffer) < batch_size:
                    # Use last real frame_id + 1 offset so frame_ids stay
                    # monotonically increasing (temporal fusion uses frame_id
                    # deltas; padding must not corrupt that)
                    pad_frame_id = buffer[-1][0] + 1
                    buffer.append((pad_frame_id, last_real_frame, True))
            yield buffer

    finally:
        cap.release()
        print(f"Video reading finished. Total frames read: {frame_id}")
