"""
VEHICLE TRACKER
================
Uses DeepSORT (VehicleTracker) on GPU when available via embedder="torchreid".
Falls back to SimpleTracker (IoU-only) when deep_sort_realtime unavailable.

DeepSORT is critical for temporal fusion — stable track IDs across frames
allow EMA scores to accumulate meaningful evidence per vehicle.
"""
from deep_sort_realtime.deepsort_tracker import DeepSort
import torch


class VehicleTracker:
    """
    DeepSORT tracker with GPU appearance embedder when CUDA is available.
    """
    def __init__(
        self,
        max_age=40,
        n_init=3,
        max_cosine_distance=0.3,
        max_iou_distance=0.6,
        nn_budget=100,
    ):
        device   = "cuda" if torch.cuda.is_available() else "cpu"
        embedder = "torchreid" if device == "cuda" else "mobilenet"

        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            nn_budget=nn_budget,
            embedder=embedder,
            embedder_gpu=(device == "cuda"),
        )
        print(f"[VehicleTracker] DeepSORT | device={device} | embedder={embedder}")

    def update(self, detections, frame):
        if not detections:
            self.tracker.update_tracks([], frame=frame)
            return []

        ds_inputs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0 or det["confidence"] < 0.4:
                continue
            ds_inputs.append(([x1, y1, w, h], det["confidence"], det["class"]))

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame)

        out = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            out.append({
                "track_id":   f"veh_{track.track_id}",
                "bbox":       [x1, y1, x2, y2],
                "confidence": 1.0,
                "class":      track.get_det_class(),
            })
        return out

    def reset(self):
        self.tracker = DeepSort()


class SimpleTracker:
    """
    Ultra-lightweight IoU-only tracker.
    Used as fallback or when speed > re-id accuracy.
    """
    def __init__(self, iou_threshold=0.4, max_age=30):
        self.tracks       = {}
        self.next_id      = 0
        self.iou_threshold = iou_threshold
        self.max_age      = max_age

    def update(self, vehicles):
        # Age existing tracks
        expired = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in expired:
            del self.tracks[tid]
        for t in self.tracks.values():
            t["age"] += 1

        out = []
        matched = set()

        for v in vehicles:
            best_iou, best_id = 0.0, None
            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                iou = self._iou(v["bbox"], track["bbox"])
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou, best_id = iou, tid

            if best_id is not None:
                v["track_id"] = f"veh_{best_id}"
                self.tracks[best_id].update({"bbox": v["bbox"], "age": 0})
                matched.add(best_id)
            else:
                v["track_id"] = f"veh_{self.next_id}"
                self.tracks[self.next_id] = {"bbox": v["bbox"], "age": 0, "class": v.get("class")}
                self.next_id += 1
            out.append(v)
        return out

    def _iou(self, a, b):
        xA, yA = max(a[0], b[0]), max(a[1], b[1])
        xB, yB = min(a[2], b[2]), min(a[3], b[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0:
            return 0.0
        areaA = (a[2] - a[0]) * (a[3] - a[1])
        areaB = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (areaA + areaB - inter)

    def reset(self):
        self.tracks.clear()
        self.next_id = 0
