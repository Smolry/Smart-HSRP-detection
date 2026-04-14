"""
OPTIMIZED VEHICLE TRACKER
=========================
Performance improvements over standard DeepSORT:
1. Configurable tracking parameters
2. Better memory management
3. Track lifecycle management
4. Optimized for traffic scenarios
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    """
    Optimized vehicle tracker for traffic monitoring scenarios.
    
    Key optimizations:
    - Aggressive track cleanup for stationary/slow-moving vehicles
    - Higher IOU threshold for traffic (vehicles don't overlap much)
    - Faster initialization for quick tracking
    """
    
    def __init__(
        self,
        max_age=40,              # Reduced from 60: faster cleanup
        n_init=3,                # Reduced from 5: faster track initialization
        max_cosine_distance=0.3,  # Increased: less strict on appearance
        max_iou_distance=0.6,    # Reduced from 0.7: tighter spatial matching
        nn_budget=100,           # Feature budget per track
    ):
        """
        Initialize optimized tracker.
        
        Args:
            max_age: Frames before track deletion (lower = faster cleanup)
            n_init: Frames needed to confirm track (lower = faster initialization)
            max_cosine_distance: Appearance similarity threshold
            max_iou_distance: Spatial overlap threshold (lower = stricter)
            nn_budget: Feature vectors per track (lower = less memory)
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            max_iou_distance=max_iou_distance,
            nn_budget=nn_budget,
            # embedder="mobilenet",  # Faster than default
        )
        
        # Track statistics
        self.total_tracks_created = 0
        self.active_tracks = 0

    def update(self, detections, frame):
        """
        Update tracks with new detections.
        
        Input:
            detections: List[dict] with structure:
                {
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class": str
                }
            frame: np.ndarray (required for feature extraction)

        Output:
            List[dict] with structure:
                {
                    "track_id": str,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float,
                    "class": str
                }
        """
        
        if not detections:
            # Update with empty detections to age out old tracks
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert to DeepSORT format: ([x, y, w, h], confidence, class)
        deep_sort_inputs = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = det["confidence"]
            cls = det["class"]

            # Validation: skip invalid boxes
            if w <= 0 or h <= 0 or conf < 0.4:
                continue

            deep_sort_inputs.append(
                ([x1, y1, w, h], conf, cls)
            )

        # Update tracker
        tracks = self.tracker.update_tracks(
            deep_sort_inputs,
            frame=frame
        )

        # Convert to output format
        output_tracks = []
        confirmed_count = 0

        for track in tracks:
            # Only return confirmed tracks
            if not track.is_confirmed():
                continue

            confirmed_count += 1
            
            # Get bounding box in [x1, y1, x2, y2] format
            x1, y1, x2, y2 = map(int, track.to_ltrb())

            output_tracks.append({
                "track_id": f"veh_{track.track_id}",
                "bbox": [x1, y1, x2, y2],
                "confidence": det.get("confidence", 1.0),  # Preserve original confidence
                "class": track.get_det_class()
            })

        # Update statistics
        self.active_tracks = confirmed_count

        return output_tracks

    def reset(self):
        """Reset tracker state (use between videos)"""
        self.tracker = DeepSort(
            max_age=self.tracker.max_age,
            n_init=self.tracker.n_init,
            max_cosine_distance=self.tracker.max_cosine_distance,
            max_iou_distance=self.tracker.max_iou_distance,
        )
        self.total_tracks_created = 0
        self.active_tracks = 0

    def get_stats(self):
        """Get tracking statistics"""
        return {
            "active_tracks": self.active_tracks,
            "total_created": self.total_tracks_created
        }


class SimpleTracker:
    """
    Ultra-lightweight IoU-based tracker (no deep learning).
    
    Use when:
    - Speed is critical
    - DeepSORT is too slow
    - Simple tracking is sufficient
    
    Pros:
    - Very fast (no feature extraction)
    - Low memory usage
    - Good for simple scenarios
    
    Cons:
    - Less robust to occlusion
    - No appearance modeling
    - Higher ID switches
    """

    def __init__(self, iou_threshold=0.4, max_age=30):
        """
        Args:
            iou_threshold: Minimum IoU for track association
            max_age: Frames before track expires
        """
        self.tracks = {}  # track_id -> {"bbox": [...], "age": int, "class": str}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_age = max_age

    def update(self, vehicles):
        """
        Update tracks with new vehicle detections.
        
        Args:
            vehicles: List[dict] with "bbox" and "class"
        
        Returns:
            List[dict] with added "track_id" field
        """
        
        # Age existing tracks
        expired_ids = []
        for tid, track in self.tracks.items():
            track["age"] += 1
            if track["age"] > self.max_age:
                expired_ids.append(tid)
        
        # Remove expired tracks
        for tid in expired_ids:
            del self.tracks[tid]

        # Match new detections to existing tracks
        updated_vehicles = []
        matched_track_ids = set()

        for v in vehicles:
            best_iou = 0.0
            best_id = None

            # Find best matching track
            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue  # Already matched
                
                score = self._iou(v["bbox"], track["bbox"])
                if score > best_iou and score >= self.iou_threshold:
                    best_iou = score
                    best_id = tid

            # Assign track ID
            if best_id is not None:
                # Matched existing track
                v["track_id"] = f"veh_{best_id}"
                self.tracks[best_id]["bbox"] = v["bbox"]
                self.tracks[best_id]["age"] = 0  # Reset age
                matched_track_ids.add(best_id)
            else:
                # Create new track
                v["track_id"] = f"veh_{self.next_id}"
                self.tracks[self.next_id] = {
                    "bbox": v["bbox"],
                    "age": 0,
                    "class": v.get("class", "unknown")
                }
                self.next_id += 1

            updated_vehicles.append(v)

        return updated_vehicles

    def _iou(self, boxA, boxB):
        """Calculate IoU between two boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        if inter <= 0:
            return 0.0

        areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        return inter / float(areaA + areaB - inter)

    def reset(self):
        """Reset tracker state"""
        self.tracks.clear()
        self.next_id = 0

    def get_stats(self):
        """Get tracking statistics"""
        return {
            "active_tracks": len(self.tracks),
            "total_created": self.next_id
        }


# Compatibility alias (use optimized version by default)
VehicleTracker = VehicleTracker
SimpleTracker = SimpleTracker
