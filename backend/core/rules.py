from config.settings import settings
from backend.utils.geometry import iou
import cv2


# =========================================================
# HELMET VIOLATION LOGIC (STRICT ENFORCEMENT SAFE)
# =========================================================
def helmet_violation(helmet_detected: bool, confidence: float) -> bool:
    """
    Violation only if:
    - Confidence above minimum usable threshold
    - Helmet explicitly NOT detected
    """

    min_conf = float(settings.HELMET_CONF_THRESHOLD)

    if confidence < min_conf:
        return False  # insufficient evidence

    return not helmet_detected


# =========================================================
# HSRP VIOLATION
# =========================================================
def hsrp_violation(is_hsrp: bool, confidence: float) -> bool:
    min_conf = float(settings.HSRP_CONF_THRESHOLD)

    if confidence < min_conf:
        return True  # low confidence = manual review

    return not is_hsrp


# =========================================================
# GENERIC ASSOCIATION
# =========================================================
def associate_by_iou(
    source_bbox,
    targets,
    *,
    iou_threshold: float,
    require_center_inside: bool = False,
):
    best = None
    best_score = 0.0

    sx1, sy1, sx2, sy2 = source_bbox

    for target in targets:
        tbbox = target["bbox"]
        score = iou(source_bbox, tbbox)

        if score < iou_threshold:
            continue

        if require_center_inside:
            cx = (tbbox[0] + tbbox[2]) / 2
            cy = (tbbox[1] + tbbox[3]) / 2
            if not (sx1 <= cx <= sx2 and sy1 <= cy <= sy2):
                continue

        if score > best_score:
            best = target
            best_score = score

    return best


# =========================================================
# RIDER ASSOCIATION (MORE ROBUST)
# =========================================================
def associate_rider(vehicle_bbox, persons):
    """
    Instead of relying purely on IoU,
    also allow partial vertical overlap.
    """

    if not persons:
        return None

    vx1, vy1, vx2, vy2 = vehicle_bbox
    v_height = vy2 - vy1

    best_person = None
    best_overlap = 0.0

    for person in persons:
        px1, py1, px2, py2 = person["bbox"]

        # Horizontal overlap check
        horizontal_overlap = max(
            0, min(vx2, px2) - max(vx1, px1)
        )

        if horizontal_overlap <= 0:
            continue

        # Vertical overlap
        vertical_overlap = max(
            0, min(vy2, py2) - max(vy1, py1)
        )

        overlap_area = horizontal_overlap * vertical_overlap

        if overlap_area > best_overlap:
            best_overlap = overlap_area
            best_person = person

    return best_person


# =========================================================
# PLATE ASSOCIATION (STABLE + DETERMINISTIC)
# =========================================================
def associate_plate(vehicle_bbox, plates):

    if not plates:
        return None

    vx1, vy1, vx2, vy2 = vehicle_bbox
    v_width = vx2 - vx1
    v_height = vy2 - vy1

    best_plate = None
    best_score = float("inf")

    for plate in plates:
        px1, py1, px2, py2 = plate["bbox"]

        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2

        # Horizontal alignment tolerance
        if not (vx1 - 0.2 * v_width <= pcx <= vx2 + 0.2 * v_width):
            continue

        # Plate may be slightly above vehicle
        if pcy < vy1 - 0.3 * v_height:
            continue

        vertical_distance = abs(pcy - vy1)

        if vertical_distance < best_score:
            best_score = vertical_distance
            best_plate = plate

    return best_plate


# =========================================================
# HEAD CROPPING (TRAINING-COMPATIBLE + STABLE)
# =========================================================
def crop_head(frame, person_bbox):
    """
    Head + shoulders crop aligned with training distribution.
    Target aspect ratio ≈ 1.16
    """

    x1, y1, x2, y2 = person_bbox
    frame_h, frame_w = frame.shape[:2]

    person_w = x2 - x1
    person_h = y2 - y1

    if person_w <= 0 or person_h <= 0:
        return None, (0, 0)

    # -----------------------------------
    # Target aspect ratio (training)
    # -----------------------------------
    target_ratio = 1.16  # height / width

    # Use upper portion of person box
    crop_w = person_w * 0.75
    crop_h = crop_w * target_ratio

    center_x = (x1 + x2) / 2
    top_y = y1

    new_x1 = center_x - crop_w / 2
    new_x2 = center_x + crop_w / 2
    new_y1 = top_y
    new_y2 = new_y1 + crop_h

    # Slight padding
    pad = crop_w * 0.05
    new_x1 -= pad
    new_x2 += pad
    new_y2 += pad

    # Clamp
    new_x1 = int(max(0, new_x1))
    new_y1 = int(max(0, new_y1))
    new_x2 = int(min(frame_w, new_x2))
    new_y2 = int(min(frame_h, new_y2))

    if new_x2 <= new_x1 or new_y2 <= new_y1:
        return None, (0, 0)

    head_crop = frame[new_y1:new_y2, new_x1:new_x2]

    if head_crop.size == 0:
        return None, (0, 0)

    return head_crop, (new_x1, new_y1)


# =========================================================
# HELMET DECISION LOGIC (THRESHOLD CONTROLLED)
# =========================================================
def decide_helmet_status(helmet_result: dict) -> str:
    """
    Enforcement-safe helmet decision logic.

    Policy:
    - NO_HELMET requires strong confidence
    - HELMET allowed at moderate confidence
    - Gray zone → UNCERTAIN
    """

    status = helmet_result.get("status")
    confidence = helmet_result.get("confidence", 0.0)

    no_threshold = float(settings.HELMET_NO_THRESHOLD)
    yes_threshold = float(settings.HELMET_YES_THRESHOLD)

    # ------------------------------------
    # Bias Safety Margin
    # ------------------------------------
    margin = 0.03  # prevents edge flipping near threshold

    # ------------------------------------
    # Strong NO_HELMET required
    # ------------------------------------
    if status == "NO_HELMET":
        if confidence >= (no_threshold + margin):
            return "NO_HELMET"
        return "UNCERTAIN"

    # ------------------------------------
    # HELMET allowed slightly easier
    # ------------------------------------
    if status == "HELMET":
        if confidence >= (yes_threshold - margin):
            return "HELMET"
        return "UNCERTAIN"

    return "UNCERTAIN"

