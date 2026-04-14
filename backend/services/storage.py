"""
STORAGE SERVICE
================
PostgreSQL persistence with:
- violations table (full metadata + quality scores + flags)
- Adaptive threshold state table
- Gate evaluation before insert
"""

from backend.db.database import get_db
from typing import Dict, Any, List, Optional
import json
import time


# ─────────────────────────────────────────────
# VIOLATIONS TABLE INSERT
# ─────────────────────────────────────────────

def store_violation(record: Dict[str, Any]) -> Optional[int]:
    """
    Insert one gated violation record into the database.
    Returns the inserted row id.
    """
    db_gen = get_db()
    conn   = next(db_gen)

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO violations (
                    track_id,
                    vehicle_number,
                    vehicle_class,
                    violation_type,
                    violation_confidence,
                    stability_score,
                    temporal_consistency,
                    consecutive_detections,
                    quality_score,
                    needs_manual_review,
                    prediction_preceded,
                    prediction_risk_max,
                    first_frame,
                    last_frame,
                    track_duration_frames,
                    screenshot_path,
                    metadata_path
                ) VALUES (
                    %(track_id)s,
                    %(vehicle_number)s,
                    %(vehicle_class)s,
                    %(violation_type)s,
                    %(violation_confidence)s,
                    %(stability_score)s,
                    %(temporal_consistency)s,
                    %(consecutive_detections)s,
                    %(quality_score)s,
                    %(needs_manual_review)s,
                    %(prediction_preceded)s,
                    %(prediction_risk_max)s,
                    %(first_frame)s,
                    %(last_frame)s,
                    %(track_duration_frames)s,
                    %(screenshot_path)s,
                    %(metadata_path)s
                )
                RETURNING id
                """,
                record,
            )
            row_id = cur.fetchone()["id"]
            conn.commit()
            return row_id

    except Exception as e:
        print(f"[storage] Insert failed: {e}")
        conn.rollback()
        return None

    finally:
        db_gen.close()


def store_violations_batch(records: List[Dict[str, Any]]) -> List[Optional[int]]:
    """Batch insert multiple violation records."""
    return [store_violation(r) for r in records]


# ─────────────────────────────────────────────
# THRESHOLD STATE PERSISTENCE
# ─────────────────────────────────────────────

def save_threshold_state(thresholds: Dict[str, float]) -> bool:
    """
    Upsert adaptive threshold state into the DB.
    Table: adaptive_thresholds (decision_type, current_threshold, updated_at)
    """
    db_gen = get_db()
    conn   = next(db_gen)

    try:
        with conn.cursor() as cur:
            for dtype, val in thresholds.items():
                cur.execute(
                    """
                    INSERT INTO adaptive_thresholds (decision_type, current_threshold, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (decision_type)
                    DO UPDATE SET current_threshold = EXCLUDED.current_threshold,
                                  updated_at = EXCLUDED.updated_at
                    """,
                    (dtype, val),
                )
            conn.commit()
            return True

    except Exception as e:
        print(f"[storage] Threshold save failed: {e}")
        conn.rollback()
        return False

    finally:
        db_gen.close()


def load_threshold_state() -> Dict[str, float]:
    """Load latest adaptive thresholds from DB."""
    db_gen = get_db()
    conn   = next(db_gen)

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT decision_type, current_threshold FROM adaptive_thresholds")
            rows = cur.fetchall()
            return {r["decision_type"]: float(r["current_threshold"]) for r in rows}

    except Exception as e:
        print(f"[storage] Threshold load failed: {e}")
        return {}

    finally:
        db_gen.close()


# ─────────────────────────────────────────────
# VIOLATIONS QUERY
# ─────────────────────────────────────────────

def get_violations(
    limit: int = 200,
    offset: int = 0,
    violation_type: Optional[str] = None,
    needs_review: Optional[bool] = None,
    min_quality: float = 0.0,
) -> List[Dict[str, Any]]:
    """Query violations with optional filters."""
    db_gen = get_db()
    conn   = next(db_gen)

    conditions = ["quality_score >= %(min_quality)s"]
    params: Dict[str, Any] = {"min_quality": min_quality, "limit": limit, "offset": offset}

    if violation_type:
        conditions.append("violation_type = %(violation_type)s")
        params["violation_type"] = violation_type

    if needs_review is not None:
        conditions.append("needs_manual_review = %(needs_review)s")
        params["needs_review"] = needs_review

    where = " AND ".join(conditions)

    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT
                    id,
                    track_id,
                    vehicle_number,
                    vehicle_class,
                    violation_type,
                    violation_confidence,
                    stability_score,
                    temporal_consistency,
                    consecutive_detections,
                    quality_score,
                    needs_manual_review,
                    prediction_preceded,
                    prediction_risk_max,
                    first_frame,
                    last_frame,
                    track_duration_frames,
                    screenshot_path,
                    metadata_path,
                    created_at
                FROM violations
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %(limit)s OFFSET %(offset)s
                """,
                params,
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    except Exception as e:
        print(f"[storage] Query failed: {e}")
        return []

    finally:
        db_gen.close()


# ─────────────────────────────────────────────
# LEGACY SHIM (backward compat)
# ─────────────────────────────────────────────

def persist_pipeline_result(event_data: dict, plates: list):
    """
    Kept for backward compatibility.
    Wraps old-style event/plates into new violation record format.
    """
    for p in plates:
        record = {
            "track_id":              None,
            "vehicle_number":        p.get("ocr_text"),
            "vehicle_class":         "unknown",
            "violation_type":        (
                "no_helmet" if event_data.get("helmet_violation")
                else "non_hsrp_plate" if not p.get("is_hsrp") else None
            ),
            "violation_confidence":  max(
                event_data.get("helmet_confidence", 0.0),
                p.get("hsrp_confidence", 0.0),
            ),
            "stability_score":       0.5,
            "temporal_consistency":  0.5,
            "consecutive_detections": 1,
            "quality_score":         0.5,
            "needs_manual_review":   True,  # legacy inserts always flagged
            "prediction_preceded":   False,
            "prediction_risk_max":   0.0,
            "first_frame":           0,
            "last_frame":            0,
            "track_duration_frames": 0,
            "screenshot_path":       event_data.get("image_path"),
            "metadata_path":         None,
        }
        if record["violation_type"]:
            store_violation(record)
