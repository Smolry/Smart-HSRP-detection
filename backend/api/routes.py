"""
API ROUTES
===========
EC2 local storage — S3 removed entirely.
Worker uses ThreadPoolExecutor (safe for GPU model imports).
WebSocket uses async pubsub.listen() — no timeout polling issues on Windows.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse
import tempfile
import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from typing import Optional
import logging
import time

import redis.asyncio as redis

from backend.core.video_pipeline import process_video, generate_violation_summary
from backend.services.storage import (
    store_violations_batch, get_violations,
    save_threshold_state, load_threshold_state,
)
from backend.db.database import get_db
from backend.utils.local_storage import (
    save_input_video, get_output_video_path, get_video_url,
    output_video_exists, cleanup_input,
)

logger   = logging.getLogger(__name__)
router   = APIRouter()
executor = ThreadPoolExecutor(max_workers=2)

redis_client: Optional[redis.Redis] = None


# ── Redis ─────────────────────────────────────────────────────────────────────

async def init_redis():
    global redis_client
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    logger.info("Redis connected")


async def close_redis():
    if redis_client:
        await redis_client.close()


async def job_set(job_id: str, data: dict):
    await redis_client.hset(f"job:{job_id}", mapping={
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        for k, v in data.items()
    })


async def job_get(job_id: str):
    data = await redis_client.hgetall(f"job:{job_id}")
    if not data:
        return None
    result = {}
    for k, v in data.items():
        try:
            result[k] = json.loads(v) if v and (v.startswith("{") or v.startswith("[")) else v
        except Exception:
            result[k] = v
    return result


# ── Background worker ─────────────────────────────────────────────────────────

def run_pipeline_sync(job_id, tmp_path, output_video_path,
                      frame_skip, annotate_violations, annotate_no_violations, ocr_mode):
    import redis as sync_redis
    import traceback

    r = sync_redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )
    channel    = f"job_progress:{job_id}"
    start_time = time.time()

    def publish(data):
        r.hset(f"job:{job_id}", mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in data.items()
        })
        r.publish(channel, json.dumps(data))

    try:
        video_path = save_input_video(tmp_path, job_id)
        logger.info(f"[{job_id}] Input saved: {video_path}")

        publish({"status": "running", "progress": 0, "total": 0, "fps": 0})

        def progress_cb(frames, total):
            elapsed = time.time() - start_time
            publish({
                "status":   "running",
                "progress": frames,
                "total":    total,
                "fps":      round(frames / elapsed if elapsed > 0 else 0, 2),
            })

        output = process_video(
            video_path=video_path,
            output_video_path=output_video_path,
            frame_skip=frame_skip,
            annotate_violations=annotate_violations,
            annotate_no_violations=annotate_no_violations,
            ocr_mode=ocr_mode,
            progress_callback=progress_cb,
            job_id=job_id,
        )

        if output.get("violations"):
            store_violations_batch(output["violations"])

        summary   = generate_violation_summary(output)
        video_url = output.get("video_url") or get_video_url(job_id)

        logger.info(f"[{job_id}] Complete | video_url={video_url}")
        publish({
            "status":          "completed",
            "summary":         summary,
            "metadata":        output.get("metadata", {}),
            "track_summaries": output.get("track_summaries", {}),
            "video_url":       video_url,
        })

    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        publish({"status": "failed", "error": str(e), "trace": traceback.format_exc()})

    finally:
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        try:
            cleanup_input(job_id)
        except Exception:
            pass


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/process-video")
async def process_video_endpoint(
    file:                  UploadFile = File(...),
    frame_skip:            int  = Form(1),
    save_output_video:     bool = Form(True),
    annotate_violations:   bool = Form(True),
    annotate_no_violations:bool = Form(False),
    ocr_mode:              str  = Form("on_violation"),
):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id            = str(uuid4())
    output_video_path = get_output_video_path(job_id) if save_output_video else None

    await job_set(job_id, {"status": "queued"})
    logger.info(f"[{job_id}] Queued → thread pool")

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor, run_pipeline_sync,
        job_id, tmp_path, output_video_path,
        frame_skip, annotate_violations, annotate_no_violations, ocr_mode,
    )
    return {"job_id": job_id}


# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    logger.info(f"[{job_id}] WS connected")

    pubsub = redis_client.pubsub()

    try:
        # If already completed before WS connected, reply immediately
        job = await job_get(job_id)
        if job and job.get("status") == "completed":
            await websocket.send_text(json.dumps({"status": "completed", **job}))
            await websocket.close(1000)
            return

        await pubsub.subscribe(f"job_progress:{job_id}")

        async for message in pubsub.listen():
            if message["type"] != "message":
                continue
            raw = message["data"]
            await websocket.send_text(raw)
            try:
                if json.loads(raw).get("status") in ("completed", "failed"):
                    break
            except Exception:
                pass

        await websocket.close(1000)

    except WebSocketDisconnect:
        logger.info(f"[{job_id}] WS disconnected by client")
    except Exception as e:
        logger.error(f"[{job_id}] WS error: {e}")
    finally:
        try:
            await pubsub.unsubscribe(f"job_progress:{job_id}")
            await pubsub.aclose()
        except Exception:
            pass


# ── Job result ────────────────────────────────────────────────────────────────

@router.get("/job-result/{job_id}")
async def job_result(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") == "failed":
        raise HTTPException(500, detail=job)
    if job.get("status") != "completed":
        raise HTTPException(400, "Job not completed yet")
    return job


@router.get("/job-video/{job_id}")
async def job_video(job_id: str):
    if not output_video_exists(job_id):
        raise HTTPException(404, "Video file not found on disk")
    return FileResponse(
        get_output_video_path(job_id),
        media_type="video/mp4",
        filename=f"output_{job_id}.mp4",
        headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache"},
    )


@router.get("/job-tracks/{job_id}")
async def job_tracks(job_id: str):
    job = await job_get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(404, "Job not ready")
    ts = job.get("track_summaries", {})
    if isinstance(ts, str):
        try:
            ts = json.loads(ts)
        except Exception:
            ts = {}
    tracks = sorted(ts.values(), key=lambda t: t.get("first_frame", 0))
    return {"job_id": job_id, "tracks": tracks, "count": len(tracks)}


# ── Violations ────────────────────────────────────────────────────────────────

@router.get("/violations")
async def list_violations(
    limit:          int            = Query(200, ge=1, le=1000),
    offset:         int            = Query(0, ge=0),
    violation_type: Optional[str]  = Query(None),
    needs_review:   Optional[bool] = Query(None),
    min_quality:    float          = Query(0.0, ge=0.0, le=1.0),
):
    rows = get_violations(limit=limit, offset=offset,
                          violation_type=violation_type,
                          needs_review=needs_review, min_quality=min_quality)
    return {"violations": rows, "count": len(rows)}


@router.get("/violations/{violation_id}")
async def get_violation(violation_id: int):
    db_gen = get_db()
    conn   = next(db_gen)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM violations WHERE id = %s", (violation_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, "Violation not found")
            return dict(row)
    finally:
        db_gen.close()


# ── Thresholds ────────────────────────────────────────────────────────────────

@router.get("/thresholds")
async def get_thresholds():
    return load_threshold_state()


@router.post("/thresholds/reset")
async def reset_thresholds():
    defaults = {"hsrp": 0.50, "helmet": 0.40, "ocr_confidence": 0.60}
    save_threshold_state(defaults)
    return {"status": "reset", "thresholds": defaults}
