"""
API ROUTES
===========
EC2 local storage version — S3 completely removed.

Videos are saved to and served from the EC2 filesystem via FastAPI StaticFiles.
- Input  → static/inputs/<job_id>.mp4  (cleaned up after processing)
- Output → static/outputs/<job_id>.mp4 (served via /static/outputs/<job_id>.mp4)
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import FileResponse
import tempfile
import os
import json
import asyncio
from concurrent.futures import ProcessPoolExecutor
from uuid import uuid4
from typing import Optional
from pathlib import Path
import logging
import time

import redis.asyncio as redis

from backend.core.video_pipeline import process_video, generate_violation_summary
from backend.services.storage import (
    store_violations_batch,
    get_violations,
    save_threshold_state,
    load_threshold_state,
)
from backend.db.database import get_db
from backend.utils.local_storage import (
    save_input_video,
    get_output_video_path,
    get_video_url,
    output_video_exists,
    cleanup_input,
)

# ── Logging ─────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Globals ─────────────────────────────────────

router = APIRouter()
redis_client: Optional[redis.Redis] = None
executor = ProcessPoolExecutor(max_workers=2)

# ── Redis Setup ─────────────────────────────────

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

# ── Redis Helpers ───────────────────────────────

async def job_set(job_id: str, data: dict):
    await redis_client.hset(f"job:{job_id}", mapping={
        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
        for k, v in data.items()
    })

async def job_get(job_id: str):
    data = await redis_client.hgetall(f"job:{job_id}")
    if not data:
        return None
    return {
        k: json.loads(v) if v.startswith("{") or v.startswith("[") else v
        for k, v in data.items()
    }

# ── Worker ─────────────────────────────────────

def run_pipeline_sync(job_id, video_path, output_video_path,
                      frame_skip, annotate_violations,
                      annotate_no_violations, ocr_mode):

    import redis as sync_redis
    import traceback

    r = sync_redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True,
    )

    channel = f"job_progress:{job_id}"
    start_time = time.time()

    def publish(data):
        r.hset(f"job:{job_id}", mapping={
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in data.items()
        })
        r.publish(channel, json.dumps(data))

    try:
        # ── Progress callback ───────────────
        def progress_cb(frames, total):
            elapsed = time.time() - start_time
            fps = frames / elapsed if elapsed > 0 else 0
            publish({
                "status": "running",
                "progress": frames,
                "total": total,
                "fps": round(fps, 2),
            })

        # ── Run pipeline ────────────────────────────────────────────────
        # process_video writes output to output_video_path on local disk.
        # It returns output["video_url"] as the /static/outputs/<job_id>.mp4 URL.
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

        # ── Store violations in DB ──────────────────────────────────────
        if output.get("violations"):
            store_violations_batch(output["violations"])

        summary = generate_violation_summary(output)

        # ── Local video URL (set by pipeline via get_video_url) ─────────
        local_video_url = output.get("video_url") or get_video_url(job_id)

        logger.info(f"[{job_id}] Output video available at: {local_video_url}")

        publish({
            "status": "completed",
            "summary": summary,
            "video_url": local_video_url,
        })

    except Exception as e:
        publish({
            "status": "failed",
            "error": str(e),
            "trace": traceback.format_exc(),
        })

    finally:
        # Clean up the raw input video to free space; output stays for serving
        cleanup_input(job_id)
        # Also clean up any leftover temp file
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass

# ── Upload Endpoint ─────────────────────────────

@router.post("/process-video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    frame_skip: int = Form(1),
    save_output_video: bool = Form(True),
    annotate_violations: bool = Form(True),
    annotate_no_violations: bool = Form(False),
    ocr_mode: str = Form("always"),
):
    suffix = os.path.splitext(file.filename)[1] or ".mp4"

    # ── Save uploaded file to a temp location ────────────────
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    job_id = str(uuid4())

    # ── Copy to static/inputs/<job_id>.mp4 for traceability ──
    video_path = save_input_video(tmp_path, job_id)
    os.remove(tmp_path)   # remove the original temp file

    logger.info(f"[{job_id}] Input saved locally: {video_path}")

    # ── Output path on EC2 disk ───────────────────────────────
    output_video_path = get_output_video_path(job_id) if save_output_video else None

    await job_set(job_id, {
        "status": "queued",
        "input_video_path": video_path,
    })

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        run_pipeline_sync,
        job_id,
        video_path,
        output_video_path,
        frame_skip,
        annotate_violations,
        annotate_no_violations,
        ocr_mode,
    )

    return {"job_id": job_id}

# ── WebSocket ─────────────────────────────────────────────────────────────────

@router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"job_progress:{job_id}")

    try:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            if message:
                await websocket.send_text(message["data"])
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {job_id}")

    finally:
        await pubsub.unsubscribe(f"job_progress:{job_id}")
        await pubsub.close()

# ── Job Result ────────────────────────────────────────────────────────────────

@router.get("/job-result/{job_id}")
async def job_result(job_id: str):
    job = await job_get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail=job)

    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    return job

# ── Job Video — served directly from EC2 disk ─────────────────────────────────

@router.get("/job-video/{job_id}")
async def job_video(job_id: str):
    job = await job_get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Video not available")

    if not output_video_exists(job_id):
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    path = get_output_video_path(job_id)
    return FileResponse(path, media_type="video/mp4", filename=f"output_{job_id}.mp4")

# ── Violations ────────────────────────────────────────────────────────────────

@router.get("/violations")
async def list_violations(
    limit:          int   = Query(200, ge=1, le=1000),
    offset:         int   = Query(0, ge=0),
    violation_type: Optional[str]  = Query(None),
    needs_review:   Optional[bool] = Query(None),
    min_quality:    float = Query(0.0, ge=0.0, le=1.0),
):
    rows = get_violations(
        limit=limit, offset=offset,
        violation_type=violation_type,
        needs_review=needs_review,
        min_quality=min_quality,
    )
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
                raise HTTPException(status_code=404, detail="Violation not found")
            return dict(row)
    finally:
        db_gen.close()

# ── Track summaries ───────────────────────────────────────────────────────────

@router.get("/job-tracks/{job_id}")
async def job_tracks(job_id: str):
    job = await job_get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Job not ready")
    track_summaries = job.get("track_summaries", {})
    tracks = sorted(track_summaries.values(), key=lambda t: t.get("first_frame", 0))
    return {"job_id": job_id, "tracks": tracks, "count": len(tracks)}

# ── Adaptive thresholds ───────────────────────────────────────────────────────

@router.get("/thresholds")
async def get_thresholds():
    return load_threshold_state()


@router.post("/thresholds/reset")
async def reset_thresholds():
    defaults = {"hsrp": 0.50, "helmet": 0.40, "ocr_confidence": 0.60}
    save_threshold_state(defaults)
    return {"status": "reset", "thresholds": defaults}

# ── Legacy ────────────────────────────────────────────────────────────────────

@router.get("/job-legacy-result/{job_id}")
async def legacy_result(job_id: str):
    return await job_result(job_id)
