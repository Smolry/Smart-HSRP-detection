"""
API ROUTES
===========
Production-ready version for AWS deployment
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, WebSocket, WebSocketDisconnect
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
import sys
import time
import boto3

import redis.asyncio as redis

from backend.core.video_pipeline import process_video, generate_violation_summary
from backend.services.storage import (
    store_violations_batch,
    get_violations,
    save_threshold_state,
    load_threshold_state,
)

# ── Logging ─────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
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
    logger.info("✅ Redis connected")

async def close_redis():
    if redis_client:
        await redis_client.close()

# ── Redis Health ────────────────────────────────

@router.get("/redis-health")
async def redis_health():
    try:
        pong = await redis_client.ping()
        return {"status": "ok", "redis": pong}
    except Exception as e:
        return {"status": "error", "error": str(e)}

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

# ── S3 upload ───────────────────────────────

s3 = boto3.client("s3")

def upload_to_s3(file_path: str, job_id: str) -> str:
    bucket = os.getenv("S3_BUCKET")

    key = f"outputs/{job_id}.mp4"

    s3.upload_file(file_path, bucket, key)

    return f"https://{bucket}.s3.amazonaws.com/{key}"

# ── Worker (FIXED) ──────────────────────────────

def run_pipeline_sync(job_id, video_path, output_video_path,
                      frame_skip, annotate_violations,
                      annotate_no_violations, ocr_mode):

    import redis as sync_redis
    import json, traceback, os, time

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
        def progress_cb(frames, total):
            elapsed = time.time() - start_time
            fps = frames / elapsed if elapsed > 0 else 0

            publish({
                "status": "running",
                "progress": frames,
                "total": total,
                "fps": round(fps, 2)
            })

        output = process_video(
            video_path=video_path,
            output_video_path=output_video_path,
            frame_skip=frame_skip,
            annotate_violations=annotate_violations,
            annotate_no_violations=annotate_no_violations,
            ocr_mode=ocr_mode,
            progress_callback=progress_cb,
        )

        if output.get("violations"):
            store_violations_batch(output["violations"])

        summary = generate_violation_summary(output)

        s3_url = None
        
        if output_video_path and os.path.exists(output_video_path):
            s3_url = upload_to_s3(output_video_path, job_id)
            
        publish({
            "status": "completed",
            "summary": summary,
            "video_url": s3_url
            })

    except Exception as e:
        publish({
            "status": "failed",
            "error": str(e),
            "trace": traceback.format_exc()
        })

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

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

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        video_path = tmp.name

    job_id = str(uuid4())

    output_video_path = None
    if save_output_video:
        Path("static/outputs").mkdir(parents=True, exist_ok=True)
        output_video_path = f"static/outputs/{job_id}.mp4"

    await job_set(job_id, {"status": "queued"})

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
        ocr_mode
    )

    return {"job_id": job_id}

# ── WebSocket (FIXED) ───────────────────────────

@router.websocket("/ws/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()

    pubsub = redis_client.pubsub()
    await pubsub.subscribe(f"job_progress:{job_id}")

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True,
                timeout=1.0
            )

            if message:
                await websocket.send_text(message["data"])

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {job_id}")

    finally:
        await pubsub.unsubscribe(f"job_progress:{job_id}")
        await pubsub.close()

# ── Job status / result (kept — used after WS closes to fetch full result) ───

@router.get("/job-status/{job_id}")
async def job_status(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":       job_id,
        "status":       job.get("status"),
        "filename":     job.get("filename"),
        "progress":     int(job.get("progress", 0)),
        "total_frames": int(job.get("total_frames", 0)),
        "mode":         job.get("mode", "batch"),
        "fps":          float(job.get("fps", 0)),
    }


@router.get("/job-result/{job_id}")
async def job_result(job_id: str):
    job = await job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") == "failed":
        raise HTTPException(status_code=500, detail={
            "error":    job.get("error", "Unknown error"),
            "trace":    job.get("trace", ""),
            "filename": job.get("filename", ""),
        })
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    return job


@router.get("/job-video/{job_id}")
async def job_video(job_id: str):
    job = await job_get(job_id)
    if not job or job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Video not available")
    path = job.get("output_video_path", "")
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video file not found")
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