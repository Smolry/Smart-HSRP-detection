"""
LIVE STREAM HANDLER — BUFFERED ADAPTIVE STREAMING
===================================================
Every frame is processed. No frames are dropped.

Architecture:
  Phone  →  /api/stream  WebSocket
               │
               ▼
         intake_queue (unbounded FIFO)
               │
               ▼
         GPU pipeline (processes at its own pace)
               │
               ▼
         _broadcast_queue  →  per-viewer buffer deques
               │
               ▼
         /api/stream-view  paced playback at PLAYBACK_FPS

Phone side (capture.html):
  - Captures at fixed 10 fps into a local JS send-buffer array
  - Drains send-buffer as fast as WS allows (not rate-limited)
  - Shows buffer depth — grows when GPU is slower than capture rate
  - Warns at MAX_PHONE_BUFFER_FRAMES cap

Server side:
  - intake_queue is UNBOUNDED — every frame queued, none discarded
  - GPU processes strictly in order
  - After each frame: ack sent to phone with queue_depth
  - Processed frames broadcast to all viewers

Viewer (dashboard) side:
  - Buffers INITIAL_BUFFER_FRAMES before starting playback
  - Plays at PLAYBACK_FPS from its own deque
  - If deque empties → sends buffering status to dashboard (spinner)
  - Resumes when buffer refills
"""

import asyncio
import collections
import cv2
import numpy as np
import json
import logging
import time
from typing import Set, Optional, Dict

from fastapi import WebSocket, WebSocketDisconnect

from backend.core.frame_pipeline import FramePipeline
from backend.core.video_annotator import annotate_frame
import backend.core.model_registry as models
from backend.services.vehicle_tracker import VehicleTracker
from backend.services.ocr_stabilizer import OCRStabilizer

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

INITIAL_BUFFER_FRAMES  = 8     # viewer buffers this many frames before playback starts
PLAYBACK_FPS           = 10    # viewer plays at this rate regardless of processing speed
VIEWER_MAX_BUFFER      = 600   # ~60s at 10fps — cap to avoid unbounded memory
MAX_PHONE_BUFFER_FRAMES = 300  # sent to phone so JS can show warning

# ── Broadcast state ───────────────────────────────────────────────────────────

class ViewerState:
    """Per-viewer playback buffer and pacing state."""
    def __init__(self):
        self.buffer: collections.deque = collections.deque()
        self.buffering = True  # True until initial buffer threshold met


_viewers: Dict[WebSocket, ViewerState] = {}
_broadcast_queue: asyncio.Queue = asyncio.Queue()  # unbounded

# ── Session state ─────────────────────────────────────────────────────────────

class LiveSession:
    """Holds per-camera temporal state that must persist across frames."""

    def __init__(self):
        self.pipeline         = FramePipeline()
        self.tracker          = VehicleTracker()
        self.ocr_stabilizer   = OCRStabilizer()
        self.frame_id         = 0
        self.frames_processed = 0
        self.start_time       = time.time()
        self.last_fps_time    = time.time()
        self.fps_counter      = 0
        self.current_fps      = 0.0
        # UNBOUNDED — every frame from phone is queued, none discarded
        self.intake_queue: asyncio.Queue = asyncio.Queue()

    def update_fps(self):
        self.fps_counter += 1
        now     = time.time()
        elapsed = now - self.last_fps_time
        if elapsed >= 1.0:
            self.current_fps   = round(self.fps_counter / elapsed, 1)
            self.fps_counter   = 0
            self.last_fps_time = now


_session: Optional[LiveSession] = None

# ── Broadcast worker ──────────────────────────────────────────────────────────

async def _broadcast_worker():
    """
    Drains _broadcast_queue and copies each processed frame into every
    connected viewer's buffer deque.
    """
    while True:
        try:
            jpeg_bytes, meta_json = await _broadcast_queue.get()

            dead = set()
            for ws, state in list(_viewers.items()):
                try:
                    # Silently drop oldest frame if viewer is extremely behind
                    # (tab backgrounded etc.) to prevent unlimited memory growth
                    if len(state.buffer) >= VIEWER_MAX_BUFFER:
                        state.buffer.popleft()
                    state.buffer.append((jpeg_bytes, meta_json))

                    # Transition out of initial buffering once threshold met
                    if state.buffering and len(state.buffer) >= INITIAL_BUFFER_FRAMES:
                        state.buffering = False

                except Exception:
                    dead.add(ws)

            for ws in dead:
                _viewers.pop(ws, None)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"[broadcast_worker] error: {e}")


_broadcast_task: Optional[asyncio.Task] = None


def ensure_broadcast_worker():
    global _broadcast_task
    try:
        loop = asyncio.get_event_loop()
        if _broadcast_task is None or _broadcast_task.done():
            _broadcast_task = loop.create_task(_broadcast_worker())
            logger.info("[live_stream] Broadcast worker started")
    except RuntimeError:
        pass


# ── /stream — phone sender endpoint ──────────────────────────────────────────

async def handle_phone_stream(websocket: WebSocket):
    """
    Phone sends raw JPEG bytes continuously.
    Every frame is enqueued and processed in order — no drops.
    """
    global _session

    ensure_broadcast_worker()
    await websocket.accept()
    logger.info("[/stream] Phone connected")

    _session = LiveSession()
    session  = _session

    async def _receive_loop():
        """Receive JPEG bytes from phone, push to unbounded intake_queue."""
        while True:
            try:
                data = await websocket.receive_bytes()
                await session.intake_queue.put(data)

                # Ack with current queue depth so phone can display backpressure
                try:
                    await websocket.send_text(json.dumps({
                        "type":                  "ack",
                        "queue_depth":           session.intake_queue.qsize(),
                        "max_phone_buffer":      MAX_PHONE_BUFFER_FRAMES,
                    }))
                except Exception:
                    pass

            except WebSocketDisconnect:
                await session.intake_queue.put(None)  # sentinel
                break
            except Exception as e:
                logger.warning(f"[/stream] receive error: {e}")
                await session.intake_queue.put(None)
                break

    async def _process_loop():
        """Drain intake_queue strictly in order, run GPU pipeline, broadcast."""
        loop = asyncio.get_event_loop()
        while True:
            data = await session.intake_queue.get()
            if data is None:
                break

            try:
                arr   = np.frombuffer(data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    logger.warning("[/stream] Bad JPEG — skipping frame")
                    continue

                session.frame_id += 1
                fid = session.frame_id

                result = await loop.run_in_executor(
                    None, _run_pipeline_sync, frame.copy(), fid, session,
                )

                annotated, _ = annotate_frame(frame, result, {})

                ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                jpeg_bytes = buf.tobytes()

                session.update_fps()
                session.frames_processed += 1

                meta = {
                    "type":             "frame",
                    "frame_id":         fid,
                    "fps":              session.current_fps,
                    "frames_processed": session.frames_processed,
                    "queue_depth":      session.intake_queue.qsize(),
                    "violations":       result.get("violations", []),
                    "tracks":           _extract_tracks(result),
                    "active_tracks":    len(result.get("entities", {}).get("vehicles", [])),
                    "temporal_stats":   result.get("temporal_stats", {}),
                }
                meta_json = json.dumps(meta, default=str)

                # Send processed result back to phone
                try:
                    await websocket.send_bytes(jpeg_bytes)
                    await websocket.send_text(meta_json)
                except Exception:
                    break

                # Broadcast to all dashboard viewers
                await _broadcast_queue.put((jpeg_bytes, meta_json))

            except Exception as e:
                logger.error(f"[/stream] pipeline error frame {session.frame_id}: {e}")

    try:
        await asyncio.gather(_receive_loop(), _process_loop())
    except Exception as e:
        logger.error(f"[/stream] session error: {e}")
    finally:
        logger.info(
            f"[/stream] Phone disconnected — "
            f"processed={session.frames_processed} "
            f"remaining_in_queue={session.intake_queue.qsize()}"
        )


def _run_pipeline_sync(frame: np.ndarray, frame_id: int, session: LiveSession) -> dict:
    return session.pipeline.process_frame(
        frame=frame,
        frame_id=frame_id,
        vehicle_detector=models.vehicle_detector,
        helmet_detector=models.helmet_detector,
        plate_detector=models.plate_detector,
        hsrp_classifier=models.hsrp_classifier,
        ocr_model=models.ocr_model,
        ocr_stabilizer=session.ocr_stabilizer,
        tracker=session.tracker,
        skip_empty_frames=False,
        frame_skip=1,
        annotate_violations=True,
        annotate_no_violations=True,
        ocr_mode="on_violation",
    )


def _extract_tracks(result: dict) -> list:
    vehicles   = result.get("entities", {}).get("vehicles", [])
    plates     = result.get("entities", {}).get("plates",   [])
    assoc_vp   = result.get("associations", {}).get("vehicle_plate", {})
    enf_list   = result.get("enforcements", {}).get("two_wheelers", [])
    enf_by_vid = {e["vehicle_id"]: e for e in enf_list}

    tracks = []
    for v in vehicles:
        vid      = v.get("id", "")
        track_id = v.get("track_id", vid)
        pid      = assoc_vp.get(vid)
        plate    = next((p for p in plates if p["id"] == pid), None) if pid else None
        enf      = enf_by_vid.get(vid)

        t = {
            "track_id":      track_id,
            "vehicle_class": v.get("class", "vehicle"),
            "bbox":          v.get("bbox", []),
            "confidence":    round(v.get("confidence", 0.0), 3),
            "hsrp":          plate.get("hsrp")                        if plate else None,
            "hsrp_conf":     round(plate.get("hsrp_confidence", 0.0), 3) if plate else 0.0,
            "ocr_text":      plate.get("ocr_text")                    if plate else None,
            "helmet":        None,
            "helmet_conf":   0.0,
        }
        if enf and enf.get("helmet"):
            t["helmet"]      = enf["helmet"].get("status")
            t["helmet_conf"] = round(enf["helmet"].get("confidence", 0.0), 3)
        tracks.append(t)
    return tracks


# ── /stream-view — dashboard viewer endpoint ──────────────────────────────────

async def handle_viewer_stream(websocket: WebSocket):
    """
    Dashboard viewer — paced playback from per-viewer buffer deque.

    Playback model:
      1. Initial buffering: wait until INITIAL_BUFFER_FRAMES frames received
      2. Play frames at PLAYBACK_FPS — one frame every (1/PLAYBACK_FPS) seconds
      3. Buffer empties → send buffering status, wait for refill
      4. Resume once buffer hits threshold again
    """
    ensure_broadcast_worker()
    await websocket.accept()

    state = ViewerState()
    _viewers[websocket] = state
    logger.info(f"[/stream-view] Viewer connected — total: {len(_viewers)}")

    frame_interval = 1.0 / PLAYBACK_FPS

    async def _playback_loop():
        rebuffering = False

        while True:
            # ── Initial / re-buffering ────────────────────────────────────
            if state.buffering:
                try:
                    await websocket.send_text(json.dumps({
                        "type":          "buffering",
                        "buffer_frames": len(state.buffer),
                        "buffer_target": INITIAL_BUFFER_FRAMES,
                    }))
                except Exception:
                    return
                await asyncio.sleep(0.2)
                continue

            # ── Buffer empty → re-buffer ──────────────────────────────────
            if not state.buffer:
                if not rebuffering:
                    rebuffering = True
                    try:
                        await websocket.send_text(json.dumps({
                            "type":          "buffering",
                            "buffer_frames": 0,
                            "buffer_target": INITIAL_BUFFER_FRAMES,
                        }))
                    except Exception:
                        return
                # Flip back to buffering state so the threshold check
                # in _broadcast_worker re-arms
                state.buffering = True
                await asyncio.sleep(0.05)
                continue

            rebuffering = False

            # ── Play next frame ───────────────────────────────────────────
            jpeg_bytes, meta_json = state.buffer.popleft()

            # Inject live viewer buffer depth so dashboard can show it
            try:
                meta = json.loads(meta_json)
                meta["viewer_buffer_frames"] = len(state.buffer)
                meta_json = json.dumps(meta)
            except Exception:
                pass

            try:
                await websocket.send_bytes(jpeg_bytes)
                await websocket.send_text(meta_json)
            except Exception:
                return

            await asyncio.sleep(frame_interval)

    async def _keepalive_loop():
        """Drain any client messages; send periodic pings."""
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=15)
            except asyncio.TimeoutError:
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    return
            except (WebSocketDisconnect, Exception):
                return

    try:
        playback_task  = asyncio.create_task(_playback_loop())
        keepalive_task = asyncio.create_task(_keepalive_loop())
        done, pending  = await asyncio.wait(
            [playback_task, keepalive_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for t in pending:
            t.cancel()
    except Exception as e:
        logger.warning(f"[/stream-view] viewer error: {e}")
    finally:
        _viewers.pop(websocket, None)
        logger.info(f"[/stream-view] Viewer disconnected — total: {len(_viewers)}")
