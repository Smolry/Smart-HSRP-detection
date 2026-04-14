/**
 * api.js — unified API service
 *
 * REST:       auth, violations, thresholds, job result/tracks  → FastAPI :8000
 * WebSocket:  job progress stream                              → FastAPI :8000/api/ws/job/:id
 */

const REST_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// WebSocket base — same host, swap http(s) → ws(s)
const WS_BASE = REST_BASE.replace(/^http/, 'ws');

// ── helpers ──────────────────────────────────────────────────────────────────

async function apiFetch(path, opts = {}, token = null) {
  const headers = {
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...opts.headers,
  };
  const resp = await fetch(`${REST_BASE}${path}`, { ...opts, headers });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${resp.status}`);
  }
  return resp.json();
}

// ── Video upload ──────────────────────────────────────────────────────────────

export async function uploadVideo(token, file, options = {}) {
  const form = new FormData();
  form.append('file', file);
  form.append('frame_skip',             String(options.frameSkip          ?? 1));
  form.append('save_output_video',      String(options.saveVideo          ?? true));
  form.append('annotate_violations',    String(options.annotateViolations ?? true));
  form.append('annotate_no_violations', String(options.annotateNoViolations ?? false));
  form.append('ocr_mode',               options.ocrMode ?? 'on_violation');
  form.append('mode',                   options.mode    ?? 'batch');

  const resp = await fetch(`${REST_BASE}/api/process-video`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
  return resp.json();
}

// ── WebSocket job stream ──────────────────────────────────────────────────────

/**
 * connectJobStream — replaces pollJobProgress entirely.
 *
 * Opens a WebSocket to /api/ws/job/:jobId.
 * The server sends a message on every progress update — no polling.
 *
 * onUpdate(data)  — called on every message while status === 'running'
 * onDone(data)    — called once when status === 'completed'
 * onError(err)    — called on WS error or status === 'failed'
 *
 * Returns a cancel() function that closes the socket.
 */
export function connectJobStream(token, jobId, onUpdate, onDone, onError) {
  // Pass token as a query param — WebSocket API has no custom headers
  const url = `${WS_BASE}/api/ws/job/${jobId}?token=${encodeURIComponent(token)}`;
  const ws = new WebSocket(url);

  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    if (data.error && !data.status) {
      onError(new Error(data.error));
      ws.close();
      return;
    }

    const status = data.status;

    if (status === 'failed') {
      onError(new Error(data.error || 'Job failed'));
      ws.close();
      return;
    }

    if (status === 'completed') {
      onDone(data);
      ws.close();
      return;
    }

    // status === 'running'
    onUpdate({
      progress:     data.progress     ?? 0,
      total_frames: data.total_frames ?? 0,
      status:       data.status,
      mode:         data.mode         ?? 'batch',
      fps:          data.fps          ?? 0,
    });
  };

  ws.onerror = () => {
    onError(new Error('WebSocket connection error'));
  };

  ws.onclose = (event) => {
    // Abnormal close that wasn't already handled above
    if (event.code !== 1000 && event.code !== 1005) {
      onError(new Error(`WebSocket closed unexpectedly (code ${event.code})`));
    }
  };

  return () => {
    ws.onmessage = null;
    ws.onerror   = null;
    ws.onclose   = null;
    ws.close();
  };
}

// ── Job result / tracks (fetched via REST after WS signals completion) ────────

export async function getJobResult(token, jobId) {
  return apiFetch(`/api/job-result/${jobId}`, {}, token);
}

export async function getJobTracks(token, jobId) {
  return apiFetch(`/api/job-tracks/${jobId}`, {}, token);
}

export function getVideoUrl(jobId) {
  return `${REST_BASE}/api/job-video/${jobId}`;
}

// ── Violations ────────────────────────────────────────────────────────────────

export async function getViolations(token, params = {}) {
  const qs = new URLSearchParams();
  if (params.limit)              qs.set('limit',          params.limit);
  if (params.offset)             qs.set('offset',         params.offset);
  if (params.violationType)      qs.set('violation_type', params.violationType);
  if (params.needsReview != null) qs.set('needs_review',  params.needsReview);
  if (params.minQuality)         qs.set('min_quality',    params.minQuality);
  return apiFetch(`/api/violations?${qs}`, {}, token);
}

// ── Thresholds ────────────────────────────────────────────────────────────────

export async function getThresholds(token) {
  return apiFetch('/api/thresholds', {}, token);
}

export async function resetThresholds(token) {
  return apiFetch('/api/thresholds/reset', { method: 'POST' }, token);
}

// ── Users (admin) ─────────────────────────────────────────────────────────────

export async function getUsers(token) {
  return apiFetch('/users', { headers: { Authorization: `Bearer ${token}` } });
}