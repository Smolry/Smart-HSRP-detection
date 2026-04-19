/**
 * api.js — unified API service
 */

const REST_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE   = REST_BASE.replace(/^http/, 'ws');

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
  form.append('file',                   file);
  form.append('frame_skip',             String(options.frameSkip            ?? 1));
  form.append('save_output_video',      String(options.saveVideo            ?? true));
  form.append('annotate_violations',    String(options.annotateViolations   ?? true));
  form.append('annotate_no_violations', String(options.annotateNoViolations ?? false));
  form.append('ocr_mode',               options.ocrMode ?? 'on_violation');

  const resp = await fetch(`${REST_BASE}/api/process-video`, {
    method:  'POST',
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body:    form,
  });
  if (!resp.ok) throw new Error(`Upload failed: ${resp.status}`);
  return resp.json();
}

// ── WebSocket job stream ──────────────────────────────────────────────────────

/**
 * connectJobStream — opens a WebSocket to /api/ws/{jobId}.
 *
 * onUpdate(data)  — called on each 'running' progress message
 * onDone()        — called once when status === 'completed'; REST fetch happens in caller
 * onError(err)    — called on failure
 *
 * Returns a cancel() function.
 */
export function connectJobStream(token, jobId, onUpdate, onDone, onError) {
  const url = `${WS_BASE}/api/ws/${jobId}`;
  const ws  = new WebSocket(url);

  let done = false;

  ws.onopen = () => console.log('[WS] connected:', url);

  ws.onmessage = (event) => {
    if (done) return;
    let data;
    try { data = JSON.parse(event.data); } catch { return; }

    const status = data.status;

    if (status === 'failed') {
      done = true;
      onError(new Error(data.error || 'Job failed'));
      ws.close(1000);
      return;
    }

    if (status === 'completed') {
      done = true;
      ws.close(1000);
      onDone();   // caller does REST fetches
      return;
    }

    if (status === 'running') {
      onUpdate({
        progress:     data.progress     ?? 0,
        total_frames: data.total        ?? 0,
        fps:          data.fps          ?? 0,
      });
    }
  };

  ws.onerror = (e) => {
    if (done) return;
    console.error('[WS] error', e);
    onError(new Error('WebSocket connection error'));
  };

  ws.onclose = (event) => {
    if (done) return;
    // 1000 = normal close, 1005 = no status (also normal)
    if (event.code !== 1000 && event.code !== 1005) {
      onError(new Error(`WebSocket closed unexpectedly (code ${event.code})`));
    }
  };

  return () => {
    done = true;
    ws.onmessage = null;
    ws.onerror   = null;
    ws.onclose   = null;
    ws.close();
  };
}

// ── Job result / tracks ───────────────────────────────────────────────────────

/**
 * Retry up to `retries` times with `delay` ms between attempts.
 * Needed because Redis write and REST read can race right after WS completion.
 */
async function fetchWithRetry(fn, retries = 5, delay = 800) {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (err) {
      if (i === retries - 1) throw err;
      await new Promise(r => setTimeout(r, delay));
    }
  }
}

export async function getJobResult(token, jobId) {
  return fetchWithRetry(() => apiFetch(`/api/job-result/${jobId}`, {}, token));
}

export async function getJobTracks(token, jobId) {
  return fetchWithRetry(() => apiFetch(`/api/job-tracks/${jobId}`, {}, token));
}

// Returns the URL to stream/download the annotated video directly
export function getVideoUrl(jobId) {
  return `${REST_BASE}/api/job-video/${jobId}`;
}

// ── Violations ────────────────────────────────────────────────────────────────

export async function getViolations(token, params = {}) {
  const qs = new URLSearchParams();
  if (params.limit)               qs.set('limit',          params.limit);
  if (params.offset)              qs.set('offset',         params.offset);
  if (params.violationType)       qs.set('violation_type', params.violationType);
  if (params.needsReview != null) qs.set('needs_review',   params.needsReview);
  if (params.minQuality)          qs.set('min_quality',    params.minQuality);
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
