const REST_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const WS_BASE   = REST_BASE.replace(/^http/, 'ws');

// ── helpers ───────────────────────────────────────────────────────────────────

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

// ── Job stream — WS for progress, REST polling as fallback ───────────────────
//
// Strategy:
//   1. Open WS to get live frame progress updates
//   2. Simultaneously poll REST /job-result every 3s
//   3. Whichever signals completion first wins — the other is cancelled
//   4. If WS drops (1006 etc) polling takes over seamlessly
//
// This means 1006 is completely harmless — polling catches the completion.

export function connectJobStream(token, jobId, onUpdate, onDone, onError) {
  let done      = false;
  let ws        = null;
  let pollTimer = null;

  function finish() {
    if (done) return;
    done = true;
    clearTimeout(pollTimer);
    if (ws) {
      ws.onmessage = null;
      ws.onerror   = null;
      ws.onclose   = null;
      try { ws.close(); } catch {}
    }
    onDone();
  }

  function fail(err) {
    if (done) return;
    done = true;
    clearTimeout(pollTimer);
    if (ws) {
      ws.onmessage = null;
      ws.onerror   = null;
      ws.onclose   = null;
      try { ws.close(); } catch {}
    }
    onError(err);
  }

  // ── REST polling ────────────────────────────────────────────────────────────
  // Runs independently of WS — if WS dies, this still catches completion.
  async function poll() {
    if (done) return;
    try {
      const resp = await fetch(`${REST_BASE}/api/job-result/${jobId}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      console.log(`[poll] status=${resp.status} done=${done}`);
      if (resp.ok) {
        finish();
        return;
      }
      if (resp.status === 500) {
        const body = await resp.json().catch(() => ({}));
        fail(new Error(body?.detail?.error || 'Job failed'));
        return;
      }
      // 400 = not completed yet, 404 = not found — keep polling
    } catch (e) {
      console.warn('[poll] fetch error:', e);
    }
    if (!done) {
      pollTimer = setTimeout(poll, 3000);
    }
  }

  // Start polling after a short delay (give pipeline time to start)
  pollTimer = setTimeout(poll, 4000);

  // ── WebSocket — live progress only ──────────────────────────────────────────
  try {
    ws = new WebSocket(`${WS_BASE}/api/ws/${jobId}`);

    ws.onmessage = (event) => {
      if (done) return;
      let data;
      try { data = JSON.parse(event.data); } catch { return; }

      if (data.status === 'failed') {
        fail(new Error(data.error || 'Job failed'));
        return;
      }
      if (data.status === 'completed') {
        finish();
        return;
      }
      if (data.status === 'running') {
        onUpdate({
          progress:     data.progress     ?? 0,
          total_frames: data.total        ?? 0,
          fps:          data.fps          ?? 0,
        });
      }
    };

    ws.onerror = () => {
      // WS error — polling will handle completion, no need to do anything
      console.warn('[WS] error — falling back to polling');
    };

    ws.onclose = (e) => {
      // Any WS close (including 1006) is fine — polling takes over
      if (e.code !== 1000) {
        console.warn(`[WS] closed with code ${e.code} — polling active`);
      }
    };

  } catch {
    // WebSocket not available — polling only
    console.warn('[WS] could not open — polling only');
  }

  // Return cancel function
  return () => {
    done = true;
    clearTimeout(pollTimer);
    if (ws) {
      ws.onmessage = null;
      ws.onerror   = null;
      ws.onclose   = null;
      try { ws.close(); } catch {}
    }
  };
}

// ── Job result / tracks ───────────────────────────────────────────────────────

async function fetchWithRetry(fn, retries = 5, delay = 800) {
  for (let i = 0; i < retries; i++) {
    try { return await fn(); } catch (err) {
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
