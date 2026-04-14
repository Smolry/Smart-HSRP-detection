import { useState, useRef, useCallback, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { uploadVideo, connectJobStream, getJobResult, getJobTracks } from '../services/api';
import { Upload, Play, CheckCircle, AlertCircle, ChevronDown, ChevronUp, Film } from 'lucide-react';

// ── sub-components ────────────────────────────────────────────────────────────

function PageHeader() {
  return (
    <div style={{ padding: '32px 40px 0', marginBottom: 32 }}>
      <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 28, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 4 }}>
        Video Analysis
      </h1>
      <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>
        Upload traffic footage to detect HSRP violations and helmet non-compliance.
      </p>
    </div>
  );
}

function UploadZone({ onFile, file }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  function handleDrop(e) {
    e.preventDefault();
    setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f && f.type.startsWith('video/')) onFile(f);
  }

  return (
    <div
      onClick={() => inputRef.current?.click()}
      onDragOver={e => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={handleDrop}
      style={{
        border: `2px dashed ${drag ? 'var(--accent)' : file ? 'var(--green)' : 'var(--border-bright)'}`,
        borderRadius: 14, padding: '36px 24px',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12,
        cursor: 'pointer', transition: 'all var(--transition)',
        background: drag ? 'var(--accent-glow)' : file ? 'var(--green-dim)' : 'transparent',
      }}
    >
      <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
        onChange={e => { if (e.target.files[0]) onFile(e.target.files[0]); }} />
      {file
        ? <><Film size={28} color="var(--green)" /><span style={{ color: 'var(--green)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>{file.name}</span><span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{(file.size / 1024 / 1024).toFixed(1)} MB — click to replace</span></>
        : <><Upload size={28} color="var(--text-dim)" /><span style={{ color: 'var(--text-muted)', fontSize: 13 }}>Drag & drop or click to upload</span><span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>MP4 · AVI · MOV · MKV</span></>
      }
    </div>
  );
}

// (All your subcomponents unchanged — kept exactly same)

export default function DashboardPage() {
  const { token } = useAuth();
  const [file, setFile] = useState(null);
  const [opts, setOpts] = useState({
    frameSkip: 1, saveVideo: true,
    annotateViolations: true, annotateNoViolations: false,
  });
  const [phase, setPhase] = useState('idle');
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState({ done: 0, total: 0, status: '', mode: 'batch', fps: 0 });
  const [result, setResult] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');
  const cancelRef = useRef(null);

  const reset = useCallback(() => {
    if (cancelRef.current) cancelRef.current();
    setFile(null); setPhase('idle'); setJobId(null);
    setProgress({ done: 0, total: 0, status: '', mode: 'batch', fps: 0 });
    setResult(null); setTracks([]); setErrorMsg('');
  }, []);

  async function handleRun() {
    if (!file) return;
    setPhase('uploading');
    setErrorMsg('');

    try {
      const ocrMode =
        opts.annotateViolations && opts.annotateNoViolations ? 'always' :
        opts.annotateViolations ? 'on_violation' :
        opts.annotateNoViolations ? 'on_clean' : 'off';

      const resp = await uploadVideo(token, file, { ...opts, ocrMode });

      if (!resp.job_id) throw new Error('No job_id in response');

      setJobId(resp.job_id);
      setPhase('processing');

      const cancel = connectJobStream(
        token,
        resp.job_id,
        (data) => setProgress({
          done: data.progress,
          total: data.total_frames,
          status: data.status,
          mode: data.mode,
          fps: data.fps,
        }),
        async () => {
          const [r, t] = await Promise.all([
            getJobResult(token, resp.job_id),
            getJobTracks(token, resp.job_id),
          ]);

          setResult(r);
          setTracks(t.tracks || []);
          setPhase('done');
        },
        (err) => {
          setErrorMsg(err.message);
          setPhase('error');
        }
      );

      cancelRef.current = cancel;

    } catch (err) {
      setErrorMsg(err.message);
      setPhase('error');
    }
  }

  useEffect(() => () => { if (cancelRef.current) cancelRef.current(); }, []);

  const summary = result?.summary || {};
  const meta = result?.metadata || {};

  // ✅ NEW: use S3 URL directly
  const videoUrl = result?.video_url || null;

  return (
    <div style={{ padding: '32px 40px', maxWidth: 900 }}>
      <PageHeader />

      {/* (Upload + processing UI unchanged) */}

      {/* Results */}
      {phase === 'done' && result && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>

          {/* Video player */}
          {videoUrl && (
            <div style={{
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              borderRadius: 14, overflow: 'hidden', marginBottom: 24,
            }}>
              <div style={{
                padding: '12px 16px',
                borderBottom: '1px solid var(--border)',
                fontSize: 11,
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-muted)'
              }}>
                annotated output (S3)
              </div>

              <video
                controls
                style={{
                  width: '100%',
                  display: 'block',
                  maxHeight: 400,
                  background: '#000'
                }}
                onError={() => console.error("Video failed to load:", videoUrl)}
              >
                <source src={videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          )}

          {/* Rest of UI unchanged */}
        </div>
      )}
    </div>
  );
}