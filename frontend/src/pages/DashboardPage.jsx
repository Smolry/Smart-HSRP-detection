import { useState, useRef, useCallback, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import { uploadVideo, connectJobStream, getJobResult, getJobTracks, getVideoUrl } from '../services/api';
import {
  Upload, Play, CheckCircle, AlertCircle, ChevronDown, ChevronUp,
  Film, Download, RefreshCw, Eye, Shield, ShieldOff, HardHat,
  HardHatIcon, Zap, BarChart2, Layers
} from 'lucide-react';

// ── Design tokens (extend existing CSS vars) ──────────────────────────────────
const T = {
  green:     'var(--green,    #22c55e)',
  greenDim:  'var(--green-dim, rgba(34,197,94,0.08))',
  red:       'var(--red,      #ef4444)',
  redDim:    'var(--red-dim,  rgba(239,68,68,0.08))',
  amber:     '#f59e0b',
  amberDim:  'rgba(245,158,11,0.10)',
  accent:    'var(--accent,   #7c3aed)',
  accentGlow:'var(--accent-glow, rgba(124,58,237,0.12))',
  card:      'var(--bg-card,  #111118)',
  border:    'var(--border,   rgba(255,255,255,0.08))',
  borderBrt: 'var(--border-bright, rgba(255,255,255,0.14))',
  textMuted: 'var(--text-muted, #888)',
  textDim:   'var(--text-dim,  #555)',
  mono:      'var(--font-mono, monospace)',
  display:   'var(--font-display, inherit)',
  trans:     'var(--transition, 0.18s ease)',
};

// ── Shared primitives ─────────────────────────────────────────────────────────

function Card({ children, style = {}, noPad = false }) {
  return (
    <div style={{
      background: T.card,
      border: `1px solid ${T.border}`,
      borderRadius: 14,
      padding: noPad ? 0 : 24,
      marginBottom: 20,
      ...style,
    }}>
      {children}
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <div style={{
      fontSize: 10, fontFamily: T.mono, letterSpacing: '0.12em',
      color: T.textDim, textTransform: 'uppercase', marginBottom: 14,
    }}>
      {children}
    </div>
  );
}

function Pill({ color, bg, children }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px', borderRadius: 20,
      fontSize: 11, fontFamily: T.mono,
      color: color, background: bg,
    }}>
      {children}
    </span>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{
      background: T.card, border: `1px solid ${T.border}`,
      borderRadius: 12, padding: '16px 20px',
      borderTop: `2px solid ${accent || T.accent}`,
    }}>
      <div style={{ fontSize: 11, color: T.textDim, fontFamily: T.mono, marginBottom: 6 }}>{label}</div>
      <div style={{ fontSize: 28, fontWeight: 800, fontFamily: T.display, lineHeight: 1 }}>{value ?? '—'}</div>
      {sub && <div style={{ fontSize: 11, color: T.textMuted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ── Upload zone ───────────────────────────────────────────────────────────────

function UploadZone({ onFile, file }) {
  const [drag, setDrag] = useState(false);
  const inputRef = useRef();

  function handleDrop(e) {
    e.preventDefault(); setDrag(false);
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
        border: `2px dashed ${drag ? T.accent : file ? T.green : T.borderBrt}`,
        borderRadius: 12, padding: '32px 20px',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10,
        cursor: 'pointer', transition: `all ${T.trans}`,
        background: drag ? T.accentGlow : file ? T.greenDim : 'transparent',
      }}
    >
      <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
        onChange={e => { if (e.target.files[0]) onFile(e.target.files[0]); }} />
      {file ? (
        <>
          <Film size={26} color={T.green} />
          <span style={{ color: T.green, fontSize: 13, fontFamily: T.mono }}>{file.name}</span>
          <span style={{ fontSize: 11, color: T.textMuted }}>{(file.size / 1024 / 1024).toFixed(1)} MB · click to replace</span>
        </>
      ) : (
        <>
          <Upload size={26} color={T.textDim} />
          <span style={{ color: T.textMuted, fontSize: 13 }}>Drag & drop or click to upload</span>
          <span style={{ fontSize: 11, color: T.textDim, fontFamily: T.mono }}>MP4 · AVI · MOV · MKV</span>
        </>
      )}
    </div>
  );
}

// ── Options panel ─────────────────────────────────────────────────────────────

function OptionsPanel({ opts, onChange }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ border: `1px solid ${T.border}`, borderRadius: 10, overflow: 'hidden', marginTop: 14 }}>
      <button onClick={() => setOpen(v => !v)} style={{
        width: '100%', padding: '11px 16px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'rgba(255,255,255,0.02)', border: 'none', cursor: 'pointer',
        fontSize: 11, fontFamily: T.mono, color: T.textMuted, letterSpacing: '0.08em',
      }}>
        PROCESSING OPTIONS {open ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
      </button>
      {open && (
        <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <div style={{ fontSize: 11, color: T.textMuted, marginBottom: 4 }}>
              Frame skip: <span style={{ fontFamily: T.mono, color: '#fff' }}>{opts.frameSkip}</span>
            </div>
            <input type="range" min={1} max={5} value={opts.frameSkip}
              onChange={e => onChange({ ...opts, frameSkip: Number(e.target.value) })}
              style={{ width: '100%' }} />
          </div>
          {[
            ['saveVideo',           'Save annotated output video'],
            ['annotateViolations',  'Annotate violation frames'],
            ['annotateNoViolations','Annotate clean frames'],
          ].map(([key, label]) => (
            <label key={key} style={{ display: 'flex', gap: 8, cursor: 'pointer', fontSize: 12, color: T.textMuted, alignItems: 'center' }}>
              <input type="checkbox" checked={opts[key]}
                onChange={e => onChange({ ...opts, [key]: e.target.checked })} />
              {label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Progress bar ──────────────────────────────────────────────────────────────

function ProgressCard({ phase, progress }) {
  const isUploading = phase === 'uploading';
  const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <Card>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
        <span style={{ fontSize: 13, color: T.textMuted }}>
          {isUploading ? 'Uploading video…' : `Processing frames…`}
        </span>
        {!isUploading && (
          <span style={{ fontSize: 11, fontFamily: T.mono, color: T.textDim }}>
            {progress.done} / {progress.total || '?'}{progress.fps > 0 ? ` · ${progress.fps} fps` : ''}
          </span>
        )}
      </div>
      <div style={{ height: 5, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
        <div style={{
          height: '100%',
          width: isUploading ? '100%' : `${pct || 4}%`,
          background: `linear-gradient(90deg, ${T.accent}, #a855f7)`,
          borderRadius: 4,
          transition: 'width 0.4s ease',
          animation: isUploading ? 'pulse 1.5s ease infinite' : 'none',
        }} />
      </div>
      {!isUploading && progress.total > 0 && (
        <div style={{ marginTop: 5, fontSize: 10, color: T.textDim, textAlign: 'right', fontFamily: T.mono }}>
          {pct}%
        </div>
      )}
      <style>{`@keyframes pulse { 0%,100%{opacity:0.7} 50%{opacity:1} }`}</style>
    </Card>
  );
}

// ── Annotated video player ────────────────────────────────────────────────────

function VideoPlayer({ jobId }) {
  const url = getVideoUrl(jobId);
  const [err, setErr] = useState(false);

  return (
    <Card noPad style={{ overflow: 'hidden' }}>
      <div style={{
        padding: '10px 16px', borderBottom: `1px solid ${T.border}`,
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Film size={14} color={T.accent} />
          <span style={{ fontSize: 11, fontFamily: T.mono, color: T.textMuted, letterSpacing: '0.08em' }}>
            ANNOTATED OUTPUT
          </span>
        </div>
        <div style={{ display: 'flex', gap: 14, fontSize: 10, fontFamily: T.mono, color: T.textDim }}>
          <span><span style={{ color: '#6b7280' }}>⬜</span> tracking</span>
          <span><span style={{ color: T.amber }}>🟡</span> predicted</span>
          <span><span style={{ color: T.red }}>🔴</span> confirmed</span>
        </div>
      </div>

      {err ? (
        <div style={{ padding: 32, textAlign: 'center', color: T.textDim, fontSize: 13 }}>
          Video not available yet. Try refreshing.
        </div>
      ) : (
        <video
          controls
          style={{ width: '100%', display: 'block', maxHeight: 420, background: '#000' }}
          onError={() => setErr(true)}
        >
          <source src={url} type="video/mp4" />
        </video>
      )}
    </Card>
  );
}

// ── Summary stat strip ────────────────────────────────────────────────────────

function SummaryStrip({ summary, meta }) {
  const cards = [
    { label: 'FRAMES PROCESSED', value: meta?.total_frames_processed ?? '—', accent: T.accent },
    { label: 'AVG FPS',          value: meta?.avg_fps              ?? '—', accent: '#6366f1' },
    { label: 'VIOLATIONS FOUND', value: summary?.total             ?? 0,   accent: T.red    },
    { label: 'NEEDS REVIEW',     value: summary?.needs_review      ?? 0,   accent: T.amber  },
  ];
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 20 }}>
      {cards.map(c => <StatCard key={c.label} {...c} />)}
    </div>
  );
}

// ── Track table ───────────────────────────────────────────────────────────────

function fmtViolation(v) {
  if (!v) return null;
  return v.replace(/_/g, ' ').replace('non hsrp plate', 'Non-HSRP Plate').replace(/\b\w/g, c => c.toUpperCase());
}

function TrackRow({ t, idx }) {
  const vclass       = (t.vehicle_class || 'unknown').toLowerCase();
  const isTwoWheeler = ['motorcycle', 'bicycle', 'bike'].includes(vclass);

  const hsrp = (() => {
    const raw = t.hsrp_label || t.hsrp;
    if (raw === 'hsrp') return { label: 'HSRP', color: T.green, bg: T.greenDim };
    if (raw === 'non_hsrp' || raw === 'non hsrp' || t.violation_type === 'non_hsrp_plate')
      return { label: 'Non-HSRP', color: T.red, bg: T.redDim };
    return null;
  })();

  const helmet = (() => {
    if (!isTwoWheeler) return null;
    const raw = t.helmet_status || t.helmet;
    if (raw === 'HELMET') return { label: 'Helmet', color: T.green, bg: T.greenDim };
    if (raw === 'NO_HELMET' || t.violation_type === 'no_helmet')
      return { label: 'No Helmet', color: T.red, bg: T.redDim };
    if (raw === 'UNCERTAIN') return { label: 'Uncertain', color: T.amber, bg: T.amberDim };
    return null;
  })();

  const violation = fmtViolation(t.violation_type);
  const stored    = t.should_store;
  const review    = t.needs_review;

  const rowBg = stored && review ? T.amberDim
              : stored           ? T.redDim
              : 'transparent';

  return (
    <tr style={{ background: rowBg, borderBottom: `1px solid ${T.border}` }}>
      <td style={td}><span style={{ fontFamily: T.mono, fontSize: 11 }}>#{t.track_id ?? idx}</span></td>
      <td style={td}>{vclass.charAt(0).toUpperCase() + vclass.slice(1)}</td>
      <td style={td}><span style={{ fontFamily: T.mono, fontSize: 11 }}>{t.plate_number || t.vehicle_number || '—'}</span></td>
      <td style={td}>{hsrp ? <Pill color={hsrp.color} bg={hsrp.bg}>{hsrp.label}</Pill> : <span style={{ color: T.textDim }}>—</span>}</td>
      <td style={td}>{isTwoWheeler ? (helmet ? <Pill color={helmet.color} bg={helmet.bg}>{helmet.label}</Pill> : <span style={{ color: T.textDim }}>—</span>) : <span style={{ color: T.textDim }}>—</span>}</td>
      <td style={td}>
        {violation
          ? <Pill color={T.red} bg={T.redDim}>{violation}</Pill>
          : <Pill color={T.green} bg={T.greenDim}>✓ Clean</Pill>}
      </td>
      <td style={{ ...td, fontFamily: T.mono, fontSize: 11 }}>{(t.violation_confidence || t.quality_score || 0).toFixed(2)}</td>
      <td style={{ ...td, fontFamily: T.mono, fontSize: 11 }}>{(t.quality_score || 0).toFixed(2)}</td>
      <td style={td}>
        {stored
          ? (review
            ? <Pill color={T.amber} bg={T.amberDim}>⚠ Review</Pill>
            : <Pill color={T.red}   bg={T.redDim}>Stored</Pill>)
          : <span style={{ color: T.textDim }}>—</span>}
      </td>
      <td style={{ ...td, fontFamily: T.mono, fontSize: 10, color: T.textDim }}>
        {t.first_frame ?? 0}–{t.last_frame ?? 0}
      </td>
    </tr>
  );
}

const td = {
  padding: '10px 14px', fontSize: 12, verticalAlign: 'middle',
  color: 'var(--text, #e5e5e5)',
};

const th = {
  ...td, fontSize: 10, fontFamily: 'var(--font-mono, monospace)',
  color: 'var(--text-dim, #555)', letterSpacing: '0.1em',
  textTransform: 'uppercase', textAlign: 'left',
  borderBottom: '1px solid var(--border, rgba(255,255,255,0.08))',
  background: 'rgba(255,255,255,0.02)',
  fontWeight: 500,
};

function TrackTable({ tracks }) {
  if (!tracks?.length) return (
    <Card>
      <div style={{ textAlign: 'center', color: T.textDim, fontSize: 13, padding: '20px 0' }}>
        No tracks found in this video.
      </div>
    </Card>
  );

  return (
    <Card noPad style={{ overflow: 'hidden' }}>
      <div style={{ padding: '14px 16px', borderBottom: `1px solid ${T.border}`, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Layers size={14} color={T.accent} />
        <span style={{ fontSize: 11, fontFamily: T.mono, color: T.textMuted, letterSpacing: '0.08em' }}>
          TRACK-LEVEL RESULTS — {tracks.length} vehicles
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>
              {['Track', 'Vehicle', 'Plate', 'HSRP', 'Helmet', 'Violation', 'Conf.', 'Quality', 'Status', 'Frames'].map(h => (
                <th key={h} style={th}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {tracks.map((t, i) => <TrackRow key={t.track_id ?? i} t={t} idx={i} />)}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ── Violation breakdown ───────────────────────────────────────────────────────

function ViolationBreakdown({ byType }) {
  if (!byType || !Object.keys(byType).length) return null;
  return (
    <Card>
      <SectionLabel><BarChart2 size={11} style={{ verticalAlign: 'middle', marginRight: 6 }} />Violation Breakdown</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {Object.entries(byType).map(([k, v]) => {
          const pct = Math.min(100, Math.round((v.count / Math.max(...Object.values(byType).map(x => x.count))) * 100));
          return (
            <div key={k}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 4 }}>
                <span style={{ color: T.textMuted }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                <span style={{ fontFamily: T.mono, fontSize: 11 }}>
                  {v.count} · <span style={{ color: T.textDim }}>avg conf {v.avg_conf?.toFixed(2)}</span>
                </span>
              </div>
              <div style={{ height: 4, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${pct}%`, background: T.red, borderRadius: 4, transition: 'width 0.5s ease' }} />
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

// ── JSON download ─────────────────────────────────────────────────────────────

function DownloadReport({ result, jobId }) {
  function download() {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href = url; a.download = `report_${jobId}.json`; a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <button onClick={download} style={{
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '9px 16px', borderRadius: 8,
      background: 'rgba(255,255,255,0.04)', border: `1px solid ${T.border}`,
      color: T.textMuted, cursor: 'pointer', fontSize: 12, fontFamily: T.mono,
      transition: `all ${T.trans}`,
      marginBottom: 20,
    }}>
      <Download size={13} /> Download Full JSON Report
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { token }   = useAuth();
  const [file, setFile]     = useState(null);
  const [opts, setOpts]     = useState({ frameSkip: 1, saveVideo: true, annotateViolations: true, annotateNoViolations: false });
  const [phase, setPhase]   = useState('idle');      // idle | uploading | processing | done | error
  const [jobId, setJobId]   = useState(null);
  const [progress, setProgress] = useState({ done: 0, total: 0, fps: 0 });
  const [result, setResult] = useState(null);
  const [tracks, setTracks] = useState([]);
  const [errorMsg, setErrorMsg] = useState('');
  const cancelRef = useRef(null);

  const reset = useCallback(() => {
    cancelRef.current?.();
    setFile(null); setPhase('idle'); setJobId(null);
    setProgress({ done: 0, total: 0, fps: 0 });
    setResult(null); setTracks([]); setErrorMsg('');
  }, []);

  useEffect(() => () => cancelRef.current?.(), []);

  async function handleRun() {
    if (!file) return;
    setPhase('uploading'); setErrorMsg('');

    try {
      const ocrMode =
        opts.annotateViolations && opts.annotateNoViolations ? 'always' :
        opts.annotateViolations ? 'on_violation' :
        opts.annotateNoViolations ? 'on_clean' : 'off';

      const resp = await uploadVideo(token, file, { ...opts, ocrMode });
      if (!resp.job_id) throw new Error('No job_id returned');

      setJobId(resp.job_id);
      setPhase('processing');

      cancelRef.current = connectJobStream(
        token, resp.job_id,
        (data) => setProgress({ done: data.progress, total: data.total_frames, fps: data.fps }),
        async () => {
          const [r, t] = await Promise.all([
            getJobResult(token, resp.job_id),
            getJobTracks(token, resp.job_id),
          ]);
          setResult(r); setTracks(t.tracks || []); setPhase('done');
        },
        (err) => { setErrorMsg(err.message); setPhase('error'); }
      );
    } catch (err) {
      setErrorMsg(err.message); setPhase('error');
    }
  }

  const summary = result?.summary  || {};
  const meta    = result?.metadata || {};

  return (
    <div style={{ padding: '32px 40px', maxWidth: 980 }}>

      {/* Header */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontFamily: T.display, fontSize: 26, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 4 }}>
          Video Analysis
        </h1>
        <p style={{ color: T.textMuted, fontSize: 13 }}>
          Upload traffic footage to detect HSRP violations and helmet non-compliance.
        </p>
      </div>

      {/* ── Upload / error card ── */}
      {(phase === 'idle' || phase === 'error') && (
        <Card style={{ animation: 'fadeIn 0.25s ease' }}>
          <UploadZone onFile={setFile} file={file} />
          <OptionsPanel opts={opts} onChange={setOpts} />

          {errorMsg && (
            <div style={{
              display: 'flex', gap: 10, marginTop: 14, padding: '12px 14px',
              background: T.redDim, border: `1px solid rgba(239,68,68,0.2)`,
              borderRadius: 8, color: T.red, fontSize: 12,
            }}>
              <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
              {errorMsg}
            </div>
          )}

          <button
            onClick={handleRun} disabled={!file}
            style={{
              marginTop: 18, width: '100%', padding: '13px',
              background: file ? T.accent : 'rgba(255,255,255,0.04)',
              color: file ? '#fff' : T.textDim,
              border: 'none', borderRadius: 10,
              cursor: file ? 'pointer' : 'not-allowed',
              fontFamily: T.display, fontSize: 14, fontWeight: 700,
              display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
              transition: `all ${T.trans}`,
            }}
          >
            <Play size={15} /> Run Detection Pipeline
          </button>
        </Card>
      )}

      {/* ── Progress ── */}
      {(phase === 'uploading' || phase === 'processing') && (
        <ProgressCard phase={phase} progress={progress} />
      )}

      {/* ── Results ── */}
      {phase === 'done' && result && (
        <div style={{ animation: 'fadeIn 0.3s ease' }}>

          {/* Reset button */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <CheckCircle size={16} color={T.green} />
              <span style={{ fontSize: 13, color: T.green, fontFamily: T.mono }}>PROCESSING COMPLETE</span>
            </div>
            <button onClick={reset} style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '7px 14px', borderRadius: 8,
              background: 'transparent', border: `1px solid ${T.border}`,
              color: T.textMuted, cursor: 'pointer', fontSize: 12,
            }}>
              <RefreshCw size={12} /> New Video
            </button>
          </div>

          {/* Stat strip */}
          <SummaryStrip summary={summary} meta={meta} />

          {/* Annotated video */}
          {opts.saveVideo && <VideoPlayer jobId={jobId} />}

          {/* Download report */}
          <DownloadReport result={result} jobId={jobId} />

          {/* Violation breakdown */}
          <ViolationBreakdown byType={summary?.by_type} />

          {/* Track table */}
          <TrackTable tracks={tracks} />
        </div>
      )}

    </div>
  );
}