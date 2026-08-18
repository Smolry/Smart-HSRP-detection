import { useState, useRef, useCallback, useEffect } from 'react';
import { useAuth } from '../hooks/useAuth';
import {
  uploadVideo, processExistingVideo, listInputVideos,
  connectJobStream, getJobResult, getJobTracks,
} from '../services/api';
import {
  Upload, Play, CheckCircle, AlertCircle, ChevronDown, ChevronUp,
  Film, Download, RefreshCw, BarChart2, Layers, HardDrive,
  Clock, Zap, Database, Eye, X,
} from 'lucide-react';

// ── Design tokens ─────────────────────────────────────────────────────────────
const T = {
  accent:    'var(--accent)',
  accentDim: 'var(--accent-dim)',
  accentHi:  'var(--accent-hi)',
  red:       'var(--red)',     redDim:   'var(--red-dim)',
  green:     'var(--green)',   greenDim: 'var(--green-dim)',
  amber:     'var(--amber)',   amberDim: 'var(--amber-dim)',
  blue:      'var(--blue)',    blueDim:  'var(--blue-dim)',
  card:      'var(--bg-card)', elevated: 'var(--bg-elevated)',
  border:    'var(--border)',  borderMed:'var(--border-med)',
  text:      'var(--text)',    muted:    'var(--text-muted)', dim: 'var(--text-dim)',
  mono:      'var(--font-mono)', display: 'var(--font-display)',
  trans:     'var(--transition)',
};

const REST_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// ── Primitives ────────────────────────────────────────────────────────────────

function Card({ children, style = {}, noPad = false, accent = false }) {
  return (
    <div style={{
      background: T.card,
      border: `1px solid ${accent ? 'rgba(255,107,26,0.25)' : T.border}`,
      borderRadius: 'var(--radius-lg)',
      padding: noPad ? 0 : 22,
      marginBottom: 16,
      position: 'relative',
      overflow: 'hidden',
      ...(accent ? { borderTop: `2px solid ${T.accent}` } : {}),
      ...style,
    }}>
      {children}
    </div>
  );
}

function SectionLabel({ icon: Icon, children }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
      {Icon && <Icon size={13} color={T.accent} />}
      <span style={{
        fontSize: 10, fontFamily: T.mono, color: T.muted,
        letterSpacing: '0.14em', textTransform: 'uppercase',
      }}>{children}</span>
    </div>
  );
}

function Pill({ color, bg, children }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px', borderRadius: 4,
      fontSize: 10, fontFamily: T.mono, color, background: bg,
    }}>{children}</span>
  );
}

function StatCard({ label, value, accent, sub }) {
  return (
    <div style={{
      background: T.elevated, border: `1px solid ${T.border}`,
      borderRadius: 'var(--radius-lg)', padding: '16px 18px',
      borderLeft: `3px solid ${accent || T.accent}`,
    }}>
      <div style={{ fontSize: 9, color: T.dim, fontFamily: T.mono, marginBottom: 8, letterSpacing: '0.12em', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 26, fontWeight: 900, fontFamily: T.display, lineHeight: 1, color: accent || T.text, letterSpacing: '0.02em' }}>{value ?? '—'}</div>
      {sub && <div style={{ fontSize: 10, color: T.muted, fontFamily: T.mono, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ── Inline Video Preview ──────────────────────────────────────────────────────
// Used for both uploaded File objects (blob URL) and existing server videos (URL string).

function VideoPreview({ src, label, onClear }) {
  const [isBlob] = useState(src instanceof File);
  const [blobUrl, setBlobUrl] = useState(null);

  useEffect(() => {
    if (src instanceof File) {
      const url = URL.createObjectURL(src);
      setBlobUrl(url);
      return () => URL.revokeObjectURL(url);
    }
  }, [src]);

  const videoSrc = isBlob ? blobUrl : src;
  if (!videoSrc) return null;

  return (
    <div style={{
      marginTop: 14,
      border: `1px solid ${T.borderMed}`,
      borderRadius: 'var(--radius)',
      overflow: 'hidden',
      animation: 'fadeUp 0.2s ease',
    }}>
      <div style={{
        padding: '8px 14px',
        borderBottom: `1px solid ${T.border}`,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(255,255,255,0.02)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
          <Eye size={11} color={T.accent} />
          <span style={{ fontSize: 9, fontFamily: T.mono, color: T.muted, letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            Preview · {label}
          </span>
        </div>
        {onClear && (
          <button
            onClick={onClear}
            style={{ background: 'none', border: 'none', cursor: 'pointer', color: T.muted, padding: 0, display: 'flex', alignItems: 'center' }}
          >
            <X size={13} />
          </button>
        )}
      </div>
      <video
        src={videoSrc}
        controls
        style={{ width: '100%', display: 'block', maxHeight: 320, background: '#000' }}
      />
    </div>
  );
}

// ── Upload zone ───────────────────────────────────────────────────────────────

function UploadZone({ onFile, file }) {
  const [drag, setDrag] = useState(false);
  const inputRef        = useRef();

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
        border: `2px dashed ${drag ? T.accent : file ? T.green : T.borderMed}`,
        borderRadius: 'var(--radius-lg)', padding: '28px 20px',
        display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
        cursor: 'pointer', transition: 'all 0.2s ease',
        background: drag ? T.accentDim : file ? T.greenDim : 'transparent',
        textAlign: 'center',
      }}
    >
      <input ref={inputRef} type="file" accept="video/*" style={{ display: 'none' }}
        onChange={e => { if (e.target.files[0]) onFile(e.target.files[0]); }} />
      {file ? (
        <>
          <Film size={24} color={T.green} />
          <span style={{ color: T.green, fontSize: 13, fontWeight: 600 }}>{file.name}</span>
          <span style={{ fontSize: 11, color: T.muted, fontFamily: T.mono }}>
            {(file.size / 1024 / 1024).toFixed(1)} MB · click to replace
          </span>
        </>
      ) : (
        <>
          <Upload size={24} color={T.muted} strokeWidth={1.5} />
          <span style={{ color: T.text, fontSize: 13, fontWeight: 500 }}>Drop video or click to browse</span>
          <span style={{ fontSize: 10, color: T.dim, fontFamily: T.mono, letterSpacing: '0.08em' }}>MP4 · AVI · MOV · MKV</span>
        </>
      )}
    </div>
  );
}

// ── Available Videos Library ──────────────────────────────────────────────────

function AvailableVideos({ token, onSelect }) {
  const [videos, setVideos]   = useState([]);
  const [loading, setLoading] = useState(true);
  const [open, setOpen]       = useState(true);

  useEffect(() => {
    listInputVideos(token)
      .then(d => setVideos(d.videos || []))
      .catch(() => setVideos([]))
      .finally(() => setLoading(false));
  }, [token]);

  if (!loading && videos.length === 0) return null;

  const fmtSize = mb => mb > 1000 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`;
  const fmtDate = ts => new Date(ts * 1000).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });

  return (
    <Card>
      <div
        style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer', marginBottom: open ? 16 : 0 }}
        onClick={() => setOpen(v => !v)}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <HardDrive size={14} color={T.accent} />
          <span style={{ fontSize: 10, fontFamily: T.mono, color: T.muted, letterSpacing: '0.14em', textTransform: 'uppercase' }}>Available Videos</span>
          {!loading && (
            <span style={{ fontSize: 9, fontFamily: T.mono, color: T.accent, background: T.accentDim, padding: '1px 7px', borderRadius: 4 }}>{videos.length}</span>
          )}
        </div>
        {open ? <ChevronUp size={14} color={T.muted} /> : <ChevronDown size={14} color={T.muted} />}
      </div>

      {open && (
        <div style={{ animation: 'fadeUp 0.2s ease' }}>
          {loading ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '12px 0', color: T.muted, fontSize: 12 }}>
              <div style={{ width: 14, height: 14, border: `2px solid ${T.borderMed}`, borderTopColor: T.accent, borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              Loading videos…
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 10 }}>
              {videos.map(v => (
                <VideoCard key={v.job_id} video={v} onSelect={onSelect} fmtSize={fmtSize} fmtDate={fmtDate} />
              ))}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function VideoCard({ video, onSelect, fmtSize, fmtDate }) {
  const [hover, setHover] = useState(false);

  return (
    <div
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
      style={{
        background: hover ? 'var(--bg-hover)' : T.elevated,
        border: `1px solid ${hover ? T.borderMed : T.border}`,
        borderRadius: 'var(--radius)', padding: '12px 14px',
        cursor: 'pointer', transition: 'all var(--transition)',
        display: 'flex', flexDirection: 'column', gap: 8,
      }}
      onClick={() => onSelect(video)}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Film size={13} color={hover ? T.accent : T.muted} style={{ transition: 'color var(--transition)', flexShrink: 0 }} />
          <span style={{ fontSize: 11, fontFamily: T.mono, color: hover ? T.text : T.muted, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 130, transition: 'color var(--transition)' }}>
            {video.filename}
          </span>
        </div>
        {video.has_output && <Pill color={T.green} bg={T.greenDim}>✓ done</Pill>}
      </div>
      <div style={{ display: 'flex', gap: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Database size={9} color={T.dim} />
          <span style={{ fontSize: 10, fontFamily: T.mono, color: T.dim }}>{fmtSize(video.size_mb)}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <Clock size={9} color={T.dim} />
          <span style={{ fontSize: 10, fontFamily: T.mono, color: T.dim }}>{fmtDate(video.uploaded_at)}</span>
        </div>
      </div>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 6, padding: '6px 10px', borderRadius: 6,
        background: hover ? T.accentDim : 'transparent',
        border: `1px solid ${hover ? 'rgba(255,107,26,0.3)' : 'transparent'}`,
        transition: 'all var(--transition)', justifyContent: 'center',
      }}>
        <Zap size={11} color={hover ? T.accent : T.dim} />
        <span style={{ fontSize: 10, fontFamily: T.mono, color: hover ? T.accent : T.dim, letterSpacing: '0.08em', textTransform: 'uppercase', transition: 'color var(--transition)' }}>
          Select · Skip Upload
        </span>
      </div>
    </div>
  );
}

// ── Options panel ─────────────────────────────────────────────────────────────

function OptionsPanel({ opts, onChange }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ border: `1px solid ${T.border}`, borderRadius: 'var(--radius)', overflow: 'hidden', marginTop: 12 }}>
      <button onClick={() => setOpen(v => !v)} style={{
        width: '100%', padding: '10px 14px',
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        background: 'rgba(255,255,255,0.02)', border: 'none', cursor: 'pointer',
        fontSize: 9, fontFamily: T.mono, color: T.muted, letterSpacing: '0.12em', textTransform: 'uppercase',
      }}>
        Processing Options {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <div style={{ padding: '14px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <div style={{ fontSize: 10, color: T.muted, marginBottom: 6, fontFamily: T.mono }}>
              Frame skip: <span style={{ color: T.text }}>{opts.frameSkip}</span>
              <span style={{ color: T.dim }}> (higher = faster, less accuracy)</span>
            </div>
            <input type="range" min={1} max={5} value={opts.frameSkip}
              onChange={e => onChange({ ...opts, frameSkip: Number(e.target.value) })}
              style={{ width: '100%', accentColor: 'var(--accent)' }} />
          </div>
          {[
            ['saveVideo',            'Save annotated output video'],
            ['annotateViolations',   'Annotate violation frames'],
            ['annotateNoViolations', 'Annotate clean frames'],
          ].map(([key, label]) => (
            <label key={key} style={{ display: 'flex', gap: 10, cursor: 'pointer', fontSize: 12, color: T.muted, alignItems: 'center' }}>
              <input type="checkbox" checked={opts[key]}
                onChange={e => onChange({ ...opts, [key]: e.target.checked })}
                style={{ accentColor: 'var(--accent)' }} />
              {label}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Progress card — Fix 5: spinner + frames count only, no broken total ───────

function ProgressCard({ phase, progress, sourceLabel }) {
  const isUploading = phase === 'uploading';
  const hasTotal    = progress.total > 0;
  const pct         = hasTotal ? Math.round((progress.done / progress.total) * 100) : null;

  return (
    <Card accent>
      <SectionLabel icon={Zap}>{isUploading ? 'Uploading' : 'Processing Pipeline'}</SectionLabel>

      <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 16 }}>
        {/* Spinner */}
        <div style={{
          width: 28, height: 28, flexShrink: 0,
          border: `2px solid ${T.borderMed}`,
          borderTopColor: T.accent,
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite',
        }} />

        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 13, color: T.muted, fontWeight: 300, marginBottom: 4 }}>
            {isUploading ? `Uploading ${sourceLabel || 'video'}…` : 'Running detection models…'}
          </div>

          {!isUploading && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
              {/* Frames processed — always shown */}
              <span style={{ fontSize: 12, fontFamily: T.mono, color: T.text }}>
                {progress.done} frames processed
              </span>

              {/* Total — only shown when backend reports a valid total */}
              {hasTotal && (
                <span style={{ fontSize: 11, fontFamily: T.mono, color: T.muted }}>
                  of {progress.total}
                </span>
              )}

              {/* FPS */}
              {progress.fps > 0 && (
                <span style={{
                  fontSize: 10, fontFamily: T.mono, color: T.accent,
                  background: T.accentDim, padding: '1px 8px', borderRadius: 4,
                }}>
                  {progress.fps} fps
                </span>
              )}

              {/* Percentage — only when total is known */}
              {pct !== null && (
                <span style={{ fontSize: 10, fontFamily: T.mono, color: T.dim, marginLeft: 'auto' }}>
                  {pct}%
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Progress bar — only rendered when total is known */}
      {!isUploading && hasTotal && (
        <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%',
            width: `${Math.max(pct, 1)}%`,
            background: `linear-gradient(90deg, var(--accent), var(--accent-hi))`,
            borderRadius: 4,
            transition: 'width 0.4s ease',
          }} />
        </div>
      )}

      {/* Upload indeterminate bar */}
      {isUploading && (
        <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{
            height: '100%', width: '40%',
            background: `linear-gradient(90deg, transparent, var(--accent), transparent)`,
            borderRadius: 4,
            animation: 'shimmer 1.5s ease infinite',
            backgroundSize: '200% 100%',
          }} />
        </div>
      )}
    </Card>
  );
}

// ── Annotated output video player ─────────────────────────────────────────────

function VideoPlayer({ jobId }) {
  const url = `${REST_BASE}/static/outputs/${jobId}.mp4`;
  return (
    <Card noPad>
      <div style={{ padding: '12px 18px', borderBottom: `1px solid ${T.border}`, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Film size={13} color={T.accent} />
          <span style={{ fontSize: 9, fontFamily: T.mono, color: T.muted, letterSpacing: '0.14em', textTransform: 'uppercase' }}>Annotated Output</span>
        </div>
        <div style={{ display: 'flex', gap: 16, fontSize: 9, fontFamily: T.mono, color: T.dim }}>
          <span style={{ color: '#666' }}>⬜ tracking</span>
          <span style={{ color: T.amber }}>🟡 predicted</span>
          <span style={{ color: T.red }}>🔴 confirmed</span>
        </div>
      </div>
      <video key={jobId} controls style={{ width: '100%', display: 'block', maxHeight: 400, background: '#000' }}>
        <source src={url} type="video/mp4" />
      </video>
      <div style={{ padding: '8px 18px', borderTop: `1px solid ${T.border}`, textAlign: 'right' }}>
        <a href={url} download style={{ fontSize: 10, fontFamily: T.mono, color: T.muted, textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
          <Download size={11} /> Download Video
        </a>
      </div>
    </Card>
  );
}

// ── Summary stats ─────────────────────────────────────────────────────────────

function SummaryStrip({ summary, meta }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10, marginBottom: 16 }}>
      <StatCard label="Frames Processed" value={meta?.total_frames_processed ?? '—'} accent={T.blue} sub={meta?.avg_fps ? `${meta.avg_fps} avg fps` : undefined} />
      <StatCard label="Processing Time" value={meta?.total_time_seconds ? `${meta.total_time_seconds.toFixed(1)}s` : '—'} accent="var(--text-muted)" />
      <StatCard label="Violations Found" value={summary?.total ?? 0} accent={T.red} />
      <StatCard label="Needs Review" value={summary?.needs_review ?? 0} accent={T.amber} />
    </div>
  );
}

// ── Track table ───────────────────────────────────────────────────────────────

const tdS = { padding: '10px 14px', fontSize: 12, verticalAlign: 'middle', color: T.text, borderBottom: `1px solid ${T.border}` };
const thS = { ...tdS, fontSize: 9, fontFamily: T.mono, color: T.dim, letterSpacing: '0.12em', textTransform: 'uppercase', textAlign: 'left', background: 'rgba(255,255,255,0.02)', fontWeight: 500 };

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
    if (raw === 'non_hsrp' || raw === 'non hsrp' || t.violation_type === 'non_hsrp_plate') return { label: 'Non-HSRP', color: T.red, bg: T.redDim };
    return null;
  })();

  const helmet = (() => {
    if (!isTwoWheeler) return null;
    const raw = t.helmet_status || t.helmet;
    if (raw === 'HELMET')   return { label: 'Helmet',    color: T.green, bg: T.greenDim };
    if (raw === 'NO_HELMET' || t.violation_type === 'no_helmet') return { label: 'No Helmet', color: T.red, bg: T.redDim };
    if (raw === 'UNCERTAIN') return { label: 'Uncertain', color: T.amber, bg: T.amberDim };
    return null;
  })();

  const violation = fmtViolation(t.violation_type);
  const stored    = t.should_store;
  const review    = t.needs_review;
  const rowBg     = stored && review ? 'rgba(255,196,61,0.04)' : stored ? 'rgba(255,59,59,0.04)' : 'transparent';

  return (
    <tr style={{ background: rowBg }}>
      <td style={tdS}><span style={{ fontFamily: T.mono, fontSize: 11 }}>#{t.track_id ?? idx}</span></td>
      <td style={tdS}>{vclass.charAt(0).toUpperCase() + vclass.slice(1)}</td>
      <td style={tdS}><span style={{ fontFamily: T.mono, fontSize: 11 }}>{t.plate_number || t.vehicle_number || '—'}</span></td>
      <td style={tdS}>{hsrp   ? <Pill color={hsrp.color}   bg={hsrp.bg}>{hsrp.label}</Pill>   : <span style={{ color: T.dim }}>—</span>}</td>
      <td style={tdS}>{isTwoWheeler ? (helmet ? <Pill color={helmet.color} bg={helmet.bg}>{helmet.label}</Pill> : <span style={{ color: T.dim }}>—</span>) : <span style={{ color: T.dim }}>N/A</span>}</td>
      <td style={tdS}>{violation ? <Pill color={T.red} bg={T.redDim}>{violation}</Pill> : <Pill color={T.green} bg={T.greenDim}>✓ Clean</Pill>}</td>
      <td style={{ ...tdS, fontFamily: T.mono, fontSize: 11 }}>{(t.violation_confidence || 0).toFixed(2)}</td>
      <td style={{ ...tdS, fontFamily: T.mono, fontSize: 11 }}>{(t.quality_score || 0).toFixed(2)}</td>
      <td style={tdS}>{stored ? (review ? <Pill color={T.amber} bg={T.amberDim}>⚠ Review</Pill> : <Pill color={T.red} bg={T.redDim}>Stored</Pill>) : <span style={{ color: T.dim }}>—</span>}</td>
      <td style={{ ...tdS, fontFamily: T.mono, fontSize: 10, color: T.dim }}>{t.first_frame ?? 0}–{t.last_frame ?? 0}</td>
    </tr>
  );
}

function TrackTable({ tracks }) {
  if (!tracks?.length) return (
    <Card><div style={{ textAlign: 'center', color: T.dim, fontSize: 13, padding: '20px 0' }}>No tracks found.</div></Card>
  );
  return (
    <Card noPad>
      <div style={{ padding: '14px 18px', borderBottom: `1px solid ${T.border}`, display: 'flex', alignItems: 'center', gap: 8 }}>
        <Layers size={13} color={T.accent} />
        <span style={{ fontSize: 9, fontFamily: T.mono, color: T.muted, letterSpacing: '0.14em', textTransform: 'uppercase' }}>
          Track Results — {tracks.length} vehicles
        </span>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr>{['Track','Vehicle','Plate','HSRP','Helmet','Violation','Conf.','Quality','Status','Frames'].map(h => <th key={h} style={thS}>{h}</th>)}</tr>
          </thead>
          <tbody>{tracks.map((t, i) => <TrackRow key={t.track_id ?? i} t={t} idx={i} />)}</tbody>
        </table>
      </div>
    </Card>
  );
}

function ViolationBreakdown({ byType }) {
  if (!byType || !Object.keys(byType).length) return null;
  const maxCount = Math.max(...Object.values(byType).map(v => v.count));
  return (
    <Card>
      <SectionLabel icon={BarChart2}>Violation Breakdown</SectionLabel>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
        {Object.entries(byType).map(([k, v]) => (
          <div key={k}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, marginBottom: 6 }}>
              <span style={{ color: T.muted }}>{k.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
              <span style={{ fontFamily: T.mono, fontSize: 11 }}>{v.count} <span style={{ color: T.dim }}>· avg {v.avg_conf?.toFixed(2)}</span></span>
            </div>
            <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${Math.round((v.count / maxCount) * 100)}%`, background: T.red, borderRadius: 4, transition: 'width 0.6s ease' }} />
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
}

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
      display: 'inline-flex', alignItems: 'center', gap: 8,
      padding: '8px 14px', borderRadius: 'var(--radius)',
      background: 'transparent', border: `1px solid ${T.borderMed}`,
      color: T.muted, cursor: 'pointer', fontSize: 11, fontFamily: T.mono,
      marginBottom: 16, letterSpacing: '0.06em', transition: 'all var(--transition)',
    }}
      onMouseEnter={e => { e.currentTarget.style.borderColor = T.accent; e.currentTarget.style.color = T.accent; }}
      onMouseLeave={e => { e.currentTarget.style.borderColor = T.borderMed; e.currentTarget.style.color = T.muted; }}
    >
      <Download size={12} /> Export JSON Report
    </button>
  );
}

// ── Main page ─────────────────────────────────────────────────────────────────

export default function DashboardPage() {
  const { token } = useAuth();

  const [file, setFile]                   = useState(null);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [opts, setOpts]                   = useState({ frameSkip: 1, saveVideo: true, annotateViolations: true, annotateNoViolations: false });
  const [phase, setPhase]                 = useState('idle');
  const [jobId, setJobId]                 = useState(null);
  const [progress, setProgress]           = useState({ done: 0, total: 0, fps: 0 });
  const [result, setResult]               = useState(null);
  const [tracks, setTracks]               = useState([]);
  const [errorMsg, setErrorMsg]           = useState('');
  const cancelRef                         = useRef(null);

  const reset = useCallback(() => {
    cancelRef.current?.();
    setFile(null); setSelectedVideo(null); setPhase('idle'); setJobId(null);
    setProgress({ done: 0, total: 0, fps: 0 });
    setResult(null); setTracks([]); setErrorMsg('');
  }, []);

  useEffect(() => () => cancelRef.current?.(), []);

  function handleSelectExisting(video) { setSelectedVideo(video); setFile(null); }
  function handleClearFile()           { setFile(null); }
  function handleClearSelected()       { setSelectedVideo(null); }

  async function handleRun() {
    if (!file && !selectedVideo) return;
    setPhase(file ? 'uploading' : 'processing');
    setErrorMsg('');

    try {
      const ocrMode =
        opts.annotateViolations && opts.annotateNoViolations ? 'always' :
        opts.annotateViolations   ? 'on_violation' :
        opts.annotateNoViolations ? 'on_clean' : 'off';

      const resp = file
        ? await uploadVideo(token, file, { ...opts, ocrMode })
        : await processExistingVideo(token, selectedVideo.job_id, { ...opts, ocrMode });

      if (!resp.job_id) throw new Error('No job_id returned from server');

      const id = resp.job_id;
      setJobId(id);
      setPhase('processing');

      cancelRef.current = connectJobStream(
        token, id,
        data => setProgress({ done: data.progress, total: data.total_frames, fps: data.fps }),
        async () => {
          try {
            const [r, t] = await Promise.all([getJobResult(token, id), getJobTracks(token, id)]);
            setResult(r); setTracks(t.tracks || []); setPhase('done');
          } catch (err) {
            setErrorMsg(`Result fetch failed: ${err.message}`); setPhase('error');
          }
        },
        err => { setErrorMsg(err.message); setPhase('error'); },
      );
    } catch (err) {
      setErrorMsg(err.message); setPhase('error');
    }
  }

  const summary  = result?.summary  || {};
  const meta     = result?.metadata || {};
  const canRun   = file || selectedVideo;

  // Derive video preview source and label
  const previewSrc   = file || (selectedVideo ? `${REST_BASE}/static/inputs/${selectedVideo.job_id}.mp4` : null);
  const previewLabel = file ? file.name : selectedVideo?.filename;

  return (
    <div style={{ padding: '28px 36px', maxWidth: 1020 }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 28, fontWeight: 900, letterSpacing: '0.04em', textTransform: 'uppercase', marginBottom: 4, lineHeight: 1 }}>
          Video Analysis
        </h1>
        <p style={{ color: T.muted, fontSize: 13, fontWeight: 300 }}>
          Detect HSRP violations, helmet non-compliance, and read licence plates from traffic footage.
        </p>
      </div>

      {/* Idle / error */}
      {(phase === 'idle' || phase === 'error') && (
        <div style={{ animation: 'fadeUp 0.25s ease' }}>

          {/* Library */}
          <AvailableVideos token={token} onSelect={handleSelectExisting} />

          {/* Selected existing video badge */}
          {selectedVideo && (
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              padding: '10px 14px', background: T.accentDim,
              border: `1px solid rgba(255,107,26,0.3)`, borderRadius: 'var(--radius)', marginBottom: 12,
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <Zap size={13} color={T.accent} />
                <span style={{ fontSize: 12, color: T.accent, fontFamily: T.mono }}>{selectedVideo.filename}</span>
                <span style={{ fontSize: 10, color: T.muted, fontFamily: T.mono }}>· will skip upload</span>
              </div>
              <button onClick={handleClearSelected} style={{ background: 'none', border: 'none', cursor: 'pointer', color: T.muted, fontSize: 11, fontFamily: T.mono, padding: '2px 8px' }}>
                ✕ clear
              </button>
            </div>
          )}

          {/* Upload zone — only when no existing video selected */}
          {!selectedVideo && (
            <Card>
              <SectionLabel icon={Upload}>Upload New Video</SectionLabel>
              <UploadZone onFile={setFile} file={file} />
              {/* Fix 4: inline preview for uploaded file */}
              {file && (
                <VideoPreview src={file} label={file.name} onClear={handleClearFile} />
              )}
            </Card>
          )}

          {/* Fix 4: inline preview for selected existing video */}
          {selectedVideo && (
            <Card>
              <SectionLabel icon={Eye}>Preview</SectionLabel>
              <VideoPreview
                src={`${REST_BASE}/static/inputs/${selectedVideo.job_id}.mp4`}
                label={selectedVideo.filename}
                onClear={null}
              />
            </Card>
          )}

          {/* Options + run */}
          <Card>
            <OptionsPanel opts={opts} onChange={setOpts} />

            {errorMsg && (
              <div style={{ display: 'flex', gap: 10, marginTop: 14, padding: '12px 14px', background: T.redDim, border: `1px solid rgba(255,59,59,0.2)`, borderRadius: 'var(--radius)', color: T.red, fontSize: 12 }}>
                <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                {errorMsg}
              </div>
            )}

            <button
              onClick={handleRun}
              disabled={!canRun}
              style={{
                marginTop: 16, width: '100%', padding: '13px',
                background: canRun ? T.accent : 'rgba(255,255,255,0.04)',
                color: canRun ? '#080c14' : T.dim,
                border: 'none', borderRadius: 'var(--radius)',
                cursor: canRun ? 'pointer' : 'not-allowed',
                fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 700,
                letterSpacing: '0.1em', textTransform: 'uppercase',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                transition: 'all var(--transition)',
              }}
            >
              <Play size={14} />
              {selectedVideo ? 'Process Existing Video' : 'Run Detection Pipeline'}
            </button>
          </Card>
        </div>
      )}

      {/* Progress */}
      {(phase === 'uploading' || phase === 'processing') && (
        <ProgressCard phase={phase} progress={progress} sourceLabel={file?.name || selectedVideo?.filename} />
      )}

      {/* Results */}
      {phase === 'done' && result && (
        <div style={{ animation: 'fadeUp 0.3s ease' }}>
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            marginBottom: 16, padding: '12px 16px',
            background: T.greenDim, border: `1px solid rgba(0,229,160,0.2)`, borderRadius: 'var(--radius)',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <CheckCircle size={15} color={T.green} />
              <span style={{ fontSize: 12, color: T.green, fontFamily: T.mono, letterSpacing: '0.1em' }}>PROCESSING COMPLETE</span>
            </div>
            <button onClick={reset} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '6px 12px', borderRadius: 6, background: 'transparent', border: `1px solid ${T.border}`, color: T.muted, cursor: 'pointer', fontSize: 11, fontFamily: T.mono }}>
              <RefreshCw size={11} /> New Analysis
            </button>
          </div>

          <SummaryStrip summary={summary} meta={meta} />
          {opts.saveVideo && <VideoPlayer jobId={jobId} />}
          <DownloadReport result={result} jobId={jobId} />
          <ViolationBreakdown byType={summary?.by_type} />
          <TrackTable tracks={tracks} />
        </div>
      )}
    </div>
  );
}
