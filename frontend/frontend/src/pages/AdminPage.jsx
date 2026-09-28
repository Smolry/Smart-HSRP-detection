import { useState, useEffect, useCallback } from 'react';
import { useAuth } from '../hooks/useAuth';
import { getViolations, getThresholds, resetThresholds, getUsers } from '../services/api';
import { RefreshCw, AlertTriangle, Shield, Users } from 'lucide-react';

const TABS = [
  { id: 'violations', label: 'Violations DB', icon: AlertTriangle },
  { id: 'thresholds', label: 'Thresholds', icon: Shield },
  { id: 'users', label: 'Users', icon: Users },
];

export default function AdminPage() {
  const [tab, setTab] = useState('violations');

  return (
    <div style={{ padding: '32px 40px' }}>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{ fontFamily: 'var(--font-display)', fontSize: 28, fontWeight: 800, letterSpacing: '-0.03em', marginBottom: 4 }}>
          Admin Panel
        </h1>
        <p style={{ color: 'var(--text-muted)', fontSize: 13 }}>System management and violation records.</p>
      </div>

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 2, background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 10, padding: 4, marginBottom: 28, width: 'fit-content' }}>
        {TABS.map(({ id, label, icon: Icon }) => (
          <button key={id} onClick={() => setTab(id)}
            style={{
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '8px 16px', borderRadius: 8, border: 'none', cursor: 'pointer',
              fontSize: 12, fontFamily: 'var(--font-mono)',
              background: tab === id ? 'var(--bg-hover)' : 'transparent',
              color: tab === id ? 'var(--accent)' : 'var(--text-muted)',
              transition: 'all var(--transition)',
            }}
          >
            <Icon size={13} /> {label}
          </button>
        ))}
      </div>

      <div style={{ animation: 'fadeIn 0.2s ease' }}>
        {tab === 'violations' && <ViolationsTab />}
        {tab === 'thresholds' && <ThresholdsTab />}
        {tab === 'users' && <UsersTab />}
      </div>
    </div>
  );
}

// ── Violations ────────────────────────────────────────────────────────────────

function ViolationsTab() {
  const { token } = useAuth();
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({ violationType: '', needsReview: '', minQuality: 0, limit: 200 });

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = { limit: filters.limit, minQuality: filters.minQuality };
      if (filters.violationType) params.violationType = filters.violationType;
      if (filters.needsReview !== '') params.needsReview = filters.needsReview === 'true';
      const resp = await getViolations(token, params);
      setData(resp.violations || []);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [token, filters]);

  useEffect(() => { load(); }, [load]);

  const needsReviewCount = data.filter(v => v.needs_manual_review).length;

  return (
    <div>
      {/* Filter bar */}
      <div style={{
        display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end',
        background: 'var(--bg-card)', border: '1px solid var(--border)',
        borderRadius: 12, padding: 16, marginBottom: 20,
      }}>
        <FilterSelect label="type" value={filters.violationType}
          onChange={v => setFilters(f => ({ ...f, violationType: v }))}
          options={[{ label: 'all', value: '' }, { label: 'non_hsrp', value: 'non_hsrp_plate' }, { label: 'no_helmet', value: 'no_helmet' }]} />
        <FilterSelect label="review" value={filters.needsReview}
          onChange={v => setFilters(f => ({ ...f, needsReview: v }))}
          options={[{ label: 'all', value: '' }, { label: 'needs review', value: 'true' }, { label: 'auto-cleared', value: 'false' }]} />
        <div>
          <div style={filterLabel}>min quality</div>
          <input type="number" min={0} max={1} step={0.05} value={filters.minQuality}
            onChange={e => setFilters(f => ({ ...f, minQuality: Number(e.target.value) }))}
            style={inputStyle} />
        </div>
        <button onClick={load}
          style={{ padding: '8px 14px', background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-muted)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontFamily: 'var(--font-mono)' }}>
          <RefreshCw size={12} /> refresh
        </button>
      </div>

      {/* Stats strip */}
      <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
        {[['total', data.length, 'var(--text)'], ['needs review', needsReviewCount, 'var(--amber)'], ['auto-cleared', data.length - needsReviewCount, 'var(--green)']].map(([l, v, c]) => (
          <div key={l} style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, padding: '10px 16px' }}>
            <div style={{ fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 800, color: c }}>{v}</div>
            <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', marginTop: 2 }}>{l}</div>
          </div>
        ))}
      </div>

      {/* Table */}
      {loading ? <Spinner /> : (
        <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 12, overflow: 'hidden' }}>
          <ViolationsTable rows={data} />
        </div>
      )}
    </div>
  );
}

function ViolationsTable({ rows }) {
  if (!rows.length) return <div style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)', fontSize: 13, fontFamily: 'var(--font-mono)' }}>no violations found</div>;
  const cols = ['id', 'vehicle_number', 'vehicle_class', 'violation_type', 'quality_score', 'needs_review', 'created_at'];
  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
        <thead>
          <tr>
            {cols.map(c => <th key={c} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap' }}>{c.replace(/_/g, ' ')}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((v, i) => (
            <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
              <td style={td}><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>#{v.id}</span></td>
              <td style={td}><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text)' }}>{v.vehicle_number || '—'}</span></td>
              <td style={td}><span style={{ ...badge, background: 'var(--bg-hover)', color: 'var(--text-muted)' }}>{v.vehicle_class || '—'}</span></td>
              <td style={td}><span style={{ ...badge, background: v.violation_type ? 'var(--red-dim)' : 'var(--green-dim)', color: v.violation_type ? 'var(--red)' : 'var(--green)' }}>{v.violation_type || 'clean'}</span></td>
              <td style={td}><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{v.quality_score != null ? (v.quality_score * 100).toFixed(1) + '%' : '—'}</span></td>
              <td style={td}>{v.needs_manual_review ? <span style={{ ...badge, background: 'var(--amber-dim)', color: 'var(--amber)' }}>review</span> : <span style={{ ...badge, background: 'var(--bg-hover)', color: 'var(--text-dim)' }}>ok</span>}</td>
              <td style={td}><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', fontSize: 11 }}>{v.created_at ? new Date(v.created_at).toLocaleString() : '—'}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── Thresholds ────────────────────────────────────────────────────────────────

function ThresholdsTab() {
  const { token } = useAuth();
  const [thresholds, setThresholds] = useState(null);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    getThresholds(token).then(setThresholds).catch(console.error);
  }, [token]);

  async function handleReset() {
    setResetting(true);
    const data = await resetThresholds(token);
    setThresholds(data.thresholds || data);
    setResetting(false);
  }

  return (
    <div style={{ maxWidth: 500 }}>
      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 14, padding: 24 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
          <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 15 }}>Adaptive thresholds</span>
          <button onClick={handleReset} disabled={resetting}
            style={{ padding: '7px 14px', background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 8, color: 'var(--text-muted)', cursor: 'pointer', fontSize: 11, fontFamily: 'var(--font-mono)', display: 'flex', alignItems: 'center', gap: 6 }}>
            <RefreshCw size={11} style={{ animation: resetting ? 'spin 0.9s linear infinite' : 'none' }} />
            reset to defaults
          </button>
        </div>

        {thresholds ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            {Object.entries(thresholds).map(([key, val]) => (
              <div key={key} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', background: 'var(--bg)', borderRadius: 8, border: '1px solid var(--border)' }}>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--text-muted)' }}>{key}</span>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ width: 80, height: 4, background: 'var(--border)', borderRadius: 2 }}>
                    <div style={{ width: `${(val * 100).toFixed(0)}%`, height: '100%', background: 'var(--accent)', borderRadius: 2 }} />
                  </div>
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent)', minWidth: 36, textAlign: 'right' }}>{(val * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        ) : <Spinner />}
      </div>
    </div>
  );
}

// ── Users ─────────────────────────────────────────────────────────────────────

function UsersTab() {
  const { token } = useAuth();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getUsers(token).then(d => setUsers(d.users || [])).catch(console.error).finally(() => setLoading(false));
  }, [token]);

  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 14, overflow: 'hidden', maxWidth: 700 }}>
      <div style={{ padding: '14px 18px', borderBottom: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>registered users</span>
        <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>{users.length}</span>
      </div>
      {loading ? <Spinner /> : (
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
          <thead>
            <tr>
              {['id', 'email', 'role', 'created'].map(c => <th key={c} style={{ padding: '10px 14px', textAlign: 'left', fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', borderBottom: '1px solid var(--border)' }}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {users.map((u, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                <td style={td}><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>#{u.id}</span></td>
                <td style={td}>{u.email}</td>
                <td style={td}><span style={{ ...badge, background: u.role === 'admin' ? 'var(--blue-dim)' : 'var(--bg-hover)', color: u.role === 'admin' ? 'var(--blue)' : 'var(--text-muted)' }}>{u.role}</span></td>
                <td style={td}><span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-dim)' }}>{u.created_at ? new Date(u.created_at).toLocaleDateString() : '—'}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Shared ────────────────────────────────────────────────────────────────────

const td = { padding: '10px 14px', color: 'var(--text-muted)' };
const badge = { display: 'inline-block', padding: '2px 8px', borderRadius: 4, fontSize: 10, fontFamily: 'var(--font-mono)' };
const filterLabel = { fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-dim)', marginBottom: 5 };
const inputStyle = {
  padding: '7px 10px', background: 'var(--bg)', border: '1px solid var(--border)',
  borderRadius: 7, color: 'var(--text)', fontSize: 12, fontFamily: 'var(--font-mono)',
  outline: 'none', width: 80,
};

function FilterSelect({ label, value, onChange, options }) {
  return (
    <div>
      <div style={filterLabel}>{label}</div>
      <select value={value} onChange={e => onChange(e.target.value)}
        style={{ ...inputStyle, width: 'auto', cursor: 'pointer' }}>
        {options.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </div>
  );
}

function Spinner() {
  return <div style={{ padding: 32, display: 'flex', justifyContent: 'center' }}>
    <div style={{ width: 28, height: 28, border: '2px solid var(--border)', borderTopColor: 'var(--accent)', borderRadius: '50%', animation: 'spin 0.9s linear infinite' }} />
  </div>;
}
