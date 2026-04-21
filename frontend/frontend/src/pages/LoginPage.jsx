import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Radio, Eye, EyeOff, AlertCircle } from 'lucide-react';

export default function LoginPage() {
  const [tab, setTab] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState('user');
  const [showPw, setShowPw] = useState(false);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login, signup } = useAuth();
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      if (tab === 'login') await login(email, password);
      else await signup(email, password, role);
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{
      minHeight: '100vh', display: 'flex',
      background: 'var(--bg)',
      position: 'relative', overflow: 'hidden',
    }}>
      {/* Background grid */}
      <div style={{
        position: 'absolute', inset: 0, opacity: 0.03,
        backgroundImage: 'linear-gradient(var(--accent) 1px, transparent 1px), linear-gradient(90deg, var(--accent) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
        pointerEvents: 'none',
      }} />

      {/* Accent blob */}
      <div style={{
        position: 'absolute', top: -200, right: -200,
        width: 600, height: 600, borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(232,255,71,0.06) 0%, transparent 70%)',
        pointerEvents: 'none',
      }} />

      {/* Left panel — branding */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column',
        justifyContent: 'center', padding: '80px 60px',
        borderRight: '1px solid var(--border)',
      }}>
        <div style={{ maxWidth: 440 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 48 }}>
            <div style={{
              width: 44, height: 44, background: 'var(--accent)',
              borderRadius: 12, display: 'flex', alignItems: 'center', justifyContent: 'center',
            }}>
              <Radio size={20} color="#0a0a0f" strokeWidth={2.5} />
            </div>
            <div>
              <div style={{ fontFamily: 'var(--font-display)', fontSize: 20, fontWeight: 700, letterSpacing: '-0.02em' }}>Smart HSRP</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>Traffic Monitoring System</div>
            </div>
          </div>

          <h1 style={{
            fontFamily: 'var(--font-display)', fontSize: 52, fontWeight: 800,
            lineHeight: 1.05, letterSpacing: '-0.04em', marginBottom: 20,
            color: 'var(--text)',
          }}>
            AI-powered<br />
            <span style={{ color: 'var(--accent)' }}>violation</span><br />
            detection.
          </h1>

          <p style={{ color: 'var(--text-muted)', fontSize: 15, lineHeight: 1.7, maxWidth: 340 }}>
            Upload traffic footage. The pipeline detects HSRP violations, helmet non-compliance, and reads plates automatically.
          </p>

          <div style={{ display: 'flex', gap: 24, marginTop: 40 }}>
            {[['Helmet', 'detection'], ['HSRP', 'classification'], ['OCR', 'plate reading']].map(([a, b]) => (
              <div key={a} style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <span style={{ fontFamily: 'var(--font-display)', fontWeight: 700, fontSize: 14, color: 'var(--accent)' }}>{a}</span>
                <span style={{ fontSize: 11, color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>{b}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — form */}
      <div style={{
        width: 460, display: 'flex', alignItems: 'center',
        justifyContent: 'center', padding: 40,
      }}>
        <div style={{
          width: '100%', background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          borderRadius: 20, padding: 36,
          animation: 'fadeIn 0.35s ease',
        }}>
          {/* Tabs */}
          <div style={{
            display: 'flex', gap: 4, background: 'var(--bg)',
            borderRadius: 10, padding: 4, marginBottom: 28,
          }}>
            {['login', 'signup'].map(t => (
              <button key={t} onClick={() => { setTab(t); setError(''); }}
                style={{
                  flex: 1, padding: '8px 0', borderRadius: 8,
                  border: 'none', cursor: 'pointer',
                  fontFamily: 'var(--font-display)', fontSize: 13, fontWeight: 600,
                  transition: 'all var(--transition)',
                  background: tab === t ? 'var(--bg-card)' : 'transparent',
                  color: tab === t ? 'var(--text)' : 'var(--text-muted)',
                  boxShadow: tab === t ? '0 1px 4px rgba(0,0,0,0.4)' : 'none',
                }}
              >
                {t === 'login' ? 'Sign in' : 'Create account'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <Field label="Email" type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="you@example.com" />

            <div style={{ position: 'relative' }}>
              <Field label="Password" type={showPw ? 'text' : 'password'} value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" />
              <button type="button" onClick={() => setShowPw(v => !v)}
                style={{
                  position: 'absolute', right: 12, top: 32, background: 'none',
                  border: 'none', cursor: 'pointer', color: 'var(--text-muted)',
                  padding: 0,
                }}
              >
                {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>

            {tab === 'signup' && (
              <div>
                <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 6, fontFamily: 'var(--font-mono)' }}>Role</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  {['user', 'admin'].map(r => (
                    <button key={r} type="button" onClick={() => setRole(r)}
                      style={{
                        flex: 1, padding: '9px 0', borderRadius: 8, cursor: 'pointer',
                        border: `1px solid ${role === r ? 'var(--accent)' : 'var(--border)'}`,
                        background: role === r ? 'var(--accent-dim)' : 'var(--bg)',
                        color: role === r ? 'var(--accent)' : 'var(--text-muted)',
                        fontSize: 12, fontFamily: 'var(--font-mono)',
                        transition: 'all var(--transition)',
                      }}
                    >
                      {r}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {error && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '10px 12px', borderRadius: 8,
                background: 'var(--red-dim)', border: '1px solid rgba(255,68,68,0.2)',
                color: 'var(--red)', fontSize: 12,
              }}>
                <AlertCircle size={14} />
                {error}
              </div>
            )}

            <button type="submit" disabled={loading}
              style={{
                padding: '12px', borderRadius: 10, border: 'none', cursor: loading ? 'not-allowed' : 'pointer',
                background: loading ? 'rgba(232,255,71,0.3)' : 'var(--accent)',
                color: '#0a0a0f', fontFamily: 'var(--font-display)',
                fontSize: 14, fontWeight: 700, marginTop: 4,
                transition: 'all var(--transition)',
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? 'Please wait…' : tab === 'login' ? 'Sign in' : 'Create account'}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}

function Field({ label, ...props }) {
  return (
    <div>
      <label style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginBottom: 6, fontFamily: 'var(--font-mono)' }}>
        {label}
      </label>
      <input {...props}
        required
        style={{
          width: '100%', padding: '10px 12px',
          background: 'var(--bg)', border: '1px solid var(--border)',
          borderRadius: 8, color: 'var(--text)', fontSize: 13,
          outline: 'none', fontFamily: 'var(--font-body)',
          transition: 'border-color var(--transition)',
        }}
        onFocus={e => e.target.style.borderColor = 'var(--accent)'}
        onBlur={e => e.target.style.borderColor = 'var(--border)'}
      />
    </div>
  );
}
