import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Cpu, Eye, EyeOff, AlertCircle, ArrowRight } from 'lucide-react';

function Field({ label, error, ...props }) {
  const [focused, setFocused] = useState(false);
  return (
    <div>
      <label style={{
        fontSize: 10,
        color: focused ? 'var(--accent)' : 'var(--text-muted)',
        display: 'block',
        marginBottom: 6,
        fontFamily: 'var(--font-mono)',
        letterSpacing: '0.1em',
        textTransform: 'uppercase',
        transition: 'color var(--transition)',
      }}>
        {label}
      </label>
      <input
        {...props}
        required
        onFocus={e => { setFocused(true); props.onFocus?.(e); }}
        onBlur={e => { setFocused(false); props.onBlur?.(e); }}
        style={{
          width: '100%',
          padding: '11px 14px',
          background: 'var(--bg)',
          border: `1px solid ${focused ? 'var(--accent)' : 'var(--border-med)'}`,
          borderRadius: 'var(--radius)',
          color: 'var(--text)',
          fontSize: 13,
          fontFamily: 'var(--font-body)',
          outline: 'none',
          transition: 'border-color var(--transition)',
          ...props.style,
        }}
      />
    </div>
  );
}

export default function LoginPage() {
  const [tab, setTab]         = useState('login');
  const [email, setEmail]     = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole]       = useState('user');
  const [showPw, setShowPw]   = useState(false);
  const [error, setError]     = useState('');
  const [loading, setLoading] = useState(false);
  const { login, signup }     = useAuth();
  const navigate              = useNavigate();

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

  const stats = [
    { value: '4', label: 'AI Models' },
    { value: 'TRT', label: 'Optimized' },
    { value: 'A10G', label: 'GPU Ready' },
  ];

  return (
    <div style={{
      minHeight: '100vh',
      display: 'flex',
      background: 'var(--bg)',
      position: 'relative',
      overflow: 'hidden',
    }}>

      {/* Background grid pattern */}
      <div style={{
        position: 'absolute',
        inset: 0,
        backgroundImage: `
          linear-gradient(rgba(255,107,26,0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,107,26,0.04) 1px, transparent 1px)
        `,
        backgroundSize: '60px 60px',
        pointerEvents: 'none',
      }} />

      {/* Glow blob */}
      <div style={{
        position: 'absolute',
        bottom: -300,
        left: -200,
        width: 700,
        height: 700,
        borderRadius: '50%',
        background: 'radial-gradient(circle, rgba(255,107,26,0.06) 0%, transparent 60%)',
        pointerEvents: 'none',
      }} />

      {/* Left — branding */}
      <div style={{
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        padding: '80px 70px',
        borderRight: '1px solid var(--border)',
        position: 'relative',
      }}>
        <div style={{ maxWidth: 480 }}>

          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 56 }}>
            <div style={{
              width: 46,
              height: 46,
              background: 'var(--accent)',
              borderRadius: 8,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Cpu size={22} color="#080c14" strokeWidth={2.5} />
            </div>
            <div>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontSize: 22,
                fontWeight: 900,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                lineHeight: 1.1,
              }}>Smart HSRP</div>
              <div style={{
                fontSize: 10,
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
              }}>Traffic Enforcement System</div>
            </div>
          </div>

          {/* Headline */}
          <h1 style={{
            fontFamily: 'var(--font-display)',
            fontSize: 64,
            fontWeight: 900,
            lineHeight: 0.95,
            letterSpacing: '-0.01em',
            marginBottom: 24,
            textTransform: 'uppercase',
          }}>
            AI-Powered<br />
            <span style={{ color: 'var(--accent)' }}>Violation</span><br />
            Detection.
          </h1>

          <p style={{
            color: 'var(--text-muted)',
            fontSize: 15,
            lineHeight: 1.7,
            maxWidth: 360,
            marginBottom: 48,
            fontWeight: 300,
          }}>
            Upload traffic footage. Detect HSRP violations, helmet non-compliance, and read plates — automatically.
          </p>

          {/* Stats */}
          <div style={{ display: 'flex', gap: 0 }}>
            {stats.map((s, i) => (
              <div key={s.label} style={{
                padding: '14px 24px',
                borderLeft: i === 0 ? '2px solid var(--accent)' : '1px solid var(--border)',
              }}>
                <div style={{
                  fontFamily: 'var(--font-display)',
                  fontSize: 28,
                  fontWeight: 900,
                  color: i === 0 ? 'var(--accent)' : 'var(--text)',
                  letterSpacing: '0.04em',
                  lineHeight: 1,
                }}>{s.value}</div>
                <div style={{
                  fontSize: 10,
                  color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)',
                  marginTop: 4,
                  textTransform: 'uppercase',
                  letterSpacing: '0.1em',
                }}>{s.label}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right — form */}
      <div style={{
        width: 460,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 48,
      }}>
        <div style={{
          width: '100%',
          background: 'var(--bg-card)',
          border: '1px solid var(--border-med)',
          borderRadius: 'var(--radius-lg)',
          padding: 36,
          animation: 'fadeUp 0.4s ease',
          position: 'relative',
          overflow: 'hidden',
        }}>
          {/* Top accent */}
          <div style={{
            position: 'absolute',
            top: 0, left: 0, right: 0,
            height: 2,
            background: 'linear-gradient(90deg, var(--accent), transparent)',
          }} />

          {/* Tabs */}
          <div style={{
            display: 'flex',
            marginBottom: 30,
            borderBottom: '1px solid var(--border)',
          }}>
            {['login', 'signup'].map(t => (
              <button key={t} onClick={() => { setTab(t); setError(''); }}
                style={{
                  flex: 1,
                  padding: '10px 0',
                  background: 'transparent',
                  border: 'none',
                  borderBottom: `2px solid ${tab === t ? 'var(--accent)' : 'transparent'}`,
                  marginBottom: -1,
                  cursor: 'pointer',
                  fontFamily: 'var(--font-display)',
                  fontSize: 14,
                  fontWeight: tab === t ? 700 : 400,
                  letterSpacing: '0.08em',
                  textTransform: 'uppercase',
                  color: tab === t ? 'var(--accent)' : 'var(--text-muted)',
                  transition: 'all var(--transition)',
                }}
              >
                {t === 'login' ? 'Sign In' : 'Register'}
              </button>
            ))}
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            <Field
              label="Email"
              type="email"
              value={email}
              onChange={e => setEmail(e.target.value)}
              placeholder="operator@traffic.gov"
            />

            <div style={{ position: 'relative' }}>
              <Field
                label="Password"
                type={showPw ? 'text' : 'password'}
                value={password}
                onChange={e => setPassword(e.target.value)}
                placeholder="••••••••"
                style={{ paddingRight: 44 }}
              />
              <button
                type="button"
                onClick={() => setShowPw(v => !v)}
                style={{
                  position: 'absolute',
                  right: 12,
                  bottom: 11,
                  background: 'none',
                  border: 'none',
                  cursor: 'pointer',
                  color: 'var(--text-muted)',
                  padding: 0,
                  lineHeight: 1,
                }}
              >
                {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
              </button>
            </div>

            {tab === 'signup' && (
              <div>
                <div style={{
                  fontSize: 10,
                  color: 'var(--text-muted)',
                  marginBottom: 8,
                  fontFamily: 'var(--font-mono)',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                }}>Role</div>
                <div style={{ display: 'flex', gap: 8 }}>
                  {['user', 'admin'].map(r => (
                    <button key={r} type="button" onClick={() => setRole(r)}
                      style={{
                        flex: 1,
                        padding: '9px 0',
                        borderRadius: 'var(--radius)',
                        cursor: 'pointer',
                        border: `1px solid ${role === r ? 'var(--accent)' : 'var(--border-med)'}`,
                        background: role === r ? 'var(--accent-dim)' : 'transparent',
                        color: role === r ? 'var(--accent)' : 'var(--text-muted)',
                        fontSize: 11,
                        fontFamily: 'var(--font-mono)',
                        textTransform: 'uppercase',
                        letterSpacing: '0.1em',
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
                display: 'flex',
                alignItems: 'flex-start',
                gap: 10,
                padding: '10px 14px',
                borderRadius: 'var(--radius)',
                background: 'var(--red-dim)',
                border: '1px solid rgba(255,59,59,0.2)',
                color: 'var(--red)',
                fontSize: 12,
              }}>
                <AlertCircle size={14} style={{ flexShrink: 0, marginTop: 1 }} />
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                padding: '13px',
                borderRadius: 'var(--radius)',
                border: 'none',
                cursor: loading ? 'not-allowed' : 'pointer',
                background: loading ? 'rgba(255,107,26,0.4)' : 'var(--accent)',
                color: '#080c14',
                fontFamily: 'var(--font-display)',
                fontSize: 14,
                fontWeight: 700,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                marginTop: 4,
                transition: 'all var(--transition)',
                opacity: loading ? 0.7 : 1,
              }}
            >
              {loading ? 'Please wait…' : tab === 'login' ? 'Sign In' : 'Create Account'}
              {!loading && <ArrowRight size={15} />}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
