import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Camera, ShieldAlert, LogOut, Cpu, Activity } from 'lucide-react';

function NavItem({ to, icon: Icon, label, badge }) {
  return (
    <NavLink
      to={to}
      style={({ isActive }) => ({
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '9px 14px',
        borderRadius: 'var(--radius)',
        textDecoration: 'none',
        fontSize: 13,
        fontWeight: isActive ? 600 : 400,
        fontFamily: 'var(--font-body)',
        color: isActive ? 'var(--accent-hi)' : 'var(--text-muted)',
        background: isActive ? 'var(--accent-dim)' : 'transparent',
        borderLeft: isActive ? '2px solid var(--accent)' : '2px solid transparent',
        transition: 'all var(--transition)',
        letterSpacing: '0.01em',
        position: 'relative',
      })}
    >
      <Icon size={14} />
      {label}
      {badge != null && (
        <span style={{
          marginLeft: 'auto',
          fontSize: 10,
          fontFamily: 'var(--font-mono)',
          color: 'var(--accent)',
          background: 'var(--accent-dim)',
          padding: '1px 6px',
          borderRadius: 4,
        }}>{badge}</span>
      )}
    </NavLink>
  );
}

function LogoutBtn() {
  const { logout } = useAuth();
  const navigate = useNavigate();
  return (
    <button
      onClick={() => { logout(); navigate('/login'); }}
      style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '9px 14px', borderRadius: 'var(--radius)',
        border: 'none', cursor: 'pointer',
        fontSize: 13, fontWeight: 400,
        color: 'var(--text-muted)', background: 'transparent',
        width: '100%', transition: 'all var(--transition)',
        fontFamily: 'var(--font-body)',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.color = 'var(--red)';
        e.currentTarget.style.background = 'var(--red-dim)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.color = 'var(--text-muted)';
        e.currentTarget.style.background = 'transparent';
      }}
    >
      <LogOut size={14} />
      Sign out
    </button>
  );
}

export default function Layout() {
  const { user } = useAuth();

  return (
    <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
      {/* Sidebar */}
      <aside style={{
        width: 230,
        minWidth: 230,
        background: 'var(--bg-card)',
        borderRight: '1px solid var(--border)',
        display: 'flex',
        flexDirection: 'column',
        padding: '0',
        position: 'relative',
        overflow: 'hidden',
      }}>
        {/* Accent line top */}
        <div style={{
          height: 2,
          background: 'linear-gradient(90deg, var(--accent), transparent)',
        }} />

        {/* Logo */}
        <div style={{
          padding: '20px 18px 18px',
          borderBottom: '1px solid var(--border)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 34,
              height: 34,
              background: 'var(--accent)',
              borderRadius: 6,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}>
              <Cpu size={16} color="#080c14" strokeWidth={2.5} />
            </div>
            <div>
              <div style={{
                fontFamily: 'var(--font-display)',
                fontSize: 18,
                fontWeight: 900,
                letterSpacing: '0.04em',
                color: 'var(--text)',
                textTransform: 'uppercase',
                lineHeight: 1.1,
              }}>HSRP</div>
              <div style={{
                fontSize: 9,
                color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.12em',
                textTransform: 'uppercase',
              }}>Vision System</div>
            </div>
          </div>
        </div>

        {/* Nav */}
        <nav style={{ flex: 1, padding: '16px 10px', display: 'flex', flexDirection: 'column', gap: 2 }}>
          <div style={{
            fontSize: 9,
            fontFamily: 'var(--font-mono)',
            color: 'var(--text-dim)',
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            padding: '4px 14px 8px',
          }}>
            Detection
          </div>
          <NavItem to="/dashboard" icon={Camera} label="Video Analysis" />

          {user?.role === 'admin' && (
            <>
              <div style={{
                fontSize: 9,
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-dim)',
                letterSpacing: '0.14em',
                textTransform: 'uppercase',
                padding: '12px 14px 8px',
              }}>
                Admin
              </div>
              <NavItem to="/admin" icon={ShieldAlert} label="Admin Panel" />
            </>
          )}
        </nav>

        {/* Bottom */}
        <div style={{
          padding: '12px 10px',
          borderTop: '1px solid var(--border)',
        }}>
          {/* System status */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            padding: '8px 14px',
            marginBottom: 4,
          }}>
            <Activity size={11} color="var(--green)" />
            <span style={{
              fontSize: 9,
              fontFamily: 'var(--font-mono)',
              color: 'var(--green)',
              letterSpacing: '0.12em',
            }}>SYSTEM ONLINE</span>
          </div>

          {/* User */}
          <div style={{
            padding: '8px 14px 10px',
            borderRadius: 'var(--radius)',
            background: 'var(--bg-elevated)',
            marginBottom: 6,
            border: '1px solid var(--border)',
          }}>
            <div style={{
              fontSize: 12,
              color: 'var(--text)',
              fontWeight: 500,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
            }}>{user?.email}</div>
            <div style={{
              fontSize: 9,
              fontFamily: 'var(--font-mono)',
              color: user?.role === 'admin' ? 'var(--accent)' : 'var(--text-muted)',
              marginTop: 2,
              textTransform: 'uppercase',
              letterSpacing: '0.1em',
            }}>{user?.role}</div>
          </div>

          <LogoutBtn />
        </div>
      </aside>

      {/* Main content */}
      <main style={{
        flex: 1,
        overflow: 'auto',
        background: 'var(--bg)',
      }}>
        <Outlet />
      </main>
    </div>
  );
}
