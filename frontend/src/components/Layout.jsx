import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { Camera, ShieldAlert, LogOut, Radio } from 'lucide-react';

const styles = {
  shell: {
    display: 'flex',
    height: '100vh',
    overflow: 'hidden',
  },
  sidebar: {
    width: 220,
    minWidth: 220,
    background: 'var(--bg-card)',
    borderRight: '1px solid var(--border)',
    display: 'flex',
    flexDirection: 'column',
    padding: '24px 0',
  },
  logo: {
    padding: '0 20px 28px',
    borderBottom: '1px solid var(--border)',
    marginBottom: 16,
  },
  logoMark: {
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  logoIcon: {
    width: 32,
    height: 32,
    background: 'var(--accent)',
    borderRadius: 8,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  logoText: {
    fontFamily: 'var(--font-display)',
    fontSize: 16,
    fontWeight: 700,
    color: 'var(--text)',
    letterSpacing: '-0.02em',
  },
  logoSub: {
    fontSize: 10,
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
    marginTop: 2,
  },
  nav: {
    flex: 1,
    padding: '0 12px',
    display: 'flex',
    flexDirection: 'column',
    gap: 2,
  },
  navLabel: {
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-dim)',
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    padding: '0 8px',
    marginBottom: 6,
    marginTop: 8,
  },
  bottom: {
    padding: '16px 12px 0',
    borderTop: '1px solid var(--border)',
  },
  userChip: {
    padding: '10px 12px',
    borderRadius: 'var(--radius)',
    marginBottom: 8,
  },
  userEmail: {
    fontSize: 12,
    color: 'var(--text)',
    fontWeight: 500,
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  userRole: {
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    color: 'var(--text-muted)',
    marginTop: 2,
  },
  main: {
    flex: 1,
    overflow: 'auto',
    background: 'var(--bg)',
  },
  liveIndicator: {
    display: 'flex',
    alignItems: 'center',
    gap: 6,
    padding: '6px 12px',
    marginBottom: 12,
  },
  liveDot: {
    width: 6,
    height: 6,
    borderRadius: '50%',
    background: 'var(--green)',
    animation: 'blink 2s ease-in-out infinite',
  },
  liveText: {
    fontSize: 10,
    fontFamily: 'var(--font-mono)',
    color: 'var(--green)',
    letterSpacing: '0.06em',
  },
};

function NavItem({ to, icon: Icon, label }) {
  return (
    <NavLink
      to={to}
      style={({ isActive }) => ({
        display: 'flex',
        alignItems: 'center',
        gap: 10,
        padding: '9px 12px',
        borderRadius: 'var(--radius)',
        textDecoration: 'none',
        fontSize: 13,
        fontWeight: 500,
        color: isActive ? 'var(--accent)' : 'var(--text-muted)',
        background: isActive ? 'var(--accent-dim)' : 'transparent',
        transition: 'all var(--transition)',
      })}
    >
      <Icon size={15} />
      {label}
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
        padding: '9px 12px', borderRadius: 'var(--radius)',
        border: 'none', cursor: 'pointer',
        fontSize: 13, fontWeight: 500,
        color: 'var(--text-muted)', background: 'transparent',
        width: '100%', transition: 'all var(--transition)',
      }}
      onMouseEnter={e => { e.currentTarget.style.color = 'var(--red)'; e.currentTarget.style.background = 'var(--red-dim)'; }}
      onMouseLeave={e => { e.currentTarget.style.color = 'var(--text-muted)'; e.currentTarget.style.background = 'transparent'; }}
    >
      <LogOut size={15} />
      Sign out
    </button>
  );
}

export default function Layout() {
  const { user } = useAuth();

  return (
    <div style={styles.shell}>
      <aside style={styles.sidebar}>
        <div style={styles.logo}>
          <div style={styles.logoMark}>
            <div style={styles.logoIcon}>
              <Radio size={16} color="#0a0a0f" strokeWidth={2.5} />
            </div>
            <div>
              <div style={styles.logoText}>HSRP</div>
              <div style={styles.logoSub}>monitor v2</div>
            </div>
          </div>
        </div>

        <nav style={styles.nav}>
          <div style={styles.navLabel}>detection</div>
          <NavItem to="/dashboard" icon={Camera} label="Video Analysis" />
          {user?.role === 'admin' && (
            <>
              <div style={styles.navLabel}>admin</div>
              <NavItem to="/admin" icon={ShieldAlert} label="Admin Panel" />
            </>
          )}
        </nav>

        <div style={styles.bottom}>
          <div style={styles.liveIndicator}>
            <div style={styles.liveDot} />
            <span style={styles.liveText}>SYSTEM ONLINE</span>
          </div>
          <div style={styles.userChip}>
            <div style={styles.userEmail}>{user?.email}</div>
            <div style={styles.userRole}>{user?.role}</div>
          </div>
          <LogoutBtn />
        </div>
      </aside>

      <main style={styles.main}>
        <Outlet />
      </main>
    </div>
  );
}
