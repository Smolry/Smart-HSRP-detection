import { createContext, useContext, useState, useEffect, useCallback } from 'react';

const AuthContext = createContext(null);
const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const stored = localStorage.getItem('hsrp_token');
    const storedUser = localStorage.getItem('hsrp_user');
    if (stored && storedUser) {
      try { setUser(JSON.parse(storedUser)); } catch {}
    }
    setLoading(false);
  }, []);

  const login = useCallback(async (email, password) => {
    const resp = await fetch(`${API}/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Login failed');
    localStorage.setItem('hsrp_token', data.access_token);
    localStorage.setItem('hsrp_user', JSON.stringify({ email: data.email, role: data.role, id: data.user_id }));
    setUser({ email: data.email, role: data.role, id: data.user_id });
    return data;
  }, []);

  const signup = useCallback(async (email, password, role = 'user') => {
    const resp = await fetch(`${API}/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, role }),
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.detail || 'Signup failed');
    localStorage.setItem('hsrp_token', data.access_token);
    localStorage.setItem('hsrp_user', JSON.stringify({ email: data.email, role: data.role, id: data.user_id }));
    setUser({ email: data.email, role: data.role, id: data.user_id });
    return data;
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem('hsrp_token');
    localStorage.removeItem('hsrp_user');
    setUser(null);
  }, []);

  const token = localStorage.getItem('hsrp_token');

  return (
    <AuthContext.Provider value={{ user, token, loading, login, signup, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}

export { API };
