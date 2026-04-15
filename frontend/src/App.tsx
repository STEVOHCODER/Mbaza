import { useState, useEffect, useRef } from 'react';
import { Send, Bot, PlusCircle, PanelLeftOpen, Copy, Check, X, Layout, MessageSquare, Trash2, MoreVertical, Calendar as CalendarIcon, Heart, LogOut } from 'lucide-react';

import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'highlight.js/styles/tokyo-night-dark.css';
import 'katex/dist/katex.min.css';
import { useAuth } from './context/AuthContext';
import { Login } from './components/Login';
import { Register } from './components/Register';

const ProjectMenu = ({ onRename, onDelete }: any) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  useEffect(() => { const handleClickOutside = (e: any) => { if (menuRef.current && !menuRef.current.contains(e.target)) setIsOpen(false); }; document.addEventListener('mousedown', handleClickOutside); return () => document.removeEventListener('mousedown', handleClickOutside); }, []);
  return (
    <div className="relative" ref={menuRef}>
      <button onClick={(e) => { e.stopPropagation(); setIsOpen(!isOpen); }} className="p-1.5 text-[#6b7fa8] hover:text-white rounded-lg"><MoreVertical size={14} /></button>
      {isOpen && (
        <div className="absolute right-0 top-8 w-40 bg-[#161b22] border border-[#30363d] rounded-xl shadow-2xl z-[60] overflow-hidden animate-in fade-in slide-in-from-top-2">
          <button onClick={(e) => { e.stopPropagation(); onRename(); setIsOpen(false); }} className="flex items-center gap-2 w-full px-4 py-2.5 text-[11px] font-bold text-[#c9d1d9] hover:bg-[#29d8a8]/10 border-b border-[#30363d]/50">RENAME</button>
          <button onClick={(e) => { e.stopPropagation(); onDelete(); setIsOpen(false); }} className="flex items-center gap-2 w-full px-4 py-2.5 text-[11px] font-bold text-red-400 hover:bg-red-400/10">DELETE</button>
        </div>
      )}
    </div>
  );
};

const CalendarView = ({ events, onAdd }: any) => {
  return (
    <div className="flex-1 bg-[#0d1117] p-8 overflow-y-auto custom-scroll">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-10">
          <h2 className="text-2xl font-black text-white tracking-tight flex items-center gap-3"><CalendarIcon className="text-[#29d8a8]" /> Schedule</h2>
          <button onClick={onAdd} className="flex items-center gap-2 px-6 py-2.5 bg-[#29d8a8] text-[#0a0c10] font-black text-xs uppercase tracking-widest rounded-full hover:scale-105 transition-all shadow-glow"><PlusCircle size={16} /> New Entry</button>
        </div>
        <div className="grid gap-4">
          {events.length === 0 ? (
            <div className="text-center py-20 bg-[#161b22] border border-[#30363d] rounded-[2.5rem] opacity-50">
              <CalendarIcon size={40} className="mx-auto mb-4 text-[#6b7fa8]" />
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-[#6b7fa8]">Your calendar is clear.</p>
            </div>
          ) : (
            events.sort((a: any, b: any) => new Date(a.event_date).getTime() - new Date(b.event_date).getTime()).map((e: any) => (
              <div key={e.id} className="bg-[#161b22] border border-[#30363d] rounded-[1.5rem] p-6 flex items-center justify-between group hover:border-[#29d8a8]/30 transition-all shadow-xl">
                <div className="flex items-center gap-6">
                  <div className="w-14 h-14 bg-gradient-to-br from-[#161b22] to-[#0d1117] border border-[#30363d] rounded-2xl flex flex-col items-center justify-center shadow-inner">
                    <span className="text-[10px] font-black text-[#6b7fa8] uppercase">{new Date(e.event_date).toLocaleString('default', { month: 'short' })}</span>
                    <span className="text-lg font-black text-white">{new Date(e.event_date).getDate()}</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-black text-white uppercase tracking-tight">{e.title}</h4>
                    <span className="text-[9px] font-bold text-[#29d8a8] uppercase tracking-[0.2em] mt-1 block">{e.category}</span>
                  </div>
                </div>
                <button className="p-2 text-[#6b7fa8] hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"><Trash2 size={16} /></button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

const PeriodView = ({ entries, onAdd }: any) => {
  return (
    <div className="flex-1 bg-[#0d1117] p-8 overflow-y-auto custom-scroll">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-10">
          <h2 className="text-2xl font-black text-white tracking-tight flex items-center gap-3"><Heart size={24} className="text-[#ff7eb3]" /> My Cycles</h2>
          <button onClick={onAdd} className="flex items-center gap-2 px-6 py-2.5 bg-[#ff7eb3] text-[#0a0c10] font-black text-xs uppercase tracking-widest rounded-full hover:scale-105 transition-all shadow-[0_0_20px_rgba(255,126,179,0.3)]"><PlusCircle size={16} /> Log Period</button>
        </div>

        <div className="grid grid-cols-2 gap-6 mb-10">
          <div className="bg-gradient-to-br from-[#161b22] to-[#0d1117] border border-[#30363d] rounded-[2.5rem] p-8 shadow-2xl relative overflow-hidden group">
            <div className="absolute -right-4 -top-4 w-24 h-24 bg-[#ff7eb3]/5 rounded-full group-hover:scale-150 transition-all duration-700" />
            <h3 className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-[0.2em] mb-4 relative z-10">Status</h3>
            <div className="flex items-end gap-2 relative z-10">
              <span className="text-4xl font-black text-white tracking-tighter">Tracking</span>
              <div className="w-2 h-2 rounded-full bg-[#29d8a8] mb-2 animate-pulse" />
            </div>
          </div>
          <div className="bg-[#161b22] border border-[#30363d] rounded-[2.5rem] p-8 shadow-2xl relative overflow-hidden group">
            <h3 className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-[0.2em] mb-4 relative z-10">AI Insights</h3>
            <p className="text-[11px] text-[#c9d1d9] leading-relaxed relative z-10 font-medium">Your data is synced with Mbaza for personalized advice.</p>
          </div>
        </div>

        <div className="grid gap-4">
          {entries.length === 0 ? (
            <div className="text-center py-20 bg-[#161b22] border border-[#30363d] rounded-[2.5rem] opacity-50 shadow-inner">
              <Heart size={40} className="mx-auto mb-4 text-[#6b7fa8]" />
              <p className="text-xs font-bold uppercase tracking-[0.2em] text-[#6b7fa8]">No data logged yet.</p>
            </div>
          ) : (
            entries.sort((a: any, b: any) => new Date(b.start_date).getTime() - new Date(a.start_date).getTime()).map((e: any) => (
              <div key={e.id} className="bg-[#161b22] border border-[#30363d] rounded-[1.5rem] p-6 flex items-center justify-between group hover:border-[#ff7eb3]/30 transition-all shadow-xl">
                <div className="flex items-center gap-6">
                  <div className="w-14 h-14 bg-gradient-to-br from-[#161b22] to-[#0d1117] border border-[#ff7eb3]/20 rounded-2xl flex flex-col items-center justify-center">
                    <span className="text-[10px] font-black text-[#ff7eb3] uppercase tracking-tighter">{new Date(e.start_date).toLocaleString('default', { month: 'short' })}</span>
                    <span className="text-lg font-black text-white">{new Date(e.start_date).getDate()}</span>
                  </div>
                  <div>
                    <h4 className="text-sm font-black text-white uppercase tracking-tight italic flex items-center gap-2">Intensity: <span className={e.intensity === 'heavy' ? 'text-red-400' : e.intensity === 'medium' ? 'text-orange-300' : 'text-yellow-200'}>{e.intensity}</span></h4>
                    <p className="text-[10px] text-[#6b7fa8] mt-1 font-medium">{e.notes || "Standard entry"}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <button onClick={() => {
                    const input = document.getElementById('main-input') as HTMLInputElement;
                    if (input) {
                      input.value = `Hi, can you explain my period log from ${new Date(e.start_date).toLocaleDateString()}? Flow was ${e.intensity}.`;
                      input.focus();
                    }
                  }} className="px-5 py-2 bg-[#0d1117] border border-[#30363d] rounded-full text-[9px] font-black text-[#6b7fa8] hover:text-[#ff7eb3] hover:border-[#ff7eb3] transition-all uppercase tracking-widest shadow-inner">Consult Mbaza</button>
                  <button className="p-2 text-[#6b7fa8] hover:text-red-400 opacity-0 group-hover:opacity-100 transition-all"><Trash2 size={16} /></button>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default function App() {
  const { user, isAuthenticated, loading, logout } = useAuth();
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');

  const [sessions, setSessions] = useState<any[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [messages, setMessages] = useState<any[]>([]);
  const [status, setStatus] = useState('Ready');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttempts = useRef(0);

  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isWorkspaceOpen, setIsWorkspaceOpen] = useState(true);
  const [activeTab, setActiveTab] = useState<'calendar' | 'period'>('period');
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const [calendarEvents, setCalendarEvents] = useState<any[]>([]);
  const [periodEntries, setPeriodEntries] = useState<any[]>([]);
  const [showAddEvent, setShowAddEvent] = useState(false);
  const [showAddPeriod, setShowAddPeriod] = useState(false);
  const [newEvent, setNewEvent] = useState({ title: '', date: '', category: 'health' });
  const [newPeriod, setNewPeriod] = useState({ start_date: '', intensity: 'medium', notes: '' });

  const [isAiTyping, setIsAiTyping] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [editingSessionId, setEditingSessionId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const userSessionsKey = user?.id ? `mbaza_sessions_${user.id}` : '';

  useEffect(() => {
    if (!isAuthenticated || !userSessionsKey) return;
    const saved = localStorage.getItem(userSessionsKey);
    setCurrentSessionId('');
    setMessages([]);
    setSessions([]);
    if (saved) {
      const parsed = JSON.parse(saved);
      setSessions(parsed);
      if (parsed.length > 0) loadSession(parsed[0].id, parsed);
      else createNewSession([]);
    } else createNewSession([]);
  }, [isAuthenticated, userSessionsKey]);

  useEffect(() => {
    if (isAuthenticated && userSessionsKey && sessions.length > 0) {
      localStorage.setItem(userSessionsKey, JSON.stringify(sessions));
    }
  }, [sessions, isAuthenticated, userSessionsKey]);

  useEffect(() => { if (currentSessionId) setSessions(prev => prev.map(s => s.id === currentSessionId ? { ...s, messages } : s)); }, [messages]);
  useEffect(() => { if (messagesEndRef.current) messagesEndRef.current.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const createNewSession = (currentList?: any[]) => {
    const id = Math.random().toString(36).substring(7);
    const newSess = { id, title: 'New Consultation', messages: [] };
    const list = currentList || sessions;
    const updated = [newSess, ...list];
    setSessions(updated);
    loadSession(id, updated);
  };

  const loadSession = (id: string, currentList?: any[]) => {
    if (currentSessionId === id) return;
    reconnectAttempts.current = 0; // reset reconnect counter on session switch
    setCurrentSessionId(id);
    const list = currentList || sessions;
    const sess = list.find((s: any) => s.id === id);
    setMessages(sess?.messages || []);
  };

  // ✅ Robust API + WS URL handling
  const API_BASE_URL =
    import.meta.env.VITE_API_URL || "https://mbaza-mzf3.onrender.com";

  const WS_BASE_URL = API_BASE_URL.replace(/^http/, "ws");
  // ─── WebSocket with auto-reconnect ───────────────────────────────────────
  useEffect(() => {
    if (!currentSessionId || !isAuthenticated) return;

    let isMounted = true;
    let shouldReconnect = true;

    const connect = () => {
      if (!isMounted) return;

      const socket = new WebSocket(
        `${WS_BASE_URL}/ws/${currentSessionId}`
      );

      wsRef.current = socket;

      socket.onopen = () => {
        if (!isMounted) return;
        reconnectAttempts.current = 0;
        setStatus("Mbaza Online");
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const reply =
            typeof data.reply === 'string'
              ? data.reply
              : typeof data.content === 'string'
                ? data.content
                : '';

          if (!reply) return;

          setMessages((prev) => [
            ...prev,
            { role: "assistant", content: reply },
          ]);
          setStatus("Ready");
          setIsAiTyping(false);
        } catch (error) {
          console.error('Invalid WebSocket message:', error);
          setStatus('Message error');
          setIsAiTyping(false);
        }
      };

      socket.onclose = () => {
        if (!isMounted || !shouldReconnect) return;

        const attempts = reconnectAttempts.current;
        const delay = Math.min(1000 * Math.pow(2, attempts), 10000);

        setStatus(`Reconnecting in ${Math.round(delay / 1000)}s…`);

        reconnectAttempts.current += 1;
        reconnectRef.current = setTimeout(connect, delay);
      };

      socket.onerror = () => {
        socket.close();
      };
    };

    connect();

    return () => {
      isMounted = false;
      shouldReconnect = false;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [currentSessionId, isAuthenticated]);
  // ─────────────────────────────────────────────────────────────────────────

  const sendMessage = async () => {
    const inputEl = document.getElementById('main-input') as HTMLInputElement;
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && inputEl.value.trim() && !isAiTyping) {
      const text = inputEl.value.trim();
      wsRef.current.send(text);
      setMessages(prev => [...prev, { role: 'user', content: text }]);
      inputEl.value = ''; setStatus('Thinking...'); setIsAiTyping(true);
    }
  };

  const fetchCalendar = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/calendar/${id}`);
      const data = await res.json();
      setCalendarEvents(data);
    } catch (e) { console.error(e); }
  };

  const addCalendarEvent = async () => {
    if (!newEvent.title || !newEvent.date) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/calendar/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newEvent.title, description: '', event_date: newEvent.date, category: newEvent.category })
      });
      if (res.ok) {
        setNewEvent({ title: '', date: '', category: 'health' });
        setShowAddEvent(false);
        fetchCalendar(currentSessionId);
      }
    } catch (e) { console.error(e); }
  };

  const fetchPeriods = async (id: string) => {
    try {
      const res = await fetch(`${API_BASE_URL}/api/period/${id}`);
      const data = await res.json();
      setPeriodEntries(data);
    } catch (e) { console.error(e); }
  };

  const addPeriodEntry = async () => {
    if (!newPeriod.start_date) return;
    try {
      const res = await fetch(`${API_BASE_URL}/api/period/${currentSessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_date: newPeriod.start_date, intensity: newPeriod.intensity, notes: newPeriod.notes })
      });
      if (res.ok) {
        setNewPeriod({ start_date: '', intensity: 'medium', notes: '' });
        setShowAddPeriod(false);
        fetchPeriods(currentSessionId);
      }
    } catch (e) { console.error(e); }
  };

  useEffect(() => {
    if (currentSessionId) {
      fetchCalendar(currentSessionId);
      fetchPeriods(currentSessionId);
    }
  }, [currentSessionId]);

  const deleteSession = (id: string) => { const updated = sessions.filter(s => s.id !== id); setSessions(updated); if (currentSessionId === id && updated.length > 0) loadSession(updated[0].id, updated); else if (updated.length === 0) createNewSession([]); };
  const startRenaming = (id: string, title: string) => { setEditingSessionId(id); setEditTitle(title); };
  const saveRename = () => { if (editingSessionId && editTitle.trim()) { setSessions(prev => prev.map(s => s.id === editingSessionId ? { ...s, title: editTitle } : s)); setEditingSessionId(null); } };
  const copyText = (text: string, id: string) => { navigator.clipboard.writeText(text); setCopiedId(id); setTimeout(() => setCopiedId(null), 2000); };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-[#0a0c10] to-[#161b22] flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-12 h-12 border-4 border-[#30363d] border-t-[#29d8a8] rounded-full animate-spin"></div>
          <p className="text-[#6b7fa8] font-medium">Loading Mbaza...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return authMode === 'login' ? (
      <Login onSwitchToRegister={() => setAuthMode('register')} />
    ) : (
      <Register onSwitchToLogin={() => setAuthMode('login')} />
    );
  }

  return (
    <div className="flex h-screen w-screen bg-[#0a0c10] text-[#c9d1d9] font-['Inter',sans-serif] overflow-hidden antialiased">
      {isSidebarOpen && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden" onClick={() => setIsSidebarOpen(false)} />
      )}

      <div className={"fixed md:relative shrink-0 transition-all duration-500 bg-[#0d1117] border-r border-[#30363d] flex flex-col z-50 h-full " + (isSidebarOpen ? 'w-72 translate-x-0' : 'w-0 -translate-x-full md:translate-x-0 md:w-0 overflow-hidden opacity-0')}>
        <div className="p-8 border-b border-[#30363d] flex justify-between items-center"><div className="flex items-center gap-3"><Heart size={18} className="text-[#ff7eb3]" /><span className="text-[11px] font-black tracking-[0.2em] text-white uppercase">MBAZA</span></div><button onClick={() => setIsSidebarOpen(false)} className="p-1.5 text-[#6b7fa8] hover:text-white rounded-lg"><X size={16} /></button></div>
        <div className="flex-1 flex flex-col p-6 overflow-hidden">
          <button onClick={() => createNewSession()} className="flex items-center justify-center gap-2 w-full px-4 py-4 text-xs font-black text-[#0a0c10] bg-white hover:bg-gray-200 rounded-full shadow-lg mb-8 tracking-widest uppercase transition-all hover:scale-105">NEW CONSULTATION</button>
          <div className="flex-1 overflow-y-auto space-y-2 custom-scroll pb-4">
            {sessions.map(s => (
              <div key={s.id} onClick={() => loadSession(s.id)} className={"group flex items-center justify-between w-full px-4 py-3.5 text-xs rounded-2xl cursor-pointer transition-all border " + (currentSessionId === s.id ? 'bg-[#161b22] text-white border-[#30363d]' : 'border-transparent text-[#6b7fa8] hover:bg-[#161b22]/50')}>
                <div className="flex items-center gap-3 flex-1 min-w-0"><MessageSquare size={14} className={currentSessionId === s.id ? "text-[#29d8a8]" : ""} />{editingSessionId === s.id ? <input value={editTitle} onChange={(e) => setEditTitle(e.target.value)} onBlur={saveRename} onKeyDown={(e) => e.key === 'Enter' && saveRename()} className="bg-transparent text-white outline-none w-full border-b border-[#29d8a8]" autoFocus onClick={(e) => e.stopPropagation()} /> : <span className="truncate font-bold tracking-tight">{s.title}</span>}</div>
                <ProjectMenu onRename={() => startRenaming(s.id, s.title)} onDelete={() => deleteSession(s.id)} />
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="flex-1 flex flex-col min-w-0 bg-[#0a0c10] relative">
        <header className="shrink-0 h-16 border-b border-[#30363d] bg-[#0d1117]/50 backdrop-blur-xl flex items-center px-4 md:px-6 justify-between z-20">
          <div className="flex items-center gap-2 md:gap-4">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 text-[#6b7fa8] hover:text-white bg-[#161b22] border border-[#30363d] rounded-xl transition-all"><PanelLeftOpen size={20} /></button>
            <h2 className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-[0.2em] md:tracking-[0.4em] flex items-center gap-2 italic">
              <div className={"w-2 h-2 rounded-full animate-pulse " + (isAiTyping ? 'bg-[#29d8a8]' : status.startsWith('Reconnecting') ? 'bg-yellow-400' : status === 'Mbaza Online' || status === 'Ready' ? 'bg-[#29d8a8]' : 'bg-gray-600')} />
              <span className="hidden sm:inline">{status}</span>
            </h2>
          </div>
          <div className="flex items-center gap-2 md:gap-3">
            <button onClick={() => setIsWorkspaceOpen(!isWorkspaceOpen)} className="flex items-center gap-2 px-3 md:px-4 py-2 bg-[#161b22] border border-[#30363d] rounded-full text-[9px] font-black text-[#c9d1d9] hover:bg-[#30363d] transition-all uppercase tracking-widest shadow-xl">
              <Layout size={14} /> <span className="hidden sm:inline">{isWorkspaceOpen ? 'Hide Trackers' : 'Show Trackers'}</span>
            </button>
            <button onClick={logout} className="flex items-center gap-2 px-3 md:px-4 py-2 bg-red-500/10 border border-red-500/30 rounded-full text-[9px] font-black text-red-400 hover:bg-red-500/20 transition-all uppercase tracking-widest shadow-xl">
              <LogOut size={14} /> <span className="hidden sm:inline">Logout</span>
            </button>
          </div>
        </header>

        <div className="flex-1 flex overflow-hidden w-full h-full relative">
          <div className="flex flex-col h-full flex-1 min-w-0 bg-gradient-to-b from-[#0a0c10] to-[#0d1117]">
            <div className="flex-1 overflow-y-auto p-4 md:p-10 space-y-6 md:space-y-10 custom-scroll">
              {messages.length === 0 && <div className="text-center mt-10 md:mt-20 max-w-xl mx-auto animate-in fade-in zoom-in duration-700"><div className="w-16 h-16 md:w-24 md:h-24 bg-gradient-to-tr from-[#ff7eb3]/10 to-[#29d8a8]/10 border border-[#30363d] rounded-[2rem] md:rounded-[3rem] mx-auto flex items-center justify-center shadow-2xl mb-6 md:mb-8"><Bot size={30} className="text-[#29d8a8]" /></div><h3 className="text-white text-xl md:text-3xl font-black tracking-tight tracking-[0.1em]">Welcome to Mbaza</h3><p className="text-[#6b7fa8] text-xs md:text-sm font-medium tracking-tight mt-4 max-w-sm mx-auto leading-relaxed px-4">Your professional and private space for health, cycle tracking, and educational consultation.</p></div>}
              {messages.map((m, i) => (
                <div key={i} className={"flex gap-3 md:gap-6 " + (m.role === 'user' ? 'justify-end' : 'justify-start') + " w-full group/msg animate-in slide-in-from-bottom-4 duration-300"}>
                  {m.role === 'assistant' && <div className="w-8 h-8 md:w-10 md:h-10 rounded-xl md:rounded-2xl bg-gradient-to-br from-[#161b22] to-[#0d1117] border border-[#30363d] flex items-center justify-center shrink-0 mt-1 shadow-2xl"><Bot size={14} className="text-[#29d8a8]" /></div>}
                  <div className={"max-w-[90%] md:max-w-[85%] rounded-[1.5rem] md:rounded-[2rem] p-4 md:p-8 shadow-2xl relative " + (m.role === 'user' ? 'bg-[#161b22] border border-[#30363d] text-white rounded-tr-none' : 'bg-[#0d1117]/50 border border-[#30363d]/50 text-[#c9d1d9]')}>
                    <div className="absolute top-2 right-4 md:right-6 flex items-center gap-3">
                      <button onClick={() => copyText(m.content, "msg-" + i)} className="p-1 text-[#6b7fa8] hover:text-white transition-all opacity-0 group-hover/msg:opacity-100">{copiedId === ("msg-" + i) ? <Check size={12} className="text-[#29d8a8]" /> : <Copy size={12} />}</button>
                    </div>
                    <div className="prose prose-invert prose-sm md:prose-base max-w-none text-[#c9d1d9] custom-markdown pt-2"><ReactMarkdown remarkPlugins={[remarkMath]} rehypePlugins={[rehypeHighlight, rehypeKatex]}>{m.content}</ReactMarkdown></div>
                  </div>
                </div>
              ))}
              <div ref={messagesEndRef} />
            </div>
            <div className="shrink-0 p-4 md:p-8">
              <div className="max-w-4xl mx-auto relative group">
                <div className="absolute -inset-1 bg-gradient-to-r from-[#29d8a8]/20 to-[#ff7eb3]/20 rounded-[1.5rem] md:rounded-[2.5rem] blur opacity-30 group-hover:opacity-100 transition duration-1000 group-hover:duration-200"></div>
                <div className="relative flex items-center w-full shadow-2xl rounded-[1.2rem] md:rounded-[2rem] bg-[#161b22] border border-[#30363d]">
                  <input id="main-input" type="text" onKeyDown={(e) => e.key === 'Enter' && sendMessage()} placeholder="Message Mbaza..." className="w-full bg-transparent pl-5 md:pl-8 pr-14 md:pr-16 py-4 md:py-6 text-xs md:text-sm font-medium focus:outline-none text-white placeholder-[#6b7fa8]" />
                  <button onClick={sendMessage} disabled={isAiTyping} className="absolute right-2 md:right-3 p-3 md:p-4 text-[#0a0c10] bg-[#29d8a8] hover:bg-[#34eeb0] rounded-[1rem] md:rounded-[1.5rem] active:scale-95 shadow-lg transition-all disabled:opacity-50"><Send size={18} /></button>
                </div>
              </div>
            </div>
          </div>

          {isWorkspaceOpen && (
            <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40 md:hidden" onClick={() => setIsWorkspaceOpen(false)} />
          )}

          <div className={"fixed md:relative right-0 shrink-0 bg-[#0d1117] flex flex-col h-full border-l border-[#30363d] transition-all duration-700 z-50 " + (isWorkspaceOpen ? 'w-[90%] sm:w-[450px] md:w-[450px] lg:w-[600px] translate-x-0' : 'w-0 translate-x-full md:translate-x-0 md:w-0 overflow-hidden opacity-0')}>
            <div className="shrink-0 h-16 border-b border-[#30363d] flex items-center px-4 md:px-8 justify-between bg-[#161b22]/50 backdrop-blur-xl">
              <div className="flex bg-[#0d1117] border border-[#30363d] rounded-full p-1 shadow-inner scale-90 origin-left">
                <button onClick={() => setActiveTab('period')} className={"flex items-center gap-2 px-4 md:px-6 py-2 rounded-full text-[9px] md:text-[10px] font-black transition-all " + (activeTab === 'period' ? 'bg-[#ff7eb3] text-[#0a0c10] shadow-lg' : 'text-[#6b7fa8] hover:text-white')}>CYCLES</button>
                <button onClick={() => setActiveTab('calendar')} className={"flex items-center gap-2 px-4 md:px-6 py-2 rounded-full text-[9px] md:text-[10px] font-black transition-all " + (activeTab === 'calendar' ? 'bg-[#29d8a8] text-[#0a0c10] shadow-lg' : 'text-[#6b7fa8] hover:text-white')}>CALENDAR</button>
              </div>
              <button onClick={() => setIsWorkspaceOpen(false)} className="md:hidden p-2 text-[#6b7fa8] hover:text-white"><X size={20} /></button>
            </div>
            <div className="flex-1 bg-gradient-to-b from-[#0d1117] to-[#0a0c10] overflow-hidden flex flex-col h-full relative">
              <div className={"flex-1 h-full " + (activeTab === 'calendar' ? 'flex' : 'hidden')}><CalendarView events={calendarEvents} onAdd={() => setShowAddEvent(true)} /></div>
              <div className={"flex-1 h-full " + (activeTab === 'period' ? 'flex' : 'hidden')}><PeriodView entries={periodEntries} onAdd={() => setShowAddPeriod(true)} /></div>
            </div>
          </div>
        </div>
      </div>

      {showAddEvent && (<div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/85 backdrop-blur-md animate-in fade-in duration-300"><div className="bg-[#0d1117] border border-[#30363d] w-full max-w-md rounded-[3rem] p-12 shadow-2xl relative border-t-[#29d8a8]/30"><button onClick={() => setShowAddEvent(false)} className="absolute top-8 right-8 text-[#6b7fa8] hover:text-white transition-all"><X size={24} /></button><h2 className="text-2xl font-black text-white tracking-tight mb-10">New Calendar Event</h2><div className="space-y-5"><div className="space-y-2"><label className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-widest ml-2">Event Title</label><input type="text" value={newEvent.title} onChange={e => setNewEvent({ ...newEvent, title: e.target.value })} className="w-full bg-[#161b22] border border-[#30363d] rounded-2xl px-6 py-4 text-sm text-white focus:border-[#29d8a8] outline-none transition-all" placeholder="Health checkup, reminder, etc." /></div><div className="space-y-2"><label className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-widest ml-2">Date</label><input type="date" value={newEvent.date} onChange={e => setNewEvent({ ...newEvent, date: e.target.value })} className="w-full bg-[#161b22] border border-[#30363d] rounded-2xl px-6 py-4 text-sm text-white focus:border-[#29d8a8] outline-none appearance-none" /></div><button onClick={addCalendarEvent} className="w-full py-5 bg-[#29d8a8] text-[#0a0c10] font-black text-xs uppercase tracking-[0.2em] rounded-2xl mt-6 hover:scale-[1.02] active:scale-95 transition-all shadow-glow">Schedule Event</button></div></div></div>)}
      {showAddPeriod && (<div className="fixed inset-0 z-[110] flex items-center justify-center bg-black/85 backdrop-blur-md animate-in fade-in duration-300"><div className="bg-[#0d1117] border border-[#30363d] w-full max-w-md rounded-[3rem] p-12 shadow-2xl relative border-t-[#ff7eb3]/30"><button onClick={() => setShowAddPeriod(false)} className="absolute top-8 right-8 text-[#6b7fa8] hover:text-white transition-all"><X size={24} /></button><h2 className="text-2xl font-black text-white tracking-tight mb-10 text-center">Log My Cycle</h2><div className="space-y-5"><div className="space-y-2"><label className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-widest ml-2">Start Date</label><input type="date" value={newPeriod.start_date} onChange={e => setNewPeriod({ ...newPeriod, start_date: e.target.value })} className="w-full bg-[#161b22] border border-[#30363d] rounded-2xl px-6 py-4 text-sm text-white focus:border-[#ff7eb3] outline-none" /></div><div className="space-y-2"><label className="text-[10px] font-black text-[#6b7fa8] uppercase tracking-widest ml-2">Intensity</label><select value={newPeriod.intensity} onChange={e => setNewPeriod({ ...newPeriod, intensity: e.target.value })} className="w-full bg-[#161b22] border border-[#30363d] rounded-2xl px-6 py-4 text-sm text-white focus:border-[#ff7eb3] outline-none appearance-none"><option value="light">Light Flow</option><option value="medium">Medium Flow</option><option value="heavy">Heavy Flow</option></select></div><textarea value={newPeriod.notes} onChange={e => setNewPeriod({ ...newPeriod, notes: e.target.value })} className="w-full bg-[#161b22] border border-[#30363d] rounded-2xl p-6 text-sm text-white h-32 focus:border-[#ff7eb3] outline-none" placeholder="Notes on mood, symptoms, or energy..." /><button onClick={addPeriodEntry} className="w-full py-5 bg-[#ff7eb3] text-[#0a0c10] font-black text-xs uppercase tracking-[0.2em] rounded-2xl mt-6 hover:scale-[1.02] active:scale-95 transition-all shadow-[0_0_20px_rgba(255,126,179,0.3)]">Save to My History</button></div></div></div>)}

      <style>{" .custom-scroll::-webkit-scrollbar { width: 4px; } .custom-scroll::-webkit-scrollbar-track { background: transparent; } .custom-scroll::-webkit-scrollbar-thumb { background: #30363d; border-radius: 10px; } .custom-markdown pre { background: #0d1117 !important; padding: 1.5rem !important; border-radius: 1.5rem !important; border: 1px solid #30363d !important; overflow-x: auto !important; } .custom-markdown code { font-family: 'JetBrains Mono', monospace !important; color: #29d8a8; font-size: 0.85rem; } .shadow-glow { filter: drop-shadow(0 0 12px rgba(41,216,168,0.4)); } "}</style>
    </div>
  );
}
