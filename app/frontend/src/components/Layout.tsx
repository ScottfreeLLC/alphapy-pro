import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  LineChart,
  ListChecks,
  Briefcase,
  BookOpen,
  FlaskConical,
  Send,
  MessageSquare,
  Zap,
} from 'lucide-react';
import StatusBadge from './StatusBadge';
import { useAlfiWebSocket } from '../lib/websocket';
import { AgentState } from '../types/agent';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Swing Agent' },
  { to: '/day', icon: Zap, label: 'Day Agent' },
  { to: '/screener', icon: LineChart, label: 'Screener' },
  { to: '/trades', icon: ListChecks, label: 'Trades' },
  { to: '/portfolio', icon: Briefcase, label: 'Portfolio' },
  { to: '/strategies', icon: BookOpen, label: 'Strategies' },
  { to: '/backtest', icon: FlaskConical, label: 'Backtest' },
  { to: '/chat', icon: MessageSquare, label: 'Chat' },
  { to: '/publish', icon: Send, label: 'Publish' },
];

function AgentStatusLine({ label, state }: { label: string; state: AgentState | null }) {
  const statusColor = {
    stopped: 'bg-gray-500',
    starting: 'bg-yellow-500',
    running: 'bg-green-500',
    paused: 'bg-yellow-500',
    error: 'bg-red-500',
  }[state?.status ?? 'stopped'];

  return (
    <div className="flex items-center justify-between text-xs">
      <div className="flex items-center gap-1.5">
        <div className={`w-1.5 h-1.5 rounded-full ${statusColor}`} />
        <span className="text-gray-400">{label}</span>
      </div>
      <span className="text-gray-500">
        {state ? `${state.cycle_count}c / ${state.pending_signals_count}p` : 'off'}
      </span>
    </div>
  );
}

export default function Layout() {
  const { swingState, dayState, connected } = useAlfiWebSocket();

  return (
    <div className="flex h-screen bg-gray-950 text-gray-100">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 border-r border-gray-800 flex flex-col">
        <div className="p-4 border-b border-gray-800">
          <h1 className="text-xl font-bold tracking-tight">Alfi</h1>
          <p className="text-xs text-gray-500 mt-1">Dual-Agent Trading</p>
        </div>

        <nav className="flex-1 p-3 space-y-1">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-colors ${
                  isActive
                    ? 'bg-blue-600/20 text-blue-400'
                    : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800'
                }`
              }
            >
              <Icon size={18} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Status */}
        <div className="p-4 border-t border-gray-800 space-y-2">
          <StatusBadge
            status={swingState?.status || dayState?.status || 'stopped'}
            connected={connected}
          />
          <AgentStatusLine label="Swing" state={swingState} />
          <AgentStatusLine label="Day" state={dayState} />
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto">
        <Outlet context={{ swingState, dayState, connected }} />
      </main>
    </div>
  );
}
