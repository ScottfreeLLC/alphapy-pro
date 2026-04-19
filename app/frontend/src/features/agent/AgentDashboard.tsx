import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useOutletContext } from 'react-router-dom';
import { Play, Square, AlertCircle, Shield } from 'lucide-react';
import { format } from 'date-fns';
import {
  fetchAgentStatus,
  startAgent,
  stopAgent,
  setAgentMode,
  fetchPerformanceMetrics,
  fetchRiskStatus,
  resetCircuitBreaker,
} from '../../lib/api';
import { AgentType, AgentState, AutonomyMode } from '../../types/agent';
import SkillMetrics from './SkillMetrics';
import GraduationPanel from './GraduationPanel';
import PatternPanel from './PatternPanel';
import DiscoveryPanel from './DiscoveryPanel';
import EncodingPanel from './EncodingPanel';

interface Props {
  agentType: AgentType;
}

export default function AgentDashboard({ agentType }: Props) {
  const { swingState, dayState } = useOutletContext<{
    swingState: AgentState | null;
    dayState: AgentState | null;
  }>();
  const queryClient = useQueryClient();

  const wsState = agentType === 'swing' ? swingState : dayState;

  // Fallback to REST if WebSocket state not available
  const { data: restState } = useQuery({
    queryKey: ['agentStatus', agentType],
    queryFn: () => fetchAgentStatus(agentType),
    refetchInterval: 5000,
    enabled: !wsState,
  });

  const state = wsState || restState;

  // Performance metrics
  const { data: perfMetrics } = useQuery({
    queryKey: ['performanceMetrics', agentType],
    queryFn: () => fetchPerformanceMetrics(agentType),
    refetchInterval: 30000,
  });

  // Risk status
  const { data: riskStatus } = useQuery({
    queryKey: ['riskStatus', agentType],
    queryFn: () => fetchRiskStatus(agentType),
    refetchInterval: 15000,
  });

  const startMutation = useMutation({
    mutationFn: () => startAgent(agentType),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['agentStatus', agentType] }),
  });

  const stopMutation = useMutation({
    mutationFn: () => stopAgent(agentType),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['agentStatus', agentType] }),
  });

  const modeMutation = useMutation({
    mutationFn: (mode: AutonomyMode) => setAgentMode({ agentType, mode }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['agentStatus', agentType] }),
  });

  const resetCBMutation = useMutation({
    mutationFn: () => resetCircuitBreaker(agentType),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['riskStatus', agentType] }),
  });

  const isRunning = state?.status === 'running';
  const title = agentType === 'swing' ? 'Swing Agent' : 'Day Agent';

  const modeOptions: { value: AutonomyMode; label: string; desc: string }[] = [
    { value: 'approval', label: 'Approval', desc: 'All trades require manual approval' },
    { value: 'semi_autonomous', label: 'Semi-Auto', desc: 'High-confidence signals auto-execute' },
    { value: 'autonomous', label: 'Autonomous', desc: 'All signals auto-execute' },
  ];

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold">{title}</h2>
          <p className="text-sm text-gray-500 mt-1">
            {agentType === 'swing' ? 'Multi-day holds on daily bars' : 'Intraday trades on 5-min bars'}
          </p>
        </div>
        <div className="flex gap-2">
          {isRunning ? (
            <button
              onClick={() => stopMutation.mutate()}
              disabled={stopMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors disabled:opacity-50"
            >
              <Square size={16} /> Stop
            </button>
          ) : (
            <button
              onClick={() => startMutation.mutate()}
              disabled={startMutation.isPending}
              className="flex items-center gap-2 px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors disabled:opacity-50"
            >
              <Play size={16} /> Start
            </button>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Cycles', value: state?.cycle_count ?? 0 },
          { label: 'Pending Signals', value: state?.pending_signals_count ?? 0 },
          { label: 'Total Trades', value: perfMetrics?.total_trades ?? 0 },
          { label: 'Win Rate', value: perfMetrics?.win_rate ? `${(perfMetrics.win_rate * 100).toFixed(0)}%` : '—' },
        ].map(({ label, value }) => (
          <div key={label} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="text-sm text-gray-400">{label}</div>
            <div className="text-2xl font-bold mt-1">{value}</div>
          </div>
        ))}
      </div>

      {/* Performance summary */}
      {perfMetrics && perfMetrics.total_trades > 0 && (
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: 'Sharpe Ratio', value: perfMetrics.sharpe_ratio.toFixed(2) },
            { label: 'Total P&L', value: `$${perfMetrics.total_pnl.toFixed(0)}`, color: perfMetrics.total_pnl >= 0 ? 'text-green-400' : 'text-red-400' },
            { label: 'Max Drawdown', value: `${perfMetrics.max_drawdown_pct.toFixed(1)}%` },
            { label: 'Profit Factor', value: typeof perfMetrics.profit_factor === 'string' ? perfMetrics.profit_factor : perfMetrics.profit_factor.toFixed(2) },
          ].map(({ label, value, color }) => (
            <div key={label} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
              <div className="text-sm text-gray-400">{label}</div>
              <div className={`text-xl font-bold mt-1 ${color || ''}`}>{value}</div>
            </div>
          ))}
        </div>
      )}

      {/* Risk Status */}
      {riskStatus?.circuit_breaker?.tripped && (
        <div className="flex items-center justify-between bg-red-900/30 border border-red-800 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Shield size={20} className="text-red-400" />
            <div>
              <div className="text-sm font-medium text-red-300">Circuit Breaker Tripped</div>
              <div className="text-xs text-red-400">{riskStatus.circuit_breaker.reason}</div>
            </div>
          </div>
          <button
            onClick={() => resetCBMutation.mutate()}
            className="px-3 py-1.5 bg-red-600 hover:bg-red-700 rounded text-sm transition-colors"
          >
            Reset
          </button>
        </div>
      )}

      {/* Autonomy Mode */}
      <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Autonomy Mode</h3>
        <div className="flex gap-3">
          {modeOptions.map(({ value, label, desc }) => (
            <button
              key={value}
              onClick={() => modeMutation.mutate(value)}
              className={`flex-1 p-3 rounded-lg border transition-colors text-left ${
                state?.autonomy_mode === value
                  ? 'border-blue-500 bg-blue-600/10'
                  : 'border-gray-700 hover:border-gray-600'
              }`}
            >
              <div className="font-medium text-sm">{label}</div>
              <div className="text-xs text-gray-500 mt-1">{desc}</div>
            </button>
          ))}
        </div>
      </div>

      {/* Skill Metrics + Graduation side by side */}
      <div className="grid grid-cols-2 gap-4">
        <SkillMetrics agentType={agentType} />
        <GraduationPanel agentType={agentType} />
      </div>

      {/* Price Encoding (both agents) */}
      <EncodingPanel />

      {/* Pattern Classifier + Discovery (Day Agent only) */}
      {agentType === 'day' && (
        <>
          <PatternPanel />
          <DiscoveryPanel />
        </>
      )}

      {/* Error banner */}
      {state?.last_error && (
        <div className="flex items-center gap-3 bg-red-900/30 border border-red-800 rounded-lg p-4">
          <AlertCircle size={20} className="text-red-400 shrink-0" />
          <div>
            <div className="text-sm font-medium text-red-300">Last Error</div>
            <div className="text-xs text-red-400 mt-1">{state.last_error}</div>
          </div>
        </div>
      )}

      {/* Activity Feed */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800">
          <h3 className="font-medium">Activity Feed</h3>
        </div>
        <div className="divide-y divide-gray-800 max-h-80 overflow-y-auto">
          {state?.recent_activity && state.recent_activity.length > 0 ? (
            [...state.recent_activity].reverse().map((entry, i) => (
              <div key={i} className="px-4 py-3 flex items-start gap-3">
                <div
                  className={`w-1.5 h-1.5 rounded-full mt-2 shrink-0 ${
                    entry.level === 'error'
                      ? 'bg-red-500'
                      : entry.level === 'warning'
                      ? 'bg-yellow-500'
                      : 'bg-blue-500'
                  }`}
                />
                <div className="min-w-0">
                  <span className="text-sm">{entry.action}</span>
                  {entry.detail && (
                    <p className="text-xs text-gray-500 mt-0.5 truncate">{entry.detail}</p>
                  )}
                </div>
                <span className="text-xs text-gray-600 ml-auto shrink-0">
                  {format(new Date(entry.timestamp), 'HH:mm:ss')}
                </span>
              </div>
            ))
          ) : (
            <div className="p-8 text-center text-gray-500 text-sm">
              No activity yet. Start the agent to begin.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
