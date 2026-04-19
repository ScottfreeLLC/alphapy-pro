import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Check, X, TrendingUp, TrendingDown } from 'lucide-react';
import { format } from 'date-fns';
import { fetchPendingSignals, fetchRecentSignals, approveSignal, rejectSignal } from '../../lib/api';

export default function TradeProposals() {
  const queryClient = useQueryClient();

  const { data: pendingData } = useQuery({
    queryKey: ['pendingSignals'],
    queryFn: fetchPendingSignals,
    refetchInterval: 3000,
  });

  const { data: recentData } = useQuery({
    queryKey: ['recentSignals'],
    queryFn: fetchRecentSignals,
    refetchInterval: 5000,
  });

  const approveMutation = useMutation({
    mutationFn: approveSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pendingSignals'] });
      queryClient.invalidateQueries({ queryKey: ['recentSignals'] });
    },
  });

  const rejectMutation = useMutation({
    mutationFn: rejectSignal,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['pendingSignals'] });
      queryClient.invalidateQueries({ queryKey: ['recentSignals'] });
    },
  });

  const pending = pendingData?.signals || [];
  const recent = recentData?.signals || [];

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <h2 className="text-3xl font-bold">Trade Proposals</h2>

      {/* Pending Signals */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <h3 className="font-medium">Pending Approval</h3>
          <span className="text-xs bg-yellow-600/20 text-yellow-400 px-2 py-1 rounded">
            {pending.length} pending
          </span>
        </div>

        {pending.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">
            No signals awaiting approval
          </div>
        ) : (
          <div className="divide-y divide-gray-800">
            {pending.map((signal) => (
              <div key={signal.id} className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    {signal.direction === 'long' ? (
                      <TrendingUp size={20} className="text-green-400" />
                    ) : (
                      <TrendingDown size={20} className="text-red-400" />
                    )}
                    <div>
                      <span className="font-bold text-lg">{signal.symbol}</span>
                      <span className="ml-2 text-xs text-gray-400 uppercase">
                        {signal.direction}
                      </span>
                    </div>
                    {signal.agent_type && (
                      <span className={`text-xs px-2 py-1 rounded ${
                        signal.agent_type === 'swing'
                          ? 'bg-blue-900/50 text-blue-400'
                          : 'bg-purple-900/50 text-purple-400'
                      }`}>
                        {signal.agent_type}
                      </span>
                    )}
                    <span className="text-xs bg-gray-800 px-2 py-1 rounded">
                      {signal.skill_name}
                    </span>
                  </div>

                  <div className="flex gap-2">
                    <button
                      onClick={() => approveMutation.mutate(signal.id)}
                      disabled={approveMutation.isPending}
                      className="flex items-center gap-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 rounded text-sm transition-colors"
                    >
                      <Check size={14} /> Approve
                    </button>
                    <button
                      onClick={() => rejectMutation.mutate(signal.id)}
                      disabled={rejectMutation.isPending}
                      className="flex items-center gap-1 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-sm transition-colors"
                    >
                      <X size={14} /> Reject
                    </button>
                  </div>
                </div>

                {/* Signal details */}
                <div className="grid grid-cols-4 gap-4 text-sm mb-3">
                  <div>
                    <span className="text-gray-500">Entry</span>
                    <div className="font-mono">${signal.entry_price.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Stop</span>
                    <div className="font-mono text-red-400">${signal.stop_loss.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">Target</span>
                    <div className="font-mono text-green-400">${signal.take_profit.toFixed(2)}</div>
                  </div>
                  <div>
                    <span className="text-gray-500">R:R</span>
                    <div className="font-mono">{signal.risk_reward_ratio.toFixed(1)}</div>
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-xs text-gray-500 w-20">Confidence</span>
                  <div className="flex-1 bg-gray-800 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        signal.confidence >= 0.8
                          ? 'bg-green-500'
                          : signal.confidence >= 0.5
                          ? 'bg-yellow-500'
                          : 'bg-red-500'
                      }`}
                      style={{ width: `${signal.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs font-mono w-12 text-right">
                    {(signal.confidence * 100).toFixed(0)}%
                  </span>
                </div>

                {/* Reasoning */}
                <p className="text-xs text-gray-400 mt-2">{signal.reasoning}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent Signals */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800">
          <h3 className="font-medium">Recent Signals</h3>
        </div>

        {recent.length === 0 ? (
          <div className="p-8 text-center text-gray-500 text-sm">
            No recent signals
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-gray-500 text-xs">
                <tr className="border-b border-gray-800">
                  <th className="text-left p-3">Symbol</th>
                  <th className="text-left p-3">Agent</th>
                  <th className="text-left p-3">Direction</th>
                  <th className="text-right p-3">Confidence</th>
                  <th className="text-right p-3">Entry</th>
                  <th className="text-right p-3">R:R</th>
                  <th className="text-left p-3">Skill</th>
                  <th className="text-left p-3">Status</th>
                  <th className="text-left p-3">Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {recent.slice(0, 20).map((signal) => (
                  <tr key={signal.id} className="hover:bg-gray-800/50">
                    <td className="p-3 font-medium">{signal.symbol}</td>
                    <td className="p-3">
                      {signal.agent_type && (
                        <span className={`text-xs px-1.5 py-0.5 rounded ${
                          signal.agent_type === 'swing'
                            ? 'bg-blue-900/50 text-blue-400'
                            : 'bg-purple-900/50 text-purple-400'
                        }`}>
                          {signal.agent_type}
                        </span>
                      )}
                    </td>
                    <td className="p-3">
                      <span className={signal.direction === 'long' ? 'text-green-400' : 'text-red-400'}>
                        {signal.direction.toUpperCase()}
                      </span>
                    </td>
                    <td className="p-3 text-right font-mono">
                      {(signal.confidence * 100).toFixed(0)}%
                    </td>
                    <td className="p-3 text-right font-mono">${signal.entry_price.toFixed(2)}</td>
                    <td className="p-3 text-right font-mono">{signal.risk_reward_ratio.toFixed(1)}</td>
                    <td className="p-3 text-gray-400">{signal.skill_name}</td>
                    <td className="p-3">
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          signal.status === 'executed'
                            ? 'bg-green-900/50 text-green-400'
                            : signal.status === 'rejected'
                            ? 'bg-red-900/50 text-red-400'
                            : signal.status === 'pending'
                            ? 'bg-yellow-900/50 text-yellow-400'
                            : 'bg-gray-800 text-gray-400'
                        }`}
                      >
                        {signal.status}
                      </span>
                    </td>
                    <td className="p-3 text-gray-500 text-xs">
                      {format(new Date(signal.created_at), 'HH:mm:ss')}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
