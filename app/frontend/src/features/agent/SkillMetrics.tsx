import { useQuery } from '@tanstack/react-query';
import { fetchSkillPerformance } from '../../lib/api';
import { AgentType } from '../../types/agent';

interface Props {
  agentType: AgentType;
}

export default function SkillMetrics({ agentType }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['skillPerformance', agentType],
    queryFn: () => fetchSkillPerformance(agentType),
    refetchInterval: 30000,
  });

  if (isLoading) {
    return <div className="text-gray-500 text-sm">Loading skill metrics...</div>;
  }

  const skills = data?.skills || {};
  const entries = Object.entries(skills);

  if (entries.length === 0) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <h3 className="text-sm font-medium text-gray-400 mb-2">Skill Performance</h3>
        <p className="text-xs text-gray-500">No trade history yet. Start the agent to generate signals.</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Skill Performance</h3>
      <div className="space-y-3">
        {entries.map(([name, metrics]) => (
          <div key={name} className="flex items-center justify-between">
            <div>
              <span className="text-sm font-medium">{name.replace(/_/g, ' ')}</span>
              <span className="text-xs text-gray-500 ml-2">{metrics.total_trades} trades</span>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <span className={metrics.win_rate >= 0.5 ? 'text-green-400' : 'text-red-400'}>
                {(metrics.win_rate * 100).toFixed(0)}% WR
              </span>
              <span className={metrics.total_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                ${metrics.total_pnl.toFixed(0)}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
