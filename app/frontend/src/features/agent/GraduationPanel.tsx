import { useQuery } from '@tanstack/react-query';
import { fetchGraduationStatus } from '../../lib/api';
import { GraduationCap } from 'lucide-react';
import { AgentType } from '../../types/agent';

function ProgressBar({ value, target, label }: { value: number; target: number; label: string }) {
  const pct = Math.min(100, (value / target) * 100);
  const met = value >= target;

  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-400">{label}</span>
        <span className={met ? 'text-green-400' : 'text-gray-500'}>
          {typeof value === 'number' && value < 1 ? (value * 100).toFixed(0) + '%' : value.toFixed ? value.toFixed(1) : value}
          {' / '}
          {typeof target === 'number' && target < 1 ? (target * 100).toFixed(0) + '%' : target}
        </span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${met ? 'bg-green-500' : 'bg-blue-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

interface Props {
  agentType: AgentType;
}

export default function GraduationPanel({ agentType }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['graduationStatus', agentType],
    queryFn: () => fetchGraduationStatus(agentType),
    refetchInterval: 30000,
  });

  if (isLoading || !data) {
    return null;
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center gap-2 mb-3">
        <GraduationCap size={16} className="text-blue-400" />
        <h3 className="text-sm font-medium text-gray-400">Graduation Progress</h3>
        {data.ready_for_promotion && (
          <span className="text-xs bg-green-600/20 text-green-400 px-2 py-0.5 rounded-full ml-auto">
            Ready
          </span>
        )}
      </div>

      <div className="space-y-2">
        <ProgressBar value={data.total_trades} target={50} label="Trades" />
        <ProgressBar value={data.win_rate} target={data.win_rate_target} label="Win Rate" />
        <ProgressBar value={data.sharpe} target={data.sharpe_target} label="Sharpe Ratio" />
        <ProgressBar
          value={Math.max(0, data.drawdown_limit - data.max_drawdown)}
          target={data.drawdown_limit}
          label="Drawdown Buffer"
        />
      </div>

      {data.consecutive_losses > 0 && (
        <div className="mt-2 text-xs text-yellow-400">
          Consecutive losses: {data.consecutive_losses} / {data.consecutive_loss_limit}
        </div>
      )}
    </div>
  );
}
