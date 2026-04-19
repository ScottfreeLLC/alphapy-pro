import { AgentStatus } from '../types/agent';

const statusColors: Record<AgentStatus, string> = {
  stopped: 'bg-gray-500',
  starting: 'bg-yellow-500 animate-pulse',
  running: 'bg-green-500',
  paused: 'bg-yellow-500',
  error: 'bg-red-500',
};

const statusLabels: Record<AgentStatus, string> = {
  stopped: 'Stopped',
  starting: 'Starting...',
  running: 'Running',
  paused: 'Paused',
  error: 'Error',
};

interface StatusBadgeProps {
  status: AgentStatus;
  connected: boolean;
}

export default function StatusBadge({ status, connected }: StatusBadgeProps) {
  return (
    <div className="flex items-center gap-2">
      <span
        className={`inline-block w-2.5 h-2.5 rounded-full ${statusColors[status]}`}
      />
      <span className="text-sm font-medium">{statusLabels[status]}</span>
      {!connected && (
        <span className="text-xs text-red-400 ml-auto">Disconnected</span>
      )}
    </div>
  );
}
