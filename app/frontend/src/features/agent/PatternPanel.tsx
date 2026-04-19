import { useQuery } from '@tanstack/react-query';
import { Brain } from 'lucide-react';
import { fetchIntradayModelStatus, PatternMetrics } from '../../lib/api';

const PATTERN_COLORS: Record<string, string> = {
  ORB_BREAKOUT: 'text-blue-400',
  MORNING_REVERSAL: 'text-orange-400',
  VWAP_RECLAIM: 'text-cyan-400',
  GAP_FILL: 'text-yellow-400',
  POWER_HOUR: 'text-purple-400',
  MEAN_REVERSION: 'text-green-400',
  MOMENTUM_BREAKOUT: 'text-red-400',
  RANGE_EXPANSION: 'text-pink-400',
  NO_PATTERN: 'text-gray-500',
};

function formatPatternName(name: string) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export default function PatternPanel() {
  const { data, isLoading } = useQuery({
    queryKey: ['intradayModelStatus'],
    queryFn: fetchIntradayModelStatus,
    refetchInterval: 30000,
  });

  if (isLoading) {
    return null;
  }

  if (!data?.loaded) {
    return (
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Brain size={16} className="text-gray-500" />
          <h3 className="text-sm font-medium text-gray-400">Pattern Classifier</h3>
        </div>
        <p className="text-xs text-gray-500">
          No trained model. Use the API to train: POST /api/ml/intraday/train
        </p>
      </div>
    );
  }

  const metrics = data.training_metrics;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center gap-2 mb-3">
        <Brain size={16} className="text-purple-400" />
        <h3 className="text-sm font-medium text-gray-400">Pattern Classifier</h3>
        <span className="text-xs bg-purple-600/20 text-purple-400 px-2 py-0.5 rounded-full ml-auto">
          Active
        </span>
      </div>

      {metrics && (
        <div className="space-y-3">
          {/* Model accuracy */}
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <span className="text-gray-500">Train Accuracy</span>
              <div className="font-medium mt-0.5">
                {metrics.train_accuracy != null ? `${(metrics.train_accuracy * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
            <div>
              <span className="text-gray-500">Eval Accuracy</span>
              <div className="font-medium mt-0.5">
                {metrics.eval_accuracy != null ? `${(metrics.eval_accuracy * 100).toFixed(1)}%` : '—'}
              </div>
            </div>
          </div>

          {/* Per-pattern performance */}
          {metrics.eval_per_pattern && (
            <div className="space-y-1.5">
              <span className="text-xs text-gray-500">Pattern Detection (Eval Set)</span>
              {(Object.entries(metrics.eval_per_pattern) as [string, PatternMetrics][])
                .filter(([, m]) => m.support > 0)
                .sort(([, a], [, b]) => b.f1 - a.f1)
                .slice(0, 6)
                .map(([name, m]) => (
                  <div key={name} className="flex items-center justify-between text-xs">
                    <span className={PATTERN_COLORS[name] || 'text-gray-400'}>
                      {formatPatternName(name)}
                    </span>
                    <div className="flex gap-3 text-gray-500">
                      <span>P: {(m.precision * 100).toFixed(0)}%</span>
                      <span>R: {(m.recall * 100).toFixed(0)}%</span>
                      <span className="text-gray-400">F1: {(m.f1 * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                ))}
            </div>
          )}

          {/* Training info */}
          <div className="text-xs text-gray-600 pt-1 border-t border-gray-800">
            {metrics.n_features ?? '?'} features, {metrics.n_classes ?? '?'} classes,{' '}
            {metrics.train_samples ?? '?'} train / {metrics.eval_samples ?? '?'} eval samples
          </div>
        </div>
      )}
    </div>
  );
}
