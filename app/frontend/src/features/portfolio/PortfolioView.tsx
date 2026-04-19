import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  DollarSign,
  TrendingUp,
  PieChart,
  Wallet,
  RefreshCw,
  BarChart3,
  Scale,
  Target,
} from 'lucide-react';
import { fetchPortfolioSummary, fetchPortfolioWeights, rebalancePortfolio } from '../../lib/api';

const STRATEGIES = [
  { value: 'hrp', label: 'HRP (Risk Parity)' },
  { value: 'mean_risk', label: 'Mean-Risk (Max Sharpe)' },
  { value: 'black_litterman', label: 'Black-Litterman' },
];

export default function PortfolioView() {
  const queryClient = useQueryClient();
  const [selectedStrategy, setSelectedStrategy] = useState('hrp');

  const { data: summary, isLoading } = useQuery({
    queryKey: ['portfolioSummary'],
    queryFn: fetchPortfolioSummary,
    refetchInterval: 10000,
  });

  const { data: weights } = useQuery({
    queryKey: ['portfolioWeights'],
    queryFn: fetchPortfolioWeights,
    refetchInterval: 30000,
    retry: false,
  });

  const rebalanceMutation = useMutation({
    mutationFn: rebalancePortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolioWeights'] });
      queryClient.invalidateQueries({ queryKey: ['portfolioSummary'] });
    },
  });

  if (isLoading) {
    return (
      <div className="p-6 flex justify-center py-20">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white" />
      </div>
    );
  }

  const metrics = [
    {
      icon: DollarSign,
      label: 'Equity',
      value: `$${(summary?.equity ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`,
    },
    {
      icon: Wallet,
      label: 'Cash',
      value: `$${(summary?.cash ?? 0).toLocaleString('en-US', { minimumFractionDigits: 2 })}`,
    },
    {
      icon: TrendingUp,
      label: 'Unrealized P&L',
      value: `$${(summary?.total_unrealized_pl ?? 0).toFixed(2)}`,
      color: (summary?.total_unrealized_pl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400',
    },
    {
      icon: PieChart,
      label: 'Exposure',
      value: `${(summary?.exposure_pct ?? 0).toFixed(1)}%`,
    },
  ];

  const positions = summary?.positions || [];
  const optimizer = weights || summary?.optimizer;
  const sortedWeights = optimizer?.weights
    ? Object.entries(optimizer.weights).sort(([, a], [, b]) => b - a)
    : [];

  return (
    <div className="p-6 max-w-5xl mx-auto space-y-6">
      <h2 className="text-3xl font-bold">Portfolio</h2>

      {/* Account metrics */}
      <div className="grid grid-cols-4 gap-4">
        {metrics.map(({ icon: Icon, label, value, color }) => (
          <div key={label} className="bg-gray-900 rounded-lg p-4 border border-gray-800">
            <div className="flex items-center gap-2 text-gray-400 mb-2">
              <Icon size={16} />
              <span className="text-sm">{label}</span>
            </div>
            <div className={`text-xl font-bold ${color || ''}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* Portfolio Optimizer */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Scale size={18} className="text-blue-400" />
            <h3 className="font-medium">Portfolio Optimizer</h3>
            {optimizer?.is_fitted && (
              <span className="text-xs bg-green-900/50 text-green-400 px-2 py-0.5 rounded">
                Fitted
              </span>
            )}
            {optimizer?.needs_rebalance && optimizer?.is_fitted && (
              <span className="text-xs bg-yellow-900/50 text-yellow-400 px-2 py-0.5 rounded">
                Needs Rebalance
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-sm text-gray-300"
            >
              {STRATEGIES.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
            <button
              onClick={() =>
                rebalanceMutation.mutate({ strategy: selectedStrategy })
              }
              disabled={rebalanceMutation.isPending}
              className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 rounded text-sm font-medium transition-colors"
            >
              <RefreshCw
                size={14}
                className={rebalanceMutation.isPending ? 'animate-spin' : ''}
              />
              {rebalanceMutation.isPending ? 'Rebalancing...' : 'Rebalance'}
            </button>
          </div>
        </div>

        {rebalanceMutation.isError && (
          <div className="mx-4 mt-4 p-3 bg-red-900/30 border border-red-800 rounded text-red-400 text-sm">
            {(rebalanceMutation.error as Error)?.message || 'Rebalance failed'}
          </div>
        )}

        {optimizer?.is_fitted ? (
          <div className="p-4 space-y-4">
            {/* Fit metrics summary */}
            {optimizer.fit_metrics && (
              <div className="grid grid-cols-4 gap-3">
                <div className="text-center">
                  <div className="text-xs text-gray-500">Strategy</div>
                  <div className="text-sm font-medium capitalize">
                    {optimizer.fit_metrics.strategy.replace('_', ' ')}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-500">Assets</div>
                  <div className="text-sm font-medium">{optimizer.fit_metrics.n_assets}</div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-500">Ann. Return</div>
                  <div className="text-sm font-medium">
                    {optimizer.fit_metrics.annualized_return != null
                      ? `${(optimizer.fit_metrics.annualized_return * 100).toFixed(1)}%`
                      : '—'}
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-xs text-gray-500">Sharpe</div>
                  <div className="text-sm font-medium">
                    {optimizer.fit_metrics.sharpe != null
                      ? optimizer.fit_metrics.sharpe.toFixed(2)
                      : '—'}
                  </div>
                </div>
              </div>
            )}

            {/* Weights bar chart */}
            {sortedWeights.length > 0 && (
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <BarChart3 size={14} className="text-gray-500" />
                  <span className="text-xs text-gray-500 uppercase tracking-wide">
                    Asset Weights ({sortedWeights.length} assets)
                  </span>
                </div>
                <div className="space-y-1.5">
                  {sortedWeights.map(([symbol, weight]) => (
                    <div key={symbol} className="flex items-center gap-3">
                      <span className="text-xs font-mono w-14 text-right text-gray-400">
                        {symbol}
                      </span>
                      <div className="flex-1 bg-gray-800 rounded-full h-4 overflow-hidden">
                        <div
                          className="bg-blue-500/70 h-full rounded-full transition-all"
                          style={{
                            width: `${Math.min(100, (weight / (optimizer.max_weight || 0.15)) * 100)}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-16 text-gray-400">
                        {(weight * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Fitted timestamp */}
            {optimizer.fitted_at && (
              <div className="text-xs text-gray-600 text-right">
                Last fitted: {new Date(optimizer.fitted_at).toLocaleString()}
                {' '}| Rebalance every {optimizer.rebalance_days} days
              </div>
            )}
          </div>
        ) : (
          <div className="p-8 text-center">
            <Target size={32} className="mx-auto text-gray-600 mb-3" />
            <p className="text-gray-500 text-sm">Portfolio optimizer not yet fitted</p>
            <p className="text-gray-600 text-xs mt-1">
              Click "Rebalance" to fit the optimizer on historical returns
            </p>
          </div>
        )}
      </div>

      {/* Positions table */}
      <div className="bg-gray-900 rounded-lg border border-gray-800">
        <div className="p-4 border-b border-gray-800 flex items-center justify-between">
          <h3 className="font-medium">Open Positions</h3>
          <span className="text-xs text-gray-500">{positions.length} positions</span>
        </div>

        {positions.length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-500 text-sm">No open positions</p>
            <p className="text-gray-600 text-xs mt-2">
              Connect Alpaca and start the agent to begin trading
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="text-gray-500 text-xs">
                <tr className="border-b border-gray-800">
                  <th className="text-left p-3">Symbol</th>
                  <th className="text-left p-3">Side</th>
                  <th className="text-right p-3">Qty</th>
                  <th className="text-right p-3">Entry</th>
                  <th className="text-right p-3">Current</th>
                  <th className="text-right p-3">Market Value</th>
                  <th className="text-right p-3">P&L</th>
                  <th className="text-right p-3">P&L %</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800">
                {positions.map((pos) => (
                  <tr key={pos.symbol} className="hover:bg-gray-800/50">
                    <td className="p-3 font-medium">{pos.symbol}</td>
                    <td className="p-3 uppercase text-xs">{pos.side}</td>
                    <td className="p-3 text-right font-mono">{pos.qty}</td>
                    <td className="p-3 text-right font-mono">
                      ${pos.avg_entry_price.toFixed(2)}
                    </td>
                    <td className="p-3 text-right font-mono">
                      ${pos.current_price.toFixed(2)}
                    </td>
                    <td className="p-3 text-right font-mono">
                      ${pos.market_value.toLocaleString('en-US', { minimumFractionDigits: 2 })}
                    </td>
                    <td
                      className={`p-3 text-right font-mono ${
                        pos.unrealized_pl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      ${pos.unrealized_pl.toFixed(2)}
                    </td>
                    <td
                      className={`p-3 text-right font-mono ${
                        pos.unrealized_plpc >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}
                    >
                      {(pos.unrealized_plpc * 100).toFixed(2)}%
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
