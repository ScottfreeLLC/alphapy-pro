import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Play, Clock, TrendingUp, TrendingDown, AlertTriangle } from 'lucide-react';
import { runBacktest, fetchBacktestRuns, fetchBacktestResult, fetchBacktestStrategies, StrategyOption } from '../../lib/api';
import { BacktestResult, BacktestRunSummary } from '../../types/backtest';
import { AgentType } from '../../types/agent';
import EquityCurveChart from './EquityCurveChart';

function MetricCard({ label, value, suffix, positive }: {
  label: string;
  value: number | null;
  suffix?: string;
  positive?: boolean | null;
}) {
  const display = value !== null && value !== undefined
    ? `${value.toLocaleString(undefined, { maximumFractionDigits: 2 })}${suffix || ''}`
    : 'N/A';

  const color = positive === null || positive === undefined
    ? 'text-gray-100'
    : positive
      ? 'text-green-400'
      : 'text-red-400';

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className={`text-lg font-semibold ${color}`}>{display}</div>
    </div>
  );
}

export default function BacktestPage() {
  const [agentType, setAgentType] = useState<AgentType | ''>('swing');
  const [strategy, setStrategy] = useState('momentum_breakout');
  const [symbolsInput, setSymbolsInput] = useState('AAPL, MSFT, NVDA');
  const [startDate, setStartDate] = useState('2025-06-01');
  const [endDate, setEndDate] = useState('2026-02-14');
  const [initialCapital, setInitialCapital] = useState(100000);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);

  // Fetch strategies dynamically based on agent type
  const strategiesQuery = useQuery({
    queryKey: ['backtest-strategies', agentType],
    queryFn: () => fetchBacktestStrategies(agentType),
  });

  const strategies: StrategyOption[] = strategiesQuery.data?.strategies ?? [];

  // Auto-select first strategy when agent type changes
  const handleAgentTypeChange = (newType: AgentType | '') => {
    setAgentType(newType);
    // Strategy selection will update when query completes
  };

  // Update strategy selection when strategies list changes
  if (strategies.length > 0 && !strategies.find(s => s.value === strategy)) {
    setStrategy(strategies[0].value);
  }

  const timeframe = agentType === 'day' ? '5min' : '1d';

  const runsQuery = useQuery({
    queryKey: ['backtest-runs'],
    queryFn: fetchBacktestRuns,
  });

  const backtestMutation = useMutation({
    mutationFn: runBacktest,
    onSuccess: (data) => {
      setResult(data);
      runsQuery.refetch();
    },
  });

  const loadRunMutation = useMutation({
    mutationFn: fetchBacktestResult,
    onSuccess: (data) => {
      setResult(data);
      setSelectedRunId(data.run_id);
    },
  });

  const handleRun = () => {
    const symbols = symbolsInput
      .split(',')
      .map((s) => s.trim().toUpperCase())
      .filter(Boolean);

    backtestMutation.mutate({
      strategy,
      symbols,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital,
      commission_pct: 0.001,
      slippage_pct: 0.0005,
      timeframe,
      agent_type: agentType,
    });
  };

  const metrics = result?.metrics;
  const totalReturn = metrics?.total_return_pct ?? null;

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Backtest</h1>

      {/* Config Form */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
        {/* Agent Type Toggle */}
        <div className="flex items-center gap-2 mb-4">
          <span className="text-sm text-gray-400">Agent:</span>
          {(['swing', 'day'] as const).map((type) => (
            <button
              key={type}
              onClick={() => handleAgentTypeChange(type)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                agentType === type
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-gray-200 hover:bg-gray-700'
              }`}
            >
              {type === 'swing' ? 'Swing (Daily)' : 'Day (5-min)'}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {/* Strategy */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {strategies.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                  {s.source === 'discovered' ? ' *' : ''}
                </option>
              ))}
            </select>
            {strategies.some(s => s.source === 'discovered') && (
              <p className="text-xs text-purple-400 mt-1">* = Discovered by ML pipeline</p>
            )}
          </div>

          {/* Symbols */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Symbols</label>
            <input
              type="text"
              value={symbolsInput}
              onChange={(e) => setSymbolsInput(e.target.value)}
              placeholder="AAPL, MSFT, NVDA"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Initial Capital */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Initial Capital</label>
            <input
              type="number"
              value={initialCapital}
              onChange={(e) => setInitialCapital(Number(e.target.value))}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Start Date */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">Start Date</label>
            <input
              type="date"
              value={startDate}
              onChange={(e) => setStartDate(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* End Date */}
          <div>
            <label className="block text-sm text-gray-400 mb-1">End Date</label>
            <input
              type="date"
              value={endDate}
              onChange={(e) => setEndDate(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          {/* Run Button */}
          <div className="flex items-end">
            <button
              onClick={handleRun}
              disabled={backtestMutation.isPending}
              className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 text-white font-medium rounded-lg px-4 py-2 text-sm flex items-center justify-center gap-2 transition-colors"
            >
              {backtestMutation.isPending ? (
                <>
                  <div className="animate-spin h-4 w-4 border-2 border-white border-t-transparent rounded-full" />
                  Running...
                </>
              ) : (
                <>
                  <Play size={16} />
                  Run Backtest
                </>
              )}
            </button>
          </div>
        </div>

        {backtestMutation.isError && (
          <div className="mt-4 bg-red-900/30 border border-red-800 rounded-lg p-3 flex items-center gap-2 text-sm text-red-300">
            <AlertTriangle size={16} />
            {(backtestMutation.error as Error)?.message || 'Backtest failed'}
          </div>
        )}
      </div>

      {/* Results */}
      {result && (
        <div className="space-y-6">
          {/* Metrics Cards */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
            <MetricCard
              label="Total Return"
              value={totalReturn}
              suffix="%"
              positive={totalReturn !== null ? totalReturn > 0 : null}
            />
            <MetricCard label="Total Trades" value={metrics?.total_trades ?? null} />
            <MetricCard
              label="Win Rate"
              value={metrics?.win_rate ?? null}
              suffix="%"
              positive={metrics?.win_rate !== null ? (metrics?.win_rate ?? 0) > 50 : null}
            />
            <MetricCard
              label="Sharpe Ratio"
              value={metrics?.sharpe_ratio ?? null}
              positive={metrics?.sharpe_ratio !== null ? (metrics?.sharpe_ratio ?? 0) > 1 : null}
            />
            <MetricCard
              label="Max Drawdown"
              value={metrics?.max_drawdown_pct ?? null}
              suffix="%"
              positive={false}
            />
            <MetricCard
              label="Profit Factor"
              value={metrics?.profit_factor ?? null}
              positive={metrics?.profit_factor !== null ? (metrics?.profit_factor ?? 0) > 1 : null}
            />
            <MetricCard
              label="Avg Trade P&L"
              value={metrics?.avg_trade_pnl ?? null}
              suffix=""
              positive={metrics?.avg_trade_pnl !== null ? (metrics?.avg_trade_pnl ?? 0) > 0 : null}
            />
            <MetricCard
              label="Best Trade"
              value={metrics?.best_trade ?? null}
              positive={true}
            />
            <MetricCard
              label="Worst Trade"
              value={metrics?.worst_trade ?? null}
              positive={false}
            />
            <MetricCard
              label="Final Equity"
              value={metrics?.final_equity ?? null}
              suffix=""
              positive={metrics?.final_equity !== null ? (metrics?.final_equity ?? 0) > result.config.initial_capital : null}
            />
          </div>

          {/* Equity Curve */}
          <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
            <h2 className="text-lg font-semibold mb-4">Equity Curve</h2>
            <EquityCurveChart
              data={result.equity_curve}
              initialCapital={result.config.initial_capital}
            />
          </div>

          {/* Per-symbol summary */}
          {Object.keys(result.symbol_summaries).length > 1 && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
              <h2 className="text-lg font-semibold mb-4">Per-Symbol Results</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-800">
                      <th className="text-left py-2 px-3">Symbol</th>
                      <th className="text-right py-2 px-3">Return</th>
                      <th className="text-right py-2 px-3">Trades</th>
                      <th className="text-right py-2 px-3">Win Rate</th>
                      <th className="text-right py-2 px-3">Final Equity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(result.symbol_summaries).map(([sym, s]) => (
                      <tr key={sym} className="border-b border-gray-800/50">
                        <td className="py-2 px-3 font-medium">{sym}</td>
                        <td className={`text-right py-2 px-3 ${(s.total_return_pct ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {s.total_return_pct?.toFixed(2) ?? 'N/A'}%
                        </td>
                        <td className="text-right py-2 px-3">{s.total_trades}</td>
                        <td className="text-right py-2 px-3">{s.win_rate?.toFixed(1) ?? 'N/A'}%</td>
                        <td className="text-right py-2 px-3">${s.final_equity?.toLocaleString() ?? 'N/A'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Trades Table */}
          {result.trades.length > 0 && (
            <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
              <h2 className="text-lg font-semibold mb-4">
                Trades ({result.trades.length})
              </h2>
              <div className="overflow-x-auto max-h-96">
                <table className="w-full text-sm">
                  <thead className="sticky top-0 bg-gray-900">
                    <tr className="text-gray-400 border-b border-gray-800">
                      {Object.keys(result.trades[0]).slice(0, 8).map((key) => (
                        <th key={key} className="text-left py-2 px-3 whitespace-nowrap">
                          {key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.trades.map((trade, idx) => (
                      <tr key={idx} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                        {Object.values(trade).slice(0, 8).map((val, ci) => {
                          const isNumeric = typeof val === 'number';
                          const isPnl = Object.keys(trade)[ci]?.toLowerCase().includes('pnl');
                          return (
                            <td
                              key={ci}
                              className={`py-2 px-3 whitespace-nowrap ${
                                isPnl && isNumeric
                                  ? val >= 0
                                    ? 'text-green-400'
                                    : 'text-red-400'
                                  : ''
                              }`}
                            >
                              {isNumeric ? (val as number).toFixed(2) : String(val ?? '')}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Past Runs */}
      <div className="bg-gray-900 rounded-xl border border-gray-800 p-5">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Clock size={18} />
          Past Runs
        </h2>
        {runsQuery.isLoading ? (
          <div className="text-gray-500 text-sm">Loading...</div>
        ) : runsQuery.data?.runs?.length ? (
          <div className="space-y-2">
            {runsQuery.data.runs.map((run: BacktestRunSummary) => (
              <button
                key={run.run_id}
                onClick={() => loadRunMutation.mutate(run.run_id)}
                className={`w-full text-left bg-gray-800 hover:bg-gray-700 rounded-lg p-3 transition-colors ${
                  selectedRunId === run.run_id ? 'ring-1 ring-blue-500' : ''
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {run.strategy.includes('momentum') || run.strategy.includes('day_') ? (
                      <TrendingUp size={16} className="text-green-400" />
                    ) : (
                      <TrendingDown size={16} className="text-blue-400" />
                    )}
                    <span className="font-medium text-sm">
                      {strategies.find((s) => s.value === run.strategy)?.label || run.strategy.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                    </span>
                    <span className="text-gray-400 text-xs">
                      {run.symbols.join(', ')}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(run.created_at).toLocaleDateString()} {new Date(run.created_at).toLocaleTimeString()}
                  </div>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {run.start_date} to {run.end_date}
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="text-gray-500 text-sm">No past runs yet. Run a backtest to get started.</div>
        )}
      </div>
    </div>
  );
}
