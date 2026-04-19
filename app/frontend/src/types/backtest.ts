export interface BacktestConfig {
  strategy: string;
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_capital: number;
  commission_pct: number;
  slippage_pct: number;
  timeframe?: string;    // "1d" or "5min"
  agent_type?: string;   // "swing", "day", or ""
}

export interface BacktestMetrics {
  total_return_pct: number | null;
  total_trades: number;
  win_rate: number | null;
  sharpe_ratio: number | null;
  max_drawdown_pct: number | null;
  profit_factor: number | null;
  avg_trade_pnl: number | null;
  best_trade: number | null;
  worst_trade: number | null;
  final_equity: number | null;
}

export interface EquityPoint {
  date: string;
  equity: number;
}

export interface BacktestTrade {
  [key: string]: string | number | null;
}

export interface SymbolSummary {
  total_return_pct: number | null;
  total_trades: number;
  win_rate: number | null;
  final_equity: number | null;
}

export interface BacktestResult {
  run_id: string;
  config: BacktestConfig;
  created_at: string;
  metrics: BacktestMetrics;
  equity_curve: EquityPoint[];
  trades: BacktestTrade[];
  symbol_summaries: Record<string, SymbolSummary>;
}

export interface BacktestRunSummary {
  run_id: string;
  strategy: string;
  symbols: string[];
  start_date: string;
  end_date: string;
  created_at: string;
}
