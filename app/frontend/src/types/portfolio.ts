export interface Position {
  symbol: string;
  qty: number;
  side: string;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  change_today: number;
}

export interface OptimizerStatus {
  strategy: string;
  is_fitted: boolean;
  fitted_at: string | null;
  needs_rebalance: boolean;
  rebalance_days: number;
  n_assets: number;
  weights: Record<string, number>;
  max_weight: number;
  fit_metrics: {
    strategy: string;
    n_assets: number;
    total_weight: number;
    top_5: [string, number][];
    annualized_return: number | null;
    annualized_vol: number | null;
    sharpe: number | null;
    views_count?: number;
  } | null;
}

export interface PortfolioSummary {
  equity: number;
  cash: number;
  buying_power: number;
  portfolio_value?: number;
  positions_count: number;
  total_unrealized_pl: number;
  total_market_value?: number;
  exposure_pct: number;
  positions: Position[];
  optimizer?: OptimizerStatus;
}

export interface RebalanceResult {
  status: string;
  result: Record<string, unknown>;
  optimizer: OptimizerStatus;
}
