export type AgentType = 'swing' | 'day';
export type AgentStatus = 'stopped' | 'starting' | 'running' | 'paused' | 'error';
export type AutonomyMode = 'approval' | 'semi_autonomous' | 'autonomous';
export type SignalDirection = 'long' | 'short';
export type SignalStatus = 'pending' | 'approved' | 'rejected' | 'executed' | 'expired' | 'cancelled';

export interface TradeSignal {
  id: string;
  symbol: string;
  direction: SignalDirection;
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  risk_reward_ratio: number;
  reasoning: string;
  skill_name: string;
  status: SignalStatus;
  created_at: string;
  position_size_pct: number;
  agent_type?: AgentType;
  pattern?: string;
  pattern_probability?: number;
}

export interface ActivityEntry {
  timestamp: string;
  action: string;
  detail: string;
  level: string;
}

export interface AgentState {
  agent_type: AgentType;
  status: AgentStatus;
  autonomy_mode: AutonomyMode;
  cycle_count: number;
  started_at: string | null;
  last_cycle_at: string | null;
  pending_signals_count: number;
  recent_signals_count: number;
  consecutive_errors: number;
  last_error: string | null;
  recent_activity: ActivityEntry[];
}

export interface SharedRiskStatus {
  total_positions: number;
  max_combined_positions: number;
  positions_by_agent: Record<string, Record<string, string>>;
  combined_daily_pnl: number;
  combined_daily_loss_limit: number;
  day_trades_today: number;
  max_day_trades: number;
  circuit_breaker_tripped: boolean;
}

export interface CombinedState {
  swing: AgentState;
  day: AgentState;
  shared_risk: SharedRiskStatus;
}

export interface Skill {
  name: string;
  enabled: boolean;
  timeframes: string[];
  tags: string[];
  risk_per_trade: number;
  max_positions: number;
}
