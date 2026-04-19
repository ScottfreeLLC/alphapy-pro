import axios from 'axios';
import { MarketScreenerResponse } from '../types/stock';
import { AgentType, AgentState, CombinedState, TradeSignal, Skill } from '../types/agent';
import { PortfolioSummary, OptimizerStatus, RebalanceResult } from '../types/portfolio';
import { BacktestConfig, BacktestResult, BacktestRunSummary } from '../types/backtest';
import { SubstackStatus, SubstackPostContent, SubstackComposeRequest } from '../types/substack';

const API_BASE_URL = 'http://localhost:8080';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Screener
export const fetchMarketScreener = async (): Promise<MarketScreenerResponse> => {
  const response = await apiClient.get<MarketScreenerResponse>('/api/stocks/screener');
  return response.data;
};

// Agent
export const fetchAgentStatus = async (agentType: AgentType = 'swing'): Promise<AgentState> => {
  const response = await apiClient.get<AgentState>(`/api/agent/${agentType}/status`);
  return response.data;
};

export const fetchCombinedStatus = async (): Promise<CombinedState> => {
  const response = await apiClient.get<CombinedState>('/api/agent/combined/status');
  return response.data;
};

export const startAgent = async (agentType: AgentType = 'swing'): Promise<{ status: string; state: AgentState }> => {
  const response = await apiClient.post(`/api/agent/${agentType}/start`);
  return response.data;
};

export const stopAgent = async (agentType: AgentType = 'swing'): Promise<{ status: string; state: AgentState }> => {
  const response = await apiClient.post(`/api/agent/${agentType}/stop`);
  return response.data;
};

export const setAgentMode = async ({ agentType, mode }: { agentType: AgentType; mode: string }): Promise<{ mode: string }> => {
  const response = await apiClient.post(`/api/agent/${agentType}/mode`, { mode });
  return response.data;
};

// Signals
export const fetchPendingSignals = async (): Promise<{ signals: TradeSignal[]; count: number }> => {
  const response = await apiClient.get('/api/signals/pending');
  return response.data;
};

export const fetchRecentSignals = async (): Promise<{ signals: TradeSignal[]; count: number }> => {
  const response = await apiClient.get('/api/signals/recent');
  return response.data;
};

export const approveSignal = async (signalId: string): Promise<{ status: string; signal: TradeSignal }> => {
  const response = await apiClient.post(`/api/signals/${signalId}/approve`);
  return response.data;
};

export const rejectSignal = async (signalId: string): Promise<{ status: string; signal: TradeSignal }> => {
  const response = await apiClient.post(`/api/signals/${signalId}/reject`);
  return response.data;
};

// Skills
export const fetchSkills = async (agentType: AgentType = 'swing'): Promise<{ skills: Skill[] }> => {
  const response = await apiClient.get(`/api/agent/${agentType}/skills`);
  return response.data;
};

export const toggleSkill = async ({ agentType, name, enabled }: { agentType: AgentType; name: string; enabled: boolean }): Promise<{ skill: string; enabled: boolean }> => {
  const response = await apiClient.put(`/api/agent/${agentType}/skills/${name}/toggle`, { enabled });
  return response.data;
};

// Portfolio
export const fetchPortfolioSummary = async (): Promise<PortfolioSummary> => {
  const response = await apiClient.get<PortfolioSummary>('/api/portfolio/summary');
  return response.data;
};

export const fetchPortfolioWeights = async (): Promise<OptimizerStatus> => {
  const response = await apiClient.get<OptimizerStatus>('/api/portfolio/weights');
  return response.data;
};

export const rebalancePortfolio = async (config?: {
  strategy?: string;
  symbols?: string[];
  days_back?: number;
}): Promise<RebalanceResult> => {
  const response = await apiClient.post<RebalanceResult>('/api/portfolio/rebalance', config || {});
  return response.data;
};

// Backtest
export const runBacktest = async (config: BacktestConfig): Promise<BacktestResult> => {
  const response = await apiClient.post<BacktestResult>('/api/backtest/run', config);
  return response.data;
};

export const fetchBacktestRuns = async (): Promise<{ runs: BacktestRunSummary[]; count: number }> => {
  const response = await apiClient.get('/api/backtest/runs');
  return response.data;
};

export const fetchBacktestResult = async (runId: string): Promise<BacktestResult> => {
  const response = await apiClient.get<BacktestResult>(`/api/backtest/${runId}`);
  return response.data;
};

// Risk & Performance
export const fetchRiskStatus = async (agentType: AgentType = 'swing'): Promise<{
  circuit_breaker: { tripped: boolean; tripped_at: string | null; reason: string | null };
  exposure: { open_positions: number; max_positions: number; daily_pnl: number };
}> => {
  const response = await apiClient.get(`/api/agent/${agentType}/risk/status`);
  return response.data;
};

export const resetCircuitBreaker = async (agentType: AgentType = 'swing'): Promise<{ status: string }> => {
  const response = await apiClient.post(`/api/agent/${agentType}/risk/reset`);
  return response.data;
};

export const fetchPerformanceMetrics = async (agentType: AgentType = 'swing'): Promise<{
  total_trades: number;
  win_rate: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_pnl: number;
  profit_factor: number | string;
  equity_curve: number[];
}> => {
  const response = await apiClient.get(`/api/agent/${agentType}/performance/metrics`);
  return response.data;
};

export const fetchSkillPerformance = async (agentType: AgentType = 'swing'): Promise<{
  skills: Record<string, { total_trades: number; win_rate: number; total_pnl: number; avg_pnl: number }>;
}> => {
  const response = await apiClient.get(`/api/agent/${agentType}/performance/skills`);
  return response.data;
};

export const recordSignalOutcome = async (signalId: string, exitPrice: number): Promise<{ status: string }> => {
  const response = await apiClient.post(`/api/signals/${signalId}/outcome`, { exit_price: exitPrice });
  return response.data;
};

// Graduation
export const fetchGraduationStatus = async (agentType: AgentType = 'swing'): Promise<{
  total_trades: number;
  trades_needed: number;
  win_rate: number;
  win_rate_target: number;
  sharpe: number;
  sharpe_target: number;
  max_drawdown: number;
  drawdown_limit: number;
  consecutive_losses: number;
  consecutive_loss_limit: number;
  ready_for_promotion: boolean;
}> => {
  const response = await apiClient.get(`/api/agent/${agentType}/graduation/status`);
  return response.data;
};

// Chat
export const chatWithAlfi = async (message: string): Promise<{ reply: string; timestamp: string }> => {
  const response = await apiClient.post('/api/chat', { message });
  return response.data;
};

// Enriched Screener
export const fetchEnrichedScreener = async (): Promise<{
  stocks: Array<{
    symbol: string;
    price: number;
    high: number;
    low: number;
    open: number;
    volume: number;
    daily_change: number;
    daily_change_pct: number;
    sma20: number | null;
    rsi_14: number | null;
    volume_ratio: number | null;
    trend_summary: string;
    patterns: string[];
    sentiment: string;
  }>;
  count: number;
  timestamp: string;
}> => {
  const response = await apiClient.get('/api/stocks/screener/enriched');
  return response.data;
};

// ML Backtest
export const runMLBacktest = async (config: {
  strategy: string;
  symbols: string[];
  initial_capital?: number;
  train_pct?: number;
  ml_threshold?: number;
}): Promise<unknown> => {
  const response = await apiClient.post('/api/backtest/ml', config);
  return response.data;
};

// Intraday Pattern ML
export interface PatternMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface IntradayModelStatus {
  loaded: boolean;
  training_metrics: {
    train_accuracy?: number;
    eval_accuracy?: number;
    train_samples?: number;
    eval_samples?: number;
    n_features?: number;
    n_classes?: number;
    eval_per_pattern?: Record<string, PatternMetrics>;
  } | null;
}

export const fetchIntradayModelStatus = async (): Promise<IntradayModelStatus> => {
  const response = await apiClient.get('/api/ml/intraday/status');
  return response.data;
};

export const trainIntradayClassifier = async (config?: {
  symbols?: string[];
  days_back?: number;
  train_pct?: number;
}): Promise<Record<string, unknown>> => {
  const response = await apiClient.post('/api/ml/intraday/train', config || {});
  return response.data;
};

export const fetchIntradayPredictions = async (symbol: string): Promise<{
  symbol: string;
  model_loaded: boolean;
  latest: { pattern: string; probability: number; top_3: Array<{ pattern: string; probability: number }> } | null;
  session_bars?: number;
}> => {
  const response = await apiClient.get(`/api/ml/intraday/predictions/${symbol}`);
  return response.data;
};

// Pattern Discovery
export interface MotifLibrary {
  total_motifs: number;
  motifs: Array<{
    motif_id: number;
    window_size: number;
    template: number[];
    occurrences: number;
    avg_forward_return: number;
    win_rate: number;
    sharpe: number;
    last_seen: string | null;
    cluster_id: number;
    symbols: string[];
  }>;
  window_sizes?: number[];
  avg_win_rate?: number;
  avg_sharpe?: number;
  loaded?: boolean;
}

export interface EvolvedRulesLibrary {
  total_rules: number;
  active_rules: number;
  rules: Array<{
    rule_id: number;
    expression: string;
    sharpe_train: number;
    sharpe_test: number;
    total_pnl_train: number;
    total_pnl_test: number;
    max_drawdown: number;
    total_trades: number;
    win_rate: number;
    symbols_validated: string[];
    created_at: string;
    status: string;
  }>;
  best_sharpe?: number;
  avg_win_rate?: number;
  loaded?: boolean;
}

export interface StrategyOption {
  value: string;
  label: string;
  agent_type: string;
  source: string;
}

export const fetchMotifLibrary = async (): Promise<MotifLibrary> => {
  const response = await apiClient.get<MotifLibrary>('/api/discovery/motifs');
  return response.data;
};

export const runMotifDiscovery = async (config: {
  symbols: string[];
  days_back?: number;
  window_sizes?: number[];
}): Promise<{ motifs_discovered: number; symbols_processed: number; summary: MotifLibrary }> => {
  const response = await apiClient.post('/api/discovery/motifs', config);
  return response.data;
};

export const fetchEvolvedRules = async (): Promise<EvolvedRulesLibrary> => {
  const response = await apiClient.get<EvolvedRulesLibrary>('/api/discovery/rules');
  return response.data;
};

export const runRuleEvolution = async (config: {
  symbols: string[];
  days_back?: number;
  generations?: number;
  population?: number;
  window_size?: number;
}): Promise<{ new_rules: number; total_rules: number; rules: EvolvedRulesLibrary['rules'] }> => {
  const response = await apiClient.post('/api/discovery/evolve', config);
  return response.data;
};

export const runDiscoveryBacktest = async (config: {
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_capital?: number;
}): Promise<{ results: BacktestResult[]; count: number }> => {
  const response = await apiClient.post('/api/discovery/backtest', config);
  return response.data;
};

export const fetchBacktestStrategies = async (agentType: string = ''): Promise<{ strategies: StrategyOption[]; count: number }> => {
  const response = await apiClient.get('/api/backtest/strategies', { params: { agent_type: agentType } });
  return response.data;
};

// Substack
export const fetchSubstackStatus = async (): Promise<SubstackStatus> => {
  const response = await apiClient.get<SubstackStatus>('/api/substack/status');
  return response.data;
};

export const composeSubstackPost = async (request: SubstackComposeRequest): Promise<SubstackPostContent> => {
  const response = await apiClient.post<SubstackPostContent>('/api/substack/compose', request);
  return response.data;
};

export const createSubstackDraft = async (request: SubstackComposeRequest): Promise<{ draft: unknown; post: SubstackPostContent }> => {
  const response = await apiClient.post('/api/substack/draft', request);
  return response.data;
};

export const publishSubstackDraft = async (draftId: number, sendEmail = true): Promise<{ status: string }> => {
  const response = await apiClient.post('/api/substack/publish', { draft_id: draftId, send_email: sendEmail });
  return response.data;
};

// Encoding
import { EncodingResponse, EncodingHistoryResponse, SimilarityResponse, PatternCatalogResponse } from '../types/encoding';

export const fetchEncoding = async (symbol: string, bars = 50): Promise<EncodingResponse> => {
  const response = await apiClient.get<EncodingResponse>(`/api/encoding/${symbol}`, { params: { bars } });
  return response.data;
};

export const fetchEncodingHistory = async (symbol: string, window = 20, bars = 100): Promise<EncodingHistoryResponse> => {
  const response = await apiClient.get<EncodingHistoryResponse>(`/api/encoding/${symbol}/history`, { params: { window, bars } });
  return response.data;
};

export const fetchEncodingSimilarity = async (symbolA: string, symbolB: string, bars = 100): Promise<SimilarityResponse> => {
  const response = await apiClient.post<SimilarityResponse>('/api/encoding/similarity', { symbol_a: symbolA, symbol_b: symbolB, bars });
  return response.data;
};

export const fetchEncodingPatterns = async (): Promise<PatternCatalogResponse> => {
  const response = await apiClient.get<PatternCatalogResponse>('/api/encoding/patterns');
  return response.data;
};
