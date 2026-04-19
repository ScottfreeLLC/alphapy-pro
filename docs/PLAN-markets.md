# Alfi: Dual-Agent Autonomous Trading Platform — Implementation Plan

## Previous Phases (COMPLETE)
- [x] Phase 1: Core Agent Engine
- [x] Phase 2: Data Pipeline
- [x] Phase 3: Alpaca Integration
- [x] Phase 4: Dashboard
- [x] Phase 5: Autonomy Graduation

## Polymarket + Substack Integration
- [x] Polymarket Backend — **REMOVED (cleanup complete)**
- [x] Polymarket Frontend — **REMOVED (cleanup complete)**
- [x] Substack Backend — COMPLETE (kept, will refactor for trading newsletter)
- [x] Substack Frontend — COMPLETE (kept)
- [x] Agent Integration — prediction_markets skill removed

## Architecture Roadmap (COMPLETE)
- [x] Phase 1: Feature Engineering
- [x] Phase 2: Safety Rails
- [x] Phase 3: Real-Time Streaming
- [x] Phase 4: ML Signal Evaluation
- [x] Phase 5: Enhanced UX
- [x] Phase 6: Autonomy & Graduation

---

## Dual-Agent Autonomous Trading Platform

### Phase 1: Cleanup and Foundation — COMPLETE
- [x] 1a. Delete Polymarket files (backend, frontend, skill)
- [x] 1b. Clean existing files (api.py, config.py, engine.py, App.tsx, Layout.tsx, api.ts, post_composer.py, pyproject.toml)
- [x] 1c. Restructure skills directories (swing/ and day/ with 5 new intraday skills)
- [x] 1d. Refactor AgentConfig for dual agents (agent_type, timeframe, factory methods)
- [x] 1d. Add AgentType enum to state.py
- [x] 1e. Verify backend boots without errors
- [x] 1f. Verify frontend builds without broken imports

### Phase 2: Dual Agent Infrastructure — COMPLETE
- [x] 2a. Create AgentCoordinator (app/agent/coordinator.py)
- [x] 2b. Parameterize data pipeline for 5-min bars (DataProvider, FeatureEngine, indicators VWAP)
- [x] 2c. Shared risk management (app/agent/risk/shared.py)
- [x] 2d. Update backend API — parameterized agent endpoints (/api/agent/{agent_type}/...)

### Phase 3: Frontend Dual Agent Support — COMPLETE
- [x] types/agent.ts: Add AgentType, agent_type fields, SharedRiskStatus, CombinedState
- [x] api.ts: All agent/skills/risk/performance/graduation functions parameterized with agentType
- [x] websocket.ts: Parse combined state {swing, day, shared_risk}, expose per-agent states
- [x] App.tsx: Add /day route with AgentDashboard agentType="day"
- [x] Layout.tsx: Nav items "Swing Agent" and "Day Agent" with per-agent status lines
- [x] AgentDashboard.tsx: Accept agentType prop, parameterized queries
- [x] SkillMetrics.tsx: Accept agentType prop
- [x] GraduationPanel.tsx: Accept agentType prop
- [x] SkillsManager.tsx: Agent type tabs (swing/day)
- [x] TradeProposals.tsx: Agent type badges on signals
- [x] TypeScript + build verification pass

### Phase 4: Intraday Pattern Engine (ML) — COMPLETE
- [x] app/agent/ml/intraday/patterns.py — 9 patterns (ORB, Morning Reversal, VWAP Reclaim, Gap Fill, Power Hour, Mean Reversion, Momentum Breakout, Range Expansion, NO_PATTERN) + heuristic labeler
- [x] app/agent/ml/intraday/features.py — 40 intraday features (time-of-day, VWAP, OR, volume, lagged returns, RSI, MACD, ATR, BB, session stats, gap, bar-level)
- [x] app/agent/ml/intraday/classifier.py — XGBoost multi:softprob with class weighting, save/load, predict_single, top-3 patterns
- [x] app/agent/ml/intraday/trainer.py — Walk-forward pipeline (fetch bars → label → features → 70/30 split → train → evaluate)
- [x] Engine integration: pattern_classifier param on AlfiEngine, _enrich_with_patterns() on intraday snapshots
- [x] Coordinator integration: day_pattern_classifier param passed to Day engine
- [x] API: POST /api/ml/intraday/train, GET /api/ml/intraday/predictions/{symbol}, GET /api/ml/intraday/status
- [x] Frontend: PatternPanel.tsx (classifier status, per-pattern P/R/F1), integrated in Day Agent dashboard
- [x] End-to-end smoke test passed (label → features → train → predict → save → load)

### Phase 5: skfolio Portfolio Optimization — COMPLETE
- [x] app/agent/portfolio/optimizer.py — PortfolioOptimizer with HRP, MeanRisk, BlackLitterman strategies + build_returns_df helper
- [x] Integration with AgentCoordinator (portfolio_optimizer param, data_provider stored)
- [x] API endpoints: GET /api/portfolio/weights, POST /api/portfolio/rebalance (with strategy selector + symbol override), updated GET /api/portfolio/summary with optimizer status
- [x] Add skfolio>=0.4.0 dependency (installed via uv sync)
- [x] Frontend: PortfolioView.tsx updated with optimizer panel (strategy selector, rebalance button, weight bars, fit metrics)
- [x] types/portfolio.ts: Added OptimizerStatus, RebalanceResult interfaces
- [x] api.ts: Added fetchPortfolioWeights, rebalancePortfolio functions
- [x] TypeScript + Vite build verification passed

### Phase 6: Autonomous Execution — COMPLETE
- [x] alpaca-py>=0.21.0 added to pyproject.toml (installed v0.43.2)
- [x] Broker module already existed: alpaca_client.py (full SDK wrapper), order_manager.py (signal→order lifecycle + SQLite), position_tracker.py (positions + trade history)
- [x] app/agent/broker/position_sizer.py — Half-Kelly criterion with volatility scaling, confidence multiplier, portfolio weight overlay, clamped to [0.5%, 5%]
- [x] app/agent/broker/monitor.py — PositionMonitor background loop (polls positions, orders, account equity; detects fills/closes/P&L)
- [x] AgentCoordinator updated: accepts broker, position_tracker, position_monitor; passes broker to both engines; get_combined_state() includes broker status
- [x] api.py _get_coordinator() creates AlpacaClient → OrderManager → PositionTracker → PositionMonitor when Alpaca keys configured; graceful degradation when not
- [x] Portfolio endpoints use real Alpaca data: GET /api/portfolio/positions, GET /api/portfolio/summary
- [x] Broker API endpoints: GET /api/broker/status, GET /api/broker/orders, GET /api/broker/order-history, GET /api/broker/trade-history
- [x] Position monitor auto-starts with first agent, auto-stops when both agents stop
- [x] All Python imports verified, TypeScript builds clean

### Phase 7: Organic Pattern Discovery + Backtesting — COMPLETE
- [x] 7a. STUMPY motif discovery — app/agent/ml/discovery/motifs.py (MotifDiscoverer with discover, cross-symbol, cluster, evaluate, match, save/load)
- [x] 7b. Automated feature extraction — app/agent/ml/discovery/features.py (AutoFeatureExtractor with tsfresh offline + pycatch22 real-time)
- [x] 7c. Genetic rule evolution — app/agent/ml/discovery/evolver.py (StrategyEvolver with DEAP NSGA-II, walk-forward validation, rule_to_strategy)
- [x] 7d. Backtest runner enhancements — timeframe support (1d/5min), agent_type filter, TIMEFRAME_FREQ map, get_available_strategies(), run_discovered_backtest()
- [x] 7e. API endpoints — POST/GET /api/discovery/motifs, POST /api/discovery/evolve, GET /api/discovery/rules, POST /api/discovery/backtest, GET /api/backtest/strategies
- [x] 7f. Frontend DiscoveryPanel.tsx — motif library viewer, evolved rules list, run discovery/evolve buttons, summary stats
- [x] 7f. BacktestPage.tsx — agent type toggle (Swing/Day), dynamic strategy list from API, timeframe in config, discovered rules in dropdown
- [x] 7f. types/backtest.ts — added timeframe, agent_type to BacktestConfig
- [x] Dependencies: stumpy>=1.12.0, tsfresh>=0.20.0, pycatch22>=0.4.0, deap>=1.4.0
- [x] TypeScript build clean, Python imports verified, Vite production build successful

### Phase 8: Price Encoding + Transformer Pattern Recognition — COMPLETE
- [x] A1: Core encoder module — app/agent/ml/encoding/ (encoder.py, price_encoder.py, patterns.py)
  - encode_price(): OHLCV → token string (H3P1R0V2 format)
  - encode_price_df(): per-bar columns + numeric features
  - PriceEncoder class: encode_bars, encode_grouped, rolling_window, similarity
  - 11 pattern definitions (bullish/bearish/continuation) with regex matching
- [x] A2: Feature engine integration — encoding features in build_feature_matrix() + FeatureEngine snapshots
- [x] A3: Backtest strategies — encoded_pattern (swing) + day_encoded_pattern (day) registered
- [x] A4: Engine + coordinator — price_encoder param on AlfiEngine, _enrich_with_encoding(), coordinator instantiates PriceEncoder
- [x] A5: API endpoints — GET /api/encoding/{symbol}, GET /api/encoding/{symbol}/history, POST /api/encoding/similarity, GET /api/encoding/patterns
- [x] A6: Frontend panel — EncodingPanel.tsx with token display, pattern badges, cross-symbol matches
- [x] B1: Trade-GPT project setup — projects/trade-gpt/ with config.yaml, requirements.txt, src/
- [x] B2: Data pipeline — fetch_data.py (Massive API), encode_corpus.py (PriceEncoder → corpus.txt)
- [x] B3: Tokenizer — train_tokenizer.py (ByteLevelBPE, 2000 vocab)
- [x] B4: Training — train_mlm.py (RoBERTa 4L/8H/256D), train_causal.py (GPT 4L/8H/256D, PyTorch)
- [x] B5: Evaluation — evaluate.py (per-type accuracy, direction accuracy, walk-forward)
- [x] B6: Inference bridge — predict.py (PricePredictor.load_causal/load_mlm, predict_next → structured output)
- [x] No new dependencies in pyproject.toml (Part B deps in projects/trade-gpt/requirements.txt)
- [x] TypeScript build clean, Python imports verified, Vite production build successful

---

## Dependencies Summary

| Add | Remove |
|-----|--------|
| `skfolio>=0.4.0` (Phase 5 — done) | `playwright` (done) |
| `alpaca-py>=0.21.0` (Phase 6 — done) | |
| `stumpy>=1.12.0` (Phase 7 — done) | |
| `tsfresh>=0.20.0` (Phase 7 — done) | |
| `pycatch22>=0.4.0` (Phase 7 — done) | |
| `deap>=1.4.0` (Phase 7 — done) | |
| `torch, transformers, tokenizers` (Phase 8 — in projects/trade-gpt/requirements.txt only) | |
