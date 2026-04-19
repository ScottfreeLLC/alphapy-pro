"""
AlphaPy Markets API Server

HOW TO RUN:
> export MASSIVE_API_KEY=<your-api-key>
> cd app/backend
> uvicorn api:app --host 0.0.0.0 --port 8080 --reload
"""

import asyncio
import logging
import os
import sys
import threading
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from massive import WebSocketClient
from massive.websocket.models import WebSocketMessage, EquityTrade

# Import pivot service components (now local)
from pivots import pivothigh, pivotlow
from pattern_analyzer import PatternAnalyzer
from data_fetcher import MassiveDataFetcher
from bar_aggregator import BarAggregator
from substack_publisher import SubstackPublisher
from post_composer import PostComposer

# Initialize logger
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s\t%(message)s",
    level=logging.INFO,
    datefmt='%m/%d/%y %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global state
stock_data = {}
pattern_data = {}
symbols = []
data_fetcher = None
pattern_analyzer = None
bar_aggregator = None
substack_publisher = None
post_composer = None

# Real-time trade tracking
live_trade_counts: Dict[str, int] = {}  # Trades in last 5 seconds
live_prices: Dict[str, float] = {}  # Latest price per symbol
live_volumes: Dict[str, int] = {}  # Volume in last 5 seconds
trade_lock = threading.Lock()  # Thread-safe updates

# Alfi dual-agent coordinator
coordinator = None
alfi_ws_clients: List[WebSocket] = []

# Pydantic models
class PivotAnalysisRequest(BaseModel):
    symbol: str
    window_length: Optional[int] = 100
    minimum_strength: Optional[int] = 5

class ScreenerFilters(BaseModel):
    patterns: Optional[List[str]] = None
    sentiment: Optional[str] = None  # bullish, bearish, neutral
    min_strength: Optional[int] = None


# Massive WebSocket handler
def handle_massive_messages(msgs: List[WebSocketMessage]):
    """Handle incoming trade messages from Massive WebSocket."""
    global live_trade_counts, live_prices, live_volumes

    for m in msgs:
        if isinstance(m, EquityTrade):
            symbol = m.symbol
            if not isinstance(symbol, str):
                continue

            with trade_lock:
                # Update trade count
                live_trade_counts[symbol] = live_trade_counts.get(symbol, 0) + 1

                # Update latest price
                if isinstance(m.price, float):
                    live_prices[symbol] = m.price

                # Update volume
                if isinstance(m.price, float) and isinstance(m.size, int):
                    live_volumes[symbol] = live_volumes.get(symbol, 0) + m.size

            # Feed trades to bar aggregator
            if bar_aggregator and isinstance(m.price, float) and isinstance(m.size, int):
                timestamp_ms = int(m.timestamp) if hasattr(m, 'timestamp') and m.timestamp else int(datetime.now().timestamp() * 1000)
                bar_aggregator.process_trade(symbol, m.price, m.size, timestamp_ms)


def run_massive_websocket():
    """Run Massive WebSocket client in background thread."""
    logger.info("Starting Massive WebSocket stream...")
    try:
        client = WebSocketClient()  # Uses MASSIVE_API_KEY environment variable
        client.subscribe("T.*")    # Subscribe to all equity trades
        client.subscribe("XT.*")   # Subscribe to all crypto trades
        client.run(handle_massive_messages)
    except Exception as e:
        logger.error(f"Massive WebSocket error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global symbols, stock_data, data_fetcher, pattern_analyzer, bar_aggregator
    global substack_publisher, post_composer

    logger.info('*' * 80)
    logger.info("AlphaPy Markets API Server Start")
    logger.info('*' * 80)

    # Initialize services
    data_fetcher = MassiveDataFetcher()
    pattern_analyzer = PatternAnalyzer()
    bar_aggregator = BarAggregator(
        timeframes=["1min", "5min"],
        on_bar_complete=lambda bar: logger.debug(f"Bar complete: {bar.symbol} {bar.timeframe}"),
    )

    # Initialize Substack publisher + composer
    substack_publisher = SubstackPublisher()
    post_composer = PostComposer()
    if substack_publisher.is_configured():
        logger.info("Substack publisher configured")
    else:
        logger.info("Substack publisher not configured (set SUBSTACK_* env vars)")

    # Load initial symbols (demo mode: first 20)
    logger.info("Loading symbols...")
    all_symbols = data_fetcher.refresh_symbols('massive')
    symbols = all_symbols[:20]  # Demo mode
    logger.info(f"Loaded {len(symbols)} symbols")

    # Fetch historical data
    logger.info("Fetching historical data...")
    stock_data = data_fetcher.get_stock_data(symbols, days_back=100)
    logger.info(f"Loaded data for {len(stock_data)} symbols")

    # Start Massive WebSocket in background thread for real-time updates
    ws_thread = threading.Thread(target=run_massive_websocket, daemon=True)
    ws_thread.start()
    logger.info("Massive WebSocket thread started")

    yield

    # Shutdown
    await substack_publisher.close()
    logger.info("Shutting down AlphaPy Markets API Server")

# Initialize FastAPI app
app = FastAPI(
    title="AlphaPy Markets API",
    description="Market analysis API with pivot patterns and trading signals",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5331", "http://localhost:3000"],  # React on 5331
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "AlphaPy Markets API",
        "version": "1.0.0",
        "symbols_loaded": len(symbols),
        "data_loaded": len(stock_data)
    }

# Get all symbols
@app.get("/api/symbols")
async def get_symbols():
    return {
        "symbols": symbols,
        "count": len(symbols)
    }

# Get market summary
@app.get("/api/market/summary")
async def get_market_summary():
    if not stock_data:
        raise HTTPException(status_code=503, detail="Market data not loaded")

    summary = {
        "total_symbols": len(symbols),
        "data_loaded": len(stock_data),
        "timestamp": datetime.now().isoformat(),
    }
    return summary

# Analyze single symbol for pivots
@app.post("/api/pivots/analyze")
async def analyze_pivot(request: PivotAnalysisRequest):
    symbol = request.symbol.upper()

    if symbol not in stock_data:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    df = stock_data[symbol]

    try:
        # Run pattern analysis
        analysis = pattern_analyzer.analyze_stock(symbol, df)
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get pivot analysis for multiple symbols
@app.get("/api/pivots/scan")
async def scan_pivots(
    sentiment: Optional[str] = None,
    limit: Optional[int] = 20
):
    if not stock_data:
        raise HTTPException(status_code=503, detail="Market data not loaded")

    results = []
    for symbol in symbols[:limit]:
        if symbol in stock_data:
            try:
                analysis = pattern_analyzer.analyze_stock(symbol, stock_data[symbol])

                # Filter by sentiment if specified
                if sentiment:
                    overall_sentiment = analysis.get('summary', {}).get('overall_sentiment', 'neutral')
                    if overall_sentiment != sentiment:
                        continue

                results.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue

    return {
        "results": results,
        "count": len(results),
        "timestamp": datetime.now().isoformat()
    }

# Get bullish stocks
@app.get("/api/pivots/bullish")
async def get_bullish():
    return await scan_pivots(sentiment="bullish", limit=20)

# Get bearish stocks
@app.get("/api/pivots/bearish")
async def get_bearish():
    return await scan_pivots(sentiment="bearish", limit=20)

# Get stock quote
@app.get("/api/quote/{symbol}")
async def get_quote(symbol: str):
    symbol = symbol.upper()

    if symbol not in stock_data:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

    df = stock_data[symbol]
    latest = df.iloc[-1]

    return {
        "symbol": symbol,
        "price": float(latest['close']),
        "high": float(latest['high']),
        "low": float(latest['low']),
        "open": float(latest['open']),
        "volume": int(latest['volume']),
        "date": str(latest['date']),
        "timestamp": datetime.now().isoformat()
    }

# Stock screener endpoint - returns all stocks with calculated fields
@app.get("/api/stocks/screener")
async def get_stock_screener():
    if not stock_data:
        raise HTTPException(status_code=503, detail="Market data not loaded")

    screener_data = []
    for symbol in symbols:
        if symbol not in stock_data:
            continue

        df = stock_data[symbol]
        if len(df) < 2:
            continue

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        # Use live price if available, otherwise use historical
        with trade_lock:
            current_price = live_prices.get(symbol, float(latest['close']))
            current_volume = live_volumes.get(symbol, int(latest['volume']))

        # Calculate daily change using live price
        daily_change = current_price - float(previous['close'])
        daily_change_pct = (daily_change / float(previous['close'])) * 100 if float(previous['close']) != 0 else 0

        screener_data.append({
            "symbol": symbol,
            "price": current_price,
            "high": float(latest['high']),
            "low": float(latest['low']),
            "open": float(latest['open']),
            "volume": current_volume,
            "daily_change": round(daily_change, 2),
            "daily_change_pct": round(daily_change_pct, 2)
        })

    return {
        "stocks": screener_data,
        "count": len(screener_data),
        "timestamp": datetime.now().isoformat()
    }

# WebSocket clients for market data
market_ws_clients: List[WebSocket] = []


# WebSocket for real-time market data
@app.websocket("/ws/market-data")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    market_ws_clients.append(websocket)
    logger.info("Market data WebSocket client connected")

    try:
        while True:
            # Build live price/volume snapshot from Massive WS trades
            with trade_lock:
                price_data = {
                    sym: {"price": price, "volume": live_volumes.get(sym, 0)}
                    for sym, price in live_prices.items()
                }

            if price_data:
                await websocket.send_json({
                    "type": "price_update",
                    "data": price_data,
                    "timestamp": datetime.now().isoformat(),
                })
            else:
                await websocket.send_json({
                    "type": "heartbeat",
                    "timestamp": datetime.now().isoformat(),
                })

            await asyncio.sleep(2)
    except WebSocketDisconnect:
        if websocket in market_ws_clients:
            market_ws_clients.remove(websocket)
        logger.info("Market data WebSocket client disconnected")


# ============================================================================
# Alfi Agent API Endpoints
# ============================================================================

VALID_AGENT_TYPES = {"swing", "day"}


def _get_coordinator():
    """Lazy-initialize the dual-agent coordinator with shared infrastructure."""
    global coordinator
    if coordinator is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from agent.config import AgentConfig
        from agent.coordinator import AgentCoordinator
        from agent.features.engine import FeatureEngine
        from agent.risk.rules import CircuitBreaker
        from agent.risk.manager import RiskManager
        from agent.risk.shared import SharedRiskManager
        from agent.autonomy.performance import PerformanceTracker
        from agent.autonomy.confidence import ConfidenceScorer
        from agent.autonomy.graduation import GraduationManager
        from data_provider import DataProvider
        from data_cache import DataCache

        # Shared data provider
        feature_engine = FeatureEngine()
        dp = DataProvider(
            fetcher=data_fetcher,
            cache=DataCache(),
            aggregator=bar_aggregator,
            stock_data=stock_data,
            live_prices=live_prices,
            trade_lock=trade_lock,
            feature_engine=feature_engine,
        )

        # Per-agent configs
        swing_config = AgentConfig.swing_config()
        day_config = AgentConfig.day_config()

        # Per-agent safety rails
        swing_cb = CircuitBreaker(swing_config)
        swing_rm = RiskManager(swing_config, swing_cb)
        swing_pt = PerformanceTracker(
            db_path=os.path.join(swing_config.data_dir, "performance", "swing_metrics.db")
        )
        swing_cs = ConfidenceScorer(swing_pt)
        swing_gm = GraduationManager(swing_pt)

        day_cb = CircuitBreaker(day_config)
        day_rm = RiskManager(day_config, day_cb)
        day_pt = PerformanceTracker(
            db_path=os.path.join(day_config.data_dir, "performance", "day_metrics.db")
        )
        day_cs = ConfidenceScorer(day_pt)
        day_gm = GraduationManager(day_pt)

        # Shared risk manager
        shared_risk = SharedRiskManager(circuit_breaker=CircuitBreaker(swing_config))

        # Portfolio optimizer (skfolio)
        portfolio_optimizer = None
        try:
            from agent.portfolio.optimizer import PortfolioOptimizer, OptStrategy
            portfolio_optimizer = PortfolioOptimizer(strategy=OptStrategy.HRP)
            logger.info("Portfolio optimizer initialized (HRP strategy)")
        except Exception as e:
            logger.warning(f"Failed to initialize portfolio optimizer: {e}")

        # Intraday pattern classifier (load latest model if available)
        day_pattern_classifier = None
        try:
            from agent.ml.intraday.classifier import IntradayClassifier
            latest_model = IntradayClassifier.find_latest_model()
            if latest_model:
                day_pattern_classifier = IntradayClassifier(model_path=latest_model)
                logger.info(f"Loaded intraday pattern classifier: {latest_model}")
            else:
                day_pattern_classifier = IntradayClassifier()
                logger.info("No intraday pattern model found — classifier available but untrained")
        except Exception as e:
            logger.warning(f"Failed to load intraday pattern classifier: {e}")

        # Broker infrastructure (Alpaca) — shared by both agents
        alpaca_client = None
        order_manager = None
        position_tracker = None
        position_monitor = None
        try:
            from agent.broker.alpaca_client import AlpacaClient
            from agent.broker.order_manager import OrderManager
            from agent.broker.position_tracker import PositionTracker
            from agent.broker.monitor import PositionMonitor

            alpaca_client = AlpacaClient(swing_config)  # shared account
            if alpaca_client.connected:
                order_manager = OrderManager(alpaca_client)
                position_tracker = PositionTracker(alpaca_client)
                position_monitor = PositionMonitor(
                    alpaca=alpaca_client,
                    position_tracker=position_tracker,
                )
                logger.info("Broker infrastructure initialized (Alpaca connected)")
            else:
                logger.info("Alpaca not configured — broker features disabled (paper trading mode)")
        except Exception as e:
            logger.warning(f"Failed to initialize broker: {e}")

        coordinator = AgentCoordinator(
            swing_config=swing_config,
            day_config=day_config,
            data_provider=dp,
            shared_risk=shared_risk,
            on_state_change=_broadcast_state,
            swing_risk_manager=swing_rm,
            day_risk_manager=day_rm,
            swing_performance=swing_pt,
            day_performance=day_pt,
            swing_confidence=swing_cs,
            day_confidence=day_cs,
            swing_graduation=swing_gm,
            day_graduation=day_gm,
            day_pattern_classifier=day_pattern_classifier,
            portfolio_optimizer=portfolio_optimizer,
            broker=order_manager,
            position_tracker=position_tracker,
            position_monitor=position_monitor,
        )
    return coordinator


def _get_engine(agent_type: str = "swing"):
    """Get an engine from the coordinator by agent type."""
    coord = _get_coordinator()
    return coord.get_engine(agent_type)


def _validate_agent_type(agent_type: str):
    """Validate agent_type path parameter."""
    if agent_type not in VALID_AGENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid agent_type '{agent_type}'. Must be 'swing' or 'day'.",
        )


async def _broadcast_state(state_dict: Dict):
    """Broadcast combined agent state to all connected WebSocket clients."""
    for ws in alfi_ws_clients[:]:
        try:
            await ws.send_json({"type": "state_update", "data": state_dict})
        except Exception:
            alfi_ws_clients.remove(ws)


# ---- Per-agent endpoints: /api/agent/{agent_type}/... ----

@app.get("/api/agent/{agent_type}/status")
async def get_agent_status(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    return engine.state.to_dict()


class AgentModeRequest(BaseModel):
    mode: str  # approval | semi_autonomous | autonomous


@app.post("/api/agent/{agent_type}/start")
async def start_agent(agent_type: str):
    _validate_agent_type(agent_type)
    coord = _get_coordinator()
    await coord.start(agent_type)
    # Start position monitor when first agent starts
    if coord.position_monitor and not coord.position_monitor.running:
        await coord.position_monitor.start()
    return {"status": "started", "agent_type": agent_type, "state": coord.get_engine(agent_type).state.to_dict()}


@app.post("/api/agent/{agent_type}/stop")
async def stop_agent(agent_type: str):
    _validate_agent_type(agent_type)
    coord = _get_coordinator()
    await coord.stop(agent_type)
    # Stop position monitor when both agents are stopped
    if coord.position_monitor and coord.position_monitor.running:
        both_stopped = not coord.swing.state.running and not coord.day.state.running
        if both_stopped:
            await coord.position_monitor.stop()
    return {"status": "stopped", "agent_type": agent_type, "state": coord.get_engine(agent_type).state.to_dict()}


@app.post("/api/agent/{agent_type}/mode")
async def set_agent_mode(agent_type: str, request: AgentModeRequest):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    engine.set_autonomy_mode(request.mode)
    return {"agent_type": agent_type, "mode": engine.state.autonomy_mode.value}


# ---- Combined status ----

@app.get("/api/agent/combined/status")
async def get_combined_status():
    coord = _get_coordinator()
    return coord.get_combined_state()


# ---- Signals (aggregate from both agents) ----

@app.get("/api/signals/pending")
async def get_pending_signals():
    coord = _get_coordinator()
    signals = coord.get_all_pending_signals()
    return {"signals": signals, "count": len(signals)}


@app.get("/api/signals/recent")
async def get_recent_signals():
    coord = _get_coordinator()
    signals = coord.get_all_recent_signals()
    return {"signals": signals, "count": len(signals)}


@app.post("/api/signals/{signal_id}/approve")
async def approve_signal(signal_id: str):
    coord = _get_coordinator()
    # Search both engines for the signal
    for agent_type in VALID_AGENT_TYPES:
        engine = coord.get_engine(agent_type)
        signal = engine.approve_signal(signal_id)
        if signal:
            return {"status": "approved", "agent_type": agent_type, "signal": signal.to_dict()}
    raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")


@app.post("/api/signals/{signal_id}/reject")
async def reject_signal(signal_id: str):
    coord = _get_coordinator()
    for agent_type in VALID_AGENT_TYPES:
        engine = coord.get_engine(agent_type)
        signal = engine.reject_signal(signal_id)
        if signal:
            return {"status": "rejected", "agent_type": agent_type, "signal": signal.to_dict()}
    raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")


# ---- Skills (per-agent) ----

@app.get("/api/agent/{agent_type}/skills")
async def get_skills(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    return {"agent_type": agent_type, "skills": engine.registry.list_skills()}


class SkillToggleRequest(BaseModel):
    enabled: bool


@app.put("/api/agent/{agent_type}/skills/{skill_name}/toggle")
async def toggle_skill(agent_type: str, skill_name: str, request: SkillToggleRequest):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    found = engine.registry.toggle_skill(skill_name, request.enabled)
    if found:
        return {"agent_type": agent_type, "skill": skill_name, "enabled": request.enabled}
    raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")


# ---- Signal outcome recording ----

class SignalOutcomeRequest(BaseModel):
    exit_price: float
    exit_time: Optional[str] = None


@app.post("/api/signals/{signal_id}/outcome")
async def record_signal_outcome(signal_id: str, request: SignalOutcomeRequest):
    """Record the outcome of a trade for performance tracking."""
    coord = _get_coordinator()

    # Find the signal in either engine
    for agent_type in VALID_AGENT_TYPES:
        engine = coord.get_engine(agent_type)
        signal = None
        for s in engine.state.recent_signals:
            if s.id == signal_id:
                signal = s
                break
        if not signal:
            continue

        if engine.performance_tracker:
            engine.performance_tracker.record_trade(
                symbol=signal.symbol,
                direction=signal.direction.value,
                skill_name=signal.skill_name,
                signal_id=signal.id,
                entry_price=signal.entry_price,
                exit_price=request.exit_price,
                qty=1.0,
                entry_time=signal.created_at.isoformat(),
                exit_time=request.exit_time or datetime.now().isoformat(),
            )
            profitable = (
                (request.exit_price > signal.entry_price and signal.direction.value == "long")
                or (request.exit_price < signal.entry_price and signal.direction.value == "short")
            )
            if engine.graduation_manager:
                engine.graduation_manager.record_trade_outcome(profitable)
            if engine.confidence_scorer:
                engine.confidence_scorer.calibrate(signal, profitable)

        return {"status": "recorded", "agent_type": agent_type, "signal_id": signal_id}

    raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")


# ---- Risk status (per-agent + shared) ----

@app.get("/api/agent/{agent_type}/risk/status")
async def get_agent_risk_status(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    result = {"agent_type": agent_type, "circuit_breaker": {"tripped": False}, "exposure": {}}
    if engine.risk_manager:
        result["exposure"] = engine.risk_manager.get_exposure()
        if engine.risk_manager.circuit_breaker:
            result["circuit_breaker"] = engine.risk_manager.circuit_breaker.get_status()
    return result


@app.get("/api/risk/status")
async def get_risk_status():
    coord = _get_coordinator()
    return {
        "shared": coord.shared_risk.get_status(),
        "swing": _risk_for_engine(coord.swing),
        "day": _risk_for_engine(coord.day),
    }


def _risk_for_engine(engine):
    """Extract risk status from a single engine."""
    if engine.risk_manager:
        result = {"exposure": engine.risk_manager.get_exposure()}
        if engine.risk_manager.circuit_breaker:
            result["circuit_breaker"] = engine.risk_manager.circuit_breaker.get_status()
        return result
    return {}


@app.post("/api/agent/{agent_type}/risk/reset")
async def reset_agent_circuit_breaker(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    if engine.risk_manager and engine.risk_manager.circuit_breaker:
        engine.risk_manager.circuit_breaker.reset()
        return {"status": "reset", "agent_type": agent_type}
    raise HTTPException(status_code=400, detail="No circuit breaker configured")


# ---- Performance metrics (per-agent) ----

@app.get("/api/agent/{agent_type}/performance/metrics")
async def get_agent_performance_metrics(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    if engine.performance_tracker:
        return engine.performance_tracker.get_metrics()
    return {"total_trades": 0, "message": "Performance tracker not configured"}


@app.get("/api/agent/{agent_type}/performance/skills")
async def get_agent_skill_performance(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    if engine.performance_tracker:
        return {"agent_type": agent_type, "skills": engine.performance_tracker.get_skill_metrics()}
    return {"skills": {}}


# ---- Graduation (per-agent) ----

@app.get("/api/agent/{agent_type}/graduation/status")
async def get_graduation_status(agent_type: str):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    if engine.graduation_manager:
        return engine.graduation_manager.get_graduation_status()
    return {"message": "Graduation manager not configured"}


class GraduationLockRequest(BaseModel):
    locked: bool


@app.post("/api/agent/{agent_type}/graduation/lock")
async def lock_graduation(agent_type: str, request: GraduationLockRequest):
    _validate_agent_type(agent_type)
    engine = _get_engine(agent_type)
    if request.locked:
        engine.graduation_manager = None
        return {"status": "locked", "agent_type": agent_type, "message": "Automatic graduation disabled"}
    return {"status": "already_active"}


# Portfolio
@app.get("/api/portfolio/positions")
async def get_portfolio_positions():
    coord = _get_coordinator()
    if coord.position_tracker:
        positions = coord.position_tracker.get_positions()
        return {"positions": positions, "count": len(positions)}
    return {"positions": [], "count": 0}


@app.get("/api/portfolio/summary")
async def get_portfolio_summary():
    coord = _get_coordinator()
    optimizer_status = {}
    if coord.portfolio_optimizer:
        optimizer_status = coord.portfolio_optimizer.get_status()

    # Use real Alpaca data if broker is connected
    if coord.position_tracker:
        summary = coord.position_tracker.get_portfolio_summary()
        summary["optimizer"] = optimizer_status
        return summary

    return {
        "equity": 0,
        "cash": 0,
        "buying_power": 0,
        "positions_count": 0,
        "total_unrealized_pl": 0,
        "exposure_pct": 0,
        "positions": [],
        "optimizer": optimizer_status,
    }


@app.get("/api/portfolio/weights")
async def get_portfolio_weights():
    """Get current portfolio optimization weights."""
    coord = _get_coordinator()
    if not coord.portfolio_optimizer:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not initialized")
    return coord.portfolio_optimizer.get_status()


class RebalanceRequest(BaseModel):
    strategy: str = "hrp"
    symbols: Optional[List[str]] = None
    days_back: int = 252


@app.post("/api/portfolio/rebalance")
async def rebalance_portfolio(request: RebalanceRequest):
    """Fit/rebalance the portfolio optimizer on historical returns."""
    coord = _get_coordinator()
    if not coord.portfolio_optimizer:
        raise HTTPException(status_code=503, detail="Portfolio optimizer not initialized")

    # Set strategy if changed
    from agent.portfolio.optimizer import OptStrategy
    try:
        coord.portfolio_optimizer.strategy = OptStrategy(request.strategy)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid strategy: {request.strategy}")

    # Collect symbols from both agents' watchlists
    symbols = request.symbols
    if not symbols:
        symbols = list(set(
            coord.swing.config.symbols + coord.day.config.symbols
        ))
    if len(symbols) < 3:
        raise HTTPException(status_code=400, detail=f"Need at least 3 symbols, got {len(symbols)}")

    # Build returns matrix
    from agent.portfolio.optimizer import build_returns_df
    returns_df = build_returns_df(coord.data_provider, symbols, days_back=request.days_back)
    if returns_df.empty:
        raise HTTPException(status_code=400, detail="Could not build returns matrix — insufficient data")

    # Fit optimizer
    result = coord.portfolio_optimizer.fit(returns_df)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "status": "rebalanced",
        "result": result,
        "optimizer": coord.portfolio_optimizer.get_status(),
    }


# ---- Broker status & orders ----

@app.get("/api/broker/status")
async def get_broker_status():
    """Get broker connection status, account info, and position monitor state."""
    coord = _get_coordinator()
    result = {"connected": False, "account": None, "monitor": None, "orders_tracked": 0}

    if coord.position_tracker and coord.position_tracker.alpaca.connected:
        result["connected"] = True
        result["account"] = coord.position_tracker.alpaca.get_account()

    if coord.position_monitor:
        result["monitor"] = coord.position_monitor.get_status()

    return result


@app.get("/api/broker/orders")
async def get_broker_orders(status: str = "open"):
    """Get orders from Alpaca."""
    coord = _get_coordinator()
    if not coord.position_tracker or not coord.position_tracker.alpaca.connected:
        return {"orders": [], "count": 0, "connected": False}

    orders = coord.position_tracker.alpaca.get_orders(status=status)
    return {"orders": orders, "count": len(orders), "connected": True}


@app.get("/api/broker/order-history")
async def get_order_history(limit: int = 50):
    """Get order history from local database."""
    coord = _get_coordinator()
    # Access order_manager through the engine's broker
    if coord.swing.broker and hasattr(coord.swing.broker, 'get_order_history'):
        history = coord.swing.broker.get_order_history(limit=limit)
        return {"orders": history, "count": len(history)}
    return {"orders": [], "count": 0}


@app.get("/api/broker/trade-history")
async def get_trade_history(limit: int = 100, skill: Optional[str] = None):
    """Get closed trade history from position tracker."""
    coord = _get_coordinator()
    if not coord.position_tracker:
        return {"trades": [], "count": 0}

    trades = coord.position_tracker.get_trade_history(limit=limit, skill_name=skill)
    return {"trades": trades, "count": len(trades)}


# Alfi WebSocket — real-time combined agent state + signals
@app.websocket("/ws/alfi")
async def alfi_websocket(websocket: WebSocket):
    await websocket.accept()
    alfi_ws_clients.append(websocket)
    logger.info("Alfi WebSocket client connected")

    try:
        # Send initial combined state
        coord = _get_coordinator()
        await websocket.send_json({
            "type": "state_update",
            "data": coord.get_combined_state(),
        })

        while True:
            # Keep connection alive + send periodic combined state
            await asyncio.sleep(3)
            await websocket.send_json({
                "type": "state_update",
                "data": coord.get_combined_state(),
            })
    except WebSocketDisconnect:
        if websocket in alfi_ws_clients:
            alfi_ws_clients.remove(websocket)
        logger.info("Alfi WebSocket client disconnected")


# ============================================================================
# Backtesting API Endpoints
# ============================================================================

class BacktestRequest(BaseModel):
    strategy: str                        # momentum_breakout | mean_reversion | crypto_momentum | pivot_pattern_entry
    symbols: List[str]                   # ["AAPL", "MSFT"] or ["X:BTCUSD"]
    start_date: str                      # "2025-06-01"
    end_date: str                        # "2026-02-14"
    initial_capital: float = 100000
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    timeframe: str = "1d"                # "1d" or "5min"
    agent_type: str = ""                 # "swing", "day", or "" for all


@app.post("/api/backtest/run")
async def run_backtest_endpoint(request: BacktestRequest):
    """Run a backtest synchronously and return full results."""
    import asyncio
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.backtest.runner import run_backtest

    config = request.model_dump()

    try:
        result = await asyncio.to_thread(run_backtest, config)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/backtest/runs")
async def list_backtest_runs():
    """List past backtest runs (summary only)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.backtest.results import list_runs

    runs = list_runs()
    return {"runs": runs, "count": len(runs)}


@app.get("/api/backtest/strategies")
async def list_backtest_strategies(agent_type: str = ""):
    """Return available backtest strategies, optionally filtered by agent type."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.backtest.runner import get_available_strategies

    strategies = get_available_strategies(agent_type)
    return {"strategies": strategies, "count": len(strategies)}


@app.get("/api/backtest/{run_id}")
async def get_backtest_run(run_id: str):
    """Get full results for a specific backtest run."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.backtest.results import get_run

    result = get_run(run_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return result


# ============================================================================
# Enriched Screener Endpoint (Phase 5)
# ============================================================================

@app.get("/api/stocks/screener/enriched")
async def get_enriched_screener():
    """Screener + technical indicators + patterns for each stock."""
    if not stock_data:
        raise HTTPException(status_code=503, detail="Market data not loaded")

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from agent.features.engine import FeatureEngine

    fe = FeatureEngine()
    screener_data = []

    for symbol in symbols:
        if symbol not in stock_data:
            continue

        df = stock_data[symbol]
        if len(df) < 2:
            continue

        latest = df.iloc[-1]
        previous = df.iloc[-2]

        with trade_lock:
            current_price = live_prices.get(symbol, float(latest['close']))
            current_volume = live_volumes.get(symbol, int(latest['volume']))

        daily_change = current_price - float(previous['close'])
        daily_change_pct = (daily_change / float(previous['close'])) * 100 if float(previous['close']) != 0 else 0

        # Compute indicators
        bars = [
            {
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": int(row.get("volume", 0)),
            }
            for _, row in df.iterrows()
        ]
        snapshot = {
            "symbol": symbol,
            "current_price": current_price,
            "bars_1d": bars,
        }
        enriched = fe.compute_features(snapshot)
        indicators = enriched.get("indicators", {})

        # Pivot patterns
        patterns = []
        try:
            analysis = pattern_analyzer.analyze_stock(symbol, df)
            patterns = analysis.get("summary", {}).get("detected_patterns", [])
            sentiment = analysis.get("summary", {}).get("overall_sentiment", "neutral")
        except Exception:
            sentiment = "neutral"

        screener_data.append({
            "symbol": symbol,
            "price": current_price,
            "high": float(latest['high']),
            "low": float(latest['low']),
            "open": float(latest['open']),
            "volume": current_volume,
            "daily_change": round(daily_change, 2),
            "daily_change_pct": round(daily_change_pct, 2),
            "sma20": indicators.get("trend", {}).get("sma20"),
            "rsi_14": indicators.get("momentum", {}).get("rsi_14"),
            "volume_ratio": indicators.get("volume", {}).get("volume_ratio"),
            "trend_summary": enriched.get("trend_summary", ""),
            "patterns": patterns,
            "sentiment": sentiment,
        })

    return {
        "stocks": screener_data,
        "count": len(screener_data),
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================================
# Chat Endpoint (Phase 5)
# ============================================================================

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = None


@app.post("/api/chat")
async def chat_with_alfi(request: ChatRequest):
    """Conversational interface to Alfi agent."""
    coord = _get_coordinator()
    swing = coord.swing
    day = coord.day

    # Build context from current market state
    context_parts = []
    recent_signals = coord.get_all_recent_signals()[:10]
    if recent_signals:
        context_parts.append(f"Recent signals: {recent_signals}")

    # Market summary
    if stock_data:
        market_summary = {
            "symbols_loaded": len(stock_data),
            "timestamp": datetime.now().isoformat(),
        }
        context_parts.append(f"Market: {market_summary}")

    # Agent states
    context_parts.append(f"Swing agent: {swing.state.status.value}, mode: {swing.state.autonomy_mode.value}, cycles: {swing.state.cycle_count}")
    context_parts.append(f"Day agent: {day.state.status.value}, mode: {day.state.autonomy_mode.value}, cycles: {day.state.cycle_count}")

    system_prompt = f"""You are Alfi, a dual-agent AI trading system with a Swing Agent (multi-day holds) and Day Agent (intraday). Answer questions about market conditions, trading signals, portfolio, and strategy performance.

Current context:
{chr(10).join(context_parts)}

{request.context or ''}

Be concise and actionable. Reference specific data when available."""

    import litellm

    try:
        response = litellm.completion(
            model=swing.config.eval_model,
            max_tokens=1024,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.message},
            ],
        )
        reply = response.choices[0].message.content.strip()
        return {"reply": reply, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ML Backtest Endpoints
# ============================================================================

class MLBacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100000
    train_pct: float = 0.7
    pt_sl: List[float] = [1.0, 1.0]
    vertical_barrier_periods: int = 10
    ml_threshold: float = 0.5


@app.post("/api/backtest/ml")
async def run_ml_backtest_endpoint(request: MLBacktestRequest):
    """Run ML backtest with triple barrier labeling and meta-model."""
    import asyncio
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.ml.backtest_ml import run_ml_backtest

    config = request.model_dump()

    try:
        result = await asyncio.to_thread(
            run_ml_backtest,
            config,
            data_fetcher=data_fetcher,
            stock_data=stock_data,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"ML Backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Intraday Pattern ML Endpoints
# ============================================================================

class IntradayTrainRequest(BaseModel):
    symbols: Optional[List[str]] = None
    days_back: int = 90
    train_pct: float = 0.7


@app.post("/api/ml/intraday/train")
async def train_intraday_classifier(request: IntradayTrainRequest):
    """Train the intraday pattern classifier on historical 5-min bars."""
    import asyncio
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.ml.intraday.trainer import IntradayTrainer

    symbols = request.symbols or DEFAULT_WATCHLIST[:10]

    trainer = IntradayTrainer(
        data_fetcher=data_fetcher,
        data_provider=_get_coordinator().day.data_provider,
    )

    try:
        result = await asyncio.to_thread(
            trainer.train,
            symbols=symbols,
            days_back=request.days_back,
            train_pct=request.train_pct,
        )

        # Reload model in the day engine if training succeeded
        if result.get("status") == "success" and result.get("model_path"):
            try:
                from agent.ml.intraday.classifier import IntradayClassifier
                coord = _get_coordinator()
                coord.day.pattern_classifier = IntradayClassifier(
                    model_path=result["model_path"]
                )
                logger.info("Day engine pattern classifier reloaded with new model")
            except Exception as e:
                logger.warning(f"Failed to reload classifier: {e}")

        return result
    except Exception as e:
        logger.error(f"Intraday training error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/intraday/predictions/{symbol}")
async def get_intraday_predictions(symbol: str):
    """Get current pattern prediction for a symbol using the day agent's classifier."""
    coord = _get_coordinator()
    classifier = coord.day.pattern_classifier

    if classifier is None or not classifier.is_loaded:
        return {
            "symbol": symbol,
            "prediction": None,
            "model_loaded": False,
            "message": "No trained model available. Use POST /api/ml/intraday/train first.",
        }

    # Get latest 5-min bars for the symbol
    dp = coord.day.data_provider
    if dp is None:
        raise HTTPException(status_code=503, detail="Data provider not available")

    try:
        bars = dp.get_bars(symbol, timeframe="5min", days_back=5)
        if not bars or len(bars) < 10:
            return {
                "symbol": symbol,
                "prediction": None,
                "model_loaded": True,
                "message": f"Insufficient 5-min bar data for {symbol}",
            }

        import pandas as pd
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from agent.ml.intraday.features import build_intraday_features
        from agent.ml.intraday.patterns import _detect_session_breaks

        df = pd.DataFrame(bars)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        if "vwap" in df.columns:
            df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # Build features for last session
        session_breaks = _detect_session_breaks(df)
        last_session_start = session_breaks[-1] if session_breaks else 0
        session_df = df.iloc[last_session_start:]

        features = build_intraday_features(session_df)
        if len(features) == 0:
            return {"symbol": symbol, "prediction": None, "model_loaded": True}

        predictions = classifier.predict(features)

        return {
            "symbol": symbol,
            "model_loaded": True,
            "latest": predictions[-1] if predictions else None,
            "session_bars": len(session_df),
            "predictions_count": len(predictions),
        }
    except Exception as e:
        logger.error(f"Pattern prediction error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/intraday/status")
async def get_intraday_model_status():
    """Get the status of the intraday pattern classifier."""
    coord = _get_coordinator()
    classifier = coord.day.pattern_classifier

    if classifier is None:
        return {"loaded": False, "message": "Classifier not initialized"}

    return {
        "loaded": classifier.is_loaded,
        "training_metrics": classifier.training_metrics,
    }


# ============================================================================
# Substack Publishing Endpoints
# ============================================================================

class SubstackComposeRequest(BaseModel):
    post_type: str  # daily_signals | backtest_report | signal_note | custom
    # For daily_signals: no extra fields needed
    # For backtest_report:
    run_id: Optional[str] = None
    # For signal_note:
    signal_id: Optional[str] = None
    # For custom:
    title: Optional[str] = None
    markdown: Optional[str] = None
    tags: Optional[List[str]] = None


class SubstackPublishRequest(BaseModel):
    draft_id: int
    send_email: bool = True


@app.get("/api/substack/status")
async def get_substack_status():
    """Check if Substack publishing is configured."""
    return {
        "configured": substack_publisher.is_configured(),
        "publication_url": substack_publisher.publication_url or None,
    }


@app.post("/api/substack/compose")
async def compose_substack_post(request: SubstackComposeRequest):
    """Preview a post (returns content without publishing)."""
    try:
        composed = await _compose_post(request)
        return composed.to_dict()
    except Exception as e:
        logger.error(f"Compose failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/substack/draft")
async def create_substack_draft(request: SubstackComposeRequest):
    """Create a draft on Substack."""
    if not substack_publisher.is_configured():
        raise HTTPException(status_code=400, detail="Substack not configured")

    try:
        composed = await _compose_post(request)
        draft = await substack_publisher.create_draft(
            title=composed.title,
            body_html=composed.to_html(),
            subtitle=composed.subtitle,
        )
        return {"draft": draft, "post": composed.to_dict()}
    except Exception as e:
        logger.error(f"Draft creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/substack/publish")
async def publish_substack_draft(request: SubstackPublishRequest):
    """Publish an existing draft."""
    if not substack_publisher.is_configured():
        raise HTTPException(status_code=400, detail="Substack not configured")

    try:
        await substack_publisher.prepublish(request.draft_id)
        result = await substack_publisher.publish(request.draft_id, request.send_email)
        return {"status": "published", "result": result}
    except Exception as e:
        logger.error(f"Publish failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SubstackCustomPublishRequest(BaseModel):
    title: str
    markdown: str
    subtitle: str = ""
    audience: str = "everyone"
    send_email: bool = True
    tags: Optional[List[str]] = None


@app.post("/api/substack/publish-custom")
async def publish_custom_substack(request: SubstackCustomPublishRequest):
    """Publish a custom markdown post directly."""
    if not substack_publisher.is_configured():
        raise HTTPException(status_code=400, detail="Substack not configured")

    try:
        html = SubstackPublisher.markdown_to_html(request.markdown)
        result = await substack_publisher.full_publish(
            title=request.title,
            body_html=html,
            subtitle=request.subtitle,
            audience=request.audience,
            send_email=request.send_email,
        )
        return {"status": "published", "result": result}
    except Exception as e:
        logger.error(f"Custom publish failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _compose_post(request: SubstackComposeRequest):
    """Internal helper to compose a post based on request type."""
    if request.post_type == "daily_signals":
        coord = _get_coordinator()
        signals = coord.get_all_recent_signals()[:20]
        market_summary = {
            "total_symbols": len(symbols),
            "timestamp": datetime.now().isoformat(),
        }
        return post_composer.compose_daily_signal_post(
            signals=signals,
            market_summary=market_summary,
        )

    elif request.post_type == "backtest_report":
        if not request.run_id:
            raise ValueError("run_id required for backtest_report")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from agent.backtest.results import get_run
        result = get_run(request.run_id)
        if not result:
            raise ValueError(f"Backtest run {request.run_id} not found")
        return post_composer.compose_backtest_report(result=result)

    elif request.post_type == "signal_note":
        if not request.signal_id:
            raise ValueError("signal_id required for signal_note")
        coord = _get_coordinator()
        signal = None
        for s_dict in coord.get_all_recent_signals():
            if s_dict.get("id") == request.signal_id:
                signal = s_dict
                break
        if not signal:
            raise ValueError(f"Signal {request.signal_id} not found")
        return post_composer.compose_signal_note(signal=signal)

    elif request.post_type == "custom":
        if not request.title or not request.markdown:
            raise ValueError("title and markdown required for custom posts")
        return post_composer.compose_custom(
            title=request.title,
            markdown=request.markdown,
            tags=request.tags,
        )

    else:
        raise ValueError(f"Unknown post_type: {request.post_type}")




# ============================================================================
# Pattern Discovery API Endpoints (Phase 7)
# ============================================================================

class DiscoveryMotifRequest(BaseModel):
    symbols: List[str] = ["SPY", "AAPL", "MSFT"]
    days_back: int = 90
    window_sizes: List[int] = [20, 40, 60]

class DiscoveryEvolveRequest(BaseModel):
    symbols: List[str] = ["SPY", "AAPL", "MSFT"]
    days_back: int = 90
    generations: int = 50
    population: int = 300
    window_size: int = 20

class DiscoveryBacktestRequest(BaseModel):
    symbols: List[str] = ["SPY", "AAPL", "MSFT"]
    start_date: str = "2025-06-01"
    end_date: str = "2026-02-14"
    initial_capital: float = 100000


@app.post("/api/discovery/motifs")
async def run_motif_discovery(request: DiscoveryMotifRequest):
    """Run STUMPY motif discovery on recent 5-min data."""
    import asyncio
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.ml.discovery.motifs import MotifDiscoverer

    discoverer = MotifDiscoverer()
    # Try to load existing library first
    discoverer.load_motif_library()

    data_provider = _get_data_provider_instance()

    all_motifs = []
    symbols_data = {}
    for symbol in request.symbols:
        bars = data_provider.get_bars(symbol, "5min", days_back=request.days_back)
        if not bars:
            continue
        import pandas as pd
        df = pd.DataFrame(bars)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        symbols_data[symbol] = df

        # Discover per-symbol
        motifs = await asyncio.to_thread(
            discoverer.discover, df, request.window_sizes, symbol
        )
        all_motifs.extend(motifs)

    # Cross-symbol discovery
    if len(symbols_data) >= 2:
        cross = await asyncio.to_thread(
            discoverer.find_cross_symbol_motifs, symbols_data
        )
        all_motifs.extend(cross)

    # Cluster and evaluate
    if all_motifs:
        all_motifs = discoverer.cluster_motifs(all_motifs)
        # Evaluate against the first symbol's data
        first_df = next(iter(symbols_data.values()))
        all_motifs = await asyncio.to_thread(
            discoverer.evaluate_motifs, all_motifs, first_df
        )

    discoverer.motif_library = all_motifs
    path = discoverer.save_motif_library()

    return {
        "status": "complete",
        "motifs_discovered": len(all_motifs),
        "symbols_processed": len(symbols_data),
        "library_path": path,
        "summary": discoverer.get_library_summary(),
    }


@app.get("/api/discovery/motifs")
async def get_motif_library():
    """Get current motif library."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.ml.discovery.motifs import MotifDiscoverer

    discoverer = MotifDiscoverer()
    loaded = discoverer.load_motif_library()
    if not loaded:
        return {"total_motifs": 0, "motifs": [], "loaded": False}

    summary = discoverer.get_library_summary()
    summary["loaded"] = True
    return summary


@app.post("/api/discovery/evolve")
async def run_rule_evolution(request: DiscoveryEvolveRequest):
    """Run DEAP rule evolution using discovered features."""
    import asyncio
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.ml.discovery.evolver import StrategyEvolver
    from agent.ml.discovery.features import AutoFeatureExtractor

    data_provider = _get_data_provider_instance()
    extractor = AutoFeatureExtractor()
    evolver = StrategyEvolver(
        population_size=request.population,
        generations=request.generations,
    )
    evolver.load_evolved_rules()

    all_features = []
    all_returns = []
    all_close = []

    for symbol in request.symbols:
        bars = data_provider.get_bars(symbol, "5min", days_back=request.days_back)
        if not bars:
            continue
        import pandas as pd
        df = pd.DataFrame(bars)
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        features = extractor.extract_catch22(df, window_size=request.window_size)
        if features.empty:
            continue

        # Forward returns (10-bar)
        close = df["close"].iloc[request.window_size:]
        close = close.iloc[:len(features)]
        fwd_ret = close.pct_change(10).shift(-10)

        # Align
        min_len = min(len(features), len(fwd_ret))
        all_features.append(features.iloc[:min_len].reset_index(drop=True))
        all_returns.append(fwd_ret.iloc[:min_len].reset_index(drop=True))
        all_close.append(close.iloc[:min_len].reset_index(drop=True))

    if not all_features:
        raise HTTPException(status_code=422, detail="No data available for evolution")

    combined_features = pd.concat(all_features, ignore_index=True)
    combined_returns = pd.concat(all_returns, ignore_index=True).fillna(0)
    combined_close = pd.concat(all_close, ignore_index=True)

    new_rules = await asyncio.to_thread(
        evolver.evolve, combined_features, combined_returns, combined_close
    )
    evolver.evolved_rules.extend(new_rules)
    path = evolver.save_evolved_rules()

    return {
        "status": "complete",
        "new_rules": len(new_rules),
        "total_rules": len(evolver.evolved_rules),
        "library_path": path,
        "rules": [r.to_dict() for r in new_rules],
    }


@app.get("/api/discovery/rules")
async def get_evolved_rules():
    """Get evolved rules with performance metrics."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.ml.discovery.evolver import StrategyEvolver

    evolver = StrategyEvolver()
    loaded = evolver.load_evolved_rules()
    if not loaded:
        return {"total_rules": 0, "active_rules": 0, "rules": [], "loaded": False}

    summary = evolver.get_rules_summary()
    summary["loaded"] = True
    return summary


@app.post("/api/discovery/backtest")
async def run_discovery_backtest(request: DiscoveryBacktestRequest):
    """Backtest all discovered rules."""
    import asyncio
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from agent.backtest.runner import run_discovered_backtest

    config = {
        "strategy": "",  # ignored — all discovered rules tested
        "symbols": request.symbols,
        "start_date": request.start_date,
        "end_date": request.end_date,
        "initial_capital": request.initial_capital,
        "timeframe": "5min",
        "agent_type": "day",
    }

    try:
        results = await asyncio.to_thread(run_discovered_backtest, config)
        return {"status": "complete", "results": results, "count": len(results)}
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Discovery backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


def _get_data_provider_instance():
    """Get or create a DataProvider for discovery endpoints."""
    from data_provider import DataProvider
    from data_fetcher import MassiveDataFetcher
    from data_cache import DataCache
    return DataProvider(fetcher=MassiveDataFetcher(), cache=DataCache())


# ============================================================================
# Price Encoding Endpoints
# ============================================================================

class SimilarityRequest(BaseModel):
    symbol_a: str
    symbol_b: str
    bars: Optional[int] = 100

@app.get("/api/encoding/{symbol}")
async def get_encoding(symbol: str, bars: int = 50):
    """Get current encoded sequence and token breakdown for a symbol."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
    from ml.encoding import PriceEncoder
    from ml.encoding.patterns import find_patterns

    sym_upper = symbol.upper()
    sym_data = stock_data.get(sym_upper)
    if not sym_data:
        raise HTTPException(status_code=404, detail=f"No data for {sym_upper}")

    df = _stock_data_to_df(sym_data, bars)
    if df is None or len(df) < 20:
        raise HTTPException(status_code=422, detail=f"Insufficient data for {sym_upper}")

    pe = PriceEncoder(period=20)
    encoded = pe.encode_bars(df)
    enc_df = pe.encode_bars_df(df)
    matches = find_patterns(encoded)

    # Token breakdown for last 10 bars
    token_breakdown = []
    for _, row in enc_df.tail(10).iterrows():
        parsed = PriceEncoder.parse_bar_token(row["encoded_str"])
        token_breakdown.append({
            "token": row["encoded_str"],
            "parsed": parsed,
        })

    return {
        "symbol": sym_upper,
        "encoded": encoded,
        "bar_count": len(df),
        "last_10_tokens": token_breakdown,
        "patterns": [
            {"name": m.name, "type": m.pattern_type, "bars": m.bars_matched,
             "description": m.description}
            for m in matches
        ],
    }


@app.get("/api/encoding/{symbol}/history")
async def get_encoding_history(symbol: str, window: int = 20, bars: int = 100):
    """Get historical encoded windows for a symbol."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
    from ml.encoding import PriceEncoder

    sym_upper = symbol.upper()
    sym_data = stock_data.get(sym_upper)
    if not sym_data:
        raise HTTPException(status_code=404, detail=f"No data for {sym_upper}")

    df = _stock_data_to_df(sym_data, bars)
    if df is None or len(df) < window:
        raise HTTPException(status_code=422, detail=f"Insufficient data for {sym_upper}")

    pe = PriceEncoder(period=20)
    windows = pe.encode_rolling_window(df, window=window)

    return {
        "symbol": sym_upper,
        "window_size": window,
        "total_windows": len(windows),
        "windows": windows[-20:],  # Last 20 windows
    }


@app.post("/api/encoding/similarity")
async def compare_encoding(req: SimilarityRequest):
    """Compare encoded sequences of two symbols."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
    from ml.encoding import PriceEncoder

    pe = PriceEncoder(period=20)
    results = {}
    for sym in [req.symbol_a, req.symbol_b]:
        sym_upper = sym.upper()
        sym_data = stock_data.get(sym_upper)
        if not sym_data:
            raise HTTPException(status_code=404, detail=f"No data for {sym_upper}")
        df = _stock_data_to_df(sym_data, req.bars)
        if df is None or len(df) < 20:
            raise HTTPException(status_code=422, detail=f"Insufficient data for {sym_upper}")
        results[sym_upper] = pe.encode_bars(df)

    syms = list(results.keys())
    sim = PriceEncoder.similarity(results[syms[0]], results[syms[1]])

    return {
        "symbol_a": syms[0],
        "symbol_b": syms[1],
        "similarity": round(sim, 4),
        "bars": req.bars,
    }


@app.get("/api/encoding/patterns")
async def list_encoding_patterns():
    """List known encoding patterns and current matches across all symbols."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agent"))
    from ml.encoding import PriceEncoder
    from ml.encoding.patterns import get_pattern_catalog, find_patterns

    catalog = get_pattern_catalog()
    pe = PriceEncoder(period=20)

    # Check current matches across loaded symbols
    current_matches = {}
    for sym, sym_data in stock_data.items():
        df = _stock_data_to_df(sym_data, 50)
        if df is None or len(df) < 20:
            continue
        encoded = pe.encode_bars(df)
        matches = find_patterns(encoded)
        if matches:
            current_matches[sym] = [
                {"name": m.name, "type": m.pattern_type}
                for m in matches[-3:]
            ]

    return {
        "catalog": catalog,
        "current_matches": current_matches,
    }


def _stock_data_to_df(sym_data: dict, bars: int):
    """Convert stock_data dict entry to a pandas DataFrame."""
    import pandas as pd
    bar_list = sym_data.get("bars", [])
    if not bar_list:
        return None
    df = pd.DataFrame(bar_list[-bars:])
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    else:
        df["volume"] = 0.0
    return df


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info"
    )
