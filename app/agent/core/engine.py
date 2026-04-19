"""Main reasoning loop for the Alfi trading agent."""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from ..config import AgentConfig
from ..skills.evaluator import SkillEvaluator
from ..skills.registry import SkillRegistry
from .reasoning import ReasoningClient
from .signal import SignalAggregator, SignalStatus, TradeSignal
from .state import AgentState, AgentStatus, AgentType, AutonomyMode, CycleResult

logger = logging.getLogger(__name__)


class AlfiEngine:
    """
    Alfi's main reasoning loop.

    Each cycle:
      1. Gather market data
      2. Evaluate enabled skills against watched symbols
      3. Aggregate signals (dedup, rank, resolve conflicts)
      4. Apply risk filters
      5. Route signals (approve queue vs. auto-execute)
      6. Track results
    """

    def __init__(
        self,
        config: AgentConfig,
        data_provider: Optional[Any] = None,
        broker: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        on_signal: Optional[Callable[[TradeSignal], None]] = None,
        on_state_change: Optional[Callable[[Dict], None]] = None,
        performance_tracker: Optional[Any] = None,
        confidence_scorer: Optional[Any] = None,
        graduation_manager: Optional[Any] = None,
        pattern_classifier: Optional[Any] = None,
        price_encoder: Optional[Any] = None,
    ):
        self.config = config
        self.data_provider = data_provider
        self.broker = broker
        self.risk_manager = risk_manager
        self.on_signal = on_signal
        self.on_state_change = on_state_change
        self.performance_tracker = performance_tracker
        self.confidence_scorer = confidence_scorer
        self.graduation_manager = graduation_manager
        self.pattern_classifier = pattern_classifier
        self.price_encoder = price_encoder

        # Core components
        self.reasoning = ReasoningClient(config)
        self.registry = SkillRegistry(config.skills_dir)
        self.evaluator = SkillEvaluator(self.reasoning)
        self.aggregator = SignalAggregator()
        self.state = AgentState(
            agent_type=AgentType(config.agent_type),
        )

        # Control
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the agent loop."""
        if self._running:
            logger.warning("Engine already running")
            return

        self._running = True
        self.state.status = AgentStatus.STARTING
        self.state.started_at = datetime.now()
        self.state.log_activity("engine_start", "Alfi engine starting")

        # Load skills
        self.registry.load()
        skills = self.registry.get_enabled()
        self.state.log_activity(
            "skills_loaded", f"Loaded {len(skills)} enabled skills"
        )

        self.state.status = AgentStatus.RUNNING
        self._notify_state_change()

        # Start the cycle loop
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        """Stop the agent loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

        self.state.status = AgentStatus.STOPPED
        self.state.log_activity("engine_stop", "Alfi engine stopped")
        self._notify_state_change()

    async def _run_loop(self):
        """Main cycle loop."""
        while self._running:
            try:
                result = await self.run_cycle()
                self.state.consecutive_errors = 0

                if result.errors:
                    for err in result.errors:
                        self.state.log_activity("cycle_error", err, level="warning")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.state.consecutive_errors += 1
                self.state.last_error = str(e)
                self.state.log_activity("cycle_exception", str(e), level="error")
                logger.exception(f"Engine cycle error: {e}")

                if self.state.consecutive_errors >= 5:
                    self.state.status = AgentStatus.ERROR
                    self.state.log_activity(
                        "engine_error",
                        "Too many consecutive errors, pausing",
                        level="error",
                    )
                    self._running = False
                    break

            # Wait for next cycle
            await asyncio.sleep(self.config.cycle_interval_seconds)

    async def run_cycle(self) -> CycleResult:
        """Execute a single reasoning cycle."""
        cycle_start = time.time()
        self.state.cycle_count += 1
        self.state.last_cycle_at = datetime.now()
        cycle_num = self.state.cycle_count
        errors: List[str] = []

        self.state.log_activity("cycle_start", f"Cycle #{cycle_num}")

        # 1. Check for skill file changes
        self.registry.reload_if_changed()

        # 2. Gather market data
        market_snapshots = await self._gather_market_data()
        if not market_snapshots:
            errors.append("No market data available")
            return CycleResult(
                cycle_number=cycle_num,
                timestamp=datetime.now(),
                signals_generated=0,
                signals_executed=0,
                signals_pending_approval=0,
                errors=errors,
                duration_ms=(time.time() - cycle_start) * 1000,
            )

        # 3. Evaluate skills
        enabled_skills = self.registry.get_enabled()
        watchlist = self._get_watchlist()

        signals = await self.evaluator.aevaluate_batch(
            skills=enabled_skills,
            symbols=watchlist,
            market_snapshots=market_snapshots,
        )

        # 4. Aggregate signals
        signals = self.aggregator.aggregate(signals)

        # Limit signals per cycle
        signals = signals[: self.config.max_signals_per_cycle]

        # 5. Apply confidence scoring (composite: LLM + skill accuracy + R:R)
        if self.confidence_scorer:
            for signal in signals:
                signal.metadata["raw_llm_confidence"] = signal.confidence
                signal.confidence = self.confidence_scorer.score(signal)

        # 6. Risk filter
        signals = self._apply_risk_filter(signals)

        # 7. Route signals
        executed = 0
        pending = 0
        for signal in signals:
            routed = self._route_signal(signal)
            if routed == "executed":
                executed += 1
            elif routed == "pending":
                pending += 1

        self.state.recent_signals = (signals + self.state.recent_signals)[:50]

        # 8. Evaluate graduation (autonomy mode transitions)
        if self.graduation_manager:
            new_mode = self.graduation_manager.evaluate(self.state.autonomy_mode)
            if new_mode is not None:
                self.state.autonomy_mode = new_mode
                self.state.log_activity(
                    "graduation",
                    f"Autonomy mode changed to {new_mode.value}",
                )

        duration_ms = (time.time() - cycle_start) * 1000
        self.state.log_activity(
            "cycle_complete",
            f"Cycle #{cycle_num}: {len(signals)} signals, {executed} executed, {pending} pending ({duration_ms:.0f}ms)",
        )
        self._notify_state_change()

        return CycleResult(
            cycle_number=cycle_num,
            timestamp=datetime.now(),
            signals_generated=len(signals),
            signals_executed=executed,
            signals_pending_approval=pending,
            errors=errors,
            duration_ms=duration_ms,
        )

    async def _gather_market_data(self) -> Dict[str, Dict]:
        """Gather market snapshots for all watched symbols."""
        if self.data_provider is None:
            self.state.log_activity(
                "data_stub",
                "No data provider configured — skipping data gathering",
                level="warning",
            )
            return {}

        try:
            snapshots = await self.data_provider.get_current_snapshots(
                self._get_watchlist(),
                timeframe=self.config.primary_timeframe,
                lookback_bars=self.config.lookback_bars,
            )
        except Exception as e:
            logger.error(f"Failed to gather market data: {e}")
            return {}

        # Enrich intraday snapshots with pattern classification
        if self.pattern_classifier and self.pattern_classifier.is_loaded:
            snapshots = self._enrich_with_patterns(snapshots)

        # Enrich with price encoding
        if self.price_encoder:
            snapshots = self._enrich_with_encoding(snapshots)

        return snapshots

    def _get_watchlist(self) -> List[str]:
        """Get the combined stock + crypto watchlist."""
        return self.config.default_watchlist + self.config.crypto_watchlist

    def _enrich_with_patterns(self, snapshots: Dict[str, Dict]) -> Dict[str, Dict]:
        """Enrich intraday snapshots with pattern classification from ML model."""
        try:
            from ..ml.intraday.features import build_intraday_features
            import pandas as pd

            bars_key = f"bars_{self.config.primary_timeframe}"

            for symbol, snapshot in snapshots.items():
                bars = snapshot.get(bars_key, [])
                if len(bars) < 10:
                    continue

                df = pd.DataFrame(bars)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
                if "vwap" in df.columns:
                    df["vwap"] = pd.to_numeric(df["vwap"], errors="coerce")

                features = build_intraday_features(df)
                if len(features) == 0:
                    continue

                # Predict on the latest bar
                latest_features = features.iloc[-1].to_dict()
                prediction = self.pattern_classifier.predict_single(latest_features)

                snapshot["pattern"] = prediction
                self.state.log_activity(
                    "pattern_detected",
                    f"{symbol}: {prediction['pattern']} ({prediction['probability']:.0%})",
                )

        except Exception as e:
            logger.warning(f"Pattern enrichment failed: {e}")

        return snapshots

    def _enrich_with_encoding(self, snapshots: Dict[str, Dict]) -> Dict[str, Dict]:
        """Enrich snapshots with price encoding tokens and pattern matches."""
        try:
            import pandas as pd
            from ..ml.encoding.patterns import find_patterns

            bars_key = f"bars_{self.config.primary_timeframe}"

            for symbol, snapshot in snapshots.items():
                bars = snapshot.get(bars_key, [])
                if len(bars) < 20:
                    continue

                df = pd.DataFrame(bars)
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

                encoded = self.price_encoder.get_last_n_encoded(df, n=10)
                full_encoded = self.price_encoder.encode_bars(df)
                matches = find_patterns(full_encoded)

                snapshot["encoding"] = {
                    "tokens": encoded,
                    "patterns": [
                        {"name": m.name, "type": m.pattern_type}
                        for m in matches[-5:]  # Last 5 matches
                    ],
                }

        except Exception as e:
            logger.warning(f"Encoding enrichment failed: {e}")

        return snapshots

    def _apply_risk_filter(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Apply risk management filters. Full implementation in Phase 5."""
        if self.risk_manager:
            return self.risk_manager.filter_signals(signals)

        # Basic filter: reject signals with poor risk/reward
        return [
            s for s in signals
            if s.risk_reward_ratio >= self.config.min_risk_reward
        ]

    def _route_signal(self, signal: TradeSignal) -> str:
        """Route a signal based on autonomy mode. Returns 'executed', 'pending', or 'rejected'."""
        mode = self.state.autonomy_mode

        if mode == AutonomyMode.AUTONOMOUS:
            return self._execute_signal(signal)

        if mode == AutonomyMode.SEMI_AUTONOMOUS and signal.confidence >= 0.8:
            return self._execute_signal(signal)

        # Approval mode or low-confidence semi-auto
        signal.status = SignalStatus.PENDING
        self.state.pending_signals.append(signal)
        if self.on_signal:
            self.on_signal(signal)
        return "pending"

    def _execute_signal(self, signal: TradeSignal) -> str:
        """Execute a signal via the broker. Full implementation in Phase 3."""
        if self.broker is None:
            signal.status = SignalStatus.PENDING
            self.state.pending_signals.append(signal)
            self.state.log_activity(
                "no_broker",
                f"No broker configured — signal {signal.id} queued for approval",
                level="warning",
            )
            if self.on_signal:
                self.on_signal(signal)
            return "pending"

        try:
            self.broker.submit_order(signal)
            signal.status = SignalStatus.EXECUTED
            self.state.log_activity(
                "signal_executed",
                f"Executed {signal.direction.value} {signal.symbol} @ {signal.entry_price}",
            )
            return "executed"
        except Exception as e:
            signal.status = SignalStatus.CANCELLED
            self.state.log_activity(
                "execution_failed",
                f"Failed to execute {signal.symbol}: {e}",
                level="error",
            )
            return "rejected"

    def approve_signal(self, signal_id: str) -> Optional[TradeSignal]:
        """Approve a pending signal for execution."""
        for signal in self.state.pending_signals:
            if signal.id == signal_id:
                self.state.pending_signals.remove(signal)
                signal.status = SignalStatus.APPROVED
                self._execute_signal(signal)
                self._notify_state_change()
                return signal
        return None

    def reject_signal(self, signal_id: str) -> Optional[TradeSignal]:
        """Reject a pending signal."""
        for signal in self.state.pending_signals:
            if signal.id == signal_id:
                self.state.pending_signals.remove(signal)
                signal.status = SignalStatus.REJECTED
                self.state.log_activity(
                    "signal_rejected", f"Rejected signal {signal_id} ({signal.symbol})"
                )
                self._notify_state_change()
                return signal
        return None

    def set_autonomy_mode(self, mode: str):
        """Change the autonomy mode."""
        try:
            self.state.autonomy_mode = AutonomyMode(mode)
            self.state.log_activity("mode_change", f"Autonomy mode set to {mode}")
            self._notify_state_change()
        except ValueError:
            logger.error(f"Invalid autonomy mode: {mode}")

    def _notify_state_change(self):
        """Notify listeners of state changes."""
        if self.on_state_change:
            import asyncio
            result = self.on_state_change(self.state.to_dict())
            # If the callback is a coroutine, schedule it
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    pass  # No running loop — skip async broadcast
