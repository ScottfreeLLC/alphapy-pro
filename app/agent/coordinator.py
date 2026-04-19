"""
AgentCoordinator: top-level orchestrator for dual-agent trading.

Manages two AlfiEngine instances (swing + day) sharing common infrastructure:
- DataProvider (daily + 5min bars)
- SharedRiskManager (cross-agent limits)
- PerformanceTracker (per-agent DBs)
"""

import asyncio
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from .config import AgentConfig
from .core.engine import AlfiEngine
from .risk.shared import SharedRiskManager

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """
    Orchestrates two AlfiEngine instances sharing common infrastructure.

    The backend instantiates this instead of a bare engine. Each engine
    gets its own config, skills directory, and cycle cadence, but they
    share a DataProvider and SharedRiskManager.
    """

    def __init__(
        self,
        swing_config: AgentConfig,
        day_config: AgentConfig,
        data_provider: Optional[Any] = None,
        shared_risk: Optional[SharedRiskManager] = None,
        on_state_change: Optional[Callable[[Dict], None]] = None,
        # Per-agent subsystems (created externally for flexibility)
        swing_risk_manager: Optional[Any] = None,
        day_risk_manager: Optional[Any] = None,
        swing_performance: Optional[Any] = None,
        day_performance: Optional[Any] = None,
        swing_confidence: Optional[Any] = None,
        day_confidence: Optional[Any] = None,
        swing_graduation: Optional[Any] = None,
        day_graduation: Optional[Any] = None,
        day_pattern_classifier: Optional[Any] = None,
        portfolio_optimizer: Optional[Any] = None,
        # Shared broker infrastructure
        broker: Optional[Any] = None,
        position_tracker: Optional[Any] = None,
        position_monitor: Optional[Any] = None,
    ):
        self.shared_risk = shared_risk or SharedRiskManager()
        self.portfolio_optimizer = portfolio_optimizer
        self.data_provider = data_provider
        self.position_tracker = position_tracker
        self.position_monitor = position_monitor
        self._on_state_change = on_state_change

        # Shared price encoder for both agents
        from .ml.encoding import PriceEncoder
        self.price_encoder = PriceEncoder(period=20)

        # Create engines (both share the same broker / Alpaca account)
        self.swing = AlfiEngine(
            config=swing_config,
            data_provider=data_provider,
            broker=broker,
            risk_manager=swing_risk_manager,
            on_state_change=lambda state: self._on_engine_state_change("swing", state),
            performance_tracker=swing_performance,
            confidence_scorer=swing_confidence,
            graduation_manager=swing_graduation,
            price_encoder=self.price_encoder,
        )
        self.day = AlfiEngine(
            config=day_config,
            data_provider=data_provider,
            broker=broker,
            risk_manager=day_risk_manager,
            on_state_change=lambda state: self._on_engine_state_change("day", state),
            performance_tracker=day_performance,
            confidence_scorer=day_confidence,
            graduation_manager=day_graduation,
            pattern_classifier=day_pattern_classifier,
            price_encoder=self.price_encoder,
        )

        self._engines = {"swing": self.swing, "day": self.day}

    def get_engine(self, agent_type: str) -> AlfiEngine:
        """Get engine by agent type."""
        engine = self._engines.get(agent_type)
        if not engine:
            raise ValueError(f"Unknown agent type: {agent_type}. Must be 'swing' or 'day'.")
        return engine

    async def start(self, agent_type: str):
        """Start a specific agent."""
        engine = self.get_engine(agent_type)
        await engine.start()
        logger.info(f"{agent_type.title()} agent started")

    async def stop(self, agent_type: str):
        """Stop a specific agent."""
        engine = self.get_engine(agent_type)
        await engine.stop()
        logger.info(f"{agent_type.title()} agent stopped")

    async def start_all(self):
        """Start both agents."""
        await asyncio.gather(
            self.start("swing"),
            self.start("day"),
        )

    async def stop_all(self):
        """Stop both agents."""
        await asyncio.gather(
            self.stop("swing"),
            self.stop("day"),
        )

    def get_combined_state(self) -> Dict:
        """Get combined state from both agents plus shared risk."""
        state = {
            "swing": self.swing.state.to_dict(),
            "day": self.day.state.to_dict(),
            "shared_risk": self.shared_risk.get_status(),
        }
        if self.position_monitor:
            state["broker"] = self.position_monitor.get_status()
        return state

    def get_all_pending_signals(self) -> List[Dict]:
        """Get pending signals from both agents."""
        signals = []
        for agent_type, engine in self._engines.items():
            for s in engine.state.pending_signals:
                d = s.to_dict()
                d["agent_type"] = agent_type
                signals.append(d)
        return signals

    def get_all_recent_signals(self) -> List[Dict]:
        """Get recent signals from both agents, sorted by time."""
        signals = []
        for agent_type, engine in self._engines.items():
            for s in engine.state.recent_signals:
                d = s.to_dict()
                d["agent_type"] = agent_type
                signals.append(d)
        # Sort by created_at descending
        signals.sort(key=lambda s: s.get("created_at", ""), reverse=True)
        return signals

    def _on_engine_state_change(self, agent_type: str, state_dict: Dict):
        """Handle state change from an individual engine."""
        if self._on_state_change:
            combined = self.get_combined_state()
            result = self._on_state_change(combined)
            if asyncio.iscoroutine(result):
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    pass
