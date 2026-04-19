"""Evaluate trading skills against market data via LLM."""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..core.reasoning import ReasoningClient
from ..core.signal import SignalDirection, TradeSignal
from .loader import Skill

logger = logging.getLogger(__name__)


class SkillEvaluator:
    """Evaluates skills by sending them + market data to Claude."""

    def __init__(self, reasoning: ReasoningClient):
        self.reasoning = reasoning

    def evaluate(
        self,
        skill: Skill,
        symbol: str,
        market_data: Dict[str, Any],
    ) -> Optional[TradeSignal]:
        """
        Evaluate a single skill for a single symbol.

        Returns a TradeSignal if conditions are met with sufficient confidence,
        or None if conditions aren't met.
        """
        result = self.reasoning.evaluate_skill(
            skill_markdown=skill.body,
            market_data=market_data,
            symbol=symbol,
        )

        if result is None:
            return None

        conditions_met = result.get("conditions_met", False)
        confidence = result.get("confidence", 0.0)

        if not conditions_met or confidence < 0.3:
            logger.debug(
                f"Skill '{skill.name}' for {symbol}: conditions not met "
                f"(confidence={confidence})"
            )
            return None

        try:
            direction = SignalDirection(result.get("direction", "long"))
        except ValueError:
            direction = SignalDirection.LONG

        signal = TradeSignal(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            entry_price=float(result.get("entry_price", 0)),
            stop_loss=float(result.get("stop_loss", 0)),
            take_profit=float(result.get("take_profit", 0)),
            reasoning=result.get("reasoning", ""),
            skill_name=skill.name,
            position_size_pct=skill.risk_per_trade,
        )

        logger.info(
            f"Signal generated: {signal.direction.value} {symbol} @ {signal.entry_price} "
            f"(confidence={signal.confidence}, skill={skill.name})"
        )
        return signal

    def evaluate_batch(
        self,
        skills: List[Skill],
        symbols: List[str],
        market_snapshots: Dict[str, Dict[str, Any]],
    ) -> List[TradeSignal]:
        """
        Evaluate multiple skills across multiple symbols (sync fallback).

        Args:
            skills: List of enabled skills to evaluate
            symbols: List of symbols to check
            market_snapshots: Dict mapping symbol -> market data snapshot

        Returns:
            List of generated trade signals
        """
        signals: List[TradeSignal] = []

        for symbol in symbols:
            snapshot = market_snapshots.get(symbol)
            if not snapshot:
                continue

            for skill in skills:
                signal = self.evaluate(skill, symbol, snapshot)
                if signal:
                    signals.append(signal)

        logger.info(f"Batch evaluation: {len(signals)} signals from {len(skills)} skills x {len(symbols)} symbols")
        return signals

    async def aevaluate_batch(
        self,
        skills: List[Skill],
        symbols: List[str],
        market_snapshots: Dict[str, Dict[str, Any]],
    ) -> List[TradeSignal]:
        """
        Async batch evaluation — runs all LLM calls concurrently via threads
        so the event loop stays free for HTTP/WebSocket traffic.
        """
        tasks = []
        task_meta = []  # track (skill, symbol) for each task

        for symbol in symbols:
            snapshot = market_snapshots.get(symbol)
            if not snapshot:
                continue
            for skill in skills:
                tasks.append(
                    self.reasoning.aevaluate_skill(
                        skill_markdown=skill.body,
                        market_data=snapshot,
                        symbol=symbol,
                    )
                )
                task_meta.append((skill, symbol))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        signals: List[TradeSignal] = []
        for (skill, symbol), result in zip(task_meta, results):
            if isinstance(result, Exception):
                logger.error(f"Async eval error for {skill.name}/{symbol}: {result}")
                continue
            if result is None:
                continue

            conditions_met = result.get("conditions_met", False)
            confidence = result.get("confidence", 0.0)

            if not conditions_met or confidence < 0.3:
                continue

            try:
                direction = SignalDirection(result.get("direction", "long"))
            except ValueError:
                direction = SignalDirection.LONG

            signal = TradeSignal(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry_price=float(result.get("entry_price", 0)),
                stop_loss=float(result.get("stop_loss", 0)),
                take_profit=float(result.get("take_profit", 0)),
                reasoning=result.get("reasoning", ""),
                skill_name=skill.name,
                position_size_pct=skill.risk_per_trade,
            )
            signals.append(signal)

        logger.info(
            f"Async batch evaluation: {len(signals)} signals from "
            f"{len(skills)} skills x {len(symbols)} symbols"
        )
        return signals
