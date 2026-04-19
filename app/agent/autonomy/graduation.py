"""Graduation logic: approval → semi-autonomous → autonomous."""

import logging
from typing import Dict, Optional

from ..core.state import AutonomyMode
from .performance import PerformanceTracker

logger = logging.getLogger(__name__)

# Promotion criteria
PROMO_MIN_TRADES = 50
PROMO_MIN_WIN_RATE = 0.55
PROMO_MIN_SHARPE = 1.0
PROMO_MAX_DRAWDOWN = 10.0  # percentage

# Demotion criteria
DEMOTE_CONSECUTIVE_LOSSES = 5
DEMOTE_DRAWDOWN_MULTIPLIER = 1.5  # 1.5x the max allowed drawdown


class GraduationManager:
    """
    Manages transitions between autonomy modes based on performance.

    Modes:
      - approval: All signals require manual approval
      - semi_autonomous: High-confidence (>=0.8) signals from proven skills auto-execute
      - autonomous: All signals auto-execute (requires strong track record)
    """

    def __init__(self, performance: PerformanceTracker):
        self.performance = performance
        self._consecutive_losses = 0

    def evaluate(self, current_mode: AutonomyMode) -> Optional[AutonomyMode]:
        """
        Check if a mode change is warranted.

        Returns the new mode if a change should happen, or None to stay.
        """
        metrics = self.performance.get_metrics()
        total_trades = metrics.get("total_trades", 0)

        # Check demotion first (safety)
        demotion = self._check_demotion(current_mode, metrics)
        if demotion is not None:
            return demotion

        # Check promotion
        promotion = self._check_promotion(current_mode, metrics, total_trades)
        if promotion is not None:
            return promotion

        return None

    def record_trade_outcome(self, profitable: bool):
        """Track consecutive losses for demotion."""
        if profitable:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def get_graduation_status(self) -> Dict:
        """Get current graduation status and progress toward next level."""
        metrics = self.performance.get_metrics()
        total = metrics.get("total_trades", 0)

        return {
            "total_trades": total,
            "trades_needed": max(0, PROMO_MIN_TRADES - total),
            "win_rate": metrics.get("win_rate", 0),
            "win_rate_target": PROMO_MIN_WIN_RATE,
            "sharpe": metrics.get("sharpe_ratio", 0),
            "sharpe_target": PROMO_MIN_SHARPE,
            "max_drawdown": metrics.get("max_drawdown_pct", 0),
            "drawdown_limit": PROMO_MAX_DRAWDOWN,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_loss_limit": DEMOTE_CONSECUTIVE_LOSSES,
            "ready_for_promotion": self._meets_promotion_criteria(metrics, total),
        }

    def _check_promotion(
        self,
        current_mode: AutonomyMode,
        metrics: Dict,
        total_trades: int,
    ) -> Optional[AutonomyMode]:
        """Check if agent should be promoted."""
        if not self._meets_promotion_criteria(metrics, total_trades):
            return None

        if current_mode == AutonomyMode.APPROVAL:
            logger.info("Promotion: approval → semi_autonomous")
            return AutonomyMode.SEMI_AUTONOMOUS

        if current_mode == AutonomyMode.SEMI_AUTONOMOUS:
            # Need even stronger criteria for full autonomous
            sharpe = metrics.get("sharpe_ratio", 0)
            if sharpe >= PROMO_MIN_SHARPE * 1.5 and total_trades >= PROMO_MIN_TRADES * 2:
                logger.info("Promotion: semi_autonomous → autonomous")
                return AutonomyMode.AUTONOMOUS

        return None

    def _check_demotion(
        self,
        current_mode: AutonomyMode,
        metrics: Dict,
    ) -> Optional[AutonomyMode]:
        """Check if agent should be demoted."""
        if current_mode == AutonomyMode.APPROVAL:
            return None  # Can't demote further

        max_dd = metrics.get("max_drawdown_pct", 0)
        dd_limit = PROMO_MAX_DRAWDOWN * DEMOTE_DRAWDOWN_MULTIPLIER

        should_demote = (
            self._consecutive_losses >= DEMOTE_CONSECUTIVE_LOSSES
            or max_dd >= dd_limit
        )

        if not should_demote:
            return None

        if current_mode == AutonomyMode.AUTONOMOUS:
            reason = (
                f"consecutive_losses={self._consecutive_losses}"
                if self._consecutive_losses >= DEMOTE_CONSECUTIVE_LOSSES
                else f"drawdown={max_dd:.1f}%"
            )
            logger.warning(f"Demotion: autonomous → semi_autonomous ({reason})")
            return AutonomyMode.SEMI_AUTONOMOUS

        if current_mode == AutonomyMode.SEMI_AUTONOMOUS:
            reason = (
                f"consecutive_losses={self._consecutive_losses}"
                if self._consecutive_losses >= DEMOTE_CONSECUTIVE_LOSSES
                else f"drawdown={max_dd:.1f}%"
            )
            logger.warning(f"Demotion: semi_autonomous → approval ({reason})")
            return AutonomyMode.APPROVAL

        return None

    def _meets_promotion_criteria(self, metrics: Dict, total_trades: int) -> bool:
        """Check if all promotion criteria are met."""
        return (
            total_trades >= PROMO_MIN_TRADES
            and metrics.get("win_rate", 0) >= PROMO_MIN_WIN_RATE
            and metrics.get("sharpe_ratio", 0) >= PROMO_MIN_SHARPE
            and metrics.get("max_drawdown_pct", 0) <= PROMO_MAX_DRAWDOWN
        )
