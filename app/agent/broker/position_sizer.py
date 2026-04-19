"""Position sizing using Half-Kelly criterion with volatility scaling."""

import logging
import math
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Guardrails
MIN_POSITION_PCT = 0.005   # 0.5% minimum
MAX_POSITION_PCT = 0.05    # 5% maximum per position
DEFAULT_POSITION_PCT = 0.02  # 2% default


class PositionSizer:
    """
    Calculates position sizes using Half-Kelly criterion.

    Kelly fraction: f* = (p * b - q) / b
    Half-Kelly:     f  = f* / 2

    Where:
      p = win probability (from skill performance or default 0.5)
      q = 1 - p
      b = average win / average loss ratio (reward-to-risk)

    Additional modifiers:
      - Volatility scaling: reduce size when ATR > historical average
      - Confidence multiplier: scale by signal confidence score
      - Portfolio weight: scale by optimizer weight if available
    """

    def __init__(
        self,
        max_position_pct: float = MAX_POSITION_PCT,
        min_position_pct: float = MIN_POSITION_PCT,
        default_position_pct: float = DEFAULT_POSITION_PCT,
        kelly_fraction: float = 0.5,  # Half-Kelly by default
    ):
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.default_position_pct = default_position_pct
        self.kelly_fraction = kelly_fraction

    def calculate_size(
        self,
        signal_confidence: float = 0.5,
        win_rate: float = 0.0,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        risk_reward_ratio: float = 0.0,
        atr_current: float = 0.0,
        atr_average: float = 0.0,
        portfolio_weight: Optional[float] = None,
    ) -> float:
        """
        Calculate position size as a fraction of equity.

        Args:
            signal_confidence: Model/LLM confidence score (0-1).
            win_rate: Historical win rate for this skill (0-1).
            avg_win: Average winning trade size ($).
            avg_loss: Average losing trade size ($).
            risk_reward_ratio: Signal's risk/reward ratio.
            atr_current: Current ATR for the symbol.
            atr_average: Historical average ATR for the symbol.
            portfolio_weight: Portfolio optimizer weight (0-1) if available.

        Returns:
            Position size as a fraction of equity (e.g. 0.02 = 2%).
        """
        # Start with Kelly sizing if we have enough data
        kelly_pct = self._kelly_size(win_rate, avg_win, avg_loss, risk_reward_ratio)

        if kelly_pct > 0:
            base_pct = kelly_pct
        else:
            base_pct = self.default_position_pct

        # Confidence multiplier: scale linearly with confidence
        # confidence 0.5 -> 0.5x, confidence 0.8 -> 1.0x, confidence 1.0 -> 1.25x
        confidence_mult = 0.25 + signal_confidence
        base_pct *= confidence_mult

        # Volatility scaling: reduce size when vol is elevated
        vol_mult = self._volatility_scale(atr_current, atr_average)
        base_pct *= vol_mult

        # Portfolio weight overlay
        if portfolio_weight is not None and portfolio_weight > 0:
            # Scale by portfolio weight relative to max weight (15%)
            pw_mult = min(2.0, max(0.25, portfolio_weight / 0.15))
            base_pct *= pw_mult

        # Clamp to guardrails
        final_pct = max(self.min_position_pct, min(self.max_position_pct, base_pct))
        return round(final_pct, 4)

    def calculate_qty(
        self,
        equity: float,
        entry_price: float,
        position_size_pct: float,
    ) -> int:
        """
        Convert a position size percentage to share quantity.

        Args:
            equity: Account equity in dollars.
            entry_price: Expected entry price.
            position_size_pct: Position size as fraction of equity.

        Returns:
            Number of shares (integer, minimum 0).
        """
        if entry_price <= 0 or equity <= 0:
            return 0
        position_value = equity * position_size_pct
        qty = int(position_value / entry_price)
        return max(0, qty)

    def _kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        risk_reward_ratio: float,
    ) -> float:
        """
        Calculate Kelly fraction.

        Uses avg_win/avg_loss if available, otherwise falls back to risk_reward_ratio.
        Returns 0 if insufficient data for Kelly calculation.
        """
        if win_rate <= 0 or win_rate >= 1:
            return 0.0

        p = win_rate
        q = 1 - p

        # Determine payoff ratio (b)
        if avg_win > 0 and avg_loss > 0:
            b = avg_win / avg_loss
        elif risk_reward_ratio > 0:
            b = risk_reward_ratio
        else:
            return 0.0

        # Kelly formula: f* = (p*b - q) / b
        kelly = (p * b - q) / b

        if kelly <= 0:
            return 0.0

        # Apply Kelly fraction (half-Kelly by default)
        return kelly * self.kelly_fraction

    def _volatility_scale(self, atr_current: float, atr_average: float) -> float:
        """
        Scale position size inversely with volatility.

        Returns multiplier: 1.0 at average vol, lower when vol is elevated.
        """
        if atr_current <= 0 or atr_average <= 0:
            return 1.0

        vol_ratio = atr_current / atr_average

        if vol_ratio <= 1.0:
            # Below-average vol: keep full size (slight increase up to 1.2x)
            return min(1.2, 1.0 + (1.0 - vol_ratio) * 0.4)
        else:
            # Above-average vol: reduce size
            # 1.5x vol -> 0.75x size, 2.0x vol -> 0.5x size
            return max(0.3, 1.0 / vol_ratio)

    def get_status(self) -> Dict:
        """Get sizer configuration for API."""
        return {
            "kelly_fraction": self.kelly_fraction,
            "max_position_pct": self.max_position_pct,
            "min_position_pct": self.min_position_pct,
            "default_position_pct": self.default_position_pct,
        }
