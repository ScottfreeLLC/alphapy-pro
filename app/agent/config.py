"""Agent configuration for Alfi."""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AgentConfig:
    """Configuration for the Alfi trading agent."""

    # Agent identity
    agent_type: str = "swing"  # "swing" | "day"
    agent_name: str = "swing"

    # Timeframe
    primary_timeframe: str = "1d"     # "1d" for swing, "5min" for day
    lookback_bars: int = 100          # 100 daily bars or 78 intraday bars (1 trading day)
    min_holding_periods: int = 2      # Minimum bars to hold
    max_holding_periods: int = 20     # Maximum bars to hold

    # LLM (provider auto-detected by LiteLLM from model name)
    eval_model: str = field(
        default_factory=lambda: os.getenv("LLM_EVAL_MODEL", "gpt-4.1")
    )
    synthesis_model: str = field(
        default_factory=lambda: os.getenv("LLM_SYNTHESIS_MODEL", "gpt-4.1")
    )

    # Cycle
    cycle_interval_seconds: int = 60
    max_signals_per_cycle: int = 20

    # Skills
    skills_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "skills", "swing"
        )
    )

    # Risk defaults
    max_position_pct: float = 0.10        # 10% max per position
    max_total_exposure_pct: float = 0.60  # 60% total exposure
    max_daily_loss_pct: float = 0.02      # 2% daily loss limit
    max_positions: int = 10
    min_risk_reward: float = 1.5

    # Autonomy
    autonomy_mode: str = "approval"  # approval | semi_autonomous | autonomous

    # Watchlist
    default_watchlist: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "AMD", "AVGO", "NFLX",
    ])
    crypto_watchlist: List[str] = field(default_factory=lambda: [
        "X:BTCUSD", "X:ETHUSD", "X:SOLUSD",
    ])

    # Backend
    backend_url: str = "http://localhost:8080"

    # Alpaca (populated from env)
    alpaca_api_key: str = field(
        default_factory=lambda: os.getenv("ALPACA_API_KEY", "")
    )
    alpaca_secret_key: str = field(
        default_factory=lambda: os.getenv("ALPACA_SECRET_KEY", "")
    )
    alpaca_paper: bool = True

    # Data
    data_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data"
        )
    )

    @classmethod
    def swing_config(cls) -> "AgentConfig":
        """Factory for Swing Agent configuration."""
        return cls(
            agent_type="swing",
            agent_name="swing",
            primary_timeframe="1d",
            lookback_bars=100,
            min_holding_periods=2,
            max_holding_periods=20,
            cycle_interval_seconds=60,
            max_signals_per_cycle=20,
            max_positions=10,
            max_total_exposure_pct=0.60,
            skills_dir=os.path.join(
                os.path.dirname(__file__), "skills", "specs", "swing"
            ),
        )

    @classmethod
    def day_config(cls) -> "AgentConfig":
        """Factory for Day Agent configuration."""
        return cls(
            agent_type="day",
            agent_name="day",
            primary_timeframe="5min",
            lookback_bars=78,          # 1 trading day of 5-min bars
            min_holding_periods=2,     # 10 minutes minimum
            max_holding_periods=78,    # 1 trading day maximum
            cycle_interval_seconds=30,
            max_signals_per_cycle=10,
            max_positions=5,
            max_total_exposure_pct=0.40,
            max_daily_loss_pct=0.01,   # Tighter daily loss for day trading
            min_risk_reward=1.2,       # Lower R:R threshold for intraday
            skills_dir=os.path.join(
                os.path.dirname(__file__), "skills", "specs", "day"
            ),
        )
