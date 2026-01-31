"""Trading Agent - Claude-powered autonomous trading orchestrator."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from anthropic import Anthropic

from .tools.base import Tool
from .tools.market_data import MarketDataTool
from .tools.signal_generator import SignalGeneratorTool
from .tools.portfolio_state import PortfolioStateTool
from .tools.order_execution import OrderExecutionTool
from .tools.risk_checks import RiskCheckTool
from .tools.memory import MemoryTool
from .utils.market_hours import MarketHours

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for the trading agent."""

    # Model settings
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096
    temperature: float = 0.7

    # Trading settings
    symbols_stocks: list[str] = field(default_factory=lambda: ["AAPL", "TSLA", "NVDA"])
    symbols_crypto: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    timeframe: str = "5Min"
    bar_lookback: int = 100

    # Model/ML settings
    run_dir: str = "projects/intraday/runs/latest"
    algo: str = "xgb"
    prob_min: float = 0.55

    # Risk settings
    max_position_value: float = 5000.0
    max_portfolio_exposure: float = 25000.0
    max_positions: int = 5
    daily_loss_limit: float = 0.02

    # Scheduling
    loop_interval_seconds: int = 300  # 5 minutes

    @classmethod
    def from_yaml(cls, path: str) -> "AgentConfig":
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Flatten nested structure
        config = {}

        if "trading" in data:
            trading = data["trading"]
            if "symbols" in trading:
                config["symbols_stocks"] = trading["symbols"].get("stocks", [])
                config["symbols_crypto"] = trading["symbols"].get("crypto", [])
            config["timeframe"] = trading.get("timeframe", "5Min")
            config["bar_lookback"] = trading.get("bar_lookback", 100)

        if "risk" in data:
            risk = data["risk"]
            config["max_position_value"] = risk.get("max_position_value", 5000)
            config["max_portfolio_exposure"] = risk.get("max_portfolio_exposure", 25000)
            config["max_positions"] = risk.get("max_positions", 5)
            config["daily_loss_limit"] = risk.get("daily_loss_limit", 0.02)

        if "model" in data:
            model = data["model"]
            config["run_dir"] = model.get("run_dir", "projects/intraday/runs/latest")
            config["algo"] = model.get("algo", "xgb")
            config["prob_min"] = model.get("prob_min", 0.55)

        if "agent" in data:
            agent = data["agent"]
            config["model"] = agent.get("model", "claude-sonnet-4-20250514")
            config["loop_interval_seconds"] = agent.get("loop_interval_seconds", 300)

        return cls(**config)


class TradingAgent:
    """Claude-powered autonomous trading agent.

    Orchestrates the trading loop:
    1. Fetch market data
    2. Generate signals
    3. Check risk constraints
    4. Execute approved trades
    5. Track state/memory
    """

    SYSTEM_PROMPT = """You are an autonomous trading agent managing a portfolio of stocks and crypto.

Your job is to:
1. Analyze market data and generate trading signals
2. Evaluate risk before any trade
3. Execute trades that pass risk checks
4. Track performance and maintain state

RULES:
- ALWAYS check risk constraints before placing any trade
- NEVER exceed position size or exposure limits
- If daily loss limit is hit, STOP trading for the day
- Log all decisions and their rationale

WORKFLOW for each cycle:
1. Get portfolio state to understand current positions
2. Fetch market data for watchlist symbols
3. Generate signals using the ML model
4. For each actionable signal:
   a. Check risk constraints
   b. If approved, execute the trade
   c. Log the decision
5. Update memory with cycle summary

Be conservative. It's better to miss a trade than to violate risk rules.
"""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        verbose: bool = True,
    ):
        """Initialize the trading agent.

        Args:
            config: Agent configuration
            verbose: Enable verbose logging
        """
        self.config = config or AgentConfig()
        self.verbose = verbose

        # Initialize Anthropic client
        self.client = Anthropic()

        # Initialize tools
        self.tools: list[Tool] = self._initialize_tools()

        # Message history
        self.messages: list[dict] = []

        # Running state
        self._running = False
        self._cycle_count = 0

        logger.info(f"Trading agent initialized with {len(self.tools)} tools")

    def _initialize_tools(self) -> list[Tool]:
        """Initialize all trading tools."""
        # Market data tool (Polygon)
        market_data = MarketDataTool()

        # Signal generator (with model loading)
        signal_generator = SignalGeneratorTool()
        run_dir = Path(self.config.run_dir)
        if run_dir.exists():
            try:
                signal_generator.load_model(str(run_dir), self.config.algo)
            except Exception as e:
                logger.warning(f"Could not load model: {e}")

        # Portfolio state tool (Alpaca)
        portfolio_state = PortfolioStateTool()

        # Order execution tool (Alpaca)
        order_execution = OrderExecutionTool()

        # Risk check tool
        risk_checks = RiskCheckTool()
        risk_checks.configure(
            max_position_value=self.config.max_position_value,
            max_portfolio_exposure=self.config.max_portfolio_exposure,
            max_positions=self.config.max_positions,
            daily_loss_limit=self.config.daily_loss_limit,
        )

        # Memory tool
        memory = MemoryTool()

        return [
            market_data,
            signal_generator,
            portfolio_state,
            order_execution,
            risk_checks,
            memory,
        ]

    def _get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for Claude API."""
        return [tool.to_dict() for tool in self.tools]

    async def _execute_tool(self, name: str, input_data: dict) -> str:
        """Execute a tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return await tool.execute(**input_data)

        return json.dumps({"error": f"Unknown tool: {name}"})

    async def _agent_loop(self, user_input: str) -> str:
        """Run the agent loop for a single user input."""
        # Add user message
        self.messages.append({
            "role": "user",
            "content": user_input,
        })

        while True:
            # Call Claude
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=self.SYSTEM_PROMPT,
                tools=self._get_tool_definitions(),
                messages=self.messages,
            )

            # Process response
            assistant_message = {"role": "assistant", "content": response.content}
            self.messages.append(assistant_message)

            # Check for tool use
            tool_use_blocks = [
                block for block in response.content
                if block.type == "tool_use"
            ]

            if not tool_use_blocks:
                # No more tool calls, extract text response
                text_blocks = [
                    block.text for block in response.content
                    if block.type == "text"
                ]
                return "\n".join(text_blocks)

            # Execute tools
            tool_results = []
            for tool_use in tool_use_blocks:
                if self.verbose:
                    logger.info(f"Tool call: {tool_use.name}")

                result = await self._execute_tool(
                    tool_use.name,
                    tool_use.input,
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            # Add tool results as user message
            self.messages.append({
                "role": "user",
                "content": tool_results,
            })

    async def run_cycle(self) -> str:
        """Run a single trading cycle."""
        self._cycle_count += 1
        timestamp = datetime.now().isoformat()

        # Get all symbols
        symbols = self.config.symbols_stocks + self.config.symbols_crypto

        # Check market hours
        market_status = MarketHours.get_market_status(symbols)

        prompt = f"""Trading cycle #{self._cycle_count} at {timestamp}

Market Status:
{json.dumps(market_status, indent=2)}

Watchlist: {symbols}

Please execute the trading workflow:
1. Get current portfolio state
2. Fetch market data for the watchlist (timeframe: {self.config.timeframe})
3. Generate signals (probability threshold: {self.config.prob_min})
4. For any actionable signals, check risk and execute if approved
5. Log summary to memory

Focus on symbols where the market is currently open.
"""

        try:
            result = await self._agent_loop(prompt)
            logger.info(f"Cycle {self._cycle_count} completed")
            return result

        except Exception as e:
            logger.error(f"Error in cycle {self._cycle_count}: {e}")
            return f"Error: {e}"

    async def run(self) -> None:
        """Run the agent continuously."""
        self._running = True
        logger.info("Starting trading agent...")

        while self._running:
            try:
                # Run a cycle
                result = await self.run_cycle()

                if self.verbose:
                    print(f"\n{'='*50}")
                    print(f"Cycle {self._cycle_count} Result:")
                    print(result)
                    print(f"{'='*50}\n")

                # Wait for next cycle
                await asyncio.sleep(self.config.loop_interval_seconds)

            except KeyboardInterrupt:
                logger.info("Received interrupt, stopping...")
                self._running = False
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying

        logger.info("Trading agent stopped")

    def stop(self) -> None:
        """Stop the agent."""
        self._running = False

    async def chat(self, message: str) -> str:
        """Interactive chat with the agent."""
        return await self._agent_loop(message)
