#!/usr/bin/env python
"""Entry point for the trading agent."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.trading_agent import TradingAgent, AgentConfig


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("agent/logs/agent.log", mode="a"),
        ],
    )

    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the AlphaPy trading agent"
    )
    parser.add_argument(
        "--config",
        "-c",
        default="agent/config/agent.yml",
        help="Path to agent configuration file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single cycle and exit",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run in interactive chat mode",
    )

    args = parser.parse_args()

    # Ensure logs directory exists
    Path("agent/logs").mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from {config_path}")
        config = AgentConfig.from_yaml(str(config_path))
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        config = AgentConfig()

    # Create agent
    agent = TradingAgent(config=config, verbose=args.verbose)

    if args.chat:
        # Interactive chat mode
        asyncio.run(interactive_mode(agent))
    elif args.once:
        # Single cycle mode
        logger.info("Running single cycle...")
        result = asyncio.run(agent.run_cycle())
        print(result)
    else:
        # Continuous trading mode
        logger.info("Starting continuous trading mode...")
        try:
            asyncio.run(agent.run())
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            agent.stop()


async def interactive_mode(agent: TradingAgent) -> None:
    """Run interactive chat mode."""
    print("\nTrading Agent Interactive Mode")
    print("Type 'quit' or 'exit' to stop")
    print("Type 'cycle' to run a trading cycle")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if user_input.lower() == "cycle":
                print("\nRunning trading cycle...")
                result = await agent.run_cycle()
                print(f"\nAgent: {result}")
                continue

            # Regular chat
            response = await agent.chat(user_input)
            print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
