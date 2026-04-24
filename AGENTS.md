# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## IMPORTANT RULES
- **NO GUESSING**: Do not attempt solutions you're not certain about. If you don't know how to fix something, say so directly instead of wasting time with attempts that might not work.
- Be honest about limitations and knowledge gaps
- **DO NOT suggest `uv run` or virtual environment activation**: After installing with `uv pip install -e ".[dev]"`, commands like `alphapy` and `mflow` work directly. Never tell the user to use `uv run` or `source .venv/bin/activate`.

## Overview

AlphaPy Pro is a machine learning framework designed for speculators and data scientists. It provides a flexible ML pipeline built on scikit-learn and pandas, with specialized pipelines for market analysis (MarketFlow).

## Common Development Commands

### Build and Installation
```bash
# Install package locally (editable mode) - commands work directly after this
pip install -e ".[dev]"

# Build distribution packages
python -m build

# Check package integrity
twine check dist/*
```

### Running AlphaPy
```bash
# Main pipeline
alphapy

# Market flow pipeline
mflow
```

### Documentation
```bash
# Build HTML documentation
cd docs
make html

# Clean documentation build
make clean
```

### Cleanup Utilities
```bash
# Remove old run directories (keeps most recent)
./utils/cleanup_runs.sh
```

## Architecture

### Core Components
- **alphapy/**: Main package containing all modules
  - `alphapy_main.py`: Main pipeline entry point
  - `mflow_main.py`: Market flow pipeline entry point
  - `model.py`: Core model management
  - `data.py`, `frame.py`: Data processing
  - `features.py`, `transforms.py`, `variables.py`: Feature engineering
  - `portfolio.py`, `system.py`: Trading functionality
  - `plots.py`: Visualization utilities

### Configuration System
All configurations use YAML format in the `config/` directory:
- `alphapy.yml`: Main configuration with project paths
- `algos.yml`: ML algorithm definitions and hyperparameters
- `model.yml`: Project-specific model configuration (in each project directory)
- `variables.yml`, `groups.yml`, `sources.yml`, `systems.yml`: Feature and data configurations

### Project Structure
Projects are organized under `projects/` with each containing:
- `config/model.yml`: Project-specific configuration
- `data/`: Input data files
- `runs/`: Output directories for each run (auto-generated)

### Pipeline Flow
1. Domain pipeline transforms raw data (market data, sports data, etc.)
2. Core ML pipeline trains and evaluates models
3. Models and results are saved to disk in timestamped run directories

## Key Development Notes

- The framework separates domain-specific logic from the core ML pipeline
- All major functionality is configuration-driven through YAML files
- Entry points are defined in pyproject.toml: `alphapy` and `mflow`
- The project uses Apache License 2.0
- Warning suppression is in place for pandas and sklearn deprecations

## Development Environment

- Python Version: 3.12+
- Development Installation: `pip install -e ".[dev]"` (editable install, commands work directly)

## Branching Workflow

- **main**: Production branch - stable releases only
- **develop**: Development branch - all new features and fixes should be tested here first
- **feature branches**: Create from develop for new features, merge back to develop

### Development Process
1. Always work on develop branch or feature branches created from develop
2. Test changes on develop before merging to main
3. Create releases from main branch only
4. Use PR workflow: feature → develop → main

---

# Alfi Markets App (merged from alphapy-markets)

The `app/` directory contains **Alfi**, a React + TypeScript market analysis UI backed by a FastAPI service for real-time market data, trading signals, and agent control. It is a distinct subsystem from the core AlphaPy ML pipeline above.

## Alfi Applications

### Frontend (Alfi UI)
- **Location**: `app/frontend/`
- **Run Command**: `npm run dev`
- **URL**: http://localhost:5331/
- **Purpose**: React + TypeScript market analysis UI with screener, agent dashboard, and chat

### Backend API
- **Location**: `app/backend/api.py`
- **Run Command**: `cd app/backend && uv run python -m uvicorn api:app --host 127.0.0.1 --port 8080 --reload`
- **Check Server**: `lsof -i :8080`
- **Purpose**: FastAPI service for real-time market data, trading signals, and agent control

## Port Allocation - Stock Market Screener App

### RESERVED PORTS (DO NOT CHANGE)
- **8080**: Backend API (FastAPI/Uvicorn) — `app/backend/api.py`, endpoint http://localhost:8080/api/*, WebSocket ws://localhost:8080/ws/market-data
- **5331**: React Frontend (Vite dev server) — http://localhost:5331/, HMR enabled

### Starting the App (Codex)

**IMPORTANT**: The backend MUST be started from `app/backend/` because `api.py` uses bare imports (e.g. `from pivots import ...`). Running from the project root will fail with `ModuleNotFoundError`.

```bash
# Step 1: Check ports are free
lsof -i :8080 -i :5331

# Step 2: Start Backend API (run from app/backend/)
cd /Users/markconway/Projects/alphapy-pro/app/backend && uv run python -m uvicorn api:app --host 127.0.0.1 --port 8080 --reload

# Step 3: Start React Frontend (run from app/frontend/)
cd /Users/markconway/Projects/alphapy-pro/app/frontend && npm run dev
```

**Key details**:
- Always use `uv run python -m uvicorn` (not bare `uvicorn`) — uvicorn is a project dependency, not globally installed
- The backend loads `.env` from `app/backend/.env` via dotenv
- Backend takes ~10 seconds to start (fetches stock data for 20 symbols on startup)
- Run both commands in background (`run_in_background: true`)

### Troubleshooting
- Check for port conflicts: `lsof -i :8080 -i :5331`
- If `ModuleNotFoundError: No module named 'pivots'` — you started uvicorn from the wrong directory, must `cd app/backend` first
- Run `uv sync` from project root if dependencies are missing
- Run `npm install` in `app/frontend/` if frontend dependencies are missing
- For CORS issues, verify backend allows frontend origin

## Alfi Agent Structure

- `app/agent/`: Alfi trading agent (engine, skills, autonomy, backtesting)
  - Dual-agent platform: Swing Agent (daily bars) + Day Agent (5-min bars)
  - Both agents share `AlfiEngine` with different configs
  - `AgentCoordinator` manages both engines with shared risk, broker, portfolio optimizer
- `app/backend/api.py`: Main API server with all endpoints
- `app/backend/config.py`: Backend configuration (portfolio sources, API keys, analysis parameters)

## Frontend Tech Stack

- React 18 + TypeScript
- Vite dev server
- Tailwind CSS (dark theme)
- TanStack Query for data fetching/caching
- WebSocket for real-time market data

### Frontend Project Structure
```
app/frontend/
├── src/
│   ├── features/        # Feature-based modules (screener, pivots, chat, etc.)
│   ├── lib/             # Shared utilities (api.ts, websocket.ts)
│   ├── App.tsx
│   └── main.tsx
├── package.json
├── tsconfig.json
└── vite.config.ts
```

### Frontend Best Practices
- TypeScript for all new components — strict type checking enabled
- Feature-based folder structure in `src/features/`
- TanStack Query for API calls (handles caching, loading, errors)
- WebSocket hooks in `lib/websocket.ts` for real-time data
- Tailwind utility classes — avoid custom CSS files
- Functional components with hooks (no class components)

## Data Sources (Alfi)

- **Portfolio Sources**: alphapy (config-based), finnhub, finviz, massive
- **Historical Data**: Fetched via Massive (formerly Polygon.io) API
- **Real-time Data**: WebSocket connections for live market feeds

## Markets-Specific Projects

Three projects under `projects/` originated in alphapy-markets:
- `price-encoding/`: Price encoding system (OHLCV → text tokens: H3P1R0V2 format)
- `trade-gpt/`: Transformer training project (RoBERTa MLM + causal GPT). Has its own `requirements.txt`.
- `two-sigma/`: Two Sigma financial news / market data notebooks

## Markets Reference Docs

- `docs/PLAN-markets.md`: Multi-phase Alfi roadmap (Phases 1–8)
- `docs/markets/design-doc.md`: Alfi design document
- `docs/markets/trading-dashboard.html`: Dashboard prototype
- `scripts/markets/`: Massive API screener scripts, encoding analysis scripts