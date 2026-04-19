# AlphaPy Markets

A market analysis application with real-time pivot pattern detection and trading signals.

## Features

✅ **Real-time Stock Screener**
- Bullish and bearish signal detection
- 12 pivot patterns (Gartley, ABCD, Three-Drive, Wolfe Wave, etc.)
- Live WebSocket connection indicator
- Click any stock for detailed analysis

✅ **Detailed Pivot Analysis**
- Pattern-by-pattern breakdown
- Pivot point statistics
- Price action metrics
- Sentiment analysis (bullish/bearish/neutral)

✅ **AI Chat Assistant**
- Interactive market Q&A
- Pattern explanation
- Real-time conversation interface

✅ **Modern Dark UI**
- Mometic-style gradient design
- Responsive grid layouts
- Smooth transitions and hover effects

## Tech Stack

**Frontend:**
- React 18 + TypeScript
- Vite (blazing fast dev server)
- Tailwind CSS (dark theme)
- TanStack Query (data fetching)
- WebSocket for real-time updates

**Backend:**
- FastAPI (Python)
- Massive (formerly Polygon.io) API integration
- 12 pivot pattern detection algorithms (Gartley, ABCD, Three-Drive, Wolfe Wave, Expansion, Squeeze, Rectangle, Wedge)
- Real-time WebSocket support

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.12+
- uv package manager
- Massive API key

### 1. Backend Setup

```bash
# From project root
cd app/backend

# Set environment variables
export MASSIVE_API_KEY="your_massive_api_key_here"

# Activate venv (from project root)
cd /Users/markconway/Projects/alphapy-markets
source .venv/bin/activate

# Run server
cd app/backend
uvicorn api:app --host 0.0.0.0 --port 8080 --reload
```

Backend will be available at: http://localhost:8080

### 2. Frontend Setup

```bash
# From project root
cd app/frontend

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

Frontend will be available at: http://localhost:5331

## API Endpoints

- `GET /` - Health check
- `GET /api/symbols` - Get all tracked symbols
- `POST /api/pivots/analyze` - Analyze single symbol
- `GET /api/pivots/scan` - Scan all symbols
- `GET /api/pivots/bullish` - Get bullish signals
- `GET /api/pivots/bearish` - Get bearish signals
- `GET /api/quote/{symbol}` - Get stock quote
- `WS /ws/market-data` - WebSocket for real-time updates

## Project Structure

```
app/
├── backend/
│   └── api.py              # Unified FastAPI server
├── frontend/
│   ├── src/
│   │   ├── features/
│   │   │   ├── screener/   # Stock screener UI
│   │   │   ├── pivots/     # Pivot detail view
│   │   │   └── chat/       # AI assistant
│   │   ├── lib/
│   │   │   ├── api.ts      # API client
│   │   │   └── websocket.ts # WebSocket hook
│   │   └── App.tsx         # Main app
│   ├── package.json
│   └── tailwind.config.js
└── README.md
```

## Development

### Backend Development

The backend auto-reloads on file changes. Edit `backend/api.py` and save to see changes.

### Frontend Development

Vite provides HMR (Hot Module Replacement). Edit any `.tsx` file and see instant updates.

### Adding New Features

1. Backend: Add endpoints to `backend/api.py`
2. Frontend: Create components in `frontend/src/features/`
3. Connect via API client in `lib/api.ts`

## Deployment

### Docker (Recommended)

```bash
# Coming soon - docker-compose.yml
docker-compose up -d
```

### Manual Deployment

**Backend:**
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app
```

**Frontend:**
```bash
npm run build
# Serve dist/ folder with nginx or similar
```

## Configuration

### Backend Environment Variables

- `MASSIVE_API_KEY` - Your Massive API key (required)
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8080)

### Frontend Environment Variables

Create `.env` file in `frontend/`:

```
VITE_API_URL=http://localhost:8080
VITE_WS_URL=ws://localhost:8080/ws/market-data
```

## License

Copyright 2024 ScottFree Analytics LLC
