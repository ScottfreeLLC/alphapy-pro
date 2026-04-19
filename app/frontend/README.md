# Alfi - React Frontend

Modern React + TypeScript frontend for the AlphaPy Markets application.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first styling
- **TanStack Query** - Data fetching and caching
- **Axios** - HTTP client

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Development Server

```bash
npm run dev
```

The app will be available at: **http://localhost:5331/**

### 3. Make sure the backend is running

The frontend expects the backend API to be running on `http://localhost:8080`.

See `../backend/api.py` for instructions on starting the backend.

## Available Scripts

- `npm run dev` - Start development server with HMR
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Run ESLint

## Project Structure

```
src/
├── components/        # Reusable UI components
│   ├── StockTable.tsx
│   ├── StatsBar.tsx
│   └── ErrorDisplay.tsx
├── lib/              # Utilities and helpers
│   └── api.ts        # API client
├── types/            # TypeScript type definitions
│   └── stock.ts
├── App.tsx           # Main application component
├── main.tsx          # Application entry point
└── index.css         # Global styles + Tailwind
```

## Features

- ✅ Real-time stock data table
- ✅ Auto-refresh every 30 seconds
- ✅ Color-coded price changes (green/red)
- ✅ Statistics dashboard (tickers, volume, last update)
- ✅ Error handling with user-friendly messages
- ✅ Loading states
- ✅ Responsive design
- ✅ Dark theme

## Configuration

The API endpoint is configured in `src/lib/api.ts`:

```typescript
const API_BASE_URL = 'http://localhost:8080';
```

## Development

Hot Module Replacement (HMR) is enabled by default. Edit any `.tsx` file and see changes instantly without page refresh.

TypeScript strict mode is enabled for better type safety.
