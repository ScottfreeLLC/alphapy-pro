import { useEffect, useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchMarketScreener } from '../../lib/api';
import { useMarketWebSocket } from '../../lib/useMarketWebSocket';
import StockTable from '../../components/StockTable';
import StatsBar from '../../components/StatsBar';
import ErrorDisplay from '../../components/ErrorDisplay';

export default function ScreenerPage() {
  const [lastUpdated, setLastUpdated] = useState('');

  // REST for initial data + fallback (slow poll at 120s)
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['marketScreener'],
    queryFn: fetchMarketScreener,
    refetchInterval: 120000,
  });

  // WebSocket for real-time price overlay
  const { livePrices, connected: wsConnected } = useMarketWebSocket();

  // Merge live prices into screener data
  const stocks = useMemo(() => {
    if (!data?.stocks) return [];
    return data.stocks.map((stock) => {
      const live = livePrices.get(stock.symbol);
      if (live && live.price > 0) {
        const prevClose = stock.price - stock.daily_change;
        const newChange = live.price - prevClose;
        const newChangePct = prevClose !== 0 ? (newChange / prevClose) * 100 : 0;
        return {
          ...stock,
          price: live.price,
          volume: live.volume > 0 ? live.volume : stock.volume,
          daily_change: Math.round(newChange * 100) / 100,
          daily_change_pct: Math.round(newChangePct * 100) / 100,
        };
      }
      return stock;
    });
  }, [data?.stocks, livePrices]);

  useEffect(() => {
    if (data) {
      setLastUpdated(new Date().toLocaleTimeString());
    }
  }, [data]);

  const totalVolume = stocks.reduce((sum, stock) => sum + stock.volume, 0);
  const tickersCount = stocks.length;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-3xl font-bold">Stock Screener</h2>
        {wsConnected && (
          <span className="flex items-center gap-1.5 text-xs text-green-400">
            <span className="w-1.5 h-1.5 bg-green-400 rounded-full animate-pulse" />
            Live
          </span>
        )}
      </div>

      <StatsBar
        tickersCount={tickersCount}
        totalVolume={totalVolume}
        lastUpdated={lastUpdated}
      />

      {error && <ErrorDisplay message={error.message} />}

      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-white" />
        </div>
      ) : stocks.length > 0 ? (
        <StockTable stocks={stocks} />
      ) : (
        <div className="text-center py-12">
          <p className="text-gray-400">No data available.</p>
          <button
            onClick={() => refetch()}
            className="mt-4 px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
          >
            Refresh Data
          </button>
        </div>
      )}
    </div>
  );
}
