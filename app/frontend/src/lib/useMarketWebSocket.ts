import { useEffect, useRef, useState, useCallback } from 'react';

const MARKET_WS_URL = 'ws://localhost:8080/ws/market-data';

export interface LivePrice {
  price: number;
  volume: number;
}

export function useMarketWebSocket() {
  const [livePrices, setLivePrices] = useState<Map<string, LivePrice>>(new Map());
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(MARKET_WS_URL);

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'price_update' && msg.data) {
          setLivePrices((prev) => {
            const next = new Map(prev);
            for (const [symbol, data] of Object.entries(msg.data)) {
              const d = data as LivePrice;
              next.set(symbol, { price: d.price, volume: d.volume });
            }
            return next;
          });
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { livePrices, connected };
}
