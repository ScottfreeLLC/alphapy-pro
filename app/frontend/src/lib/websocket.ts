import { useEffect, useRef, useState, useCallback } from 'react';
import { AgentState, CombinedState, SharedRiskStatus } from '../types/agent';

const ALFI_WS_URL = 'ws://localhost:8080/ws/alfi';

export function useAlfiWebSocket() {
  const [combinedState, setCombinedState] = useState<CombinedState | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(ALFI_WS_URL);

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'state_update' && msg.data) {
          setCombinedState(msg.data);
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 3 seconds
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

  // Derive per-agent states from combined
  const swingState: AgentState | null = combinedState?.swing ?? null;
  const dayState: AgentState | null = combinedState?.day ?? null;
  const sharedRisk: SharedRiskStatus | null = combinedState?.shared_risk ?? null;

  return { combinedState, swingState, dayState, sharedRisk, connected };
}
