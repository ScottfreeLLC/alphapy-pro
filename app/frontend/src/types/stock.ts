export interface Stock {
  symbol: string;
  price: number;
  high: number;
  low: number;
  open: number;
  volume: number;
  daily_change: number;
  daily_change_pct: number;
}

export interface MarketScreenerResponse {
  stocks: Stock[];
  count: number;
}
