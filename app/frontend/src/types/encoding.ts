export interface ParsedToken {
  type: string;
  direction: number;
  magnitude: number;
  raw: string;
}

export interface TokenBreakdown {
  token: string;
  parsed: Record<string, ParsedToken> | null;
}

export interface PatternMatch {
  name: string;
  type: string;
  bars?: number;
  description?: string;
}

export interface EncodingResponse {
  symbol: string;
  encoded: string;
  bar_count: number;
  last_10_tokens: TokenBreakdown[];
  patterns: PatternMatch[];
}

export interface EncodingHistoryResponse {
  symbol: string;
  window_size: number;
  total_windows: number;
  windows: string[];
}

export interface SimilarityResponse {
  symbol_a: string;
  symbol_b: string;
  similarity: number;
  bars: number;
}

export interface PatternCatalogEntry {
  name: string;
  type: string;
  description: string;
  min_bars: number;
}

export interface PatternCatalogResponse {
  catalog: PatternCatalogEntry[];
  current_matches: Record<string, PatternMatch[]>;
}
