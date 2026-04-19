import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchEncoding, fetchEncodingPatterns } from '../../lib/api';
import type { TokenBreakdown, PatternMatch } from '../../types/encoding';

const TOKEN_COLORS: Record<string, string> = {
  pivot: 'text-amber-400',
  net: 'text-emerald-400',
  range: 'text-sky-400',
  volume: 'text-purple-400',
};

const TYPE_COLORS: Record<string, string> = {
  bullish: 'bg-green-900/40 border-green-700 text-green-300',
  bearish: 'bg-red-900/40 border-red-700 text-red-300',
  continuation: 'bg-blue-900/40 border-blue-700 text-blue-300',
};

function TokenDisplay({ breakdown }: { breakdown: TokenBreakdown }) {
  const { token, parsed } = breakdown;
  if (!parsed) return <span className="text-gray-500">{token}</span>;

  return (
    <span className="inline-flex gap-0.5 font-mono text-xs">
      {Object.entries(parsed).map(([type, info]) => (
        <span key={type} className={TOKEN_COLORS[type] || 'text-gray-400'} title={`${type}: ${info.raw}`}>
          {info.raw}
        </span>
      ))}
    </span>
  );
}

function PatternBadge({ pattern }: { pattern: PatternMatch }) {
  const colorClass = TYPE_COLORS[pattern.type] || 'bg-gray-800 border-gray-700 text-gray-300';
  return (
    <span
      className={`inline-block px-2 py-0.5 text-xs rounded border ${colorClass}`}
      title={pattern.description || pattern.name}
    >
      {pattern.name.replace(/_/g, ' ')}
    </span>
  );
}

export default function EncodingPanel() {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputValue, setInputValue] = useState('AAPL');

  const { data: encoding, isLoading } = useQuery({
    queryKey: ['encoding', symbol],
    queryFn: () => fetchEncoding(symbol, 50),
    refetchInterval: 30000,
  });

  const { data: patternCatalog } = useQuery({
    queryKey: ['encodingPatterns'],
    queryFn: fetchEncodingPatterns,
    refetchInterval: 60000,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSymbol(inputValue.toUpperCase());
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between">
        <h3 className="font-medium">Price Encoding</h3>
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1 text-xs w-20 uppercase"
            placeholder="Symbol"
          />
          <button type="submit" className="bg-gray-700 hover:bg-gray-600 rounded px-2 py-1 text-xs">
            Go
          </button>
        </form>
      </div>

      <div className="p-4 space-y-4">
        {isLoading ? (
          <div className="text-center text-gray-500 text-sm py-4">Loading encoding...</div>
        ) : encoding ? (
          <>
            {/* Token sequence */}
            <div>
              <div className="text-xs text-gray-500 mb-2">Last 10 Bars ({encoding.bar_count} total)</div>
              <div className="flex flex-wrap gap-1.5">
                {encoding.last_10_tokens.map((tb, i) => (
                  <div
                    key={i}
                    className="bg-gray-800 rounded px-1.5 py-1 border border-gray-700"
                  >
                    <TokenDisplay breakdown={tb} />
                  </div>
                ))}
              </div>
            </div>

            {/* Legend */}
            <div className="flex gap-3 text-xs">
              {Object.entries(TOKEN_COLORS).map(([type, color]) => (
                <span key={type} className={`${color} opacity-70`}>
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </span>
              ))}
            </div>

            {/* Pattern matches */}
            {encoding.patterns.length > 0 && (
              <div>
                <div className="text-xs text-gray-500 mb-2">
                  Detected Patterns ({encoding.patterns.length})
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {encoding.patterns.map((p, i) => (
                    <PatternBadge key={i} pattern={p} />
                  ))}
                </div>
              </div>
            )}

            {/* Cross-symbol matches from catalog */}
            {patternCatalog && Object.keys(patternCatalog.current_matches).length > 0 && (
              <div>
                <div className="text-xs text-gray-500 mb-2">Active Patterns Across Symbols</div>
                <div className="space-y-1">
                  {Object.entries(patternCatalog.current_matches).slice(0, 8).map(([sym, matches]) => (
                    <div key={sym} className="flex items-center gap-2 text-xs">
                      <span className="text-gray-400 font-mono w-12">{sym}</span>
                      <div className="flex gap-1">
                        {matches.map((m, i) => (
                          <PatternBadge key={i} pattern={m} />
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="text-center text-gray-500 text-sm py-4">
            No encoding data available
          </div>
        )}
      </div>
    </div>
  );
}
