import { useState, useMemo } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';
import { Stock } from '../types/stock';

interface StockTableProps {
  stocks: Stock[];
}

type SortField = 'symbol' | 'price' | 'daily_change' | 'daily_change_pct' | 'volume' | 'high' | 'low';
type SortDir = 'asc' | 'desc';

export default function StockTable({ stocks }: StockTableProps) {
  const [sortField, setSortField] = useState<SortField>('symbol');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const [filter, setFilter] = useState('');

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortDir(sortDir === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDir(field === 'symbol' ? 'asc' : 'desc');
    }
  };

  const sorted = useMemo(() => {
    let filtered = stocks;
    if (filter) {
      const q = filter.toUpperCase();
      filtered = stocks.filter((s) => s.symbol.includes(q));
    }
    return [...filtered].sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (typeof aVal === 'string' && typeof bVal === 'string') {
        return sortDir === 'asc' ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
      }
      const diff = (aVal as number) - (bVal as number);
      return sortDir === 'asc' ? diff : -diff;
    });
  }, [stocks, sortField, sortDir, filter]);

  const formatChangePercent = (pct: number) => {
    return pct >= 0 ? `+${pct.toFixed(2)}%` : `${pct.toFixed(2)}%`;
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (sortField !== field) return null;
    return sortDir === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />;
  };

  const thClass = 'px-6 py-3 text-xs font-medium text-gray-400 uppercase tracking-wider cursor-pointer hover:text-gray-200 select-none';

  return (
    <div>
      <div className="mb-3">
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter by symbol..."
          className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-100 placeholder-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 w-48"
        />
      </div>
      <div className="overflow-x-auto rounded-lg border border-gray-700">
        <table className="w-full text-left">
          <thead className="bg-gray-800 border-b border-gray-700">
            <tr>
              <th className={thClass} onClick={() => handleSort('symbol')}>
                <span className="flex items-center gap-1">Symbol <SortIcon field="symbol" /></span>
              </th>
              <th className={thClass} onClick={() => handleSort('price')}>
                <span className="flex items-center gap-1">Price <SortIcon field="price" /></span>
              </th>
              <th className={thClass} onClick={() => handleSort('daily_change')}>
                <span className="flex items-center gap-1">Change <SortIcon field="daily_change" /></span>
              </th>
              <th className={thClass} onClick={() => handleSort('daily_change_pct')}>
                <span className="flex items-center gap-1">Change % <SortIcon field="daily_change_pct" /></span>
              </th>
              <th className={thClass} onClick={() => handleSort('volume')}>
                <span className="flex items-center gap-1">Volume <SortIcon field="volume" /></span>
              </th>
              <th className={thClass} onClick={() => handleSort('high')}>
                <span className="flex items-center gap-1">High <SortIcon field="high" /></span>
              </th>
              <th className={thClass} onClick={() => handleSort('low')}>
                <span className="flex items-center gap-1">Low <SortIcon field="low" /></span>
              </th>
            </tr>
          </thead>
          <tbody className="bg-gray-900 divide-y divide-gray-800">
            {sorted.map((stock) => (
              <tr key={stock.symbol} className="hover:bg-gray-800/50 transition-colors">
                <td className="px-6 py-4 whitespace-nowrap font-bold">{stock.symbol}</td>
                <td className="px-6 py-4 whitespace-nowrap">${stock.price.toFixed(2)}</td>
                <td className="px-6 py-4 whitespace-nowrap">${stock.daily_change.toFixed(2)}</td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span className={`font-bold ${stock.daily_change_pct >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                    {formatChangePercent(stock.daily_change_pct)}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">{stock.volume.toLocaleString()}</td>
                <td className="px-6 py-4 whitespace-nowrap">${stock.high.toFixed(2)}</td>
                <td className="px-6 py-4 whitespace-nowrap">${stock.low.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
