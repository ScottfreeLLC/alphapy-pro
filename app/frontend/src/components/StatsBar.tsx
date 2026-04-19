interface StatsBarProps {
  tickersCount: number;
  totalVolume: number;
  lastUpdated: string;
}

export default function StatsBar({ tickersCount, totalVolume, lastUpdated }: StatsBarProps) {
  return (
    <div className="flex gap-8 w-full justify-center p-4 bg-gray-800/50 rounded-lg">
      <div className="flex flex-col items-center gap-1">
        <span className="text-xs text-gray-400">Tickers</span>
        <span className="text-2xl font-bold">{tickersCount}</span>
      </div>
      <div className="flex flex-col items-center gap-1">
        <span className="text-xs text-gray-400">Total Volume</span>
        <span className="text-2xl font-bold">{totalVolume.toLocaleString()}</span>
      </div>
      <div className="flex flex-col items-center gap-1">
        <span className="text-xs text-gray-400">Last Updated</span>
        <span className="text-2xl font-bold">{lastUpdated}</span>
      </div>
    </div>
  );
}
