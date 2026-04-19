import { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Search, Dna, RefreshCw, Hash } from 'lucide-react';
import {
  fetchMotifLibrary,
  runMotifDiscovery,
  fetchEvolvedRules,
  runRuleEvolution,
} from '../../lib/api';

export default function DiscoveryPanel() {
  const [discoverySymbols, setDiscoverySymbols] = useState('SPY, AAPL, MSFT');

  const motifsQuery = useQuery({
    queryKey: ['motifLibrary'],
    queryFn: fetchMotifLibrary,
    refetchInterval: 60000,
  });

  const rulesQuery = useQuery({
    queryKey: ['evolvedRules'],
    queryFn: fetchEvolvedRules,
    refetchInterval: 60000,
  });

  const discoverMutation = useMutation({
    mutationFn: () => {
      const symbols = discoverySymbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
      return runMotifDiscovery({ symbols, days_back: 90, window_sizes: [20, 40, 60] });
    },
    onSuccess: () => {
      motifsQuery.refetch();
    },
  });

  const evolveMutation = useMutation({
    mutationFn: () => {
      const symbols = discoverySymbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
      return runRuleEvolution({ symbols, days_back: 90, generations: 50, population: 300, window_size: 20 });
    },
    onSuccess: () => {
      rulesQuery.refetch();
    },
  });

  const motifs = motifsQuery.data;
  const rules = rulesQuery.data;

  return (
    <div className="space-y-4">
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Search size={18} className="text-purple-400" />
            <h3 className="font-medium">Pattern Discovery</h3>
          </div>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-3 mb-4">
          <input
            type="text"
            value={discoverySymbols}
            onChange={(e) => setDiscoverySymbols(e.target.value)}
            placeholder="SPY, AAPL, MSFT"
            className="flex-1 bg-gray-800 border border-gray-700 rounded-lg px-3 py-1.5 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <button
            onClick={() => discoverMutation.mutate()}
            disabled={discoverMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:bg-gray-700 rounded-lg text-sm transition-colors"
          >
            {discoverMutation.isPending ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <Search size={14} />
            )}
            Discover Motifs
          </button>
          <button
            onClick={() => evolveMutation.mutate()}
            disabled={evolveMutation.isPending}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 disabled:bg-gray-700 rounded-lg text-sm transition-colors"
          >
            {evolveMutation.isPending ? (
              <RefreshCw size={14} className="animate-spin" />
            ) : (
              <Dna size={14} />
            )}
            Evolve Rules
          </button>
        </div>

        {/* Status messages */}
        {discoverMutation.isSuccess && (
          <div className="mb-3 bg-purple-900/30 border border-purple-800 rounded-lg p-2 text-xs text-purple-300">
            Discovered {discoverMutation.data.motifs_discovered} motifs across {discoverMutation.data.symbols_processed} symbols
          </div>
        )}
        {evolveMutation.isSuccess && (
          <div className="mb-3 bg-emerald-900/30 border border-emerald-800 rounded-lg p-2 text-xs text-emerald-300">
            Evolved {evolveMutation.data.new_rules} new rules ({evolveMutation.data.total_rules} total)
          </div>
        )}

        {/* Motif Library */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Hash size={14} className="text-purple-400" />
              <span className="text-sm font-medium text-gray-300">Motif Library</span>
              <span className="text-xs text-gray-500">
                ({motifs?.total_motifs ?? 0} patterns)
              </span>
            </div>
            {motifs?.motifs && motifs.motifs.length > 0 ? (
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {motifs.motifs.slice(0, 10).map((m: Record<string, unknown>) => (
                  <div
                    key={m.motif_id as number}
                    className="bg-gray-800 rounded px-3 py-2 text-xs flex items-center justify-between"
                  >
                    <div>
                      <span className="text-gray-300">
                        Motif #{m.motif_id as number}
                      </span>
                      <span className="text-gray-500 ml-2">
                        {m.window_size as number} bars
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-gray-400">
                        {m.occurrences as number} occ.
                      </span>
                      <span className={(m.win_rate as number) >= 0.5 ? 'text-green-400' : 'text-red-400'}>
                        {((m.win_rate as number) * 100).toFixed(0)}% win
                      </span>
                      <span className="text-gray-400">
                        S: {(m.sharpe as number).toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-gray-500 bg-gray-800 rounded p-3">
                No motifs discovered yet. Run discovery to find patterns.
              </div>
            )}
          </div>

          {/* Evolved Rules */}
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Dna size={14} className="text-emerald-400" />
              <span className="text-sm font-medium text-gray-300">Evolved Rules</span>
              <span className="text-xs text-gray-500">
                ({rules?.active_rules ?? 0} active)
              </span>
            </div>
            {rules?.rules && rules.rules.length > 0 ? (
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {rules.rules.slice(0, 10).map((r: Record<string, unknown>) => (
                  <div
                    key={r.rule_id as number}
                    className="bg-gray-800 rounded px-3 py-2 text-xs"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-gray-300 font-medium">
                        Rule #{r.rule_id as number}
                      </span>
                      <span className={
                        (r.status as string) === 'active'
                          ? 'text-green-400'
                          : 'text-gray-500'
                      }>
                        {r.status as string}
                      </span>
                    </div>
                    <div className="flex items-center gap-3 text-gray-400">
                      <span>Sharpe: {(r.sharpe_test as number).toFixed(2)}</span>
                      <span>{((r.win_rate as number) * 100).toFixed(0)}% win</span>
                      <span>{r.total_trades as number} trades</span>
                    </div>
                    <div className="text-gray-600 truncate mt-0.5" title={r.expression as string}>
                      {(r.expression as string).slice(0, 60)}...
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-xs text-gray-500 bg-gray-800 rounded p-3">
                No evolved rules yet. Run evolution to generate trading rules.
              </div>
            )}
          </div>
        </div>

        {/* Summary stats */}
        {((motifs?.total_motifs ?? 0) > 0 || (rules?.active_rules ?? 0) > 0) && (
          <div className="grid grid-cols-4 gap-2 mt-3">
            {[
              { label: 'Motifs', value: motifs?.total_motifs ?? 0 },
              { label: 'Avg Win Rate', value: motifs?.avg_win_rate ? `${(motifs.avg_win_rate * 100).toFixed(0)}%` : '—' },
              { label: 'Active Rules', value: rules?.active_rules ?? 0 },
              { label: 'Best Sharpe', value: rules?.best_sharpe?.toFixed(2) ?? '—' },
            ].map(({ label, value }) => (
              <div key={label} className="bg-gray-800 rounded p-2 text-center">
                <div className="text-xs text-gray-500">{label}</div>
                <div className="text-sm font-medium mt-0.5">{value}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
