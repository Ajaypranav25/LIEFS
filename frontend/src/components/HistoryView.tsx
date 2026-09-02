import React, { useState, useEffect } from 'react';
import {
  Clock,
  RefreshCw,
  Trash2,
  Download,
  Database,
} from 'lucide-react';
import { getBenchmarkHistory, clearLocalHistory } from '../lib/supabase';
import type { SavedBenchmarkRun } from '../types';

export const HistoryView: React.FC = () => {
  const [history, setHistory] = useState<SavedBenchmarkRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await getBenchmarkHistory();
      setHistory(data);
    } catch (e) {
      console.warn('Failed to load history:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, []);

  const handleExportJson = () => {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(history, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `liefs_benchmark_history_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const handleClear = () => {
    if (confirm('Clear all local benchmark history?')) {
      clearLocalHistory();
      setHistory([]);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 space-y-6">
      {/* Top Header */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold font-mono text-slate-100 flex items-center gap-2">
            <Clock className="w-5 h-5 text-emerald-400" />
            Historical Benchmark Runs
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Persisted runs stored in Supabase Postgres & synced locally across sessions.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadHistory}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg bg-obsidian-900 border border-slate-800 text-slate-300 hover:text-white text-xs font-mono flex items-center gap-1.5 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleExportJson}
            disabled={history.length === 0}
            className="px-3 py-1.5 rounded-lg bg-obsidian-900 border border-slate-800 text-slate-300 hover:text-white text-xs font-mono flex items-center gap-1.5 transition-all disabled:opacity-40"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export JSON</span>
          </button>

          <button
            onClick={handleClear}
            disabled={history.length === 0}
            className="px-3 py-1.5 rounded-lg bg-obsidian-900 border border-slate-800 text-rose-400 hover:text-rose-300 text-xs font-mono flex items-center gap-1.5 transition-all disabled:opacity-40"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Local</span>
          </button>
        </div>
      </div>

      {/* Runs List */}
      {history.length === 0 ? (
        <div className="glass-panel p-12 rounded-2xl text-center flex flex-col items-center justify-center">
          <Database className="w-12 h-12 text-slate-600 mb-3" />
          <h3 className="text-sm font-mono font-semibold text-slate-300">No Benchmark Runs Recorded Yet</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-sm">
            Execute benchmarks from the "Benchmarks" tab and click "Save Run" to persist results to Supabase.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((run, idx) => (
            <div key={run.id || idx} className="glass-panel p-5 rounded-2xl space-y-4 border border-slate-800/80">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-xs font-mono font-bold text-slate-100">
                    Run #{history.length - idx} • {new Date(run.created_at).toLocaleString()}
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">
                    {run.device || 'CUDA GPU'}
                  </span>
                </div>
                <div className="text-xs font-mono text-slate-400">
                  Max Tokens: <span className="text-emerald-400">{run.max_tokens}</span>
                </div>
              </div>

              <div className="text-xs font-sans text-slate-300 italic bg-obsidian-950 p-2.5 rounded-lg border border-slate-800/60">
                "{run.prompt}"
              </div>

              {/* Mini comparative table for this run */}
              <div className="overflow-x-auto">
                <table className="w-full text-left font-mono text-xs">
                  <thead>
                    <tr className="text-slate-500 text-[11px] border-b border-slate-800/60">
                      <th className="py-1 px-2">Engine</th>
                      <th className="py-1 px-2 text-right">Throughput</th>
                      <th className="py-1 px-2 text-right">TTFT</th>
                      <th className="py-1 px-2 text-right">TPOT</th>
                      <th className="py-1 px-2 text-right">Peak VRAM</th>
                      <th className="py-1 px-2 text-right">Speedup</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/40 text-slate-300">
                    {run.results.map((r, rIdx) => (
                      <tr key={rIdx}>
                        <td className="py-2 px-2 font-semibold text-slate-200">{r.engine_name}</td>
                        <td className="py-2 px-2 text-right font-bold text-emerald-400">
                          {(r.throughput_tok_s || r.tokens_per_sec || 0).toFixed(1)} t/s
                        </td>
                        <td className="py-2 px-2 text-right text-cyan-300">{r.ttft_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-2 text-right text-slate-300">{r.tpot_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-2 text-right text-purple-300">{Math.round(r.peak_vram_mb)} MB</td>
                        <td className="py-2 px-2 text-right">
                          <span className="text-emerald-400 font-bold">
                            {(r.speedup || r.speedup_vs_naive || 1.0).toFixed(2)}x
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
