import React, { useState, useEffect } from 'react';
import {
  Clock,
  RefreshCw,
  Trash2,
  Download,
  Database,
} from 'lucide-react';
import { getBenchmarkHistory, clearLocalHistory } from '../lib/supabase';
import { useAuth } from '../context/AuthContext';
import type { SavedBenchmarkRun } from '../types';

export const HistoryView: React.FC = () => {
  const { user } = useAuth();
  const [history, setHistory] = useState<SavedBenchmarkRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await getBenchmarkHistory(user?.id);
      setHistory(data);
    } catch (e) {
      console.warn('Failed to load history:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHistory();
  }, [user?.id]);

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
    <div className="pt-6 px-4 sm:px-8 max-w-[1440px] mx-auto pb-12 space-y-6 font-sans text-xs">
      {/* Top Header */}
      <div className="p-5 rounded-xl border border-outline-variant bg-surface-container flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shadow-subtle">
        <div>
          <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
            <Clock className="w-4 h-4 text-primary" />
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
            className="px-3 py-1.5 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] text-slate-300 hover:text-white text-xs flex items-center gap-1.5 transition-all cursor-pointer shadow-subtle font-medium"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleExportJson}
            disabled={history.length === 0}
            className="px-3 py-1.5 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] text-slate-300 hover:text-white text-xs flex items-center gap-1.5 transition-all disabled:opacity-40 cursor-pointer shadow-subtle font-medium"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export JSON</span>
          </button>

          <button
            onClick={handleClear}
            disabled={history.length === 0}
            className="px-3 py-1.5 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-rose-500/30 text-rose-400 hover:text-rose-300 text-xs flex items-center gap-1.5 transition-all disabled:opacity-40 cursor-pointer shadow-subtle font-medium"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Local</span>
          </button>
        </div>
      </div>

      {/* Runs List */}
      {history.length === 0 ? (
        <div className="p-12 rounded-xl border border-outline-variant bg-surface-container text-center flex flex-col items-center justify-center shadow-subtle">
          <Database className="w-10 h-10 text-slate-600 mb-3" />
          <h3 className="text-sm font-medium text-slate-300">No Benchmark Runs Recorded Yet</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-sm">
            Execute benchmarks from the "Benchmarks" tab to persist and review comparative performance runs.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {history.map((run, idx) => (
            <div key={run.id || idx} className="p-5 rounded-xl border border-outline-variant bg-surface-container space-y-4 shadow-subtle">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-outline-variant pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400" />
                  <span className="text-xs font-semibold text-slate-200">
                    Run #{history.length - idx} • {new Date(run.created_at).toLocaleString()}
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant text-slate-400 font-mono">
                    {run.device || 'CUDA GPU'}
                  </span>
                </div>
                <div className="text-xs text-slate-400 font-mono">
                  Max Tokens: <span className="text-primary font-medium">{run.max_tokens}</span>
                </div>
              </div>

              <div className="text-xs text-slate-300 bg-surface-lowest p-3 rounded-lg border border-outline-variant font-sans">
                "{run.prompt}"
              </div>

              {/* Mini comparative table for this run */}
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="text-slate-400 text-[10px] uppercase font-semibold font-mono border-b border-outline-variant">
                      <th className="py-2 px-2">Engine</th>
                      <th className="py-2 px-2 text-right">Throughput</th>
                      <th className="py-2 px-2 text-right">TTFT</th>
                      <th className="py-2 px-2 text-right">TPOT</th>
                      <th className="py-2 px-2 text-right">Peak VRAM</th>
                      <th className="py-2 px-2 text-right">Speedup</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-outline-variant font-mono">
                    {run.results.map((r, rIdx) => (
                      <tr key={rIdx} className="hover:bg-white/[0.02]">
                        <td className="py-2 px-2 font-sans font-medium text-slate-200">{r.engine_name}</td>
                        <td className="py-2 px-2 text-right font-semibold text-sky-400 tabular-nums">
                          {(r.throughput_tok_s || r.tokens_per_sec || 0).toFixed(1)} tok/s
                        </td>
                        <td className="py-2 px-2 text-right text-slate-300 tabular-nums">{r.ttft_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-2 text-right text-slate-300 tabular-nums">{r.tpot_ms.toFixed(1)} ms</td>
                        <td className="py-2 px-2 text-right text-slate-300 tabular-nums">{Math.round(r.peak_vram_mb)} MB</td>
                        <td className="py-2 px-2 text-right">
                          <span className="text-amber-400 font-semibold tabular-nums">
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
