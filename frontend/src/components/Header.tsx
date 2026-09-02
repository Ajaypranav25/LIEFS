import React from 'react';
import { Zap, Activity, HardDrive, BarChart3, Clock, Layers } from 'lucide-react';
import type { SystemInfo } from '../types';

interface HeaderProps {
  activeTab: 'playground' | 'benchmarks' | 'architecture' | 'history';
  setActiveTab: (tab: 'playground' | 'benchmarks' | 'architecture' | 'history') => void;
  systemInfo: SystemInfo | null;
  serverOnline: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  systemInfo,
  serverOnline,
}) => {
  return (
    <header className="border-b border-slate-800/80 bg-obsidian-900/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left: Brand / Title */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-glow-emerald">
              <Zap className="w-5 h-5 text-emerald-400 fill-emerald-400/20" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-bold text-lg tracking-wider text-slate-100 font-mono">LIEFS</span>
                <span className="text-xs px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono font-medium">
                  v1.0.0
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                LLM Inference Engine From Scratch • Qwen2.5-0.5B
              </p>
            </div>
          </div>

          {/* Center: Navigation Tabs */}
          <nav className="flex items-center gap-1 bg-obsidian-950/80 p-1 rounded-lg border border-slate-800">
            <button
              onClick={() => setActiveTab('playground')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'playground'
                  ? 'bg-emerald-500 text-obsidian-950 font-semibold shadow-glow-emerald'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Live Playground</span>
            </button>
            <button
              onClick={() => setActiveTab('benchmarks')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'benchmarks'
                  ? 'bg-emerald-500 text-obsidian-950 font-semibold shadow-glow-emerald'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5" />
              <span>Benchmarks</span>
            </button>
            <button
              onClick={() => setActiveTab('architecture')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'architecture'
                  ? 'bg-emerald-500 text-obsidian-950 font-semibold shadow-glow-emerald'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Architecture & Sizing</span>
            </button>
            <button
              onClick={() => setActiveTab('history')}
              className={`flex items-center gap-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                activeTab === 'history'
                  ? 'bg-emerald-500 text-obsidian-950 font-semibold shadow-glow-emerald'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
              }`}
            >
              <Clock className="w-3.5 h-3.5" />
              <span>Run History</span>
            </button>
          </nav>

          {/* Right: Hardware Telemetry HUD */}
          <div className="flex items-center gap-3">
            {/* Status indicator */}
            <div className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs font-mono">
              <span
                className={`w-2 h-2 rounded-full ${
                  serverOnline ? 'bg-emerald-400 animate-pulse' : 'bg-rose-500'
                }`}
              />
              <span className={serverOnline ? 'text-emerald-400' : 'text-rose-400'}>
                {serverOnline ? 'ENGINE ONLINE' : 'DISCONNECTED'}
              </span>
            </div>

            {/* GPU / VRAM Meter */}
            {systemInfo && systemInfo.cuda_available && (
              <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs font-mono">
                <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
                <div className="flex flex-col">
                  <span className="text-[10px] text-slate-400 leading-tight">
                    {systemInfo.device_name.replace('NVIDIA ', '')}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-200 font-semibold">
                      {Math.round(systemInfo.vram_allocated_mb)}MB
                    </span>
                    <span className="text-slate-500 text-[10px]">
                      / {Math.round(systemInfo.vram_total_mb)}MB
                    </span>
                    <div className="w-12 h-1.5 bg-slate-800 rounded-full overflow-hidden ml-1">
                      <div
                        className="h-full bg-cyan-400 rounded-full transition-all duration-500"
                        style={{ width: `${Math.min(100, systemInfo.vram_usage_percent)}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
