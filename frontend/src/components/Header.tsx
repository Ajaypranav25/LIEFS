import React from 'react';
import { Cpu, Activity, BarChart3, Layers, Clock, Settings, HardDrive } from 'lucide-react';
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
    <nav className="fixed top-0 left-0 w-full z-50 flex justify-between items-center px-4 sm:px-8 h-16 bg-surface/90 border-b border-outline-variant/30 backdrop-blur-xl shadow-[0_0_15px_rgba(78,222,163,0.08)]">
      {/* Brand & Main Nav Links */}
      <div className="flex items-center gap-6 lg:gap-8">
        <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => setActiveTab('playground')}>
          <div className="w-8 h-8 rounded bg-surface-container-low border border-white/10 flex items-center justify-center text-primary glow-hover">
            <Cpu className="w-5 h-5 text-primary" />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-bold tracking-tight text-primary font-sans">LIEFS Engine</span>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-container-high border border-outline-variant/40 text-on-surface-variant font-medium">
              v1.0
            </span>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="hidden md:flex items-center h-full gap-2">
          {[
            { id: 'playground', label: 'Live Playground', icon: Activity },
            { id: 'benchmarks', label: 'Benchmarks', icon: BarChart3 },
            { id: 'architecture', label: 'Architecture', icon: Layers },
            { id: 'history', label: 'Run History', icon: Clock },
          ].map((tab) => {
            const Icon = tab.icon;
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded text-xs font-mono transition-all duration-200 ${
                  isActive
                    ? 'text-primary border-b-2 border-primary font-bold bg-surface-container-low/80 shadow-glow-active'
                    : 'text-on-surface-variant hover:text-primary hover:bg-surface-container-low/40'
                }`}
              >
                <Icon className="w-3.5 h-3.5" />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Right Side Status & HUD */}
      <div className="flex items-center gap-4">
        {/* Engine Online Badge */}
        <div className="flex items-center gap-3 bg-[#0f172a] border border-white/10 rounded px-3 py-1.5 font-mono text-xs">
          <div className="flex items-center gap-2">
            <div
              className={`w-2 h-2 rounded-full ${
                serverOnline ? 'bg-primary border border-primary status-led animate-pulse' : 'bg-rose-500'
              }`}
            />
            <span className={serverOnline ? 'text-primary font-semibold' : 'text-rose-400 font-semibold'}>
              {serverOnline ? 'Engine: ONLINE' : 'DISCONNECTED'}
            </span>
          </div>

          {systemInfo && systemInfo.cuda_available && (
            <>
              <div className="w-px h-3.5 bg-outline-variant/40 hidden sm:block" />
              <div className="hidden sm:flex items-center gap-1.5 text-on-surface-variant text-[11px]">
                <HardDrive className="w-3 h-3 text-secondary" />
                <span>GPU VRAM:</span>
                <span className="text-secondary font-semibold">
                  {(systemInfo.vram_allocated_mb / 1024).toFixed(1)}GB
                </span>
                <span>/ {(systemInfo.vram_total_mb / 1024).toFixed(0)}GB</span>
              </div>
            </>
          )}
        </div>

        {/* Action icons */}
        <div className="hidden sm:flex items-center gap-1 text-on-surface-variant">
          <button
            onClick={() => setActiveTab('architecture')}
            className="p-1.5 rounded hover:text-primary hover:bg-surface-container-low transition-colors"
            title="System Architecture"
          >
            <Layers className="w-4 h-4" />
          </button>
          <button
            onClick={() => setActiveTab('benchmarks')}
            className="p-1.5 rounded hover:text-primary hover:bg-surface-container-low transition-colors"
            title="Engine Settings & Benchmarks"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>
    </nav>
  );
};
