import React from 'react';
import {
  Menu,
  BarChart2,
  MessageSquare,
  HardDrive,
} from 'lucide-react';
import type { SystemInfo } from '../types';

interface HeaderProps {
  activeView: 'chat' | 'analysis';
  setActiveView: (view: 'chat' | 'analysis') => void;
  systemInfo: SystemInfo | null;
  serverOnline: boolean;
  onToggleSidebar: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeView,
  setActiveView,
  systemInfo,
  serverOnline,
  onToggleSidebar,
}) => {
  return (
    <header className="h-14 bg-surface-dim border-b border-surface-container flex items-center justify-between px-4 md:px-6 shrink-0 z-30">
      {/* Left Section: Mobile Menu + View Navigation Switcher */}
      <div className="flex items-center gap-4">
        <button
          onClick={onToggleSidebar}
          className="md:hidden p-1.5 rounded-lg text-on-surface-variant hover:text-primary hover:bg-surface-high transition-colors"
          title="Toggle Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* View Mode Switcher Pills */}
        <div className="flex bg-surface-lowest p-1 rounded-xl border border-outline-variant/30 font-mono text-xs">
          <button
            onClick={() => setActiveView('chat')}
            className={`px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${
              activeView === 'chat'
                ? 'bg-primary-container text-on-primary-container font-bold shadow-glow-cyan'
                : 'text-on-surface-variant hover:text-on-background'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5" />
            <span>AI Chat</span>
          </button>

          <button
            onClick={() => setActiveView('analysis')}
            className={`px-3 py-1 rounded-lg flex items-center gap-1.5 transition-all cursor-pointer ${
              activeView === 'analysis'
                ? 'bg-primary-container text-on-primary-container font-bold shadow-glow-cyan'
                : 'text-on-surface-variant hover:text-on-background'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5" />
            <span>Model Analysis</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-primary/20 text-primary font-bold">
              PRO
            </span>
          </button>
        </div>
      </div>

      {/* Right Section: Live Telemetry HUD & Status */}
      <div className="flex items-center gap-4 text-xs font-mono">
        {/* Engine Online Status */}
        <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-surface-lowest border border-outline-variant/40">
          <span
            className={`w-2 h-2 rounded-full ${
              serverOnline ? 'bg-primary status-led animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className="text-[11px] text-on-surface-variant hidden sm:inline">Engine:</span>
          <span className={`text-[11px] font-semibold ${serverOnline ? 'text-primary' : 'text-rose-400'}`}>
            {serverOnline ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>

        {/* GPU VRAM Monitor */}
        {systemInfo && systemInfo.cuda_available && (
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-lg bg-surface-lowest border border-outline-variant/40">
            <HardDrive className="w-3.5 h-3.5 text-secondary" />
            <span className="text-[11px] text-on-surface-variant">GPU VRAM:</span>
            <span className="text-[11px] font-bold text-secondary">
              {(systemInfo.vram_allocated_mb / 1024).toFixed(1)}GB / {(systemInfo.vram_total_mb / 1024).toFixed(0)}GB
            </span>
          </div>
        )}
      </div>
    </header>
  );
};
