import React from 'react';
import {
  Menu,
  BarChart2,
  MessageSquare,
  HardDrive,
  Cpu,
  Layers,
} from 'lucide-react';
import { UserAuth } from './UserAuth';
import type { SystemInfo, ModelArchitectureMetadata, HardwareProfile } from '../types';

interface HeaderProps {
  activeView: 'chat' | 'analysis';
  setActiveView: (view: 'chat' | 'analysis') => void;
  systemInfo: SystemInfo | null;
  hardware: HardwareProfile | null;
  modelMeta: ModelArchitectureMetadata | null;
  serverOnline: boolean;
  onToggleSidebar: () => void;
  onOpenModelLoader: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeView,
  setActiveView,
  systemInfo,
  hardware,
  modelMeta,
  serverOnline,
  onToggleSidebar,
  onOpenModelLoader,
}) => {
  const modelShortName = modelMeta?.model_name.split('/').pop() || systemInfo?.model_name.split('/').pop() || 'Qwen2.5-0.5B';

  return (
    <header className="h-13 bg-surface-dim border-b border-outline-variant flex items-center justify-between px-4 md:px-6 shrink-0 z-30 font-sans text-xs">
      {/* Left Section: Mobile Menu + View Navigation Switcher */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="md:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-surface-high transition-colors"
          title="Toggle Navigation"
        >
          <Menu className="w-4 h-4" />
        </button>

        {/* View Mode Switcher: Apple/Linear-style Segmented Control */}
        <div className="flex bg-surface-lowest p-0.5 rounded-lg border border-outline-variant shadow-subtle">
          <button
            onClick={() => setActiveView('chat')}
            className={`px-3 py-1 rounded-md flex items-center gap-1.5 transition-all text-xs cursor-pointer ${
              activeView === 'chat'
                ? 'bg-surface-container text-slate-100 font-medium shadow-subtle border border-outline-variant'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            <MessageSquare className="w-3.5 h-3.5 text-primary" />
            <span>Chat</span>
          </button>

          <button
            onClick={() => setActiveView('analysis')}
            className={`px-3 py-1 rounded-md flex items-center gap-1.5 transition-all text-xs cursor-pointer ${
              activeView === 'analysis'
                ? 'bg-surface-container text-slate-100 font-medium shadow-subtle border border-outline-variant'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            <BarChart2 className="w-3.5 h-3.5 text-primary" />
            <span>Benchmarks</span>
          </button>
        </div>
      </div>

      {/* Right Section: Active Model Pill, Hardware Status & VRAM */}
      <div className="flex items-center gap-2.5">
        {/* Active Model Selector Pill */}
        <button
          onClick={onOpenModelLoader}
          className="flex items-center gap-2 px-2.5 py-1 rounded-lg bg-surface-lowest hover:bg-surface-container border border-outline-variant hover:border-white/[0.16] text-slate-300 hover:text-slate-100 transition-all cursor-pointer shadow-subtle group"
          title="Click to load or switch models"
        >
          <Layers className="w-3.5 h-3.5 text-primary" />
          <span className="text-[11px] font-mono font-medium truncate max-w-[130px] sm:max-w-[180px]">
            {modelShortName}
          </span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-white/[0.06] text-slate-400 font-mono group-hover:text-slate-200 transition-colors hidden sm:inline">
            Switch
          </span>
        </button>

        {/* Host Computer Hardware Pill */}
        {hardware && (
          <div className="hidden lg:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-surface-lowest border border-outline-variant">
            <Cpu className="w-3.5 h-3.5 text-secondary" />
            <span className="text-[11px] text-slate-400">Host:</span>
            <span className="text-[11px] font-medium text-slate-200 truncate max-w-[120px]" title={hardware.gpu_name !== 'None (CPU Only)' ? hardware.gpu_name : hardware.cpu_model}>
              {hardware.cuda_available ? hardware.gpu_name.replace('NVIDIA GeForce ', '') : 'CPU'}
            </span>
          </div>
        )}

        {/* GPU VRAM Monitor */}
        {systemInfo && systemInfo.cuda_available && (
          <div className="hidden sm:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-surface-lowest border border-outline-variant">
            <HardDrive className="w-3.5 h-3.5 text-secondary" />
            <span className="text-[11px] text-slate-400">VRAM:</span>
            <span className="text-[11px] font-mono font-medium text-slate-200 tabular-nums">
              {(systemInfo.vram_allocated_mb / 1024).toFixed(1)}GB
            </span>
          </div>
        )}

        {/* Server Online Status Indicator */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-lowest border border-outline-variant">
          <span
            className={`w-2 h-2 rounded-full ${
              serverOnline ? 'bg-emerald-400 ring-2 ring-emerald-400/20' : 'bg-rose-500 ring-2 ring-rose-500/20'
            }`}
          />
          <span className={`text-[11px] font-medium hidden md:inline ${serverOnline ? 'text-slate-300' : 'text-rose-400'}`}>
            {serverOnline ? 'Online' : 'Offline'}
          </span>
        </div>

        {/* Google OAuth Login / User Profile */}
        <UserAuth />
      </div>
    </header>
  );
};
