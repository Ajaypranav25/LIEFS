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
    <header className="h-14 bg-surface-dim border-b border-surface-container flex items-center justify-between px-4 md:px-6 shrink-0 z-30 font-mono text-xs">
      {/* Left Section: Mobile Menu + View Navigation Switcher */}
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="md:hidden p-1.5 rounded-lg text-on-surface-variant hover:text-primary hover:bg-surface-high transition-colors"
          title="Toggle Navigation"
        >
          <Menu className="w-5 h-5" />
        </button>

        {/* View Mode Switcher Pills */}
        <div className="flex bg-surface-lowest p-1 rounded-xl border border-outline-variant/30">
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
            <span>Computer Benchmarks</span>
            <span className="text-[9px] px-1 py-0.2 rounded bg-primary/20 text-primary font-bold">
              PRO
            </span>
          </button>
        </div>
      </div>

      {/* Right Section: Active Model Pill, Hardware Status & VRAM */}
      <div className="flex items-center gap-3">
        {/* Active Model Selector Pill */}
        <button
          onClick={onOpenModelLoader}
          className="flex items-center gap-2 px-3 py-1 rounded-lg bg-surface-lowest border border-primary/40 hover:border-primary text-on-surface hover:text-primary transition-all cursor-pointer group shadow-glow-cyan/50"
          title="Click to load or switch models"
        >
          <Layers className="w-3.5 h-3.5 text-primary" />
          <span className="text-[11px] font-bold truncate max-w-[130px] sm:max-w-[180px]">
            {modelShortName}
          </span>
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-primary/20 text-primary uppercase font-bold group-hover:bg-primary group-hover:text-on-primary transition-colors hidden sm:inline">
            Switch
          </span>
        </button>

        {/* Host Computer Hardware Pill */}
        {hardware && (
          <div className="hidden lg:flex items-center gap-2 px-3 py-1 rounded-lg bg-surface-lowest border border-outline-variant/40">
            <Cpu className="w-3.5 h-3.5 text-secondary" />
            <span className="text-[11px] text-on-surface-variant">Host:</span>
            <span className="text-[11px] font-bold text-on-surface truncate max-w-[120px]" title={hardware.gpu_name !== 'None (CPU Only)' ? hardware.gpu_name : hardware.cpu_model}>
              {hardware.cuda_available ? hardware.gpu_name.replace('NVIDIA GeForce ', '') : 'CPU'}
            </span>
          </div>
        )}

        {/* GPU VRAM Monitor */}
        {systemInfo && systemInfo.cuda_available && (
          <div className="hidden sm:flex items-center gap-2 px-3 py-1 rounded-lg bg-surface-lowest border border-outline-variant/40">
            <HardDrive className="w-3.5 h-3.5 text-secondary" />
            <span className="text-[11px] text-on-surface-variant">VRAM:</span>
            <span className="text-[11px] font-bold text-secondary">
              {(systemInfo.vram_allocated_mb / 1024).toFixed(1)}GB
            </span>
          </div>
        )}

        {/* Server Online Status LED */}
        <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-lowest border border-outline-variant/40">
          <span
            className={`w-2 h-2 rounded-full ${
              serverOnline ? 'bg-primary status-led animate-pulse' : 'bg-rose-500'
            }`}
          />
          <span className={`text-[11px] font-semibold hidden md:inline ${serverOnline ? 'text-primary' : 'text-rose-400'}`}>
            {serverOnline ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>

        {/* Google OAuth Login / User Profile */}
        <UserAuth />
      </div>
    </header>
  );
};
