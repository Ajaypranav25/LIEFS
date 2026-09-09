import React, { useState } from 'react';
import {
  BarChart2,
  Zap,
  Calculator,
  Clock,
  ArrowLeft,
} from 'lucide-react';
import { BenchmarksView } from './BenchmarksView';
import { ArchitectureView } from './ArchitectureView';
import { HistoryView } from './HistoryView';
import { HardwareProfileHUD } from './HardwareProfileHUD';
import type { HardwareProfile, ModelArchitectureMetadata } from '../types';

interface ModelAnalysisViewProps {
  serverOnline: boolean;
  hardware: HardwareProfile | null;
  modelMeta: ModelArchitectureMetadata | null;
  onBackToChat: () => void;
  onOpenModelLoader: () => void;
}

export const ModelAnalysisView: React.FC<ModelAnalysisViewProps> = ({
  serverOnline,
  hardware,
  modelMeta,
  onBackToChat,
  onOpenModelLoader,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'benchmarks' | 'architecture' | 'history'>('benchmarks');

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-background text-on-background">
      {/* Top Model Analysis Sub-Header */}
      <div className="bg-surface-dim border-b border-outline-variant px-4 md:px-8 py-3 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shrink-0 z-20 font-sans text-xs">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToChat}
            className="px-2.5 py-1.5 rounded-lg bg-surface-lowest hover:bg-surface-container border border-outline-variant hover:border-white/[0.16] text-slate-300 hover:text-white transition-all flex items-center gap-1.5 text-xs font-sans cursor-pointer shadow-subtle"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">Back to Chat</span>
          </button>

          <div className="h-4 w-px bg-outline-variant" />

          <div>
            <h2 className="text-sm font-semibold text-slate-100 font-sans flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-primary" />
              <span>Inference Benchmarks & Architecture</span>
            </h2>
            <p className="text-[11px] text-slate-400 font-mono truncate max-w-[320px] sm:max-w-none">
              {modelMeta?.model_name || 'Qwen2.5-0.5B'} • {hardware?.cpu_model || 'Host Machine'}
            </p>
          </div>
        </div>

        {/* Sub-Tab Navigation Switcher */}
        <div className="flex bg-surface-lowest p-0.5 rounded-lg border border-outline-variant shadow-subtle font-sans text-xs">
          <button
            onClick={() => setActiveSubTab('benchmarks')}
            className={`px-3 py-1 rounded-md transition-all flex items-center gap-1.5 cursor-pointer ${
              activeSubTab === 'benchmarks'
                ? 'bg-surface-container text-slate-100 font-medium shadow-subtle border border-outline-variant'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            <Zap className="w-3.5 h-3.5 text-primary" />
            <span>Hardware Benchmarks</span>
          </button>

          <button
            onClick={() => setActiveSubTab('architecture')}
            className={`px-3 py-1 rounded-md transition-all flex items-center gap-1.5 cursor-pointer ${
              activeSubTab === 'architecture'
                ? 'bg-surface-container text-slate-100 font-medium shadow-subtle border border-outline-variant'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            <Calculator className="w-3.5 h-3.5 text-primary" />
            <span>Architecture & Sizing</span>
          </button>

          <button
            onClick={() => setActiveSubTab('history')}
            className={`px-3 py-1 rounded-md transition-all flex items-center gap-1.5 cursor-pointer ${
              activeSubTab === 'history'
                ? 'bg-surface-container text-slate-100 font-medium shadow-subtle border border-outline-variant'
                : 'text-slate-400 hover:text-slate-200 border border-transparent'
            }`}
          >
            <Clock className="w-3.5 h-3.5 text-primary" />
            <span>Run History</span>
          </button>
        </div>
      </div>

      {/* Sub-View Content Viewport */}
      <div className="flex-1 overflow-y-auto scrollbar-hide space-y-4 p-4 md:p-6">
        {/* Persistent Hardware Profile HUD */}
        <HardwareProfileHUD
          hardware={hardware}
          modelMeta={modelMeta}
          onOpenModelLoader={onOpenModelLoader}
        />

        {activeSubTab === 'benchmarks' && (
          <BenchmarksView
            serverOnline={serverOnline}
            onOpenModelLoader={onOpenModelLoader}
          />
        )}
        {activeSubTab === 'architecture' && <ArchitectureView />}
        {activeSubTab === 'history' && <HistoryView />}
      </div>
    </div>
  );
};
