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

interface ModelAnalysisViewProps {
  serverOnline: boolean;
  onBackToChat: () => void;
}

export const ModelAnalysisView: React.FC<ModelAnalysisViewProps> = ({
  serverOnline,
  onBackToChat,
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'benchmarks' | 'architecture' | 'history'>('benchmarks');

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-background text-on-background">
      {/* Top Model Analysis Sub-Header */}
      <div className="bg-surface-dim border-b border-surface-container px-4 md:px-8 py-3.5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 shrink-0 z-20">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToChat}
            className="p-1.5 rounded-lg bg-surface-low border border-outline-variant/40 hover:border-primary/50 text-on-surface-variant hover:text-primary transition-all flex items-center gap-1 text-xs font-mono cursor-pointer"
          >
            <ArrowLeft className="w-4 h-4" />
            <span className="hidden sm:inline">Back to Chat</span>
          </button>

          <div className="h-5 w-px bg-surface-container" />

          <div>
            <h2 className="text-base font-bold text-primary font-sans flex items-center gap-2">
              <BarChart2 className="w-4 h-4 text-primary" />
              <span>Model Analysis & Performance Suite</span>
            </h2>
            <p className="text-[11px] text-on-surface-variant font-mono">
              Qwen2.5-0.5B-Instruct • Local PyTorch CUDA Engine Deep-Dive
            </p>
          </div>
        </div>

        {/* Sub-Tab Navigation Switcher */}
        <div className="flex bg-surface-lowest p-1 rounded-xl border border-outline-variant/30 font-mono text-xs">
          <button
            onClick={() => setActiveSubTab('benchmarks')}
            className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer ${
              activeSubTab === 'benchmarks'
                ? 'bg-primary-container text-on-primary-container font-bold shadow-glow-cyan'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Benchmarks</span>
          </button>

          <button
            onClick={() => setActiveSubTab('architecture')}
            className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer ${
              activeSubTab === 'architecture'
                ? 'bg-primary-container text-on-primary-container font-bold shadow-glow-cyan'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <Calculator className="w-3.5 h-3.5" />
            <span>Architecture & Sizing</span>
          </button>

          <button
            onClick={() => setActiveSubTab('history')}
            className={`px-3.5 py-1.5 rounded-lg transition-all flex items-center gap-1.5 cursor-pointer ${
              activeSubTab === 'history'
                ? 'bg-primary-container text-on-primary-container font-bold shadow-glow-cyan'
                : 'text-on-surface-variant hover:text-on-surface'
            }`}
          >
            <Clock className="w-3.5 h-3.5" />
            <span>Run History</span>
          </button>
        </div>
      </div>

      {/* Sub-View Content Viewport */}
      <div className="flex-1 overflow-y-auto scrollbar-hide">
        {activeSubTab === 'benchmarks' && <BenchmarksView serverOnline={serverOnline} />}
        {activeSubTab === 'architecture' && <ArchitectureView />}
        {activeSubTab === 'history' && <HistoryView />}
      </div>
    </div>
  );
};
