import React from 'react';
import {
  Cpu,
  Zap,
  Layers,
  Sparkles,
} from 'lucide-react';
import type { HardwareProfile, ModelArchitectureMetadata } from '../types';

interface HardwareProfileHUDProps {
  hardware: HardwareProfile | null;
  modelMeta: ModelArchitectureMetadata | null;
  onOpenModelLoader: () => void;
}

export const HardwareProfileHUD: React.FC<HardwareProfileHUDProps> = ({
  hardware,
  modelMeta,
  onOpenModelLoader,
}) => {
  if (!hardware) return null;

  const vramPct = hardware.vram_usage_percent || 0;
  const ramPct = hardware.ram_usage_percent || 0;

  return (
    <div className="p-4 rounded-xl border border-outline-variant bg-surface-container grid grid-cols-1 md:grid-cols-4 gap-4 font-mono text-xs shadow-subtle">
      {/* 1. Computer CPU Info */}
      <div className="flex items-start gap-3 border-b md:border-b-0 md:border-r border-outline-variant pb-3 md:pb-0 md:pr-3">
        <div className="p-2 rounded-lg bg-surface-lowest border border-outline-variant text-primary shrink-0">
          <Cpu className="w-4 h-4" />
        </div>
        <div className="space-y-1 min-w-0 flex-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-sans">
            Host Processor
          </div>
          <div className="font-semibold text-slate-200 text-[11px] truncate font-sans" title={hardware.cpu_model}>
            {hardware.cpu_model}
          </div>
          <div className="text-[10px] text-slate-400 flex items-center justify-between tabular-nums">
            <span>{hardware.cpu_physical_cores}C / {hardware.cpu_logical_cores}T</span>
            <span>RAM: {hardware.ram_total_gb} GB ({ramPct}%)</span>
          </div>
          {/* RAM Meter */}
          <div className="w-full bg-surface-lowest h-1 rounded-full overflow-hidden mt-1 border border-outline-variant">
            <div
              className="h-full bg-sky-400 transition-all duration-500 rounded-full"
              style={{ width: `${Math.min(100, Math.max(5, ramPct))}%` }}
            />
          </div>
        </div>
      </div>

      {/* 2. Computer GPU / VRAM Info */}
      <div className="flex items-start gap-3 border-b md:border-b-0 md:border-r border-outline-variant pb-3 md:pb-0 md:pr-3">
        <div className="p-2 rounded-lg bg-surface-lowest border border-outline-variant text-secondary shrink-0">
          <Zap className="w-4 h-4" />
        </div>
        <div className="space-y-1 min-w-0 flex-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-sans">
            GPU Accelerator
          </div>
          <div className="font-semibold text-slate-200 text-[11px] truncate font-sans" title={hardware.gpu_name}>
            {hardware.gpu_name}
          </div>
          <div className="text-[10px] text-slate-400 flex items-center justify-between tabular-nums">
            <span>CUDA: {hardware.cuda_version || 'N/A'}</span>
            <span>VRAM: {Math.round(hardware.vram_allocated_mb)} / {Math.round(hardware.vram_total_mb)} MB</span>
          </div>
          {/* VRAM Meter */}
          <div className="w-full bg-surface-lowest h-1 rounded-full overflow-hidden mt-1 border border-outline-variant">
            <div
              className={`h-full transition-all duration-500 rounded-full ${
                vramPct > 85 ? 'bg-rose-400' : vramPct > 60 ? 'bg-amber-400' : 'bg-teal-400'
              }`}
              style={{ width: `${Math.min(100, Math.max(5, vramPct))}%` }}
            />
          </div>
        </div>
      </div>

      {/* 3. Loaded Model Specs */}
      <div className="flex items-start gap-3 border-b md:border-b-0 md:border-r border-outline-variant pb-3 md:pb-0 md:pr-3">
        <div className="p-2 rounded-lg bg-surface-lowest border border-outline-variant text-tertiary shrink-0">
          <Layers className="w-4 h-4" />
        </div>
        <div className="space-y-1 min-w-0 flex-1">
          <div className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-sans">
            Active Model Specs
          </div>
          <div className="font-semibold text-slate-200 text-[11px] truncate" title={modelMeta?.model_name || 'Qwen2.5-0.5B'}>
            {modelMeta?.model_name.split('/').pop() || 'Qwen2.5-0.5B'}
          </div>
          <div className="text-[10px] text-slate-400 flex items-center justify-between tabular-nums">
            <span>{modelMeta ? `${modelMeta.parameter_count_m}M Params` : '~494M Params'}</span>
            <span>{modelMeta?.num_layers || 24} Layers</span>
          </div>
          <div className="text-[10px] text-slate-400 truncate">
            {modelMeta ? `${modelMeta.num_attention_heads} Q / ${modelMeta.num_kv_heads} KV Heads` : 'GQA 14:2 Heads'}
          </div>
        </div>
      </div>

      {/* 4. Action / Switch Model */}
      <div className="flex flex-col justify-center items-stretch sm:items-end gap-2">
        <button
          onClick={onOpenModelLoader}
          className="w-full sm:w-auto px-3.5 py-1.5 rounded-lg bg-white/[0.05] hover:bg-white/[0.09] border border-outline-variant hover:border-white/[0.16] text-slate-200 hover:text-white transition-all font-sans font-medium text-xs flex items-center justify-center gap-1.5 cursor-pointer shadow-subtle"
        >
          <Sparkles className="w-3.5 h-3.5 text-primary" />
          <span>Load / Switch Model</span>
        </button>
        <span className="text-[10px] text-slate-400 text-center sm:text-right font-mono">
          Device: <span className="text-slate-200 font-medium">{hardware.cuda_available ? 'CUDA' : 'CPU'}</span> ({modelMeta?.dtype_str || 'FP16'})
        </span>
      </div>
    </div>
  );
};
