import React, { useState, useEffect } from 'react';
import {
  X,
  Cpu,
  Layers,
  Sparkles,
  Search,
  HardDrive,
  Check,
  AlertCircle,
  Loader2,
  Lock,
  ArrowRight,
} from 'lucide-react';
import { fetchModelPresets, loadCustomModel } from '../lib/api';
import type { ModelPreset, ModelArchitectureMetadata } from '../types';

interface ModelLoaderModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentModelName: string;
  onModelLoaded: (meta: ModelArchitectureMetadata) => void;
}

export const ModelLoaderModal: React.FC<ModelLoaderModalProps> = ({
  isOpen,
  onClose,
  currentModelName,
  onModelLoaded,
}) => {
  const [presets, setPresets] = useState<ModelPreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>(currentModelName);
  const [customModelId, setCustomModelId] = useState<string>('');
  const [device, setDevice] = useState<string>('auto');
  const [precision, setPrecision] = useState<string>('float16');
  const [hfToken, setHfToken] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadingStep, setLoadingStep] = useState<string>('');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchModelPresets()
        .then((res) => {
          if (res && res.presets) setPresets(res.presets);
        })
        .catch(console.warn);
      setErrorMsg(null);
      setSelectedPresetId(currentModelName);
    }
  }, [isOpen, currentModelName]);

  if (!isOpen) return null;

  const targetModel = customModelId.trim() || selectedPresetId;

  const handleLoad = async () => {
    if (!targetModel) return;
    setIsLoading(true);
    setErrorMsg(null);
    setLoadingStep('Downloading weights and tokenizer from Hugging Face...');

    try {
      setTimeout(() => {
        setLoadingStep('Allocating tensor memory and loading onto device...');
      }, 1500);

      setTimeout(() => {
        setLoadingStep('Initializing KV-Cache & Paged attention engines...');
      }, 3000);

      const res = await loadCustomModel({
        model_name: targetModel,
        device,
        precision,
        hf_token: hfToken.trim() || undefined,
      });

      if (res && res.metadata) {
        onModelLoaded(res.metadata);
        onClose();
      }
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to load model. Please verify model ID and VRAM limits.');
    } finally {
      setIsLoading(false);
      setLoadingStep('');
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
      <div className="w-full max-w-2xl max-h-[90vh] flex flex-col rounded-xl overflow-hidden border border-outline-variant shadow-card bg-surface-container font-sans text-xs">
        {/* Header */}
        <div className="p-5 border-b border-outline-variant flex items-center justify-between bg-surface-low">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-lg bg-surface-lowest border border-outline-variant text-primary">
              <Layers className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-slate-100 font-sans flex items-center gap-2">
                <span>Model Hub & Custom Importer</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-white/[0.05] text-slate-400 border border-outline-variant">
                  BYOM
                </span>
              </h2>
              <p className="text-xs text-slate-400 font-sans mt-0.5">
                Load any Hugging Face model repo ID or local checkpoint to benchmark
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isLoading}
            className="p-1.5 rounded-lg hover:bg-surface-high text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs">
          {/* Custom Input Field */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1.5">
              <Search className="w-3.5 h-3.5 text-primary" />
              <span>Hugging Face Model ID or Local Path</span>
            </label>
            <div className="relative">
              <input
                type="text"
                value={customModelId}
                onChange={(e) => setCustomModelId(e.target.value)}
                placeholder="e.g. meta-llama/Llama-3.2-1B-Instruct or ./my_local_weights"
                disabled={isLoading}
                className="w-full bg-surface-lowest border border-outline-variant focus:border-primary/50 rounded-lg px-3.5 py-2.5 text-slate-100 placeholder:text-slate-500 outline-none text-xs font-mono transition-all shadow-subtle"
              />
              {customModelId && (
                <button
                  onClick={() => setCustomModelId('')}
                  className="absolute right-3 top-2.5 text-slate-400 hover:text-slate-200 cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Preset Grid */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-medium text-slate-300">
                Popular Benchmark Presets
              </span>
              <span className="text-[10px] text-slate-400">Click to select</span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {presets.map((p) => {
                const isSelected = !customModelId && selectedPresetId === p.id;
                const isCurrent = currentModelName === p.id;
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      setCustomModelId('');
                      setSelectedPresetId(p.id);
                    }}
                    disabled={isLoading}
                    className={`p-3 rounded-lg border text-left transition-all cursor-pointer flex flex-col justify-between gap-1.5 shadow-subtle ${
                      isSelected
                        ? 'bg-surface-high border-primary/50 text-slate-100'
                        : 'bg-surface-lowest hover:bg-surface-high border-outline-variant text-slate-300'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-semibold text-xs text-slate-200 truncate">{p.name}</div>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-surface-container font-mono text-slate-400 border border-outline-variant">
                        {p.size_label}
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400 leading-relaxed line-clamp-2 font-sans">
                      {p.description}
                    </p>
                    <div className="flex items-center justify-between text-[10px] text-slate-400 pt-1 border-t border-outline-variant font-mono">
                      <span>VRAM: ~{p.recommended_vram_mb} MB</span>
                      {isCurrent && (
                        <span className="text-emerald-400 flex items-center gap-1 font-medium font-sans">
                          <Check className="w-3 h-3" /> Active
                        </span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Configuration Options */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-1">
            {/* Compute Device */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-primary" /> Target Device
              </label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                disabled={isLoading}
                className="w-full bg-surface-lowest border border-outline-variant focus:border-primary/50 rounded-lg px-3 py-2 text-slate-200 outline-none cursor-pointer text-xs"
              >
                <option value="auto">Auto (CUDA if available)</option>
                <option value="cuda">CUDA GPU (Accelerated)</option>
                <option value="cpu">CPU (Universal Benchmark)</option>
              </select>
            </div>

            {/* Precision */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-medium text-slate-300 flex items-center gap-1">
                <HardDrive className="w-3.5 h-3.5 text-teal-400" /> Precision (Dtype)
              </label>
              <select
                value={precision}
                onChange={(e) => setPrecision(e.target.value)}
                disabled={isLoading}
                className="w-full bg-surface-lowest border border-outline-variant focus:border-primary/50 rounded-lg px-3 py-2 text-slate-200 outline-none cursor-pointer text-xs"
              >
                <option value="float16">FP16 (Half Precision - Recommended)</option>
                <option value="bfloat16">BF16 (Brain Float 16)</option>
                <option value="float32">FP32 (Full Precision)</option>
              </select>
            </div>
          </div>

          {/* Optional Hugging Face Token */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-medium text-slate-400 flex items-center gap-1">
              <Lock className="w-3.5 h-3.5 text-violet-400" /> Hugging Face Access Token (Optional for gated models)
            </label>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              disabled={isLoading}
              className="w-full bg-surface-lowest border border-outline-variant focus:border-violet-500/50 rounded-lg px-3 py-2 text-slate-200 placeholder:text-slate-500 outline-none text-xs font-mono"
            />
          </div>

          {/* Error Message */}
          {errorMsg && (
            <div className="p-3.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400 flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="leading-relaxed text-[11px]">{errorMsg}</div>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="p-4 rounded-lg bg-surface-lowest border border-primary/40 space-y-2 animate-pulse">
              <div className="flex items-center gap-2 text-primary font-medium text-xs">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Loading Model: {targetModel}</span>
              </div>
              <p className="text-[11px] text-slate-400">{loadingStep}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-outline-variant flex items-center justify-between bg-surface-low text-xs">
          <div className="text-[11px] text-slate-400 truncate max-w-[280px] font-mono">
            Target: <span className="text-slate-200 font-medium">{targetModel}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              disabled={isLoading}
              className="px-3.5 py-1.5 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] text-slate-300 hover:text-white transition-all cursor-pointer font-medium shadow-subtle"
            >
              Cancel
            </button>
            <button
              onClick={handleLoad}
              disabled={isLoading || !targetModel}
              className="px-4 py-1.5 rounded-lg bg-primary hover:bg-sky-300 text-slate-950 font-medium shadow-subtle flex items-center gap-1.5 transition-all disabled:opacity-50 cursor-pointer active:scale-[0.99]"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Loading...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Load Model</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
