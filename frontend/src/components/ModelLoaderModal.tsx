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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-background/80 backdrop-blur-md animate-fade-in">
      <div className="cyber-card w-full max-w-2xl max-h-[90vh] flex flex-col rounded-2xl overflow-hidden border border-primary/30 shadow-glow-cyan bg-surface-dim">
        {/* Header */}
        <div className="p-5 border-b border-surface-container flex items-center justify-between bg-surface-low/80">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-primary/10 border border-primary/30 text-primary">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-on-surface font-sans flex items-center gap-2">
                <span>Model Hub & Custom Importer</span>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-primary/20 text-primary border border-primary/40">
                  Universal BYOM
                </span>
              </h2>
              <p className="text-xs text-on-surface-variant font-mono">
                Load any Hugging Face model repo ID or local checkpoint to benchmark
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isLoading}
            className="p-1.5 rounded-lg hover:bg-surface-lowest text-on-surface-variant hover:text-on-surface transition-colors cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 overflow-y-auto space-y-6 flex-1 text-xs font-mono">
          {/* Custom Input Field */}
          <div className="space-y-2">
            <label className="text-[11px] font-bold text-on-surface flex items-center gap-1.5 uppercase tracking-wider text-primary">
              <Search className="w-3.5 h-3.5" />
              <span>Hugging Face Model ID or Local Path</span>
            </label>
            <div className="relative">
              <input
                type="text"
                value={customModelId}
                onChange={(e) => setCustomModelId(e.target.value)}
                placeholder="e.g. meta-llama/Llama-3.2-1B-Instruct or ./my_local_weights"
                disabled={isLoading}
                className="w-full bg-surface-lowest border border-outline-variant/50 focus:border-primary rounded-xl px-4 py-3 text-on-surface placeholder:text-on-surface-variant/40 outline-none text-xs transition-all shadow-inner"
              />
              {customModelId && (
                <button
                  onClick={() => setCustomModelId('')}
                  className="absolute right-3 top-3 text-on-surface-variant hover:text-on-surface cursor-pointer"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>
          </div>

          {/* Preset Grid */}
          <div className="space-y-2.5">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold text-on-surface uppercase tracking-wider text-secondary">
                Popular Benchmark Presets
              </span>
              <span className="text-[10px] text-on-surface-variant">Click to quick-select</span>
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
                    className={`p-3 rounded-xl border text-left transition-all cursor-pointer flex flex-col justify-between gap-1.5 ${
                      isSelected
                        ? 'bg-primary/10 border-primary shadow-glow-cyan'
                        : 'bg-surface-lowest/70 border-outline-variant/30 hover:border-primary/40 hover:bg-surface-low'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-bold text-xs text-on-surface truncate">{p.name}</div>
                      <span className="text-[9px] px-1.5 py-0.5 rounded bg-surface-container font-mono text-primary font-bold">
                        {p.size_label}
                      </span>
                    </div>
                    <p className="text-[10px] text-on-surface-variant leading-relaxed line-clamp-2">
                      {p.description}
                    </p>
                    <div className="flex items-center justify-between text-[10px] text-on-surface-variant/70 pt-1 border-t border-outline-variant/20">
                      <span>VRAM: ~{p.recommended_vram_mb} MB</span>
                      {isCurrent && (
                        <span className="text-primary flex items-center gap-1 font-bold">
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
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pt-2">
            {/* Compute Device */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-bold text-on-surface flex items-center gap-1">
                <Cpu className="w-3.5 h-3.5 text-primary" /> Target Device
              </label>
              <select
                value={device}
                onChange={(e) => setDevice(e.target.value)}
                disabled={isLoading}
                className="w-full bg-surface-lowest border border-outline-variant/50 focus:border-primary rounded-xl px-3 py-2 text-on-surface outline-none cursor-pointer"
              >
                <option value="auto">Auto (CUDA if available)</option>
                <option value="cuda">CUDA GPU (Accelerated)</option>
                <option value="cpu">CPU (Universal Benchmark)</option>
              </select>
            </div>

            {/* Precision */}
            <div className="space-y-1.5">
              <label className="text-[11px] font-bold text-on-surface flex items-center gap-1">
                <HardDrive className="w-3.5 h-3.5 text-secondary" /> Precision (Dtype)
              </label>
              <select
                value={precision}
                onChange={(e) => setPrecision(e.target.value)}
                disabled={isLoading}
                className="w-full bg-surface-lowest border border-outline-variant/50 focus:border-primary rounded-xl px-3 py-2 text-on-surface outline-none cursor-pointer"
              >
                <option value="float16">FP16 (Half Precision - Recommended)</option>
                <option value="bfloat16">BF16 (Brain Float 16)</option>
                <option value="float32">FP32 (Full Precision)</option>
              </select>
            </div>
          </div>

          {/* Optional Hugging Face Token */}
          <div className="space-y-1.5">
            <label className="text-[11px] font-bold text-on-surface-variant flex items-center gap-1">
              <Lock className="w-3.5 h-3.5 text-tertiary" /> Hugging Face Access Token (Optional for gated models)
            </label>
            <input
              type="password"
              value={hfToken}
              onChange={(e) => setHfToken(e.target.value)}
              placeholder="hf_..."
              disabled={isLoading}
              className="w-full bg-surface-lowest border border-outline-variant/40 focus:border-tertiary rounded-xl px-3 py-2 text-on-surface placeholder:text-on-surface-variant/40 outline-none text-xs"
            />
          </div>

          {/* Error Message */}
          {errorMsg && (
            <div className="p-3.5 rounded-xl bg-error/10 border border-error/40 text-error flex items-start gap-2.5 animate-shake">
              <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="leading-relaxed text-[11px]">{errorMsg}</div>
            </div>
          )}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="p-4 rounded-xl bg-surface-lowest border border-primary/40 space-y-2 animate-pulse">
              <div className="flex items-center gap-2 text-primary font-bold text-xs">
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Loading Model: {targetModel}</span>
              </div>
              <p className="text-[11px] text-on-surface-variant">{loadingStep}</p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-surface-container flex items-center justify-between bg-surface-low/80 font-mono text-xs">
          <div className="text-[11px] text-on-surface-variant truncate max-w-[280px]">
            Target: <span className="text-primary font-bold">{targetModel}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              disabled={isLoading}
              className="px-4 py-2 rounded-xl bg-surface-lowest hover:bg-surface-container text-on-surface-variant hover:text-on-surface transition-all cursor-pointer"
            >
              Cancel
            </button>
            <button
              onClick={handleLoad}
              disabled={isLoading || !targetModel}
              className="px-5 py-2 rounded-xl bg-primary text-on-primary font-bold hover:brightness-110 shadow-glow-cyan flex items-center gap-1.5 transition-all disabled:opacity-50 cursor-pointer"
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
