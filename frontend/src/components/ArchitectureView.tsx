import React, { useState } from 'react';
import {
  Layers,
  Calculator,
  Sparkles,
  Zap,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';

export const ArchitectureView: React.FC = () => {
  // Calculator state
  const [contextLength, setContextLength] = useState<number>(2048);
  const [batchSize, setBatchSize] = useState<number>(16);
  const [numLayers, setNumLayers] = useState<number>(24);
  const [numKVHeads, setNumKVHeads] = useState<number>(2); // GQA for Qwen2.5-0.5B
  const [headDim, setHeadDim] = useState<number>(64);
  const [precisionBytes, setPrecisionBytes] = useState<number>(2); // 2 bytes for FP16, 1 byte for INT8
  const gpuVramGb = 16;

  // KV Cache Formula:
  // KV_size_per_token = 2 (for K and V) * num_layers * num_kv_heads * head_dim * precision_bytes
  const bytesPerToken = 2 * numLayers * numKVHeads * headDim * precisionBytes;
  const memoryPerReqMb = (bytesPerToken * contextLength) / (1024 * 1024);
  const totalKVCacheMb = memoryPerReqMb * batchSize;
  const totalKVCacheGb = totalKVCacheMb / 1024;
  const maxConcurrentRequests = Math.floor((gpuVramGb * 1024 * 0.7) / memoryPerReqMb); // Assuming 70% available for KV

  return (
    <div className="pt-20 px-4 sm:px-8 max-w-[1440px] mx-auto pb-12 space-y-8 font-mono">
      {/* Header */}
      <div className="cyber-card p-6 rounded-lg">
        <h2 className="text-lg font-bold text-primary flex items-center gap-2">
          <Layers className="w-5 h-5 text-primary" />
          LIEFS Architectural Deep-Dive & Mathematical Foundations
        </h2>
        <p className="text-xs text-on-surface-variant mt-1 max-w-3xl">
          Understanding the core engineering mechanisms that transform naive autoregressive generation into a production-grade inference engine (KV-Caching, Continuous Batching, Paged Attention, and INT8 Quantization).
        </p>
      </div>

      {/* 5 Stage Deep-Dive Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Stage 1: Naive */}
        <div className="cyber-card p-5 rounded-lg border border-rose-500/30 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                Stage 1: Naive Baseline
              </span>
              <span className="text-xs text-rose-400 font-semibold">O(N²) Cost</span>
            </div>
            <h3 className="text-sm font-bold text-slate-100 mb-2 font-sans">
              Full Sequence Recomputation
            </h3>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              At every token step <code className="text-rose-300">t</code>, the entire sequence of <code className="text-slate-300">(P + t)</code> tokens is fed forward through all 24 transformer layers.
            </p>
            <div className="mt-3 p-2.5 rounded bg-[#020617] border border-slate-800 text-[11px] text-on-surface space-y-1">
              <div className="text-rose-400 font-semibold">Quadratic Attention Flops:</div>
              <div>Total Compute ≈ Σ(P + t)²</div>
              <div className="text-slate-500 text-[10px]">Linear projections recomputed redundantly every step.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center gap-1.5 text-xs text-rose-400">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Severe latency scaling degradation</span>
          </div>
        </div>

        {/* Stage 2: KV Cache */}
        <div className="cyber-card p-5 rounded-lg border border-primary/40 flex flex-col justify-between shadow-glow-active">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-primary/20 text-primary border border-primary/40">
                Stage 2: KV-Cache
              </span>
              <span className="text-xs text-primary font-semibold">O(N) Decode</span>
            </div>
            <h3 className="text-sm font-bold text-slate-100 mb-2 font-sans">
              Prefill + Decode Separation
            </h3>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              Computes Key and Value tensors for historical tokens once and caches them. Each decode step only processes <code className="text-primary">1 token</code> through the network.
            </p>
            <div className="mt-3 p-2.5 rounded bg-[#020617] border border-slate-800 text-[11px] text-on-surface space-y-1">
              <div className="text-primary font-semibold">Linear Attention Cost:</div>
              <div>Prefill: O(P²) once → Decode: O(1) per step</div>
              <div className="text-slate-500 text-[10px]">Eliminates Q/K/V/MLP redundant projections.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center gap-1.5 text-xs text-primary">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>~4x to 10.7x speedup vs Naive</span>
          </div>
        </div>

        {/* Stage 3: Continuous Batching */}
        <div className="cyber-card p-5 rounded-lg border border-amber-500/30 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                Stage 3: Cont. Batching
              </span>
              <span className="text-xs text-amber-400 font-semibold">Iteration Scheduling</span>
            </div>
            <h3 className="text-sm font-bold text-slate-100 mb-2 font-sans">
              Zero Pipeline Bubbles
            </h3>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              Instead of waiting for the longest sequence in a batch to finish (static batching), new prompts enter the active forward pass at the very next token iteration.
            </p>
            <div className="mt-3 p-2.5 rounded bg-[#020617] border border-slate-800 text-[11px] text-on-surface space-y-1">
              <div className="text-amber-400 font-semibold">Multi-Tenant Throughput:</div>
              <div>Saturates GPU tensor cores continuously</div>
              <div className="text-slate-500 text-[10px]">Up to 39x aggregate throughput under load.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center gap-1.5 text-xs text-amber-400">
            <Zap className="w-3.5 h-3.5" />
            <span>High multi-user serving efficiency</span>
          </div>
        </div>

        {/* Stage 4: Paged Attention */}
        <div className="cyber-card p-5 rounded-lg border border-secondary/30 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-secondary/20 text-secondary border border-secondary/30">
                Stage 4: Paged Attention
              </span>
              <span className="text-xs text-secondary font-semibold">Virtual Memory Pool</span>
            </div>
            <h3 className="text-sm font-bold text-slate-100 mb-2 font-sans">
              BlockAllocator Pool (16 tok/block)
            </h3>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              Eliminates contiguous VRAM pre-allocation waste and external fragmentation by allocating KV-cache tensors in fixed 16-token virtual memory blocks.
            </p>
            <div className="mt-3 p-2.5 rounded bg-[#020617] border border-slate-800 text-[11px] text-on-surface space-y-1">
              <div className="text-secondary font-semibold">Zero Memory Fragmentation:</div>
              <div>Block table maps logical tokens → physical blocks</div>
              <div className="text-slate-500 text-[10px]">Supports copy-on-write branching & prefix caching.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center gap-1.5 text-xs text-secondary">
            <HardDrive className="w-3.5 h-3.5" />
            <span>Optimal VRAM packing</span>
          </div>
        </div>

        {/* Stage 5: Quantization */}
        <div className="cyber-card p-5 rounded-lg border border-tertiary/30 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-bold px-2 py-0.5 rounded bg-tertiary/20 text-tertiary border border-tertiary/30">
                Stage 5: INT8 Quantized
              </span>
              <span className="text-xs text-tertiary font-semibold">Symmetric Linear</span>
            </div>
            <h3 className="text-sm font-bold text-slate-100 mb-2 font-sans">
              Per-Channel INT8 Weights
            </h3>
            <p className="text-xs text-on-surface-variant leading-relaxed">
              Weights quantized to 8-bit integers with per-channel scale factors: <code className="text-tertiary">W_int8 = round(W / scale)</code>. Dequantized on the fly during forward pass.
            </p>
            <div className="mt-3 p-2.5 rounded bg-[#020617] border border-slate-800 text-[11px] text-on-surface space-y-1">
              <div className="text-tertiary font-semibold">50% Weight VRAM Cut:</div>
              <div>Dequantize: W_fp16 = W_int8 * scale</div>
              <div className="text-slate-500 text-[10px]">Preserves lm_head to protect generation quality.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center gap-1.5 text-xs text-tertiary">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Fits larger models on smaller GPUs</span>
          </div>
        </div>
      </div>

      {/* Interactive KV-Cache Memory Sizing Calculator */}
      <div className="cyber-card p-6 rounded-lg">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-base font-bold text-primary flex items-center gap-2">
              <Calculator className="w-5 h-5 text-primary" />
              Interactive KV-Cache GPU Memory Sizing Calculator
            </h3>
            <p className="text-xs text-on-surface-variant mt-1">
              Formula: <code className="text-primary font-mono">VRAM_KV = 2 × N_layers × N_kv_heads × d_head × precision_bytes × Seq_len × Batch_size</code>
            </p>
          </div>

          {/* Model Preset Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-on-surface-variant">Model Preset:</span>
            <button
              onClick={() => {
                setNumLayers(24);
                setNumKVHeads(2);
                setHeadDim(64);
              }}
              className="px-2.5 py-1 rounded bg-[#020617] border border-primary/40 text-primary text-xs font-medium hover:bg-primary/10 transition-all cursor-pointer"
            >
              Qwen2.5-0.5B (GQA)
            </button>
            <button
              onClick={() => {
                setNumLayers(32);
                setNumKVHeads(8);
                setHeadDim(128);
              }}
              className="px-2.5 py-1 rounded bg-[#020617] border border-slate-800 text-on-surface-variant text-xs font-medium hover:text-on-surface transition-all cursor-pointer"
            >
              Llama-3-8B (GQA)
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          {/* Controls (7 cols) */}
          <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Context Length */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-on-surface-variant">Context Length (Tokens)</span>
                <span className="text-primary font-semibold">{contextLength.toLocaleString()}</span>
              </div>
              <input
                type="range"
                min="512"
                max="32768"
                step="512"
                value={contextLength}
                onChange={(e) => setContextLength(Number(e.target.value))}
                className="w-full accent-primary bg-slate-800 h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Batch Size */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-on-surface-variant">Batch Size (Concurrent Streams)</span>
                <span className="text-secondary font-semibold">{batchSize}</span>
              </div>
              <input
                type="range"
                min="1"
                max="64"
                step="1"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full accent-secondary bg-slate-800 h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Layers */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-on-surface-variant">Transformer Layers</span>
                <span className="text-on-surface font-semibold">{numLayers}</span>
              </div>
              <input
                type="range"
                min="12"
                max="80"
                step="2"
                value={numLayers}
                onChange={(e) => setNumLayers(Number(e.target.value))}
                className="w-full accent-slate-400 bg-slate-800 h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* KV Heads (GQA) */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <span className="text-on-surface-variant">KV Attention Heads (GQA)</span>
                <span className="text-on-surface font-semibold">{numKVHeads}</span>
              </div>
              <input
                type="range"
                min="1"
                max="32"
                step="1"
                value={numKVHeads}
                onChange={(e) => setNumKVHeads(Number(e.target.value))}
                className="w-full accent-slate-400 bg-slate-800 h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Precision */}
            <div className="space-y-1 sm:col-span-2">
              <span className="text-xs text-on-surface-variant block mb-1">KV-Cache Quantization Precision</span>
              <div className="flex gap-2">
                {[
                  { label: 'FP16 (2 bytes)', bytes: 2 },
                  { label: 'INT8 (1 byte)', bytes: 1 },
                  { label: 'FP8 (1 byte)', bytes: 1 },
                ].map((p, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => setPrecisionBytes(p.bytes)}
                    className={`flex-1 py-1.5 rounded border text-xs transition-all ${
                      precisionBytes === p.bytes
                        ? 'bg-primary/20 text-primary border-primary/50 shadow-glow-active font-semibold'
                        : 'bg-[#020617] border-slate-800 text-on-surface-variant hover:text-on-surface'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Card HUD (5 cols) */}
          <div className="lg:col-span-5 bg-[#020617] p-5 rounded border border-slate-800 flex flex-col justify-between space-y-4">
            <div>
              <span className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold block mb-3">
                Calculated KV Memory Demand
              </span>

              <div className="space-y-3">
                <div className="flex items-baseline justify-between border-b border-slate-800/80 pb-2">
                  <span className="text-xs text-on-surface-variant">Memory Per Request:</span>
                  <span className="text-base font-bold text-on-surface">
                    {memoryPerReqMb.toFixed(2)} MB
                  </span>
                </div>

                <div className="flex items-baseline justify-between border-b border-slate-800/80 pb-2">
                  <span className="text-xs text-on-surface-variant">Total KV Cache (Batch={batchSize}):</span>
                  <span className="text-lg font-bold text-primary">
                    {totalKVCacheGb >= 1.0 ? `${totalKVCacheGb.toFixed(2)} GB` : `${totalKVCacheMb.toFixed(1)} MB`}
                  </span>
                </div>

                <div className="flex items-baseline justify-between border-b border-slate-800/80 pb-2">
                  <span className="text-xs text-on-surface-variant">Max Concurrency ({gpuVramGb}GB GPU):</span>
                  <span className="text-base font-bold text-secondary">
                    ~{maxConcurrentRequests} streams
                  </span>
                </div>
              </div>
            </div>

            <div className="p-3 rounded bg-primary/5 border border-primary/20 text-[11px] text-primary">
              💡 GQA (Grouped Query Attention) with 2 KV heads reduces KV memory by <strong className="text-white">7x</strong> compared to full MHA (14 heads).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
