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
    <div className="pt-6 px-4 sm:px-8 max-w-[1440px] mx-auto pb-12 space-y-6 font-sans">
      {/* Header */}
      <div className="p-5 rounded-xl border border-outline-variant bg-surface-container shadow-subtle">
        <h2 className="text-base font-semibold text-slate-100 flex items-center gap-2">
          <Layers className="w-4 h-4 text-primary" />
          LIEFS Architectural Deep-Dive & Mathematical Foundations
        </h2>
        <p className="text-xs text-slate-400 mt-1 max-w-3xl leading-relaxed">
          Understanding the core engineering mechanisms that transform naive autoregressive generation into a production-grade inference engine (KV-Caching, Continuous Batching, Paged Attention, and INT8 Quantization).
        </p>
      </div>

      {/* 5 Stage Deep-Dive Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {/* Stage 1: Naive */}
        <div className="p-5 rounded-xl border border-outline-variant bg-surface-container flex flex-col justify-between shadow-subtle">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/20">
                Stage 1: Naive Baseline
              </span>
              <span className="text-xs font-mono text-rose-400">O(N²) Cost</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-100 mb-1.5 font-sans">
              Full Sequence Recomputation
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              At every token step <code className="text-rose-400 font-mono">t</code>, the entire sequence of <code className="text-slate-300 font-mono">(P + t)</code> tokens is fed forward through all 24 transformer layers.
            </p>
            <div className="mt-3 p-3 rounded-lg bg-surface-lowest border border-outline-variant text-xs text-slate-300 space-y-1 font-mono">
              <div className="text-rose-400 font-medium">Quadratic Attention Flops:</div>
              <div>Total Compute ≈ Σ(P + t)²</div>
              <div className="text-slate-500 text-[10px] font-sans">Linear projections recomputed redundantly every step.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-outline-variant flex items-center gap-1.5 text-xs text-rose-400 font-medium">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Severe latency scaling degradation</span>
          </div>
        </div>

        {/* Stage 2: KV Cache */}
        <div className="p-5 rounded-xl border border-outline-variant bg-surface-container flex flex-col justify-between shadow-subtle">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-sky-500/10 text-sky-400 border border-sky-500/20">
                Stage 2: KV-Cache
              </span>
              <span className="text-xs font-mono text-sky-400">O(N) Decode</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-100 mb-1.5 font-sans">
              Prefill + Decode Separation
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Computes Key and Value tensors for historical tokens once and caches them. Each decode step only processes <code className="text-sky-400 font-mono">1 token</code> through the network.
            </p>
            <div className="mt-3 p-3 rounded-lg bg-surface-lowest border border-outline-variant text-xs text-slate-300 space-y-1 font-mono">
              <div className="text-sky-400 font-medium">Linear Attention Cost:</div>
              <div>Prefill: O(P²) once → Decode: O(1) per step</div>
              <div className="text-slate-500 text-[10px] font-sans">Eliminates redundant linear projections.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-outline-variant flex items-center gap-1.5 text-xs text-sky-400 font-medium">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>~4x to 10.7x speedup vs Naive</span>
          </div>
        </div>

        {/* Stage 3: Continuous Batching */}
        <div className="p-5 rounded-xl border border-outline-variant bg-surface-container flex flex-col justify-between shadow-subtle">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                Stage 3: Cont. Batching
              </span>
              <span className="text-xs font-mono text-amber-400">Iteration Scheduling</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-100 mb-1.5 font-sans">
              Zero Pipeline Bubbles
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Instead of waiting for the longest sequence in a batch to finish (static batching), new prompts enter the active forward pass at the very next token iteration.
            </p>
            <div className="mt-3 p-3 rounded-lg bg-surface-lowest border border-outline-variant text-xs text-slate-300 space-y-1 font-mono">
              <div className="text-amber-400 font-medium">Multi-Tenant Throughput:</div>
              <div>Saturates GPU tensor cores continuously</div>
              <div className="text-slate-500 text-[10px] font-sans">Up to 39x aggregate throughput under load.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-outline-variant flex items-center gap-1.5 text-xs text-amber-400 font-medium">
            <Zap className="w-3.5 h-3.5" />
            <span>High multi-user serving efficiency</span>
          </div>
        </div>

        {/* Stage 4: Paged Attention */}
        <div className="p-5 rounded-xl border border-outline-variant bg-surface-container flex flex-col justify-between shadow-subtle">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                Stage 4: Paged Attention
              </span>
              <span className="text-xs font-mono text-teal-400">Virtual Memory Pool</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-100 mb-1.5 font-sans">
              BlockAllocator Pool (16 tok/block)
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Eliminates contiguous VRAM pre-allocation waste and external fragmentation by allocating KV-cache tensors in fixed 16-token virtual memory blocks.
            </p>
            <div className="mt-3 p-3 rounded-lg bg-surface-lowest border border-outline-variant text-xs text-slate-300 space-y-1 font-mono">
              <div className="text-teal-400 font-medium">Zero Memory Fragmentation:</div>
              <div>Block table maps logical tokens → physical blocks</div>
              <div className="text-slate-500 text-[10px] font-sans">Supports copy-on-write branching & prefix caching.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-outline-variant flex items-center gap-1.5 text-xs text-teal-400 font-medium">
            <HardDrive className="w-3.5 h-3.5" />
            <span>Optimal VRAM packing</span>
          </div>
        </div>

        {/* Stage 5: Quantization */}
        <div className="p-5 rounded-xl border border-outline-variant bg-surface-container flex flex-col justify-between shadow-subtle">
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded bg-violet-500/10 text-violet-400 border border-violet-500/20">
                Stage 5: INT8 Quantized
              </span>
              <span className="text-xs font-mono text-violet-400">Symmetric Linear</span>
            </div>
            <h3 className="text-sm font-semibold text-slate-100 mb-1.5 font-sans">
              Per-Channel INT8 Weights
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Weights quantized to 8-bit integers with per-channel scale factors: <code className="text-violet-400 font-mono">W_int8 = round(W / scale)</code>. Dequantized on the fly during forward pass.
            </p>
            <div className="mt-3 p-3 rounded-lg bg-surface-lowest border border-outline-variant text-xs text-slate-300 space-y-1 font-mono">
              <div className="text-violet-400 font-medium">50% Weight VRAM Cut:</div>
              <div>Dequantize: W_fp16 = W_int8 * scale</div>
              <div className="text-slate-500 text-[10px] font-sans">Preserves lm_head to protect generation quality.</div>
            </div>
          </div>
          <div className="mt-4 pt-3 border-t border-outline-variant flex items-center gap-1.5 text-xs text-violet-400 font-medium">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Fits larger models on smaller GPUs</span>
          </div>
        </div>
      </div>

      {/* Interactive KV-Cache Memory Sizing Calculator */}
      <div className="p-6 rounded-xl border border-outline-variant bg-surface-container shadow-subtle">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <Calculator className="w-4 h-4 text-primary" />
              <span>Interactive KV-Cache GPU Memory Sizing Calculator</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Formula: <code className="text-slate-300 font-mono text-[11px]">VRAM_KV = 2 × N_layers × N_kv_heads × d_head × precision_bytes × Seq_len × Batch_size</code>
            </p>
          </div>

          {/* Model Preset Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-medium">Preset:</span>
            <button
              onClick={() => {
                setNumLayers(24);
                setNumKVHeads(2);
                setHeadDim(64);
              }}
              className="px-2.5 py-1 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant text-slate-300 hover:text-white text-xs font-sans transition-all cursor-pointer shadow-subtle"
            >
              Qwen2.5-0.5B (GQA)
            </button>
            <button
              onClick={() => {
                setNumLayers(32);
                setNumKVHeads(8);
                setHeadDim(128);
              }}
              className="px-2.5 py-1 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant text-slate-300 hover:text-white text-xs font-sans transition-all cursor-pointer shadow-subtle"
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
                <label htmlFor="calc-context-length" className="text-slate-400">Context Length (Tokens)</label>
                <span className="text-primary font-mono font-medium">{contextLength.toLocaleString()}</span>
              </div>
              <input
                id="calc-context-length"
                name="contextLength"
                type="range"
                min="512"
                max="32768"
                step="512"
                value={contextLength}
                onChange={(e) => setContextLength(Number(e.target.value))}
                className="w-full accent-primary bg-surface-lowest h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Batch Size */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <label htmlFor="calc-batch-size" className="text-slate-400">Batch Size (Concurrent Streams)</label>
                <span className="text-teal-400 font-mono font-medium">{batchSize}</span>
              </div>
              <input
                id="calc-batch-size"
                name="batchSize"
                type="range"
                min="1"
                max="64"
                step="1"
                value={batchSize}
                onChange={(e) => setBatchSize(Number(e.target.value))}
                className="w-full accent-teal-400 bg-surface-lowest h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Layers */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <label htmlFor="calc-num-layers" className="text-slate-400">Transformer Layers</label>
                <span className="text-slate-200 font-mono font-medium">{numLayers}</span>
              </div>
              <input
                id="calc-num-layers"
                name="numLayers"
                type="range"
                min="12"
                max="80"
                step="2"
                value={numLayers}
                onChange={(e) => setNumLayers(Number(e.target.value))}
                className="w-full accent-slate-400 bg-surface-lowest h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* KV Heads (GQA) */}
            <div className="space-y-1">
              <div className="flex justify-between text-xs">
                <label htmlFor="calc-kv-heads" className="text-slate-400">KV Attention Heads (GQA)</label>
                <span className="text-slate-200 font-mono font-medium">{numKVHeads}</span>
              </div>
              <input
                id="calc-kv-heads"
                name="numKVHeads"
                type="range"
                min="1"
                max="32"
                step="1"
                value={numKVHeads}
                onChange={(e) => setNumKVHeads(Number(e.target.value))}
                className="w-full accent-slate-400 bg-surface-lowest h-1.5 rounded cursor-pointer"
              />
            </div>

            {/* Precision */}
            <div className="space-y-1 sm:col-span-2">
              <span className="text-xs text-slate-400 block mb-1">KV-Cache Quantization Precision</span>
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
                    className={`flex-1 py-1.5 rounded-lg border text-xs font-sans transition-all cursor-pointer shadow-subtle ${
                      precisionBytes === p.bytes
                        ? 'bg-surface-high text-primary border-primary/50 font-medium'
                        : 'bg-surface-lowest border-outline-variant text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Results Card HUD (5 cols) */}
          <div className="lg:col-span-5 bg-surface-lowest p-5 rounded-xl border border-outline-variant flex flex-col justify-between space-y-4 shadow-subtle">
            <div>
              <span className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold font-mono block mb-3">
                Calculated KV Memory Demand
              </span>

              <div className="space-y-3">
                <div className="flex items-baseline justify-between border-b border-outline-variant pb-2">
                  <span className="text-xs text-slate-400">Memory Per Request:</span>
                  <span className="text-sm font-semibold text-slate-200 font-mono tabular-nums">
                    {memoryPerReqMb.toFixed(2)} MB
                  </span>
                </div>

                <div className="flex items-baseline justify-between border-b border-outline-variant pb-2">
                  <span className="text-xs text-slate-400">Total KV Cache (Batch={batchSize}):</span>
                  <span className="text-base font-semibold text-sky-400 font-mono tabular-nums">
                    {totalKVCacheGb >= 1.0 ? `${totalKVCacheGb.toFixed(2)} GB` : `${totalKVCacheMb.toFixed(1)} MB`}
                  </span>
                </div>

                <div className="flex items-baseline justify-between border-b border-outline-variant pb-2">
                  <span className="text-xs text-slate-400">Max Concurrency ({gpuVramGb}GB GPU):</span>
                  <span className="text-sm font-semibold text-teal-400 font-mono tabular-nums">
                    ~{maxConcurrentRequests} streams
                  </span>
                </div>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-surface-container border border-outline-variant text-xs text-slate-300">
              GQA (Grouped Query Attention) with 2 KV heads reduces KV memory by <strong className="text-white font-semibold">7x</strong> compared to full MHA (14 heads).
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
