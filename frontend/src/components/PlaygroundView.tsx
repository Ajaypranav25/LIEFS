import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Square,
  Sparkles,
  Gauge,
  Timer,
  Zap,
  HardDrive,
  Copy,
  Check,
  RotateCcw,
  Terminal as TerminalIcon,
  Info,
} from 'lucide-react';
import { streamCompletion } from '../lib/api';
import type { GenerationMetrics, EngineInfo } from '../types';

interface PlaygroundViewProps {
  engines?: EngineInfo[];
  serverOnline: boolean;
}

const PRESETS = [
  {
    label: 'KV-Cache Analysis',
    prompt: 'Explain why KV-caching reduces autoregressive generation complexity from O(N^2) to O(N) in transformer models.',
    maxTokens: 160,
  },
  {
    label: 'Python Algorithm',
    prompt: 'Write an efficient Python function to calculate Fibonacci numbers using memoization and dynamic programming.',
    maxTokens: 180,
  },
  {
    label: 'Reasoning & Logic',
    prompt: 'A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show your step-by-step logic.',
    maxTokens: 128,
  },
  {
    label: 'Short QA',
    prompt: 'What are the main differences between continuous batching and static batching in LLM serving engines like vLLM?',
    maxTokens: 140,
  },
];

export const PlaygroundView: React.FC<PlaygroundViewProps> = ({ serverOnline }) => {
  const [prompt, setPrompt] = useState<string>(PRESETS[0].prompt);
  const [selectedEngine, setSelectedEngine] = useState<string>('kv_cache');
  const [maxTokens, setMaxTokens] = useState<number>(128);
  const [temperature, setTemperature] = useState<number>(0.0);

  const [generating, setGenerating] = useState<boolean>(false);
  const [streamedText, setStreamedText] = useState<string>('');
  const [tokensList, setTokensList] = useState<{ id: number; text: string; time: number }[]>([]);
  const [copied, setCopied] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Real-time telemetry metrics
  const [metrics, setMetrics] = useState<GenerationMetrics>({
    ttft_ms: 0,
    tpot_ms: 0,
    total_time_ms: 0,
    tokens_per_sec: 0,
    peak_vram_mb: 0,
    generated_tokens: 0,
    prompt_tokens: 0,
  });

  const cancelStreamRef = useRef<(() => void) | null>(null);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (generating && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [streamedText, generating]);

  const handleStartGeneration = async () => {
    if (!prompt.trim() || generating) return;
    setErrorMsg(null);
    setGenerating(true);
    setStreamedText('');
    setTokensList([]);
    setMetrics({
      ttft_ms: 0,
      tpot_ms: 0,
      total_time_ms: 0,
      tokens_per_sec: 0,
      peak_vram_mb: 0,
      generated_tokens: 0,
      prompt_tokens: 0,
    });

    const startTime = performance.now();

    try {
      const cancel = await streamCompletion(
        prompt,
        {
          maxTokens,
          temperature,
          engine: selectedEngine,
        },
        {
          onToken: (token) => {
            setStreamedText((prev) => prev + token);
            setTokensList((prev) => [
              ...prev,
              { id: prev.length, text: token, time: Math.round(performance.now() - startTime) },
            ]);
          },
          onMetrics: (updatedMetrics) => {
            setMetrics(updatedMetrics);
          },
          onDone: (fullText, finalMetrics) => {
            setStreamedText(fullText);
            setMetrics(finalMetrics);
            setGenerating(false);
            cancelStreamRef.current = null;
          },
          onError: (err) => {
            setErrorMsg(err.message || 'Generation error occurred');
            setGenerating(false);
            cancelStreamRef.current = null;
          },
        }
      );
      cancelStreamRef.current = cancel;
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to connect to backend server');
      setGenerating(false);
    }
  };

  const handleStop = () => {
    if (cancelStreamRef.current) {
      cancelStreamRef.current();
      cancelStreamRef.current = null;
    }
    setGenerating(false);
  };

  const handleCopy = () => {
    navigator.clipboard.writeText(streamedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Top Banner / Error */}
      {errorMsg && (
        <div className="mb-4 p-3 rounded-lg bg-rose-950/60 border border-rose-800 text-rose-300 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Info className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{errorMsg}</span>
          </div>
          <button onClick={() => setErrorMsg(null)} className="text-rose-400 hover:text-white font-mono">
            Dismiss
          </button>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Prompt & Generation Configuration (5 cols) */}
        <div className="lg:col-span-5 flex flex-col gap-5">
          {/* Preset Prompts Pill Bar */}
          <div className="glass-panel p-4 rounded-xl">
            <div className="flex items-center justify-between mb-2.5">
              <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-emerald-400" />
                Prompt Presets
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((preset, idx) => (
                <button
                  key={idx}
                  onClick={() => {
                    setPrompt(preset.prompt);
                    setMaxTokens(preset.maxTokens);
                  }}
                  className={`text-xs px-2.5 py-1 rounded-md border font-mono transition-all ${
                    prompt === preset.prompt
                      ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-glow-emerald'
                      : 'bg-obsidian-900/80 text-slate-400 border-slate-800 hover:border-slate-700 hover:text-slate-200'
                  }`}
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* Prompt Input Form */}
          <div className="glass-panel p-4 rounded-xl flex flex-col flex-1">
            <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono mb-2 flex items-center justify-between">
              <span>Input Prompt</span>
              <span className="text-[11px] text-slate-500 font-mono">
                ~{Math.ceil(prompt.length / 4)} tokens
              </span>
            </label>
            <textarea
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter your prompt here..."
              rows={5}
              disabled={generating}
              className="w-full bg-obsidian-950 text-slate-100 placeholder-slate-600 rounded-lg p-3 text-sm font-sans border border-slate-800 focus:border-emerald-500/50 focus:ring-1 focus:ring-emerald-500/50 outline-none transition-all resize-none"
            />

            {/* Inference Engine Selector */}
            <div className="mt-4">
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono mb-2 block">
                Inference Architecture Stage
              </label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  {
                    id: 'kv_cache',
                    name: 'KV-Cache (Stage 2)',
                    desc: 'O(N) Attention Cache',
                    badge: 'Fastest Single',
                  },
                  {
                    id: 'naive',
                    name: 'Naive (Stage 1)',
                    desc: 'O(N²) Recompute',
                    badge: 'Baseline',
                  },
                  {
                    id: 'paged',
                    name: 'Paged Attention (Stage 4)',
                    desc: 'Block Allocator Pool',
                    badge: 'Zero Waste',
                  },
                  {
                    id: 'quantized',
                    name: 'INT8 Quantized (Stage 5)',
                    desc: 'Per-channel Linear',
                    badge: '50% VRAM Cut',
                  },
                ].map((eng) => (
                  <button
                    key={eng.id}
                    type="button"
                    onClick={() => setSelectedEngine(eng.id)}
                    className={`p-2.5 rounded-lg border text-left transition-all relative ${
                      selectedEngine === eng.id
                        ? 'bg-emerald-500/10 border-emerald-500/50 text-slate-100 shadow-glow-emerald'
                        : 'bg-obsidian-950/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-300'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-mono font-semibold">{eng.name}</span>
                      <span
                        className={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
                          selectedEngine === eng.id
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-slate-800 text-slate-400'
                        }`}
                      >
                        {eng.badge}
                      </span>
                    </div>
                    <p className="text-[10px] text-slate-500 font-sans">{eng.desc}</p>
                  </button>
                ))}
              </div>
            </div>

            {/* Sliders: Max Tokens & Temperature */}
            <div className="mt-4 space-y-3 pt-3 border-t border-slate-800/80">
              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">Max New Tokens</span>
                  <span className="text-emerald-400 font-semibold">{maxTokens}</span>
                </div>
                <input
                  type="range"
                  min="16"
                  max="512"
                  step="16"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  disabled={generating}
                  className="w-full accent-emerald-500 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
                />
              </div>

              <div>
                <div className="flex justify-between text-xs font-mono mb-1">
                  <span className="text-slate-400">Temperature (Greedy argmax default)</span>
                  <span className="text-cyan-400 font-semibold">{temperature.toFixed(1)}</span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="1.0"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(Number(e.target.value))}
                  disabled={generating}
                  className="w-full accent-cyan-500 bg-slate-800 h-1.5 rounded-lg cursor-pointer"
                />
              </div>
            </div>

            {/* Action Buttons */}
            <div className="mt-5 flex gap-2">
              {generating ? (
                <button
                  onClick={handleStop}
                  className="flex-1 py-2.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-mono font-semibold text-xs flex items-center justify-center gap-2 transition-all shadow-lg shadow-rose-900/30"
                >
                  <Square className="w-4 h-4 fill-white" />
                  <span>Abort Generation</span>
                </button>
              ) : (
                <button
                  onClick={handleStartGeneration}
                  disabled={!serverOnline}
                  className={`flex-1 py-2.5 rounded-lg font-mono font-semibold text-xs flex items-center justify-center gap-2 transition-all ${
                    serverOnline
                      ? 'bg-emerald-500 hover:bg-emerald-400 text-obsidian-950 shadow-glow-emerald cursor-pointer'
                      : 'bg-slate-800 text-slate-500 cursor-not-allowed'
                  }`}
                >
                  <Play className="w-4 h-4 fill-current" />
                  <span>Execute Generation</span>
                </button>
              )}

              <button
                onClick={() => {
                  setStreamedText('');
                  setTokensList([]);
                  setMetrics({
                    ttft_ms: 0,
                    tpot_ms: 0,
                    total_time_ms: 0,
                    tokens_per_sec: 0,
                    peak_vram_mb: 0,
                  });
                }}
                disabled={generating || !streamedText}
                className="px-3 py-2.5 rounded-lg border border-slate-800 bg-obsidian-950 hover:bg-slate-800 text-slate-400 hover:text-slate-200 disabled:opacity-40 font-mono text-xs flex items-center gap-1.5 transition-all"
                title="Reset Terminal"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Live Streaming Terminal Canvas & Telemetry HUD (7 cols) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          {/* Live Telemetry Ribbon HUD */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
            {/* TTFT */}
            <div className="glass-panel p-3 rounded-xl border border-slate-800/80">
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 mb-1">
                <Timer className="w-3.5 h-3.5 text-cyan-400" />
                <span>TTFT</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-xl font-bold font-mono text-slate-100">
                  {metrics.ttft_ms > 0 ? metrics.ttft_ms.toFixed(1) : '--'}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">ms</span>
              </div>
              <span className="text-[9px] text-slate-500 font-mono">Time-to-First-Token</span>
            </div>

            {/* TPOT */}
            <div className="glass-panel p-3 rounded-xl border border-slate-800/80">
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 mb-1">
                <Gauge className="w-3.5 h-3.5 text-emerald-400" />
                <span>TPOT</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-xl font-bold font-mono text-slate-100">
                  {metrics.tpot_ms > 0 ? metrics.tpot_ms.toFixed(1) : '--'}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">ms</span>
              </div>
              <span className="text-[9px] text-slate-500 font-mono">Time-Per-Output-Token</span>
            </div>

            {/* Throughput */}
            <div className="glass-panel p-3 rounded-xl border border-slate-800/80">
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 mb-1">
                <Zap className="w-3.5 h-3.5 text-amber-400" />
                <span>Throughput</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-xl font-bold font-mono text-emerald-400">
                  {metrics.tokens_per_sec > 0 ? metrics.tokens_per_sec.toFixed(1) : '--'}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">tok/s</span>
              </div>
              <span className="text-[9px] text-slate-500 font-mono">Live Generation Rate</span>
            </div>

            {/* Peak VRAM */}
            <div className="glass-panel p-3 rounded-xl border border-slate-800/80">
              <div className="flex items-center gap-1.5 text-[11px] font-mono text-slate-400 mb-1">
                <HardDrive className="w-3.5 h-3.5 text-indigo-400" />
                <span>Peak VRAM</span>
              </div>
              <div className="flex items-baseline gap-1">
                <span className="text-xl font-bold font-mono text-slate-100">
                  {metrics.peak_vram_mb > 0 ? Math.round(metrics.peak_vram_mb) : '--'}
                </span>
                <span className="text-[10px] text-slate-400 font-mono">MB</span>
              </div>
              <span className="text-[9px] text-slate-500 font-mono">
                {metrics.generated_tokens ? `${metrics.generated_tokens} tokens gen` : 'Memory peak'}
              </span>
            </div>
          </div>

          {/* Terminal Container */}
          <div className="glass-panel-glow rounded-xl overflow-hidden flex flex-col flex-1 min-h-[420px]">
            {/* Terminal Top Window Bar */}
            <div className="bg-obsidian-950 px-4 py-2.5 border-b border-slate-800/80 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-rose-500/80" />
                  <div className="w-2.5 h-2.5 rounded-full bg-amber-500/80" />
                  <div className="w-2.5 h-2.5 rounded-full bg-emerald-500/80" />
                </div>
                <div className="flex items-center gap-1.5 ml-2 text-xs font-mono text-slate-400">
                  <TerminalIcon className="w-3.5 h-3.5 text-emerald-400" />
                  <span>stream://{selectedEngine}/output</span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                {generating && (
                  <span className="text-[11px] font-mono text-emerald-400 flex items-center gap-1.5">
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                    STREAMING TOKENS...
                  </span>
                )}
                <button
                  onClick={handleCopy}
                  disabled={!streamedText}
                  className="px-2 py-1 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-400 hover:text-slate-200 disabled:opacity-30 text-xs font-mono flex items-center gap-1 transition-all cursor-pointer"
                  title="Copy generated text"
                >
                  {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
            </div>

            {/* Terminal Body */}
            <div className="p-5 bg-obsidian-950/90 flex-1 overflow-y-auto font-mono text-sm leading-relaxed text-slate-200 select-text">
              {streamedText ? (
                <div>
                  <span className="whitespace-pre-wrap">{streamedText}</span>
                  {generating && <span className="terminal-cursor" />}
                </div>
              ) : (
                <div className="h-full min-h-[300px] flex flex-col items-center justify-center text-slate-600 text-center select-none">
                  <TerminalIcon className="w-12 h-12 mb-3 text-slate-700 stroke-[1.5]" />
                  <p className="font-mono text-xs text-slate-400">LIEFS Inference Engine Ready</p>
                  <p className="text-[11px] text-slate-600 mt-1 max-w-sm">
                    Select an engine configuration and click "Execute Generation" to inspect real-time token streaming and per-token telemetry.
                  </p>
                </div>
              )}
              <div ref={terminalEndRef} />
            </div>

            {/* Terminal Footer Info Bar */}
            <div className="bg-obsidian-950/80 px-4 py-2 border-t border-slate-800 text-[11px] font-mono text-slate-500 flex items-center justify-between">
              <div>
                Prompt Tokens: <span className="text-slate-300">{metrics.prompt_tokens || 0}</span> | Generated: <span className="text-emerald-400 font-semibold">{tokensList.length}</span>
              </div>
              <div>
                Total Time: <span className="text-slate-300">{metrics.total_time_ms ? `${(metrics.total_time_ms / 1000).toFixed(2)}s` : '--'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
