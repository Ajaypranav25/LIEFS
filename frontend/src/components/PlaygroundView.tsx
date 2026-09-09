import React, { useState, useRef, useEffect } from 'react';
import {
  Play,
  Square,
  Terminal as TerminalIcon,
  Copy,
  Check,
  RotateCcw,
  Info,
  Code,
  Gauge,
  Timer,
  Zap,
  HardDrive,
} from 'lucide-react';
import { streamCompletion } from '../lib/api';
import type { GenerationMetrics, EngineInfo } from '../types';

interface PlaygroundViewProps {
  engines?: EngineInfo[];
  serverOnline: boolean;
}

const PRESETS = [
  {
    label: 'Python Algorithm',
    prompt: 'def fibonacci(n):\n    """Generates the first n Fibonacci numbers."""',
    maxTokens: 160,
  },
  {
    label: 'KV-Cache Explanation',
    prompt: 'Explain why KV-caching reduces autoregressive generation complexity from O(N^2) to O(N) in transformer models.',
    maxTokens: 180,
  },
  {
    label: 'Logic & Reasoning',
    prompt: 'A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show step-by-step logic.',
    maxTokens: 128,
  },
  {
    label: 'Continuous Batching',
    prompt: 'What are the main differences between continuous batching and static batching in serving engines like vLLM and TGI?',
    maxTokens: 150,
  },
];

export const PlaygroundView: React.FC<PlaygroundViewProps> = ({ serverOnline }) => {
  const [systemPrompt, setSystemPrompt] = useState<string>('You are a helpful high-performance coding and ML infrastructure assistant.');
  const [prompt, setPrompt] = useState<string>(PRESETS[0].prompt);
  const [selectedEngine, setSelectedEngine] = useState<string>('kv_cache');
  const [maxTokens, setMaxTokens] = useState<number>(160);
  const [temperature, setTemperature] = useState<number>(0.0);
  const [topP, setTopP] = useState<number>(0.95);

  const [generating, setGenerating] = useState<boolean>(false);
  const [streamedText, setStreamedText] = useState<string>('');
  const [tokensList, setTokensList] = useState<{ id: number; text: string; time: number }[]>([]);
  const [throughputHistory, setThroughputHistory] = useState<number[]>([30, 45, 60, 75, 88, 98]);
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
      const fullPrompt = systemPrompt ? `${systemPrompt}\n\n${prompt}` : prompt;
      const cancel = await streamCompletion(
        fullPrompt,
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
            if (updatedMetrics.tokens_per_sec > 0) {
              setThroughputHistory((prev) => [...prev.slice(-10), Math.round(updatedMetrics.tokens_per_sec)]);
            }
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

  // Sparkline points calculation
  const sparklinePoints = throughputHistory
    .map((val, idx) => {
      const x = (idx / Math.max(1, throughputHistory.length - 1)) * 100;
      const y = 38 - Math.min(35, (val / 120) * 35);
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <div className="flex flex-col lg:flex-row w-full pt-16 min-h-screen bg-background text-on-surface font-sans">
      {/* Left SideNavBar Configuration Panel (280px on desktop) */}
      <aside className="w-full lg:w-[280px] lg:fixed lg:top-16 lg:bottom-0 bg-surface-lowest border-r border-outline-variant flex flex-col p-4 z-40 overflow-y-auto no-scrollbar">
        {/* Panel Header */}
        <div className="flex items-center gap-3 mb-5 p-1">
          <div className="w-8 h-8 rounded-lg bg-surface-container border border-outline-variant flex items-center justify-center text-primary">
            <TerminalIcon className="w-4 h-4" />
          </div>
          <div>
            <h2 className="text-xs font-semibold text-slate-200">Configuration</h2>
            <p className="text-[11px] text-slate-400">Inference Parameters</p>
          </div>
        </div>

        {/* Configuration Form */}
        <div className="flex flex-col gap-4 flex-grow text-xs">
          {/* Presets */}
          <div>
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block mb-1.5 font-semibold font-mono">
              Prompt Presets
            </span>
            <div className="flex flex-wrap gap-1">
              {PRESETS.map((p, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setPrompt(p.prompt);
                    setMaxTokens(p.maxTokens);
                  }}
                  className={`px-2 py-1 rounded-md text-[11px] border transition-all cursor-pointer shadow-subtle ${
                    prompt === p.prompt
                      ? 'bg-surface-high text-primary border-primary/50 font-medium'
                      : 'bg-surface-lowest border-outline-variant text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* System Prompt */}
          <div className="flex flex-col gap-1">
            <label htmlFor="system-prompt-input" className="text-[11px] text-slate-300 font-medium">System Instructions</label>
            <textarea
              id="system-prompt-input"
              name="systemPrompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              disabled={generating}
              rows={2}
              className="w-full bg-surface-container border border-outline-variant rounded-lg p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-primary/50 resize-none"
              placeholder="System prompt..."
            />
          </div>

          {/* User Prompt */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between items-baseline">
              <label htmlFor="user-prompt-input" className="text-[11px] text-slate-300 font-medium">User Prompt</label>
              <span className="text-[10px] text-slate-500 font-mono">~{Math.ceil(prompt.length / 4)} tokens</span>
            </div>
            <textarea
              id="user-prompt-input"
              name="userPrompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={generating}
              rows={4}
              className="w-full bg-surface-container border border-outline-variant rounded-lg p-2 text-xs font-mono text-slate-200 focus:outline-none focus:border-primary/50 resize-none"
              placeholder="Enter prompt..."
            />
          </div>

          {/* Inference Strategy Selector */}
          <div className="flex flex-col gap-1.5 pt-1">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold font-mono">
              Inference Strategy
            </span>
            <div className="flex flex-col gap-1.5">
              {[
                { id: 'kv_cache', name: 'KV-Cache (Stage 2)', badge: 'O(N) Fast' },
                { id: 'naive', name: 'Naive (Stage 1)', badge: 'O(N²)' },
                { id: 'continuous_batching', name: 'Continuous Batching', badge: 'Stage 3' },
                { id: 'paged', name: 'Paged Attention', badge: 'Stage 4' },
                { id: 'quantized', name: 'INT8 Quantized', badge: 'Stage 5' },
              ].map((strategy) => (
                <label
                  key={strategy.id}
                  htmlFor={`strategy-${strategy.id}`}
                  onClick={() => setSelectedEngine(strategy.id)}
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded-lg border cursor-pointer transition-all shadow-subtle ${
                    selectedEngine === strategy.id
                      ? 'bg-surface-high border-primary/50 text-slate-100'
                      : 'bg-surface-lowest border-outline-variant text-slate-400 hover:text-slate-200 hover:bg-surface-container'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <input
                      id={`strategy-${strategy.id}`}
                      type="radio"
                      name="strategy"
                      value={strategy.id}
                      checked={selectedEngine === strategy.id}
                      onChange={() => setSelectedEngine(strategy.id)}
                      className="text-primary accent-primary"
                    />
                    <span className="text-xs font-medium">{strategy.name}</span>
                  </div>
                  <span className="text-[9px] px-1.5 py-0.5 rounded bg-surface-lowest text-slate-400 font-mono border border-outline-variant">
                    {strategy.badge}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Sliders: Max Tokens, Temperature, Top-P */}
          <div className="flex flex-col gap-3 pt-2 border-t border-outline-variant">
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <label htmlFor="max-tokens-slider" className="text-slate-300">Max Tokens</label>
                <span className="text-primary font-mono font-medium">{maxTokens}</span>
              </div>
              <input
                id="max-tokens-slider"
                name="maxTokens"
                type="range"
                min="16"
                max="512"
                step="16"
                value={maxTokens}
                onChange={(e) => setMaxTokens(Number(e.target.value))}
                disabled={generating}
                className="w-full accent-primary h-1 bg-surface-container rounded appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <label htmlFor="temperature-slider" className="text-slate-300">Temperature</label>
                <span className="text-teal-400 font-mono font-medium">{temperature.toFixed(1)}</span>
              </div>
              <input
                id="temperature-slider"
                name="temperature"
                type="range"
                min="0.0"
                max="1.0"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number(e.target.value))}
                disabled={generating}
                className="w-full accent-teal-400 h-1 bg-surface-container rounded appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <label htmlFor="topp-slider" className="text-slate-300">Top-P</label>
                <span className="text-violet-400 font-mono font-medium">{topP.toFixed(2)}</span>
              </div>
              <input
                id="topp-slider"
                name="topP"
                type="range"
                min="0.5"
                max="1.0"
                step="0.05"
                value={topP}
                onChange={(e) => setTopP(Number(e.target.value))}
                disabled={generating}
                className="w-full accent-violet-400 h-1 bg-surface-container rounded appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Side Footer CTA Button */}
        <div className="mt-4 pt-3 border-t border-outline-variant flex flex-col gap-2">
          {generating ? (
            <button
              onClick={handleStop}
              className="w-full bg-rose-600 hover:bg-rose-500 text-white text-xs py-2 rounded-lg font-medium flex items-center justify-center gap-2 shadow-subtle transition-all cursor-pointer"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Abort Generation</span>
            </button>
          ) : (
            <button
              onClick={handleStartGeneration}
              disabled={!serverOnline}
              className={`w-full text-xs py-2.5 rounded-lg font-medium flex items-center justify-center gap-2 transition-all shadow-subtle ${
                serverOnline
                  ? 'bg-primary hover:bg-sky-300 text-slate-950 cursor-pointer active:scale-[0.99]'
                  : 'bg-surface-high text-slate-500 cursor-not-allowed'
              }`}
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Deploy & Execute</span>
            </button>
          )}
        </div>
      </aside>

      {/* Main Content Area (Fluid width on right of 280px sidebar) */}
      <main className="lg:ml-[280px] flex-1 p-4 sm:p-6 flex flex-col lg:flex-row gap-6 overflow-hidden">
        {/* Terminal Streaming Canvas */}
        <div className="flex-1 flex flex-col rounded-xl border border-outline-variant bg-surface-container relative overflow-hidden group min-h-[500px] shadow-subtle">
          {/* Terminal Header */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-outline-variant bg-surface-low">
            <div className="flex items-center gap-2">
              <Code className="w-4 h-4 text-primary" />
              <span className="text-xs font-mono text-slate-300">stream://{selectedEngine}/output.py</span>
            </div>

            <div className="flex items-center gap-3">
              {generating ? (
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-mono text-emerald-400 font-medium">Streaming Tokens...</span>
                </div>
              ) : (
                <span className="text-xs font-mono text-slate-500">Ready</span>
              )}

              <button
                onClick={handleCopy}
                disabled={!streamedText}
                className="px-2 py-1 rounded-md bg-surface-lowest border border-outline-variant hover:border-white/[0.16] text-slate-400 hover:text-slate-200 disabled:opacity-30 text-xs font-mono flex items-center gap-1 transition-all cursor-pointer"
                title="Copy output"
              >
                {copied ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>

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
                className="p-1 rounded-md bg-surface-lowest border border-outline-variant text-slate-400 hover:text-slate-200 disabled:opacity-30 cursor-pointer"
                title="Clear Terminal"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Terminal Code Body */}
          <div className="flex-1 p-5 bg-surface-lowest overflow-y-auto font-mono text-xs leading-relaxed text-slate-200 select-text no-scrollbar">
            {errorMsg && (
              <div className="mb-3 p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-2">
                <Info className="w-4 h-4 text-rose-400 shrink-0" />
                <span>{errorMsg}</span>
              </div>
            )}

            {streamedText ? (
              <div>
                <pre className="whitespace-pre-wrap font-mono text-xs leading-6 text-slate-200">
                  <code>{streamedText}</code>
                  {generating && <span className="terminal-cursor" />}
                </pre>
              </div>
            ) : (
              <div className="h-full min-h-[350px] flex flex-col items-center justify-center text-slate-600 text-center select-none">
                <TerminalIcon className="w-10 h-10 mb-3 text-slate-700" />
                <p className="font-sans font-medium text-xs text-slate-400">LIEFS Engine Standby</p>
                <p className="text-[11px] text-slate-500 mt-1 max-w-sm font-sans">
                  Select an engine strategy in the configuration rail and click "Deploy & Execute" to stream tokens live.
                </p>
              </div>
            )}
            <div ref={terminalEndRef} />
          </div>

          {/* Terminal Footer Bar */}
          <div className="bg-surface-low px-4 py-2 border-t border-outline-variant text-[11px] font-mono text-slate-400 flex items-center justify-between">
            <div>
              Tokens: Prompt <span className="text-slate-200">{metrics.prompt_tokens || 0}</span> | Generated <span className="text-primary font-medium">{tokensList.length}</span>
            </div>
            <div>
              Wall Time: <span className="text-slate-200">{metrics.total_time_ms ? `${(metrics.total_time_ms / 1000).toFixed(2)}s` : '--'}</span>
            </div>
          </div>
        </div>

        {/* Right Telemetry HUD Rail (320px) */}
        <div className="w-full lg:w-[320px] flex-shrink-0 flex flex-col gap-4 font-mono">
          {/* Telemetry HUD Card */}
          <div className="rounded-xl border border-outline-variant bg-surface-container p-4 flex flex-col gap-4 relative overflow-hidden shadow-subtle">
            <h3 className="text-xs font-semibold text-slate-200 border-b border-outline-variant pb-2 flex items-center gap-1.5 font-sans">
              <Zap className="w-3.5 h-3.5 text-primary" />
              Telemetry HUD
            </h3>

            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-baseline">
                <span className="text-[11px] text-slate-400 flex items-center gap-1">
                  <Timer className="w-3 h-3 text-teal-400" />
                  TTFT (Time to First Token)
                </span>
                <span className="text-sm font-semibold text-teal-400 tabular-nums">
                  {metrics.ttft_ms > 0 ? `${metrics.ttft_ms.toFixed(1)}ms` : '--'}
                </span>
              </div>

              <div className="flex justify-between items-baseline">
                <span className="text-[11px] text-slate-400 flex items-center gap-1">
                  <Gauge className="w-3 h-3 text-teal-400" />
                  TPOT (Time Per Token)
                </span>
                <span className="text-sm font-semibold text-teal-400 tabular-nums">
                  {metrics.tpot_ms > 0 ? `${metrics.tpot_ms.toFixed(1)}ms` : '--'}
                </span>
              </div>

              <div className="pt-2 border-t border-outline-variant">
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-[11px] text-primary font-medium">Throughput Rate</span>
                  <span className="text-sm font-semibold text-primary tabular-nums">
                    {metrics.tokens_per_sec > 0 ? `${metrics.tokens_per_sec.toFixed(1)} T/s` : '--'}
                  </span>
                </div>

                {/* SVG Live Sparkline */}
                <div className="h-10 w-full relative">
                  <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 100 40">
                    <polyline
                      fill="none"
                      points={sparklinePoints}
                      stroke="#38bdf8"
                      strokeWidth="1.5"
                    />
                    <polygon
                      fill="url(#sparkline-gradient)"
                      opacity="0.25"
                      points={`0,40 ${sparklinePoints} 100,40`}
                    />
                    <defs>
                      <linearGradient id="sparkline-gradient" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="#38bdf8" stopOpacity="1" />
                        <stop offset="100%" stopColor="#38bdf8" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
              </div>

              <div className="pt-2 border-t border-outline-variant flex justify-between items-baseline">
                <span className="text-[11px] text-slate-400 flex items-center gap-1">
                  <HardDrive className="w-3 h-3 text-violet-400" />
                  Peak VRAM
                </span>
                <span className="text-xs font-semibold text-violet-300 tabular-nums">
                  {metrics.peak_vram_mb > 0 ? `${Math.round(metrics.peak_vram_mb)} MB` : '--'}
                </span>
              </div>
            </div>
          </div>

          {/* Model Performance Radar / Geometry Card */}
          <div className="rounded-xl border border-outline-variant bg-surface-container p-4 flex-1 flex flex-col items-center justify-center relative overflow-hidden min-h-[180px] shadow-subtle">
            <span className="text-[10px] uppercase tracking-wider text-slate-400 mb-3 font-semibold font-mono">
              Inference Hardware Saturation
            </span>

            <div className="w-28 h-28 relative border border-outline-variant rounded-full flex items-center justify-center">
              <div className="absolute w-20 h-20 border border-outline-variant/60 rounded-full" />
              <div className="absolute w-12 h-12 border border-outline-variant/40 rounded-full" />
              <div className="absolute w-full h-px bg-outline-variant" />
              <div className="absolute h-full w-px bg-outline-variant" />
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
                <polygon
                  fill="rgba(45, 212, 191, 0.15)"
                  points="50,18 82,38 72,82 28,76 22,34"
                  stroke="#2dd4bf"
                  strokeWidth="1.2"
                />
              </svg>
            </div>

            <span className="text-[10px] text-slate-400 mt-3 text-center font-mono">
              Qwen2.5-0.5B • CUDA FP16
            </span>
          </div>
        </div>
      </main>
    </div>
  );
};
