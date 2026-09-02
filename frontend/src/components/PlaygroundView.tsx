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
    <div className="flex flex-col lg:flex-row w-full pt-16 min-h-screen bg-[#020617] text-on-surface">
      {/* Left SideNavBar Configuration Panel (280px on desktop) */}
      <aside className="w-full lg:w-[280px] lg:fixed lg:top-16 lg:bottom-0 bg-surface-container-low border-r border-outline-variant/30 flex flex-col p-4 z-40 overflow-y-auto no-scrollbar">
        {/* Panel Header */}
        <div className="flex items-center gap-3 mb-5 p-1">
          <div className="w-9 h-9 rounded bg-[#0f172a] border border-white/10 flex items-center justify-center glow-hover">
            <TerminalIcon className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h2 className="text-xs font-mono font-bold text-primary">Configuration</h2>
            <p className="text-[11px] font-mono text-on-surface-variant">Inference Parameters</p>
          </div>
        </div>

        {/* Configuration Form */}
        <div className="flex flex-col gap-4 flex-grow text-xs font-mono">
          {/* Presets */}
          <div>
            <span className="text-[10px] text-on-surface-variant uppercase tracking-wider block mb-1.5 font-semibold">
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
                  className={`px-2 py-1 rounded text-[11px] border transition-all ${
                    prompt === p.prompt
                      ? 'bg-primary/20 text-primary border-primary/50 font-semibold'
                      : 'bg-surface border-slate-800 text-on-surface-variant hover:text-on-surface hover:border-slate-700'
                  }`}
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* System Prompt */}
          <div className="flex flex-col gap-1">
            <label htmlFor="system-prompt-input" className="text-[11px] text-secondary font-medium">System Instructions</label>
            <textarea
              id="system-prompt-input"
              name="systemPrompt"
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
              disabled={generating}
              rows={2}
              className="w-full bg-[#020617] border border-slate-800 rounded p-2 text-xs font-mono text-on-surface focus:outline-none focus:border-primary glow-hover resize-none"
              placeholder="System prompt..."
            />
          </div>

          {/* User Prompt */}
          <div className="flex flex-col gap-1">
            <div className="flex justify-between items-baseline">
              <label htmlFor="user-prompt-input" className="text-[11px] text-secondary font-medium">User Prompt</label>
              <span className="text-[10px] text-slate-500">~{Math.ceil(prompt.length / 4)} tokens</span>
            </div>
            <textarea
              id="user-prompt-input"
              name="userPrompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              disabled={generating}
              rows={4}
              className="w-full bg-[#020617] border border-slate-800 rounded p-2 text-xs font-mono text-on-surface focus:outline-none focus:border-primary glow-hover resize-none"
              placeholder="Enter prompt..."
            />
          </div>

          {/* Inference Strategy Selector */}
          <div className="flex flex-col gap-1.5 pt-1">
            <span className="text-[10px] text-on-surface-variant uppercase tracking-wider font-semibold">
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
                  className={`flex items-center justify-between px-2.5 py-1.5 rounded border cursor-pointer transition-all ${
                    selectedEngine === strategy.id
                      ? 'bg-[#0f172a] border-primary/50 text-primary glow-active'
                      : 'bg-surface/50 border-slate-800/80 text-on-surface-variant hover:text-on-surface hover:border-slate-700'
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
                    <span className="text-xs font-semibold">{strategy.name}</span>
                  </div>
                  <span className="text-[9px] px-1 py-0.5 rounded bg-slate-800/80 text-slate-400">
                    {strategy.badge}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Sliders: Max Tokens, Temperature, Top-P */}
          <div className="flex flex-col gap-3 pt-2 border-t border-slate-800/80">
            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <label htmlFor="max-tokens-slider" className="text-on-surface">Max Tokens</label>
                <span className="text-secondary font-semibold">{maxTokens}</span>
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
                className="w-full accent-primary h-1 bg-slate-800 rounded appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <label htmlFor="temperature-slider" className="text-on-surface">Temperature</label>
                <span className="text-secondary font-semibold">{temperature.toFixed(1)}</span>
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
                className="w-full accent-primary h-1 bg-slate-800 rounded appearance-none cursor-pointer"
              />
            </div>

            <div>
              <div className="flex justify-between text-[11px] mb-1">
                <label htmlFor="topp-slider" className="text-on-surface">Top-P</label>
                <span className="text-secondary font-semibold">{topP.toFixed(2)}</span>
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
                className="w-full accent-primary h-1 bg-slate-800 rounded appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Side Footer CTA Button */}
        <div className="mt-4 pt-3 border-t border-outline-variant/30 flex flex-col gap-2">
          {generating ? (
            <button
              onClick={handleStop}
              className="w-full bg-rose-600 hover:bg-rose-500 text-white font-mono text-xs py-2 rounded font-bold flex items-center justify-center gap-2 shadow-lg transition-all"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
              <span>Abort Generation</span>
            </button>
          ) : (
            <button
              onClick={handleStartGeneration}
              disabled={!serverOnline}
              className={`w-full font-mono text-xs py-2.5 rounded font-bold flex items-center justify-center gap-2 transition-all ${
                serverOnline
                  ? 'bg-primary hover:bg-primary-fixed text-[#000000] shadow-[0_0_12px_rgba(78,222,163,0.3)] cursor-pointer'
                  : 'bg-slate-800 text-slate-500 cursor-not-allowed'
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
        <div className="flex-1 flex flex-col cyber-card rounded-lg relative overflow-hidden group min-h-[500px]">
          <div className="absolute inset-0 border border-primary/20 rounded-lg pointer-events-none group-hover:shadow-[0_0_15px_rgba(78,222,163,0.15)] transition-shadow duration-300" />

          {/* Terminal Header */}
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-white/10 bg-[#1e293b]/50">
            <div className="flex items-center gap-2">
              <Code className="w-4 h-4 text-secondary" />
              <span className="text-xs font-mono text-secondary">stream://{selectedEngine}/output.py</span>
            </div>

            <div className="flex items-center gap-3">
              {generating ? (
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-primary animate-ping" />
                  <span className="text-xs font-mono text-primary font-semibold">Streaming Tokens...</span>
                </div>
              ) : (
                <span className="text-xs font-mono text-slate-500">Ready</span>
              )}

              <button
                onClick={handleCopy}
                disabled={!streamedText}
                className="px-2 py-1 rounded bg-[#0f172a] border border-slate-800 hover:border-slate-700 text-on-surface-variant hover:text-on-surface disabled:opacity-30 text-xs font-mono flex items-center gap-1 transition-all"
                title="Copy output"
              >
                {copied ? <Check className="w-3 h-3 text-primary" /> : <Copy className="w-3 h-3" />}
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
                className="p-1 rounded bg-[#0f172a] border border-slate-800 text-on-surface-variant hover:text-on-surface disabled:opacity-30"
                title="Clear Terminal"
              >
                <RotateCcw className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Terminal Code Body */}
          <div className="flex-1 p-5 bg-[#000000] overflow-y-auto font-mono text-xs leading-relaxed text-on-surface select-text no-scrollbar">
            {errorMsg && (
              <div className="mb-3 p-2.5 rounded bg-rose-950/60 border border-rose-800 text-rose-300 text-xs flex items-center gap-2">
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
                <TerminalIcon className="w-12 h-12 mb-3 text-slate-800" />
                <p className="font-mono text-xs text-slate-400">LIEFS Engine Standby</p>
                <p className="text-[11px] text-slate-600 mt-1 max-w-sm">
                  Select an engine strategy in the configuration rail and click "Deploy & Execute" to stream tokens live.
                </p>
              </div>
            )}
            <div ref={terminalEndRef} />
          </div>

          {/* Terminal Footer Bar */}
          <div className="bg-[#0f172a] px-4 py-2 border-t border-white/10 text-[11px] font-mono text-on-surface-variant flex items-center justify-between">
            <div>
              Tokens: Prompt <span className="text-slate-200">{metrics.prompt_tokens || 0}</span> | Generated <span className="text-primary font-bold">{tokensList.length}</span>
            </div>
            <div>
              Wall Time: <span className="text-slate-200">{metrics.total_time_ms ? `${(metrics.total_time_ms / 1000).toFixed(2)}s` : '--'}</span>
            </div>
          </div>
        </div>

        {/* Right Telemetry HUD Rail (320px) */}
        <div className="w-full lg:w-[320px] flex-shrink-0 flex flex-col gap-4 font-mono">
          {/* Telemetry HUD Card */}
          <div className="cyber-card rounded-lg p-4 flex flex-col gap-4 relative overflow-hidden">
            <h3 className="text-xs font-bold text-primary border-b border-white/10 pb-2 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-primary" />
              Telemetry HUD
            </h3>

            <div className="flex flex-col gap-3">
              <div className="flex justify-between items-baseline">
                <span className="text-[11px] text-on-surface-variant flex items-center gap-1">
                  <Timer className="w-3 h-3 text-secondary" />
                  TTFT (Time to First Token)
                </span>
                <span className="text-sm font-bold text-secondary">
                  {metrics.ttft_ms > 0 ? `${metrics.ttft_ms.toFixed(1)}ms` : '--'}
                </span>
              </div>

              <div className="flex justify-between items-baseline">
                <span className="text-[11px] text-on-surface-variant flex items-center gap-1">
                  <Gauge className="w-3 h-3 text-secondary" />
                  TPOT (Time Per Token)
                </span>
                <span className="text-sm font-bold text-secondary">
                  {metrics.tpot_ms > 0 ? `${metrics.tpot_ms.toFixed(1)}ms` : '--'}
                </span>
              </div>

              <div className="pt-2 border-t border-white/5">
                <div className="flex justify-between items-baseline mb-2">
                  <span className="text-[11px] text-primary font-semibold">Throughput Rate</span>
                  <span className="text-sm font-bold text-primary">
                    {metrics.tokens_per_sec > 0 ? `${metrics.tokens_per_sec.toFixed(1)} T/s` : '--'}
                  </span>
                </div>

                {/* SVG Live Sparkline */}
                <div className="h-10 w-full relative">
                  <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 100 40">
                    <polyline
                      fill="none"
                      points={sparklinePoints}
                      stroke="#4edea3"
                      strokeWidth="1.5"
                    />
                    <polygon
                      fill="url(#sparkline-gradient)"
                      opacity="0.25"
                      points={`0,40 ${sparklinePoints} 100,40`}
                    />
                    <defs>
                      <linearGradient id="sparkline-gradient" x1="0" x2="0" y1="0" y2="1">
                        <stop offset="0%" stopColor="#4edea3" stopOpacity="1" />
                        <stop offset="100%" stopColor="#4edea3" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
              </div>

              <div className="pt-2 border-t border-white/5 flex justify-between items-baseline">
                <span className="text-[11px] text-on-surface-variant flex items-center gap-1">
                  <HardDrive className="w-3 h-3 text-purple-400" />
                  Peak VRAM
                </span>
                <span className="text-xs font-bold text-purple-300">
                  {metrics.peak_vram_mb > 0 ? `${Math.round(metrics.peak_vram_mb)} MB` : '--'}
                </span>
              </div>
            </div>
          </div>

          {/* Model Performance Radar / Geometry Card */}
          <div className="cyber-card rounded-lg p-4 flex-1 flex flex-col items-center justify-center relative overflow-hidden min-h-[180px]">
            <span className="text-[10px] uppercase tracking-wider text-on-surface-variant/70 mb-3 font-semibold">
              Inference Hardware Saturation
            </span>

            <div className="w-28 h-28 relative border border-secondary/30 rounded-full flex items-center justify-center">
              <div className="absolute w-20 h-20 border border-secondary/20 rounded-full" />
              <div className="absolute w-12 h-12 border border-secondary/10 rounded-full" />
              <div className="absolute w-full h-px bg-secondary/20" />
              <div className="absolute h-full w-px bg-secondary/20" />
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100">
                <polygon
                  fill="rgba(76, 215, 246, 0.2)"
                  points="50,18 82,38 72,82 28,76 22,34"
                  stroke="#4cd7f6"
                  strokeWidth="1.2"
                />
              </svg>
            </div>

            <span className="text-[10px] text-on-surface-variant mt-3 text-center">
              Qwen2.5-0.5B • CUDA FP16
            </span>
          </div>
        </div>
      </main>
    </div>
  );
};
