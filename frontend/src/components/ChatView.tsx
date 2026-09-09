import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Cpu,
  User,
  Copy,
  Check,
  Zap,
  ChevronDown,
  Code2,
  Layers,
  ArrowRight,
} from 'lucide-react';
import type { ChatMessage } from '../types';

interface ChatViewProps {
  messages: ChatMessage[];
  onSendMessage: (userText: string, engine: string) => Promise<void>;
  onStopGeneration: () => void;
  isGenerating: boolean;
  selectedEngine: string;
  setSelectedEngine: (engine: string) => void;
  serverOnline: boolean;
  onNavigateToAnalysis: () => void;
}

const SUGGESTIONS = [
  {
    icon: Zap,
    title: 'KV-Cache Speedup',
    desc: 'Why KV-cache cuts autoregressive cost from O(N²) to O(N)',
    prompt: 'Explain why KV-caching reduces autoregressive generation complexity from O(N^2) to O(N) in transformer models.',
  },
  {
    icon: Code2,
    title: 'Python Memoization',
    desc: 'Generate Fibonacci memoization function in Python',
    prompt: 'Write an efficient Python function to calculate Fibonacci numbers using memoization and dynamic programming.',
  },
  {
    icon: Cpu,
    title: 'Continuous Batching',
    desc: 'Iteration-level scheduling vs static batching',
    prompt: 'What are the main differences between continuous batching and static batching in LLM serving engines like vLLM?',
  },
  {
    icon: Layers,
    title: 'Paged Attention',
    desc: 'How virtual memory eliminates VRAM fragmentation',
    prompt: 'How does Paged Attention eliminate internal and external memory fragmentation in GPU KV-caching?',
  },
];

const ENGINES = [
  { id: 'kv_cache', name: 'KV-Cache (Stage 2)', badge: 'O(N) Fast', desc: 'Saves past KV tokens in GPU VRAM' },
  { id: 'naive', name: 'Naive (Stage 1)', badge: 'O(N²) Base', desc: 'Recomputes entire sequence each step' },
  { id: 'continuous_batching', name: 'Continuous Batching', badge: 'Stage 3', desc: 'Iteration-level concurrent scheduling' },
  { id: 'paged', name: 'Paged Attention', badge: 'Stage 4', desc: 'Virtual block table allocation' },
  { id: 'quantized', name: 'INT8 Quantized', badge: 'Stage 5', desc: 'Per-channel 8-bit weight compression' },
];

export const ChatView: React.FC<ChatViewProps> = ({
  messages,
  onSendMessage,
  onStopGeneration,
  isGenerating,
  selectedEngine,
  setSelectedEngine,
  serverOnline,
  onNavigateToAnalysis,
}) => {
  const [inputText, setInputText] = useState<string>('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [engineDropdownOpen, setEngineDropdownOpen] = useState<boolean>(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setEngineDropdownOpen(false);
      }
    };

    if (engineDropdownOpen) {
      document.addEventListener('mousedown', handleClickOutside);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [engineDropdownOpen]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  const handleSend = () => {
    if (!inputText.trim() || isGenerating || !serverOnline) return;
    const text = inputText;
    setInputText('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
    onSendMessage(text, selectedEngine);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const activeEngineObj = ENGINES.find((e) => e.id === selectedEngine) || ENGINES[0];

  return (
    <div className="flex-1 flex flex-col h-full relative overflow-hidden bg-background text-on-background">
      {/* Scrollable Conversation Viewport */}
      <div className="flex-1 overflow-y-auto scrollbar-hide pt-6 pb-40 px-4 md:px-12">
        <div className="max-w-4xl mx-auto space-y-6">
          {/* Welcome Screen (When no messages yet) */}
          {messages.length === 0 && (
            <div className="py-8 md:py-14 flex flex-col items-center text-center space-y-7">
              <div className="w-11 h-11 rounded-xl bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-subtle">
                <Cpu className="w-5 h-5 text-primary" />
              </div>

              <div className="space-y-1.5 max-w-xl">
                <h2 className="text-xl font-semibold font-sans text-slate-100 tracking-tight">
                  LIEFS Inference Console
                </h2>
                <p className="text-xs md:text-sm text-slate-400 font-sans leading-relaxed">
                  Interactive latency profiling and token generation across PyTorch execution stages serving <span className="text-slate-300 font-mono">Qwen2.5-0.5B</span> with CUDA acceleration.
                </p>
              </div>

              {/* Suggestion Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
                {SUGGESTIONS.map((s, idx) => {
                  const Icon = s.icon;
                  return (
                    <button
                      key={idx}
                      onClick={() => {
                        setInputText(s.prompt);
                        if (textareaRef.current) textareaRef.current.focus();
                      }}
                      className="p-3.5 rounded-xl text-left bg-surface-container hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] transition-all group cursor-pointer shadow-subtle"
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-2">
                          <Icon className="w-3.5 h-3.5 text-primary" />
                          <span className="text-xs font-medium text-slate-200 group-hover:text-white">
                            {s.title}
                          </span>
                        </div>
                        <ArrowRight className="w-3 h-3 text-slate-500 opacity-0 group-hover:opacity-100 group-hover:translate-x-0.5 transition-all" />
                      </div>
                      <p className="text-xs text-slate-400 line-clamp-2 leading-relaxed">{s.desc}</p>
                    </button>
                  );
                })}
              </div>

              {/* Model Analysis Shortcut Banner */}
              <div
                onClick={onNavigateToAnalysis}
                className="w-full max-w-2xl p-3.5 rounded-xl bg-surface-container border border-outline-variant flex items-center justify-between hover:border-white/[0.16] hover:bg-surface-high cursor-pointer transition-all shadow-subtle group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2 rounded-lg bg-white/[0.05] border border-outline-variant text-primary">
                    <Zap className="w-4 h-4 text-primary" />
                  </div>
                  <div className="text-left">
                    <span className="text-xs font-medium text-slate-200 block">
                      Inspect Architectural Benchmarks & Sizing
                    </span>
                    <span className="text-[11px] text-slate-400 font-sans">
                      Side-by-side latency (TTFT/TPOT), throughput speedup, and GPU VRAM calculator
                    </span>
                  </div>
                </div>
                <span className="text-xs font-medium text-primary flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
                  Explore →
                </span>
              </div>
            </div>
          )}

          {/* Message List */}
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3.5 w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {/* Assistant Avatar */}
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-subtle shrink-0 mt-0.5">
                  <Cpu className="w-3.5 h-3.5 text-primary" />
                </div>
              )}

              {/* Message Bubble Container */}
              <div
                className={`max-w-[85%] md:max-w-[80%] rounded-xl p-4 text-sm leading-relaxed shadow-subtle ${
                  msg.role === 'user'
                    ? 'bg-surface-container border border-outline-variant text-slate-100 rounded-tr-sm font-sans'
                    : 'bg-surface-low border border-outline-variant text-slate-200 rounded-tl-sm font-sans'
                }`}
              >
                {/* Message Body Content */}
                <div className="whitespace-pre-wrap font-sans text-sm text-slate-200">
                  {msg.content}
                  {msg.isStreaming && <span className="terminal-cursor" />}
                </div>

                {/* Assistant Telemetry HUD Badge */}
                {msg.role === 'assistant' && msg.metrics && (
                  <div className="mt-3 pt-2.5 border-t border-outline-variant flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant text-primary font-medium tabular-nums">
                        {(msg.metrics.tokens_per_sec || 0).toFixed(1)} tok/s
                      </span>
                      <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant text-slate-300 tabular-nums">
                        TTFT: {msg.metrics.ttft_ms.toFixed(1)}ms
                      </span>
                      <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant text-slate-400 tabular-nums">
                        TPOT: {msg.metrics.tpot_ms.toFixed(1)}ms
                      </span>
                      {msg.metrics.peak_vram_mb > 0 && (
                        <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant text-slate-400 tabular-nums">
                          VRAM: {Math.round(msg.metrics.peak_vram_mb)}MB
                        </span>
                      )}
                      {msg.engine && (
                        <span className="px-1.5 py-0.5 rounded bg-white/[0.04] text-slate-400 text-[10px] border border-outline-variant">
                          {msg.engine}
                        </span>
                      )}
                    </div>

                    <button
                      onClick={() => handleCopy(msg.id, msg.content)}
                      className="p-1 rounded hover:bg-white/[0.06] hover:text-slate-200 transition-colors cursor-pointer text-slate-400"
                      title="Copy response"
                    >
                      {copiedId === msg.id ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                )}
              </div>

              {/* User Avatar */}
              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-lg bg-surface-container border border-outline-variant flex items-center justify-center shrink-0 mt-0.5 shadow-subtle">
                  <User className="w-3.5 h-3.5 text-slate-300" />
                </div>
              )}
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Floating Bottom Input Bar */}
      <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-background via-background/95 to-transparent pt-6 pb-4 px-4 md:px-12 z-30">
        <div className="max-w-4xl mx-auto space-y-2">
          {/* Main Input Container Card */}
          <div className="bg-surface-container rounded-xl border border-outline-variant shadow-card p-2 flex flex-col focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20 transition-all">
            {/* Input Textarea */}
            <textarea
              ref={textareaRef}
              id="chat-message-input"
              name="chatMessage"
              value={inputText}
              onChange={(e) => {
                setInputText(e.target.value);
                e.target.style.height = 'auto';
                e.target.style.height = `${Math.min(e.target.scrollHeight, 120)}px`;
              }}
              onKeyDown={handleKeyDown}
              disabled={isGenerating || !serverOnline}
              placeholder={serverOnline ? "Ask LIEFS or run a prompt (e.g. 'Write a Python function to calculate Fibonacci')..." : "Inference Engine Disconnected..."}
              rows={1}
              className="w-full bg-transparent border-none focus:ring-0 text-slate-100 placeholder:text-slate-500 resize-none max-h-32 min-h-[44px] py-2 px-3 text-sm font-sans scrollbar-hide outline-none"
            />

            {/* Bottom Input Controls Ribbon */}
            <div className="flex items-center justify-between pt-2 px-2 border-t border-outline-variant">
              {/* Engine Selector Dropdown Pill */}
              <div className="relative" ref={dropdownRef}>
                <button
                  type="button"
                  onClick={() => setEngineDropdownOpen(!engineDropdownOpen)}
                  disabled={isGenerating}
                  className="px-2.5 py-1 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] text-xs font-sans text-slate-300 hover:text-slate-100 flex items-center gap-1.5 transition-all cursor-pointer shadow-subtle"
                >
                  <Cpu className="w-3.5 h-3.5 text-primary" />
                  <span className="font-medium text-slate-200">{activeEngineObj.name}</span>
                  <ChevronDown className="w-3 h-3 text-slate-400" />
                </button>

                {engineDropdownOpen && (
                  <div className="absolute bottom-full mb-2 left-0 w-72 bg-surface-low border border-outline-variant rounded-xl shadow-popover p-1.5 z-50 font-sans text-xs space-y-0.5">
                    <div className="px-2.5 py-1 text-[10px] text-slate-400 font-mono uppercase tracking-wider border-b border-outline-variant pb-1 mb-1">
                      Inference Strategy
                    </div>
                    {ENGINES.map((eng) => (
                      <button
                        key={eng.id}
                        type="button"
                        onClick={() => {
                          setSelectedEngine(eng.id);
                          setEngineDropdownOpen(false);
                        }}
                        className={`w-full px-2.5 py-1.5 rounded-lg flex items-center justify-between text-left transition-all cursor-pointer ${
                          selectedEngine === eng.id
                            ? 'bg-surface-high text-slate-100 font-medium border border-outline-variant shadow-subtle'
                            : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200 border border-transparent'
                        }`}
                      >
                        <div className="flex flex-col">
                          <span className="text-xs font-medium text-slate-200">{eng.name}</span>
                          <span className="text-[11px] text-slate-400 font-sans">{eng.desc}</span>
                        </div>
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-surface-lowest border border-outline-variant text-slate-400 shrink-0 ml-2">
                          {eng.badge}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* Action Buttons: Send or Stop */}
              <div className="flex items-center gap-2">
                {isGenerating ? (
                  <button
                    onClick={onStopGeneration}
                    className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-sans text-xs font-medium flex items-center gap-1.5 shadow-subtle transition-all cursor-pointer"
                  >
                    <Square className="w-3 h-3 fill-current" />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    onClick={handleSend}
                    disabled={!inputText.trim() || !serverOnline}
                    className={`p-1.5 rounded-lg transition-all flex items-center justify-center h-7 w-7 ${
                      inputText.trim() && serverOnline
                        ? 'bg-primary text-slate-950 hover:bg-sky-300 shadow-subtle cursor-pointer active:scale-95'
                        : 'bg-white/[0.03] border border-outline-variant text-slate-500 cursor-not-allowed'
                    }`}
                    title="Send message"
                  >
                    <Send className={`w-3.5 h-3.5 ${inputText.trim() ? 'fill-current' : ''}`} />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Disclaimer Footer */}
          <div className="text-center">
            <span className="text-[10px] font-mono text-slate-500">
              LIEFS Engine • Qwen2.5-0.5B-Instruct • PyTorch CUDA
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
