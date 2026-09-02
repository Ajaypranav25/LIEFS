import React, { useState, useRef, useEffect } from 'react';
import {
  Send,
  Square,
  Sparkles,
  Cpu,
  User,
  Copy,
  Check,
  Zap,
  ChevronDown,
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
    title: '⚡ KV-Cache Speedup',
    desc: 'Why KV-cache cuts autoregressive cost from O(N²) to O(N)',
    prompt: 'Explain why KV-caching reduces autoregressive generation complexity from O(N^2) to O(N) in transformer models.',
  },
  {
    title: '🐍 Python Algorithm',
    desc: 'Generate Fibonacci memoization function in Python',
    prompt: 'Write an efficient Python function to calculate Fibonacci numbers using memoization and dynamic programming.',
  },
  {
    title: '🔄 Continuous Batching',
    desc: 'Iteration-level scheduling vs static batching',
    prompt: 'What are the main differences between continuous batching and static batching in LLM serving engines like vLLM?',
  },
  {
    title: '💾 Paged Attention',
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
            <div className="py-8 md:py-14 flex flex-col items-center text-center space-y-8 animate-fadeIn">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-tr from-primary-container via-primary to-secondary flex items-center justify-center text-on-primary shadow-glow-cyan">
                <Cpu className="w-9 h-9 text-slate-950" />
              </div>

              <div className="space-y-2 max-w-xl">
                <h2 className="text-2xl md:text-3xl font-bold font-sans text-on-background tracking-tight">
                  How can I help you today?
                </h2>
                <p className="text-xs md:text-sm text-on-surface-variant font-sans">
                  Query the <strong className="text-primary font-mono">LIEFS</strong> custom inference engine serving{' '}
                  <span className="text-secondary font-mono">Qwen2.5-0.5B-Instruct</span> with raw PyTorch CUDA acceleration.
                </p>
              </div>

              {/* Suggestion Cards Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 w-full max-w-2xl">
                {SUGGESTIONS.map((s, idx) => (
                  <button
                    key={idx}
                    onClick={() => {
                      setInputText(s.prompt);
                      if (textareaRef.current) textareaRef.current.focus();
                    }}
                    className="glass-card p-4 rounded-xl text-left hover:border-primary/50 hover:shadow-glow-cyan transition-all duration-200 group cursor-pointer active:scale-95"
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <span className="text-xs font-mono font-bold text-primary group-hover:text-primary-light">
                        {s.title}
                      </span>
                      <Sparkles className="w-3.5 h-3.5 text-secondary opacity-70 group-hover:opacity-100" />
                    </div>
                    <p className="text-xs text-on-surface-variant line-clamp-2">{s.desc}</p>
                  </button>
                ))}
              </div>

              {/* Model Analysis Shortcut Banner */}
              <div
                onClick={onNavigateToAnalysis}
                className="w-full max-w-2xl p-4 rounded-xl bg-surface-container-high border border-outline-variant/30 flex items-center justify-between hover:border-primary/40 cursor-pointer transition-all shadow-sm group"
              >
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-primary/20 text-primary group-hover:scale-105 transition-transform">
                    <Zap className="w-5 h-5 text-primary" />
                  </div>
                  <div className="text-left">
                    <span className="text-xs font-mono font-bold text-on-background block">
                      Inspect Architectural Benchmarks & Sizing
                    </span>
                    <span className="text-[11px] text-on-surface-variant font-sans">
                      Side-by-side latency (TTFT/TPOT), throughput speedup, and GPU VRAM calculator
                    </span>
                  </div>
                </div>
                <span className="text-xs font-mono text-primary font-bold flex items-center gap-1 group-hover:translate-x-0.5 transition-transform">
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
              {/* Assistant Avatar - Matches Top-Left Logo */}
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-xl bg-primary/20 border border-primary/40 flex items-center justify-center text-primary shadow-glow-cyan shrink-0 mt-0.5">
                  <Cpu className="w-4.5 h-4.5 text-primary" />
                </div>
              )}

              {/* Message Bubble Container */}
              <div
                className={`max-w-[85%] md:max-w-[80%] rounded-2xl p-4 shadow-sm text-sm leading-relaxed ${
                  msg.role === 'user'
                    ? 'bg-surface-container-high border border-outline-variant/40 text-on-background rounded-tr-sm font-sans'
                    : 'glass-card border border-outline-variant/20 text-on-background rounded-tl-sm font-sans'
                }`}
              >
                {/* Message Body Content */}
                <div className="whitespace-pre-wrap font-sans text-sm text-slate-100">
                  {msg.content}
                  {msg.isStreaming && <span className="terminal-cursor" />}
                </div>

                {/* Assistant Telemetry HUD Badge */}
                {msg.role === 'assistant' && msg.metrics && (
                  <div className="mt-3 pt-2.5 border-t border-surface-container flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-on-surface-variant">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant/40 text-primary font-bold">
                        ⚡ {(msg.metrics.tokens_per_sec || 0).toFixed(1)} t/s
                      </span>
                      <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant/40 text-secondary">
                        TTFT: {msg.metrics.ttft_ms.toFixed(1)}ms
                      </span>
                      <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant/40 text-on-background">
                        TPOT: {msg.metrics.tpot_ms.toFixed(1)}ms
                      </span>
                      {msg.metrics.peak_vram_mb > 0 && (
                        <span className="px-2 py-0.5 rounded bg-surface-lowest border border-outline-variant/40 text-tertiary">
                          VRAM: {Math.round(msg.metrics.peak_vram_mb)}MB
                        </span>
                      )}
                      {msg.engine && (
                        <span className="px-1.5 py-0.5 rounded bg-surface-container-highest text-slate-300 text-[10px]">
                          {msg.engine}
                        </span>
                      )}
                    </div>

                    <button
                      onClick={() => handleCopy(msg.id, msg.content)}
                      className="p-1 rounded hover:bg-surface-high hover:text-primary transition-colors cursor-pointer"
                      title="Copy response"
                    >
                      {copiedId === msg.id ? (
                        <Check className="w-3.5 h-3.5 text-primary" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                )}
              </div>

              {/* User Avatar */}
              {msg.role === 'user' && (
                <div className="w-8 h-8 rounded-xl bg-secondary-container border border-secondary/40 flex items-center justify-center shrink-0 mt-0.5">
                  <User className="w-4 h-4 text-secondary-light" />
                </div>
              )}
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Floating Bottom Input Bar (ChatGPT / Gemini Style) */}
      <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-background via-background/95 to-transparent pt-6 pb-4 px-4 md:px-12 z-30">
        <div className="max-w-4xl mx-auto space-y-2">
          {/* Main Input Container Card */}
          <div className="glass-card rounded-2xl border border-outline-variant/40 shadow-glass-card p-2 flex flex-col focus-within:border-primary focus-within:ring-1 focus-within:ring-primary/60 transition-all">
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
              placeholder={serverOnline ? "Message LIEFS Engine (e.g. 'Write a Python function to calculate Fibonacci')..." : "Inference Engine Disconnected..."}
              rows={1}
              className="w-full bg-transparent border-none focus:ring-0 text-on-background placeholder:text-outline/70 resize-none max-h-32 min-h-[44px] py-2 px-3 text-sm font-sans scrollbar-hide outline-none"
            />

            {/* Bottom Input Controls Ribbon */}
            <div className="flex items-center justify-between pt-2 px-2 border-t border-surface-container/60">
              {/* Engine Selector Dropdown Pill */}
              <div className="relative" ref={dropdownRef}>
                <button
                  type="button"
                  onClick={() => setEngineDropdownOpen(!engineDropdownOpen)}
                  disabled={isGenerating}
                  className="px-2.5 py-1.5 rounded-lg bg-surface-lowest border border-outline-variant/50 hover:border-primary text-xs font-mono text-on-surface-variant hover:text-primary flex items-center gap-1.5 transition-all cursor-pointer shadow-sm"
                >
                  <Cpu className="w-3.5 h-3.5 text-primary" />
                  <span className="font-semibold text-primary">{activeEngineObj.name}</span>
                  <ChevronDown className="w-3 h-3 text-slate-400" />
                </button>

                {engineDropdownOpen && (
                  <div className="absolute bottom-full mb-2 left-0 w-72 bg-[#081326]/95 backdrop-blur-3xl border border-primary/50 rounded-xl shadow-[0_12px_40px_rgba(0,0,0,0.9)] p-2 z-50 font-mono text-xs space-y-1">
                    <div className="px-2 py-1 text-[10px] text-primary uppercase font-bold tracking-wider border-b border-surface-container/80 pb-1 mb-1">
                      Select Inference Strategy
                    </div>
                    {ENGINES.map((eng) => (
                      <button
                        key={eng.id}
                        type="button"
                        onClick={() => {
                          setSelectedEngine(eng.id);
                          setEngineDropdownOpen(false);
                        }}
                        className={`w-full px-2.5 py-2 rounded-lg flex items-center justify-between text-left transition-all cursor-pointer ${
                          selectedEngine === eng.id
                            ? 'bg-primary-container text-on-primary-container font-bold shadow-glow-cyan'
                            : 'text-on-surface-variant hover:bg-surface-high hover:text-on-background'
                        }`}
                      >
                        <div className="flex flex-col">
                          <span className="text-xs font-semibold">{eng.name}</span>
                          <span className="text-[10px] text-slate-400 font-sans">{eng.desc}</span>
                        </div>
                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-surface-dim border border-outline-variant/40 text-primary shrink-0 ml-2">
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
                    className="px-3 py-1.5 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-mono text-xs font-bold flex items-center gap-1.5 shadow-lg shadow-rose-950 transition-all cursor-pointer"
                  >
                    <Square className="w-3.5 h-3.5 fill-current" />
                    <span>Stop</span>
                  </button>
                ) : (
                  <button
                    onClick={handleSend}
                    disabled={!inputText.trim() || !serverOnline}
                    className={`p-2 rounded-xl transition-all flex items-center justify-center h-8 w-8 ${
                      inputText.trim() && serverOnline
                        ? 'bg-primary text-slate-950 hover:bg-primary-light shadow-glow-cyan cursor-pointer active:scale-95'
                        : 'bg-surface-high border border-outline-variant/60 text-on-surface-variant/80 hover:text-primary hover:border-primary/50 cursor-not-allowed'
                    }`}
                    title="Send message"
                  >
                    <Send className={`w-4 h-4 ${inputText.trim() ? 'fill-current' : ''}`} />
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Disclaimer Footer */}
          <div className="text-center">
            <span className="text-[10px] font-mono text-outline/80">
              LIEFS Engine v1.0 • Qwen2.5-0.5B-Instruct • Local PyTorch CUDA
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
