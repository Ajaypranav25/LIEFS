import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatView } from './components/ChatView';
import { ModelAnalysisView } from './components/ModelAnalysisView';
import { fetchHealth, fetchSystemInfo, streamCompletion } from './lib/api';
import type { SystemInfo, ChatSession, ChatMessage, GenerationMetrics } from './types';

const INITIAL_SESSIONS: ChatSession[] = [
  {
    id: 'session-1',
    title: 'KV-Cache Speedup Analysis',
    createdAt: Date.now() - 3600000,
    engine: 'kv_cache',
    messages: [
      {
        id: 'msg-1',
        role: 'user',
        content: 'Explain why KV-caching reduces autoregressive generation complexity from O(N^2) to O(N).',
        timestamp: Date.now() - 3600000,
      },
      {
        id: 'msg-2',
        role: 'assistant',
        content: `In autoregressive generation, each new token requires attending to all prior tokens.\n\n### Without KV-Cache (Stage 1 Naive Baseline):\nAt step $t$, the full sequence $x_{1:t}$ is passed through all transformer layers. Keys and values for all past tokens $1 \\dots t-1$ are **recomputed from scratch**, leading to total compute complexity:\n\\[ \\sum_{t=1}^N \\mathcal{O}(t) = \\mathcal{O}(N^2) \\]\n\n### With KV-Cache (Stage 2 Optimized):\nWe store past Keys ($K_{1:t-1}$) and Values ($V_{1:t-1}$) in GPU VRAM. At step $t$, we only pass the single newly generated token $x_t$ through the model, compute $(q_t, k_t, v_t)$, append $k_t, v_t$ to the cache, and compute attention in $\\mathcal{O}(1)$ query vector operations.\n\nTotal generation complexity drops to:\n\\[ \\sum_{t=1}^N \\mathcal{O}(1) = \\mathcal{O}(N) \\]\n\nThis yields a **3.5x to 8x throughput speedup** on Qwen2.5-0.5B on CUDA.`,
        timestamp: Date.now() - 3590000,
        engine: 'KV-Cache (Stage 2)',
        metrics: {
          ttft_ms: 14.8,
          tpot_ms: 10.4,
          total_time_ms: 540.2,
          tokens_per_sec: 96.1,
          peak_vram_mb: 998.5,
          generated_tokens: 52,
          prompt_tokens: 28,
        },
      },
    ],
  },
];

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<'chat' | 'analysis'>('chat');
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    try {
      const saved = localStorage.getItem('liefs_chat_sessions');
      return saved ? JSON.parse(saved) : INITIAL_SESSIONS;
    } catch {
      return INITIAL_SESSIONS;
    }
  });
  const [activeSessionId, setActiveSessionId] = useState<string>(() => {
    return sessions[0]?.id || 'session-1';
  });
  const [selectedEngine, setSelectedEngine] = useState<string>('kv_cache');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [sidebarOpen, setSidebarOpen] = useState<boolean>(false);

  // System & Health State
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean>(true);

  // Cancel generation reference
  const cancelStreamRef = useRef<(() => void) | null>(null);

  // Sync sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('liefs_chat_sessions', JSON.stringify(sessions));
    } catch (e) {
      console.warn('Failed to persist sessions:', e);
    }
  }, [sessions]);

  // Telemetry Polling Loop
  useEffect(() => {
    const poll = async () => {
      try {
        const [health, sys] = await Promise.all([fetchHealth(), fetchSystemInfo()]);
        setServerOnline(health.status === 'ok');
        setSystemInfo(sys);
      } catch {
        setServerOnline(false);
      }
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  const currentSession = sessions.find((s) => s.id === activeSessionId) || sessions[0];
  const messages = currentSession ? currentSession.messages : [];

  const handleNewChat = () => {
    const newSession: ChatSession = {
      id: `session-${Date.now()}`,
      title: 'New Conversation',
      createdAt: Date.now(),
      engine: selectedEngine,
      messages: [],
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
    setActiveView('chat');
    setSidebarOpen(false);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    const filtered = sessions.filter((s) => s.id !== id);
    if (filtered.length === 0) {
      const fresh: ChatSession = {
        id: `session-${Date.now()}`,
        title: 'New Conversation',
        createdAt: Date.now(),
        engine: selectedEngine,
        messages: [],
      };
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
    } else {
      setSessions(filtered);
      if (activeSessionId === id) {
        setActiveSessionId(filtered[0].id);
      }
    }
  };

  const handleSendMessage = async (userText: string, engine: string) => {
    if (isGenerating || !userText.trim()) return;

    const userMsg: ChatMessage = {
      id: `msg-${Date.now()}-u`,
      role: 'user',
      content: userText,
      timestamp: Date.now(),
    };

    const assistantMsgId = `msg-${Date.now()}-a`;
    const assistantMsgPlaceholder: ChatMessage = {
      id: assistantMsgId,
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
      engine,
      isStreaming: true,
    };

    // Update session title if first message
    const updatedTitle =
      currentSession.messages.length === 0
        ? userText.slice(0, 30) + (userText.length > 30 ? '...' : '')
        : currentSession.title;

    setSessions((prev) =>
      prev.map((s) => {
        if (s.id === currentSession.id) {
          return {
            ...s,
            title: updatedTitle,
            messages: [...s.messages, userMsg, assistantMsgPlaceholder],
          };
        }
        return s;
      })
    );

    setIsGenerating(true);

    let fullText = '';

    try {
      const cancelFn = await streamCompletion(
        userText,
        {
          maxTokens: 256,
          temperature: 0.7,
          engine,
        },
        {
          onToken: (token: string, metrics?: GenerationMetrics) => {
            fullText += token;
            setSessions((prev) =>
              prev.map((s) => {
                if (s.id === currentSession.id) {
                  return {
                    ...s,
                    messages: s.messages.map((m) => {
                      if (m.id === assistantMsgId) {
                        return {
                          ...m,
                          content: fullText,
                          metrics: metrics || m.metrics,
                        };
                      }
                      return m;
                    }),
                  };
                }
                return s;
              })
            );
          },
          onMetrics: (metrics: GenerationMetrics) => {
            setSessions((prev) =>
              prev.map((s) => {
                if (s.id === currentSession.id) {
                  return {
                    ...s,
                    messages: s.messages.map((m) => {
                      if (m.id === assistantMsgId) {
                        return {
                          ...m,
                          metrics,
                        };
                      }
                      return m;
                    }),
                  };
                }
                return s;
              })
            );
          },
          onDone: (completedText: string, finalMetrics: GenerationMetrics) => {
            setSessions((prev) =>
              prev.map((s) => {
                if (s.id === currentSession.id) {
                  return {
                    ...s,
                    messages: s.messages.map((m) => {
                      if (m.id === assistantMsgId) {
                        return {
                          ...m,
                          content: completedText || fullText,
                          metrics: finalMetrics,
                          isStreaming: false,
                        };
                      }
                      return m;
                    }),
                  };
                }
                return s;
              })
            );
            setIsGenerating(false);
            cancelStreamRef.current = null;
          },
          onError: (error: Error) => {
            console.error('Streaming error:', error);
            setIsGenerating(false);
            cancelStreamRef.current = null;
          },
        }
      );

      cancelStreamRef.current = cancelFn;
    } catch (err: any) {
      console.error('Failed to start stream:', err);
      setIsGenerating(false);
      cancelStreamRef.current = null;
    }
  };

  const handleStopGeneration = () => {
    if (cancelStreamRef.current) {
      cancelStreamRef.current();
      cancelStreamRef.current = null;
    }
    setIsGenerating(false);
  };

  return (
    <div className="h-screen w-screen flex overflow-hidden bg-background text-on-background font-sans">
      {/* ChatGPT / Gemini Style Left Sidebar */}
      <Sidebar
        activeView={activeView}
        setActiveView={setActiveView}
        sessions={sessions}
        activeSessionId={activeSessionId}
        onSelectSession={(id) => {
          setActiveSessionId(id);
          setActiveView('chat');
          setSidebarOpen(false);
        }}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
        systemInfo={systemInfo}
        serverOnline={serverOnline}
        isOpen={sidebarOpen}
        setIsOpen={setSidebarOpen}
      />

      {/* Main Content Area (16:9 Laptop Ratio Canvas) */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Top Header */}
        <Header
          activeView={activeView}
          setActiveView={setActiveView}
          systemInfo={systemInfo}
          serverOnline={serverOnline}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        />

        {/* View Switcher: AI Chat vs Model Analysis */}
        {activeView === 'chat' ? (
          <ChatView
            messages={messages}
            onSendMessage={handleSendMessage}
            onStopGeneration={handleStopGeneration}
            isGenerating={isGenerating}
            selectedEngine={selectedEngine}
            setSelectedEngine={setSelectedEngine}
            serverOnline={serverOnline}
            onNavigateToAnalysis={() => setActiveView('analysis')}
          />
        ) : (
          <ModelAnalysisView
            serverOnline={serverOnline}
            onBackToChat={() => setActiveView('chat')}
          />
        )}
      </div>
    </div>
  );
};

export default App;
