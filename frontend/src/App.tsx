import React, { useState, useEffect, useRef } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ChatView } from './components/ChatView';
import { ModelAnalysisView } from './components/ModelAnalysisView';
import { ModelLoaderModal } from './components/ModelLoaderModal';
import { AuthProvider, useAuth } from './context/AuthContext';
import {
  fetchHealth,
  fetchSystemInfo,
  fetchHardwareInfo,
  fetchCurrentModel,
  streamCompletion,
} from './lib/api';
import {
  fetchUserChatSessions,
  saveChatSessionToDb,
  saveChatMessageToDb,
  deleteChatSessionFromDb,
} from './lib/supabase';
import type {
  SystemInfo,
  HardwareProfile,
  ModelArchitectureMetadata,
  ChatSession,
  ChatMessage,
  GenerationMetrics,
} from './types';

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
        content: `In autoregressive generation, each new token requires attending to all prior tokens.\n\n### Without KV-Cache (Stage 1 Naive Baseline):\nAt step $t$, the full sequence $x_{1:t}$ is passed through all transformer layers. Keys and values for all past tokens $1 \\dots t-1$ are **recomputed from scratch**, leading to total compute complexity:\n\\[ \\sum_{t=1}^N \\mathcal{O}(t) = \\mathcal{O}(N^2) \\]\n\n### With KV-Cache (Stage 2 Optimized):\nWe store past Keys ($K_{1:t-1}$) and Values ($V_{1:t-1}$) in GPU VRAM. At step $t$, we only pass the single newly generated token $x_t$ through the model, compute $(q_t, k_t, v_t)$, append $k_t, v_t$ to the cache, and compute attention in $\\mathcal{O}(1)$ query vector operations.\n\nTotal generation complexity drops to:\n\\[ \\sum_{t=1}^N \\mathcal{O}(1) = \\mathcal{O}(N) \\]\n\nThis yields a **3.5x to 8x throughput speedup** on local consumer GPUs.`,
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

const AppContent: React.FC = () => {
  const { user } = useAuth();
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
  const [isModelLoaderOpen, setIsModelLoaderOpen] = useState<boolean>(false);

  // System, Hardware & Model State
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [hardware, setHardware] = useState<HardwareProfile | null>(null);
  const [modelMeta, setModelMeta] = useState<ModelArchitectureMetadata | null>(null);
  const [serverOnline, setServerOnline] = useState<boolean>(true);

  // Cancel generation reference
  const cancelStreamRef = useRef<(() => void) | null>(null);

  // Load user's chat sessions from Supabase when signed in
  useEffect(() => {
    const syncUserSessions = async () => {
      try {
        const cloudSessions = await fetchUserChatSessions(user?.id);
        if (cloudSessions && cloudSessions.length > 0) {
          setSessions(cloudSessions);
          if (!cloudSessions.some((s) => s.id === activeSessionId)) {
            setActiveSessionId(cloudSessions[0].id);
          }
        }
      } catch {
        console.warn('Failed to sync cloud sessions');
      }
    };

    syncUserSessions();
  }, [user?.id]);

  // Sync sessions to localStorage
  useEffect(() => {
    try {
      localStorage.setItem('liefs_chat_sessions', JSON.stringify(sessions));
    } catch {
      console.warn('Failed to persist sessions:', e);
    }
  }, [sessions]);

  // Initial Hardware and Model Profile Fetch
  useEffect(() => {
    const fetchEnv = async () => {
      try {
        const [hw, meta] = await Promise.all([
          fetchHardwareInfo().catch(() => null),
          fetchCurrentModel().catch(() => null),
        ]);
        if (hw) setHardware(hw);
        if (meta) setModelMeta(meta);
      } catch {
        console.warn('Failed to fetch environment:', e);
      }
    };
    fetchEnv();
  }, []);

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
      user_id: user?.id,
      messages: [],
    };
    setSessions([newSession, ...sessions]);
    setActiveSessionId(newSession.id);
    setActiveView('chat');
    setSidebarOpen(false);

    // Save new session to Supabase
    saveChatSessionToDb(newSession, user?.id).catch(console.warn);
  };

  const handleDeleteSession = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    deleteChatSessionFromDb(id).catch(console.warn);

    const filtered = sessions.filter((s) => s.id !== id);
    if (filtered.length === 0) {
      const fresh: ChatSession = {
        id: `session-${Date.now()}`,
        title: 'New Conversation',
        createdAt: Date.now(),
        engine: selectedEngine,
        user_id: user?.id,
        messages: [],
      };
      setSessions([fresh]);
      setActiveSessionId(fresh.id);
      saveChatSessionToDb(fresh, user?.id).catch(console.warn);
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

    const updatedSession: ChatSession = {
      ...currentSession,
      title: updatedTitle,
      user_id: user?.id,
      messages: [...currentSession.messages, userMsg, assistantMsgPlaceholder],
    };

    setSessions((prev) =>
      prev.map((s) => (s.id === currentSession.id ? updatedSession : s))
    );

    setIsGenerating(true);

    // Save session metadata and user message to Supabase
    saveChatSessionToDb(updatedSession, user?.id).catch(console.warn);
    saveChatMessageToDb(currentSession.id, userMsg, user?.id).catch(console.warn);

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
            const finalAssistantMsg: ChatMessage = {
              id: assistantMsgId,
              role: 'assistant',
              content: completedText || fullText,
              timestamp: Date.now(),
              engine,
              metrics: finalMetrics,
              isStreaming: false,
            };

            setSessions((prev) =>
              prev.map((s) => {
                if (s.id === currentSession.id) {
                  return {
                    ...s,
                    messages: s.messages.map((m) =>
                      m.id === assistantMsgId ? finalAssistantMsg : m
                    ),
                  };
                }
                return s;
              })
            );

            // Persist complete assistant response to Supabase
            saveChatMessageToDb(currentSession.id, finalAssistantMsg, user?.id).catch(console.warn);

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
      {/* Sidebar */}
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

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden relative">
        {/* Header */}
        <Header
          activeView={activeView}
          setActiveView={setActiveView}
          systemInfo={systemInfo}
          hardware={hardware}
          modelMeta={modelMeta}
          serverOnline={serverOnline}
          onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
          onOpenModelLoader={() => setIsModelLoaderOpen(true)}
        />

        {/* View Switcher: AI Chat vs Universal Model Analysis & Benchmarks */}
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
            hardware={hardware}
            modelMeta={modelMeta}
            onBackToChat={() => setActiveView('chat')}
            onOpenModelLoader={() => setIsModelLoaderOpen(true)}
          />
        )}
      </div>

      {/* Custom Model Importer & Hub Modal */}
      <ModelLoaderModal
        isOpen={isModelLoaderOpen}
        onClose={() => setIsModelLoaderOpen(false)}
        currentModelName={modelMeta?.model_name || systemInfo?.model_name || 'Qwen/Qwen2.5-0.5B-Instruct'}
        onModelLoaded={(meta) => {
          setModelMeta(meta);
          // Refresh system info
          fetchSystemInfo().then(setSystemInfo).catch(console.warn);
          fetchHardwareInfo().then(setHardware).catch(console.warn);
        }}
      />
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
};

export default App;
