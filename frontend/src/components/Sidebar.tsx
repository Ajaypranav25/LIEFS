import React from 'react';
import {
  MessageSquarePlus,
  BarChart2,
  Cpu,
  Trash2,
} from 'lucide-react';
import type { SystemInfo, ChatSession } from '../types';

interface SidebarProps {
  activeView: 'chat' | 'analysis';
  setActiveView: (view: 'chat' | 'analysis') => void;
  sessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession: (id: string, e: React.MouseEvent) => void;
  systemInfo: SystemInfo | null;
  serverOnline: boolean;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeView,
  setActiveView,
  sessions,
  activeSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
  systemInfo,
  serverOnline,
  isOpen,
}) => {
  return (
    <aside
      className={`fixed md:static inset-y-0 left-0 z-40 w-64 bg-surface-low border-r border-surface-container flex flex-col transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
      }`}
    >
      {/* Brand / Logo */}
      <div className="p-4 border-b border-surface-container flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-primary/20 border border-primary/40 flex items-center justify-center text-primary shadow-glow-cyan">
            <Cpu className="w-5 h-5 text-primary" />
          </div>
          <div>
            <h1 className="font-sans font-bold text-sm text-primary tracking-tight flex items-center gap-1.5">
              <span>LIEFS Studio</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary-light border border-primary/30 font-mono">
                v1.0
              </span>
            </h1>
            <p className="text-[11px] text-on-surface-variant font-mono">Qwen2.5-0.5B • CUDA</p>
          </div>
        </div>
      </div>

      {/* New Chat & Primary Navigation */}
      <div className="p-3 space-y-2">
        <button
          onClick={() => {
            setActiveView('chat');
            onNewChat();
          }}
          className="w-full bg-primary/15 hover:bg-primary/25 text-primary border border-primary/40 font-medium text-xs py-2.5 px-3.5 rounded-xl flex items-center justify-center gap-2 transition-all shadow-glow-cyan cursor-pointer active:scale-95"
        >
          <MessageSquarePlus className="w-4 h-4 text-primary" />
          <span>New Chat</span>
        </button>

        {/* Model Analysis Nav Pill */}
        <button
          onClick={() => setActiveView('analysis')}
          className={`w-full text-xs font-medium py-2.5 px-3.5 rounded-xl flex items-center justify-between transition-all cursor-pointer ${
            activeView === 'analysis'
              ? 'bg-secondary-container text-primary border border-primary/50 shadow-glow-cyan font-bold'
              : 'text-on-surface-variant hover:bg-surface-high hover:text-on-background'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <BarChart2 className="w-4 h-4 text-primary" />
            <span>Model Analysis</span>
          </div>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-primary/20 text-primary font-bold">
            PRO
          </span>
        </button>
      </div>

      {/* Recent Chats Section */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-3 py-2 space-y-1">
        <div className="px-2 py-1.5 flex items-center justify-between text-[11px] font-mono uppercase tracking-wider text-outline">
          <span>Recent Chats</span>
          <span className="text-[10px] text-slate-500">{sessions.length}</span>
        </div>

        {sessions.map((session) => {
          const isActive = activeView === 'chat' && activeSessionId === session.id;
          return (
            <div
              key={session.id}
              onClick={() => {
                setActiveView('chat');
                onSelectSession(session.id);
              }}
              className={`group flex items-center justify-between px-3 py-2 rounded-lg text-xs font-sans transition-all cursor-pointer ${
                isActive
                  ? 'bg-surface-high text-primary font-semibold border border-outline-variant/50'
                  : 'text-on-surface-variant hover:bg-surface-bright hover:text-on-background'
              }`}
            >
              <span className="truncate flex-1 pr-2">{session.title || 'New Conversation'}</span>
              <button
                onClick={(e) => onDeleteSession(session.id, e)}
                className="opacity-0 group-hover:opacity-100 hover:text-rose-400 p-1 transition-opacity"
                title="Delete Chat"
              >
                <Trash2 className="w-3 h-3" />
              </button>
            </div>
          );
        })}
      </div>

      {/* Bottom Footer Telemetry HUD */}
      <div className="p-3 border-t border-surface-container bg-surface-dim space-y-2.5 font-mono text-xs">
        {/* Engine Status Badge */}
        <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-surface-lowest border border-outline-variant/40">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                serverOnline ? 'bg-primary status-led animate-pulse' : 'bg-rose-500'
              }`}
            />
            <span className={serverOnline ? 'text-primary font-semibold' : 'text-rose-400'}>
              {serverOnline ? 'Engine Online' : 'Offline'}
            </span>
          </div>
          <span className="text-[10px] text-on-surface-variant">RTX 4060</span>
        </div>

        {/* GPU VRAM Bar */}
        {systemInfo && systemInfo.cuda_available && (
          <div className="px-2.5 py-1.5 rounded-lg bg-surface-lowest border border-outline-variant/40 space-y-1">
            <div className="flex justify-between text-[11px] text-on-surface-variant">
              <span>VRAM Usage</span>
              <span className="text-secondary font-bold">
                {(systemInfo.vram_allocated_mb / 1024).toFixed(1)}GB / {(systemInfo.vram_total_mb / 1024).toFixed(0)}GB
              </span>
            </div>
            <div className="w-full h-1 bg-surface-high rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary to-secondary transition-all duration-500"
                style={{ width: `${Math.min(100, systemInfo.vram_usage_percent)}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};
