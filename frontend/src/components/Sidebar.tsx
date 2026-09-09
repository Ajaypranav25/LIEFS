import React from 'react';
import {
  MessageSquarePlus,
  BarChart2,
  Cpu,
  Trash2,
} from 'lucide-react';
import { UserAuth } from './UserAuth';
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
      className={`fixed md:static inset-y-0 left-0 z-40 w-64 bg-surface-low border-r border-outline-variant flex flex-col transition-transform duration-300 ease-in-out ${
        isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
      }`}
    >
      {/* Brand / Logo */}
      <div className="p-4 border-b border-outline-variant flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-surface-container border border-outline-variant flex items-center justify-center text-primary shadow-subtle">
            <Cpu className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h1 className="font-sans font-semibold text-sm text-slate-100 tracking-tight flex items-center gap-1.5">
              <span>LIEFS Studio</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/[0.05] text-slate-400 border border-outline-variant font-mono">
                v1.0
              </span>
            </h1>
            <p className="text-[11px] text-slate-400 font-mono">Qwen2.5-0.5B • CUDA</p>
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
          className="w-full bg-white/[0.05] hover:bg-white/[0.09] text-slate-200 hover:text-white border border-outline-variant hover:border-white/[0.16] font-medium text-xs py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-all cursor-pointer shadow-subtle active:scale-[0.99]"
        >
          <MessageSquarePlus className="w-3.5 h-3.5 text-primary" />
          <span>New Chat</span>
        </button>

        {/* Model Analysis Nav Item */}
        <button
          onClick={() => setActiveView('analysis')}
          className={`w-full text-xs font-medium py-2 px-3 rounded-lg flex items-center justify-between transition-all cursor-pointer ${
            activeView === 'analysis'
              ? 'bg-surface-container text-slate-100 border border-outline-variant shadow-subtle font-semibold'
              : 'text-slate-400 hover:bg-surface-high hover:text-slate-200 border border-transparent'
          }`}
        >
          <div className="flex items-center gap-2.5">
            <BarChart2 className="w-3.5 h-3.5 text-primary" />
            <span>Model Analysis</span>
          </div>
        </button>
      </div>

      {/* Recent Chats Section */}
      <div className="flex-1 overflow-y-auto scrollbar-hide px-3 py-2 space-y-1">
        <div className="px-2 py-1.5 flex items-center justify-between text-[11px] font-medium text-slate-400">
          <span>Recent Chats</span>
          <span className="text-[10px] font-mono text-slate-500 tabular-nums">{sessions.length}</span>
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
              className={`group flex items-center justify-between px-2.5 py-1.5 rounded-lg text-xs font-sans transition-all cursor-pointer ${
                isActive
                  ? 'bg-surface-container text-slate-100 font-medium border border-outline-variant shadow-subtle'
                  : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200 border border-transparent'
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
      <div className="p-3 border-t border-outline-variant bg-surface-dim space-y-2.5 text-xs font-mono">
        {/* Engine Status Badge */}
        <div className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-surface-lowest border border-outline-variant">
          <div className="flex items-center gap-2">
            <span
              className={`w-2 h-2 rounded-full ${
                serverOnline ? 'bg-emerald-400 ring-2 ring-emerald-400/20' : 'bg-rose-500 ring-2 ring-rose-500/20'
              }`}
            />
            <span className={serverOnline ? 'text-slate-300 font-sans font-medium text-xs' : 'text-rose-400 font-sans font-medium text-xs'}>
              {serverOnline ? 'Engine Online' : 'Offline'}
            </span>
          </div>
          <span className="text-[10px] text-slate-400">RTX 4060</span>
        </div>

        {/* GPU VRAM Bar */}
        {systemInfo && systemInfo.cuda_available && (
          <div className="px-2.5 py-1.5 rounded-lg bg-surface-lowest border border-outline-variant space-y-1">
            <div className="flex justify-between text-[11px] text-slate-400">
              <span className="font-sans">VRAM</span>
              <span className="text-slate-200 font-medium tabular-nums">
                {(systemInfo.vram_allocated_mb / 1024).toFixed(1)}GB / {(systemInfo.vram_total_mb / 1024).toFixed(0)}GB
              </span>
            </div>
            <div className="w-full h-1 bg-surface-high rounded-full overflow-hidden">
              <div
                className="h-full bg-sky-400 transition-all duration-500 rounded-full"
                style={{ width: `${Math.min(100, systemInfo.vram_usage_percent)}%` }}
              />
            </div>
          </div>
        )}

        {/* User Profile / Auth Status */}
        <div className="pt-1">
          <UserAuth />
        </div>
      </div>
    </aside>
  );
};
