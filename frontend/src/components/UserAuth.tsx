import React, { useState, useRef, useEffect } from 'react';
import {
  LogOut,
  Cloud,
  ChevronDown,
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const UserAuth: React.FC = () => {
  const { user, loading, signInWithGoogle, signOut } = useAuth();
  const [dropdownOpen, setDropdownOpen] = useState<boolean>(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  if (loading) {
    return (
      <div className="h-8 w-24 bg-surface-lowest animate-pulse rounded-lg border border-outline-variant/30" />
    );
  }

  if (!user) {
    return (
      <button
        onClick={signInWithGoogle}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] text-slate-200 hover:text-white transition-all font-sans text-xs cursor-pointer shadow-subtle"
        title="Sign in with Google to sync your chats and benchmarks"
      >
        <svg className="w-3.5 h-3.5" viewBox="0 0 24 24">
          <path
            fill="#4285F4"
            d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
          />
          <path
            fill="#34A853"
            d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
          />
          <path
            fill="#FBBC05"
            d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.06H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.94l2.85-2.22.81-.63z"
          />
          <path
            fill="#EA4335"
            d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.06l3.66 2.84c.87-2.6 3.3-4.52 6.16-4.52z"
          />
        </svg>
        <span className="font-medium hidden sm:inline">Sign in with Google</span>
        <span className="font-medium sm:hidden">Sign In</span>
      </button>
    );
  }

  const displayName = user.full_name || user.email?.split('@')[0] || 'User';
  const initial = displayName.charAt(0).toUpperCase();

  return (
    <div className="relative font-sans text-xs" ref={dropdownRef}>
      <button
        onClick={() => setDropdownOpen(!dropdownOpen)}
        className="flex items-center gap-2 p-1 pl-2 pr-2.5 rounded-lg bg-surface-lowest hover:bg-surface-high border border-outline-variant hover:border-white/[0.16] transition-all cursor-pointer shadow-subtle"
      >
        {user.avatar_url ? (
          <img
            src={user.avatar_url}
            alt={displayName}
            className="w-5 h-5 rounded-full border border-outline-variant object-cover"
          />
        ) : (
          <div className="w-5 h-5 rounded-full bg-surface-high border border-outline-variant text-slate-300 flex items-center justify-center font-medium text-[10px]">
            {initial}
          </div>
        )}
        <div className="text-left hidden md:block">
          <div className="text-[11px] font-medium text-slate-200 leading-none truncate max-w-[100px]">
            {displayName}
          </div>
        </div>
        <ChevronDown className="w-3 h-3 text-slate-400" />
      </button>

      {/* Dropdown Menu */}
      {dropdownOpen && (
        <div className="absolute right-0 mt-2 w-64 rounded-xl bg-surface-container border border-outline-variant shadow-popover p-3 space-y-3 z-50 animate-fade-in">
          {/* User Info Header */}
          <div className="flex items-center gap-3 p-2.5 rounded-lg bg-surface-lowest border border-outline-variant">
            {user.avatar_url ? (
              <img
                src={user.avatar_url}
                alt={displayName}
                className="w-8 h-8 rounded-full border border-outline-variant object-cover"
              />
            ) : (
              <div className="w-8 h-8 rounded-full bg-surface-high border border-outline-variant text-slate-200 flex items-center justify-center font-medium text-xs">
                {initial}
              </div>
            )}
            <div className="min-w-0 flex-1">
              <div className="font-semibold text-slate-200 text-xs truncate">{displayName}</div>
              <div className="text-[10px] text-slate-400 truncate">{user.email}</div>
            </div>
          </div>

          {/* Sync Status Badge */}
          <div className="p-2 rounded-lg bg-white/[0.04] border border-outline-variant flex items-center gap-2 text-slate-300 text-[10px]">
            <Cloud className="w-3.5 h-3.5 shrink-0 text-primary" />
            <span>Cloud Sync Active • Chats & Benchmarks Saved</span>
          </div>

          {/* Sign Out Action */}
          <button
            onClick={() => {
              setDropdownOpen(false);
              signOut();
            }}
            className="w-full flex items-center gap-2 p-2 rounded-lg hover:bg-rose-500/10 text-slate-400 hover:text-rose-400 border border-transparent hover:border-rose-500/20 transition-all cursor-pointer text-left text-xs font-medium"
          >
            <LogOut className="w-3.5 h-3.5" />
            <span>Sign Out</span>
          </button>
        </div>
      )}
    </div>
  );
};
