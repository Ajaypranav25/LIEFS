import { createClient } from '@supabase/supabase-js';
import type {
  SavedBenchmarkRun,
  BenchmarkResult,
  ChatSession,
  ChatMessage,
  ComputerScoreData,
  HardwareProfile,
} from '../types';

const SUPABASE_URL =
  import.meta.env?.VITE_SUPABASE_URL || 'https://agcbxmiijjbvxfoosvwz.supabase.co';
const SUPABASE_ANON_KEY =
  import.meta.env?.VITE_SUPABASE_ANON_KEY ||
  'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFnY2J4bWlpampidnhmb29zdnd6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY1NjQ2MjcsImV4cCI6MjA4MjE0MDYyN30.fniKk7olCaPs_oTtwK81VdqPLcAqNsYk6Sjyyn7Dmw0';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const LOCAL_STORAGE_BENCHMARKS = 'liefs_benchmark_runs_v1';
const LOCAL_STORAGE_CHATS = 'liefs_chat_sessions';

// ── BENCHMARK RUNS PERSISTENCE ─────────────────────────────────────────

export async function saveBenchmarkRun(
  prompt: string,
  maxTokens: number,
  results: BenchmarkResult[],
  model: string = 'Qwen2.5-0.5B-Instruct',
  device: string = 'NVIDIA CUDA GPU',
  score?: ComputerScoreData,
  hardware?: HardwareProfile,
  userId?: string
): Promise<SavedBenchmarkRun> {
  const newRun: SavedBenchmarkRun = {
    id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `run-${Date.now()}`,
    created_at: new Date().toISOString(),
    prompt,
    max_tokens: maxTokens,
    model,
    device,
    results,
    score,
    hardware,
    user_id: userId,
  };

  // 1. Try persisting to Supabase
  try {
    const payload: any = {
      id: newRun.id,
      prompt,
      max_tokens: maxTokens,
      model,
      device,
      results,
      score: score || null,
      hardware: hardware || null,
    };
    if (userId) {
      payload.user_id = userId;
    }

    const { data, error } = await supabase
      .from('benchmark_runs')
      .insert([payload])
      .select()
      .single();

    if (error) {
      console.warn('Supabase insert warning, saving locally:', error.message);
    } else if (data) {
      newRun.id = data.id;
      newRun.created_at = data.created_at;
    }
  } catch (err) {
    console.warn('Supabase connection error, falling back to local storage:', err);
  }

  // 2. Always persist locally for instant availability and resilience
  try {
    const localRuns = getLocalRuns();
    const updated = [newRun, ...localRuns].slice(0, 50);
    localStorage.setItem(LOCAL_STORAGE_BENCHMARKS, JSON.stringify(updated));
  } catch {
    console.error('Failed to save to localStorage');
  }

  return newRun;
}

export async function getBenchmarkHistory(userId?: string): Promise<SavedBenchmarkRun[]> {
  try {
    let query = supabase
      .from('benchmark_runs')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(50);

    if (userId) {
      query = query.eq('user_id', userId);
    }

    const { data, error } = await query;

    if (!error && data && data.length > 0) {
      try {
        localStorage.setItem(LOCAL_STORAGE_BENCHMARKS, JSON.stringify(data));
      } catch {
        // ignore
      }
      return data as SavedBenchmarkRun[];
    }
  } catch (err) {
    console.warn('Could not fetch from Supabase, loading from localStorage:', err);
  }

  return getLocalRuns();
}

export function getLocalRuns(): SavedBenchmarkRun[] {
  try {
    const item = localStorage.getItem(LOCAL_STORAGE_BENCHMARKS);
    return item ? JSON.parse(item) : [];
  } catch {
    return [];
  }
}

export function clearLocalHistory() {
  localStorage.removeItem(LOCAL_STORAGE_BENCHMARKS);
}

// ── CHAT SESSIONS & MESSAGES PERSISTENCE ────────────────────────────────

export async function fetchUserChatSessions(userId?: string): Promise<ChatSession[]> {
  try {
    let query = supabase
      .from('chat_sessions')
      .select('*, chat_messages(*)')
      .order('created_at', { ascending: false });

    if (userId) {
      query = query.eq('user_id', userId);
    }

    const { data, error } = await query;

    if (!error && data && data.length > 0) {
      const sessions: ChatSession[] = data.map((s: any) => ({
        id: s.id,
        title: s.title,
        createdAt: new Date(s.created_at).getTime(),
        engine: s.engine || 'kv_cache',
        user_id: s.user_id,
        messages: (s.chat_messages || [])
          .sort((a: any, b: any) => (a.timestamp || 0) - (b.timestamp || 0))
          .map((m: any) => ({
            id: m.id,
            role: m.role,
            content: m.content,
            timestamp: m.timestamp || new Date(m.created_at).getTime(),
            engine: m.engine,
            metrics: m.metrics,
          })),
      }));

      // Cache locally
      try {
        localStorage.setItem(LOCAL_STORAGE_CHATS, JSON.stringify(sessions));
      } catch {
        // ignore
      }

      return sessions;
    }
  } catch (err) {
    console.warn('Failed to fetch sessions from Supabase:', err);
  }

  // Fallback to local storage
  try {
    const saved = localStorage.getItem(LOCAL_STORAGE_CHATS);
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
}

export async function saveChatSessionToDb(session: ChatSession, userId?: string): Promise<void> {
  try {
    const payload: any = {
      id: session.id,
      title: session.title,
      engine: session.engine,
      updated_at: new Date().toISOString(),
    };
    if (userId) {
      payload.user_id = userId;
    }

    await supabase.from('chat_sessions').upsert(payload);
  } catch (err) {
    console.warn('Failed to upsert chat session:', err);
  }
}

export async function saveChatMessageToDb(
  sessionId: string,
  message: ChatMessage,
  userId?: string
): Promise<void> {
  try {
    const payload: any = {
      id: message.id,
      session_id: sessionId,
      role: message.role,
      content: message.content,
      engine: message.engine || null,
      metrics: message.metrics || null,
      timestamp: message.timestamp,
    };
    if (userId) {
      payload.user_id = userId;
    }

    await supabase.from('chat_messages').upsert(payload);
  } catch (err) {
    console.warn('Failed to upsert chat message:', err);
  }
}

export async function deleteChatSessionFromDb(sessionId: string): Promise<void> {
  try {
    await supabase.from('chat_sessions').delete().eq('id', sessionId);
  } catch (err) {
    console.warn('Failed to delete session from Supabase:', err);
  }
}
