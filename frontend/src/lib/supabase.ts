import { createClient } from '@supabase/supabase-js';
import type { SavedBenchmarkRun, BenchmarkResult } from '../types';

const SUPABASE_URL = 'https://agcbxmiijjbvxfoosvwz.supabase.co';
const SUPABASE_ANON_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFnY2J4bWlpampidnhmb29zdnd6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY1NjQ2MjcsImV4cCI6MjA4MjE0MDYyN30.fniKk7olCaPs_oTtwK81VdqPLcAqNsYk6Sjyyn7Dmw0';

export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

const LOCAL_STORAGE_KEY = 'liefs_benchmark_runs_v1';

export async function saveBenchmarkRun(
  prompt: string,
  maxTokens: number,
  results: BenchmarkResult[],
  model: string = 'Qwen2.5-0.5B-Instruct',
  device: string = 'NVIDIA CUDA GPU'
): Promise<SavedBenchmarkRun> {
  const newRun: SavedBenchmarkRun = {
    id: typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `run-${Date.now()}`,
    created_at: new Date().toISOString(),
    prompt,
    max_tokens: maxTokens,
    model,
    device,
    results,
  };

  // 1. Try persisting to Supabase
  try {
    const { data, error } = await supabase
      .from('benchmark_runs')
      .insert([
        {
          id: newRun.id,
          prompt,
          max_tokens: maxTokens,
          model,
          device,
          results,
        },
      ])
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
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(updated));
  } catch (e) {
    console.error('Failed to save to localStorage:', e);
  }

  return newRun;
}

export async function getBenchmarkHistory(): Promise<SavedBenchmarkRun[]> {
  // Try fetching from Supabase first
  try {
    const { data, error } = await supabase
      .from('benchmark_runs')
      .select('*')
      .order('created_at', { ascending: false })
      .limit(50);

    if (!error && data && data.length > 0) {
      // Sync into local storage
      try {
        localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(data));
      } catch (e) {
        // ignore
      }
      return data as SavedBenchmarkRun[];
    }
  } catch (err) {
    console.warn('Could not fetch from Supabase, loading from localStorage:', err);
  }

  // Fallback to local storage
  return getLocalRuns();
}

export function getLocalRuns(): SavedBenchmarkRun[] {
  try {
    const item = localStorage.getItem(LOCAL_STORAGE_KEY);
    return item ? JSON.parse(item) : [];
  } catch (e) {
    return [];
  }
}

export function clearLocalHistory(): void {
  try {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  } catch (e) {
    // ignore
  }
}
