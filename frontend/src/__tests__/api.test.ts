import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fetchHealth, fetchSystemInfo, fetchEngines, fetchPresetBenchmarks } from '../lib/api';

describe('API Client Suite', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('fetchHealth queries /health and returns server health status', async () => {
    const mockHealth = {
      status: 'ok',
      service: 'LIEFS Inference Engine',
      model: 'Qwen/Qwen2.5-0.5B-Instruct',
      device: 'NVIDIA GeForce RTX 4060 Laptop GPU',
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockHealth,
    } as Response);

    const result = await fetchHealth();
    expect(result.status).toBe('ok');
    expect(result.service).toBe('LIEFS Inference Engine');
    expect(global.fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/health');
  });

  it('fetchSystemInfo returns GPU and memory stats', async () => {
    const mockSys = {
      cuda_available: true,
      device_name: 'NVIDIA GeForce RTX 4060 Laptop GPU',
      pytorch_version: '2.13.0+cu126',
      model_name: 'Qwen/Qwen2.5-0.5B-Instruct',
      vram_allocated_mb: 1000.5,
      vram_reserved_mb: 1200.0,
      vram_total_mb: 8188.0,
      vram_usage_percent: 12.2,
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockSys,
    } as Response);

    const result = await fetchSystemInfo();
    expect(result.cuda_available).toBe(true);
    expect(result.vram_allocated_mb).toBe(1000.5);
  });

  it('fetchEngines returns the 5 inference engine architectures', async () => {
    const mockEngines = {
      engines: [
        { id: 'kv_cache', name: 'KV-Cache Engine (Stage 2)' },
        { id: 'naive', name: 'Naive Baseline (Stage 1)' },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockEngines,
    } as Response);

    const result = await fetchEngines();
    expect(result.engines).toHaveLength(2);
    expect(result.engines[0].id).toBe('kv_cache');
  });

  it('fetchPresetBenchmarks returns benchmark scales', async () => {
    const mockPresets = {
      model: 'Qwen2.5-0.5B-Instruct',
      device: 'NVIDIA RTX',
      benchmarks: [
        {
          scale: 'Short (32 tokens)',
          prompt: 'What is 2+2?',
          max_tokens: 32,
          results: [
            { engine: 'naive', throughput_tok_s: 24.8, ttft_ms: 14.2, tpot_ms: 40.3, total_time_ms: 1289.6, peak_vram_mb: 1184.2 },
            { engine: 'kv_cache', throughput_tok_s: 96.4, ttft_ms: 14.5, tpot_ms: 10.3, total_time_ms: 331.8, peak_vram_mb: 1192.5 },
          ],
        },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockPresets,
    } as Response);

    const result = await fetchPresetBenchmarks();
    expect(result.benchmarks).toHaveLength(1);
    expect(result.benchmarks[0].results[1].engine).toBe('kv_cache');
  });
});
