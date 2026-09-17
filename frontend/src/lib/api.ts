import type {
  SystemInfo,
  HardwareProfile,
  ModelArchitectureMetadata,
  ModelPreset,
  EngineInfo,
  PresetBenchmarksResponse,
  BenchmarkResult,
  GenerationMetrics,
  ComputerBenchmarkResponse,
} from '../types';

const API_BASE = 'http://127.0.0.1:8000';

export async function fetchHealth(): Promise<{ status: string; service: string; model: string; device: string }> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
  return res.json();
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  const res = await fetch(`${API_BASE}/v1/system`);
  if (!res.ok) throw new Error(`Failed to fetch system info: ${res.statusText}`);
  return res.json();
}

export async function fetchHardwareInfo(): Promise<HardwareProfile> {
  const res = await fetch(`${API_BASE}/v1/hardware`);
  if (!res.ok) throw new Error(`Failed to fetch hardware specs: ${res.statusText}`);
  return res.json();
}

export async function fetchModelPresets(): Promise<{ presets: ModelPreset[]; current_model: string }> {
  const res = await fetch(`${API_BASE}/v1/models/presets`);
  if (!res.ok) throw new Error(`Failed to fetch model presets: ${res.statusText}`);
  return res.json();
}

export async function fetchCurrentModel(): Promise<ModelArchitectureMetadata> {
  const res = await fetch(`${API_BASE}/v1/models/current`);
  if (!res.ok) throw new Error(`Failed to fetch current model info: ${res.statusText}`);
  return res.json();
}

export async function loadCustomModel(payload: {
  model_name: string;
  precision?: string;
  device?: string;
  hf_token?: string;
}): Promise<{ status: string; message: string; load_time_sec: number; model_name: string; metadata: ModelArchitectureMetadata }> {
  const res = await fetch(`${API_BASE}/v1/models/load`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to load model');
  }
  return res.json();
}

export async function unloadModel(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/v1/models/unload`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error(`Failed to unload model: ${res.statusText}`);
  return res.json();
}

export async function fetchEngines(): Promise<{ engines: EngineInfo[] }> {
  const res = await fetch(`${API_BASE}/v1/engines`);
  if (!res.ok) throw new Error(`Failed to fetch engines: ${res.statusText}`);
  return res.json();
}

export async function fetchPresetBenchmarks(): Promise<PresetBenchmarksResponse> {
  const res = await fetch(`${API_BASE}/v1/benchmarks/preset`);
  if (!res.ok) throw new Error(`Failed to fetch preset benchmarks: ${res.statusText}`);
  return res.json();
}

export async function runLiveBenchmark(
  prompt: string,
  maxTokens: number = 128,
  engines: string[] = ['naive', 'kv_cache', 'paged']
): Promise<{ timestamp: number; prompt: string; max_tokens: number; results: BenchmarkResult[] }> {
  const res = await fetch(`${API_BASE}/v1/benchmarks/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt, max_tokens: maxTokens, engines }),
  });
  if (!res.ok) throw new Error(`Benchmark execution failed: ${res.statusText}`);
  return res.json();
}

export async function runComputerBenchmark(payload: {
  prompt?: string;
  max_tokens?: number;
  engines?: string[];
  include_batch_scaling?: boolean;
}): Promise<ComputerBenchmarkResponse> {
  const res = await fetch(`${API_BASE}/v1/benchmarks/computer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      prompt: payload.prompt || 'Explain how transformer self-attention works step by step in plain terms.',
      max_tokens: payload.max_tokens ?? 128,
      engines: payload.engines || ['naive', 'kv_cache', 'paged'],
      include_batch_scaling: payload.include_batch_scaling ?? true,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Computer benchmark failed');
  }
  return res.json();
}

export interface StreamCallbacks {
  onToken: (token: string, metrics?: GenerationMetrics) => void;
  onMetrics: (metrics: GenerationMetrics) => void;
  onDone: (fullText: string, finalMetrics: GenerationMetrics) => void;
  onError: (error: Error) => void;
}

export async function streamCompletion(
  prompt: string,
  options: {
    maxTokens?: number;
    temperature?: number;
    engine?: string;
  },
  callbacks: StreamCallbacks
): Promise<() => void> {
  const controller = new AbortController();
  let accumulatedText = '';
  let lastMetrics: GenerationMetrics = {
    ttft_ms: 0,
    tpot_ms: 0,
    total_time_ms: 0,
    tokens_per_sec: 0,
    peak_vram_mb: 0,
  };

  try {
    const response = await fetch(`${API_BASE}/v1/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens ?? 128,
        temperature: options.temperature ?? 0.0,
        stream: true,
        engine: options.engine ?? 'kv_cache',
      }),
      signal: controller.signal,
    });

    if (!response.ok) {
      throw new Error(`Server returned status ${response.status}: ${response.statusText}`);
    }

    if (!response.body) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    const readLoop = async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            callbacks.onDone(accumulatedText, lastMetrics);
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || !trimmed.startsWith('data: ')) continue;

            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') {
              callbacks.onDone(accumulatedText, lastMetrics);
              return;
            }

            try {
              const parsed = JSON.parse(dataStr);
              if (parsed.error) {
                callbacks.onError(new Error(parsed.error));
                return;
              }

              if (parsed.choices && parsed.choices.length > 0) {
                const choice = parsed.choices[0];
                const textChunk = choice.text || '';
                accumulatedText += textChunk;

                if (parsed.metrics) {
                  lastMetrics = parsed.metrics;
                  callbacks.onToken(textChunk, parsed.metrics);
                } else {
                  callbacks.onToken(textChunk);
                }
              }
            } catch {
              console.warn('Failed to parse SSE line:', dataStr);
            }
          }
        }
      } catch (err: any) {
        if (err.name === 'AbortError') {
          callbacks.onDone(accumulatedText, lastMetrics);
        } else {
          callbacks.onError(err);
        }
      }
    };

    readLoop();

    return () => {
      controller.abort();
    };
  } catch (err: any) {
    callbacks.onError(err);
    return () => {};
  }
}
