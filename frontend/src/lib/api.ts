import type {
  SystemInfo,
  EngineInfo,
  PresetBenchmarksResponse,
  BenchmarkResult,
  GenerationMetrics,
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

    const read = async () => {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;

          const dataPayload = trimmed.replace(/^data:\s*/, '');
          if (dataPayload === '[DONE]') {
            callbacks.onDone(accumulatedText, lastMetrics);
            return;
          }

          try {
            const parsed = JSON.parse(dataPayload);
            if (parsed.error) {
              throw new Error(parsed.error);
            }

            if (parsed.choices && parsed.choices.length > 0) {
              const textPiece = parsed.choices[0].text;
              if (textPiece) {
                accumulatedText += textPiece;
              }
              if (parsed.metrics) {
                lastMetrics = parsed.metrics;
                callbacks.onMetrics(parsed.metrics);
              }
              if (textPiece) {
                callbacks.onToken(textPiece, parsed.metrics);
              }
            }
          } catch (err) {
            console.warn('Failed to parse SSE line:', line, err);
          }
        }
      }
      callbacks.onDone(accumulatedText, lastMetrics);
    };

    read().catch((err) => {
      if (err.name !== 'AbortError') {
        callbacks.onError(err);
      }
    });
  } catch (err: any) {
    if (err.name !== 'AbortError') {
      callbacks.onError(err);
    }
  }

  // Return cancel function
  return () => controller.abort();
}
