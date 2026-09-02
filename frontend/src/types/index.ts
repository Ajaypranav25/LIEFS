export interface SystemInfo {
  cuda_available: boolean;
  device_name: string;
  pytorch_version: string;
  model_name: string;
  vram_allocated_mb: number;
  vram_reserved_mb: number;
  vram_total_mb: number;
  vram_usage_percent: number;
}

export interface EngineInfo {
  id: string;
  name: string;
  description: string;
  complexity: string;
  recommended: boolean;
  badge: string;
}

export interface GenerationMetrics {
  ttft_ms: number;
  tpot_ms: number;
  total_time_ms: number;
  tokens_per_sec: number;
  peak_vram_mb: number;
  generated_tokens?: number;
  prompt_tokens?: number;
  prompt_len?: number;
  gen_len?: number;
  engine?: string;
  token_index?: number;
}

export interface StreamChunk {
  id: string;
  object: string;
  created: number;
  model: string;
  engine?: string;
  choices: {
    text: string;
    index: number;
    finish_reason: string | null;
  }[];
  metrics?: GenerationMetrics;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export interface BenchmarkResult {
  engine: string;
  engine_name: string;
  throughput_tok_s?: number;
  tokens_per_sec?: number;
  ttft_ms: number;
  tpot_ms: number;
  total_time_ms: number;
  peak_vram_mb: number;
  speedup?: number;
  speedup_vs_naive?: number;
  memory_savings_percent?: number;
  sample_output?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface PresetBenchmarkScale {
  scale: string;
  prompt: string;
  max_tokens: number;
  results: BenchmarkResult[];
}

export interface PresetBenchmarksResponse {
  model: string;
  device: string;
  benchmarks: PresetBenchmarkScale[];
}

export interface SavedBenchmarkRun {
  id: string;
  created_at: string;
  prompt: string;
  max_tokens: number;
  model: string;
  device: string;
  results: BenchmarkResult[];
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
  engine?: string;
  metrics?: GenerationMetrics;
  isStreaming?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  createdAt: number;
  messages: ChatMessage[];
  engine: string;
}
