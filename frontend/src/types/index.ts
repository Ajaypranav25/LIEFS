export interface HardwareProfile {
  os_name: string;
  os_release: string;
  architecture: string;
  python_version: string;
  pytorch_version: string;
  cpu_model: string;
  cpu_physical_cores: number;
  cpu_logical_cores: number;
  cpu_freq_mhz?: number | null;
  ram_total_gb: number;
  ram_available_gb: number;
  ram_usage_percent: number;
  cuda_available: boolean;
  gpu_count: number;
  gpu_name: string;
  gpu_compute_capability?: string | null;
  vram_total_mb: number;
  vram_free_mb: number;
  vram_allocated_mb: number;
  vram_reserved_mb: number;
  vram_usage_percent: number;
  cuda_version?: string | null;
}

export interface ModelArchitectureMetadata {
  model_name: string;
  parameter_count: number;
  parameter_count_m: number;
  num_layers: number;
  hidden_size: number;
  num_attention_heads: number;
  num_kv_heads: number;
  head_dim: number;
  vocab_size: number;
  max_position_embeddings: number;
  dtype_str: string;
  device_str: string;
  memory_footprint_mb: number;
  fp16_vram_estimate_mb: number;
  int8_vram_estimate_mb: number;
  int4_vram_estimate_mb: number;
  architectures: string[];
}

export interface ModelPreset {
  id: string;
  name: string;
  family: string;
  size_label: string;
  params_m: number;
  recommended_vram_mb: number;
  description: string;
  badge: string;
}

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
  memory_bandwidth_gbs?: number;
  memory_savings_percent?: number;
  sample_output?: string;
  prompt_tokens?: number;
  completion_tokens?: number;
}

export interface BatchScalePoint {
  batch_size: number;
  total_tokens: number;
  wall_time_ms: number;
  aggregate_throughput_tok_s: number;
}

export interface ComputerScoreBreakdown {
  throughput_points: number;
  bandwidth_points: number;
  latency_points: number;
}

export interface ComputerScoreData {
  score: number;
  tier: string;
  badge: string;
  breakdown?: ComputerScoreBreakdown;
}

export interface ComputerBenchmarkResponse {
  timestamp: number;
  model: ModelArchitectureMetadata;
  hardware: HardwareProfile;
  score: ComputerScoreData;
  results: BenchmarkResult[];
  batch_scaling: BatchScalePoint[];
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
  score?: ComputerScoreData;
  hardware?: HardwareProfile;
  user_id?: string;
}

export interface UserProfile {
  id: string;
  email?: string;
  full_name?: string;
  avatar_url?: string;
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
  user_id?: string;
}
