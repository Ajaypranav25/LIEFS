"""
Pydantic schemas for the LIEFS Universal LLM & Computer Benchmarking Server.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=128, ge=1, le=2048)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)  # 0.0 = greedy
    stream: bool = False
    engine: str = Field(default="kv_cache", description="Inference engine: kv_cache, naive, quantized, paged")


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str | None = None  # 'stop' or 'length' or None


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = 'text_completion'
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: CompletionUsage
    metrics: dict | None = None


class ModelLoadRequest(BaseModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    precision: str = "float16"
    device: str = "auto"
    hf_token: Optional[str] = None


class ModelLoadResponse(BaseModel):
    status: str
    message: str
    load_time_sec: float
    model_name: str
    metadata: dict


class BenchmarkRunRequest(BaseModel):
    prompt: str = "Explain how a transformer model works step by step."
    max_tokens: int = 128
    engines: list[str] = ["naive", "kv_cache", "paged"]


class BenchmarkEngineResult(BaseModel):
    engine: str
    engine_name: str
    ttft_ms: float
    tpot_ms: float
    total_time_ms: float
    tokens_per_sec: float
    peak_vram_mb: float
    prompt_tokens: int
    completion_tokens: int
    speedup_vs_naive: float
    memory_bandwidth_gbs: float = 0.0
    sample_output: str


class ComputerBenchmarkRequest(BaseModel):
    prompt: str = "Explain how transformer self-attention works step by step in plain terms."
    max_tokens: int = 128
    engines: list[str] = ["naive", "kv_cache", "paged"]
    include_batch_scaling: bool = True


class BatchScalePoint(BaseModel):
    batch_size: int
    total_tokens: int
    wall_time_ms: float
    aggregate_throughput_tok_s: float


class ComputerBenchmarkResponse(BaseModel):
    timestamp: int
    model: dict
    hardware: dict
    score: dict
    results: list[BenchmarkEngineResult]
    batch_scaling: list[BatchScalePoint]
