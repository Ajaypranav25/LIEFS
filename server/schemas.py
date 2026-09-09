"""
Pydantic schemas for the LIEFS Universal LLM & Computer Benchmarking Server.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., max_length=100000, description="Input prompt text (max 100k chars)")
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
    model_name: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", max_length=256)
    precision: str = Field(default="float16", max_length=32)
    device: str = Field(default="auto", max_length=32)
    hf_token: Optional[str] = Field(default=None, max_length=256)


class ModelLoadResponse(BaseModel):
    status: str
    message: str
    load_time_sec: float
    model_name: str
    metadata: dict


class BenchmarkRunRequest(BaseModel):
    prompt: str = Field(default="Explain how a transformer model works step by step.", max_length=50000)
    max_tokens: int = Field(default=128, ge=1, le=2048)
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
    prompt: str = Field(default="Explain how transformer self-attention works step by step in plain terms.", max_length=50000)
    max_tokens: int = Field(default=128, ge=1, le=2048)
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
