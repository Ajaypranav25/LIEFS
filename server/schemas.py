"""
Pydantic schemas for the OpenAI-compatible completion API.
These schemas represent a simplified version of the standard OpenAI API.
"""

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
    # Custom fields for benchmarking
    metrics: dict | None = None

class BenchmarkRunRequest(BaseModel):
    prompt: str = "Explain how a transformer model works step by step."
    max_tokens: int = 128
    engines: list[str] = ["naive", "kv_cache", "quantized", "continuous_batching"]

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
    sample_output: str

