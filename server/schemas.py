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

class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str  # 'stop' or 'length'

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
