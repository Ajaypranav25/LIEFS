"""
FastAPI Serving Layer for LIEFS.

This module provides a simplified OpenAI-compatible /v1/completions endpoint.
It handles model loading on startup and synchronous request processing.

Design Decisions & Interview Context:
1. **Simplified API**: We implement a minimal subset of the OpenAI schema. 
   A production server like vLLM would implement the full spec, including streaming SSE, 
   logit bias, stop sequences, etc.
2. **Synchronous Generation**: Currently, our KV cache engine processes requests 
   synchronously. In FastAPI, this means a request blocks the async event loop if run 
   directly, or runs in a thread pool (blocking a thread). We simulate async behavior 
   here, but the actual GPU forward pass is the bottleneck.
3. **Production Differences**:
   - **Continuous Batching**: A real inference engine (vLLM) batches concurrent requests 
     at the token level (iteration-level scheduling).
   - **Streaming**: Token-by-token streaming via Server-Sent Events (SSE) provides lower 
     Time-To-First-Token (TTFT) perception to users.
   - **Inference Threading**: Production servers run generation in a separate dedicated 
     thread/process loop, feeding requests from a queue, to decouple HTTP I/O from GPU work.
"""

import time
import uuid
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from server.schemas import CompletionRequest, CompletionResponse, CompletionChoice, CompletionUsage
from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.kv_cache_engine import KVCacheEngine

# Global state
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup event: Load model and tokenizer, initialize KVCacheEngine.
    """
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    engine = KVCacheEngine(model, tokenizer)
    
    app_state["model"] = model
    app_state["tokenizer"] = tokenizer
    app_state["engine"] = engine
    app_state["recent_metrics"] = []
    
    print("Model loaded successfully. Server ready.")
    yield
    app_state.clear()


app = FastAPI(title="LIEFS Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
async def get_metrics():
    """Return the last N generation metrics."""
    return {"recent_metrics": app_state["recent_metrics"][-100:]}


@app.post("/v1/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest):
    """
    Handles completion requests.
    NOTE: In this implementation, temperature is ignored (always greedy).
    """
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming is not yet implemented.")
        
    tokenizer = app_state["tokenizer"]
    engine = app_state["engine"]
    
    # Prompt formatting
    prompt_tensor = format_chat_prompt(tokenizer, request.prompt)
    prompt_len = prompt_tensor.shape[1]
    
    # Generate (Synchronous)
    try:
        generated_ids, metrics = engine.generate(
            prompt_tensor, 
            max_new_tokens=request.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Process output
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Determine finish reason
    finish_reason = "length"
    if len(generated_ids) < request.max_tokens or (generated_ids and generated_ids[-1] in engine.eos_token_ids):
        finish_reason = "stop"
        
    usage = CompletionUsage(
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        total_tokens=prompt_len + len(generated_ids)
    )
    
    # Update metrics history
    metrics_dict = {
        "ttft_ms": metrics.ttft_ms if metrics else 0.0,
        "tpot_ms": metrics.tpot_ms if metrics else 0.0,
        "total_time_ms": metrics.total_time_ms if metrics else 0.0,
        "tokens_per_sec": metrics.tokens_per_sec if metrics else 0.0,
        "peak_vram_mb": metrics.peak_vram_mb if metrics else 0.0,
        "prompt_len": prompt_len,
        "gen_len": len(generated_ids)
    }
    app_state["recent_metrics"].append(metrics_dict)
    
    response_id = f"cmpl-{uuid.uuid4().hex}"
    
    return CompletionResponse(
        id=response_id,
        created=int(time.time()),
        model="Qwen2.5-0.5B-Instruct",
        choices=[
            CompletionChoice(
                text=generated_text,
                index=0,
                finish_reason=finish_reason
            )
        ],
        usage=usage,
        metrics=metrics_dict
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
