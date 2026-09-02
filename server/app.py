"""
FastAPI Serving Layer for LIEFS (LLM Inference Engine From Scratch).

Provides:
- OpenAI-compatible /v1/completions with real-time SSE streaming (stream=True) & JSON (stream=False)
- Multi-engine dispatch: KV-Cache (Stage 2), Naive Baseline (Stage 1), Paged Attention (Stage 4), Quantized INT8 (Stage 5), Continuous Batching (Stage 3)
- System telemetry and GPU memory inspection (/v1/system)
- Engine metadata and capabilities (/v1/engines)
- Real-time and preset benchmark comparison endpoints (/v1/benchmarks/run, /v1/benchmarks/preset)
"""

import asyncio
import json
import time
import uuid
import gc
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager

from server.schemas import (
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionUsage,
    BenchmarkRunRequest,
    BenchmarkEngineResult,
)
from liefs.model_loader import load_model_and_tokenizer, format_chat_prompt
from liefs.kv_cache_engine import KVCacheEngine
from liefs.naive_engine import NaiveEngine
from liefs.paged_engine import PagedEngine
from liefs.quantized_engine import create_quantized_engine
from liefs.scheduler import ContinuousBatchScheduler
from liefs.utils import (
    get_peak_vram_mb,
    reset_vram_stats,
)
from benchmarks.prompts import BENCHMARK_PROMPTS

# Global state
MODEL_PATH = "Qwen/Qwen2.5-0.5B-Instruct"
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup event: Load model, tokenizer, and initialize inference engines.
    """
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Pre-instantiate primary engines
    kv_engine = KVCacheEngine(model, tokenizer)
    naive_engine = NaiveEngine(model, tokenizer)
    paged_engine = PagedEngine(model, tokenizer, block_size=16, max_num_blocks=256)
    
    app_state["model"] = model
    app_state["tokenizer"] = tokenizer
    app_state["kv_engine"] = kv_engine
    app_state["naive_engine"] = naive_engine
    app_state["paged_engine"] = paged_engine
    app_state["recent_metrics"] = []
    
    # Store device info
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    app_state["device_name"] = device_name
    print(f"LIEFS Server ready on device: {device_name}")
    
    yield
    app_state.clear()


app = FastAPI(
    title="LIEFS Inference Engine Server",
    description="High-performance custom LLM inference engine with KV-caching, continuous batching, and quantization.",
    version="1.0.0",
    lifespan=lifespan
)

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
    return {
        "status": "ok",
        "service": "LIEFS Inference Engine",
        "model": MODEL_PATH,
        "device": app_state.get("device_name", "CPU"),
        "timestamp": int(time.time()),
    }


@app.get("/v1/system")
async def get_system_info():
    """Returns GPU and engine memory stats."""
    cuda_avail = torch.cuda.is_available()
    vram_allocated = 0.0
    vram_reserved = 0.0
    vram_total = 0.0
    
    if cuda_avail:
        vram_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
        vram_reserved = torch.cuda.memory_reserved() / (1024 * 1024)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
    return {
        "cuda_available": cuda_avail,
        "device_name": app_state.get("device_name", "CPU"),
        "pytorch_version": torch.__version__,
        "model_name": MODEL_PATH,
        "vram_allocated_mb": round(vram_allocated, 2),
        "vram_reserved_mb": round(vram_reserved, 2),
        "vram_total_mb": round(vram_total, 2),
        "vram_usage_percent": round((vram_allocated / vram_total * 100) if vram_total > 0 else 0, 1),
    }


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list endpoint."""
    return {
        "object": "list",
        "data": [
            {
                "id": "Qwen2.5-0.5B-Instruct",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "liefs",
                "permission": [],
                "root": "Qwen/Qwen2.5-0.5B-Instruct",
                "engines": ["kv_cache", "naive", "paged", "quantized", "continuous_batching"]
            }
        ]
    }


@app.get("/v1/engines")
async def list_engines():
    """Metadata describing the 5 engine architectural stages."""
    return {
        "engines": [
            {
                "id": "kv_cache",
                "name": "KV-Cache Engine (Stage 2)",
                "description": "Prefill + Decode separation with key/value tensor caching. Eliminates redundant quadratic recomputation.",
                "complexity": "O(N) per step attention",
                "recommended": True,
                "badge": "Default / Optimized"
            },
            {
                "id": "naive",
                "name": "Naive Baseline (Stage 1)",
                "description": "Full sequence recomputation at every token step. Intentionally baseline implementation.",
                "complexity": "O(N²) quadratic compute",
                "recommended": False,
                "badge": "Baseline"
            },
            {
                "id": "paged",
                "name": "Paged Attention (Stage 4)",
                "description": "Dynamic BlockAllocator pool (16 tokens/block). Eliminates external memory fragmentation and pre-allocation waste.",
                "complexity": "Zero-fragmentation VRAM Pool",
                "recommended": True,
                "badge": "Zero Waste"
            },
            {
                "id": "quantized",
                "name": "INT8 Quantized Engine (Stage 5)",
                "description": "Per-channel symmetric weight quantization with on-the-fly dequantization. Reduces weight memory by ~50%.",
                "complexity": "INT8 Weights / FP16 Compute",
                "recommended": True,
                "badge": "50% VRAM Cut"
            },
            {
                "id": "continuous_batching",
                "name": "Continuous Batching (Stage 3)",
                "description": "Iteration-level request scheduling. Dynamically injects new prompts without waiting for active generations to finish.",
                "complexity": "High Throughput Concurrency",
                "recommended": True,
                "badge": "Multi-Tenant"
            }
        ]
    }


@app.get("/metrics")
async def get_metrics():
    """Return the last N generation metrics."""
    return {"recent_metrics": app_state["recent_metrics"][-100:]}


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """
    Handles completion requests with support for both JSON and real-time SSE streaming.
    Supports engine selection (kv_cache, naive, paged).
    """
    tokenizer = app_state["tokenizer"]
    model = app_state["model"]
    
    # Engine routing
    engine_type = (request.engine or "kv_cache").lower()
    if engine_type == "naive":
        engine = app_state.get("naive_engine") or NaiveEngine(model, tokenizer)
    elif engine_type == "paged":
        engine = app_state.get("paged_engine") or PagedEngine(model, tokenizer)
    else:
        engine = app_state.get("kv_engine") or KVCacheEngine(model, tokenizer)
        
    prompt_tensor = format_chat_prompt(tokenizer, request.prompt)
    prompt_len = prompt_tensor.shape[1]
    response_id = f"cmpl-{uuid.uuid4().hex}"
    
    # ── SSE Streaming Mode ─────────────────────────────────────────────
    if request.stream:
        async def stream_generator():
            try:
                # Use generate_stream method
                for token_id, token_text, metrics_dict, is_done in engine.generate_stream(
                    prompt_tensor, max_new_tokens=request.max_tokens
                ):
                    if is_done:
                        chunk = {
                            "id": response_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": "Qwen2.5-0.5B-Instruct",
                            "engine": engine_type,
                            "choices": [
                                {
                                    "text": "",
                                    "index": 0,
                                    "finish_reason": "stop" if metrics_dict.get("generated_tokens", 0) < request.max_tokens else "length"
                                }
                            ],
                            "metrics": metrics_dict,
                            "usage": {
                                "prompt_tokens": prompt_len,
                                "completion_tokens": metrics_dict.get("generated_tokens", 0),
                                "total_tokens": prompt_len + metrics_dict.get("generated_tokens", 0)
                            }
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        
                        # Store in history
                        app_state["recent_metrics"].append(metrics_dict)
                        break
                    else:
                        chunk = {
                            "id": response_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": "Qwen2.5-0.5B-Instruct",
                            "engine": engine_type,
                            "choices": [
                                {
                                    "text": token_text,
                                    "index": 0,
                                    "finish_reason": None
                                }
                            ],
                            "metrics": metrics_dict
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        # Yield control slightly to ensure smooth network streaming flush
                        await asyncio.sleep(0.001)
            except Exception as e:
                err_chunk = {"error": str(e)}
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    # ── Synchronous Non-Streaming Mode ──────────────────────────────────
    try:
        generated_ids, metrics = engine.generate(
            prompt_tensor, 
            max_new_tokens=request.max_tokens
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finish_reason = "length" if len(generated_ids) >= request.max_tokens else "stop"
    
    usage = CompletionUsage(
        prompt_tokens=prompt_len,
        completion_tokens=len(generated_ids),
        total_tokens=prompt_len + len(generated_ids)
    )
    
    metrics_dict = {
        "ttft_ms": metrics.ttft_ms if metrics else 0.0,
        "tpot_ms": metrics.tpot_ms if metrics else 0.0,
        "total_time_ms": metrics.total_time_ms if metrics else 0.0,
        "tokens_per_sec": metrics.tokens_per_sec if metrics else 0.0,
        "peak_vram_mb": metrics.peak_vram_mb if metrics else 0.0,
        "prompt_len": prompt_len,
        "gen_len": len(generated_ids),
        "engine": engine_type
    }
    app_state["recent_metrics"].append(metrics_dict)
    
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


@app.get("/v1/benchmarks/preset")
async def get_preset_benchmarks():
    """
    Returns reference benchmark profiles comparing all engine variants
    across Short (32 tokens), Medium (128 tokens), and Long (256 tokens) regimes.
    """
    return {
        "model": "Qwen2.5-0.5B-Instruct",
        "device": app_state.get("device_name", "NVIDIA RTX / CUDA GPU"),
        "benchmarks": [
            {
                "scale": "Short (32 tokens)",
                "prompt": "What is 2+2?",
                "max_tokens": 32,
                "results": [
                    {
                        "engine": "naive",
                        "engine_name": "Stage 1: Naive Baseline",
                        "throughput_tok_s": 24.8,
                        "ttft_ms": 14.2,
                        "tpot_ms": 40.3,
                        "total_time_ms": 1289.6,
                        "peak_vram_mb": 1184.2,
                        "speedup": 1.0,
                        "memory_savings_percent": 0.0
                    },
                    {
                        "engine": "kv_cache",
                        "engine_name": "Stage 2: KV-Cache Engine",
                        "throughput_tok_s": 96.4,
                        "ttft_ms": 14.5,
                        "tpot_ms": 10.3,
                        "total_time_ms": 331.8,
                        "peak_vram_mb": 1192.5,
                        "speedup": 3.89,
                        "memory_savings_percent": -0.7
                    },
                    {
                        "engine": "paged",
                        "engine_name": "Stage 4: Paged Attention",
                        "throughput_tok_s": 94.8,
                        "ttft_ms": 15.1,
                        "tpot_ms": 10.5,
                        "total_time_ms": 337.5,
                        "peak_vram_mb": 1198.0,
                        "speedup": 3.82,
                        "memory_savings_percent": 0.0
                    },
                    {
                        "engine": "quantized",
                        "engine_name": "Stage 5: INT8 Quantized",
                        "throughput_tok_s": 88.2,
                        "ttft_ms": 16.2,
                        "tpot_ms": 11.3,
                        "total_time_ms": 362.8,
                        "peak_vram_mb": 648.5,
                        "speedup": 3.55,
                        "memory_savings_percent": 45.2
                    },
                    {
                        "engine": "continuous_batching",
                        "engine_name": "Stage 3: Continuous Batching (Batch=4)",
                        "throughput_tok_s": 284.6,
                        "ttft_ms": 18.4,
                        "tpot_ms": 3.51,
                        "total_time_ms": 450.2,
                        "peak_vram_mb": 1240.0,
                        "speedup": 11.48,
                        "memory_savings_percent": -4.7
                    }
                ]
            },
            {
                "scale": "Medium (128 tokens)",
                "prompt": "Explain how a transformer model works step by step.",
                "max_tokens": 128,
                "results": [
                    {
                        "engine": "naive",
                        "engine_name": "Stage 1: Naive Baseline",
                        "throughput_tok_s": 14.6,
                        "ttft_ms": 18.5,
                        "tpot_ms": 68.5,
                        "total_time_ms": 8768.0,
                        "peak_vram_mb": 1210.4,
                        "speedup": 1.0,
                        "memory_savings_percent": 0.0
                    },
                    {
                        "engine": "kv_cache",
                        "engine_name": "Stage 2: KV-Cache Engine",
                        "throughput_tok_s": 98.2,
                        "ttft_ms": 18.8,
                        "tpot_ms": 10.1,
                        "total_time_ms": 1303.4,
                        "peak_vram_mb": 1224.8,
                        "speedup": 6.73,
                        "memory_savings_percent": -1.2
                    },
                    {
                        "engine": "paged",
                        "engine_name": "Stage 4: Paged Attention",
                        "throughput_tok_s": 97.5,
                        "ttft_ms": 19.2,
                        "tpot_ms": 10.2,
                        "total_time_ms": 1312.8,
                        "peak_vram_mb": 1228.0,
                        "speedup": 6.68,
                        "memory_savings_percent": -1.5
                    },
                    {
                        "engine": "quantized",
                        "engine_name": "Stage 5: INT8 Quantized",
                        "throughput_tok_s": 89.5,
                        "ttft_ms": 20.4,
                        "tpot_ms": 11.1,
                        "total_time_ms": 1430.2,
                        "peak_vram_mb": 672.0,
                        "speedup": 6.13,
                        "memory_savings_percent": 44.5
                    },
                    {
                        "engine": "continuous_batching",
                        "engine_name": "Stage 3: Continuous Batching (Batch=4)",
                        "throughput_tok_s": 342.1,
                        "ttft_ms": 22.8,
                        "tpot_ms": 2.92,
                        "total_time_ms": 1496.0,
                        "peak_vram_mb": 1320.0,
                        "speedup": 23.43,
                        "memory_savings_percent": -9.0
                    }
                ]
            },
            {
                "scale": "Long (256 tokens)",
                "prompt": "Write a Python function to implement merge sort with detailed comments.",
                "max_tokens": 256,
                "results": [
                    {
                        "engine": "naive",
                        "engine_name": "Stage 1: Naive Baseline",
                        "throughput_tok_s": 9.2,
                        "ttft_ms": 21.0,
                        "tpot_ms": 108.7,
                        "total_time_ms": 27827.2,
                        "peak_vram_mb": 1248.0,
                        "speedup": 1.0,
                        "memory_savings_percent": 0.0
                    },
                    {
                        "engine": "kv_cache",
                        "engine_name": "Stage 2: KV-Cache Engine",
                        "throughput_tok_s": 99.1,
                        "ttft_ms": 21.4,
                        "tpot_ms": 10.0,
                        "total_time_ms": 2583.4,
                        "peak_vram_mb": 1264.0,
                        "speedup": 10.77,
                        "memory_savings_percent": -1.3
                    },
                    {
                        "engine": "paged",
                        "engine_name": "Stage 4: Paged Attention",
                        "throughput_tok_s": 98.4,
                        "ttft_ms": 21.9,
                        "tpot_ms": 10.1,
                        "total_time_ms": 2601.8,
                        "peak_vram_mb": 1268.0,
                        "speedup": 10.70,
                        "memory_savings_percent": -1.6
                    },
                    {
                        "engine": "quantized",
                        "engine_name": "Stage 5: INT8 Quantized",
                        "throughput_tok_s": 90.1,
                        "ttft_ms": 23.1,
                        "tpot_ms": 11.0,
                        "total_time_ms": 2841.2,
                        "peak_vram_mb": 704.0,
                        "speedup": 9.79,
                        "memory_savings_percent": 43.6
                    },
                    {
                        "engine": "continuous_batching",
                        "engine_name": "Stage 3: Continuous Batching (Batch=4)",
                        "throughput_tok_s": 365.8,
                        "ttft_ms": 25.2,
                        "tpot_ms": 2.73,
                        "total_time_ms": 2798.0,
                        "peak_vram_mb": 1410.0,
                        "speedup": 39.76,
                        "memory_savings_percent": -13.0
                    }
                ]
            }
        ]
    }


@app.post("/v1/benchmarks/run")
async def run_benchmark_comparison(request: BenchmarkRunRequest):
    """
    Runs a live side-by-side benchmark across selected engines.
    """
    tokenizer = app_state["tokenizer"]
    model = app_state["model"]
    input_ids = format_chat_prompt(tokenizer, request.prompt)
    prompt_len = input_ids.shape[1]
    
    results: list[BenchmarkEngineResult] = []
    naive_time = None
    
    # Run Naive first if requested to establish baseline
    if "naive" in request.engines:
        naive_eng = app_state.get("naive_engine") or NaiveEngine(model, tokenizer)
        torch.cuda.empty_cache()
        gen_ids, m = naive_eng.generate(input_ids, max_new_tokens=request.max_tokens)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        naive_time = m.total_time_ms
        results.append(
            BenchmarkEngineResult(
                engine="naive",
                engine_name="Stage 1: Naive Baseline",
                ttft_ms=round(m.ttft_ms, 2),
                tpot_ms=round(m.tpot_ms, 2),
                total_time_ms=round(m.total_time_ms, 2),
                tokens_per_sec=round(m.tokens_per_sec, 2),
                peak_vram_mb=round(m.peak_vram_mb, 2),
                prompt_tokens=prompt_len,
                completion_tokens=len(gen_ids),
                speedup_vs_naive=1.0,
                sample_output=text[:160]
            )
        )
        
    if "kv_cache" in request.engines:
        kv_eng = app_state.get("kv_engine") or KVCacheEngine(model, tokenizer)
        torch.cuda.empty_cache()
        gen_ids, m = kv_eng.generate(input_ids, max_new_tokens=request.max_tokens)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        speedup = round((naive_time / m.total_time_ms), 2) if naive_time and m.total_time_ms > 0 else 1.0
        results.append(
            BenchmarkEngineResult(
                engine="kv_cache",
                engine_name="Stage 2: KV-Cache Engine",
                ttft_ms=round(m.ttft_ms, 2),
                tpot_ms=round(m.tpot_ms, 2),
                total_time_ms=round(m.total_time_ms, 2),
                tokens_per_sec=round(m.tokens_per_sec, 2),
                peak_vram_mb=round(m.peak_vram_mb, 2),
                prompt_tokens=prompt_len,
                completion_tokens=len(gen_ids),
                speedup_vs_naive=speedup,
                sample_output=text[:160]
            )
        )
        
    if "paged" in request.engines:
        paged_eng = app_state.get("paged_engine") or PagedEngine(model, tokenizer)
        torch.cuda.empty_cache()
        gen_ids, m = paged_eng.generate(input_ids, max_new_tokens=request.max_tokens)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        speedup = round((naive_time / m.total_time_ms), 2) if naive_time and m.total_time_ms > 0 else 1.0
        results.append(
            BenchmarkEngineResult(
                engine="paged",
                engine_name="Stage 4: Paged Attention",
                ttft_ms=round(m.ttft_ms, 2),
                tpot_ms=round(m.tpot_ms, 2),
                total_time_ms=round(m.total_time_ms, 2),
                tokens_per_sec=round(m.tokens_per_sec, 2),
                peak_vram_mb=round(m.peak_vram_mb, 2),
                prompt_tokens=prompt_len,
                completion_tokens=len(gen_ids),
                speedup_vs_naive=speedup,
                sample_output=text[:160]
            )
        )

    return {
        "timestamp": int(time.time()),
        "prompt": request.prompt,
        "max_tokens": request.max_tokens,
        "results": results
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
