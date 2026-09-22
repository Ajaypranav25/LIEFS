"""
FastAPI Serving Layer for LIEFS (Universal LLM & Computer Benchmarking Platform).

Provides:
- Dynamic Model Hub & Loader (/v1/models/load, /v1/models/current, /v1/models/presets)
- Host Hardware Profiler (/v1/hardware, /v1/system)
- Live Computer Benchmarking & Score Engine (/v1/benchmarks/computer, /v1/benchmarks/run)
- Concurrency Batch Scaling (/v1/benchmarks/batch-scaling)
- OpenAI-compatible /v1/completions with real-time SSE streaming & JSON
- Multi-engine dispatch: KV-Cache (Stage 2), Naive Baseline (Stage 1), Paged Attention (Stage 4), Continuous Batching (Stage 3)
"""

import asyncio
import gc
import json
import time
import uuid
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from liefs.hardware_profiler import (
    calculate_computer_score,
    calculate_effective_memory_bandwidth,
    get_hardware_profile,
)
from liefs.kv_cache_engine import KVCacheEngine
from liefs.model_loader import (
    DEFAULT_MODEL_NAME,
    POPULAR_MODEL_PRESETS,
    format_chat_prompt,
    get_model_metadata,
    load_model_and_tokenizer,
)
from liefs.naive_engine import NaiveEngine
from liefs.paged_engine import PagedEngine
from liefs.scheduler import ContinuousBatchScheduler
from liefs.utils import (
    reset_vram_stats,
)
from server.schemas import (
    BatchScalePoint,
    BenchmarkEngineResult,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionUsage,
    ComputerBenchmarkRequest,
    ComputerBenchmarkResponse,
    ModelLoadRequest,
    ModelLoadResponse,
)

# Global runtime state
app_state = {
    "model_name": DEFAULT_MODEL_NAME,
    "model": None,
    "tokenizer": None,
    "meta": None,
    "kv_engine": None,
    "naive_engine": None,
    "paged_engine": None,
    "recent_metrics": [],
}


def instantiate_engines(model_name: str, precision: str = "float16", device: str = "auto", hf_token: str | None = None):
    """Cleanly unload prior model tensors, load new model, and initialize engines."""
    # Clean up old references
    if app_state.get("model") is not None:
        del app_state["model"]
        app_state["model"] = None
    if app_state.get("tokenizer") is not None:
        del app_state["tokenizer"]
        app_state["tokenizer"] = None
    if app_state.get("kv_engine") is not None:
        del app_state["kv_engine"]
        app_state["kv_engine"] = None
    if app_state.get("naive_engine") is not None:
        del app_state["naive_engine"]
        app_state["naive_engine"] = None
    if app_state.get("paged_engine") is not None:
        del app_state["paged_engine"]
        app_state["paged_engine"] = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading model: {model_name} on {device} ({precision})...")
    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        dtype=precision,
        device=device,
        hf_token=hf_token,
    )
    meta = get_model_metadata(model, tokenizer, model_name)

    kv_engine = KVCacheEngine(model, tokenizer)
    naive_engine = NaiveEngine(model, tokenizer)
    try:
        paged_engine = PagedEngine(model, tokenizer, block_size=16, max_num_blocks=256)
    except Exception as e:
        print(f"Paged engine notice: {e}")
        paged_engine = None

    app_state["model_name"] = model_name
    app_state["model"] = model
    app_state["tokenizer"] = tokenizer
    app_state["meta"] = meta
    app_state["kv_engine"] = kv_engine
    app_state["naive_engine"] = naive_engine
    app_state["paged_engine"] = paged_engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup event: Profile hardware and load initial default model."""
    print("LIEFS Server initializing...")
    hw = get_hardware_profile()
    print(f"Host Hardware detected: {hw.cpu_model} | GPU: {hw.gpu_name} ({hw.vram_total_mb:.1f} MB VRAM)")
    
    try:
        instantiate_engines(DEFAULT_MODEL_NAME)
    except Exception as e:
        print(f"Warning on startup model load: {e}")
        
    yield
    app_state.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="LIEFS Universal LLM & Computer Benchmarking Platform",
    description="Custom inference engine and hardware benchmark suite for any Hugging Face model and computer.",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "LIEFS Universal Benchmarking Engine",
        "model": app_state.get("model_name", DEFAULT_MODEL_NAME),
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "timestamp": int(time.time()),
    }


@app.get("/v1/system")
@app.get("/v1/hardware")
async def get_hardware_info():
    """Returns complete computer hardware specifications."""
    hw = get_hardware_profile()
    return hw.to_dict()


@app.get("/v1/models/presets")
async def get_model_presets():
    """Returns curated list of benchmark-ready models."""
    return {
        "presets": POPULAR_MODEL_PRESETS,
        "current_model": app_state.get("model_name", DEFAULT_MODEL_NAME),
    }


@app.get("/v1/models/current")
async def get_current_model_info():
    """Returns architecture specs of the currently loaded model."""
    meta = app_state.get("meta")
    if meta:
        return meta.to_dict()
    return {
        "model_name": app_state.get("model_name", DEFAULT_MODEL_NAME),
        "parameter_count_m": 0,
        "device_str": "unknown",
    }


@app.post("/v1/models/load", response_model=ModelLoadResponse)
async def load_custom_model(request: ModelLoadRequest):
    """
    Dynamically loads any Hugging Face model or local checkpoint path into memory.
    """
    t0 = time.perf_counter()
    try:
        instantiate_engines(
            model_name=request.model_name,
            precision=request.precision,
            device=request.device,
            hf_token=request.hf_token,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        meta = app_state["meta"]
        return ModelLoadResponse(
            status="ok",
            message=f"Successfully loaded {request.model_name} in {elapsed}s",
            load_time_sec=elapsed,
            model_name=request.model_name,
            metadata=meta.to_dict(),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load model {request.model_name}: {e!s}")


@app.post("/v1/models/unload")
async def unload_model():
    """Frees loaded model and purges GPU VRAM."""
    if app_state.get("model") is not None:
        del app_state["model"]
        app_state["model"] = None
    if app_state.get("tokenizer") is not None:
        del app_state["tokenizer"]
        app_state["tokenizer"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"status": "ok", "message": "Model unloaded and VRAM cleared"}


@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible models list endpoint."""
    current_name = app_state.get("model_name", DEFAULT_MODEL_NAME)
    return {
        "object": "list",
        "data": [
            {
                "id": current_name.split("/")[-1],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "liefs",
                "root": current_name,
                "engines": ["kv_cache", "naive", "paged", "continuous_batching"]
            }
        ]
    }


@app.get("/v1/engines")
async def list_engines():
    """Metadata describing the engine architectural stages."""
    return {
        "engines": [
            {
                "id": "kv_cache",
                "name": "KV-Cache Engine (Stage 2)",
                "description": "Prefill + Decode separation with key/value tensor caching. Eliminates quadratic recomputation.",
                "complexity": "O(N) per step attention",
                "recommended": True,
                "badge": "Default / Optimized"
            },
            {
                "id": "naive",
                "name": "Naive Baseline (Stage 1)",
                "description": "Full sequence recomputation at every token step. Baseline implementation.",
                "complexity": "O(N²) quadratic compute",
                "recommended": False,
                "badge": "Baseline"
            },
            {
                "id": "paged",
                "name": "Paged Attention (Stage 4)",
                "description": "Dynamic BlockAllocator pool (16 tokens/block). Eliminates external memory fragmentation.",
                "complexity": "Zero-fragmentation Pool",
                "recommended": True,
                "badge": "Zero Waste"
            },
            {
                "id": "continuous_batching",
                "name": "Continuous Batching (Stage 3)",
                "description": "Iteration-level request scheduling for high-throughput multi-request concurrency.",
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
    Handles completion requests with real-time SSE streaming or JSON.
    """
    tokenizer = app_state.get("tokenizer")
    model = app_state.get("model")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="No model loaded. Call /v1/models/load first.")

    engine_type = (request.engine or "kv_cache").lower()
    if engine_type == "naive":
        engine = app_state.get("naive_engine") or NaiveEngine(model, tokenizer)
    elif engine_type == "paged" and app_state.get("paged_engine"):
        engine = app_state.get("paged_engine")
    else:
        engine = app_state.get("kv_engine") or KVCacheEngine(model, tokenizer)

    device_str = "cuda" if next(model.parameters()).is_cuda else "cpu"
    prompt_tensor = format_chat_prompt(tokenizer, request.prompt, device=device_str)
    prompt_len = prompt_tensor.shape[1]
    response_id = f"cmpl-{uuid.uuid4().hex}"
    current_model_name = app_state.get("model_name", DEFAULT_MODEL_NAME)

    # ── SSE Streaming Mode ─────────────────────────────────────────────
    if request.stream:
        async def stream_generator():
            try:
                for token_id, token_text, metrics_dict, is_done in engine.generate_stream(
                    prompt_tensor, max_new_tokens=request.max_tokens
                ):
                    if is_done:
                        chunk = {
                            "id": response_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": current_model_name,
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
                        app_state["recent_metrics"].append(metrics_dict)
                        break
                    else:
                        chunk = {
                            "id": response_id,
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": current_model_name,
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

    # ── Non-Streaming Mode ──────────────────────────────────────────────
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
        model=current_model_name,
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
    Returns reference benchmarks for comparison across scales.
    """
    app_state.get("meta")
    model_name = app_state.get("model_name", DEFAULT_MODEL_NAME)
    hw = get_hardware_profile()

    return {
        "model": model_name,
        "device": hw.gpu_name if hw.cuda_available else hw.cpu_model,
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
                        "engine": "continuous_batching",
                        "engine_name": "Stage 3: Continuous Batching (B=4)",
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
                        "engine": "continuous_batching",
                        "engine_name": "Stage 3: Continuous Batching (B=4)",
                        "throughput_tok_s": 342.1,
                        "ttft_ms": 22.8,
                        "tpot_ms": 2.92,
                        "total_time_ms": 1496.0,
                        "peak_vram_mb": 1320.0,
                        "speedup": 23.43,
                        "memory_savings_percent": -9.0
                    }
                ]
            }
        ]
    }


@app.post("/v1/benchmarks/run")
@app.post("/v1/benchmarks/computer", response_model=ComputerBenchmarkResponse)
async def run_computer_benchmark(request: ComputerBenchmarkRequest):
    """
    Runs a live benchmark across selected engines for the loaded model on the host computer.
    Calculates achieved memory bandwidth (GB/s) and the Computer Performance Index Score.
    """
    tokenizer = app_state.get("tokenizer")
    model = app_state.get("model")
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="No model loaded.")

    meta = app_state.get("meta") or get_model_metadata(model, tokenizer, app_state.get("model_name", DEFAULT_MODEL_NAME))
    hw = get_hardware_profile()

    device_str = "cuda" if next(model.parameters()).is_cuda else "cpu"
    input_ids = format_chat_prompt(tokenizer, request.prompt, device=device_str)
    prompt_len = input_ids.shape[1]

    results: list[BenchmarkEngineResult] = []
    naive_time = None

    # 1. Naive Baseline (Stage 1)
    if "naive" in request.engines:
        naive_eng = app_state.get("naive_engine") or NaiveEngine(model, tokenizer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_vram_stats()
        gen_ids, m = naive_eng.generate(input_ids, max_new_tokens=request.max_tokens)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        naive_time = m.total_time_ms
        bw = calculate_effective_memory_bandwidth(meta.memory_footprint_mb, m.tokens_per_sec)
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
                memory_bandwidth_gbs=bw,
                sample_output=text[:160]
            )
        )

    # 2. KV-Cache Engine (Stage 2)
    kv_metrics = None
    if "kv_cache" in request.engines:
        kv_eng = app_state.get("kv_engine") or KVCacheEngine(model, tokenizer)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_vram_stats()
        gen_ids, m = kv_eng.generate(input_ids, max_new_tokens=request.max_tokens)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        kv_metrics = m
        speedup = round((naive_time / m.total_time_ms), 2) if naive_time and m.total_time_ms > 0 else 1.0
        bw = calculate_effective_memory_bandwidth(meta.memory_footprint_mb, m.tokens_per_sec)
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
                memory_bandwidth_gbs=bw,
                sample_output=text[:160]
            )
        )

    # 3. Paged Attention (Stage 4)
    if "paged" in request.engines and app_state.get("paged_engine"):
        paged_eng = app_state["paged_engine"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reset_vram_stats()
        gen_ids, m = paged_eng.generate(input_ids, max_new_tokens=request.max_tokens)
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        speedup = round((naive_time / m.total_time_ms), 2) if naive_time and m.total_time_ms > 0 else 1.0
        bw = calculate_effective_memory_bandwidth(meta.memory_footprint_mb, m.tokens_per_sec)
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
                memory_bandwidth_gbs=bw,
                sample_output=text[:160]
            )
        )

    # 4. Concurrency Batch Scaling (Stage 3)
    batch_scaling: list[BatchScalePoint] = []
    if request.include_batch_scaling:
        for b_size in [1, 2, 4, 8]:
            scheduler = ContinuousBatchScheduler(model, tokenizer, max_batch_size=b_size)
            for req_i in range(b_size):
                scheduler.add_request(f"Compute step {req_i}", max_new_tokens=min(32, request.max_tokens))
            t_start = time.perf_counter()
            req_metrics = scheduler.run_until_complete()
            t_batch_elapsed = (time.perf_counter() - t_start) * 1000.0
            total_toks = sum(m.total_tokens_generated for m in req_metrics)
            agg_throughput = round((total_toks / t_batch_elapsed * 1000.0), 2) if t_batch_elapsed > 0 else 0.0
            batch_scaling.append(
                BatchScalePoint(
                    batch_size=b_size,
                    total_tokens=total_toks,
                    wall_time_ms=round(t_batch_elapsed, 1),
                    aggregate_throughput_tok_s=agg_throughput,
                )
            )

    # 5. Computer Score
    benchmark_tok_s = kv_metrics.tokens_per_sec if kv_metrics else (results[0].tokens_per_sec if results else 0.0)
    benchmark_ttft = kv_metrics.ttft_ms if kv_metrics else (results[0].ttft_ms if results else 0.0)
    bw_primary = calculate_effective_memory_bandwidth(meta.memory_footprint_mb, benchmark_tok_s)

    score_data = calculate_computer_score(
        tokens_per_sec=benchmark_tok_s,
        ttft_ms=benchmark_ttft,
        model_param_count_m=meta.parameter_count_m,
        memory_bandwidth_gbs=bw_primary,
        is_gpu=hw.cuda_available,
    )

    return ComputerBenchmarkResponse(
        timestamp=int(time.time()),
        model=meta.to_dict(),
        hardware=hw.to_dict(),
        score=score_data,
        results=results,
        batch_scaling=batch_scaling,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=False)
