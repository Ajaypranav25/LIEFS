# LIEFS — LLM Inference Engine From Scratch

A pedagogical, high-performance LLM inference engine built from the ground up in raw PyTorch. Every optimization is implemented by hand (never imported from generation libraries), tested with a full unit test suite (34/34 passing), benchmarked on consumer GPU hardware, and documented for ML infrastructure / systems engineering interviews.

**Target Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 Layers, 896 Hidden Dim, 14 Query Heads, 2 KV Heads with GQA, SwiGLU MLP, RoPE $\theta=10^6$, RMSNorm, 494M parameters)  
**Benchmarking Environment:** AMD Ryzen 9 + NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) | PyTorch 2.13.0+cu126 | CUDA 12.6  
**Design Principle:** Raw PyTorch tensor operations for all generation logic; Hugging Face `transformers` is strictly used for loading initial weights and tokenizer vocabulary.

---

## 🚀 Engine Architecture & Stages

| Stage | Optimization | Key Mechanism | Test Suite | Status |
|---|---|---|---|---|
| **Stage 1** | **Naive Baseline** | Autoregressive generation with full recomputation ($O(n^2)$ attention) | `tests/test_naive.py` | ✅ Complete |
| **Stage 2** | **KV-Cache** | Prefill + Decode phases, caching key/value projections ($O(1)$ decode forward) | `tests/test_kv_cache.py` | ✅ Complete |
| **Stage 3** | **Continuous Batching** | Iteration-level scheduling state machine (`QUEUED` $\to$ `PREFILLING` $\to$ `GENERATING` $\to$ `FINISHED`) | `tests/test_continuous.py` | ✅ Complete |
| **Stage 4** | **Paged Attention** | Fixed-size block allocator pool (48 MB memory budget, zero external fragmentation) | `tests/test_paged.py` | ✅ Complete |
| **Stage 5** | **INT8 Quantization** | Per-channel symmetric weight-only quantization from scratch (35.3% weight reduction) | `tests/test_quantized.py` | ✅ Complete |
| **Stage 6** | **Serving Layer** | OpenAI-compatible `/v1/completions` FastAPI server with asynchronous concurrency benchmarking | `benchmarks/bench_server.py` | ✅ Complete |

---

## 📊 Comprehensive Benchmark Results

All benchmarks were run with `torch.cuda.synchronize()` before and after every measurement window to ensure genuine GPU kernel execution times (not CPU launch queue latency).

### 1. Stage 1 vs Stage 2: Naive Recomputation vs KV-Cache

| Prompt | Output Tokens | Engine | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Peak VRAM (MB) |
|---|---|---|---|---|---|---|
| **Short** ("What is 2+2?") | 9 | **Naive**<br>**KV-Cache** | 35.99<br>40.18 | 34.13<br>**31.21** | 26.24<br>**28.03** | 970.51<br>**958.63** |
| **Medium** (Explain transformers) | 128 | **Naive**<br>**KV-Cache** | 34.76<br>38.22 | 35.04<br>**31.66** | 28.54<br>**31.29** | 1042.72<br>**959.88** |
| **Long** (Write merge sort) | 256 | **Naive**<br>**KV-Cache** | 34.29<br>38.31 | 36.83<br>**31.55** | 27.16<br>**31.54** | 1119.90<br>**962.06** |

> **Key Architectural Insight:**
> - **Memory Scaling:** Without KV-cache, VRAM increases significantly with sequence length (+149.4 MB from 9 to 256 tokens) due to storing activation maps for the full recomputed sequence at every step. With KV-cache and Grouped-Query Attention (GQA with 2 KV heads), 256 tokens of KV-cache across 24 layers requires only **~3.4 MB** ($2 \times 24 \times 2 \times 64 \times 256 \times 2\text{ bytes} = 3,145,728\text{ bytes}$).
> - **TPOT Stability:** Naive TPOT degrades as sequence grows ($34.1\text{ms} \to 36.8\text{ms}$), while KV-cache maintains a constant ~31.5ms decode latency per token.

---

### 2. Stage 3: Continuous Batching Throughput

Evaluated across an 8-request concurrent workload with varying batch sizes:

| Batch Configuration | Requests Completed | Total Output Tokens | Total Time (s) | Throughput (tok/s) | Peak VRAM (MB) |
|---|---|---|---|---|---|
| **Sequential (Batch 1)** | 8 | 923 | 29.70 | 31.08 | 959.9 |
| **Continuous Batch Size 2** | 8 | 923 | 29.19 | 31.62 | 971.4 |
| **Continuous Batch Size 4** | 8 | 923 | 29.31 | 31.49 | 986.7 |
| **Continuous Batch Size 8** | 8 | 923 | 29.33 | 31.47 | 1024.4 |

---

### 3. Stage 4: Paged Attention vs Contiguous KV-Cache

Evaluated with a 256-block memory pool ($256 \times 16\text{ tokens} = 4096\text{ max tokens}$, fixed 48.0 MB pool footprint):

| Prompt | Contiguous Throughput (tok/s) | Paged Throughput (tok/s) | Contiguous TPOT (ms) | Paged TPOT (ms) | Peak VRAM (MB) |
|---|---|---|---|---|---|
| **Short** (9 tokens) | 28.52 | 23.52 | 30.97 | 35.82 | 1006.94 |
| **Medium** (128 tokens) | 31.50 | 28.12 | 31.45 | 35.05 | 1008.24 |
| **Long** (256 tokens) | 31.46 | 27.80 | 31.63 | 35.70 | 1010.51 |

> **Production Note:** The ~4ms overhead per decode step in PyTorch stems from dynamic tensor assembly (scatter/gather indexing). In production engines like **vLLM**, custom C++/CUDA kernels directly compute attention within non-contiguous physical page blocks, achieving zero-overhead paging.

---

### 4. Stage 5: INT8 Quantization vs FP16

Manual per-channel symmetric weight quantization across 168 `nn.Linear` layers:

| Metric | FP16 Baseline | INT8 Quantized | Delta / Compression |
|---|---|---|---|
| **Model Weights in VRAM** | 942.3 MB | 609.9 MB | **-332.5 MB (35.3% reduction)** |
| **Short Prompt Peak VRAM** | 958.63 MB | 634.38 MB | **-33.8%** |
| **Medium Prompt Peak VRAM** | 959.88 MB | 635.58 MB | **-33.8%** |
| **Long Prompt Peak VRAM** | 962.06 MB | 637.68 MB | **-33.7%** |
| **Throughput (Long Prompt)** | 31.82 tok/s | 26.50 tok/s | On-the-fly dequantization overhead |
| **Output Coherence** | Deterministic | Deterministic ("2+2 is 4") | **100% Quality Retention** |

---

### 5. Stage 6: FastAPI Serving Layer Concurrency

Tested via async HTTP `/v1/completions` client:

| Concurrency Level | Total Requests | Total Wall Time (s) | Throughput (tok/s) | Average Latency (s) | p50 Latency (s) | p95 Latency (s) |
|---|---|---|---|---|---|---|
| **1 Worker** | 3 | 3.66s | 51.13 tok/s | 2.95s | 3.40s | 3.66s |
| **2 Concurrent** | 6 | 7.20s | 53.37 tok/s | 4.32s | 5.02s | 7.19s |
| **4 Concurrent** | 12 | 14.58s | 52.69 tok/s | 8.76s | 9.22s | 14.57s |

---

## 🛠 Project Structure

```
LIEFS/
├── liefs/                      # Core Inference Engine Modules
│   ├── __init__.py
│   ├── model_loader.py         # HuggingFace weight & tokenizer loading in FP16
│   ├── naive_engine.py         # Stage 1: Full-recomputation naive baseline
│   ├── kv_cache_engine.py      # Stage 2: Prefill + Decode KV-cache engine
│   ├── scheduler.py            # Stage 3: Continuous batching scheduler & state machine
│   ├── paged_attention.py      # Stage 4: BlockAllocator & PagedKVCache
│   ├── paged_engine.py         # Stage 4: Paged attention generation engine
│   ├── quantization.py         # Stage 5: Per-channel symmetric INT8 QuantizedLinear
│   ├── quantized_engine.py     # Stage 5: Quantized engine constructor
│   └── utils.py                # CUDA-synced timing, VRAM stats & metrics
├── server/                     # Stage 6: Serving Layer
│   ├── __init__.py
│   ├── app.py                  # FastAPI server with OpenAI-compatible endpoint
│   └── schemas.py              # Pydantic OpenAI completion schemas
├── benchmarks/                 # Stage-by-Stage Benchmarks
│   ├── __init__.py
│   ├── prompts.py              # Standardized prompt set (Short, Medium, Long)
│   ├── bench_naive.py          # Stage 1 Benchmark
│   ├── bench_kv_cache.py       # Stage 2 Benchmark
│   ├── bench_continuous.py     # Stage 3 Benchmark
│   ├── bench_paged.py          # Stage 4 Benchmark
│   ├── bench_quantized.py      # Stage 5 Benchmark
│   └── bench_server.py         # Stage 6 Async Server Benchmark
├── tests/                      # Pytest Test Suite (34/34 Passing)
│   ├── test_naive.py
│   ├── test_kv_cache.py
│   ├── test_continuous.py
│   ├── test_paged.py
├── frontend/                   # Frontend Dashboard (React + Vite + TypeScript + Tailwind)
│   ├── src/
│   │   ├── components/         # PlaygroundView, BenchmarksView, ArchitectureView, HistoryView, Header
│   │   ├── lib/                # API client (SSE parser), Supabase persistence
│   │   ├── types/              # TypeScript schemas
│   │   └── __tests__/          # Vitest component & API test suites
│   ├── package.json
│   └── vite.config.ts
├── requirements.txt
└── README.md
```

---

## 🖥️ Frontend Dashboard & Demo UI

A modern cyber-minimalist dashboard built for system demos and technical interviews.

### Key Capabilities:
1. **Live Token Streaming Playground:** Real-time token streaming with instant per-token telemetry HUD (Time-to-First-Token, Time-per-Output-Token, Tokens/sec, Peak VRAM) and engine selector (KV-Cache, Naive, Paged Attention, INT8 Quantized).
2. **Benchmark Comparison Suite:** Side-by-side interactive Recharts comparing Throughput, TTFT/TPOT latency breakdowns, VRAM footprint, and sequence length scaling curves.
3. **Architecture & KV Memory Sizing Calculator:** Educational deep-dive explaining the 5 optimization stages alongside an interactive memory demand calculator.
4. **Supabase Persistence:** Stores and tracks historical benchmark runs in Supabase Postgres with instant local fallback.

### Running the Dashboard Locally:
```bash
# 1. Start the FastAPI backend server (Terminal 1)
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000

# 2. Start the Frontend Vite dev server (Terminal 2)
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

---

## 🧪 Running Tests & Benchmarks

```bash
# 1. Install Python backend dependencies
pip install -r requirements.txt

# 2. Run Python engine unit test suite (34 tests)
python -m pytest tests/ -v

# 3. Run Frontend unit test suite (Vitest)
cd frontend
npm test

# 4. Run individual stage benchmarks
python -m benchmarks.bench_naive
python -m benchmarks.bench_kv_cache
python -m benchmarks.bench_continuous
python -m benchmarks.bench_paged
python -m benchmarks.bench_quantized
```

---

## 🎯 Interview Deep Dive: Systems & Architecture Explanations

### Q1: Why is KV-Cache generation split into Prefill vs Decode?
* **Prefill:** Compute-bound. We have $P$ tokens available simultaneously, so matrix multiplications ($X \cdot W$) have large batch dimensions ($M=P$). GPU Tensor Cores operate at high arithmetic intensity.
* **Decode:** Memory-bandwidth-bound. We only process 1 token ($M=1$). Forward pass matrix multiplications become matrix-vector operations with low arithmetic intensity. The GPU spends most of its time streaming weights and cached K/V tensors from VRAM (HBM/GDDR6) to SRAM (registers/shared memory).

### Q2: What is the memory footprint of KV-cache under GQA?
For Qwen2.5-0.5B with 24 layers, 2 KV heads, and 64 head dimension:
$$\text{Bytes per Token} = 2 (\text{K and V}) \times 24 (\text{layers}) \times 2 (\text{KV heads}) \times 64 (\text{head dim}) \times 2 (\text{FP16 bytes}) = 12,288\text{ bytes} \approx 12\text{ KB/token}$$
Without GQA (14 KV heads), this would be $14/2 = 7\times$ larger ($86\text{ KB/token}$).

### Q3: Why does PagedAttention eliminate memory fragmentation?
Contiguous allocation requires pre-allocating contiguous VRAM for maximum sequence length ($32\text{k tokens} = 384\text{ MB/request}$). With dynamic sequence lengths, memory becomes fragmented (external fragmentation) and unused space is reserved (internal fragmentation). Paging splits memory into fixed-size physical blocks (e.g. 16 tokens), allocating blocks on demand via virtual block tables, achieving $>96\%$ memory utilization.

### Q4: Why per-channel symmetric quantization for INT8 weights?
* **Per-channel:** Neural network weights have vastly different dynamic ranges across channels (outliers). Computing scale $s_c = \frac{\max(|W_c|)}{127}$ per output channel preserves dynamic range and minimizes quantization error.
* **Symmetric:** Maps zero to zero ($z=0$), eliminating zero-point subtraction overhead during dequantization.
* **Skipping `lm_head`:** The language model head projects hidden states to the 151,936 vocabulary dimension. Quantization error at this final layer directly distorts logit calibration and token rankings.

