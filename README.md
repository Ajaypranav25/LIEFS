# LIEFS — Universal LLM & Computer Hardware Benchmarking Platform

LIEFS is an open, high-performance LLM inference engine built from the ground up in raw PyTorch, paired with a universal hardware profiling and benchmark suite. Load **any Hugging Face model or local checkpoint** to evaluate your computer's CPU/GPU inference throughput (tok/s), prefill latency (TTFT), decode latency (TPOT), achieved memory bandwidth (GB/s), 5-engine optimization speedups, and calculate your standardized **Computer Performance Index Score**.

---

## ⚡ Key Capabilities

1. **Universal Model Loader (BYOM - Bring Your Own Model):**
   - Load any Hugging Face causal LM (`meta-llama/Llama-3.2-1B`, `Qwen/Qwen2.5-0.5B`, `google/gemma-2-2b`, `TinyLlama/TinyLlama-1.1B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `HuggingFaceTB/SmolLM2-360M`, etc.) or local checkpoint paths.
   - Dynamic device selection (`cuda` GPU, `cpu` universal) and precision (`float16`, `bfloat16`, `float32`).
   - Supports private/gated models via Hugging Face access tokens.

2. **Host Hardware Profiling:**
   - Auto-detects processor brand, physical & logical core count, total & available RAM.
   - Inspects GPU accelerator, compute capability (e.g. CUDA 8.9), VRAM allocated/free/total.
   - Calculates real-world achieved memory bandwidth ($B = \text{Model Size in GB} \times \text{Tokens/sec}$).

3. **LIEFS Computer Performance Index Score:**
   - Computes a normalized performance score measuring compute throughput, memory streaming bandwidth, and prefill responsiveness.
   - Assigns standardized hardware tier ratings (Tier S Workstation, Tier A High-Performance GPU, Tier B Laptop GPU, Tier C Fast CPU).

4. **5-Engine Optimization Comparison:**
   - **Stage 1 (Naive Baseline):** Full recomputation autoregressive baseline ($O(N^2)$ quadratic attention).
   - **Stage 2 (KV-Cache):** Key/Value tensor caching for $O(1)$ decode forward passes.
   - **Stage 3 (Continuous Batching):** Iteration-level scheduling across concurrency levels ($B=1, 2, 4, 8, 16$).
   - **Stage 4 (Paged Attention):** Zero-fragmentation virtual block memory allocator.
   - **Stage 5 (INT8 Quantization):** Weight-only quantization with on-the-fly FP16 dequantization.

5. **Dual Interface (Web Dashboard & Standalone CLI):**
   - **Web UI:** Interactive dashboard with Model Hub, Hardware HUD, Live Recharts visualizations, and Chat Playground.
   - **CLI Tool:** `python -m benchmarks.bench_computer` for headless terminal benchmarking and JSON report exporting.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Ajaypranav25/LIEFS.git
cd LIEFS

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend
npm install
cd ..
```

---

### 2. Run Computer Benchmarks via CLI

Run a comprehensive benchmark of your computer using any model:

```bash
# Quick benchmark on default model
python -m benchmarks.bench_computer --quick

# Benchmark with a custom Hugging Face model
python -m benchmarks.bench_computer --model meta-llama/Llama-3.2-1B-Instruct --device cuda

# Benchmark on CPU and export JSON results
python -m benchmarks.bench_computer --model HuggingFaceTB/SmolLM2-360M-Instruct --device cpu --export results.json
```

**Example Terminal Output:**
```
======================================================================================================
  BENCHMARK RESULTS: Qwen/Qwen2.5-0.5B-Instruct on AMD Ryzen 9 / NVIDIA GeForce RTX 4060 Laptop GPU
======================================================================================================
| Engine / Optimization          | Throughput   | TTFT       | TPOT       | VRAM       | Bandwidth  | Speedup  |
|--------------------------------|--------------|------------|------------|------------|------------|----------|
| KV-Cache (Stage 2)             |     98.2 tok/s |    18.5 ms |    10.1 ms |  1009.1 MB |   92.5 GB/s |   6.73x |
| Naive Recomputation (Stage 1)  |     14.6 tok/s |    18.5 ms |    68.5 ms |  1035.9 MB |   13.7 GB/s |   1.00x |
| Paged Attention (Stage 4)      |     97.5 tok/s |    19.2 ms |    10.2 ms |  1009.5 MB |   91.8 GB/s |   6.68x |
======================================================================================================

######################################################################
  LIEFS COMPUTER PERFORMANCE SCORE:  875 POINTS  [Tier A]
  Hardware Tier:                     High-Performance Discrete GPU
  Effective Memory Bandwidth:        92.5 GB/s
  KV-Cache Elimination Speedup:      6.73x faster than Naive
######################################################################
```

---

### 3. Run the Interactive Web Platform

Start the backend server and Vite frontend:

```bash
# Terminal 1: Backend Server
python -m server.app

# Terminal 2: Frontend Dashboard
cd frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

- Click **"Load / Switch Model"** in the top bar to paste any Hugging Face repo ID or select quick presets.
- Navigate to **"Computer Benchmarks"** to run live hardware tests, view interactive latency/throughput charts, and export shareable benchmark cards.
- Test real-time token streaming with live per-token telemetry HUD in **"AI Chat"**.

---

## 🛠 Project Structure

```
LIEFS/
├── liefs/                      # Core Inference & Hardware Engine Modules
│   ├── __init__.py
│   ├── hardware_profiler.py    # CPU/GPU inspection, bandwidth calculation & scoring
│   ├── model_loader.py         # Universal HF & local model loader with metadata extraction
│   ├── kv_cache_engine.py      # Stage 2: Prefill + Decode KV-cache engine
│   ├── naive_engine.py         # Stage 1: Full-recomputation naive baseline
│   ├── paged_attention.py      # Stage 4: BlockAllocator & virtual paged pool
│   ├── paged_engine.py         # Stage 4: Paged attention generation engine
│   ├── quantization.py         # Stage 5: Per-channel symmetric INT8 QuantizedLinear
│   ├── quantized_engine.py     # Stage 5: Quantized engine constructor
│   ├── scheduler.py            # Stage 3: Continuous batching scheduler & state machine
│   └── utils.py                # CUDA/CPU synced timing, VRAM stats & stop tokens
├── server/                     # FastAPI Serving & Benchmarking Layer
│   ├── __init__.py
│   ├── app.py                  # Dynamic model loading, hardware info & benchmark API
│   └── schemas.py              # Pydantic request/response schemas
├── benchmarks/                 # CLI Benchmark Suites
│   ├── bench_computer.py       # Standalone CLI computer & model benchmark runner
│   ├── bench_kv_cache.py       # Stage 2 benchmark
│   ├── bench_continuous.py     # Stage 3 benchmark
│   ├── bench_paged.py          # Stage 4 benchmark
│   ├── bench_quantized.py      # Stage 5 benchmark
│   └── prompts.py              # Standardized prompt suites
├── tests/                      # Automated Unit Test Suite (44/44 Passing)
│   ├── test_hardware.py        # Hardware profiler & score algorithm tests
│   ├── test_model_loader_general.py # Model resolution & metadata tests
│   ├── test_server_api.py      # FastAPI endpoint integration tests
│   ├── test_continuous.py      # Continuous batching tests
│   ├── test_kv_cache.py        # KV-Cache engine tests
│   ├── test_naive.py           # Naive baseline tests
│   ├── test_paged.py           # Paged attention tests
│   └── test_quantized.py       # INT8 quantization tests
└── frontend/                   # React + TypeScript + Tailwind + Vite Dashboard
    ├── src/
    │   ├── components/
    │   │   ├── BenchmarksView.tsx       # Live computer benchmark runner & Recharts
    │   │   ├── HardwareProfileHUD.tsx   # Host CPU/GPU/VRAM status banner
    │   │   ├── ModelLoaderModal.tsx     # Custom Hugging Face model importer
    │   │   ├── ArchitectureView.tsx     # Model KV memory calculator
    │   │   ├── ChatView.tsx             # Live streaming AI playground
    │   │   ├── Header.tsx               # Top navigation with active model switcher
    │   │   └── Sidebar.tsx              # Conversation history & status
    │   ├── lib/                         # API client & SSE streaming parser
    │   └── types/                       # TypeScript schemas
    ├── package.json
    └── vite.config.ts
```

---

## 🧪 Testing

Run the full pytest suite:

```bash
pytest
```

Run frontend component tests:

```bash
cd frontend && npm test
```
