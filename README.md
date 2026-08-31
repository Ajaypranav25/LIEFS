# LIEFS — LLM Inference Engine From Scratch

A pedagogical implementation of an LLM inference engine, built from the ground
up in raw PyTorch. Every optimization is implemented by hand (not imported from
a library), benchmarked, and documented.

**Model:** Qwen2.5-0.5B-Instruct (494M params, GQA, SwiGLU, RoPE)  
**Hardware:** AMD Ryzen 9 + NVIDIA RTX 4060 (8GB VRAM)  
**Stack:** PyTorch for inference, HuggingFace for weight/tokenizer loading only

## Stages

| Stage | What | Status |
|-------|------|--------|
| 1 | Naive baseline (full recomputation, no KV-cache) | 🔧 In Progress |
| 2 | Manual KV-cache implementation | ⬜ |
| 3 | Continuous batching scheduler | ⬜ |
| 4 | Simplified paged attention | ⬜ |
| 5 | INT8 weight quantization | ⬜ |
| 6 | FastAPI serving + benchmark report | ⬜ |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run correctness tests
python -m pytest tests/ -v

# Run Stage 1 benchmark
python -m benchmarks.bench_naive
```

## Project Structure

```
LIEFS/
├── liefs/                   # Main inference engine package
│   ├── model_loader.py      # HuggingFace model/tokenizer loading
│   ├── naive_engine.py      # Stage 1: naive generation (full recomputation)
│   └── utils.py             # Timing, metrics, and logging utilities
├── benchmarks/              # Benchmark scripts and data
│   ├── prompts.py           # Fixed prompt set for reproducible benchmarks
│   └── bench_naive.py       # Stage 1 benchmark
├── tests/                   # Correctness tests
│   └── test_naive.py        # Stage 1 tests
├── requirements.txt
└── README.md
```

## Benchmark Results

### Stage 1 — Naive Baseline (full recomputation, no KV-cache)

**Hardware:** RTX 4060 Laptop (8GB VRAM) | PyTorch 2.13.0+cu126 | fp16

| Prompt | Tokens | TTFT (ms) | TPOT (ms) | Throughput (tok/s) | Peak VRAM (MB) |
|--------|--------|-----------|-----------|--------------------|----------------|
| Short ("What is 2+2?") | 9 | 35.3 ± 2.3 | 34.6 ± 1.0 | 26.0 ± 0.6 | 971 |
| Medium (explain transformers) | 128 | 33.9 ± 0.2 | 34.1 ± 0.3 | 29.4 ± 0.3 | 1,043 |
| Long (write merge sort) | 256 | 37.0 ± 3.8 | 35.6 ± 0.2 | 28.1 ± 0.2 | 1,120 |

**vs. HuggingFace `model.generate()` (with KV-cache):**

| Prompt | Naive (tok/s) | HF generate (tok/s) | HF Speedup | HF VRAM (MB) |
|--------|---------------|---------------------|------------|--------------|
| Short | 26.0 | 31.8 | 1.2× | 953 |
| Medium | 29.4 | 31.9 | 1.1× | 955 |
| Long | 28.1 | 32.1 | 1.1× | 958 |

## Design Decisions

See inline documentation in each module for detailed explanations of the
"why" behind each design choice.
