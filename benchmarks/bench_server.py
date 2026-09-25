"""
Benchmark Script for the Serving Layer.

Measures throughput and latency distribution under concurrent load.
Compares single-request vs concurrent performance.
"""

import asyncio
import time

import aiohttp

# Minimal prompts to test with
PROMPTS = [
    "Write a short poem about the sea.",
    "Explain quantum computing in simple terms.",
    "What is the capital of France?",
    "Write a python function to compute Fibonacci numbers.",
    "Translate 'Hello world' to Spanish.",
]

API_URL = "http://127.0.0.1:8000/v1/completions"


async def send_request(session, prompt):
    payload = {"prompt": prompt, "max_tokens": 50, "temperature": 0.0, "stream": False}
    start = time.time()
    try:
        async with session.post(API_URL, json=payload) as response:
            res = await response.json()
            latency = time.time() - start
            return {"latency": latency, "status": response.status, "data": res}
    except Exception as e:  # noqa: BLE001
        return {"latency": time.time() - start, "status": 500, "error": str(e)}


async def run_batch(concurrency, num_requests=10):
    print(
        f"\n--- Running benchmark with concurrency={concurrency}, requests={num_requests} ---"
    )

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i in range(num_requests):
            prompt = PROMPTS[i % len(PROMPTS)]
            tasks.append(send_request(session, prompt))

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

    latencies = [r["latency"] for r in results if r.get("status") == 200]
    errors = [r for r in results if r.get("status") != 200]

    if errors:
        print(f"Errors occurred: {len(errors)}")
        for e in errors:
            print(e)

    if not latencies:
        print("No successful requests.")
        return

    latencies.sort()
    avg_latency = sum(latencies) / len(latencies)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) > 1 else latencies[-1]
    p99 = latencies[int(len(latencies) * 0.99)] if len(latencies) > 1 else latencies[-1]

    total_tokens = sum(
        [
            r.get("data", {}).get("usage", {}).get("total_tokens", 0)
            for r in results
            if r.get("status") == 200
        ]
    )
    throughput = num_requests / total_time
    token_throughput = total_tokens / total_time

    print(f"Completed in {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} req/s ({token_throughput:.2f} tokens/s)")
    print(
        f"Latency: Avg={avg_latency:.3f}s | p50={p50:.3f}s | p95={p95:.3f}s | p99={p99:.3f}s"
    )


async def main():
    print("Checking if server is running...")
    try:
        async with (
            aiohttp.ClientSession() as session,
            session.get("http://127.0.0.1:8000/health") as resp,
        ):
            if resp.status != 200:
                print("Server health check failed.")
                return
    except Exception:  # noqa: BLE001
        print(
            "Could not connect to server. Ensure it's running with 'uvicorn server.app:app'"
        )
        return

    print("Server is up. Starting benchmarks.")
    await run_batch(concurrency=1, num_requests=3)
    await run_batch(concurrency=2, num_requests=6)
    await run_batch(concurrency=4, num_requests=12)


if __name__ == "__main__":
    asyncio.run(main())
