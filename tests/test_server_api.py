"""
API endpoint integration tests for LIEFS Universal Model & Computer Benchmark Server.
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "device" in data


def test_hardware_endpoint(client):
    res = client.get("/v1/hardware")
    assert res.status_code == 200
    data = res.json()
    assert "cpu_model" in data
    assert "ram_total_gb" in data
    assert "cuda_available" in data


def test_model_presets_endpoint(client):
    res = client.get("/v1/models/presets")
    assert res.status_code == 200
    data = res.json()
    assert "presets" in data
    assert len(data["presets"]) >= 5


def test_current_model_endpoint(client):
    res = client.get("/v1/models/current")
    assert res.status_code == 200
    data = res.json()
    assert "model_name" in data
    assert "parameter_count_m" in data


def test_engines_endpoint(client):
    res = client.get("/v1/engines")
    assert res.status_code == 200
    data = res.json()
    assert "engines" in data
    assert isinstance(data["engines"], list)
    assert len(data["engines"]) > 0
    engine_ids = [e["id"] for e in data["engines"]]
    assert "kv_cache" in engine_ids


def test_system_endpoint(client):
    res = client.get("/v1/system")
    assert res.status_code == 200
    data = res.json()
    assert "ram_total_gb" in data


def test_list_models(client):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


def test_load_and_unload_model(client):
    # Test unload first
    res = client.post("/v1/models/unload")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "message": "Model unloaded and VRAM cleared"}

    # Test load
    # Use a small model for test if we want, or just the default.
    # The default one in server is DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
    # Actually loading a real model might take some time or VRAM.
    # Since we are testing endpoints, we can just trigger it and see it works.
    payload = {"model_name": "Qwen/Qwen2.5-0.5B-Instruct"}
    res = client.post("/v1/models/load", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "model_name" in data


def test_benchmarks_preset(client):
    res = client.get("/v1/benchmarks/preset")
    assert res.status_code == 200
    data = res.json()
    assert "model" in data
    assert "device" in data
    assert "benchmarks" in data
    assert isinstance(data["benchmarks"], list)


def test_completions_stream(client):
    # Ensure a model is loaded first
    client.post("/v1/models/load", json={"model_name": "Qwen/Qwen2.5-0.5B-Instruct"})

    payload = {"prompt": "Say hi", "max_tokens": 10, "stream": True, "engine": "naive"}

    with client.stream("POST", "/v1/completions", json=payload) as response:
        assert response.status_code == 200

        chunks_received = 0
        for chunk in response.iter_lines():
            if chunk:
                chunks_received += 1
                assert chunk.startswith("data: ")

        assert chunks_received > 0


def test_run_computer_benchmark(client):
    # Ensure a model is loaded first
    client.post("/v1/models/load", json={"model_name": "Qwen/Qwen2.5-0.5B-Instruct"})

    payload = {
        "prompt": "Test prompt",
        "max_tokens": 5,
        "engines": ["naive", "kv_cache", "quantized"],
        "include_batch_scaling": True,
    }
    res = client.post("/v1/benchmarks/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "timestamp" in data
    assert "score" in data
    assert "results" in data
    assert len(data["results"]) > 0

def test_metrics_endpoint(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "recent_metrics" in res.json()

def test_unload_model_handles_none(client):
    res = client.post("/v1/models/unload")
    assert res.status_code == 200
    res2 = client.post("/v1/models/unload")
    assert res2.status_code == 200

def test_completions_no_model(client):
    client.post("/v1/models/unload")
    payload = {"prompt": "Hello", "max_tokens": 5}
    res = client.post("/v1/completions", json=payload)
    assert res.status_code == 503

def test_benchmark_no_model(client):
    client.post("/v1/models/unload")
    payload = {"prompt": "Test", "max_tokens": 5, "engines": ["naive"]}
    res = client.post("/v1/benchmarks/run", json=payload)
    assert res.status_code == 503

def test_completions_non_stream(client):
    client.post("/v1/models/load", json={"model_name": "Qwen/Qwen2.5-0.5B-Instruct"})
    payload = {"prompt": "Hello", "max_tokens": 5, "stream": False, "engine": "naive"}
    res = client.post("/v1/completions", json=payload)
    assert res.status_code == 200
    assert "choices" in res.json()

def test_benchmark_engines_options(client):
    client.post("/v1/models/load", json={"model_name": "Qwen/Qwen2.5-0.5B-Instruct"})
    payload = {
        "prompt": "Test prompt",
        "max_tokens": 5,
        "engines": ["kv_cache", "paged"],
        "include_batch_scaling": False,
    }
    res = client.post("/v1/benchmarks/run", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "results" in data

def test_load_model_failure(client):
    payload = {"model_name": "invalid_model_name_does_not_exist"}
    res = client.post("/v1/models/load", json=payload)
    assert res.status_code == 400
