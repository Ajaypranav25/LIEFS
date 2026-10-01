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
