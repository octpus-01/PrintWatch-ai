"""Contract tests for API v1 endpoints.

These tests verify that the existing API behavior remains stable.
They serve as a regression test suite when making changes to the server.
"""

import base64
import io
import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.RLserver.api import create_app
from src.RLserver import auth, database
from src.RLserver.config import Config


def _dummy_jpeg_b64():
    img = Image.new("RGB", (32, 32), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def client(tmp_path):
    original_db = Config.DB_PATH
    original_inbox = Config.INBOX_DIR
    original_token = Config.API_TOKEN
    original_auto_retrain = Config.AUTO_RETRAIN_ON_SYNC

    Config.DB_PATH = str(tmp_path / "test.db")
    Config.INBOX_DIR = str(tmp_path / "inbox")
    Config.API_TOKEN = "test-token"
    Config.AUTO_RETRAIN_ON_SYNC = False

    os.makedirs(Config.INBOX_DIR, exist_ok=True)
    database.init_db()

    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client

    Config.DB_PATH = original_db
    Config.INBOX_DIR = original_inbox
    Config.API_TOKEN = original_token
    Config.AUTO_RETRAIN_ON_SYNC = original_auto_retrain


# ── Healthz ──

def test_healthz(client):
    resp = client.get("/healthz", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    assert resp.get_json()["code"] == 200


# ── Auth ──

def test_no_token_returns_401(client):
    resp = client.get("/api/v1/get_latest_model")
    assert resp.status_code == 401
    assert resp.get_json()["code"] == 401


def test_wrong_token_returns_401(client):
    resp = client.get(
        "/api/v1/get_latest_model",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


# ── sync_data validation ──

def test_sync_data_missing_device_id(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={"records": [{"timestamp": "2026-01-01T00:00:00Z", "label": "ok", "confidence": 0.9, "image_base64": _dummy_jpeg_b64()}]},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 400


def test_sync_data_empty_records(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={"device_id": "d1", "records": []},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 400


def test_sync_data_invalid_record(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={"device_id": "d1", "records": [{}]},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 400


def test_sync_data_bad_base64(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={
            "device_id": "d1",
            "records": [{"timestamp": "2026-01-01T00:00:00Z", "label": "ok", "confidence": 0.9, "image_base64": "not-valid-base64!!!"}],
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 400


def test_sync_data_confidence_out_of_range(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={
            "device_id": "d1",
            "records": [{"timestamp": "2026-01-01T00:00:00Z", "label": "ok", "confidence": 1.5, "image_base64": _dummy_jpeg_b64()}],
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 400


# ── sync_data happy path ──

def test_sync_data_valid_record(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={
            "device_id": "pi-test-001",
            "records": [{"timestamp": "2026-01-01T00:00:00Z", "label": "good_print", "confidence": 0.95, "image_base64": _dummy_jpeg_b64()}],
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["synced_count"] == 1


def test_sync_data_with_model_path(client):
    resp = client.post(
        "/api/v1/sync_data",
        json={
            "device_id": "pi-test-001",
            "records": [{"timestamp": "2026-01-01T00:00:00Z", "label": "spaghetti", "confidence": 0.85, "image_base64": _dummy_jpeg_b64(), "model_path": "models/server_v2.onnx"}],
        },
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["synced_count"] == 1


def test_sync_data_multi_record(client):
    records = [{"timestamp": "2026-01-01T00:00:00Z", "label": "good_print", "confidence": 0.9 + i * 0.01, "image_base64": _dummy_jpeg_b64()} for i in range(3)]
    resp = client.post(
        "/api/v1/sync_data",
        json={"device_id": "pi-test-001", "records": records},
        headers={"Authorization": "Bearer test-token"},
    )
    assert resp.status_code == 200
    assert resp.get_json()["synced_count"] == 3


# ── get_latest_model ──

def test_get_latest_model_no_models(client):
    resp = client.get("/api/v1/get_latest_model", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 304


def test_get_latest_model_after_publish(client, tmp_path):
    model_file = tmp_path / "test_model.onnx"
    model_file.write_text("dummy onnx content")
    from src.RLserver import models_store
    models_store.publish_model("v1", str(model_file), note="test")

    resp = client.get("/api/v1/get_latest_model", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Model-Version") == "v1"
    assert resp.data == b"dummy onnx content"


def test_get_latest_model_same_version_304(client, tmp_path):
    model_file = tmp_path / "test_model.onnx"
    model_file.write_text("dummy")
    from src.RLserver import models_store
    models_store.publish_model("v1", str(model_file))

    resp = client.get("/api/v1/get_latest_model?current_version=v1", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 304


def test_get_latest_model_outdated_version_200(client, tmp_path):
    model_file = tmp_path / "test_model.onnx"
    model_file.write_text("dummy")
    from src.RLserver import models_store
    models_store.publish_model("v2", str(model_file))

    resp = client.get("/api/v1/get_latest_model?current_version=v1", headers={"Authorization": "Bearer test-token"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Model-Version") == "v2"