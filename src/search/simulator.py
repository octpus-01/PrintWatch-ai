"""EdgeSimulator: simulates edge device behaviour for contract testing.

Simulates the edge-side behaviour defined in API_INTEGRATION_GUIDE:
  - sync_data: confidence-based stratified sampling upload
  - get_latest_model: download, ONNX validation, dual-buffer rollback
  - telemetry: periodic latency/inference stats reporting
  - baseline model protection: /models/baseline/ghostnet_v1.onnx is never overwritten
"""

import base64
import io
import json
import logging
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Confidence thresholds matching the edge sampling strategy
CONFIDENCE_TIERS = {
    "uncertain": (0.3, 0.7),       # report all
    "defect_high": (0.9, 1.0),     # report all if defect label
    "normal_high": (0.9, 1.0),     # report 5% sample if normal label
}

DEFECT_LABELS = {"spaghetti", "stringing", "warping", "under_extrusion", "over_extrusion"}
NORMAL_LABELS = {"good_print"}


def _sample_tier(confidence: float, label: str) -> Optional[str]:
    """Determine the sampling tier for a prediction record.
    
    Returns None if the record should be skipped.
    """
    if confidence < 0.5:
        return None
    if confidence < 0.7:
        return "uncertain"
    if label in DEFECT_LABELS and confidence >= 0.9:
        return "defect_high"
    if label in NORMAL_LABELS and confidence >= 0.9:
        return "normal_high"
    return None


def _should_report(tier: str) -> bool:
    """Decide whether to include a record based on its tier."""
    if tier == "uncertain":
        return True
    if tier == "defect_high":
        return True
    if tier == "normal_high":
        return random.random() < 0.05
    return False


def _dummy_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (224, 224), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


try:
    from PIL import Image
except ImportError:
    Image = None


class EdgeSimulator:
    """Simulates the edge device (Raspberry Pi) behaviour.

    Can be used in tests to verify server-side API changes without
    a real edge device present.
    """

    def __init__(self, server_url: str, token: str, device_id: str = "sim-pi-001"):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self.device_id = device_id
        self.current_version: Optional[str] = None
        self.models_dir = Path(f"models/edge_{device_id}")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._baseline_path = self.models_dir / "ghostnet_v1.onnx"
        self._baseline_path.write_text("dummy baseline onnx")
        self._running = False

    # ── sync_data integration ──

    def generate_sync_payload(self, predictions: list[dict]) -> dict:
        """Build a sync_data request body from raw predictions.
        
        Applies the confidence-based stratified sampling strategy.
        """
        records = []
        for pred in predictions:
            tier = _sample_tier(pred["confidence"], pred["label"])
            if tier is None:
                continue
            if not _should_report(tier):
                continue
            records.append({
                "timestamp": pred.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                "label": pred["label"],
                "confidence": round(pred["confidence"], 4),
                "image_base64": base64.b64encode(pred.get("image_bytes", _dummy_jpeg_bytes())).decode(),
                "model_path": pred.get("model_path", "models/server_v1.onnx"),
            })
        return {"device_id": self.device_id, "records": records}

    def sync_data(self, predictions: list[dict]) -> dict:
        """Send sync_data to the server and return the response."""
        payload = self.generate_sync_payload(predictions)
        req = urllib.request.Request(
            f"{self.server_url}/api/v1/sync_data",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"code": e.code, "message": e.read().decode()}

    # ── get_latest_model integration ──

    def get_latest_model(self) -> Optional[str]:
        """Fetch the latest model from the server.
        
        Returns the model version string on success, None on failure.
        Implements dual-buffer rollback: if the downloaded model is invalid,
        the previous version is kept.
        """
        url = f"{self.server_url}/api/v1/get_latest_model"
        if self.current_version:
            url += f"?current_version={self.current_version}"

        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {self.token}"},
            method="GET",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                if resp.status == 304:
                    logger.info("No new model available")
                    return self.current_version
                new_version = resp.headers.get("X-Model-Version")
                if not new_version:
                    logger.warning("Response missing X-Model-Version header")
                    return self.current_version
                model_bytes = resp.read()
                if not self._validate_onnx(model_bytes):
                    logger.error("Downloaded model failed ONNX validation, keeping previous version")
                    return self.current_version
                new_path = self.models_dir / f"server_{new_version}.onnx"
                old_path = self.models_dir / f"server_{self.current_version}.onnx" if self.current_version else None
                new_path.write_bytes(model_bytes)
                if old_path and old_path.exists() and old_path != self._baseline_path:
                    old_path.unlink(missing_ok=True)
                self.current_version = new_version
                logger.info("Updated to model version %s", new_version)
                return new_version
        except urllib.error.HTTPError as e:
            logger.error("Failed to fetch model: %s", e)
            return self.current_version

    def _validate_onnx(self, data: bytes) -> bool:
        """Basic ONNX validation: check for the protobuf magic bytes."""
        return len(data) >= 4 and data[:4] == b"\x08\x00\x00\x00"

    # ── telemetry ──

    def report_telemetry(self, latency_p50_ms: float, latency_p95_ms: float,
                         inference_count_24h: int, load_result: str = "success",
                         error_detail: str = "", memory_peak_mb: float = 0.0) -> dict:
        """Report telemetry data to the server."""
        payload = {
            "device_id": self.device_id,
            "model_version": self.current_version or "unknown",
            "latency_p50_ms": latency_p50_ms,
            "latency_p95_ms": latency_p95_ms,
            "inference_count_24h": inference_count_24h,
            "load_result": load_result,
            "error_detail": error_detail,
            "memory_peak_mb": memory_peak_mb,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        req = urllib.request.Request(
            f"{self.server_url}/api/v1/telemetry",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"code": e.code, "message": e.read().decode()}

    # ── baseline protection ──

    @property
    def baseline_path(self) -> Path:
        return self._baseline_path

    def is_baseline(self, version: str) -> bool:
        """Check if a version refers to the baseline ghostnet model."""
        return version == "baseline" or "ghostnet_v1" in version

    # ── synthetic prediction generators for testing ──

    @staticmethod
    def make_prediction(label: str, confidence: float, **kwargs) -> dict:
        return {"label": label, "confidence": confidence, **kwargs}

    @staticmethod
    def make_batch(n: int, label_pool: Optional[list[str]] = None) -> list[dict]:
        if label_pool is None:
            label_pool = list(DEFECT_LABELS | NORMAL_LABELS)
        return [
            {"label": random.choice(label_pool), "confidence": round(random.uniform(0.5, 1.0), 4)}
            for _ in range(n)
        ]