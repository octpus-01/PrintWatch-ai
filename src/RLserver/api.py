import base64
import logging
import os
from datetime import datetime

from flask import Flask, jsonify, request, send_file

from . import auth, database, models_store, retrain
from .config import Config

logger = logging.getLogger(__name__)


def _parse_iso_timestamp(value):
    text = str(value).strip()
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return text


def _build_iso_timestamp():
    return datetime.utcnow().isoformat()


def _save_image(device_id, timestamp, payload):
    folder = os.path.join(Config.INBOX_DIR, device_id)
    os.makedirs(folder, exist_ok=True)
    name = f"{timestamp.replace(':', '-').replace('+', '_')}.jpg"
    path = os.path.join(folder, name)
    counter = 1
    while os.path.exists(path):
        stem, ext = os.path.splitext(name)
        path = os.path.join(folder, f"{stem}_{counter}{ext}")
        counter += 1
    with open(path, "wb") as fh:
        fh.write(payload)
    return path


def _validate_records(device_id, records):
    cleaned = []
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(f"records[{idx}] is not an object")
        timestamp = _parse_iso_timestamp(rec.get("timestamp"))
        if timestamp is None:
            raise ValueError(f"records[{idx}].timestamp is not a valid ISO-8601 datetime")
        label = rec.get("label")
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"records[{idx}].label is required")
        try:
            confidence = float(rec["confidence"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"records[{idx}].confidence is required and must be a number")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"records[{idx}].confidence must be in [0, 1]")
        raw = rec.get("image_base64")
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"records[{idx}].image_base64 is required")
        try:
            image_bytes = base64.b64decode(raw, validate=True)
        except Exception:
            raise ValueError(f"records[{idx}].image_base64 is not valid base64")
        if not image_bytes:
            raise ValueError(f"records[{idx}].image_base64 decodes to empty data")
        if len(image_bytes) > Config.MAX_IMAGE_BYTES:
            raise ValueError(f"records[{idx}].image exceeds {Config.MAX_IMAGE_BYTES} bytes")
        model_path = rec.get("model_path")
        if model_path is not None and not isinstance(model_path, str):
            raise ValueError(f"records[{idx}].model_path must be a string")
        cleaned.append(
            {
                "device_id": device_id,
                "timestamp": timestamp,
                "label": label.strip(),
                "confidence": confidence,
                "image_bytes": image_bytes,
                "model_path": model_path,
            }
        )
    return cleaned


def _store_records(cleaned):
    rows = []
    for rec in cleaned:
        image_path = _save_image(rec["device_id"], rec["timestamp"], rec["image_bytes"])
        rows.append(
            {
                "device_id": rec["device_id"],
                "timestamp": rec["timestamp"],
                "label": rec["label"],
                "confidence": rec["confidence"],
                "image_path": image_path,
                "model_path": rec["model_path"],
            }
        )
    database.insert_records(rows)
    return len(rows)


def create_app():
    app = Flask(__name__)
    app.json.ensure_ascii = False

    @app.before_request
    def check_auth():
        if not auth.is_authorized():
            return jsonify({"code": 401, "message": "unauthorized"}), 401

    @app.errorhandler(404)
    def not_found(_err):
        return jsonify({"code": 404, "message": "not found"}), 404

    @app.errorhandler(500)
    def server_error(_err):
        return jsonify({"code": 500, "message": "internal server error"}), 500

    @app.get("/healthz")
    def healthz():
        return jsonify({"code": 200, "message": "ok"}), 200

    @app.post("/api/v1/sync_data")
    def sync_data():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"code": 400, "message": "request body must be a JSON object"}), 400

        device_id = body.get("device_id")
        if not isinstance(device_id, str) or not device_id.strip():
            return jsonify({"code": 400, "message": "device_id is required"}), 400

        records = body.get("records")
        if not isinstance(records, list) or not records:
            return jsonify({"code": 400, "message": "records must be a non-empty list"}), 400

        if len(records) > Config.MAX_BATCH:
            return jsonify(
                {
                    "code": 400,
                    "message": f"batch too large: {len(records)} > {Config.MAX_BATCH}",
                }
            ), 400

        try:
            cleaned = _validate_records(device_id.strip(), records)
        except ValueError as exc:
            return jsonify({"code": 400, "message": str(exc)}), 400

        try:
            synced_count = _store_records(cleaned)
        except Exception:
            logger.exception("Failed to store synced records")
            return jsonify({"code": 500, "message": "failed to store records"}), 500

        if Config.AUTO_RETRAIN_ON_SYNC and synced_count > 0:
            retrain.trigger_retrain()

        return jsonify({"code": 200, "message": "success", "synced_count": synced_count}), 200

    @app.get("/api/v1/get_latest_model")
    def get_latest_model():
        latest = models_store.get_latest_model()
        if latest is None:
            return "", 304

        current_version = request.args.get("current_version")
        if current_version is not None and current_version == latest["version"]:
            return "", 304

        if not os.path.isfile(latest["file_path"]):
            logger.error("Latest model file missing: %s", latest["file_path"])
            return jsonify({"code": 500, "message": "model file missing"}), 500

        from flask import make_response

        response = make_response(
            send_file(
                latest["file_path"],
                mimetype="application/octet-stream",
                as_attachment=False,
                conditional=False,
                download_name=f"server_{latest['version']}.onnx",
            )
        )
        response.headers["X-Model-Version"] = latest["version"]
        return response

    return app