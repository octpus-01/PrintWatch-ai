import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _as_bool(value, default):
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class Config:
    API_TOKEN = os.environ.get("RLSERVER_API_TOKEN", "change-me-token")

    HOST = os.environ.get("RLSERVER_HOST", "0.0.0.0")
    PORT = _as_int(os.environ.get("RLSERVER_PORT"), 8000)

    DATA_DIR = os.environ.get("RLSERVER_DATA_DIR", os.path.join(BASE_DIR, "data"))
    DB_PATH = os.environ.get("RLSERVER_DB_PATH", os.path.join(DATA_DIR, "server.db"))
    INBOX_DIR = os.environ.get("RLSERVER_INBOX_DIR", os.path.join(DATA_DIR, "inbox"))

    MODELS_DIR = os.environ.get("RLSERVER_MODELS_DIR", os.path.join(BASE_DIR, "models"))

    MAX_BATCH = _as_int(os.environ.get("RLSERVER_MAX_BATCH"), 50)
    MAX_IMAGE_BYTES = _as_int(os.environ.get("RLSERVER_MAX_IMAGE_BYTES"), 10 * 1024 * 1024)

    AUTO_RETRAIN_ON_SYNC = _as_bool(os.environ.get("RLSERVER_AUTO_RETRAIN_ON_SYNC"), True)

    LOG_LEVEL = os.environ.get("RLSERVER_LOG_LEVEL", "INFO")