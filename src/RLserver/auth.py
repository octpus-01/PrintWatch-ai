import hmac
import os

from flask import request

from .config import Config

AUTH_HEADER = "Authorization"
AUTH_PREFIX = "Bearer "


def is_authorized() -> bool:
    header = request.headers.get(AUTH_HEADER, "")
    if not header.startswith(AUTH_PREFIX):
        return False
    token = header[len(AUTH_PREFIX):].strip()
    if not token:
        return False
    return hmac.compare_digest(token, Config.API_TOKEN)


def get_bearer_token() -> str:
    header = request.headers.get(AUTH_HEADER, "")
    if header.startswith(AUTH_PREFIX):
        return header[len(AUTH_PREFIX):].strip()
    return ""


def token_is_configured() -> bool:
    return bool(Config.API_TOKEN) and Config.API_TOKEN not in ("", "change-me-token")


def ensure_directories() -> None:
    for path in (Config.DATA_DIR, Config.INBOX_DIR, Config.MODELS_DIR):
        os.makedirs(path, exist_ok=True)