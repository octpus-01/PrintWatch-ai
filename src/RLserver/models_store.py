import os
import re

from . import database


def parse_version(version):
    match = re.search(r"(\d+)", str(version or ""))
    if not match:
        return 0
    return int(match.group(1))


def next_version():
    latest = database.get_latest_model()
    current = parse_version(latest["version"]) if latest else 0
    return f"v{current + 1}"


def publish_model(version, file_path, note=None) -> None:
    file_path = os.path.abspath(file_path)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")
    database.publish_model(version, file_path, note=note)


def get_latest_model():
    return database.get_latest_model()


def get_all_models():
    return database.get_all_models()