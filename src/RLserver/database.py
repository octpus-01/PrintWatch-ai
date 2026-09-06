import sqlite3
import threading
from contextlib import contextmanager

from .config import Config

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    image_path TEXT NOT NULL,
    model_path TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    used_for_retrain INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT NOT NULL UNIQUE,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    note TEXT
);

CREATE TABLE IF NOT EXISTS devices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL UNIQUE,
    first_seen TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);
"""

_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _transaction():
    with _lock:
        conn = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def init_db() -> None:
    with _transaction() as conn:
        conn.executescript(SCHEMA)


def insert_records(records):
    with _transaction() as conn:
        for rec in records:
            conn.execute(
                """
                INSERT INTO records (device_id, timestamp, label, confidence,
                                     image_path, model_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    rec["device_id"],
                    rec["timestamp"],
                    rec["label"],
                    rec["confidence"],
                    rec["image_path"],
                    rec.get("model_path"),
                ),
            )
            conn.execute(
                """
                INSERT INTO devices (device_id, last_seen)
                VALUES (?, datetime('now'))
                ON CONFLICT(device_id) DO UPDATE SET last_seen = excluded.last_seen
                """,
                (rec["device_id"],),
            )


def get_unused_records(limit=1000):
    with _transaction() as conn:
        rows = conn.execute(
            """
            SELECT * FROM records
            WHERE used_for_retrain = 0
            ORDER BY id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def mark_records_used(record_ids) -> None:
    if not record_ids:
        return
    with _transaction() as conn:
        conn.executemany(
            "UPDATE records SET used_for_retrain = 1 WHERE id = ?",
            [(rid,) for rid in record_ids],
        )


def count_records() -> int:
    with _transaction() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c, "
            "SUM(CASE WHEN used_for_retrain = 0 THEN 1 ELSE 0 END) AS unused "
            "FROM records"
        ).fetchone()
    return dict(row)


def publish_model(version, file_path, note=None) -> None:
    with _transaction() as conn:
        conn.execute(
            """
            INSERT INTO models (version, file_path, note)
            VALUES (?, ?, ?)
            ON CONFLICT(version) DO UPDATE SET
                file_path = excluded.file_path,
                note = excluded.note,
                created_at = datetime('now')
            """,
            (version, file_path, note),
        )


def get_latest_model():
    with _transaction() as conn:
        row = conn.execute(
            "SELECT * FROM models ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def get_all_models():
    with _transaction() as conn:
        rows = conn.execute(
            "SELECT * FROM models ORDER BY id DESC"
        ).fetchall()
    return [dict(row) for row in rows]