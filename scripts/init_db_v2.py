"""Database schema migration script (v1 -> v2).

Idempotent: can be run multiple times safely.
Adds new columns to models/records tables and creates telemetry table.
"""

import sqlite3
import sys
from pathlib import Path


def column_exists(cursor, table, column):
    """Check if a column exists in a table."""
    cursor.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cursor.fetchall()]
    return column in columns


def table_exists(cursor, table):
    """Check if a table exists."""
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    )
    return cursor.fetchone() is not None


def migrate(db_path):
    """Run database migration."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        print("Run the server first to initialize the database.")
        sys.exit(1)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print(f"Migrating database: {db_path}")
    
    models_cols = [
        ("config_json", "TEXT"),
        ("metrics_json", "TEXT"),
        ("model_family", "TEXT"),
        ("status", "TEXT DEFAULT 'active'"),
    ]
    
    for col, col_type in models_cols:
        if not column_exists(cursor, "models", col):
            sql = f"ALTER TABLE models ADD COLUMN {col} {col_type}"
            print(f"  Adding column: {sql}")
            cursor.execute(sql)
        else:
            print(f"  Column already exists: models.{col}")
    
    records_cols = [
        ("ground_truth", "TEXT"),
        ("annotation_status", "TEXT DEFAULT 'pending'"),
        ("sample_tier", "TEXT"),
        ("latency_ms", "REAL"),
    ]
    
    for col, col_type in records_cols:
        if not column_exists(cursor, "records", col):
            sql = f"ALTER TABLE records ADD COLUMN {col} {col_type}"
            print(f"  Adding column: {sql}")
            cursor.execute(sql)
        else:
            print(f"  Column already exists: records.{col}")
    
    if not table_exists(cursor, "telemetry"):
        sql = """
        CREATE TABLE telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            latency_p50_ms REAL,
            latency_p95_ms REAL,
            inference_count_24h INTEGER,
            load_result TEXT,
            error_detail TEXT,
            memory_peak_mb REAL,
            created_at TEXT NOT NULL
        )
        """
        print(f"  Creating table: telemetry")
        cursor.execute(sql)
    else:
        print(f"  Table already exists: telemetry")
    
    conn.commit()
    conn.close()
    
    print("Migration completed successfully.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/server.db"
    migrate(db_path)
