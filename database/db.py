"""
database/db.py
--------------
SQLite logging module for the Smart Pothole Detection System.

Creates a local SQLite database (database/pothole_log.db) and provides
helper functions to log every detection run and query the history.

Tables
------
  detections      - one row per pipeline run
  pothole_details - one row per individual pothole in a run

Fixes applied:
  - init_db() is now guarded by a module-level flag (_INITIALIZED) so the
    CREATE TABLE IF NOT EXISTS statements execute only once per process,
    not on every log_detection() call.
  - log_detection() no longer calls init_db() redundantly.
"""

import os
import sqlite3
from datetime import datetime

# ── Database path ──────────────────────────────────────────────
DB_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(DB_DIR, "pothole_log.db")

# Guard: only run CREATE TABLE once per Python process
_INITIALIZED = False


def get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    return conn


def init_db():
    """Create tables if they don't exist. Safe to call multiple times (idempotent)."""
    global _INITIALIZED
    if _INITIALIZED:
        return  # FIX: skip redundant CREATE TABLE round-trips

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            image_path       TEXT,
            location         TEXT,
            pothole_count    INTEGER NOT NULL DEFAULT 0,
            highest_severity TEXT,
            avg_confidence   REAL,
            alert_sent       INTEGER NOT NULL DEFAULT 0,
            output_image     TEXT,
            notes            TEXT
        )
    """)

    # Individual potholes per run (optional detail table)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pothole_details (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            detection_id INTEGER NOT NULL REFERENCES detections(id),
            x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
            width   INTEGER,
            height  INTEGER,
            area    INTEGER,
            confidence REAL,
            severity   TEXT
        )
    """)

    conn.commit()
    conn.close()
    _INITIALIZED = True


def log_detection(
    image_path: str,
    location: str,
    pothole_count: int,
    highest_severity: str,
    avg_confidence: float,
    alert_sent: bool,
    output_image: str,
    detections: list = None,
    severities: list = None,
    timestamp: str = None,
) -> int:
    """
    Insert a detection run into the database.

    Returns
    -------
    int - row ID of the inserted detection record
    """
    # Ensure tables exist (cheap no-op after first call)
    init_db()

    ts = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections
            (timestamp, image_path, location, pothole_count,
             highest_severity, avg_confidence, alert_sent, output_image)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (ts, image_path, location, pothole_count,
          str(highest_severity),
          round(float(avg_confidence), 4) if avg_confidence else 0.0,
          1 if alert_sent else 0,
          output_image))

    row_id = cursor.lastrowid

    # Log individual potholes if provided
    if detections and severities:
        for det, sev in zip(detections, severities):
            cursor.execute("""
                INSERT INTO pothole_details
                    (detection_id, x1, y1, x2, y2,
                     width, height, area, confidence, severity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (row_id,
                  det.get("x1"), det.get("y1"),
                  det.get("x2"), det.get("y2"),
                  det.get("width"), det.get("height"), det.get("area"),
                  det.get("confidence"), str(sev)))

    conn.commit()
    conn.close()
    return row_id


def get_recent_detections(limit: int = 20) -> list:
    """Return the most recent *limit* detection records as a list of dicts."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM detections
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_stats() -> dict:
    """Return aggregate statistics across all stored detections."""
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            COUNT(*)                                         AS total_runs,
            COALESCE(SUM(pothole_count), 0)                 AS total_potholes,
            SUM(CASE WHEN alert_sent = 1 THEN 1 ELSE 0 END) AS total_alerts,
            ROUND(AVG(avg_confidence), 4)                   AS mean_confidence,
            SUM(CASE WHEN highest_severity='High'   THEN 1 ELSE 0 END) AS high_count,
            SUM(CASE WHEN highest_severity='Medium' THEN 1 ELSE 0 END) AS medium_count,
            SUM(CASE WHEN highest_severity='Low'    THEN 1 ELSE 0 END) AS low_count
        FROM detections
    """)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else {}


# ── Quick self-test ────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    test_id = log_detection(
        image_path="test_image.jpg",
        location="Test Road",
        pothole_count=2,
        highest_severity="High",
        avg_confidence=0.91,
        alert_sent=True,
        output_image="results/test_out.jpg",
    )
    print(f"[DB] Inserted test record - ID: {test_id}")
    print("[DB] Recent records:", get_recent_detections(3))
    print("[DB] Stats:", get_stats())
