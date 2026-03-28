"""
pyodbc connection factory for MS SQL Server.

Uses per-thread persistent connections — no reconnect overhead on every query.
Each worker thread (pipeline, attendance, API) gets its own long-lived connection.
"""
import threading
import pyodbc
from contextlib import contextmanager

from app.config import settings
from app.utils.logger import logger

# Per-thread persistent connections
_local = threading.local()


def is_db_enabled() -> bool:
    return bool(settings.database_url)


def _get_thread_connection() -> pyodbc.Connection:
    """
    Returns a persistent pyodbc connection for the current thread.
    Reconnects automatically if the connection was dropped.
    """
    conn = getattr(_local, "conn", None)
    if conn is not None:
        try:
            conn.cursor().execute("SELECT 1")
            return conn
        except Exception:
            logger.warning("DB: thread connection lost — reconnecting")
            try:
                conn.close()
            except Exception:
                pass
            _local.conn = None

    _local.conn = pyodbc.connect(settings.database_url, timeout=10)
    return _local.conn


@contextmanager
def get_db():
    """
    Context manager that yields the current thread's persistent connection.
    Commits on success, rolls back on error. Does NOT close the connection.
    """
    conn = _get_thread_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


def get_raw_connection() -> pyodbc.Connection:
    """Open a fresh one-off connection (used only during startup)."""
    return pyodbc.connect(settings.database_url, timeout=10)


def test_connection() -> bool:
    try:
        conn = get_raw_connection()
        conn.cursor().execute("SELECT 1")
        conn.close()
        return True
    except Exception as e:
        logger.error(f"DB: connection test failed — {e}")
        return False
