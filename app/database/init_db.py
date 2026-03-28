"""
Creates all MS SQL Server tables if they do not already exist.
Uses T-SQL IF OBJECT_ID checks so this is safe to run on every startup.
"""
import pyodbc
from app.utils.logger import logger

# ── Table definitions (T-SQL) ────────────────────────────────────────────────

_CREATE_EMPLOYEES = """
IF OBJECT_ID('dbo.employees', 'U') IS NULL
BEGIN
    CREATE TABLE employees (
        id              INT PRIMARY KEY IDENTITY(1,1),
        name            NVARCHAR(200)   NOT NULL,
        department      NVARCHAR(100)   NOT NULL DEFAULT '',
        designation     NVARCHAR(100)   NOT NULL DEFAULT '',
        email           NVARCHAR(200)   NOT NULL DEFAULT '',
        face_registered BIT             NOT NULL DEFAULT 0,
        created_at      DATETIME2       NOT NULL DEFAULT GETUTCDATE()
    )
END
"""

_CREATE_CAMERAS = """
IF OBJECT_ID('dbo.cameras', 'U') IS NULL
BEGIN
    CREATE TABLE cameras (
        id              INT PRIMARY KEY IDENTITY(1,1),
        name            NVARCHAR(200)   NOT NULL,
        location_label  NVARCHAR(100)   NOT NULL,
        rtsp_url        NVARCHAR(500)   NOT NULL,
        is_active       BIT             NOT NULL DEFAULT 1
    )
END
"""

_CREATE_FACE_EMBEDDINGS = """
IF OBJECT_ID('dbo.face_embeddings', 'U') IS NULL
BEGIN
    CREATE TABLE face_embeddings (
        id              INT PRIMARY KEY IDENTITY(1,1),
        employee_id     INT             NOT NULL,
        embedding       VARBINARY(MAX)  NOT NULL,
        created_at      DATETIME2       NOT NULL DEFAULT GETUTCDATE(),
        CONSTRAINT fk_fe_employee FOREIGN KEY (employee_id)
            REFERENCES employees(id) ON DELETE CASCADE
    )
END
"""

_CREATE_ATTENDANCE_LOGS = """
IF OBJECT_ID('dbo.attendance_logs', 'U') IS NULL
BEGIN
    CREATE TABLE attendance_logs (
        id              INT PRIMARY KEY IDENTITY(1,1),
        employee_id     INT             NOT NULL,
        camera_id       INT,
        check_in        DATETIME2       NOT NULL,
        check_out       DATETIME2,
        total_hours     FLOAT,
        date            DATE            NOT NULL,
        created_at      DATETIME2       NOT NULL DEFAULT GETUTCDATE(),
        CONSTRAINT fk_al_employee FOREIGN KEY (employee_id) REFERENCES employees(id),
        CONSTRAINT fk_al_camera   FOREIGN KEY (camera_id)   REFERENCES cameras(id)
    )
END
"""

_CREATE_BREAK_LOGS = """
IF OBJECT_ID('dbo.break_logs', 'U') IS NULL
BEGIN
    CREATE TABLE break_logs (
        id                  INT PRIMARY KEY IDENTITY(1,1),
        employee_id         INT         NOT NULL,
        attendance_log_id   INT         NOT NULL,
        break_start         DATETIME2   NOT NULL,
        break_end           DATETIME2,
        duration_minutes    FLOAT,
        break_type          NVARCHAR(20),
        CONSTRAINT fk_bl_employee   FOREIGN KEY (employee_id)
            REFERENCES employees(id),
        CONSTRAINT fk_bl_attendance FOREIGN KEY (attendance_log_id)
            REFERENCES attendance_logs(id)
    )
END
"""

_CREATE_MOVEMENT_LOGS = """
IF OBJECT_ID('dbo.movement_logs', 'U') IS NULL
BEGIN
    CREATE TABLE movement_logs (
        id              INT PRIMARY KEY IDENTITY(1,1),
        employee_id     INT,
        camera_id       INT,
        track_id        INT,
        detected_at     DATETIME2   NOT NULL DEFAULT GETUTCDATE(),
        CONSTRAINT fk_ml_employee FOREIGN KEY (employee_id) REFERENCES employees(id),
        CONSTRAINT fk_ml_camera   FOREIGN KEY (camera_id)   REFERENCES cameras(id)
    )
END
"""

_TABLES = [
    ("employees",       _CREATE_EMPLOYEES),
    ("cameras",         _CREATE_CAMERAS),
    ("face_embeddings", _CREATE_FACE_EMBEDDINGS),
    ("attendance_logs", _CREATE_ATTENDANCE_LOGS),
    ("break_logs",      _CREATE_BREAK_LOGS),
    ("movement_logs",   _CREATE_MOVEMENT_LOGS),
]


def create_tables(conn: pyodbc.Connection) -> None:
    """Run all CREATE TABLE statements. Safe to call on every startup."""
    cursor = conn.cursor()
    for table_name, sql in _TABLES:
        cursor.execute(sql)
        logger.info(f"DB: table '{table_name}' ready")
    conn.commit()
    cursor.close()
    logger.info("DB: all tables initialized")


def load_all_employees(conn: pyodbc.Connection) -> list[dict]:
    """Return all rows from employees table as dicts."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, department, designation, email, face_registered, created_at "
        "FROM employees ORDER BY id"
    )
    rows = cursor.fetchall()
    cursor.close()
    return [
        {
            "id": r[0],
            "name": r[1],
            "department": r[2] or "",
            "designation": r[3] or "",
            "email": r[4] or "",
            "face_registered": bool(r[5]),
            "created_at": r[6].isoformat() if r[6] else "",
        }
        for r in rows
    ]


def load_all_cameras(conn: pyodbc.Connection) -> list[dict]:
    """Return all rows from cameras table as dicts."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, location_label, rtsp_url, is_active FROM cameras ORDER BY id"
    )
    rows = cursor.fetchall()
    cursor.close()
    return [
        {
            "id": r[0],
            "name": r[1],
            "location_label": r[2],
            "rtsp_url": r[3],
            "is_active": bool(r[4]),
        }
        for r in rows
    ]
