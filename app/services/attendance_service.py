"""
Attendance service — check-in, check-out, and break detection.

All queries are raw T-SQL executed via pyodbc.
Only runs when DATABASE_URL is configured.

Event routing:
  location_label == 'entry'  → check-in or return-from-break
  location_label == 'exit'   → start break or final check-out
  anything else              → movement log only
"""
import threading
from datetime import datetime, date
from typing import Optional

from app.config import settings
from app.utils.logger import logger

# ── Debounce ─────────────────────────────────────────────────────────────────
# Prevents duplicate events within the same 30-second window.
_debounce: dict[tuple, datetime] = {}
_debounce_lock = threading.Lock()
_DEBOUNCE_SECONDS = 30


def _is_debounced(employee_id: int, camera_id: int) -> bool:
    key = (employee_id, camera_id)
    now = datetime.utcnow()
    with _debounce_lock:
        last = _debounce.get(key)
        if last and (now - last).total_seconds() < _DEBOUNCE_SECONDS:
            return True
        _debounce[key] = now
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def handle_event(employee_id: int, camera_id: int, location_label: str) -> None:
    """
    Called by the pipeline every time an identified employee is seen on a camera.
    Routes to the appropriate attendance logic based on the camera's location label.
    No-op if DATABASE_URL is not configured.
    """
    from app.database.connection import is_db_enabled
    if not is_db_enabled():
        return

    if _is_debounced(employee_id, camera_id):
        return

    try:
        from app.database.connection import get_db
        with get_db() as conn:
            cursor = conn.cursor()
            now = datetime.utcnow()
            today = now.date()

            if location_label == "entry":
                _handle_entry(cursor, employee_id, camera_id, now, today)
            elif location_label == "exit":
                _handle_exit(cursor, employee_id, camera_id, now, today)
            else:
                _log_movement(cursor, employee_id, camera_id, now)
    except Exception as e:
        logger.error(f"AttendanceService: error for employee {employee_id}: {e}")


def get_today_status(employee_id: int) -> dict:
    """
    Returns current attendance state for an employee today.
    Possible statuses: absent | inside | on_break | checked_out
    """
    from app.database.connection import get_db
    today = date.today()
    with get_db() as conn:
        cursor = conn.cursor()
        log = _get_open_attendance(cursor, employee_id, today)

        if log is None:
            # Check for a completed (checked-out) record today
            cursor.execute(
                "SELECT id, check_in, check_out, total_hours "
                "FROM attendance_logs "
                "WHERE employee_id=? AND date=? AND check_out IS NOT NULL "
                "ORDER BY check_in DESC",
                employee_id, today,
            )
            row = cursor.fetchone()
            if row:
                return {
                    "status": "checked_out",
                    "check_in": row[1],
                    "check_out": row[2],
                    "total_hours": row[3],
                }
            return {
                "status": "absent",
                "check_in": None,
                "check_out": None,
                "total_hours": None,
            }

        brk = _get_open_break(cursor, log[0])
        status = "on_break" if brk else "inside"
        elapsed = (datetime.utcnow() - log[1]).total_seconds() / 3600
        return {
            "status": status,
            "check_in": log[1],
            "check_out": None,
            "total_hours_so_far": round(elapsed, 2),
        }


def list_attendance(target_date: Optional[date] = None) -> list[dict]:
    """
    Return all attendance records for a given date (defaults to today).
    Each row includes employee name and break count.
    """
    from app.database.connection import get_db
    if target_date is None:
        target_date = date.today()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT al.id, al.employee_id, e.name, al.check_in, al.check_out, "
            "       al.total_hours, al.date, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id) AS break_count, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id AND bl.break_end IS NULL) AS open_breaks "
            "FROM attendance_logs al "
            "JOIN employees e ON e.id = al.employee_id "
            "WHERE al.date = ? "
            "ORDER BY al.check_in",
            target_date,
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "employee_id": r[1],
                "employee_name": r[2],
                "check_in": r[3].isoformat() + "Z" if r[3] else None,
                "check_out": r[4].isoformat() + "Z" if r[4] else None,
                "total_hours": r[5],
                "date": str(r[6]),
                "break_count": r[7],
                "on_break": r[4] is None and r[8] > 0,
            }
            for r in rows
        ]


def list_breaks(attendance_log_id: int) -> list[dict]:
    """Return all break records for a given attendance log."""
    from app.database.connection import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, employee_id, attendance_log_id, break_start, break_end, "
            "       duration_minutes, break_type "
            "FROM break_logs WHERE attendance_log_id=? ORDER BY break_start",
            attendance_log_id,
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "employee_id": r[1],
                "attendance_log_id": r[2],
                "break_start": r[3].isoformat() + "Z" if r[3] else None,
                "break_end": r[4].isoformat() + "Z" if r[4] else None,
                "duration_minutes": r[5],
                "break_type": r[6],
            }
            for r in rows
        ]


def get_employee_attendance_history(employee_id: int, limit: int = 30) -> list[dict]:
    """Return the last N attendance records for an employee across all dates."""
    from app.database.connection import get_db
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT TOP (?) al.id, al.check_in, al.check_out, al.total_hours, al.date, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id) "
            "FROM attendance_logs al "
            "WHERE al.employee_id=? "
            "ORDER BY al.date DESC, al.check_in DESC",
            limit, employee_id,
        )
        rows = cursor.fetchall()
        return [
            {
                "id": r[0],
                "check_in": r[1].isoformat() + "Z" if r[1] else None,
                "check_out": r[2].isoformat() + "Z" if r[2] else None,
                "total_hours": r[3],
                "date": str(r[4]),
                "break_count": r[5],
            }
            for r in rows
        ]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_open_attendance(cursor, employee_id: int, today: date):
    """Return (id, check_in) for the first open attendance log today, or None."""
    cursor.execute(
        "SELECT id, check_in FROM attendance_logs "
        "WHERE employee_id=? AND date=? AND check_out IS NULL",
        employee_id, today,
    )
    return cursor.fetchone()


def _get_open_break(cursor, attendance_log_id: int):
    """Return (id, break_start) for the open break under this log, or None."""
    cursor.execute(
        "SELECT id, break_start FROM break_logs "
        "WHERE attendance_log_id=? AND break_end IS NULL",
        attendance_log_id,
    )
    return cursor.fetchone()


def _get_employee_name(employee_id: int) -> str:
    from app.store import state
    emp = state.get_employee(employee_id)
    return emp["name"] if emp else str(employee_id)


def _handle_entry(cursor, employee_id: int, camera_id: int, now: datetime, today: date) -> None:
    log = _get_open_attendance(cursor, employee_id, today)
    name = _get_employee_name(employee_id)

    if log is None:
        # First check-in of the day
        cursor.execute(
            "INSERT INTO attendance_logs (employee_id, camera_id, check_in, date) "
            "VALUES (?, ?, ?, ?)",
            employee_id, camera_id, now, today,
        )
        logger.info(f"Attendance: employee {employee_id} checked IN at {now.strftime('%H:%M:%S')}")
        try:
            from app.store import state as _state
            cam = _state.get_camera(camera_id)
            label = cam["location_label"] if cam else "entry"
            from app.api.ws import emit_checkin
            emit_checkin(employee_id, name, camera_id, label)
        except Exception:
            pass
    else:
        # Returning from break — close any open break
        brk = _get_open_break(cursor, log[0])
        if brk:
            duration = (now - brk[1]).total_seconds() / 60
            break_type = _classify_break(duration)
            cursor.execute(
                "UPDATE break_logs "
                "SET break_end=?, duration_minutes=?, break_type=? "
                "WHERE id=?",
                now, round(duration, 2), break_type, brk[0],
            )
            logger.info(
                f"Attendance: employee {employee_id} returned — "
                f"{break_type} break ({duration:.1f} min)"
            )
            try:
                from app.store import state as _state2
                cam2 = _state2.get_camera(camera_id)
                label2 = cam2["location_label"] if cam2 else "entry"
                from app.api.ws import emit_break_end
                emit_break_end(employee_id, name, duration, break_type, camera_id, label2)
            except Exception:
                pass


def _handle_exit(cursor, employee_id: int, camera_id: int, now: datetime, today: date) -> None:
    log = _get_open_attendance(cursor, employee_id, today)

    if log is None:
        return

    name = _get_employee_name(employee_id)
    brk  = _get_open_break(cursor, log[0])

    if brk is None:
        # Not on break yet — start break timer
        cursor.execute(
            "INSERT INTO break_logs (employee_id, attendance_log_id, break_start) "
            "VALUES (?, ?, ?)",
            employee_id, log[0], now,
        )
        logger.info(f"Attendance: employee {employee_id} left — break started at {now.strftime('%H:%M:%S')}")
        try:
            from app.api.ws import emit_break_start
            emit_break_start(employee_id, name, camera_id)
        except Exception:
            pass
    else:
        # Already on break — check if duration exceeds medium threshold (final checkout)
        duration = (now - brk[1]).total_seconds() / 60
        if duration >= settings.break_medium_min:
            cursor.execute(
                "UPDATE break_logs "
                "SET break_end=?, duration_minutes=?, break_type='long' "
                "WHERE id=?",
                now, round(duration, 2), brk[0],
            )
            total_hours = (now - log[1]).total_seconds() / 3600
            cursor.execute(
                "UPDATE attendance_logs "
                "SET check_out=?, total_hours=? "
                "WHERE id=?",
                now, round(total_hours, 4), log[0],
            )
            logger.info(
                f"Attendance: employee {employee_id} checked OUT "
                f"(total {total_hours:.2f}h)"
            )
            try:
                from app.store import state as _state3
                cam3 = _state3.get_camera(camera_id)
                label3 = cam3["location_label"] if cam3 else "exit"
                from app.api.ws import emit_checkout
                emit_checkout(employee_id, name, total_hours, auto=False, camera_id=camera_id, camera_label=label3)
            except Exception:
                pass


def _log_movement(cursor, employee_id: int, camera_id: int, now: datetime) -> None:
    cursor.execute(
        "INSERT INTO movement_logs (employee_id, camera_id, detected_at) VALUES (?, ?, ?)",
        employee_id, camera_id, now,
    )


def _classify_break(duration_minutes: float) -> str:
    if duration_minutes < settings.break_short_min:
        return "short"
    elif duration_minutes < settings.break_medium_min:
        return "medium"
    return "long"


def get_missing_employees(absent_threshold_min: int = 0) -> list[dict]:
    """
    Returns employees who are absent or overdue.

    absent_threshold_min=0  → all employees with no check-in today (absent)
    absent_threshold_min=N  → employees on break for more than N minutes (overdue)
                              PLUS employees with no check-in today

    Each result includes employee_id, name, status, and minutes_absent.
    """
    from app.database.connection import get_db
    today = date.today()
    now = datetime.utcnow()

    with get_db() as conn:
        cursor = conn.cursor()

        # All employees
        cursor.execute("SELECT id, name, department, designation FROM employees")
        all_employees = {r[0]: {"id": r[0], "name": r[1], "department": r[2], "designation": r[3]}
                         for r in cursor.fetchall()}

        # Employees with check-in today (open or closed)
        cursor.execute(
            "SELECT employee_id, check_out FROM attendance_logs WHERE date=?", today
        )
        checked_in_ids = {r[0]: r[1] for r in cursor.fetchall()}  # id → check_out (None if open)

        missing = []
        for emp_id, emp in all_employees.items():
            if emp_id not in checked_in_ids:
                # Never seen today
                missing.append({**emp, "status": "absent", "minutes_absent": None})
            elif checked_in_ids[emp_id] is not None:
                # Already checked out — not missing
                pass
            else:
                # Has open attendance — check if on a long break
                if absent_threshold_min > 0:
                    cursor.execute(
                        "SELECT break_start FROM break_logs "
                        "WHERE employee_id=? AND break_end IS NULL "
                        "ORDER BY break_start DESC",
                        emp_id,
                    )
                    brk = cursor.fetchone()
                    if brk:
                        minutes_away = (now - brk[0]).total_seconds() / 60
                        if minutes_away >= absent_threshold_min:
                            missing.append({
                                **emp,
                                "status": "overdue_break",
                                "minutes_absent": round(minutes_away, 1),
                            })

        return missing


def auto_checkout_stale() -> int:
    """
    Background job — auto check-out employees based on these rules:

    Rule 1 (end of day): After office_end_hour (default 7 PM local time),
      if employee has been outside for >= office_end_break_min (default 15 min)
      → checkout. Checkout time = break_start (last time they were seen leaving).

    Rule 2 (safety net): At any time, if employee has been outside for
      >= max_break_min (default 120 min / 2 hours) → checkout.

    Checkout time is always break_start (last time seen), NOT now.
    This means total_hours is accurate — it reflects when they actually left.
    """
    from app.database.connection import is_db_enabled, get_db
    if not is_db_enabled():
        return 0

    today = date.today()
    now_utc = datetime.utcnow()
    now_local = datetime.now()          # local time — used only for hour check
    after_end_of_day = now_local.hour >= settings.office_end_hour
    count = 0

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT bl.id, bl.employee_id, bl.attendance_log_id, bl.break_start "
                "FROM break_logs bl "
                "JOIN attendance_logs al ON al.id = bl.attendance_log_id "
                "WHERE bl.break_end IS NULL AND al.date=? AND al.check_out IS NULL",
                today,
            )
            stale_breaks = cursor.fetchall()

            for brk_id, emp_id, log_id, break_start in stale_breaks:
                duration_min = (now_utc - break_start).total_seconds() / 60

                # Decide whether to check out
                if after_end_of_day and duration_min >= settings.office_end_break_min:
                    reason = f"after {settings.office_end_hour}:00, out {duration_min:.0f} min"
                elif duration_min >= settings.max_break_min:
                    reason = f"out {duration_min:.0f} min (max {settings.max_break_min} min exceeded)"
                else:
                    continue  # still within allowed break time

                # Checkout time = break_start (last time seen at exit camera)
                checkout_time = break_start

                cursor.execute(
                    "UPDATE break_logs SET break_end=?, duration_minutes=?, break_type='long' "
                    "WHERE id=?",
                    checkout_time, round(duration_min, 2), brk_id,
                )
                cursor.execute(
                    "SELECT check_in FROM attendance_logs WHERE id=?", log_id
                )
                row = cursor.fetchone()
                if row:
                    total_hours = (checkout_time - row[0]).total_seconds() / 3600
                    cursor.execute(
                        "UPDATE attendance_logs SET check_out=?, total_hours=? WHERE id=?",
                        checkout_time, round(total_hours, 4), log_id,
                    )
                    logger.info(
                        f"AutoCheckout: employee {emp_id} — {reason} — "
                        f"check_out={checkout_time.strftime('%H:%M')} total={total_hours:.2f}h"
                    )
                    try:
                        name = _get_employee_name(emp_id)
                        from app.api.ws import emit_checkout
                        emit_checkout(emp_id, name, total_hours, auto=True)
                    except Exception:
                        pass
                    count += 1
    except Exception as e:
        logger.error(f"auto_checkout_stale error: {e}")

    return count
