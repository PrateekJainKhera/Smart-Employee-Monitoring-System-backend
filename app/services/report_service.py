"""
Report service — daily summaries, monthly reports, movement timelines, CSV export.

All queries are raw T-SQL via pyodbc.
"""
import csv
import io
from datetime import date, datetime
from typing import Optional

from app.utils.logger import logger


def daily_summary(target_date: Optional[date] = None) -> dict:
    """
    Returns a summary of attendance for a given date.
    Includes: total employees, present, absent, on_break, checked_out, avg_hours.
    """
    from app.database.connection import get_db
    if target_date is None:
        target_date = date.today()

    with get_db() as conn:
        cursor = conn.cursor()

        # Total registered employees
        cursor.execute("SELECT COUNT(*) FROM employees")
        total = cursor.fetchone()[0]

        # Attendance records for the day
        cursor.execute(
            "SELECT al.employee_id, al.check_in, al.check_out, al.total_hours, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id AND bl.break_end IS NULL) AS open_breaks "
            "FROM attendance_logs al "
            "WHERE al.date = ?",
            target_date,
        )
        rows = cursor.fetchall()

        present_ids   = {r[0] for r in rows}
        checked_out   = [r for r in rows if r[2] is not None]
        on_break      = [r for r in rows if r[2] is None and r[4] > 0]
        inside        = [r for r in rows if r[2] is None and r[4] == 0]
        absent        = total - len(present_ids)

        hours_list = [r[3] for r in checked_out if r[3] is not None]
        avg_hours  = round(sum(hours_list) / len(hours_list), 2) if hours_list else 0.0

        # Early arrivals (first 5 check-ins sorted by time)
        cursor.execute(
            "SELECT TOP 5 al.employee_id, e.name, al.check_in "
            "FROM attendance_logs al JOIN employees e ON e.id = al.employee_id "
            "WHERE al.date = ? ORDER BY al.check_in",
            target_date,
        )
        early = [{"employee_id": r[0], "name": r[1], "check_in": r[2].isoformat() + "Z" if r[2] else None}
                 for r in cursor.fetchall()]

        return {
            "date": str(target_date),
            "total_employees": total,
            "present": len(present_ids),
            "absent": absent,
            "inside": len(inside),
            "on_break": len(on_break),
            "checked_out": len(checked_out),
            "avg_hours_worked": avg_hours,
            "early_arrivals": early,
        }


def monthly_summary(employee_id: int, year: int, month: int) -> dict:
    """
    Per-employee monthly report: days present, total hours, avg hours/day, break breakdown.
    """
    from app.database.connection import get_db

    with get_db() as conn:
        cursor = conn.cursor()

        # Employee name
        cursor.execute("SELECT name, department, designation FROM employees WHERE id=?", employee_id)
        emp = cursor.fetchone()
        if emp is None:
            return {"error": f"Employee {employee_id} not found"}

        # All attendance records for the month
        cursor.execute(
            "SELECT al.id, al.date, al.check_in, al.check_out, al.total_hours "
            "FROM attendance_logs al "
            "WHERE al.employee_id=? AND YEAR(al.date)=? AND MONTH(al.date)=? "
            "ORDER BY al.date",
            employee_id, year, month,
        )
        logs = cursor.fetchall()

        days_present = len(logs)
        total_hours  = sum(r[4] for r in logs if r[4] is not None)

        # Break stats
        log_ids = [r[0] for r in logs]
        break_counts = {"short": 0, "medium": 0, "long": 0}
        total_break_min = 0.0

        if log_ids:
            placeholders = ",".join(["?"] * len(log_ids))
            cursor.execute(
                f"SELECT break_type, duration_minutes FROM break_logs "
                f"WHERE attendance_log_id IN ({placeholders}) AND break_end IS NOT NULL",
                *log_ids,
            )
            for btype, dur in cursor.fetchall():
                if btype in break_counts:
                    break_counts[btype] += 1
                total_break_min += (dur or 0)

        # Days in month for absence calculation
        import calendar
        days_in_month = calendar.monthrange(year, month)[1]

        return {
            "employee_id": employee_id,
            "name": emp[0],
            "department": emp[1],
            "designation": emp[2],
            "year": year,
            "month": month,
            "days_in_month": days_in_month,
            "days_present": days_present,
            "days_absent": days_in_month - days_present,
            "total_hours": round(total_hours, 2),
            "avg_hours_per_day": round(total_hours / days_present, 2) if days_present else 0.0,
            "total_break_minutes": round(total_break_min, 1),
            "breaks": break_counts,
            "daily": [
                {
                    "date": str(r[1]),
                    "check_in": r[2].isoformat() + "Z" if r[2] else None,
                    "check_out": r[3].isoformat() + "Z" if r[3] else None,
                    "hours": r[4],
                }
                for r in logs
            ],
        }


def movement_timeline(employee_id: int, target_date: Optional[date] = None) -> list[dict]:
    """
    Full movement timeline for an employee on a given date.
    Merges attendance events (check-in, break-start, break-end, check-out)
    and movement_log detections into a single chronological list.
    """
    from app.database.connection import get_db
    if target_date is None:
        target_date = date.today()

    with get_db() as conn:
        cursor = conn.cursor()
        events: list[dict] = []

        # Attendance events
        cursor.execute(
            "SELECT al.id, al.check_in, al.check_out, c.name, c.location_label "
            "FROM attendance_logs al "
            "JOIN cameras c ON c.id = al.camera_id "
            "WHERE al.employee_id=? AND al.date=?",
            employee_id, target_date,
        )
        for log_id, check_in, check_out, cam_name, cam_label in cursor.fetchall():
            events.append({
                "type": "check_in",
                "timestamp": check_in.isoformat() + "Z" if check_in else None,
                "camera": cam_name,
                "location": cam_label,
            })
            if check_out:
                events.append({
                    "type": "check_out",
                    "timestamp": check_out.isoformat() + "Z",
                    "camera": cam_name,
                    "location": cam_label,
                })

            # Breaks for this log
            cursor.execute(
                "SELECT break_start, break_end, duration_minutes, break_type "
                "FROM break_logs WHERE attendance_log_id=? ORDER BY break_start",
                log_id,
            )
            for bs, be, dur, btype in cursor.fetchall():
                events.append({
                    "type": "break_start",
                    "timestamp": bs.isoformat() + "Z" if bs else None,
                    "break_type": btype,
                })
                if be:
                    events.append({
                        "type": "break_end",
                        "timestamp": be.isoformat() + "Z",
                        "duration_min": dur,
                        "break_type": btype,
                    })

        # Movement log detections (internal cameras)
        cursor.execute(
            "SELECT ml.detected_at, c.name, c.location_label "
            "FROM movement_logs ml "
            "JOIN cameras c ON c.id = ml.camera_id "
            "WHERE ml.employee_id=? AND CAST(ml.detected_at AS DATE)=? "
            "ORDER BY ml.detected_at",
            employee_id, target_date,
        )
        for detected_at, cam_name, cam_label in cursor.fetchall():
            events.append({
                "type": "detected",
                "timestamp": detected_at.isoformat() + "Z" if detected_at else None,
                "camera": cam_name,
                "location": cam_label,
            })

        # Sort all events by timestamp
        events.sort(key=lambda e: e["timestamp"] or "")
        return events


def export_csv(target_date: Optional[date] = None) -> bytes:
    """
    Export daily attendance as CSV bytes.
    Columns: employee_id, name, department, check_in, check_out, total_hours, break_count
    """
    from app.database.connection import get_db
    if target_date is None:
        target_date = date.today()

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT al.employee_id, e.name, e.department, e.designation, "
            "       al.check_in, al.check_out, al.total_hours, "
            "       (SELECT COUNT(*) FROM break_logs bl WHERE bl.attendance_log_id = al.id) "
            "FROM attendance_logs al "
            "JOIN employees e ON e.id = al.employee_id "
            "WHERE al.date=? ORDER BY al.check_in",
            target_date,
        )
        rows = cursor.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Employee ID", "Name", "Department", "Designation",
        "Check In", "Check Out", "Total Hours", "Break Count"
    ])
    for r in rows:
        writer.writerow([
            r[0], r[1], r[2], r[3],
            r[4].strftime("%Y-%m-%d %H:%M:%S") if r[4] else "",
            r[5].strftime("%Y-%m-%d %H:%M:%S") if r[5] else "",
            r[6] if r[6] is not None else "",
            r[7],
        ])

    return output.getvalue().encode("utf-8")
