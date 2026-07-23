"""
AlertService — background thread that fires WebSocket alerts.

Runs every 60 seconds and checks:
  1. Missing alert  — employee checked in today but not seen for > MISSING_ALERT_MIN minutes
  2. After-hours    — employee detected after OFFICE_END_HOUR (if after_hours_alerts=True)

Alerts are rate-limited: same alert won't fire more than once per 10 minutes per employee.
"""
import threading
import time
from datetime import datetime

from app.utils.logger import logger
from app.config import settings


# {(alert_type, employee_id): last_fired_monotonic}
_alert_cooldown: dict[tuple, float] = {}
_COOLDOWN_SEC = 600  # 10 minutes between same alert for same employee


def _should_fire(alert_type: str, employee_id: int) -> bool:
    key = (alert_type, employee_id)
    now = time.monotonic()
    last = _alert_cooldown.get(key, 0)
    if now - last >= _COOLDOWN_SEC:
        _alert_cooldown[key] = now
        return True
    return False


def run_alert_checks() -> None:
    """Run one round of alert checks. Called by the background loop."""
    try:
        from app.services.attendance_service import list_attendance
        from app.sightings.sighting_store import sighting_store
        from app.store import state
        from app.api.ws import emit_alert

        now_dt = datetime.now()
        now_utc = datetime.utcnow()

        today_logs = list_attendance()  # all attendance logs for today

        for log in today_logs:
            emp_id = log.get("employee_id") if isinstance(log, dict) else log.employee_id
            emp_name = log.get("employee_name") if isinstance(log, dict) else getattr(log, "employee_name", str(emp_id))
            check_in = log.get("check_in") if isinstance(log, dict) else log.check_in
            check_out = log.get("check_out") if isinstance(log, dict) else log.check_out
            on_break = log.get("on_break") if isinstance(log, dict) else log.on_break

            # Skip employees who have already checked out
            if check_out:
                continue

            # Skip employees who haven't checked in
            if not check_in:
                continue

            last_seen_dt = sighting_store.get_last_seen(emp_id)

            # ── Missing alert ──────────────────────────────────────────────
            if last_seen_dt is not None:
                minutes_ago = (now_utc - last_seen_dt).total_seconds() / 60
                if minutes_ago >= settings.missing_alert_min:
                    if _should_fire("missing", emp_id):
                        emit_alert(
                            "missing",
                            emp_id,
                            emp_name,
                            minutes_since_seen=round(minutes_ago),
                        )
                        logger.info(
                            f"Alert: {emp_name} missing — last seen {round(minutes_ago)}m ago"
                        )

            # ── After-hours alert ──────────────────────────────────────────
            if settings.after_hours_alerts and last_seen_dt is not None:
                if now_dt.hour >= settings.office_end_hour:
                    # Only alert if seen within the last 5 minutes (recently active after hours)
                    minutes_ago = (now_utc - last_seen_dt).total_seconds() / 60
                    if minutes_ago <= 5:
                        if _should_fire("after_hours", emp_id):
                            emit_alert(
                                "after_hours",
                                emp_id,
                                emp_name,
                                office_end_hour=settings.office_end_hour,
                            )
                            logger.info(
                                f"Alert: {emp_name} detected after hours ({now_dt.strftime('%H:%M')})"
                            )

    except Exception as e:
        logger.warning(f"AlertService check error: {e}")


def start_alert_worker() -> None:
    """Start the background alert checker thread."""
    def _loop():
        logger.info("AlertService worker started (checks every 60s)")
        while True:
            time.sleep(60)
            run_alert_checks()

    threading.Thread(target=_loop, daemon=True, name="alert-worker").start()
