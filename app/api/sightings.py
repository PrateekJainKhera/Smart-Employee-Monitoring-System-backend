"""
Sightings REST endpoints.

GET /sightings              → all sightings today (all employees, all cameras)
GET /sightings/{employee_id} → sightings for one employee today (per camera breakdown)
"""
from typing import Optional
from fastapi import APIRouter, Query

router = APIRouter(prefix="/sightings", tags=["Sightings"])


@router.get("")
def list_sightings(date: Optional[str] = Query(None, description="Date YYYY-MM-DD, default today")):
    """All sighting counts for today grouped by employee + camera."""
    from app.sightings.sighting_store import sighting_store
    from app.store import state
    records = sighting_store.get_all_today() if date is None else []

    # Enrich with employee name and camera label
    result = []
    for r in records:
        emp = state.get_employee(r["employee_id"])
        cam = state.get_camera(r["camera_id"])
        result.append({
            "employee_id":   r["employee_id"],
            "employee_name": emp["name"] if emp else str(r["employee_id"]),
            "camera_id":     r["camera_id"],
            "camera_label":  cam["location_label"] if cam else str(r["camera_id"]),
            "date":          r["date"],
            "count":         r["count"],
        })

    # Sort by employee name then camera
    result.sort(key=lambda x: (x["employee_name"], x["camera_label"]))
    return result


@router.get("/summary")
def sightings_summary(date: Optional[str] = Query(None, description="Date YYYY-MM-DD, default today")):
    """
    Per-employee summary: total sightings + breakdown by camera label.
    Returns [{employee_id, name, total, cameras: [{label, count}]}]
    """
    from app.sightings.sighting_store import sighting_store
    from app.store import state

    records = sighting_store.get_all_today()
    # Group by employee
    by_emp: dict[int, dict] = {}
    for r in records:
        eid = r["employee_id"]
        if eid not in by_emp:
            emp = state.get_employee(eid)
            by_emp[eid] = {
                "employee_id": eid,
                "name": emp["name"] if emp else str(eid),
                "total": 0,
                "cameras": [],
            }
        cam = state.get_camera(r["camera_id"])
        label = cam["location_label"] if cam else str(r["camera_id"])
        by_emp[eid]["total"] += r["count"]
        by_emp[eid]["cameras"].append({"label": label, "camera_id": r["camera_id"], "count": r["count"]})

    return sorted(by_emp.values(), key=lambda x: x["name"])
