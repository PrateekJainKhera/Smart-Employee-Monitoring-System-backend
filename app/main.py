import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.utils.logger import setup_logger, logger
from app.camera.camera_manager import camera_manager
from app.pipeline.processing_pipeline import pipeline_manager
from app.store import state
from app.api import employees, cameras, attendance, reports, snapshots, sightings, settings as settings_api
from app.api import ws as ws_module
import app.recognition.face_recognizer as _fr_module
import app.services.employee_service as _es_module

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    setup_logger(settings.log_level)
    logger.info("Starting Smart Employee Monitoring System")

    # ── Initialize Face Recognition engine ───────────────────
    logger.info("Loading InsightFace engine (may download model on first run)...")
    from app.recognition.insightface_engine import InsightFaceEngine
    from app.recognition.deepface_engine import DeepFaceEngine
    from app.recognition.embedding_store import EmbeddingStore

    insightface_engine = InsightFaceEngine(model_name="buffalo_l", det_size=(1280, 1280))
    deepface_engine = DeepFaceEngine()
    embedding_store = EmbeddingStore(settings.embeddings_path)

    from app.recognition.face_recognizer import FaceRecognizer
    from app.services.employee_service import EmployeeService

    _fr_module.face_recognizer = FaceRecognizer(
        insightface_engine, deepface_engine, embedding_store
    )
    _es_module.employee_service = EmployeeService(insightface_engine, embedding_store)
    logger.info(
        f"Face recognition ready — {embedding_store.count()} embedding(s) loaded from store"
    )

    # ── Initialize SQL Server (Phase 5) ──────────────────────
    from app.database.connection import is_db_enabled, test_connection, get_raw_connection
    if is_db_enabled():
        if test_connection():
            logger.info("DB: connected to MS SQL Server")
            from app.database.init_db import create_tables, load_all_employees, load_all_cameras
            conn = get_raw_connection()
            try:
                create_tables(conn)
                # Load existing employees + cameras from DB into AppState cache
                db_employees = load_all_employees(conn)
                db_cameras = load_all_cameras(conn)
            finally:
                conn.close()

            for emp in db_employees:
                state.add_employee(
                    name=emp["name"],
                    department=emp["department"],
                    designation=emp["designation"],
                    email=emp["email"],
                    employee_id=emp["id"],
                    face_registered=emp["face_registered"],
                    created_at=emp["created_at"],
                )
            for cam in db_cameras:
                state.add_camera(
                    name=cam["name"],
                    location_label=cam["location_label"],
                    rtsp_url=cam["rtsp_url"],
                    camera_id=cam["id"],
                    is_active=cam["is_active"],
                )
            logger.info(
                f"DB: loaded {len(db_employees)} employee(s) and "
                f"{len(db_cameras)} camera(s) from SQL Server"
            )
            logger.info("Storage mode: SQL Server (Phase 5)")
        else:
            logger.warning("DB: DATABASE_URL set but connection failed — running in-memory only")
    else:
        logger.info("Storage mode: in-memory (set DATABASE_URL in .env to enable SQL Server)")

    # ── Store event loop for WebSocket thread-safe broadcasts ─
    import asyncio
    ws_module.set_event_loop(asyncio.get_event_loop())
    logger.info("WS: event loop registered for thread-safe broadcast")

    # ── Auto check-out background scheduler ──────────────────
    if is_db_enabled():
        import threading as _threading

        def _auto_checkout_loop():
            import time as _time
            while True:
                _time.sleep(30)
                try:
                    from app.services.attendance_service import auto_checkout_stale
                    n = auto_checkout_stale()
                    if n:
                        logger.info(f"AutoCheckout scheduler: {n} employee(s) auto-checked-out")
                except Exception as _e:
                    logger.warning(f"AutoCheckout scheduler error: {_e}")

        _threading.Thread(target=_auto_checkout_loop, daemon=True, name="auto-checkout").start()
        logger.info("AutoCheckout scheduler started (runs every 60s, threshold=20min)")

    # ── Start camera threads + processing pipelines ──────────
    existing_cameras = state.list_cameras()
    if existing_cameras:
        logger.info(f"Starting {len(existing_cameras)} camera thread(s) and pipeline(s)...")
        camera_manager.start_all(existing_cameras)
        pipeline_manager.start_all(existing_cameras)
    else:
        logger.info("No cameras registered yet — add via POST /api/v1/cameras")

    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("Shutting down pipelines and camera threads...")
    pipeline_manager.stop_all()
    camera_manager.stop_all()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Smart Employee Monitoring System",
    description="AI-powered attendance and employee tracking via CCTV",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.get("/health", tags=["System"])
def health():
    from app.database.connection import is_db_enabled
    return {
        "status": "ok",
        "storage": "sql_server" if is_db_enabled() else "in-memory",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "version": "1.0.0",
        "active_cameras": camera_manager.active_cameras(),
        "db_enabled": is_db_enabled(),
    }


app.include_router(employees.router,  prefix="/api/v1")
app.include_router(cameras.router,    prefix="/api/v1")
app.include_router(attendance.router, prefix="/api/v1")
app.include_router(reports.router,    prefix="/api/v1")
app.include_router(snapshots.router,  prefix="/api/v1")
app.include_router(sightings.router,     prefix="/api/v1")
app.include_router(settings_api.router, prefix="/api/v1")
app.include_router(ws_module.router)   # WS lives at /ws/live (no /api/v1 prefix)
