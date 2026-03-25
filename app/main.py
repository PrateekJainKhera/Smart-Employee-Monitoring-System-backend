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
from app.api import employees, cameras

START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    setup_logger(settings.log_level)
    logger.info("Starting Smart Employee Monitoring System")
    logger.info(f"Storage mode: in-memory (Phase 1-4)")

    # Start camera threads + processing pipelines for pre-registered cameras
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
    return {
        "status": "ok",
        "storage": "in-memory",
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "version": "1.0.0",
        "active_cameras": camera_manager.active_cameras(),
    }


app.include_router(employees.router, prefix="/api/v1")
app.include_router(cameras.router, prefix="/api/v1")
