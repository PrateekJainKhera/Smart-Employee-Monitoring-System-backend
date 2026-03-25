"""
Shared FastAPI dependencies.

Phase 1-4: only `get_state` is used.
Phase 5+:  `get_db` is added here (SQLAlchemy session).
"""
from app.store import state, AppState


def get_state() -> AppState:
    return state


# Phase 5+ — uncomment when database/connection.py is ready:
#
# from app.database.connection import SessionLocal
# from sqlalchemy.orm import Session
#
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
