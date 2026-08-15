"""
backend/app/db/__init__.py
Re-export the most commonly used symbols so callers can do:
    from backend.app.db import Base, engine, get_db, create_tables
"""
from backend.app.db.session import Base, engine, get_db, SessionLocal, create_tables  # noqa: F401
