"""
backend/app/api/__init__.py
Aggregated API router — mounts all sub-routers under /api.
"""
from fastapi import APIRouter

from backend.app.api import colorize, health, history

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(colorize.router)
api_router.include_router(history.router)
