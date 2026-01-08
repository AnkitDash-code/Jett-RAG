"""
API v1 Router - Aggregates all domain routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import auth, users, documents, chat, admin, health, jobs, admin_jobs, memory, sessions, search

api_router = APIRouter()

# Mount domain routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat & RAG"])
api_router.include_router(search.router, prefix="/search", tags=["Search & Suggestions"])
api_router.include_router(memory.router, prefix="/memory", tags=["Memory & Recall"])
api_router.include_router(sessions.router, prefix="/sessions", tags=["Sessions & Devices"])
api_router.include_router(admin.router, prefix="/admin", tags=["Admin & Monitoring"])
api_router.include_router(health.router, prefix="/health", tags=["Health & Monitoring"])
api_router.include_router(jobs.router, tags=["Jobs & Background Tasks"])
api_router.include_router(admin_jobs.router, prefix="/admin", tags=["Admin Job Management"])
