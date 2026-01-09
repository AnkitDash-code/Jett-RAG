"""
RAG Backend - Main FastAPI Application
Unified API server for auth, documents, retrieval, chat, and monitoring.
"""
import os

# Force HuggingFace offline mode BEFORE importing any transformers/sentence-transformers
# This prevents network requests when models are already cached
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import logging
from contextlib import asynccontextmanager
import subprocess
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db, async_session_maker
from app.api.v1 import api_router
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.error_handler import global_exception_handler, AppException
from app.middleware.rate_limit import setup_rate_limiting

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Start KoboldCpp embedding server at startup (if not already running)
try:
    embedding_launcher = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'init_embedding.py'))
    if os.path.exists(embedding_launcher):
        logger.info(f"Starting KoboldCpp embedding server...")
        # Run synchronously to ensure server is ready before app starts
        result = subprocess.run(
            [sys.executable, embedding_launcher],
            cwd=os.path.dirname(embedding_launcher),
            capture_output=True,
            text=True,
            timeout=60  # Wait up to 60 seconds for server to start
        )
        if result.returncode == 0:
            logger.info("KoboldCpp embedding server started successfully")
        else:
            logger.warning(f"Embedding server startup issue: {result.stderr}")
    else:
        logger.warning(f"init_embedding.py not found at {embedding_launcher}")
except subprocess.TimeoutExpired:
    logger.warning("Embedding server startup timed out, continuing anyway...")
except Exception as e:
    logger.warning(f"Failed to launch embedding server: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # FIRST RUN: Cache all models for offline operation
    from app.utils.model_cache import ensure_models_cached
    ensure_models_cached()
    
    # Preload embedding model on GPU for fast inference
    from app.services.retrieval_service import EmbeddingService
    EmbeddingService.preload_model()
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Create default admin user if not exists
    await create_default_admin()
    
    # Start background task queue (Phase 5)
    from app.services.task_queue import setup_task_queue_lifespan, teardown_task_queue_lifespan
    await setup_task_queue_lifespan()
    logger.info("Task queue started")
    
    # Start maintenance scheduler (Phase 5)
    from app.services.maintenance_jobs import setup_maintenance_scheduler, teardown_maintenance_scheduler
    await setup_maintenance_scheduler()
    logger.info("Maintenance scheduler started")
    
    yield
    
    # Shutdown maintenance scheduler
    await teardown_maintenance_scheduler()
    logger.info("Maintenance scheduler stopped")
    
    # Shutdown task queue
    await teardown_task_queue_lifespan()
    logger.info("Task queue stopped")
    
    # Shutdown - dispose engine to close all connections immediately
    logger.info("Shutting down application...")
    from app.database import engine
    await engine.dispose()
    logger.info("Database connections closed")


async def create_default_admin():
    """Create a default admin user if no admin exists."""
    from sqlmodel import select
    from app.models.user import User
    from app.services.auth_service import AuthService
    
    async with async_session_maker() as session:
        # Check if any admin exists
        result = await session.execute(
            select(User).where(User.is_admin == True)
        )
        admin = result.scalars().first()
        
        if not admin:
            # Create default admin
            auth_service = AuthService(session)
            try:
                admin_user = await auth_service.create_user(
                    email="admin@graphrag.com",
                    password="admin123",
                    name="Admin",
                )
                # Set admin flag
                admin_user.is_admin = True
                session.add(admin_user)
                await session.commit()
                logger.info("Default admin created: admin@graphrag.com / admin123")
            except Exception as e:
                logger.warning(f"Could not create default admin: {e}")
        else:
            logger.info(f"Admin user exists: {admin.email}")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Unified RAG Backend with Auth, Document Processing, and Chat",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware - allow all origins for dev/tunneling
    # When allow_origins is ["*"], we must set allow_credentials=False
    # OR use allow_origin_regex to match all while keeping credentials
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins for port-forwarding
        allow_credentials=False,  # Must be False when using wildcard origins
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    
    # Global exception handler - register AppException first for proper handling
    app.add_exception_handler(AppException, global_exception_handler)
    app.add_exception_handler(Exception, global_exception_handler)
    
    # Setup rate limiting
    setup_rate_limiting(app)
    
    # Mount API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    
    # Health check endpoint
    @app.get("/health", tags=["Health"])
    async def health_check():
        return JSONResponse(
            content={
                "status": "healthy",
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
            }
        )
    
    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8081,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
