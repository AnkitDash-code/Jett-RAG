"""
RAG Backend - Main FastAPI Application
Unified API server for auth, documents, retrieval, chat, and monitoring.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database import init_db, async_session_maker
from app.api.v1 import api_router
from app.middleware.logging import RequestLoggingMiddleware
from app.middleware.error_handler import global_exception_handler
from app.middleware.rate_limit import setup_rate_limiting

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    
    # Initialize database
    await init_db()
    logger.info("Database initialized")
    
    # Create default admin user if not exists
    await create_default_admin()
    
    yield
    
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
    
    # Global exception handler
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
