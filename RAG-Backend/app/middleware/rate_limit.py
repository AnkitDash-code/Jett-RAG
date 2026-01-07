"""
Rate limiting middleware using slowapi.
Protects API endpoints from abuse.
"""
import logging
from typing import Optional, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import settings

logger = logging.getLogger(__name__)

# Try to import slowapi
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    SLOWAPI_AVAILABLE = True
except ImportError:
    SLOWAPI_AVAILABLE = False
    logger.warning("slowapi not installed. Rate limiting disabled.")


def get_user_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting.
    Uses user ID if authenticated, else IP address.
    """
    # Check for authenticated user
    if hasattr(request.state, "user") and request.state.user:
        user_id = getattr(request.state.user, "id", None)
        if user_id:
            return f"user:{user_id}"
    
    # Fall back to IP address
    return get_remote_address(request) if SLOWAPI_AVAILABLE else "unknown"


def create_limiter() -> Optional["Limiter"]:
    """Create rate limiter instance."""
    if not SLOWAPI_AVAILABLE:
        return None
    
    if not settings.RATE_LIMIT_ENABLED:
        logger.info("Rate limiting is disabled in settings")
        return None
    
    limiter = Limiter(
        key_func=get_user_identifier,
        default_limits=[settings.RATE_LIMIT_DEFAULT],
        storage_uri="memory://",  # Use in-memory storage (no Redis needed)
    )
    
    logger.info(f"Rate limiter created with default limit: {settings.RATE_LIMIT_DEFAULT}")
    return limiter


# Global limiter instance
limiter = create_limiter()


def get_limiter() -> Optional["Limiter"]:
    """Get the rate limiter instance."""
    return limiter


def rate_limit(limit: str) -> Callable:
    """
    Decorator for rate limiting endpoints.
    
    Usage:
        @router.post("/chat")
        @rate_limit("30/minute")
        async def chat(...):
            ...
    """
    if limiter is None:
        # Return passthrough decorator if limiter not available
        def passthrough(func):
            return func
        return passthrough
    
    return limiter.limit(limit)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for global rate limiting.
    Applied to all requests as a safety net.
    """
    
    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip rate limiting if disabled or not available
        if limiter is None:
            return await call_next(request)
        
        # Skip for health checks
        if request.url.path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # The actual rate limiting is done per-endpoint with decorators
        # This middleware just attaches the limiter to the app state
        request.state.limiter = limiter
        
        return await call_next(request)


def setup_rate_limiting(app):
    """
    Setup rate limiting for FastAPI app.
    Call this in main.py after app creation.
    """
    if not SLOWAPI_AVAILABLE or limiter is None:
        logger.warning("Rate limiting not available, skipping setup")
        return
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(RateLimitMiddleware)
    
    logger.info("Rate limiting middleware configured")
