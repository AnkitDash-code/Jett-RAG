"""
Request logging middleware for structured JSON logs.
"""
import time
import uuid
import logging
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests with timing and context."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request ID
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.perf_counter()
        
        # Get user ID if authenticated (will be set by auth middleware)
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Process request
        response = None
        try:
            response = await call_next(request)
            return response
        finally:
            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            # Log request details
            log_data = {
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "user_id": user_id,
                "client_ip": request.client.host if request.client else "unknown",
                "duration_ms": round(duration_ms, 2),
                "status_code": response.status_code if response else 500,
            }
            
            if response and response.status_code >= 400:
                logger.warning(f"Request completed with error: {log_data}")
            else:
                logger.info(f"Request completed: {log_data}")
