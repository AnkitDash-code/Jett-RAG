"""
Tracing Middleware for Phase 7.
Lightweight request tracing for debugging and performance analysis.
"""
import uuid
import time
import logging
from typing import Optional, Callable, Any, Dict
from contextvars import ContextVar
from datetime import datetime

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

logger = logging.getLogger(__name__)

# Context variable for request-scoped trace ID
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_trace_context: ContextVar[Dict[str, Any]] = ContextVar("trace_context", default={})


def get_current_trace_id() -> str:
    """Get the current request's trace ID."""
    return _trace_id.get()


def get_trace_context() -> Dict[str, Any]:
    """Get the current trace context."""
    return _trace_context.get()


def set_trace_span(name: str, data: Optional[Dict[str, Any]] = None) -> None:
    """
    Add a span to the current trace.
    
    Use this to mark significant operations within a request.
    """
    ctx = _trace_context.get()
    if "spans" not in ctx:
        ctx["spans"] = []
    
    ctx["spans"].append({
        "name": name,
        "timestamp": time.time(),
        "data": data or {},
    })
    _trace_context.set(ctx)


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds tracing to all requests.
    
    Features:
    - Generates trace IDs for each request
    - Propagates trace IDs from headers (X-Trace-ID, X-Request-ID)
    - Logs request/response with timing
    - Collects spans for debugging
    - Sets audit context for logging
    """
    
    HEADER_TRACE_ID = "X-Trace-ID"
    HEADER_REQUEST_ID = "X-Request-ID"
    HEADER_PARENT_SPAN = "X-Parent-Span-ID"
    
    def __init__(
        self,
        app,
        log_requests: bool = True,
        log_responses: bool = True,
        log_slow_requests_ms: float = 1000.0,
        exclude_paths: Optional[list] = None,
    ):
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
        self.log_slow_requests_ms = log_slow_requests_ms
        self.exclude_paths = exclude_paths or ["/health", "/health/live", "/metrics"]
    
    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        # Skip tracing for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Get or generate trace ID
        trace_id = (
            request.headers.get(self.HEADER_TRACE_ID) or
            request.headers.get(self.HEADER_REQUEST_ID) or
            str(uuid.uuid4())
        )
        
        # Set context variables
        _trace_id.set(trace_id)
        _trace_context.set({
            "trace_id": trace_id,
            "parent_span_id": request.headers.get(self.HEADER_PARENT_SPAN),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.query_params),
            "start_time": time.time(),
            "spans": [],
        })
        
        # Set audit context
        try:
            from app.services.audit_service import set_audit_context
            
            set_audit_context(
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("User-Agent"),
                request_id=trace_id,
            )
        except ImportError:
            pass
        
        # Log request
        if self.log_requests:
            logger.info(
                f"[{trace_id[:8]}] --> {request.method} {request.url.path}",
                extra={"trace_id": trace_id},
            )
        
        # Process request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration_ms = (time.time() - start_time) * 1000
            
            # Add trace ID to response headers
            response.headers[self.HEADER_TRACE_ID] = trace_id
            
            # Log response
            if self.log_responses:
                log_level = logging.INFO
                if duration_ms > self.log_slow_requests_ms:
                    log_level = logging.WARNING
                
                logger.log(
                    log_level,
                    f"[{trace_id[:8]}] <-- {response.status_code} "
                    f"{request.method} {request.url.path} ({duration_ms:.1f}ms)",
                    extra={
                        "trace_id": trace_id,
                        "status_code": response.status_code,
                        "duration_ms": duration_ms,
                    },
                )
            
            return response
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(
                f"[{trace_id[:8]}] !!! {request.method} {request.url.path} "
                f"({duration_ms:.1f}ms) - {type(e).__name__}: {e}",
                extra={"trace_id": trace_id, "error": str(e)},
            )
            raise
        
        finally:
            # Clear context
            try:
                from app.services.audit_service import clear_audit_context
                clear_audit_context()
            except ImportError:
                pass


class SpanContext:
    """
    Context manager for creating trace spans.
    
    Usage:
        async with SpanContext("database_query") as span:
            result = await db.execute(query)
            span.set_data({"rows": len(result)})
    """
    
    def __init__(self, name: str, data: Optional[Dict[str, Any]] = None):
        self.name = name
        self.data = data or {}
        self.start_time: float = 0
        self.end_time: float = 0
    
    async def __aenter__(self):
        self.start_time = time.time()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000
        
        self.data["duration_ms"] = duration_ms
        if exc_type:
            self.data["error"] = str(exc_val)
        
        set_trace_span(self.name, self.data)
        return False
    
    def set_data(self, data: Dict[str, Any]) -> None:
        """Add data to the span."""
        self.data.update(data)


def trace_function(name: Optional[str] = None):
    """
    Decorator to trace a function execution.
    
    Usage:
        @trace_function("database_query")
        async def query_database():
            ...
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        async def wrapper(*args, **kwargs):
            async with SpanContext(span_name):
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator
