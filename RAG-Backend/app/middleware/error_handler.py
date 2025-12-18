"""
Global exception handler for standardized error responses.
"""
import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

logger = logging.getLogger(__name__)


class AppException(Exception):
    """Base application exception with status code and detail."""
    
    def __init__(
        self,
        status_code: int = 500,
        detail: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR",
    ):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        super().__init__(detail)


class AuthenticationError(AppException):
    """Authentication failed."""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(status_code=401, detail=detail, error_code="AUTH_ERROR")


class AuthorizationError(AppException):
    """Authorization/permission denied."""
    
    def __init__(self, detail: str = "Permission denied"):
        super().__init__(status_code=403, detail=detail, error_code="FORBIDDEN")


class NotFoundError(AppException):
    """Resource not found."""
    
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=404, detail=detail, error_code="NOT_FOUND")


class ValidationAppError(AppException):
    """Validation error."""
    
    def __init__(self, detail: str = "Validation error"):
        super().__init__(status_code=422, detail=detail, error_code="VALIDATION_ERROR")


class RateLimitError(AppException):
    """Rate limit exceeded."""
    
    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(status_code=429, detail=detail, error_code="RATE_LIMIT")


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler that returns standardized error responses.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Handle our custom exceptions
    if isinstance(exc, AppException):
        logger.warning(
            f"Application error: {exc.error_code} - {exc.detail}",
            extra={"request_id": request_id}
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.detail,
                    "request_id": request_id,
                }
            }
        )
    
    # Handle Pydantic validation errors
    if isinstance(exc, ValidationError):
        logger.warning(
            f"Validation error: {exc.errors()}",
            extra={"request_id": request_id}
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": exc.errors(),
                    "request_id": request_id,
                }
            }
        )
    
    # Handle unexpected exceptions
    logger.exception(
        f"Unexpected error: {str(exc)}",
        extra={"request_id": request_id}
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "request_id": request_id,
            }
        }
    )
