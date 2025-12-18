"""
Pydantic schemas for API request/response models.
"""
from app.schemas.auth import (
    SignupRequest,
    SignupResponse,
    LoginRequest,
    LoginResponse,
    TokenRefreshRequest,
    TokenRefreshResponse,
    LogoutRequest,
)
from app.schemas.user import (
    UserResponse,
    UserUpdateRequest,
    UserListResponse,
)
from app.schemas.document import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    ChunkResponse,
)
from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamEvent,
    ConversationResponse,
    MessageResponse,
    FeedbackRequest,
    SourceCitation,
)

__all__ = [
    # Auth
    "SignupRequest",
    "SignupResponse", 
    "LoginRequest",
    "LoginResponse",
    "TokenRefreshRequest",
    "TokenRefreshResponse",
    "LogoutRequest",
    # User
    "UserResponse",
    "UserUpdateRequest",
    "UserListResponse",
    # Document
    "DocumentUploadResponse",
    "DocumentResponse",
    "DocumentListResponse",
    "ChunkResponse",
    # Chat
    "ChatRequest",
    "ChatResponse",
    "ChatStreamEvent",
    "ConversationResponse",
    "MessageResponse",
    "FeedbackRequest",
    "SourceCitation",
]
