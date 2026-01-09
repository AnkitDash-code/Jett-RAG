# Models package
from app.models.user import User, Role, UserRole, RefreshToken, APIKey
from app.models.document import Document, DocumentVersion, Chunk
from app.models.chat import Conversation, Message, QueryLog, Feedback
from app.models.memory import Memory, MemoryAccess, MemoryLink, MemoryType, MemoryStatus
from app.models.session import UserSession, SessionActivity, SessionStatus
from app.models.audit import AuditLog, AuditAction, AuditSeverity, AuditQuery

__all__ = [
    "User",
    "Role", 
    "UserRole",
    "RefreshToken",
    "APIKey",
    "Document",
    "DocumentVersion",
    "Chunk",
    "Conversation",
    "Message",
    "QueryLog",
    "Feedback",
    # Phase 6: Memory models
    "Memory",
    "MemoryAccess",
    "MemoryLink",
    "MemoryType",
    "MemoryStatus",
    # Phase 7: Session models
    "UserSession",
    "SessionActivity",
    "SessionStatus",
    # Phase 7: Audit models
    "AuditLog",
    "AuditAction",
    "AuditSeverity",
    "AuditQuery",
]
